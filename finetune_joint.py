"""Finetune the MAE-pretrained encoder jointly on more than one labelled
data source, reporting per-source val PER.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
from utils.config import (
    ParseKwargs, DictConfig, update_config, default_trainer_config,
    config_from_kwargs, find_key,
)
from utils.datasets import (
    DATASET_TO_IDX, DATASET_CHANNELS,
    SpikingDatasetForDecoding, GroupBatchSampler, pad_collate_fn,
)
import re as _re

from utils.eval_llm import per as per_metric, cer as cer_metric, cer_typing as cer_typing_metric
from models.trainer import Trainer
from models.ndt import NDT

_HANDWRITING_SUBJECTS = frozenset({"willett_handwriting", "fan_handwriting"})
_CHAR_VOCAB_PATH = os.path.join(_HERE, "tokenizers", "wav2vec2-base", "vocab.json")
_CHAR_VOCAB_SIZE = 32
_CHAR_NORM = _re.compile(r"[^A-Z ]")

_TYPING_SUBJECTS = frozenset({"jude_typing_t17", "jude_typing_t18"})
_TYPING_VOCAB_PATH = os.path.join(_HERE, "vocab_typing.json")
_TYPING_VOCAB_SIZE = 31


def _apply_char_ctc_record(rec: Dict[str, Any], vocab: dict) -> None:
    """Overwrite phonemes_idx with wav2vec2 char IDs for one record (in-place)."""
    space_id = vocab["|"]
    id_to_char = {v: k for k, v in vocab.items()}
    sentence = _CHAR_NORM.sub("", str(rec.get("sentence", "")).upper()).strip()
    char_ids = [space_id if c == " " else vocab[c] for c in sentence]
    rec["phonemes_idx"] = np.asarray(char_ids, dtype=np.int64)
    rec["phonemes"] = [id_to_char[i] for i in char_ids]


def _apply_typing_ctc_record(rec: Dict[str, Any], vocab: list) -> None:
    """Fix phonemes display strings for a typing record (phonemes_idx already correct)."""
    rec["phonemes"] = [vocab[int(i)] for i in rec["phonemes_idx"]]

from data.prepare_data import ALL_HUMAN_SUBJECTS


# ---------------------------------------------------------------------------
# Multi-source SL dataset
# ---------------------------------------------------------------------------

class MultiSourceDecodingDataset(SpikingDatasetForDecoding):
    """Like ``SpikingDatasetForDecoding`` but reads per-record dataset
    indices (so each record can route to its own ReadIn). Also retains a
    ``source_tag`` field that downstream code uses to split val PER by
    source.

    Two invocation styles supported:
      * trainer-style:  ``MultiSourceDecodingDataset(dataset_dict, split,
                            length=None, spikes_name='spikes',
                            targets_name='phonemes_idx', **kwargs)``
      * direct-style:   ``MultiSourceDecodingDataset(records)``  (single
                            positional arg is a list of records).
    """

    def __init__(self, dataset_or_records, split=None, length=None, *,
                 spikes_name="spikes", targets_name="phonemes_idx", **kw):
        if isinstance(dataset_or_records, dict):
            records = dataset_or_records[split]
        else:
            records = dataset_or_records
        if length is not None:
            records = records[:length]
        self._records = records
        self.spikes_name = spikes_name
        self.targets_name = targets_name
        # GroupBatchSampler reads this to keep batches single-subject:
        self.dataset_indices = np.asarray(
            [int(r["dataset_idx"]) for r in records], dtype=np.int64
        )

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        inputs = deepcopy(self._records[idx])
        spikes  = inputs.pop(self.spikes_name)
        targets = inputs.pop(self.targets_name)
        ds_idx  = int(inputs.pop("dataset_idx"))
        # ``source_tag`` is the *original* source name (e.g. "kunz_t15"),
        # carried through unused_inputs so per-source PER can be aggregated.
        inputs.setdefault("source_tag", inputs.get("source_tag", "unknown"))
        inputs.update({
            "spikes":            spikes,
            "spikes_mask":       np.ones(spikes.shape[0], dtype=np.int64),
            "spikes_timestamp":  np.arange(0, spikes.shape[0]),
            "spikes_spacestamp": np.arange(0, spikes.shape[1]),
            "spikes_lengths":    np.asarray(spikes.shape[0]),
            "targets":           targets,
            "targets_mask":      np.ones_like(targets),
            "targets_lengths":   np.asarray(targets.shape[0]),
            "dataset_name":      np.array(ds_idx),
            "day_idx":           inputs.get("day_idx", np.asarray(0)),
            "block_idx":         inputs.get("block_idx", np.asarray(0)),
        })
        return inputs


# ---------------------------------------------------------------------------
# Per-source val PER (call after training, on the BEST checkpoint)
# ---------------------------------------------------------------------------

def per_source_val_per(cfg, model, val_records: List[Dict[str, Any]],
                       metric_fn=None):
    """Compute val PER/CER grouped by ``source_tag`` on the loaded model."""
    if metric_fn is None:
        metric_fn = per_metric

    model.eval()
    pad_dict = dict(cfg.method.dataloader_kwargs.pad_dict)
    model_inputs = ["spikes", "spikes_mask", "spikes_timestamp",
                    "dataset_name", "targets", "targets_lengths"]

    results = {}
    for tag in sorted({r["source_tag"] for r in val_records}):
        recs = [r for r in val_records if r["source_tag"] == tag]
        ds = MultiSourceDecodingDataset(recs)
        sampler = GroupBatchSampler(ds, batch_size=32, shuffle=False,
                                    drop_last=False, seed=0)
        loader = DataLoader(ds, batch_sampler=sampler,
                            collate_fn=partial(pad_collate_fn,
                                               model_inputs=model_inputs,
                                               pad_dict=pad_dict),
                            pin_memory=False)
        wsum, wtot = 0.0, 0
        with torch.no_grad():
            for inputs, unused in loader:
                inputs = {k: v.cuda() if torch.is_tensor(v) else v
                          for k, v in inputs.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**inputs)
                ratio = metric_fn(model, inputs, unused, out.to_dict(),
                                  cfg, n_print=0).item()
                n = int(inputs["targets_lengths"].sum().item())
                wsum += ratio * n
                wtot += n
        results[tag] = (wsum / wtot if wtot else float("nan"), wtot, len(ds))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sources", required=True,
        help="Comma-separated subject keys to train on jointly "
             "(e.g. 'card_t15,kunz_t15').",
    )
    p.add_argument(
        "--shared_as", default=None,
        help="If set, every record from all --sources is routed through "
             "this subject's ReadIn/ReadOut (Experiment 2). "
             "If omitted, each source uses its own (Experiment 3) — in that "
             "case also pass --features <bucket-listing-all-sources>.",
    )
    p.add_argument(
        "--features", default=None,
        help="DATASET_CHANNELS bucket the model is built from. Defaults to "
             "the --shared_as key when sharing layers, else 'human_all'.",
    )
    p.add_argument(
        "--pretrained_ckpt", default="none",
        help="Path to pretrain checkpoint dir (encoder.bin + encoder_config.pth).",
    )
    p.add_argument(
        "--config", default="configs/finetune/phoneme/ndt/trainer.yaml",
    )
    p.add_argument("--savestring_suffix", default=None,
                   help="Optional override for the checkpoint subdir name.")
    p.add_argument("--kwargs", nargs="*", action=ParseKwargs)
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_HERE)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_HERE, args.config)
    config = update_config(default_trainer_config(), cfg_path)
    config["data"]["vocab_file"] = os.path.join(_HERE, "vocab.json")

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    assert len(sources) >= 1, "give at least one --sources entry"

    # Choose features (which read-in modules the model will allocate).
    if args.features is not None:
        features = args.features
    elif args.shared_as is not None:
        features = args.shared_as
    elif args.pretrained_ckpt != "none":
        features = "human_all"
    else:
        raise ValueError("specify --features when training from scratch.")
    print(f"[joint] features bucket: {features}")

    # Decide per-record dataset_idx routing.
    if args.shared_as is not None:
        if args.shared_as not in DATASET_TO_IDX:
            raise KeyError(f"--shared_as {args.shared_as!r} not in DATASET_TO_IDX")
        shared_idx = DATASET_TO_IDX[args.shared_as]
        print(f"[joint] SHARED layers — all records routed via "
              f"{args.shared_as} (idx={shared_idx})")
    else:
        shared_idx = None
        print(f"[joint] SEPARATE layers — each source uses its own ReadIn/ReadOut")

    # Load + tag each source.
    train_recs, val_recs = [], []
    for src in sources:
        if src not in ALL_HUMAN_SUBJECTS:
            raise KeyError(src)
        splits = ALL_HUMAN_SUBJECTS[src]["loader"]()
        own_idx = DATASET_TO_IDX[src]
        idx = shared_idx if shared_idx is not None else own_idx
        for r in splits["train"]:
            r = dict(r); r["dataset_idx"] = idx; r["source_tag"] = src
            train_recs.append(r)
        for r in splits["val"]:
            r = dict(r); r["dataset_idx"] = idx; r["source_tag"] = src
            val_recs.append(r)
        print(f"[joint]   {src}: train={len(splits['train'])} val={len(splits['val'])}")
    print(f"[joint] TOTAL train={len(train_recs)} val={len(val_recs)}")

    # ---- Task type: typing / handwriting (CER) vs speech (PER) -------------
    is_handwriting = all(s in _HANDWRITING_SUBJECTS for s in sources)
    is_typing = all(s in _TYPING_SUBJECTS for s in sources)
    if is_handwriting:
        char_vocab = json.load(open(_CHAR_VOCAB_PATH))
        for r in train_recs + val_recs:
            _apply_char_ctc_record(r, char_vocab)
        config["method"]["model_kwargs"]["vocab_size"] = _CHAR_VOCAB_SIZE
        config["method"]["metric_kwargs"]["metric_name"] = "CER"
        metric_fns = {"CER": cer_metric}
        eval_metric_fn = cer_metric
        print(f"[joint] handwriting mode → CER, vocab_size={_CHAR_VOCAB_SIZE}")
    elif is_typing:
        typing_vocab = json.load(open(_TYPING_VOCAB_PATH))
        for r in train_recs + val_recs:
            _apply_typing_ctc_record(r, typing_vocab)
        config["method"]["model_kwargs"]["vocab_size"] = _TYPING_VOCAB_SIZE
        config["method"]["metric_kwargs"]["metric_name"] = "CER"
        metric_fns = {"CER": cer_typing_metric}
        eval_metric_fn = cer_typing_metric
        print(f"[joint] typing mode → CER, vocab_size={_TYPING_VOCAB_SIZE}")
    else:
        metric_fns = {"PER": per_metric}
        eval_metric_fn = per_metric

    # Time masking default: off (T15-style 0). Override per-run via the flat
    # convenience kwargs `mask_ratio` / `mask_active` — e.g. for a T12 joint
    # (willet_t12+kunz_t12) pass `--kwargs mask_ratio=0.5`. Using a flat key
    # avoids having to spell out the deep model.encoder...masker path.
    masker = find_key(config["model"], "masker")
    masker["ratio"] = 0.0
    masker["active"] = False

    # Plug in the pretrained encoder.
    if args.pretrained_ckpt != "none":
        ckpt = args.pretrained_ckpt
        if not os.path.exists(os.path.join(ckpt, "encoder.bin")):
            raise FileNotFoundError(f"no encoder.bin in {ckpt!r}")
        find_key(config["model"], "encoder")["from_pt"] = ckpt

    # Apply --kwargs overrides last.
    if args.kwargs is not None:
        config = update_config(config, config_from_kwargs(args.kwargs))

    # Honour the flat mask kwargs after the merge, applying them to the (deeply
    # nested) masker via find_key, then drop them so they don't linger in config.
    if "mask_ratio" in config:
        masker = find_key(config["model"], "masker")
        masker["ratio"] = float(config.pop("mask_ratio"))
        masker["active"] = masker["ratio"] > 0.0
    if "mask_active" in config:
        find_key(config["model"], "masker")["active"] = bool(config.pop("mask_active"))

    # Naming
    if args.savestring_suffix:
        config["savestring"] = args.savestring_suffix
    else:
        tag = "shared" if shared_idx is not None else "separate"
        config["savestring"] = f"bit_joint_{tag}_{'_'.join(sources)}"

    # We need a dict-of-splits to satisfy Trainer.set_dataset; the dataset
    # class we plug in below ignores the split key naming and reads from the
    # global lists we built above.
    dataset_dict = {"train": train_recs, "val": val_recs, "test": val_recs}

    # Monkey-patch NAME2DATASET to register our multi-source class.
    from models import trainer as trainer_mod
    trainer_mod.NAME2DATASET["multi_sl"] = MultiSourceDecodingDataset
    config["data"]["dataset_class"] = "multi_sl"
    config["data"]["train_name"] = "train"
    config["data"]["val_name"] = "val"
    config["data"]["test_name"] = "test"

    extra_model_kwargs = {"features": features}

    trainer = Trainer(
        config, dataset=dataset_dict,
        metric_fns=metric_fns, extra_model_kwargs=extra_model_kwargs,
    )

    # ---- Per-source best-checkpointing ------------------------------------
    # After EVERY eval epoch, evaluate the current model SEPARATELY on each
    # source and keep, per source, the checkpoint with that source's best val
    # PER/CER (dir BEST_<source>) — instead of only the single combined-val BEST.
    # NOTE: per_source_val_per moves data to a single CUDA device; this path is
    # intended for single-GPU joint sweeps (no DDP gather).
    ckpt_root = os.path.join(config.dirs.checkpoint_dir, config.savestring)
    best_src = {}   # source -> (per, n_phonemes, n_trials, epoch)
    out_json = os.path.join(ckpt_root, "per_source_per.json")

    def _flush_per_source_json():
        with open(out_json, "w") as _f:
            json.dump({src: {"per": p, "n_phonemes": nph, "n_trials": ntr, "best_epoch": ep}
                       for src, (p, nph, ntr, ep) in best_src.items()}, _f, indent=2)

    def _per_source_hook(epoch):
        res = per_source_val_per(config, trainer.model, val_recs,
                                 metric_fn=eval_metric_fn)
        improved = False
        for src, (per_val, n_ph, n_tr) in res.items():
            if per_val < best_src.get(src, (float("inf"),))[0]:
                best_src[src] = (per_val, n_ph, n_tr, epoch)
                trainer.save_checkpoint(
                    save_name=f"BEST_{src}", epoch=epoch,
                    save_to_path=os.path.join(ckpt_root, f"BEST_{src}"),
                )
                print(f"[joint] new BEST_{src}: PER {100*per_val:.2f}% @epoch {epoch}",
                      flush=True)
                improved = True
        if improved and best_src:
            _flush_per_source_json()

    trainer._post_eval_hook = _per_source_hook
    trainer.train()

    # ---- Report: each source's OWN best PER + checkpoint ------------------
    print("\n" + "=" * 60)
    print(f"Per-source BEST validation PER — {config.savestring}")
    print("=" * 60)
    for src, (per_val, n_ph, n_tr, ep) in best_src.items():
        print(f"  {src:20s}  trials={n_tr:4d}  phonemes={n_ph:6d}"
              f"  PER={per_val:.5f} ({100*per_val:.2f}%)  @epoch {ep}  → BEST_{src}")
    _flush_per_source_json()
    print(f"\n[joint] Saved per-source best PER + checkpoints (BEST_<src>) under {ckpt_root}")


if __name__ == "__main__":
    main()
