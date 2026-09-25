"""Finetune the MAE-pretrained BIT encoder for phoneme decoding (CTC).

Usage
-----
    python finetune.py \
        --config configs/finetune/phoneme/ndt/trainer.yaml \
        --subject willet_t12 \
        --pretrained_ckpt checkpoints/bit_pretrain_human_only/<step_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from utils.config import (  # noqa: E402
    ParseKwargs,
    DictConfig,
    update_config,
    default_trainer_config,
    config_from_kwargs,
    find_key,
)
from utils.datasets import DATASET_TO_IDX, DATASET_CHANNELS  # noqa: E402
from utils.eval_llm import per, cer, cer_typing  # noqa: E402
from models.trainer import Trainer  # noqa: E402

from data.prepare_data import ALL_HUMAN_SUBJECTS  # noqa: E402

# Handwriting subjects use character-level CTC (wav2vec2 vocab) instead of phoneme CTC.
_HANDWRITING_SUBJECTS = frozenset({"willett_handwriting", "fan_handwriting"})
_CHAR_VOCAB_PATH = os.path.join(_HERE, "tokenizers", "wav2vec2-base", "vocab.json")
_CHAR_VOCAB_SIZE = 32
_CHAR_NORM = re.compile(r"[^A-Z ]")

# Typing subjects use character-level CTC with the 30-key QWERTY vocab (30 chars + BLANK = 31).
# seq_class_ids in the typing pickles already contains char indices 1-30; only the
# phonemes strings (display labels) need fixing — phonemes_idx values are correct as-is.
_TYPING_SUBJECTS = frozenset({"jude_typing_t17", "jude_typing_t18"})
_TYPING_VOCAB_PATH = os.path.join(_HERE, "vocab_typing.json")
_TYPING_VOCAB_SIZE = 31


def _apply_typing_ctc(splits: Dict[str, list]) -> None:
    """Fix phonemes display strings for typing records (phonemes_idx values 1-30 are already correct char indices)."""
    typing_vocab = json.load(open(_TYPING_VOCAB_PATH))
    for records in splits.values():
        for rec in records:
            rec["phonemes"] = [typing_vocab[int(i)] for i in rec["phonemes_idx"]]


def _apply_char_ctc(splits: Dict[str, list], vocab: dict) -> None:
    """Replace phonemes_idx with wav2vec2 character IDs in every record (in-place)."""
    space_id = vocab["|"]
    id_to_char = {v: k for k, v in vocab.items()}
    for records in splits.values():
        for rec in records:
            sentence = _CHAR_NORM.sub("", str(rec.get("sentence", "")).upper()).strip()
            char_ids = [space_id if c == " " else vocab[c] for c in sentence]
            rec["phonemes_idx"] = np.asarray(char_ids, dtype=np.int64)
            rec["phonemes"] = [id_to_char[i] for i in char_ids]


def load_subject_dataset(subject: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return ``train/val/test`` splits for a single phoneme-decoding subject."""

    if subject not in ALL_HUMAN_SUBJECTS:
        raise KeyError(f"Unknown subject: {subject}")
    splits = ALL_HUMAN_SUBJECTS[subject]["loader"]()

    # Sanity check: this loader must have produced phoneme labels.
    if splits["train"] and "phonemes_idx" not in splits["train"][0]:
        raise ValueError(
            f"Subject {subject!r} has no phoneme labels — only the speech "
            "datasets (willet_t12, card_t15) can be used for CTC finetuning."
        )
    for split in ("train", "val", "test"):
        print(f"[finetune] {subject:>12s} | {split:5s}: {len(splits.get(split, []))} records",
              flush=True)
    return splits


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="configs/finetune/phoneme/ndt/trainer.yaml",
    )
    parser.add_argument(
        "--subject", type=str, default="willet_t12",
        choices=["willet_t12", "card_t15", "kunz_t15", "kunz_t12",
                 "kunz_t16", "kunz_t17", "wairagkar", "jude_speech_anarthia",
                 "willett_handwriting", "fan_handwriting",
                 "jude_typing_t17", "jude_typing_t18"],
        help="Phoneme decoding target.",
    )
    parser.add_argument(
        "--pretrained_ckpt", type=str, default="none",
        help="Path to a directory containing encoder.bin + encoder_config.pth "
             "(produced by NDT.save_checkpoint during pretraining). "
             "Pass 'none' to train from scratch.",
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help="Key into utils.datasets.DATASET_CHANNELS that picks which "
             "per-subject ReadIn/ReadOut layers to build. Defaults to "
             "'human_all' when loading a pretrained checkpoint (so the "
             "subject-specific layers match the checkpoint) and to the "
             "single-subject key (e.g. 'card_t15') when training from "
             "scratch.",
    )
    parser.add_argument(
        "--resume_ckpt", type=str, default=None,
        help="Path to a checkpoint directory produced by save_accelerator_state_to_resume "
             "(contains optimizer/scheduler/model state + epoch.pth). Mutually exclusive "
             "with --pretrained_ckpt. Example: checkpoints/bit_finetune_phoneme_willet_t12_pretrained/LAST",
    )
    parser.add_argument("--reinit-readin", action="store_true",
        help="When plugging in a pretrained encoder, skip its per-subject read_in "
             "weights so the read-in starts from a fresh init (diagnostic for "
             "negative transfer: keeps the encoder body, re-learns the read-in).")
    parser.add_argument("--mask-ratio", type=float, default=None,
        help="Override the per-subject input masking ratio (default is chosen by "
             "subject: 0.5 for T12-family, 0 otherwise). E.g. --mask-ratio 0.5. "
             "active is set to (ratio > 0).")
    parser.add_argument("--kwargs", nargs="*", action=ParseKwargs)
    return parser.parse_args()


def main(args):
    os.chdir(_HERE)
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_HERE, cfg_path)

    config = update_config(default_trainer_config(), cfg_path)

    # ---- Subject-specific overrides ----
    config["method"]["dataset_kwargs"]["dataset_name"] = args.subject

    # T12 arrays use mask ratio 0.5; T15 arrays and kunz_t16/t17 use 0.0.
    # Both T12 datasets (willet_t12 and its same-subject twin kunz_t12) use the
    # T12 setting; callers can override with --kwargs.
    if args.subject in ("willet_t12", "kunz_t12", "wairagkar", "jude_speech_anarthia"):
        find_key(config["model"], "masker")["ratio"] = 0.5
        find_key(config["model"], "masker")["active"] = True
    else:  # card_t15, kunz_t15, kunz_t16, kunz_t17
        find_key(config["model"], "masker")["ratio"] = 0.0
        find_key(config["model"], "masker")["active"] = False

    # Explicit CLI override of the masking ratio (takes precedence over the
    # per-subject default above).
    if args.mask_ratio is not None:
        find_key(config["model"], "masker")["ratio"] = args.mask_ratio
        find_key(config["model"], "masker")["active"] = args.mask_ratio > 0

    # ---- Resume a finetuning run OR plug in a pretrained encoder ----
    if args.resume_ckpt is not None and args.pretrained_ckpt != "none":
        raise ValueError("--resume_ckpt and --pretrained_ckpt are mutually exclusive.")

    if args.resume_ckpt is not None:
        if not os.path.exists(os.path.join(args.resume_ckpt, "epoch.pth")):
            raise FileNotFoundError(
                f"No epoch.pth found under {args.resume_ckpt!r} — "
                "pass a directory produced by save_accelerator_state_to_resume."
            )
        config["training"]["resume_from"] = args.resume_ckpt
        config["savestring"] = f"bit_finetune_phoneme_{args.subject}_pretrained"
    elif args.pretrained_ckpt != "none":
        if not os.path.exists(os.path.join(args.pretrained_ckpt, "encoder.bin")):
            raise FileNotFoundError(
                f"No encoder.bin found under {args.pretrained_ckpt!r} — did "
                "you finish a pretraining run?"
            )
        find_key(config["model"], "encoder")["from_pt"] = args.pretrained_ckpt
        config["savestring"] = f"bit_finetune_phoneme_{args.subject}_pretrained"
    else:
        config["savestring"] = f"bit_finetune_phoneme_{args.subject}_from_scratch"

    # Diagnostic: re-initialize the per-subject read-in even when loading a
    # pretrained encoder (only fires inside NDT when from_pt is set).
    if args.reinit_readin:
        config["method"]["model_kwargs"]["reinit_readin"] = True

    # ---- CLI overrides ----
    if args.kwargs is not None:
        config = update_config(config, config_from_kwargs(args.kwargs))

    # ---- Load the dataset ----
    dataset = load_subject_dataset(args.subject)

    # Pick the DATASET_CHANNELS bucket. When loading the pretrained encoder
    # we *must* use the same bucket as during pretraining ("human_all") so the
    # per-subject ReadIn keys match the checkpoint. Otherwise default to just
    # this subject — yielding a tighter single-subject BIT-TFS model.
    if args.features is not None:
        features = args.features
    elif args.resume_ckpt is not None or args.pretrained_ckpt != "none":
        features = "human_all"
    else:
        features = args.subject
    extra_model_kwargs = {"features": features}

    # Typing: character-level CTC with 30-key QWERTY vocab (seq_class_ids already correct).
    if args.subject in _TYPING_SUBJECTS:
        _apply_typing_ctc(dataset)
        config["method"]["model_kwargs"]["vocab_size"] = _TYPING_VOCAB_SIZE
        config["method"]["metric_kwargs"]["metric_name"] = "CER"
        metric_fns = {"CER": cer_typing}
    # Handwriting: character-level CTC via local wav2vec2 vocab (no internet needed).
    elif args.subject in _HANDWRITING_SUBJECTS:
        char_vocab = json.load(open(_CHAR_VOCAB_PATH))
        _apply_char_ctc(dataset, char_vocab)
        config["method"]["model_kwargs"]["vocab_size"] = _CHAR_VOCAB_SIZE
        config["method"]["metric_kwargs"]["metric_name"] = "CER"
        metric_fns = {"CER": cer}
    else:
        metric_fns = {"PER": per}

    trainer = Trainer(
        config,
        dataset=dataset,
        metric_fns=metric_fns,
        extra_model_kwargs=extra_model_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    main(parse_arguments())
