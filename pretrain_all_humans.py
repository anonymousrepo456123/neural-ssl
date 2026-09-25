"""BIT MAE pretraining: all humans + primates opzionally.

Human pool, complete:
    willet_t12, kunz_t12/t15/t16/t17, card_t15
    willett_handwriting, fan_handwriting
    wairagkar
    jude_speech_anarthia, jude_typing_t17, jude_typing_t18
    karpowicz_falcon  (FALCON handwriting )
    cursor_karpowicz, cursor_singer_clark, cursor_wilson

Primates, optionally, via --primate. Available options:
    temmar | chowdhury | ma | evenchen | churchland | odoherty | perich | neural_pile

Usage
-----
    python pretrain_all_humans.py \
        --config configs/pretrain/ndt/trainer.yaml \
        [--primate perich] \
        [--kwargs training.num_epochs=200 optimizer.lr=5e-4 ...]

Saves to checkpoints/bit_mae_all_humans[_<primate>]/
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np

from utils.config import (
    ParseKwargs,
    DictConfig,
    update_config,
    default_trainer_config,
    config_from_kwargs,
)
from utils.datasets import DATASET_TO_IDX, DATASET_CHANNELS
from utils.eval_spike import eval_neurons_metric
from models.ndt import eval_rankme
from models.trainer import Trainer, _JEPA_MODEL_CLASSES
from data.prepare_data import ALL_HUMAN_SUBJECTS
from pretrain import load_multi_subject_dataset, eval_jepa_r2

_JEPA_MODEL_CLASSES = set(_JEPA_MODEL_CLASSES)


ALL_HUMAN_LIST = [
    # Speech / typing
    "willet_t12",
    "kunz_t12",
    "kunz_t15",
    "kunz_t16",
    "kunz_t17",
    "card_t15",
    "wairagkar",
    "willett_handwriting",
    "fan_handwriting",
    "jude_speech_anarthia",
    "jude_typing_t17",
    "jude_typing_t18",
    # FALCON handwriting (Karpowicz 2024) — richiede prepare_karpowicz.py
    "karpowicz_falcon",
    # NEJM cursor-control
    "cursor_karpowicz",
    "cursor_singer_clark",
    "cursor_wilson",
]

PRIMATE_CHOICES = [
    "temmar",
    "chowdhury",
    "ma",
    "evenchen",
    "churchland",
    "odoherty",
    "perich",
    "neural_pile",
]


def cap_train_per_subject(records: List[Dict[str, Any]], cap: int, seed: int = 42):
    """Subsample TRAIN records so no single dataset_idx exceeds ``cap`` trials.

    Balances gradient steps across subjects: with GroupBatchSampler the number
    of batches per subject is ∝ its trial count, so a high-trial-count source
    like cursor_wilson (~55k train trials, ~67% of the corpus by count) dominates
    SSL and collapses the val signal on the speech subjects — even though by
    *hours* it is only ~25%. Capping removes the step-count dominance while
    keeping val/test untouched. Stratified per dataset_idx, seeded.
    """
    rng = np.random.default_rng(seed)
    by_idx: Dict[int, List[int]] = {}
    for i, r in enumerate(records):
        by_idx.setdefault(int(r.get("dataset_idx", 0)), []).append(i)
    keep: List[int] = []
    for idx, idxs in by_idx.items():
        if len(idxs) > cap:
            idxs = rng.choice(idxs, cap, replace=False).tolist()
        keep.extend(idxs)
    keep.sort()
    return [records[i] for i in keep]


_PERICH_REGISTRY = "/path/to/perich/processed/perich_session_registry.json"


def _perich_idx_set() -> set:
    """The set of dataset_idx values that belong to perich sessions."""
    import json as _json
    if os.path.exists(_PERICH_REGISTRY):
        return {int(info["idx"]) for info in _json.load(open(_PERICH_REGISTRY)).values()}
    return set()


def cap_perich_sessions(dataset: Dict[str, List[Dict[str, Any]]], n_keep: int):
    """Keep only the ``n_keep`` perich sessions with the most TRAIN trials.

    Perich is fragmented into ~111 per-session dataset_idx, each getting its own
    read-in/read-out pair (~40M params total, ~6x the shared trunk). The many
    freshly-initialised subject heads produce huge reconstruction losses whose
    gradients destabilise the shared trunk (rankme collapses, human val R² goes
    negative). Keeping fewer sessions → fewer heads → stable trunk. All non-perich
    records are left untouched.
    """
    from collections import Counter
    pidx = _perich_idx_set()
    counts = Counter(int(r["dataset_idx"]) for r in dataset["train"]
                     if int(r.get("dataset_idx", -1)) in pidx)
    keep = {idx for idx, _ in counts.most_common(n_keep)}
    for split in ("train", "val", "test"):
        dataset[split] = [
            r for r in dataset[split]
            if int(r.get("dataset_idx", -1)) not in pidx or int(r["dataset_idx"]) in keep
        ]
    return keep, len(counts)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=str,
        default="configs/pretrain/ndt/trainer.yaml",
        help="Path (relative to this script) of the YAML trainer config.",
    )
    p.add_argument(
        "--primate", type=str, nargs="+", default=None, choices=PRIMATE_CHOICES,
        metavar="PRIMATE",
        help="One or more primate datasets to add to the human pool "
             "(e.g. --primate temmar churchland ... perich).",
    )
    p.add_argument(
        "--cap-per-subject", type=int, default=None,
        help="FIXED cap on TRAIN trials per subject (stratified, seed 42). "
             "Balances gradient steps and prevents collapse from dominant subjects "
             "(e.g. cursor_wilson). Val/test are untouched. "
             "Recommended ~8000-10000. Note: fixed subset → discarded trials are NEVER seen.",
    )
    p.add_argument(
        "--cap-per-subject-per-epoch", type=int, default=None,
        help="Like --cap-per-subject but NOT fixed: every epoch the sampler "
             "re-samples up to N trials per subject (rotating subset). "
             "In the limit of infinite epochs the model sees ALL data, "
             "while keeping gradient steps balanced. Mutually exclusive with "
             "--cap-per-subject.",
    )
    p.add_argument(
        "--max-perich-sessions", type=int, default=None,
        help="Keep only the N Perich sessions with the most train trials "
             "(Perich is fragmented into ~111 dataset_idx → ~40M read-in params "
             "that destabilise the trunk). Reduces subject-specific heads. "
             "For an initial test try N=10.",
    )
    p.add_argument("--kwargs", nargs="*", action=ParseKwargs)
    return p.parse_args()


def main(args):
    os.chdir(_HERE)

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_HERE, cfg_path)

    config = update_config(default_trainer_config(), cfg_path)
    if args.kwargs is not None:
        config = update_config(config, config_from_kwargs(args.kwargs))

    subjects = list(ALL_HUMAN_LIST)
    if args.primate is not None:
        for p in args.primate:
            subjects.append(f"primate_{p}")

    run_name = "bit_mae_all_humans"
    if args.primate is not None:
        run_name += "_" + "_".join(args.primate)
    if not getattr(config, "savestring", None):
        config["savestring"] = run_name

    print(f"[all_humans] subjects: {subjects}", flush=True)
    print(f"[all_humans] savestring: {run_name}", flush=True)

    dataset = load_multi_subject_dataset(
        subjects,
        neural_pile_max_train=getattr(config.data, "neural_pile_max_train", None),
        neural_pile_source_filter=getattr(config.data, "neural_pile_source_filter", None),
        primate_max_train=getattr(config.data, "primate_max_train", None),
        add_human_test_to_train=getattr(config.data, "add_human_test_to_train", False),
        add_primate_val_test_to_train=getattr(config.data, "add_primate_val_test_to_train", False),
        add_holdout_to_train=getattr(config.data, "add_holdout_to_train", False),
    )

    # Val filtrato a willet_t12 + card_t15
    _val_idx = {DATASET_TO_IDX["willet_t12"], DATASET_TO_IDX["card_t15"]}
    n_val_before = len(dataset["val"])
    dataset["val"]  = [r for r in dataset["val"]  if r.get("dataset_idx") in _val_idx]
    dataset["test"] = [r for r in dataset["test"] if r.get("dataset_idx") in _val_idx]
    print(
        f"[all_humans] val filtered to t12+t15: {n_val_before} → {len(dataset['val'])} records",
        flush=True,
    )

    # Keep only the N largest perich sessions (reduce read-in fragmentation).
    if args.max_perich_sessions is not None:
        n_before = len(dataset["train"])
        kept, n_total = cap_perich_sessions(dataset, args.max_perich_sessions)
        print(
            f"[all_humans] perich sessions: {n_total} → {len(kept)} kept; "
            f"train {n_before} → {len(dataset['train'])} records",
            flush=True,
        )

    if args.cap_per_subject is not None and args.cap_per_subject_per_epoch is not None:
        sys.exit("--cap-per-subject (fixed) and --cap-per-subject-per-epoch "
                 "(rotating) are mutually exclusive; pick one.")

    # FIXED cap: subsample train once (dropped trials never seen again).
    if args.cap_per_subject is not None:
        n_before = len(dataset["train"])
        dataset["train"] = cap_train_per_subject(dataset["train"], args.cap_per_subject)
        print(
            f"[all_humans] train capped (FIXED) at {args.cap_per_subject}/subject: "
            f"{n_before} → {len(dataset['train'])} records",
            flush=True,
        )

    # PER-EPOCH cap: keep all records, let the sampler draw a fresh ≤N subset per
    # subject each epoch → all trials seen over enough epochs, steps stay balanced.
    if args.cap_per_subject_per_epoch is not None:
        config["training"]["train_per_subject_cap"] = args.cap_per_subject_per_epoch
        print(
            f"[all_humans] train capped (PER-EPOCH, rotating) at "
            f"{args.cap_per_subject_per_epoch}/subject — all trials seen over ∞ epochs",
            flush=True,
        )

    # Log the *actual* subjects/train composition (not the stale YAML default),
    # so W&B reflects what really trained.
    config["data"]["subjects"] = subjects
    from collections import Counter
    comp = Counter(int(r.get("dataset_idx", 0)) for r in dataset["train"])
    print(f"[all_humans] train composition by dataset_idx: {dict(sorted(comp.items()))}", flush=True)

    used_idx = {
        int(r["dataset_idx"])
        for split_records in dataset.values()
        for r in split_records
        if "dataset_idx" in r
    }
    full_ch = DATASET_CHANNELS["human_all"]
    _idx_to_entry = {DATASET_TO_IDX[name]: (name, n_ch) for name, n_ch in full_ch.items()}
    filtered_channel_dict = {
        name: n_ch
        for idx, (name, n_ch) in _idx_to_entry.items()
        if idx in used_idx
    }
    print(
        f"[all_humans] channel_dict: {len(filtered_channel_dict)} datasets",
        flush=True,
    )
    extra_model_kwargs = {"features": filtered_channel_dict}

    model_class_name = getattr(config.model, "model_class", "NDT")
    if model_class_name in _JEPA_MODEL_CLASSES:
        metric_fns = {"r2_latent": eval_jepa_r2, "rankme": eval_rankme}
    else:
        metric_name = "r2" if config.method.model_kwargs.loss == "mse" else "bps"
        metric_fns = {metric_name: eval_neurons_metric, "rankme": eval_rankme}

    trainer = Trainer(
        config,
        dataset=dataset,
        metric_fns=metric_fns,
        extra_model_kwargs=extra_model_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    main(parse_arguments())
