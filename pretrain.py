"""Run BIT MAE pretraining on the human-only data pool.

Usage
-----
    python pretrain.py --config configs/pretrain/ndt/trainer.yaml \
                      [--kwargs key=value ...]

The ``--kwargs`` mechanism is the same as the upstream BIT trainer: any
dotted key can be overridden, e.g.
``--kwargs training.num_epochs=100 optimizer.lr=3e-4``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

# Make local imports resolve regardless of where the script is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

from utils.config import (  # noqa: E402
    ParseKwargs,
    DictConfig,
    update_config,
    default_trainer_config,
    config_from_kwargs,
)
from utils.datasets import DATASET_TO_IDX, DATASET_CHANNELS  # noqa: E402
from utils.eval_spike import eval_neurons_metric  # noqa: E402
from models.ndt import eval_rankme  # noqa: E402
from models.trainer import Trainer, _JEPA_MODEL_CLASSES  # noqa: E402

from data.prepare_data import ALL_HUMAN_SUBJECTS  # noqa: E402

_JEPA_MODEL_CLASSES = set(_JEPA_MODEL_CLASSES)


def eval_jepa_r2(model, model_inputs, unused_inputs, outputs, config, **kwargs):
    """R² between predictor outputs and target-encoder outputs at masked positions."""
    from sklearn.metrics import r2_score

    pred   = outputs["preds"].float().detach().cpu().numpy()      # (B, T_p, D)
    target = outputs["targets"].float().detach().cpu().numpy()    # (B, T_p, D)
    mask   = outputs["mask"].detach().cpu().numpy().astype(bool)  # (B, T_p)

    if mask.sum() < 2:
        return torch.tensor(float("nan"), device=model_inputs["spikes"].device)

    pred_v   = pred[mask]    # (N_valid, D)
    target_v = target[mask]  # (N_valid, D)

    # Drop dimensions with degenerate target variance.
    var  = target_v.astype(np.float64).var(axis=0)
    keep = var > 1e-3
    if not keep.any():
        return torch.tensor(float("nan"), device=model_inputs["spikes"].device)

    r2 = r2_score(
        target_v[:, keep].astype(np.float64),
        pred_v[:, keep].astype(np.float64),
        multioutput="raw_values",
    )
    r2 = np.where(np.isfinite(r2), r2, np.nan)
    return torch.tensor(float(np.nanmean(r2)), device=model_inputs["spikes"].device)


def _tag_records(records: List[Dict[str, Any]], dataset_idx: Optional[int]) -> List[Dict[str, Any]]:
    """Annotate every record with the global ``dataset_idx`` consumed by
    :class:`MultiSpikingDataset`.

    If ``dataset_idx`` is None (composite subjects like neural pile whose
    records already carry per-source dataset_idx), the existing value is
    preserved and not overwritten.
    """
    out = []
    for r in records:
        r = dict(r)
        if dataset_idx is not None:
            r["dataset_idx"] = int(dataset_idx)
        out.append(r)
    return out


def load_multi_subject_dataset(
    subjects: List[str],
    neural_pile_max_train: Optional[int] = None,
    neural_pile_source_filter: Optional[List[str]] = None,
    primate_max_train: Optional[int] = None,
    add_human_test_to_train: bool = False,
    add_primate_val_test_to_train: bool = False,
    add_holdout_to_train: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build the train/val/test splits for SSL pretraining.

    For every subject in ``subjects`` we call its loader from
    :mod:`data.prepare_data` and merge the records, tagging each with
    its ``dataset_idx``.

    Composite subjects (e.g. ``primate_*``) are not in DATASET_TO_IDX because
    their records already carry per-source dataset_idx values set by the loader.
    For these, _tag_records preserves the existing per-record dataset_idx.

    ``neural_pile_max_train``: cap for neural pile train records.
    ``primate_max_train``: cap for locally-preprocessed primate train records
        (applied across all primate_* datasets combined, stratified by dataset_idx).
    ``add_human_test_to_train``: if True, merge human test records into train.
    ``add_primate_val_test_to_train``: if True, merge primate val+test records into train.
    """
    merged: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for subject in subjects:
        if subject not in ALL_HUMAN_SUBJECTS:
            raise KeyError(
                f"Unknown subject {subject!r}. Known subjects: "
                f"{sorted(ALL_HUMAN_SUBJECTS)}"
            )

        ds_idx = DATASET_TO_IDX.get(subject, None)
        loader = ALL_HUMAN_SUBJECTS[subject]["loader"]
        tag = f"dataset_idx={ds_idx}" if ds_idx is not None else "per-source dataset_idx"
        print(f"[pretrain] loading {subject} ({tag}) ...", flush=True)
        loader_kwargs = {}
        if subject == "primate_neural_pile":
            if neural_pile_max_train is not None:
                loader_kwargs["max_train_samples"] = neural_pile_max_train
            if neural_pile_source_filter is not None:
                loader_kwargs["source_filter"] = neural_pile_source_filter
        splits = loader(**loader_kwargs)

        for split in ("train", "val", "test"):
            if split in splits:
                merged[split].extend(_tag_records(splits[split], ds_idx))

    # Optionally add the SSL-only holdout (card cleaned_test + willett competition)
    # to TRAIN. Never touches merged["val"]/["test"] → the evaluation test set
    # (second half of the val split) is unchanged.
    if add_holdout_to_train:
        from data.prepare_data import load_ssl_pretrain_holdout
        holdout = load_ssl_pretrain_holdout(subjects)
        total = 0
        for subj, recs in holdout.items():
            merged["train"].extend(_tag_records(recs, DATASET_TO_IDX.get(subj)))
            total += len(recs)
            print(f"[pretrain] SSL holdout: +{len(recs)} {subj} records", flush=True)
        print(f"[pretrain] added {total} SSL-only holdout records to train", flush=True)

    # Optionally add human test records to training
    if add_human_test_to_train:
        # Human records are eagerly loaded (have "spikes"), primate are lazy (have "npz_path")
        human_test = [r for r in merged["test"] if "npz_path" not in r]
        merged["train"].extend(human_test)
        print(
            f"[pretrain] added {len(human_test)} human test records to train",
            flush=True,
        )

    # Optionally add primate val+test records to training
    if add_primate_val_test_to_train:
        primate_val  = [r for r in merged["val"]  if "npz_path" in r]
        primate_test = [r for r in merged["test"] if "npz_path" in r]
        merged["train"].extend(primate_val)
        merged["train"].extend(primate_test)
        print(
            f"[pretrain] added {len(primate_val)} primate val + {len(primate_test)} primate test records to train",
            flush=True,
        )

    # Optionally cap locally-preprocessed primate train records (stratified)
    if primate_max_train is not None:
        primate_train = [r for r in merged["train"] if "npz_path" in r]
        other_train   = [r for r in merged["train"] if "npz_path" not in r]
        if len(primate_train) > primate_max_train:
            rng = np.random.default_rng(42)
            by_idx = {}
            for i, r in enumerate(primate_train):
                k = r.get("dataset_idx", 0)
                by_idx.setdefault(k, []).append(i)
            selected = []
            for k, idxs in by_idx.items():
                n_keep = max(1, round(primate_max_train * len(idxs) / len(primate_train)))
                chosen = rng.choice(idxs, min(n_keep, len(idxs)), replace=False)
                selected.extend(chosen.tolist())
            primate_train = [primate_train[i] for i in sorted(selected)]
            print(
                f"[pretrain] primate train capped: {len(primate_train)}/{primate_max_train} records",
                flush=True,
            )
        merged["train"] = other_train + primate_train

    for split in ("train", "val", "test"):
        print(f"[pretrain] split {split:5s}: {len(merged[split])} records", flush=True)

    return merged


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="configs/pretrain/ndt/trainer.yaml",
        help="Path (relative to this script) of the YAML trainer config.",
    )
    parser.add_argument("--kwargs", nargs="*", action=ParseKwargs)
    return parser.parse_args()


def main(args):
    # The YAML files use relative ``include:`` paths (e.g. ``configs/models/...``)
    # and ``utils.config`` reads ``configs/trainer.yaml`` from CWD, so anchor
    # the working directory here.
    os.chdir(_HERE)

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_HERE, cfg_path)

    # Load YAML on top of the package default and resolve any ``include:``
    # directives — this mirrors the behaviour of ConfigBuilder._load_base_config
    # without dragging in the upstream dataset/padding/embedder registry.
    config = update_config(default_trainer_config(), cfg_path)
    if args.kwargs is not None:
        config = update_config(config, config_from_kwargs(args.kwargs))

    # Load the multi-subject SSL dataset.
    subjects = list(config.data.subjects)
    neural_pile_max_train = getattr(config.data, "neural_pile_max_train", None)
    neural_pile_source_filter = getattr(config.data, "neural_pile_source_filter", None)
    if neural_pile_source_filter is not None:
        neural_pile_source_filter = list(neural_pile_source_filter)
    primate_max_train = getattr(config.data, "primate_max_train", None)
    add_human_test_to_train = getattr(config.data, "add_human_test_to_train", False)
    add_primate_val_test_to_train = getattr(config.data, "add_primate_val_test_to_train", False)
    dataset = load_multi_subject_dataset(
        subjects,
        neural_pile_max_train=neural_pile_max_train,
        neural_pile_source_filter=neural_pile_source_filter,
        primate_max_train=primate_max_train,
        add_human_test_to_train=add_human_test_to_train,
        add_primate_val_test_to_train=add_primate_val_test_to_train,
    )

    # Validate on willet_t12 (idx=2) and card_t15 (idx=3) only.
    # These are the two main human decoding subjects and have consistent
    # trial structure that makes R² a meaningful validation signal.
    _val_idx = {DATASET_TO_IDX["willet_t12"], DATASET_TO_IDX["card_t15"]}
    n_val_before = len(dataset["val"])
    dataset["val"]  = [r for r in dataset["val"]  if r.get("dataset_idx") in _val_idx]
    dataset["test"] = [r for r in dataset["test"] if r.get("dataset_idx") in _val_idx]
    print(
        f"[pretrain] val filtered to t12+t15: {n_val_before} → {len(dataset['val'])} records",
        flush=True,
    )

    # Build a channel_dict restricted to the dataset_idx values actually present
    # in the loaded records — avoids allocating ReadIn/ReadOut for unused datasets.
    used_idx = {
        int(r["dataset_idx"])
        for split_records in dataset.values()
        for r in split_records
        if "dataset_idx" in r
    }
    full_ch = DATASET_CHANNELS["human_all"]
    # idx→(name, n_ch) from the canonical registry (no duplicate indices)
    _idx_to_entry = {DATASET_TO_IDX[name]: (name, n_ch) for name, n_ch in full_ch.items()}
    filtered_channel_dict = {
        name: n_ch
        for idx, (name, n_ch) in _idx_to_entry.items()
        if idx in used_idx
    }
    print(
        f"[pretrain] channel_dict filtered to {len(filtered_channel_dict)}/{len(full_ch)} datasets",
        flush=True,
    )
    extra_model_kwargs = {"features": filtered_channel_dict}

    # Choose eval metrics based on the pretraining objective.
    model_class_name = getattr(config.model, "model_class", "NDT")
    if model_class_name in _JEPA_MODEL_CLASSES:
        # JEPA: R² in latent space + RankMe on target-encoder representations.
        metric_fns = {"r2_latent": eval_jepa_r2, "rankme": eval_rankme}
    else:
        # MAE: evaluate R² (or BPS for Poisson loss) on spike reconstructions.
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
