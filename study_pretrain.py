"""Shared pool construction for the pretraining-strategy study.

"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.datasets import DATASET_TO_IDX, DATASET_CHANNELS
from pretrain import load_multi_subject_dataset
from pretrain_all_humans import cap_perich_sessions

import study_config as SC

REGIMES = {"h": "POOL_H", "hm": "POOL_HM"}


def pool_for(regime: str) -> List[str]:
    regime = regime.lower()
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}, got {regime!r}")
    return list(SC.POOL_H if regime == "h" else SC.POOL_HM)


#: A deliberately tiny pool used only by ``--smoke`` to exercise the training
#: code paths in minutes. Never use for a real run.
SMOKE_POOL = ["willet_t12", "card_t15"]


def build_study_dataset(regime: str, config,
                        subjects_override: List[str] | None = None,
                        ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict]:
    """Load the frozen pretraining pool for ``regime`` ('h' or 'hm').

    Returns ``(dataset, extra_model_kwargs)``. Mutates ``config`` to record the
    actual subject list and the per-epoch cap, so the saved
    ``trainer_config.pth`` reflects what really trained.

    ``subjects_override`` is for smoke tests only; it is recorded in the config
    so a smoke checkpoint can never be mistaken for a study encoder.
    """
    subjects = list(subjects_override) if subjects_override else pool_for(regime)
    if subjects_override:
        print(f"[study] *** SMOKE POOL {subjects} -- NOT a study encoder ***",
              flush=True)
        config["data"]["smoke_pool"] = True

    # Held-out neural data into SSL train only; matches the MAE encoders.
    config["data"]["add_holdout_to_train"] = SC.POOL_OPTIONS["add_holdout_to_train"]

    dataset = load_multi_subject_dataset(
        subjects,
        neural_pile_max_train=getattr(config.data, "neural_pile_max_train", None),
        neural_pile_source_filter=getattr(config.data, "neural_pile_source_filter", None),
        primate_max_train=getattr(config.data, "primate_max_train", None),
        add_human_test_to_train=getattr(config.data, "add_human_test_to_train", False),
        add_primate_val_test_to_train=getattr(config.data, "add_primate_val_test_to_train", False),
        add_holdout_to_train=True,
    )

    # SSL model selection happens on the two canonical speech subjects only.
    val_idx = {DATASET_TO_IDX[s] for s in SC.POOL_OPTIONS["val_subjects"]}
    n_val_before, n_test_before = len(dataset["val"]), len(dataset["test"])
    dataset["val"] = [r for r in dataset["val"] if r.get("dataset_idx") in val_idx]
    dataset["test"] = [r for r in dataset["test"] if r.get("dataset_idx") in val_idx]
    print(f"[study] val {n_val_before} -> {len(dataset['val'])}, "
          f"test {n_test_before} -> {len(dataset['test'])} "
          f"(restricted to {SC.POOL_OPTIONS['val_subjects']})", flush=True)

    # Perich fragmentation control (only bites in the H+M regime).
    if any(s == "primate_perich" for s in subjects):
        n_before = len(dataset["train"])
        kept, n_total = cap_perich_sessions(dataset, SC.POOL_OPTIONS["max_perich_sessions"])
        print(f"[study] perich sessions {n_total} -> {len(kept)} kept; "
              f"train {n_before} -> {len(dataset['train'])}", flush=True)

    # Rotating per-epoch cap: all trials are seen eventually, but no single
    # high-count source (cursor_wilson ~55k trials) dominates gradient steps.
    config["training"]["train_per_subject_cap"] = SC.POOL_OPTIONS["train_per_subject_cap"]
    config["data"]["subjects"] = subjects

    comp = Counter(int(r.get("dataset_idx", 0)) for r in dataset["train"])
    print(f"[study] regime={regime}  subjects={len(subjects)}  "
          f"train={len(dataset['train'])}  read-in groups={len(comp)}", flush=True)

    # Only allocate read-in/read-out for read-ins that actually occur, so the
    # parameter count does not depend on unused bucket entries.
    used_idx = {int(r["dataset_idx"]) for recs in dataset.values()
                for r in recs if "dataset_idx" in r}
    idx_to_entry = {DATASET_TO_IDX[name]: (name, n_ch)
                    for name, n_ch in DATASET_CHANNELS["human_all"].items()}
    channel_dict = {name: n_ch for idx, (name, n_ch) in idx_to_entry.items()
                    if idx in used_idx}
    missing = sorted(used_idx - set(idx_to_entry))
    if missing:
        raise KeyError(f"dataset_idx present in data but absent from the "
                       f"human_all channel bucket: {missing}")
    print(f"[study] channel_dict: {len(channel_dict)} datasets", flush=True)

    return dataset, {"features": channel_dict}


def apply_frozen_schedule(config) -> None:
    """Force the frozen pretraining schedule and fail loudly on drift."""
    p = SC.PRETRAIN
    config["training"]["num_epochs"] = p["num_epochs"]
    config["training"]["train_batch_size"] = p["train_batch_size"]
    config["training"]["test_batch_size"] = p["test_batch_size"]
    config["training"]["mixed_precision"] = p["mixed_precision"]
    config["optimizer"]["lr"] = p["lr"]
    config["optimizer"]["wd"] = p["wd"]
    config["optimizer"]["eps"] = p["eps"]
    config["optimizer"]["scheduler"] = p["scheduler"]
    config["optimizer"]["warmup_pct"] = p["warmup_pct"]
    config["optimizer"]["div_factor"] = p["div_factor"]
    config["optimizer"]["gradient_accumulation_steps"] = p["gradient_accumulation_steps"]
    config["seed"] = p["seed"]


def set_causal(config) -> None:
    """Make the encoder causal and disable masking (autoregressive setting).

    ``include:`` in the YAML replaces the whole ``model`` node, so these two
    settings cannot be expressed as sibling overrides in the config file --
    they are applied here and asserted, so an AR run can never silently
    pretrain with bidirectional attention.
    """
    enc = config["model"]["encoder"]
    enc["context"]["forward"] = 0     # attend to <= t only
    enc["context"]["backward"] = -1    # unlimited past
    enc["embedder"]["masker"]["active"] = False
    assert enc["context"]["forward"] == 0 and enc["context"]["backward"] == -1, \
        "AR pretraining requires causal attention (context.forward=0, backward=-1)"
    assert enc["embedder"]["masker"]["active"] is False, \
        "AR pretraining must not mask: causality already restricts the context"
    print("[study] encoder set to CAUSAL (forward=0, backward=-1), masking OFF",
          flush=True)
