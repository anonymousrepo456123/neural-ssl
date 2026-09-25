"""Frozen configuration for the pretraining-strategy study.


Import from here; do not copy the lists.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Pretraining pools (frozen)
# ---------------------------------------------------------------------------

#: Regime **H** — human-only pretraining pool (16 read-in groups).
#: Order is significant and must not be changed: it is the order recorded in
#: the existing MAE checkpoints, and the per-subject cap draws its subsample
#: per read-in group in iteration order.
POOL_H: List[str] = [
    "willet_t12",           # speech
    "kunz_t12",             # speech
    "kunz_t15",             # speech
    "kunz_t16",             # speech
    "kunz_t17",             # speech
    "card_t15",             # speech
    "wairagkar",            # speech
    "willett_handwriting",  # handwriting
    "fan_handwriting",      # handwriting
    "jude_speech_anarthia", # speech
    "jude_typing_t17",      # typing
    "jude_typing_t18",      # typing
    "karpowicz_falcon",     # handwriting (FALCON)
    "cursor_karpowicz",     # cursor control
    "cursor_singer_clark",  # cursor control
    "cursor_wilson",        # cursor control
]

#: The monkey datasets added in regime **H+M**.
PRIMATES: List[str] = [
    "temmar",
    "chowdhury",
    "ma",
    "evenchen",
    "churchland",
    "odoherty",
    "perich",
]

#: Regime **H+M** — human + monkey (cross-species) pool.
POOL_HM: List[str] = POOL_H + [f"primate_{p}" for p in PRIMATES]

#: Pool knobs that must match across objectives.
#:  * ``add_holdout_to_train``: held-out neural data (card cleaned_test +
#:    willett competition) is added to the SSL train stream only. Labels are
#:    never used, and the downstream test splits are untouched.
#:  * ``train_per_subject_cap``: rotating per-epoch cap. Without it
#:    cursor_wilson (~55k trials) supplies ~2/3 of all gradient steps.
#:  * ``max_perich_sessions``: Perich is fragmented into ~111 read-ins;
#:    keeping the 10 largest avoids ~40M freshly-initialised head params
#:    destabilising the shared trunk.
POOL_OPTIONS: Dict[str, object] = {
    "add_holdout_to_train": True,
    "train_per_subject_cap": 10_000,
    "max_perich_sessions": 10,
    # SSL model selection: val/test restricted to willet_t12 + card_t15.
    "val_subjects": ["willet_t12", "card_t15"],
}

# ---------------------------------------------------------------------------
# Pretraining schedule (frozen; identical for MAE / JEPA / AR)
# ---------------------------------------------------------------------------

PRETRAIN: Dict[str, object] = {
    "num_epochs": 400,
    "train_batch_size": 64,
    "test_batch_size": 64,
    "mixed_precision": "bf16",
    "lr": 5.0e-4,
    "wd": 1.0e-5,
    "eps": 1.0e-8,
    "scheduler": "cosine",
    "warmup_pct": 0.0,
    "div_factor": 25,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    # Encoder-side masking used during pretraining (BIT Table 10).
    "mask_ratio": 0.5,
    "mask_max_timespan": 15,
    "mask_expand_prob": 1.0,
    # Checkpoint to transfer from. BEST = best SSL val on willet_t12+card_t15.
    "select": "BEST",
}

# ---------------------------------------------------------------------------
# Downstream finetuning (frozen — ONE config for every cell)
# ---------------------------------------------------------------------------

#: The controlled protocol: every (condition x task) cell is finetuned with
#: exactly these hyperparameters. This is deliberately *not* a per-cell sweep
#: — a per-cell sweep lets a lucky lr rescue a bad encoder and turns the
#: comparison into a comparison of tuning budgets. The 30-config sweeps kept
#: under ``checkpoints_leonardo/`` are reported separately as a
#: tuned-per-condition robustness check.
#: lr and wd are the **medians of the 30 best-validation sweep winners** in
#: ``checkpoints_leonardo/`` (3 conditions x 10 tasks), recovered from their
#: ``trainer_config.pth``:
#:
#:     lr:  min 2.1e-4  median 8.2e-4  geo-mean 6.8e-4  max 9.9e-4
#:     wd:  min 4.8e-4  median 4.2e-3  geo-mean 6.4e-3  max 8.6e-2
#:
#: Using the winners' central tendency is legitimate model selection, not
#: leakage: every winner was chosen on **validation** PER, never on test.
#:
#: This replaces the earlier lr 3e-4 / wd 1e-2, which had simply been copied
#: from a README example command. wd 1e-2 was defensible (63rd percentile of
#: winners), but lr 3e-4 sat at the **10th percentile** -- 20 of 30 winners
#: chose >= 6.6e-4. Under a fixed 800-epoch budget a systematically low lr
#: under-trains every cell, and, worse, need not under-train them equally:
#: pretrained and randomly-initialised encoders prefer different learning
#: rates, so a badly-centred fixed lr could bias the very comparison this
#: study exists to make.
#:
#: For reference, the repo's default finetune config uses wd 1e-5 -- which is
#: *below* the finetuning search range of LogUniform(5e-5, 1e-1),
#: and no sweep winner came near it. The 1e-5/1e-6 figure belongs to
#: PRETRAINING (see ``PRETRAIN["wd"]``), a different phase.
FINETUNE: Dict[str, object] = {
    "num_epochs": 800,
    "train_batch_size": 32,     # winners: {8:4, 16:10, 32:6, 64:10} -> mid
    "test_batch_size": 32,
    "lr": 8.0e-4,               # median of sweep winners (was 3e-4)
    "wd": 4.0e-3,               # median of sweep winners (was 1e-2)
    "eps": 1.0e-8,
    "scheduler": "cosine",
    "warmup_pct": 0.05,
    "max_grad_norm": 1.0,
    "div_factor": 25,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "seed": 42,
}

FINETUNE_MASK_RATIO: Dict[str, float] = {
    # mask 0.5 (masker active)
    "willet_t12": 0.5,
    "kunz_t12": 0.5,
    "wairagkar": 0.5,
    "jude_speech_anarthia": 0.5,
    "willett_handwriting": 0.5,   
    "fan_handwriting": 0.5,       
    "jude_typing_t17": 0.5,       
    "jude_typing_t18": 0.5,       
    "card_t15": 0.0,
    "kunz_t15": 0.0,
    "kunz_t16": 0.0,
    "kunz_t17": 0.0,
}

# ---------------------------------------------------------------------------
# Downstream task battery
# ---------------------------------------------------------------------------

#: task -> input modality (what the participant was doing).


TASKS: Dict[str, str] = {
    "card_t15": "speech",
    "willet_t12": "speech",
    "kunz_t12": "speech",
    "kunz_t15": "speech",
    "kunz_t16": "speech",
    "kunz_t17": "speech",
    "wairagkar": "speech",
    "jude_speech_anarthia": "speech",
    "willett_handwriting": "handwriting",
    "fan_handwriting": "handwriting",
    "jude_typing_t17": "typing",
    "jude_typing_t18": "typing",
}

#: task -> metric, determined by the *target alphabet actually decoded*:

TASK_METRIC: Dict[str, str] = {
    "card_t15": "PER",
    "willet_t12": "PER",
    "kunz_t12": "PER",
    "kunz_t15": "PER",
    "kunz_t16": "PER",
    "kunz_t17": "PER",
    "wairagkar": "PER",
    "jude_speech_anarthia": "PER",
    "willett_handwriting": "CER",
    "fan_handwriting": "CER",
    "jude_typing_t17": "CER",
    "jude_typing_t18": "CER",
}

#: Each task's symbol table. Getting this wrong silently corrupts the metric,
#: because ``format_ctc`` maps predicted ids through whichever vocab it is given.
VOCAB_FILE: Dict[str, str] = {
    "willett_handwriting": "vocab_handwriting.json",
    "fan_handwriting": "vocab_handwriting.json",
    "jude_typing_t17": "vocab_typing.json",
    "jude_typing_t18": "vocab_typing.json",
}


def vocab_file(subject: str) -> str:
    """Vocabulary file for a task (defaults to the 41-symbol phoneme set)."""
    return VOCAB_FILE.get(subject, "vocab.json")


#: Tier-A = cleanest, best-instrumented tasks; the headline table.
TIER_A: List[str] = ["card_t15", "willet_t12", "kunz_t12", "kunz_t15"]

# ---------------------------------------------------------------------------
# Conditions (rows of the results matrix)
# ---------------------------------------------------------------------------

#: condition key -> (objective, regime, checkpoint dir relative to this file).
#: ``None`` checkpoint means train from scratch.
CONDITIONS: Dict[str, Dict[str, object]] = {
    "scratch": {
        "objective": "none",
        "regime": "-",
        "ckpt": None,
        "label": r"From scratch (BIT-TFS)",
    },
    "mae_h": {
        "objective": "MAE",
        "regime": "H",
        "ckpt": "checkpoints/bit_mae_all_humans_wairagkar_full_holdout/BEST",
        "label": r"MAE, human",
    },
    "mae_hm": {
        "objective": "MAE",
        "regime": "H+M",
        "ckpt": "checkpoints/bit_mae_allhumans_allprimates_perich10_wairagkar_full_holdout/BEST",
        "label": r"MAE, human+monkey",
    },
    "jepa_h": {
        "objective": "V-JEPA 2.1",
        "regime": "H",
        "ckpt": "checkpoints/bit_jepa21_h/BEST",
        "label": r"V-JEPA 2.1, human",
    },

    "jepa_hm": {
        "objective": "V-JEPA 2.1",
        "regime": "H+M",
        "ckpt": "checkpoints/bit_jepa21_hm/EPOCH100",
        "label": r"V-JEPA 2.1, human+monkey$^{\dagger}$",
        "collapsed": True,
        "epochs_trained": 100,
        "note": "collapsed at epoch 123; using last healthy save (EPOCH100)",
    },
    "ar_h": {
        "objective": "AR",
        "regime": "H",
        "ckpt": "checkpoints/bit_ar_h/BEST",
        "label": r"Autoregressive, human",
    },
    "ar_hm": {
        "objective": "AR",
        "regime": "H+M",
        "ckpt": "checkpoints/bit_ar_hm/BEST",
        "label": r"Autoregressive, human+monkey",
    },
}

#: RankMe thresholds for the collapse gate (STUDY_PLAN section 2.3).
RANKME_HEALTHY = 200
RANKME_COLLAPSED = 50


def features_bucket(condition: str, subject: str) -> str:
    """Which ``DATASET_CHANNELS`` bucket a finetune/eval must use.

    A pretrained encoder carries read-ins keyed by *global* dataset index for
    the whole ``human_all`` bucket, so finetuning from one must use that
    bucket or the read-in keys will not match. A from-scratch run has no such
    constraint and uses the single-subject bucket.
    """
    return subject if CONDITIONS[condition]["ckpt"] is None else "human_all"
