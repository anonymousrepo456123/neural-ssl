"""Bridge between preprocessed dataset pickles and the neural-ssl pipeline.

Each dataset has its own prepare_*.py that loads raw .mat / .hdf5 files and
returns split -> list[record].  Here we replicate that contract but read from
pre-processed pickles already on disk.

ALL_HUMAN_SUBJECTS maps each subject name (must match DATASET_TO_IDX in
utils/datasets.py) to {"loader": callable}.  The callable returns:
    {
        "train": List[Dict],
        "val":   List[Dict],
        "test":  List[Dict],
    }
where every record dict contains at minimum:
    spikes:    np.ndarray (T, C) float32  — C matches DATASET_CHANNELS["human_all"]
    day_idx:   np.ndarray scalar
    block_idx: np.ndarray scalar  (0 when not present in the pickle)

Channel counts verified from the actual pickles (no padding or doubling required):
    willet_t12          256 ch
    card_t15            512 ch
    kunz_t12/t15/t16/t17  512 ch
    willett_handwriting 192 ch
    fan_handwriting     192 ch
"""
from __future__ import annotations

import os
import pickle
import sys
from typing import Any, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

from data.data_paths import (
    TRAIN_CARD_DATASET, VAL_CARD_DATASET, TEST_CARD_DATASET,
    TRAIN_WILLET_DATASET, TEST_WILLET_DATASET,
    TRAIN_HANDWRITING_DATASET, TEST_HANDWRITING_DATASET,
    TRAIN_FAN_HANDWRITING_DATSET, VAL_FAN_HANDWRITING_DATSET, TEST_FAN_HANDWRITING_DATSET,
    KUNZ_DATASET_PATH,
    TRAIN_WAIRAGKAR_DATASET,
    JUDE_SPEECH_LARGE_TRAIN,
    JUDE_TYPING_T17, JUDE_TYPING_T18,
    TRAIN_NEURALPILE_DATASET, TEST_NEURALPILE_DATASET,
)

# Primate dataset processed directories — set to your local paths
PRIMATE_PROCESSED = {
    "temmar":     "/path/to/temmar/processed",
    "chowdhury":  "/path/to/chowdhury/processed",
    "ma":         "/path/to/ma/processed",
    "evenchen":   "/path/to/evenchen/processed",
    "churchland": "/path/to/churchland/processed",
    "odoherty":   "/path/to/odoherty/processed_reach",
    "perich":     "/path/to/perich/processed",
}

CLIP_AT = 10.0  # matches BaseNeuralTextDataset(clip_values_at=10)

# Cursor-control dataset paths — set to your local paths
_CURSOR_PATHS = {
    "cursor_karpowicz":    ("/path/to/cursor_karpowicz_train.pkl",
                            "/path/to/cursor_karpowicz_val.pkl"),
    "cursor_singer_clark": ("/path/to/cursor_singer_clark_train.pkl",
                            "/path/to/cursor_singer_clark_val.pkl"),
    "cursor_wilson":       ("/path/to/cursor_wilson_train.pkl",
                            "/path/to/cursor_wilson_val.pkl"),
}

import json
_VOCAB: List[str] = json.load(open(os.path.join(_HERE, "..", "vocab.json")))


# ── helpers ────────────────────────────────────────────────────────────────

def _load_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_records(data_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a raw pickle data_dict to the list-of-dicts record format.

    Reads neural_features, dayIdx, and optionally block_num and phoneme labels.
    All channel counts are already correct in the pickles — no padding or
    channel duplication is applied.

    Phoneme labels: seq_class_ids (padded with zeros) trimmed to seq_len gives
    phonemes_idx, which SpikingDatasetForDecoding expects for CTC finetuning.
    """
    neural_features = data_dict["neural_features"]
    day_idx_raw = data_dict["dayIdx"]
    # card / handwriting / fan_handwriting store block as "block_num";
    # willet and kunz have no block info — default to 0.
    block_raw = data_dict.get("block_num", [0] * len(neural_features))

    seq_class_ids = data_dict.get("seq_class_ids")
    seq_len = data_dict.get("seq_len")
    sentence_label = data_dict.get("sentence_label")
    has_phonemes = seq_class_ids is not None and seq_len is not None

    records: List[Dict[str, Any]] = []
    for i in range(len(neural_features)):
        record = {
            "spikes":    np.clip(neural_features[i], -CLIP_AT, CLIP_AT).astype(np.float32),
            "day_idx":   np.asarray(day_idx_raw[i]),
            "block_idx": np.asarray(block_raw[i]),
        }
        if sentence_label is not None:
            record["sentence"] = sentence_label[i]
        if has_phonemes and seq_len[i] is not None:
            length = int(seq_len[i])
            phonemes_idx = np.asarray(seq_class_ids[i][:length], dtype=np.int64)
            record["phonemes_idx"] = phonemes_idx
            record["phonemes"] = [_VOCAB[idx] for idx in phonemes_idx]
        records.append(record)
    return records


# ── per-subject loaders ────────────────────────────────────────────────────

def _split_val(records: List[Dict[str, Any]], seed: int = 42) -> tuple:
    """Mirror BIT's split_val(): random 50/50 split with seed=42."""
    n = len(records)
    perm = np.random.RandomState(seed).permutation(n)
    half = n // 2
    return [records[i] for i in perm[:half]], [records[i] for i in perm[half:]]


def _load_willet_t12() -> Dict[str, List[Dict[str, Any]]]:
    train    = _to_records(_load_pkl(TRAIN_WILLET_DATASET))
    test_all = _to_records(_load_pkl(TEST_WILLET_DATASET))
    # BIT splits the test set 50/50 (seed=42): first half → val, second half → test
    val, test = _split_val(test_all)
    return {"train": train, "val": val, "test": test}


def _load_card_t15() -> Dict[str, List[Dict[str, Any]]]:
    train   = _to_records(_load_pkl(TRAIN_CARD_DATASET))
    val_all = _to_records(_load_pkl(VAL_CARD_DATASET))
    # BIT splits the val pickle 50/50 (seed=42): first half → val, second half → test
    # cleaned_test_data.pkl is the true holdout (no phoneme labels), not used here
    val, test = _split_val(val_all)
    return {"train": train, "val": val, "test": test}


def load_ssl_pretrain_holdout(subjects) -> Dict[str, List[Dict[str, Any]]]:
    """Extra neural records used ONLY for SSL pretraining (never the finetune):
      - card_t15   -> cleaned_test_data.pkl               (true holdout, 512ch)
      - willet_t12 -> willet_cleaned_competition_data.pkl (competition,  256ch)
    Returned per-subject (untagged); the caller assigns the dataset_idx. The
    EVALUATION test set (second half of the val split) is left untouched.
    """
    from data.data_paths import TEST_CARD_DATASET, COMPETITION_WILLET_DATASET
    spec = {"card_t15": TEST_CARD_DATASET, "willet_t12": COMPETITION_WILLET_DATASET}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for subj, path in spec.items():
        if subj in subjects:
            out[subj] = _to_records(_load_pkl(path))
    return out


_KUNZ_REAL_CHANNELS = {"t12": 256, "t15": 512, "t16": 128, "t17": 256}

# t12/t15: new cleaned files (125k sentences already excluded, no DO_NOTHING trials).
# val_50words.pkl is split 50/50 seed=42 into val + held-out test.
_KUNZ_50W_PATH = os.environ.get("KUNZ_50W_PATH", KUNZ_DATASET_PATH)

def _load_kunz(tag: str) -> Dict[str, List[Dict[str, Any]]]:
    real_ch = _KUNZ_REAL_CHANNELS[tag]
    def _strip_padding(data: Dict) -> Dict:
        data = dict(data)
        data["neural_features"] = [f[:, :real_ch] for f in data["neural_features"]]
        return data
    if tag in ("t12", "t15"):
        train_raw = _to_records(_strip_padding(_load_pkl(
            os.path.join(_KUNZ_50W_PATH, f"kunz_{tag}_train.pkl"))))
        val_raw   = _to_records(_strip_padding(_load_pkl(
            os.path.join(_KUNZ_50W_PATH, f"kunz_{tag}_val_50words.pkl"))))
        val, test = _split_val(val_raw, seed=42)
        return {"train": train_raw, "val": val, "test": test}
    # t16/t17: single-word datasets, old path, no held-out test.
    train_raw = _to_records(_strip_padding(_load_pkl(os.path.join(KUNZ_DATASET_PATH, f"kunz_{tag}_train.pkl"))))
    val_raw   = _to_records(_strip_padding(_load_pkl(os.path.join(KUNZ_DATASET_PATH, f"kunz_{tag}_val.pkl"))))
    return {"train": train_raw, "val": val_raw, "test": val_raw}


def _load_willett_handwriting() -> Dict[str, List[Dict[str, Any]]]:
    train    = _to_records(_load_pkl(TRAIN_HANDWRITING_DATASET))
    test_all = _to_records(_load_pkl(TEST_HANDWRITING_DATASET))
    # Split the test pickle 50/50 (seed 42) into val (selection) + held-out test,
    # like card_t15/willet_t12 — avoids val==test (selection-biased) numbers.
    val, test = _split_val(test_all)
    return {"train": train, "val": val, "test": test}


def _load_fan_handwriting() -> Dict[str, List[Dict[str, Any]]]:
    train = _to_records(_load_pkl(TRAIN_FAN_HANDWRITING_DATSET))
    val   = _to_records(_load_pkl(VAL_FAN_HANDWRITING_DATSET))
    test  = _to_records(_load_pkl(TEST_FAN_HANDWRITING_DATSET))
    return {"train": train, "val": val, "test": test}


_KARPOWICZ_FALCON_DIR = "/path/to/karpowicz/processed"


def _load_karpowicz_falcon() -> Dict[str, List[Dict[str, Any]]]:
    from data.prepare_karpowicz import load_karpowicz
    return load_karpowicz(_KARPOWICZ_FALCON_DIR)


def _load_cursor(name: str) -> Dict[str, List[Dict[str, Any]]]:
    train_path, val_path = _CURSOR_PATHS[name]
    def _prep(data: Dict) -> Dict:
        data = dict(data)
        if "block_ids" in data and "block_num" not in data:
            data["block_num"] = data["block_ids"]
        # Channels with no spikes in a trial produce std=0 → NaN after z-scoring.
        # Replace with 0 (= mean, a neutral value post z-score).
        if "neural_features" in data:
            data["neural_features"] = [
                np.nan_to_num(x, nan=0.0) for x in data["neural_features"]
            ]
        return data
    train = _to_records(_prep(_load_pkl(train_path)))
    val   = _to_records(_prep(_load_pkl(val_path)))
    return {"train": train, "val": val, "test": val}


# ── additional loaders ────────────────────────────────────────────────────

def _split_records(records: List[Dict[str, Any]], val_ratio: float = 0.1, seed: int = 42):
    """Random train/val/test split when no official splits exist."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n_val = max(1, int(len(records) * val_ratio))
    val_idx = idx[:n_val]
    test_idx = idx[n_val:2 * n_val]
    train_idx = idx[2 * n_val:]
    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )


def _load_wairagkar() -> Dict[str, List[Dict[str, Any]]]:
    all_records = _to_records(_load_pkl(TRAIN_WAIRAGKAR_DATASET))
    train, val, test = _split_records(all_records, val_ratio=0.1)
    return {"train": train, "val": val, "test": test}


def _load_jude_speech_anarthia() -> Dict[str, List[Dict[str, Any]]]:
    all_records = _to_records(_load_pkl(JUDE_SPEECH_LARGE_TRAIN))
    train, val, test = _split_records(all_records, val_ratio=0.1)
    return {"train": train, "val": val, "test": test}


def _load_jude_typing(path: str) -> Dict[str, List[Dict[str, Any]]]:
    data = _load_pkl(path)
    # jude_typing uses "block_ids" instead of "block_num"
    if "block_ids" in data and "block_num" not in data:
        data["block_num"] = data["block_ids"]
    all_records = _to_records(data)
    train, val, test = _split_records(all_records, val_ratio=0.1)
    return {"train": train, "val": val, "test": test}


def _load_neural_pile(
    max_train_samples: int = None,
    source_filter: list = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Lazy loader for the neural pile (primate data, 103k trials).

    Records contain only ``npz_path`` + metadata — the actual neural_features
    are loaded from disk in MultiSpikingDataset.__getitem__ to avoid loading
    ~634 GB into RAM. neural_features in the npz are (C_padded, T); the Dataset
    slices to the real channel count (npz_n_channels) and transposes to (T, C).

    Each source_dataset gets its own ReadIn layer via its specific dataset_idx
    in DATASET_TO_IDX. The real channel count comes from
    neural_pile_channel_map_complete.json.

    ``max_train_samples``: if set, subsample the train split to this many
    records (stratified by source_dataset to keep all species represented).
    The neural pile has 103k train trials vs ~5-15k human trials; without
    capping, 90%+ of gradient steps come from the pile and the trunk loses
    human temporal structure (flat rasters, R²<0 on human val).
    """
    from utils.datasets import DATASET_TO_IDX

    _ch_map_path = os.path.join(_HERE, "neural_pile_alive_channels.json")
    channel_map: Dict[str, int] = json.load(open(_ch_map_path))

    def _index_to_records(index_path: str, pile_root: str) -> List[Dict[str, Any]]:
        records = []
        with open(index_path) as f:
            for line in f:
                entry = json.loads(line)
                src = entry["source_dataset"]
                if src not in DATASET_TO_IDX:
                    continue
                if source_filter is not None and src not in source_filter:
                    continue
                records.append({
                    "npz_path":            os.path.join(pile_root, entry["path"]),
                    "npz_n_channels":      channel_map[src],
                    "npz_source_dataset":  src,
                    "dataset_idx":         DATASET_TO_IDX[src],
                    "day_idx":             np.asarray(0),
                    "block_idx":           np.asarray(0),
                })
        return records

    train_index = os.path.join(TRAIN_NEURALPILE_DATASET, "index_train.jsonl")
    test_index  = os.path.join(TEST_NEURALPILE_DATASET,  "index_test.jsonl")
    train = _index_to_records(train_index, TRAIN_NEURALPILE_DATASET)
    test  = _index_to_records(test_index,  TEST_NEURALPILE_DATASET)

    if max_train_samples is not None and len(train) > max_train_samples:
        # Stratified subsample: keep each source_dataset proportional to its size.
        rng = np.random.default_rng(42)
        by_src: Dict[str, List[int]] = {}
        for i, r in enumerate(train):
            by_src.setdefault(r["npz_source_dataset"], []).append(i)
        selected: List[int] = []
        for src, idxs in by_src.items():
            n_keep = max(1, round(max_train_samples * len(idxs) / len(train)))
            chosen = rng.choice(idxs, min(n_keep, len(idxs)), replace=False)
            selected.extend(chosen.tolist())
        train = [train[i] for i in sorted(selected)]
        print(f"[neural_pile] subsampled train: {len(train)}/{103125} records", flush=True)

    return {"train": train, "val": test, "test": test}


# ── primate loaders (preprocessed locally) ───────────────────────────────────

def _load_primate(name: str) -> Dict[str, List[Dict[str, Any]]]:
    """Lazy loader for a preprocessed primate dataset.

    Records already carry per-subject dataset_idx set by each prepare_*.py.
    Chowdhury has no val/test (<=2 sessions per subject); returns empty lists.
    """
    from data.prepare_temmar     import load_temmar
    from data.prepare_chowdhury  import load_chowdhury
    from data.prepare_ma         import load_ma
    from data.prepare_evenchen   import load_evenchen
    from data.prepare_churchland import load_churchland
    from data.prepare_odoherty   import load_odoherty
    from data.prepare_perich     import load_perich

    loader_fn = {
        "temmar":     load_temmar,
        "chowdhury":  load_chowdhury,
        "ma":         load_ma,
        "evenchen":   load_evenchen,
        "churchland": load_churchland,
        "odoherty":   load_odoherty,
        "perich":     load_perich,
    }[name]
    return loader_fn(PRIMATE_PROCESSED[name])


# ── public registry ────────────────────────────────────────────────────────
# Human-subject keys must match DATASET_TO_IDX in utils/datasets.py.
# Primate composite keys (primate_*) use per-record dataset_idx — they
# need NOT appear in DATASET_TO_IDX; _tag_records preserves existing values.

ALL_HUMAN_SUBJECTS: Dict[str, Dict[str, Any]] = {
    # Human BCI datasets
    "willet_t12":           {"loader": _load_willet_t12},
    "card_t15":             {"loader": _load_card_t15},
    "kunz_t12":             {"loader": lambda: _load_kunz("t12")},
    "kunz_t15":             {"loader": lambda: _load_kunz("t15")},
    "kunz_t16":             {"loader": lambda: _load_kunz("t16")},
    "kunz_t17":             {"loader": lambda: _load_kunz("t17")},
    "willett_handwriting":  {"loader": _load_willett_handwriting},
    "fan_handwriting":      {"loader": _load_fan_handwriting},
    "wairagkar":            {"loader": _load_wairagkar},
    "jude_speech_anarthia": {"loader": _load_jude_speech_anarthia},
    "jude_typing_t17":      {"loader": lambda: _load_jude_typing(JUDE_TYPING_T17)},
    "jude_typing_t18":      {"loader": lambda: _load_jude_typing(JUDE_TYPING_T18)},
    # Neural pile (pre-built pile, separate from our loaders)
    "primate_neural_pile":  {"loader": _load_neural_pile},
    # Primate datasets preprocessed locally
    "primate_temmar":     {"loader": lambda: _load_primate("temmar")},
    "primate_chowdhury":  {"loader": lambda: _load_primate("chowdhury")},
    "primate_ma":         {"loader": lambda: _load_primate("ma")},
    "primate_evenchen":   {"loader": lambda: _load_primate("evenchen")},
    "primate_churchland": {"loader": lambda: _load_primate("churchland")},
    "primate_odoherty":   {"loader": lambda: _load_primate("odoherty")},
    "primate_perich":     {"loader": lambda: _load_primate("perich")},
    # NEJM cursor-control human subjects
    "cursor_karpowicz":    {"loader": lambda: _load_cursor("cursor_karpowicz")},
    "cursor_singer_clark": {"loader": lambda: _load_cursor("cursor_singer_clark")},
    "cursor_wilson":       {"loader": lambda: _load_cursor("cursor_wilson")},
    # Karpowicz 2024 / FALCON handwriting
    "karpowicz_falcon":    {"loader": _load_karpowicz_falcon},
}
