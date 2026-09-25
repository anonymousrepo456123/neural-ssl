"""Preprocessing for the Karpowicz 2024 / FALCON handwriting dataset (DANDI:000950).

Participant T5, 192 channels, already binned at 20ms in `acquisition/binned_spikes/data`.

Pipeline:
    1. For each NWB file: reads binned_spikes/data (T, 192) and timestamps
    2. Z-score within-session per channel
    3. Extracts trials (start/stop from intervals/trials)
    4. Saves one NPZ per trial (192, T_trial) + index JSONL

Pre-defined split from folder structure:
    sub-T5-held-in-calib   → train
    sub-T5-held-in-minival → val
    sub-T5-held-out-calib  → test

Usage:
    python prepare_karpowicz.py [--out_dir ./data/karpowicz/processed
                                [--data_dir ./data/karpowicz/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

DATA_DIR = Path("./data/karpowicz/000950")
OUT_DIR  = Path("./data/karpowicz/processed")
BIN_S    = 0.020   # 20ms — verified from NWB timestamps
N_CH     = 192

SUBSET_TO_SPLIT = {
    "sub-T5-held-in-calib":   "train",
    "sub-T5-held-in-minival": "val",
    "sub-T5-held-out-calib":  "test",
}


def read_nwb_session(fp: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read pre-binned data and trial times from an NWB file.

    Returns:
        binned  (T, 192) float32 — spike counts already at 20ms
        ts      (T,)     float64 — timestamps
        t_start (N_trials,)
        t_stop  (N_trials,)
    """
    with h5py.File(fp, "r") as f:
        binned  = f["acquisition/binned_spikes/data"][:]       # (T, 192)
        ts      = f["acquisition/binned_spikes/timestamps"][:] # (T,)
        t_start = f["intervals/trials/start_time"][:]
        t_stop  = f["intervals/trials/stop_time"][:]
    return binned.astype(np.float32), ts, t_start, t_stop


def zscore_session(data: np.ndarray) -> np.ndarray:
    """Z-score within-session per channel. Input/output: (T, 192)."""
    mean = data.mean(axis=0)           # (192,)
    std  = data.std(axis=0) + 1e-8
    return ((data - mean) / std).astype(np.float32)


def extract_trials(
    data_z: np.ndarray,
    ts: np.ndarray,
    t_start: np.ndarray,
    t_stop: np.ndarray,
) -> List[np.ndarray]:
    """Extract windows (192, T_trial) for each trial.

    Uses searchsorted on timestamps — robust to small irregularities.
    Trials with fewer than 2 bins are discarded.
    """
    windows = []
    for ts_s, ts_e in zip(t_start, t_stop):
        b_start = int(np.searchsorted(ts, ts_s))
        b_end   = int(np.searchsorted(ts, ts_e, side="right"))
        b_start = max(0, b_start)
        b_end   = min(len(ts), b_end)
        if b_end - b_start < 2:
            continue
        # data_z is (T, 192) → transpose to (192, T_trial)
        windows.append(data_z[b_start:b_end].T.copy())
    return windows


def prepare_karpowicz(
    out_dir: Path = OUT_DIR,
    data_dir: Path = DATA_DIR,
) -> None:
    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, List] = {"train": [], "val": [], "test": []}

    for subset, split in SUBSET_TO_SPLIT.items():
        subset_dir = data_dir / subset
        if not subset_dir.exists():
            print(f"[karpowicz] WARNING: {subset_dir} not found, skipping")
            continue

        files = sorted(subset_dir.glob("*.nwb"))
        print(f"\n[karpowicz] {subset} → {split}: {len(files)} files")

        for day_idx, fp in enumerate(files):
            print(f"  {fp.name}", flush=True)

            binned, ts, t_start, t_stop = read_nwb_session(fp)
            data_z = zscore_session(binned)
            windows = extract_trials(data_z, ts, t_start, t_stop)
            print(f"    trials={len(windows)}  bins/session={len(ts):,}  "
                  f"dur={len(ts)*BIN_S/3600:.3f}h", flush=True)

            ses_out = out_dir / subset / fp.stem
            (ses_out / "trials").mkdir(parents=True, exist_ok=True)

            for t_idx, arr in enumerate(windows):
                fname    = f"trial_{t_idx:05d}.npz"
                npz_path = ses_out / "trials" / fname
                if not npz_path.exists():
                    np.savez_compressed(npz_path, neural_features=arr)
                all_records[split].append({
                    "path":        str(npz_path.relative_to(out_dir)),
                    "subset":      subset,
                    "session":     fp.stem,
                    "day_idx":     day_idx,
                    "trial_id":    t_idx,
                    "n_channels":  N_CH,
                    "n_time_bins": arr.shape[1],
                })

    for split, recs in all_records.items():
        idx_path = out_dir / f"index_{split}.jsonl"
        with open(idx_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        total_bins = sum(r["n_time_bins"] for r in recs)
        print(f"\n[karpowicz] {split}: {len(recs)} trials  "
              f"{total_bins:,} bins  {total_bins*BIN_S/3600:.3f}h")

    print(f"\n[karpowicz] done. Output in {out_dir}")


def load_karpowicz(data_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Lazy loader: reads index JSONL, returns records with npz_path + metadata."""
    data_dir = Path(data_dir)
    dataset_idx = 203  # karpowicz_falcon in DATASET_TO_IDX
    splits = {}
    for split in ("train", "val", "test"):
        idx_path = data_dir / f"index_{split}.jsonl"
        if not idx_path.exists():
            raise FileNotFoundError(f"{idx_path} — run prepare_karpowicz() first")
        records = []
        with open(idx_path) as f:
            for line in f:
                entry = json.loads(line)
                records.append({
                    "npz_path":           str(data_dir / entry["path"]),
                    "npz_n_channels":     entry["n_channels"],
                    "npz_source_dataset": "karpowicz_falcon",
                    "dataset_idx":        dataset_idx,
                    "day_idx":            np.asarray(entry["day_idx"]),
                    "block_idx":          np.asarray(entry["trial_id"]),
                })
        splits[split] = records
        print(f"[karpowicz] {split}: {len(records)} records", flush=True)
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",  default=str(OUT_DIR))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    args = parser.parse_args()
    prepare_karpowicz(out_dir=args.out_dir, data_dir=args.data_dir)
