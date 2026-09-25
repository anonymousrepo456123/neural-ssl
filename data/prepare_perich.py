"""Preprocessing for the Perich dataset (M1+PMd, center-out/random-target reaching, NWB).

Each session gets a distinct dataset_idx (spike sorting → units are not comparable
across sessions). The session_key → idx map is written to perich_session_registry.json
and loaded dynamically by utils/datasets.py.

Pipeline:
    1. For each NWB file: reads per-unit spike times
    2. Bins at 20ms (within-session)
    3. Z-score within-session per unit (mean/std over the full session)
    4. Extracts trials (start/stop from intervals/trials)
    5. Saves one NPZ per trial + index JSONL
    6. Writes perich_session_registry.json (session_key → idx, n_units, monkey, date)

Split: temporal per monkey — last val_ratio sessions = val, last test_ratio = test.
Sub-J (3 sessions) → all train.

Usage:
    python prepare_perich.py [--out_dir ./data/perich/processed
                             [--data_dir ./data/perich/processed
                             [--val_ratio 0.10] [--test_ratio 0.10]

NPZ format: neural_features shape (n_units, T_trial)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
DATA_DIR    = Path("./data/perich")
OUT_DIR     = Path("./data/perich/processed")
BIN_S       = 0.020   # 20ms
IDX_START   = 83      # first free idx in DATASET_TO_IDX

# ── NWB file reading ────────────────────────────────────────────────────────

def read_nwb_session(fp: Path) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Read per-unit spike times and trial start/stop from an NWB file.

    Returns:
        per_unit_spikes: List[np.ndarray]  — spike times (s) per unit
        t_start: np.ndarray (N_trials,)
        t_stop:  np.ndarray (N_trials,)
    """
    with h5py.File(fp, "r") as f:
        st_flat = f["units/spike_times"][:]
        st_idx  = f["units/spike_times_index"][:]   # cumulative end indices
        t_start = f["intervals/trials/start_time"][:]
        t_stop  = f["intervals/trials/stop_time"][:]

    per_unit_spikes: List[np.ndarray] = []
    prev = 0
    for end in st_idx:
        per_unit_spikes.append(st_flat[prev:end])
        prev = int(end)

    return per_unit_spikes, t_start, t_stop


# ── binning ──────────────────────────────────────────────────────────────────

def bin_session(per_unit_spikes: List[np.ndarray]) -> np.ndarray:
    """Bin spike times into 20ms counts over the full session.

    Returns counts (n_units, T_total).
    """
    n_units = len(per_unit_spikes)
    all_times = np.concatenate([s for s in per_unit_spikes if len(s) > 0], axis=0)
    if len(all_times) == 0:
        return np.zeros((n_units, 1), dtype=np.float32)
    t_max = all_times.max()
    T = int(np.ceil(t_max / BIN_S)) + 1

    counts = np.zeros((n_units, T), dtype=np.float32)
    for u, spikes in enumerate(per_unit_spikes):
        if len(spikes) == 0:
            continue
        bidx = np.clip(np.floor(spikes / BIN_S).astype(np.int64), 0, T - 1)
        np.add.at(counts[u], bidx, 1)
    return counts


def zscore_session(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score within-session per unit.

    Returns (counts_z, mean, std); mean/std are both of shape (n_units,).
    """
    mean = counts.mean(axis=1)          # (n_units,)
    std  = counts.std(axis=1) + 1e-8
    counts_z = (counts - mean[:, None]) / std[:, None]
    return counts_z.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


# ── trial extraction ──────────────────────────────────────────────────────────

def extract_trials(
    counts_z: np.ndarray,
    t_start: np.ndarray,
    t_stop: np.ndarray,
) -> List[np.ndarray]:
    """Extract windows (n_units, T_trial) for each trial.

    Trials with fewer than 2 bins are discarded.
    """
    windows = []
    for ts, te in zip(t_start, t_stop):
        b_start = int(np.floor(ts / BIN_S))
        b_end   = int(np.ceil(te  / BIN_S))
        b_start = max(0, b_start)
        b_end   = min(counts_z.shape[1], b_end)
        if b_end - b_start < 2:
            continue
        windows.append(counts_z[:, b_start:b_end].astype(np.float32))
    return windows


# ── session key ──────────────────────────────────────────────────────────────

def session_key_from_path(fp: Path) -> Tuple[str, str, str]:
    """Derive (session_key, monkey, date) from the NWB file path.

    Example: sub-C/sub-C_ses-CO-20131003_... → ('perich_C_20131003', 'C', '20131003')
    """
    monkey = fp.parts[-2].replace("sub-", "")   # 'C', 'J', 'M', 'T'
    # stem: 'sub-C_ses-CO-20131003_behavior+ecephys'
    parts = fp.stem.split("_ses-")
    date_part = parts[1].split("_")[0]           # 'CO-20131003'
    date = date_part.split("-")[-1]              # '20131003'
    key = f"perich_{monkey}_{date}"
    return key, monkey, date


# ── main pipeline ─────────────────────────────────────────────────────────────

def prepare_perich(
    out_dir: Path = OUT_DIR,
    data_dir: Path = DATA_DIR,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> None:
    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all files per monkey, sorted by date
    by_monkey: Dict[str, List[Path]] = {}
    for fp in sorted(data_dir.rglob("*.nwb")):
        _, monkey, _ = session_key_from_path(fp)
        by_monkey.setdefault(monkey, []).append(fp)

    # Assign deterministic idx: order (monkey, date), starting from IDX_START
    all_files = []
    for monkey in sorted(by_monkey):
        all_files.extend(by_monkey[monkey])
    idx_map: Dict[str, int] = {}
    for i, fp in enumerate(all_files):
        key, _, _ = session_key_from_path(fp)
        idx_map[key] = IDX_START + i

    registry: Dict[str, Dict] = {}
    all_records: Dict[str, List] = {"train": [], "val": [], "test": []}

    for monkey, files in sorted(by_monkey.items()):
        n = len(files)
        # Sub-J has only 3 sessions → all train
        if n <= 3:
            split_files = {"train": files, "val": [], "test": []}
        else:
            n_test  = max(1, round(n * test_ratio))
            n_val   = max(1, round(n * val_ratio))
            n_train = n - n_test - n_val
            split_files = {
                "train": files[:n_train],
                "val":   files[n_train:n_train + n_val],
                "test":  files[n_train + n_val:],
            }

        print(f"\n[perich] sub-{monkey}: {n} sessions  "
              f"(train={len(split_files['train'])} val={len(split_files['val'])} "
              f"test={len(split_files['test'])})")

        for split, split_fs in split_files.items():
            for day_idx, fp in enumerate(split_fs):
                key, _, date = session_key_from_path(fp)
                ses_idx = idx_map[key]
                print(f"  [{split}] {fp.name}  idx={ses_idx}", flush=True)

                per_unit_spikes, t_start, t_stop = read_nwb_session(fp)
                n_units = len(per_unit_spikes)

                counts = bin_session(per_unit_spikes)
                counts_z, mean, std = zscore_session(counts)
                windows = extract_trials(counts_z, t_start, t_stop)
                print(f"    n_units={n_units}  trials={len(windows)}", flush=True)

                # Save statistics + trials
                ses_out = out_dir / f"sub-{monkey}" / key
                (ses_out / "stats").mkdir(parents=True, exist_ok=True)
                (ses_out / "trials").mkdir(parents=True, exist_ok=True)
                np.save(ses_out / "stats" / "mean.npy", mean)
                np.save(ses_out / "stats" / "std.npy",  std)

                for t_idx, arr in enumerate(windows):
                    fname    = f"trial_{t_idx:05d}.npz"
                    npz_path = ses_out / "trials" / fname
                    if not npz_path.exists():
                        np.savez_compressed(npz_path, neural_features=arr)
                    all_records[split].append({
                        "path":           str(npz_path.relative_to(out_dir)),
                        "session_key":    key,
                        "dataset_idx":    ses_idx,
                        "n_channels":     n_units,
                        "monkey":         monkey,
                        "date":           date,
                        "day_idx":        day_idx,
                        "trial_id":       t_idx,
                        "n_time_bins":    arr.shape[1],
                    })

                # Registry entry (written only the first time we see this session)
                if key not in registry:
                    registry[key] = {
                        "idx":     ses_idx,
                        "n_units": n_units,
                        "monkey":  monkey,
                        "date":    date,
                    }

    # Write index JSONL
    for split, recs in all_records.items():
        idx_path = out_dir / f"index_{split}.jsonl"
        with open(idx_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"\n[perich] {split}: {len(recs)} trials total")

    # Write registry
    reg_path = out_dir / "perich_session_registry.json"
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\n[perich] registry written to {reg_path} ({len(registry)} sessions)")
    print(f"[perich] idx range: {IDX_START} – {IDX_START + len(registry) - 1}")
    print(f"[perich] done. Output in {out_dir}")


# ── loader compatible with MultiSpikingDataset ────────────────────────────────

def load_perich(
    data_dir: str | Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Lazy loader: reads index JSONL files and returns records with npz_path + metadata."""
    data_dir = Path(data_dir)
    splits = {}
    for split in ("train", "val", "test"):
        idx_path = data_dir / f"index_{split}.jsonl"
        if not idx_path.exists():
            raise FileNotFoundError(f"{idx_path} — run prepare_perich() first")
        records = []
        with open(idx_path) as f:
            for line in f:
                entry = json.loads(line)
                records.append({
                    "npz_path":           str(data_dir / entry["path"]),
                    "npz_n_channels":     entry["n_channels"],
                    "npz_source_dataset": entry["session_key"],
                    "dataset_idx":        int(entry["dataset_idx"]),
                    "day_idx":            np.asarray(entry["day_idx"]),
                    "block_idx":          np.asarray(entry["trial_id"]),
                })
        splits[split] = records
        print(f"[perich] {split}: {len(records)} records", flush=True)
    return splits


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",    default=str(OUT_DIR))
    parser.add_argument("--data_dir",   default=str(DATA_DIR))
    parser.add_argument("--val_ratio",  type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    args = parser.parse_args()
    prepare_perich(
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
