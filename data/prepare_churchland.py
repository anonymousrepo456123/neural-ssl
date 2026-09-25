"""Preprocessing for the Churchland dataset (M1+PMd, continuous spike times, NWB).

Pipeline:
    1. Bin spike times at 20ms for each of the 192 units
    2. Cross-day z-score per unit (online formula)
    3. Extract trials and save as NPZ + index JSONL

Subjects:
    sub-Jenkins   — 4 sessions, 192 fixed units
    sub-Nitschke  — 6 sessions, 192 fixed units

NWB structure:
    units/spike_times + units/spike_times_index  — continuous spike times (float64, sec)
    intervals/trials/start_time + stop_time       — trial boundaries

Usage:
    python prepare_churchland.py --out_dir ./data/churchland/processed

NPZ format: neural_features shape (192, T_trial)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import h5py
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path('./data/churchland')
N_UNITS  = 192
BIN_S    = 0.020   # 20ms


# ── binning spike times → counts at 20ms ─────────────────────────────────────

def bin_session(fp: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an NWB file and return (counts, t_start, t_stop).

    counts: (N_units, T) float32 — all spike times binned at 20ms
            T spans the full temporal range of the session.
    t_start, t_stop: (N_trials,) float64 in seconds.
    """
    with h5py.File(fp, 'r') as f:
        st_all  = f['units/spike_times'][:]
        st_idx  = f['units/spike_times_index'][:]
        uid     = f['units/id'][:]
        t_start = f['intervals/trials/start_time'][:]
        t_stop  = f['intervals/trials/stop_time'][:]

    t_max = float(st_all.max()) + BIN_S
    T     = int(np.ceil(t_max / BIN_S))

    counts = np.zeros((len(uid), T), dtype=np.float32)
    prev = 0
    for i, end in enumerate(st_idx):
        spk = st_all[prev:int(end)]
        if len(spk):
            bidx = np.clip(np.floor(spk / BIN_S).astype(np.int64), 0, T - 1)
            np.add.at(counts[i], bidx, 1)
        prev = int(end)

    # Sort by unit ID (usually already sorted)
    order = np.argsort(uid)
    return counts[order], t_start, t_stop


# ── global statistics (online formula) ───────────────────────────────────────

def compute_subject_stats(files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-unit mean/std pooling all sessions. Returns (mean, std) of shape (N_UNITS,)."""
    sum_x  = np.zeros(N_UNITS, dtype=np.float64)
    sum_x2 = np.zeros(N_UNITS, dtype=np.float64)
    n_tot  = 0

    for fp in files:
        print(f'    {fp.name[len(fp.parent.name)+5:26]}...', flush=True)
        counts, _, _ = bin_session(fp)
        sum_x  += counts.sum(axis=1).astype(np.float64)
        sum_x2 += (counts.astype(np.float64) ** 2).sum(axis=1)
        n_tot  += counts.shape[1]

    mean = sum_x / n_tot
    std  = np.sqrt(np.maximum(sum_x2 / n_tot - mean ** 2, 0.0)) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


# ── extract z-scored trials ───────────────────────────────────────────────────

def extract_trials(
    fp: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> List[np.ndarray]:
    """Bin, z-score and return a list of (N_UNITS, T_trial) arrays."""
    counts, t_start, t_stop = bin_session(fp)
    counts_z = (counts - mean[:, None]) / std[:, None]
    T = counts_z.shape[1]

    trials = []
    for s, e in zip(t_start, t_stop):
        i0 = int(np.floor(s / BIN_S))
        i1 = min(int(np.ceil(e  / BIN_S)), T)
        if i1 <= i0:
            continue
        trials.append(counts_z[:, i0:i1].astype(np.float32))
    return trials


# ── full pipeline ─────────────────────────────────────────────────────────────

def zscore_and_save(
    out_dir: str | Path,
    data_dir: str | Path = DATA_DIR,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """Preprocess all Churchland subjects and save NPZ + index JSONL.

    Layout:
        out_dir/
            <subj>/
                stats/mean.npy
                stats/std.npy
                trials/<date>_t<idx>.npz
            index_train.jsonl
            index_val.jsonl
            index_test.jsonl
    """
    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, List] = {'train': [], 'val': [], 'test': []}

    for subj_dir in sorted(data_dir.iterdir()):
        files = sorted(subj_dir.glob('*.nwb'))
        if not files:
            continue
        subj = subj_dir.name
        print(f'\n[churchland] {subj}: {len(files)} sessions')

        subj_out = out_dir / subj
        (subj_out / 'stats').mkdir(parents=True, exist_ok=True)
        (subj_out / 'trials').mkdir(parents=True, exist_ok=True)

        # Global statistics
        mean_f = subj_out / 'stats' / 'mean.npy'
        std_f  = subj_out / 'stats' / 'std.npy'
        if mean_f.exists() and std_f.exists():
            mean = np.load(mean_f)
            std  = np.load(std_f)
            print(f'  statistics loaded from disk')
        else:
            print(f'  computing mean/std...')
            mean, std = compute_subject_stats(files)
            np.save(mean_f, mean)
            np.save(std_f, std)

        # Temporal split
        n = len(files)
        n_test  = max(1, round(n * test_ratio)) if n > 2 else 0
        n_val   = max(1, round(n * val_ratio))  if n > 2 else 0
        n_train = n - n_test - n_val
        if n <= 2:
            splits_files = {'train': files, 'val': [], 'test': []}
        else:
            splits_files = {
                'train': files[:n_train],
                'val':   files[n_train:n_train + n_val],
                'test':  files[n_train + n_val:],
            }

        for split, split_fs in splits_files.items():
            for day_idx, fp in enumerate(split_fs):
                date = fp.name[len(subj)+5:len(subj)+13]   # YYYYMMDD
                print(f'  [{split}] {date}...', flush=True)
                trials = extract_trials(fp, mean, std)
                for t_idx, arr in enumerate(trials):
                    fname    = f'{date}_t{t_idx:05d}.npz'
                    npz_path = subj_out / 'trials' / fname
                    if not npz_path.exists():
                        np.savez_compressed(npz_path, neural_features=arr)
                    all_records[split].append({
                        'path':        str(npz_path.relative_to(out_dir)),
                        'subject':     subj,
                        'n_channels':  N_UNITS,
                        'date':        date,
                        'day_idx':     day_idx,
                        'trial_id':    t_idx,
                        'n_time_bins': int(arr.shape[1]),
                    })
            n_tr = sum(1 for r in all_records[split] if r['subject'] == subj)
            print(f'  {split}: {n_tr} trials')

    for split, recs in all_records.items():
        idx_path = out_dir / f'index_{split}.jsonl'
        with open(idx_path, 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        print(f'\n[churchland] {split} total: {len(recs)} trials')

    print(f'\n[churchland] done. Output in {out_dir}')


# ── loader ────────────────────────────────────────────────────────────────────

# Subject → dataset_idx map (aliased to neural pile indices)
DATASET_IDX_MAP: Dict[str, int] = {
    "sub-Jenkins":  33,   # → neural_pile_sub-Jenkins_churchland
    "sub-Nitschke": 43,   # → neural_pile_sub-Nitschke_churchland
}


def load_churchland(
    data_dir: str | Path,
    dataset_idx_map: Dict[str, int] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Lazy loader compatible with MultiSpikingDataset."""
    data_dir = Path(data_dir)
    if dataset_idx_map is None:
        dataset_idx_map = DATASET_IDX_MAP

    splits = {}
    for split in ('train', 'val', 'test'):
        idx_path = data_dir / f'index_{split}.jsonl'
        if not idx_path.exists():
            raise FileNotFoundError(f'{idx_path} — run zscore_and_save() first')
        records = []
        with open(idx_path) as f:
            for line in f:
                entry = json.loads(line)
                subj  = entry['subject']
                records.append({
                    'npz_path':           str(data_dir / entry['path']),
                    'npz_n_channels':     entry['n_channels'],
                    'npz_source_dataset': f'churchland_{subj}',
                    'dataset_idx':        dataset_idx_map[subj],
                    'day_idx':            np.asarray(entry['day_idx']),
                    'block_idx':          np.asarray(0),
                })
        splits[split] = records
        print(f'[churchland] {split}: {len(records)} records', flush=True)
    return splits


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',    default='./data/churchland/processed')
    parser.add_argument('--data_dir',   default=str(DATA_DIR))
    parser.add_argument('--val_ratio',  type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    args = parser.parse_args()
    zscore_and_save(args.out_dir, args.data_dir, args.val_ratio, args.test_ratio)
