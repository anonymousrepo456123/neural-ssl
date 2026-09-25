"""Preprocessing for the O'Doherty dataset (M1+S1 MUA, spike times, .mat HDF5).

Two segmentation modes (--segment):
    fixed (default): fixed 3s windows (150 bins × 20ms)
    reach:           one window per reach, segmented by target_pos changes
                     (filter: 200ms ≤ duration ≤ 6000ms)

Common pipeline:
    1. Reads spike times from segment 0 (standard threshold crossing)
    2. Bins at 20ms per channel (bins in absolute seconds)
    3. Cross-day z-score per channel (online formula, over the full session)
    4. Segments (fixed or reach)
    5. Saves NPZ + index JSONL

Subjects (treated separately due to different n_channels):
    indy_96   — 30 sessions, 96 channels (M1 only)
    indy_192  — 7  sessions, 192 channels (M1 + S1)
    loco_192  — 10 sessions, 192 channels (M1 + S1)

Usage:
    python prepare_odoherty.py --out_dir ./data/odoherty/processed
    python prepare_odoherty.py --out_dir ./data/odoherty/processed --segment reach

NPZ format: neural_features shape (N_ch, T_trial)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import h5py
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR    = Path('./data/odoherty')
BIN_S       = 0.020          # 20ms
WINDOW_BINS = 150            # 3s window (150 bins × 20ms)
SEG_IDX     = 0              # use segment 0 only


# ── reading spike times from segment 0 ───────────────────────────────────────

def read_spikes_seg0(fp: Path) -> Tuple[List[np.ndarray], int]:
    """Read spike times of segment 0 for all channels.

    Returns (spike_times_list, n_channels) where spike_times_list[i] is
    a 1D array of timestamps in seconds for channel i.
    """
    with h5py.File(fp, 'r') as f:
        spk    = f['spikes']
        n_seg, n_ch = spk.shape
        spike_times = []
        for ch in range(n_ch):
            ref = spk[SEG_IDX, ch]
            val = f[ref][:].flatten()
            # Remove dummy spikes (val==0 when the channel is empty)
            val = val[val > 0]
            spike_times.append(val.astype(np.float64))
    return spike_times, n_ch


# ── binning spike times → counts at 20ms ────────────────────────────────────

def bin_session(fp: Path) -> np.ndarray:
    """Read a file and return counts at 20ms, shape (N_ch, T).

    T is determined by the temporal range of the spikes.
    """
    spike_times, n_ch = read_spikes_seg0(fp)

    # Determine t_max from non-empty channels
    t_max = max((st.max() for st in spike_times if len(st) > 0), default=0.0)
    T = int(np.ceil(t_max / BIN_S)) + 1

    counts = np.zeros((n_ch, T), dtype=np.float32)
    for ch, st in enumerate(spike_times):
        if len(st) == 0:
            continue
        bidx = np.clip(np.floor(st / BIN_S).astype(np.int64), 0, T - 1)
        np.add.at(counts[ch], bidx, 1)
    return counts


# ── group files by subject ────────────────────────────────────────────────────

def group_files(data_dir: Path) -> Dict[str, List[Path]]:
    """Group files by subject: indy_96, indy_192, loco_192."""
    groups: Dict[str, List[Path]] = {}
    for fp in sorted(data_dir.glob('*.mat')):
        with h5py.File(fp, 'r') as f:
            n_ch = f['spikes'].shape[1]
        prefix = fp.name.split('_')[0]   # 'indy' or 'loco'
        key = f'{prefix}_{n_ch}'
        groups.setdefault(key, []).append(fp)
    return groups


# ── global statistics (online formula) ───────────────────────────────────────

def compute_subject_stats(files: List[Path], n_ch: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std pooling all sessions. Returns (mean, std) (N_ch,)."""
    sum_x  = np.zeros(n_ch, dtype=np.float64)
    sum_x2 = np.zeros(n_ch, dtype=np.float64)
    n_tot  = 0

    for fp in files:
        print(f'    {fp.name}...', flush=True)
        counts = bin_session(fp).astype(np.float64)   # (N_ch, T)
        sum_x  += counts.sum(axis=1)
        sum_x2 += (counts ** 2).sum(axis=1)
        n_tot  += counts.shape[1]

    mean = sum_x / n_tot
    std  = np.sqrt(np.maximum(sum_x2 / n_tot - mean ** 2, 0.0)) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


# ── segmentation into fixed windows ──────────────────────────────────────────

def extract_windows(
    fp: Path,
    mean: np.ndarray,
    std: np.ndarray,
    window_bins: int = WINDOW_BINS,
) -> List[np.ndarray]:
    """Bin, z-score and return windows of window_bins bins (3s), shape (N_ch, W)."""
    counts = bin_session(fp)                             # (N_ch, T)
    counts_z = (counts - mean[:, None]) / std[:, None]  # z-score per channel

    T = counts_z.shape[1]
    n_wins = T // window_bins
    windows = []
    for i in range(n_wins):
        w = counts_z[:, i * window_bins:(i + 1) * window_bins].astype(np.float32)
        windows.append(w)
    return windows


# ── reach-based segmentation ──────────────────────────────────────────────────

REACH_MIN_MS = 200    # discard spurious target changes (< 200ms)
REACH_MAX_MS = 6000   # discard very long idle periods (> 6s)


def find_reach_boundaries(fp: Path, min_ms: int = REACH_MIN_MS, max_ms: int = REACH_MAX_MS):
    """Read target_pos and t, return list of (t_start_s, t_end_s) for valid reaches."""
    with h5py.File(fp, 'r') as f:
        t          = f['t'][:].flatten()           # (k,) absolute seconds, 4ms step
        target_pos = f['target_pos'][:].T          # (k, 2)

    # Each target change indicates a new reach
    changes = np.where(np.any(np.diff(target_pos, axis=0) != 0, axis=1))[0] + 1
    boundaries_idx = np.concatenate([[0], changes, [len(t)]])
    starts = boundaries_idx[:-1]
    ends   = boundaries_idx[1:]
    durations_ms = (ends - starts) * 4  # 4ms per sample

    mask = (durations_ms >= min_ms) & (durations_ms <= max_ms)
    return list(zip(t[starts[mask]], t[ends[mask] - 1]))


def extract_reach_windows(
    fp: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> List[np.ndarray]:
    """Bin, z-score and return an array for each valid reach, shape (N_ch, T_reach)."""
    counts = bin_session(fp)                              # (N_ch, T_total) — absolute bins
    counts_z = (counts - mean[:, None]) / std[:, None]

    reach_bounds = find_reach_boundaries(fp)
    windows = []
    for t_start, t_end in reach_bounds:
        b_start = int(np.floor(t_start / BIN_S))
        b_end   = int(np.floor(t_end   / BIN_S)) + 1
        b_start = max(0, b_start)
        b_end   = min(counts_z.shape[1], b_end)
        if b_end > b_start:
            windows.append(counts_z[:, b_start:b_end].astype(np.float32))
    return windows


# ── full pipeline ─────────────────────────────────────────────────────────────

def zscore_and_save(
    out_dir: str | Path,
    data_dir: str | Path = DATA_DIR,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
    segment: str = 'fixed',
) -> None:
    """Preprocess all O'Doherty subjects and save NPZ + index JSONL.

    segment: 'fixed' (3s windows) | 'reach' (one window per reach)

    Layout:
        out_dir/
            <subj>/
                stats/mean.npy
                stats/std.npy
                windows/<date>_w<idx>.npz
            index_train.jsonl
            index_val.jsonl
            index_test.jsonl
    """
    assert segment in ('fixed', 'reach'), f"segment must be 'fixed' or 'reach', got {segment!r}"
    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = group_files(data_dir)
    print(f'[odoherty] subjects found: {sorted(groups.keys())}')
    print(f'[odoherty] segmentation: {segment}')

    all_records: Dict[str, List] = {'train': [], 'val': [], 'test': []}

    for subj, files in sorted(groups.items()):
        n_ch = int(subj.split('_')[1])
        print(f'\n[odoherty] {subj}: {len(files)} sessions, {n_ch} channels')

        subj_out = out_dir / subj
        (subj_out / 'stats').mkdir(parents=True, exist_ok=True)
        (subj_out / 'windows').mkdir(parents=True, exist_ok=True)

        # Global statistics (always computed over the full binned session)
        mean_f = subj_out / 'stats' / 'mean.npy'
        std_f  = subj_out / 'stats' / 'std.npy'
        if mean_f.exists() and std_f.exists():
            mean = np.load(mean_f)
            std  = np.load(std_f)
            print(f'  statistics loaded from disk')
        else:
            print(f'  computing mean/std...')
            mean, std = compute_subject_stats(files, n_ch)
            np.save(mean_f, mean)
            np.save(std_f, std)

        # Temporal split (per session, not per trial)
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
                parts = fp.stem.split('_')
                date  = parts[1]
                print(f'  [{split}] {fp.name}...', flush=True)

                if segment == 'reach':
                    windows = extract_reach_windows(fp, mean, std)
                else:
                    windows = extract_windows(fp, mean, std)

                print(f'    → {len(windows)} windows', flush=True)
                for w_idx, arr in enumerate(windows):
                    fname    = f'{date}_w{w_idx:04d}.npz'
                    npz_path = subj_out / 'windows' / fname
                    if not npz_path.exists():
                        np.savez_compressed(npz_path, neural_features=arr)
                    all_records[split].append({
                        'path':        str(npz_path.relative_to(out_dir)),
                        'subject':     subj,
                        'n_channels':  n_ch,
                        'date':        date,
                        'day_idx':     day_idx,
                        'window_id':   w_idx,
                        'n_time_bins': arr.shape[1],
                        'segment':     segment,
                    })
            n_wins = sum(1 for r in all_records[split] if r['subject'] == subj)
            print(f'  {split}: {n_wins} windows')

    for split, recs in all_records.items():
        idx_path = out_dir / f'index_{split}.jsonl'
        with open(idx_path, 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        print(f'\n[odoherty] {split} total: {len(recs)} windows')

    print(f'\n[odoherty] done. Output in {out_dir}')


# ── loader ────────────────────────────────────────────────────────────────────

# Subject → dataset_idx map (aliased to neural pile idx where they exist)
DATASET_IDX_MAP: Dict[str, int] = {
    "indy_96":  19,   # → neural_pile_indy_makin  (96ch M1)
    "indy_192": 82,   # new subject              (192ch M1+S1)
    "loco_192": 20,   # → neural_pile_loco_makin  (192ch M1+S1)
}


def load_odoherty(
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
                    'npz_source_dataset': f'odoherty_{subj}',
                    'dataset_idx':        dataset_idx_map[subj],
                    'day_idx':            np.asarray(entry['day_idx']),
                    'block_idx':          np.asarray(entry['window_id']),
                })
        splits[split] = records
        print(f'[odoherty] {split}: {len(records)} records', flush=True)
    return splits


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',    default='./data/odoherty/processed_reach')
    parser.add_argument('--data_dir',   default=str(DATA_DIR))
    parser.add_argument('--val_ratio',  type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--segment',    default='reach', choices=['fixed', 'reach'],
                        help="'fixed': 3s windows | 'reach': one window per reach (default)")
    args = parser.parse_args()
    zscore_and_save(args.out_dir, args.data_dir, args.val_ratio, args.test_ratio, args.segment)
