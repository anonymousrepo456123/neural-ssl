"""Preprocessing for the Ma dataset (M1 MUA, Utah array, .mat HDF5/XDS).

Pipeline:
    1. Identify channels common to all sessions of each subject
    2. Resample 1ms → 20ms (sum of 20 consecutive bins)
    3. Cross-day z-score per channel (online formula)
    4. Extract trials and save as NPZ + index JSONL

Subjects:
    Chewie_CO_2016    — 12 sessions, 92 common channels
    Greyson_Key_2019  — 9  sessions, 96 common channels (fixed)
    Jango_ISO_2015    — 20 sessions, 85 common channels
    Mihili_CO_2014    — 11 sessions, 94 common channels
    Mihili_RT_2013_2014 — 11 sessions, 94 common channels
    Spike_ISO_2012    — 18 sessions, 73 common channels (fixed)

Usage:
    python prepare_ma.py --out_dir ./data/ma/processed

NPZ format: neural_features shape (N_ch_common, T_trial)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import h5py
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path('./data/ma')
RESAMPLE_F = 20      # 1ms → 20ms
BIN_SRC_MS = 1
BIN_TGT_MS = 20


# ── reading metadata from XDS ────────────────────────────────────────────────

def _read_unit_names(f: h5py.File) -> List[str]:
    refs = f['xds/unit_names'][:, 0]
    return [''.join(chr(int(c)) for c in f[r][:].flatten()) for r in refs]


def get_elec_ids(fp: Path) -> List[int]:
    """Return sorted list of electrode IDs present in the file."""
    with h5py.File(fp, 'r') as f:
        names = _read_unit_names(f)
    return sorted(int(n.replace('elec', '')) for n in names)


def _make_ch_idx(elec_ids_in_file: List[int], common_elec: List[int]):
    """Return (sorted_row_idx, inv_perm) for reading from h5py in ascending order.

    h5py requires fancy indexing in ascending order. We read with sorted_row_idx
    (sorted) and then apply inv_perm to remap rows back to the order of common_elec.
    """
    elec_to_row = {e: i for i, e in enumerate(elec_ids_in_file)}
    ch_idx = [elec_to_row[e] for e in common_elec]        # order of common_elec
    sort_order = np.argsort(ch_idx)                        # to sort ch_idx
    sorted_row_idx = [ch_idx[i] for i in sort_order]      # ascending → h5py ok
    inv_perm = np.argsort(sort_order)                      # to restore common_elec order
    return sorted_row_idx, inv_perm


def get_common_channels(files: List[Path]) -> List[int]:
    """Intersection of electrode IDs across all sessions, sorted."""
    sets = [set(get_elec_ids(fp)) for fp in files]
    return sorted(set.intersection(*sets))


# ── step 1: resample 1ms → 20ms ──────────────────────────────────────────────

def resample_20ms(counts: np.ndarray) -> np.ndarray:
    """(N_ch, T) at 1ms → (N_ch, T//20) at 20ms by summing blocks of 20 bins."""
    N, T = counts.shape
    T_trim = (T // RESAMPLE_F) * RESAMPLE_F
    return counts[:, :T_trim].reshape(N, -1, RESAMPLE_F).sum(axis=2)


def resample_idx(idx: np.ndarray) -> np.ndarray:
    """Convert temporal indices from 1ms to 20ms."""
    return idx // RESAMPLE_F


# ── step 2: compute global statistics per subject ────────────────────────────

def compute_subject_stats(
    files: List[Path],
    common_elec: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std pooling all sessions (online formula).

    Considers only the common channels (common_elec).
    Returns (mean, std) of shape (N_common,).
    """
    N = len(common_elec)
    sum_x  = np.zeros(N, dtype=np.float64)
    sum_x2 = np.zeros(N, dtype=np.float64)
    n_tot  = 0

    for fp in files:
        with h5py.File(fp, 'r') as f:
            names    = _read_unit_names(f)
            elec_ids = [int(n.replace('elec', '')) for n in names]
            sorted_idx, inv_perm = _make_ch_idx(elec_ids, common_elec)
            sc = f['xds/spike_counts'][sorted_idx, :][inv_perm, :]  # (N_common, T_1ms)

        sc20 = resample_20ms(sc.astype(np.float64))  # (N_common, T_20ms)
        sum_x  += sc20.sum(axis=1)
        sum_x2 += (sc20 ** 2).sum(axis=1)
        n_tot  += sc20.shape[1]

    mean = sum_x / n_tot
    std  = np.sqrt(np.maximum(sum_x2 / n_tot - mean ** 2, 0.0)) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


# ── step 3: extract z-scored trials ──────────────────────────────────────────

def extract_trials(
    fp: Path,
    common_elec: List[int],
    mean: np.ndarray,
    std: np.ndarray,
) -> List[np.ndarray]:
    """Resample, z-score and return a list of (N_common, T_trial) arrays."""
    with h5py.File(fp, 'r') as f:
        names    = _read_unit_names(f)
        elec_ids = [int(n.replace('elec', '')) for n in names]
        sorted_idx, inv_perm = _make_ch_idx(elec_ids, common_elec)
        sc = f['xds/spike_counts'][sorted_idx, :][inv_perm, :].astype(np.float32)
        t_start  = f['xds/trial_start_time'][0, :]   # (N_trials,) in seconds
        t_end    = f['xds/trial_end_time'][0, :]
        time_frame = f['xds/time_frame'][0, :]        # (T,) timestamp at 1ms

    sc20    = resample_20ms(sc)                        # (N_common, T_20ms)
    sc20_z  = (sc20 - mean[:, None]) / std[:, None]   # z-score per channel

    # time_frame at 1ms → indices at 20ms
    # More robust than division: searchsorted on time_frame then // RESAMPLE_F
    trials = []
    T = sc20_z.shape[1]
    for s, e in zip(t_start, t_end):
        i0_1ms = int(np.searchsorted(time_frame, s))
        i1_1ms = int(np.searchsorted(time_frame, e, side='right'))
        i0 = i0_1ms // RESAMPLE_F
        i1 = min(i1_1ms // RESAMPLE_F, T)
        if i1 <= i0:
            continue
        trials.append(sc20_z[:, i0:i1].astype(np.float32))  # (N_common, T_trial)
    return trials


# ── full pipeline ─────────────────────────────────────────────────────────────

def zscore_and_save(
    out_dir: str | Path,
    data_dir: str | Path = DATA_DIR,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """Preprocess all Ma subjects and save NPZ + index JSONL.

    Layout:
        out_dir/
            <subj>/
                stats/mean.npy
                stats/std.npy
                stats/common_elec.json    list of electrode IDs used
                trials/<date>_t<idx>.npz
            index_train.jsonl   (all subjects)
            index_val.jsonl
            index_test.jsonl
    """
    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subj_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

    all_records: Dict[str, List] = {'train': [], 'val': [], 'test': []}

    for subj_dir in subj_dirs:
        files = sorted(subj_dir.glob('*.mat'))
        if not files:
            continue
        subj = subj_dir.name
        print(f'\n[ma] {subj}: {len(files)} sessions')

        subj_out = out_dir / subj
        (subj_out / 'stats').mkdir(parents=True, exist_ok=True)
        (subj_out / 'trials').mkdir(parents=True, exist_ok=True)

        # Common channels
        elec_file = subj_out / 'stats' / 'common_elec.json'
        if elec_file.exists():
            common_elec = json.loads(elec_file.read_text())
        else:
            common_elec = get_common_channels(files)
            elec_file.write_text(json.dumps(common_elec))
        print(f'  common channels: {len(common_elec)}  ({common_elec[0]}–{common_elec[-1]})')

        # Global statistics
        mean_f = subj_out / 'stats' / 'mean.npy'
        std_f  = subj_out / 'stats' / 'std.npy'
        if mean_f.exists() and std_f.exists():
            mean = np.load(mean_f)
            std  = np.load(std_f)
            print(f'  statistics loaded from disk')
        else:
            print(f'  computing mean/std...')
            mean, std = compute_subject_stats(files, common_elec)
            np.save(mean_f, mean)
            np.save(std_f, std)

        # Temporal split (sessions sorted by date)
        n = len(files)
        n_test  = max(1, round(n * test_ratio)) if n > 2 else 0
        n_val   = max(1, round(n * val_ratio))  if n > 2 else 0
        n_train = n - n_test - n_val
        splits_files = {
            'train': files[:n_train],
            'val':   files[n_train:n_train + n_val],
            'test':  files[n_train + n_val:],
        }
        if n <= 2:
            splits_files = {'train': files, 'val': [], 'test': []}

        for split, split_fs in splits_files.items():
            for day_idx, fp in enumerate(split_fs):
                date = fp.stem[len(subj.split('_')[0])+1:len(subj.split('_')[0])+9]
                trials = extract_trials(fp, common_elec, mean, std)
                for t_idx, arr in enumerate(trials):
                    fname    = f'{date}_t{t_idx:04d}.npz'
                    npz_path = subj_out / 'trials' / fname
                    if not npz_path.exists():
                        np.savez_compressed(npz_path, neural_features=arr)
                    all_records[split].append({
                        'path':        str(npz_path.relative_to(out_dir)),
                        'subject':     subj,
                        'n_channels':  len(common_elec),
                        'date':        date,
                        'day_idx':     day_idx,
                        'trial_id':    t_idx,
                        'n_time_bins': int(arr.shape[1]),
                    })
            print(f'  {split}: {sum(1 for r in all_records[split] if r["subject"]==subj)} trials')

    # Write global index files
    for split, recs in all_records.items():
        idx_path = out_dir / f'index_{split}.jsonl'
        with open(idx_path, 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        print(f'\n[ma] {split} total: {len(recs)} trials')

    print(f'\n[ma] done. Output in {out_dir}')


# ── loader ────────────────────────────────────────────────────────────────────

# subject_dir_name → dataset_idx map (from DATASET_TO_IDX in utils/datasets.py)
DATASET_IDX_MAP: Dict[str, int] = {
    "Chewie_CO_2016":      76,
    "Greyson_Key_2019":    77,
    "Jango_ISO_2015":      78,
    "Mihili_CO_2014":      79,
    "Mihili_RT_2013_2014": 80,
    "Spike_ISO_2012":      81,
}


def load_ma(
    data_dir: str | Path,
    dataset_idx_map: Dict[str, int] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Lazy loader compatible with MultiSpikingDataset.

    dataset_idx_map: maps subject name to int.
    If None, uses DATASET_IDX_MAP (values from DATASET_TO_IDX in utils/datasets.py).
    """
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
                    'npz_source_dataset': f'ma_{subj}',
                    'dataset_idx':        dataset_idx_map[subj],
                    'day_idx':            np.asarray(entry['day_idx']),
                    'block_idx':          np.asarray(0),
                })
        splits[split] = records
        print(f'[ma] {split}: {len(records)} records', flush=True)
    return splits


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',    default='./data/ma/processed')
    parser.add_argument('--data_dir',   default=str(DATA_DIR))
    parser.add_argument('--val_ratio',  type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    args = parser.parse_args()
    zscore_and_save(args.out_dir, args.data_dir, args.val_ratio, args.test_ratio)
