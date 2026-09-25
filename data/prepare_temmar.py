"""Preprocessing and loading for the Temmar dataset.

Main functions:
    zscore_and_save(out_dir)
        — Computes global mean/std over all 312 sessions and saves the z-scored
          trials as NPZ files compatible with the neural pile format.
          Each NPZ contains neural_features of shape (96, T_trial).

    load_temmar(data_dir, val_ratio, test_ratio, seed)
        — Returns {"train": [...], "val": [...], "test": [...]}
          where each record is a dict with npz_path / npz_n_channels / dataset_idx
          ready for MultiSpikingDataset (lazy loading).

Usage standalone:
    python prepare_temmar.py --out_dir ./data/temmar/processed

Record format compatible with MultiSpikingDataset:
    {
        "npz_path":           str,
        "npz_n_channels":     int  (= 96),
        "npz_source_dataset": str  (= "neural_pile_sub-Monkey-N_temmar"),
        "dataset_idx":        int  (= 38, from DATASET_TO_IDX),
        "day_idx":            np.ndarray scalar,
        "block_idx":          np.ndarray scalar,
    }
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import h5py
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR    = Path('./data/temmar/sub-Monkey-N')
N_CHANNELS  = 96
BIN_S       = 0.02          # 20 ms
SOURCE_NAME = 'neural_pile_sub-Monkey-N_temmar'
DATASET_IDX = 38            # from DATASET_TO_IDX in utils/datasets.py


# ── step 1: compute global statistics ────────────────────────────────────────

def compute_global_stats(nwb_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std pooling all bins across all sessions.

    Uses the online formula (sum / sum^2) to avoid loading everything into RAM.
    Returns (mean, std) of shape (N_CHANNELS,).
    """
    sum_x  = np.zeros(N_CHANNELS, dtype=np.float64)
    sum_x2 = np.zeros(N_CHANNELS, dtype=np.float64)
    n_total = 0

    for fp in nwb_files:
        with h5py.File(fp, 'r') as f:
            tc = f['analysis/ThresholdCrossings/data'][:].astype(np.float64)
        sum_x  += tc.sum(axis=0)
        sum_x2 += (tc ** 2).sum(axis=0)
        n_total += tc.shape[0]

    mean = sum_x / n_total
    std  = np.sqrt(np.maximum(sum_x2 / n_total - mean ** 2, 0.0)) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


# ── step 2: extract z-scored trials and save as NPZ ──────────────────────────

def _extract_trials(fp: Path, mean: np.ndarray, std: np.ndarray
                    ) -> List[np.ndarray]:
    """Return a list of (96, T_trial) z-scored arrays, one per trial."""
    with h5py.File(fp, 'r') as f:
        tc       = f['analysis/ThresholdCrossings/data'][:].astype(np.float32)
        ts       = f['analysis/ThresholdCrossings/timestamps'][:]
        t_start  = f['intervals/trials/start_time'][:]
        t_stop   = f['intervals/trials/stop_time'][:]

    tc_z = (tc - mean) / std   # (T_session, 96)

    trials = []
    for s, e in zip(t_start, t_stop):
        i0 = int(np.searchsorted(ts, s))
        i1 = int(np.searchsorted(ts, e, side='right'))
        if i1 <= i0:
            continue
        chunk = tc_z[i0:i1, :].T.astype(np.float32)  # (96, T_trial)
        trials.append(chunk)
    return trials


def zscore_and_save(
    out_dir: str | Path,
    nwb_dir: str | Path = DATA_DIR,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Compute z-scores and save trials as NPZ + index JSONL.

    Disk layout:
        out_dir/
            stats/global_mean.npy
            stats/global_std.npy
            trials/
                <session_date>_t<trial_idx>.npz
            index_train.jsonl
            index_val.jsonl
            index_test.jsonl
    """
    out_dir  = Path(out_dir)
    nwb_dir  = Path(nwb_dir)
    nwb_files = sorted(nwb_dir.glob('*.nwb'))
    assert nwb_files, f'No .nwb files found in {nwb_dir}'

    # Output folders
    (out_dir / 'stats').mkdir(parents=True, exist_ok=True)
    (out_dir / 'trials').mkdir(parents=True, exist_ok=True)

    # 1. Global statistics
    stats_mean = out_dir / 'stats' / 'global_mean.npy'
    stats_std  = out_dir / 'stats' / 'global_std.npy'
    if stats_mean.exists() and stats_std.exists():
        print('[temmar] loading pre-computed statistics...')
        mean = np.load(stats_mean)
        std  = np.load(stats_std)
    else:
        print(f'[temmar] computing mean/std over {len(nwb_files)} sessions...')
        mean, std = compute_global_stats(nwb_files)
        np.save(stats_mean, mean)
        np.save(stats_std, std)
        print(f'[temmar] statistics saved to {out_dir}/stats/')

    # 2. Split sessions into train/val/test (by date, not random)
    n = len(nwb_files)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    # most recent sessions → test/val (temporal split)
    test_files  = nwb_files[-n_test:]
    val_files   = nwb_files[-(n_test + n_val):-n_test]
    train_files = nwb_files[:-(n_test + n_val)]
    print(f'[temmar] split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}')

    # 3. Extract, z-score, save NPZ
    split_map = {'train': train_files, 'val': val_files, 'test': test_files}
    for split, files in split_map.items():
        records = []
        for day_idx, fp in enumerate(files):
            sess_date = fp.stem[18:26]  # e.g. '20200127'
            trials = _extract_trials(fp, mean, std)
            for t_idx, trial_arr in enumerate(trials):
                fname = f'{sess_date}_t{t_idx:04d}.npz'
                npz_path = out_dir / 'trials' / fname
                if not npz_path.exists():
                    np.savez_compressed(
                        npz_path,
                        neural_features=trial_arr,   # (96, T_trial)
                        source_dataset=SOURCE_NAME,
                        subject_id='sub-Monkey-N',
                        session_id=fp.stem,
                        trial_id=t_idx,
                    )
                records.append({
                    'path':           str(npz_path.relative_to(out_dir)),
                    'source_dataset': SOURCE_NAME,
                    'n_time_bins':    int(trial_arr.shape[1]),
                    'session_date':   sess_date,
                    'trial_id':       t_idx,
                    'day_idx':        day_idx,
                })

        index_path = out_dir / f'index_{split}.jsonl'
        with open(index_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        print(f'[temmar] {split}: {len(records)} trials saved → {index_path}')

    print(f'[temmar] done. Output in {out_dir}')


# ── step 3: loader for prepare_data.py ───────────────────────────────────────

def load_temmar(
    data_dir: str | Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Loader compatible with MultiSpikingDataset.

    Reads the index JSONL files produced by zscore_and_save() and returns
    records with lazy loading (npz_path), ready for MultiSpikingDataset.

    Args:
        data_dir: output directory produced by zscore_and_save()

    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    data_dir = Path(data_dir)
    splits = {}
    for split in ('train', 'val', 'test'):
        index_path = data_dir / f'index_{split}.jsonl'
        if not index_path.exists():
            raise FileNotFoundError(
                f'{index_path} not found. Run zscore_and_save() first.'
            )
        records = []
        with open(index_path) as f:
            for line in f:
                entry = json.loads(line)
                npz_abs = data_dir / entry['path']
                records.append({
                    'npz_path':           str(npz_abs),
                    'npz_n_channels':     N_CHANNELS,
                    'npz_source_dataset': SOURCE_NAME,
                    'dataset_idx':        DATASET_IDX,
                    'day_idx':            np.asarray(entry['day_idx']),
                    'block_idx':          np.asarray(0),
                })
        splits[split] = records
        print(f'[temmar] {split}: {len(records)} records loaded', flush=True)
    return splits


# ── eager loader (alternative without lazy loading) ───────────────────────────

def load_temmar_eager(
    data_dir: str | Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Variant that loads all trials into memory as spikes arrays (T, C).

    Useful for small datasets or debugging. Slower to initialise but
    does not require per-batch I/O.
    """
    data_dir = Path(data_dir)
    splits = {}
    for split in ('train', 'val', 'test'):
        index_path = data_dir / f'index_{split}.jsonl'
        if not index_path.exists():
            raise FileNotFoundError(f'{index_path} not found.')
        records = []
        with open(index_path) as f:
            entries = [json.loads(l) for l in f]
        for entry in entries:
            npz = np.load(data_dir / entry['path'])
            feats = npz['neural_features'][:N_CHANNELS]   # (96, T)
            records.append({
                'spikes':     feats.T.astype(np.float32),  # (T, 96)
                'dataset_idx': DATASET_IDX,
                'day_idx':    np.asarray(entry['day_idx']),
                'block_idx':  np.asarray(0),
            })
        splits[split] = records
        print(f'[temmar] eager {split}: {len(records)} records', flush=True)
    return splits


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./data/temmar/processed')
    parser.add_argument('--nwb_dir', default=str(DATA_DIR))
    parser.add_argument('--val_ratio',  type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    args = parser.parse_args()

    zscore_and_save(
        out_dir=args.out_dir,
        nwb_dir=args.nwb_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
