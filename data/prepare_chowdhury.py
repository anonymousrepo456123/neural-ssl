"""Preprocessing for the Chowdhury dataset (S1 reaching, 3 monkeys, .mat v5).

Pipeline:
    1. Resample 10ms → 20ms (sum of 2 consecutive bins)
    2. Z-score per channel
Usage:
    from data.prepare_chowdhury import zscore_and_save, load_chowdhury
    zscore_and_save('./data/chowdhury/processed
    records = load_chowdhury('./data/chowdhury/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import scipy.io

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path('./data/chowdhury/reaching_experiments')
BIN_SRC_S  = 0.01   # original bin: 10ms
BIN_TGT_S  = 0.02   # target bin:   20ms
RESAMPLE_F = 2      # source bins per target bin


# ── raw data loading ──────────────────────────────────────────────────────────

def _load_mat(path: Path) -> List[Dict]:
    """Load a Chowdhury .mat file and return a list of sessions (1 or 2 per file)."""
    raw = scipy.io.loadmat(str(path), simplify_cells=True)
    td = raw['trial_data']
    return [td] if isinstance(td, dict) else list(td)


def load_all_sessions(data_dir: Path = DATA_DIR) -> List[Dict]:
    """Load all sessions in alphabetical file order."""
    sessions = []
    for fp in sorted(data_dir.glob('*.mat')):
        for s in _load_mat(fp):
            s['_file'] = fp.name
            sessions.append(s)
    return sessions


# ── step 1: resample 10ms → 20ms ─────────────────────────────────────────────

def resample_session(spikes_10ms: np.ndarray) -> np.ndarray:
    """(T, N) at 10ms → (T//2, N) at 20ms by summing pairs of bins."""
    T, N = spikes_10ms.shape
    T_even = (T // RESAMPLE_F) * RESAMPLE_F
    return spikes_10ms[:T_even].reshape(-1, RESAMPLE_F, N).sum(axis=1)


def resample_indices(idx: np.ndarray) -> np.ndarray:
    """Convert indices from 10ms to 20ms (integer division)."""
    return idx // RESAMPLE_F


# ── step 2: group sessions by subject ────────────────────────────────────────

def group_sessions_by_subject(sessions: List[Dict]) -> Dict[str, List[int]]:
    """Group session indices by subject (monkey, n_units).

    Returns a dict:
        key: "C_74", "C_82", "C_98", "H_122", ...
        value: list of indices into `sessions`
    """
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(sessions):
        mk     = s['monkey']
        n_unit = s['S1_spikes'].shape[1]
        key    = f"{mk}_{n_unit}"
        groups.setdefault(key, []).append(i)
    return groups


# ── step 3: z-score per subject ───────────────────────────────────────────────

def compute_subject_stats(
    sessions: List[Dict],
    sess_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std pooling all sessions for this subject.

    Uses the online formula to avoid loading everything into RAM.
    Returns (mean, std) of shape (N_units,).
    """
    n_ch = sessions[sess_indices[0]]['S1_spikes'].shape[1]
    sum_x  = np.zeros(n_ch, dtype=np.float64)
    sum_x2 = np.zeros(n_ch, dtype=np.float64)
    n_tot  = 0

    for i in sess_indices:
        sp = resample_session(sessions[i]['S1_spikes']).astype(np.float64)
        sum_x  += sp.sum(axis=0)
        sum_x2 += (sp ** 2).sum(axis=0)
        n_tot  += sp.shape[0]

    mean = sum_x / n_tot
    std  = np.sqrt(np.maximum(sum_x2 / n_tot - mean ** 2, 0.0)) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_spikes(
    spikes_10ms: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Resample to 20ms and z-score. Returns (T//2, N) float32."""
    sp20 = resample_session(spikes_10ms).astype(np.float32)
    return (sp20 - mean) / std


# ── step 4: extract trials ────────────────────────────────────────────────────

def extract_trials(
    spikes_z: np.ndarray,           # (T, N) z-scored at 20ms
    idx_start: np.ndarray,          # indices at 10ms
    idx_end:   np.ndarray,          # indices at 10ms
) -> List[np.ndarray]:
    """Extract each trial as an (N, T_trial) array at 20ms."""
    starts20 = resample_indices(idx_start)
    ends20   = resample_indices(idx_end)
    T = spikes_z.shape[0]
    trials = []
    for s, e in zip(starts20, ends20):
        s, e = int(s), int(e)
        e = min(e, T)
        if e <= s:
            continue
        trials.append(spikes_z[s:e, :].T.astype(np.float32))  # (N, T_trial)
    return trials


# ── full pipeline ─────────────────────────────────────────────────────────────

def zscore_and_save(
    out_dir: str | Path,
    data_dir: str | Path = DATA_DIR,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Preprocess and save all Chowdhury trials as NPZ + index JSONL.

    Layout:
        out_dir/
            stats/
                <subject_key>_mean.npy     e.g. C_98_mean.npy
                <subject_key>_std.npy
            trials/
                <monkey>_<n_units>_<date>_t<trial_idx>.npz
            index_train.jsonl
            index_val.jsonl
            index_test.jsonl
    """
    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    (out_dir / 'stats').mkdir(parents=True, exist_ok=True)
    (out_dir / 'trials').mkdir(parents=True, exist_ok=True)

    print('[chowdhury] loading sessions...')
    sessions = load_all_sessions(data_dir)
    groups   = group_sessions_by_subject(sessions)

    print(f'[chowdhury] {len(sessions)} sessions → {len(groups)} subjects:')
    for k, idxs in sorted(groups.items()):
        n_sess = len(idxs)
        flag   = '← cross-day z-score' if n_sess > 1 else ''
        print(f'  {k}: {n_sess} session(s)  {flag}')

    # Compute and save statistics for each subject
    subject_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for subj_key, idxs in groups.items():
        mp = out_dir / 'stats' / f'{subj_key}_mean.npy'
        sp = out_dir / 'stats' / f'{subj_key}_std.npy'
        if mp.exists() and sp.exists():
            mean = np.load(mp)
            std  = np.load(sp)
        else:
            mean, std = compute_subject_stats(sessions, idxs)
            np.save(mp, mean)
            np.save(sp, std)
        subject_stats[subj_key] = (mean, std)

    # Build list of all trials with metadata
    all_records = []
    for subj_key, idxs in groups.items():
        mean, std = subject_stats[subj_key]
        n_units   = sessions[idxs[0]]['S1_spikes'].shape[1]
        for day_idx, sess_i in enumerate(idxs):
            s = sessions[sess_i]
            date = s.get('date_time', '')[:10].replace('/', '')
            spikes_z = zscore_spikes(s['S1_spikes'], mean, std)
            trials   = extract_trials(spikes_z, s['idx_startTime'], s['idx_endTime'])
            for t_idx, trial_arr in enumerate(trials):
                fname = f"{subj_key}_{date}_t{t_idx:04d}.npz"
                npz_path = out_dir / 'trials' / fname
                if not npz_path.exists():
                    np.savez_compressed(npz_path, neural_features=trial_arr)
                all_records.append({
                    'path':        str(npz_path.relative_to(out_dir)),
                    'subject_key': subj_key,
                    'n_units':     n_units,
                    'date':        date,
                    'day_idx':     day_idx,
                    'trial_id':    t_idx,
                    'n_time_bins': int(trial_arr.shape[1]),
                })

    # Temporal split per subject: sessions sorted by date
    # train=70%, val=15%, test=15% of sessions per subject
    rng = np.random.default_rng(seed)
    split_records: Dict[str, List] = {'train': [], 'val': [], 'test': []}

    for subj_key, idxs in groups.items():
        subj_recs = [r for r in all_records if r['subject_key'] == subj_key]
        # group by session (day_idx)
        days = sorted(set(r['day_idx'] for r in subj_recs))
        n = len(days)
        n_test = max(1, round(n * test_ratio)) if n > 2 else 0
        n_val  = max(1, round(n * val_ratio))  if n > 2 else 0
        n_train = n - n_test - n_val

        # temporal split (most recent → test)
        train_days = set(days[:n_train])
        val_days   = set(days[n_train:n_train + n_val])
        test_days  = set(days[n_train + n_val:])

        # if subject has only 1 session: all goes to train
        if n == 1:
            train_days, val_days, test_days = set(days), set(), set()

        for r in subj_recs:
            if r['day_idx'] in train_days:
                split_records['train'].append(r)
            elif r['day_idx'] in val_days:
                split_records['val'].append(r)
            else:
                split_records['test'].append(r)

    for split, recs in split_records.items():
        idx_path = out_dir / f'index_{split}.jsonl'
        with open(idx_path, 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        print(f'[chowdhury] {split}: {len(recs)} trials')

    print(f'[chowdhury] done. Output in {out_dir}')


# ── loader for prepare_data.py ────────────────────────────────────────────────

# subject_key → dataset_idx map (from DATASET_TO_IDX in utils/datasets.py)
DATASET_IDX_MAP: Dict[str, int] = {
    "C_57":  64,
    "C_74":  65,
    "C_82":  66,
    "C_98":  67,
    "C_118": 68,
    "H_102": 69,
    "H_113": 70,
    "H_122": 71,
    "H_155": 72,
    "H_167": 73,
    "L_42":  74,
    "L_58":  75,
}


def load_chowdhury(
    data_dir: str | Path,
    dataset_idx_map: Dict[str, int] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Lazy loader compatible with MultiSpikingDataset.

    dataset_idx_map: maps subject_key (e.g. "C_98") to integer dataset_idx.
    If None, uses DATASET_IDX_MAP (values from DATASET_TO_IDX in utils/datasets.py).
    """
    data_dir = Path(data_dir)
    if dataset_idx_map is None:
        dataset_idx_map = DATASET_IDX_MAP

    splits = {}
    for split in ('train', 'val', 'test'):
        idx_path = data_dir / f'index_{split}.jsonl'
        if not idx_path.exists():
            raise FileNotFoundError(f'{idx_path} not found — run zscore_and_save() first')
        records = []
        with open(idx_path) as f:
            for line in f:
                entry = json.loads(line)
                subj  = entry['subject_key']
                records.append({
                    'npz_path':           str(data_dir / entry['path']),
                    'npz_n_channels':     entry['n_units'],
                    'npz_source_dataset': f'chowdhury_{subj}',
                    'dataset_idx':        dataset_idx_map[subj],
                    'day_idx':            np.asarray(entry['day_idx']),
                    'block_idx':          np.asarray(0),
                })
        splits[split] = records
        print(f'[chowdhury] {split}: {len(records)} records', flush=True)
    return splits


# ── eager loader ──────────────────────────────────────────────────────────────

def load_chowdhury_eager(
    data_dir: str | Path,
    dataset_idx_map: Dict[str, int] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Variant that loads everything into RAM. spikes shape: (T, N)."""
    data_dir = Path(data_dir)
    if dataset_idx_map is None:
        dataset_idx_map = DATASET_IDX_MAP

    splits = {}
    for split in ('train', 'val', 'test'):
        idx_path = data_dir / f'index_{split}.jsonl'
        records = []
        with open(idx_path) as f:
            entries = [json.loads(l) for l in f]
        for entry in entries:
            npz   = np.load(data_dir / entry['path'])
            feats = npz['neural_features']              # (N, T)
            subj  = entry['subject_key']
            records.append({
                'spikes':      feats.T.astype(np.float32),  # (T, N)
                'dataset_idx': dataset_idx_map[subj],
                'day_idx':     np.asarray(entry['day_idx']),
                'block_idx':   np.asarray(0),
            })
        splits[split] = records
        print(f'[chowdhury] eager {split}: {len(records)} records', flush=True)
    return splits


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',    default='./data/chowdhury/processed')
    parser.add_argument('--data_dir',   default=str(DATA_DIR))
    parser.add_argument('--val_ratio',  type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    args = parser.parse_args()
    zscore_and_save(args.out_dir, args.data_dir, args.val_ratio, args.test_ratio)
