"""Ray-Tune-style random hyperparameter sweep for BIT phoneme finetuning.

"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PRETRAINED = os.path.join(_HERE, "checkpoints", "bit_mae_zhang_humans_v2", "BEST")
BATCH_CHOICES = [8, 16, 32, 64]
LR_RANGE = (5e-5, 1e-3)
WD_RANGE = (5e-5, 1e-1)

_PER_GLOBAL_RE = re.compile(r"(?:PER|CER) \(global\):\s+([0-9.]+)")
_PER_BATCH_RE = re.compile(r"(?:PER|CER) \(trainer, mean batch\):\s+([0-9.]+)")


def loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_configs(n_runs: int, seed: int,
                   lr_range=LR_RANGE, wd_range=WD_RANGE,
                   batch_choices=BATCH_CHOICES,
                   mask_choices=None) -> List[Dict]:
    """Reproducibly sample N (batch, lr, wd[, mask_ratio]) configs.

    When mask_choices is given (e.g. [0.0, 0.25, 0.5]), each config gets a
    fixed mask value assigned by cycling through the list (idx % len). With
    30 configs and 3 mask values → 10 configs per mask value.
    """
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n_runs):
        cfg = {
            "idx": i,
            "batch_size": int(rng.choice(batch_choices)),
            "lr": loguniform(rng, *lr_range),
            "wd": loguniform(rng, *wd_range),
        }
        if mask_choices is not None:
            cfg["mask_ratio"] = float(mask_choices[i % len(mask_choices)])
        configs.append(cfg)
    return configs


def savestring_for(tag: str, subject: str, idx: int) -> str:
    return f"sweep_{tag}_{subject}_{idx:02d}"


def build_command(cfg: Dict, args, savestring: str, log_path: str) -> str:
    """A single shell command: finetune, then eval val PER, appending to log."""
    ckpt_base = getattr(args, "checkpoint_dir", None) or os.path.join(_HERE, "checkpoints")
    ckpt_dir = os.path.join(ckpt_base, savestring, "BEST")
    py = sys.executable
    # Per-config mask takes precedence over the global --mask-ratio arg.
    _mr = cfg.get("mask_ratio")
    if _mr is None:
        _mr = getattr(args, "mask_ratio", None)
    _mask_flag = f"--mask-ratio {_mr} " if _mr is not None else ""
    _extra_kw = (" ".join(getattr(args, "extra_kwargs", None) or []) + " ")
    train = (
        f"{py} -u finetune.py "
        f"--subject {args.subject} "
        f"--pretrained_ckpt {args.pretrained} "
        f"{'--reinit-readin ' if getattr(args, 'reinit_readin', False) else ''}"
        f"{_mask_flag}"
        f"--kwargs "
        f"optimizer.lr={cfg['lr']:.6e} "
        f"optimizer.wd={cfg['wd']:.6e} "
        f"training.train_batch_size={cfg['batch_size']} "
        f"training.test_batch_size={cfg['batch_size']} "
        f"training.num_epochs={args.epochs} "
        f"training.num_workers={args.num_workers} "
        f"training.save_every=100000 "
        f"savestring={savestring} "
        f"dirs.checkpoint_dir={ckpt_base} "
        f"{_extra_kw}"
        f"wandb_project={args.wandb_project}"
    )
    evl = (
        f"{py} -u eval_phoneme_per.py "
        f"--subject {args.subject} "
        f"--checkpoint {ckpt_dir} "
        f"--split {args.eval_split}"
    )
    if args.no_eval:
        return f"set -o pipefail; {train} 2>&1 | tee {log_path}"
    # Only eval if training actually produced a BEST checkpoint — a crashed
    # train otherwise triggers a confusing FileNotFoundError in the eval.
    return (
        f"set -o pipefail; "
        f"{{ {train}; }} 2>&1 | tee {log_path}; "
        f"if [ -f {ckpt_dir}/encoder.bin ]; then {{ {evl}; }} 2>&1 | tee -a {log_path}; "
        f"else echo '[sweep] no BEST checkpoint — training failed, skipping eval' | tee -a {log_path}; fi"
    )


def parse_log(log_path: str) -> Dict[str, Optional[float]]:
    if not os.path.exists(log_path):
        return {"global": None, "mean_batch": None}
    txt = open(log_path, errors="ignore").read()
    g = _PER_GLOBAL_RE.findall(txt)
    b = _PER_BATCH_RE.findall(txt)
    return {
        "global": float(g[-1]) if g else None,
        "mean_batch": float(b[-1]) if b else None,
    }


def print_summary(configs: List[Dict], log_dir: str, tag: str, subject: str) -> None:
    rows = []
    for cfg in configs:
        ss = savestring_for(tag, subject, cfg["idx"])
        res = parse_log(os.path.join(log_dir, f"{ss}.log"))
        rows.append((cfg, res))
    rows.sort(key=lambda r: (r[1]["global"] is None, r[1]["global"] if r[1]["global"] is not None else 1e9))

    print("\n" + "=" * 76)
    print(f"SWEEP RESULTS  (tag={tag}, subject={subject})  — ranked by validation global PER")
    print("=" * 76)
    has_mask = any("mask_ratio" in cfg for cfg, _ in rows)
    hdr = f"{'rank':>4} {'idx':>3} {'batch':>5} {'lr':>11} {'wd':>11} "
    hdr += f"{'mask':>6} " if has_mask else ""
    hdr += f"{'global PER':>11} {'mean-batch':>11}"
    print(hdr)
    for rank, (cfg, res) in enumerate(rows):
        g = f"{100*res['global']:.2f}%" if res["global"] is not None else "  —  "
        b = f"{100*res['mean_batch']:.2f}%" if res["mean_batch"] is not None else "  —  "
        row = f"{rank:>4} {cfg['idx']:>3} {cfg['batch_size']:>5} {cfg['lr']:>11.3e} {cfg['wd']:>11.3e} "
        row += f"{cfg['mask_ratio']:>6.2f} " if has_mask else ""
        row += f"{g:>11} {b:>11}"
        print(row)
    best = rows[0]
    if best[1]["global"] is not None:
        print("-" * 76)
        mask_str = (f"  mask={best[0]['mask_ratio']:.2f}" if "mask_ratio" in best[0] else "")
        print(f"BEST: idx={best[0]['idx']}  batch={best[0]['batch_size']}  "
              f"lr={best[0]['lr']:.3e}  wd={best[0]['wd']:.3e}{mask_str}  "
              f"→ val PER {100*best[1]['global']:.2f}%  "
              f"(checkpoints/{savestring_for(tag, subject, best[0]['idx'])}/BEST)")
    print("=" * 76 + "\n")


def run_scheduler(configs: List[Dict], args, log_dir: str) -> None:
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    pending = list(configs)

    # --only-idx: run just a subset of indices. Because sampling is seed-fixed,
    # a second sweep instance can pick up the *tail* indices on spare GPUs while
    # the main instance keeps chewing the head — same tag/savestrings, disjoint
    # index sets → no collision (stop the main instance before it reaches them).
    if args.only_idx:
        only = {int(x) for x in str(args.only_idx).split(",") if x.strip()}
        pending = [c for c in pending if c["idx"] in only]

    running: Dict[str, Dict] = {}   # gpu -> {"proc", "cfg"}
    done = 0
    total = len(pending)

    print(f"[sweep] {total} configs to run on GPUs {gpus} "
          f"({len(gpus)} concurrent), {args.epochs} epochs each: "
          f"idx {[c['idx'] for c in pending]}")

    while pending or running:
        # Fill at most one free GPU per iteration, then wait `launch_stagger`
        # seconds before starting the next run — staggering CUDA context init
        # avoids the transient errors seen when 7 processes start at once.
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            cfg = pending.pop(0)
            ss = savestring_for(args.tag, args.subject, cfg["idx"])
            log_path = os.path.join(log_dir, f"{ss}.log")
            cmd = build_command(cfg, args, ss, log_path)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            print(f"[sweep] GPU {gpu} ◀ idx={cfg['idx']:02d} "
                  f"batch={cfg['batch_size']} lr={cfg['lr']:.2e} wd={cfg['wd']:.2e} "
                  f"→ {ss}  (workers={args.num_workers})")
            proc = subprocess.Popen(["bash", "-lc", cmd], cwd=_HERE, env=env)
            running[gpu] = {"proc": proc, "cfg": cfg, "ss": ss, "log": log_path}
            if pending and args.launch_stagger > 0:
                time.sleep(args.launch_stagger)
            break  # one launch per outer iteration → clean staggering

        # poll
        time.sleep(5)
        for gpu, job in list(running.items()):
            rc = job["proc"].poll()
            if rc is None:
                continue
            done += 1
            res = parse_log(job["log"])
            gper = f"{100*res['global']:.2f}%" if res["global"] is not None else "FAILED"
            print(f"[sweep] GPU {gpu} ✔ idx={job['cfg']['idx']:02d} rc={rc} "
                  f"val PER {gper}  ({done}/{total})")
            del running[gpu]

    print_summary(configs, log_dir, args.tag, args.subject)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default="card_t15",
                   choices=["card_t15", "willet_t12", "kunz_t12", "kunz_t15",
                            "kunz_t16", "kunz_t17", "wairagkar",
                            "jude_speech_anarthia", "willett_handwriting",
                            "fan_handwriting", "jude_typing_t17", "jude_typing_t18"])
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                   help="Pretrained encoder dir (encoder.bin + encoder_config.pth).")
    p.add_argument("--gpus", default="4", help="Comma-separated GPU ids, e.g. '4,5,6'.")
    p.add_argument("--n-runs", type=int, default=30, help="Number of sampled configs.")
    p.add_argument("--epochs", type=int, default=800, help="Epochs per run.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (reproducible).")
    p.add_argument("--lr-range", nargs=2, type=float, default=list(LR_RANGE),
                   metavar=("LOW", "HIGH"),
                   help="LogUniform lr search range. Default Table 12: 5e-5 1e-3. "
                        "Focused pass suggested: 5e-4 2e-3.")
    p.add_argument("--wd-range", nargs=2, type=float, default=list(WD_RANGE),
                   metavar=("LOW", "HIGH"),
                   help="LogUniform wd search range. Default Table 12: 5e-5 1e-1. "
                        "Focused pass suggested: 5e-3 1e-1.")
    p.add_argument("--batch-choices", type=str, default="8,16,32,64",
                   help="Comma-separated batch sizes to sample from, e.g. '16,32,64'.")
    p.add_argument("--only-idx", type=str, default=None,
                   help="Run only these config indices (comma-separated), e.g. "
                        "'20,21,...,29'. Use to run the sweep tail on spare GPUs "
                        "in a second instance (same --tag/--seed/ranges). Stop the "
                        "main instance before it reaches these indices.")
    p.add_argument("--tag", default="t15", help="Sweep tag (namespaces savestrings/logs).")
    p.add_argument("--reinit-readin", action="store_true",
                   help="Pass --reinit-readin to finetune.py: keep the pretrained "
                        "encoder body but re-initialize the per-subject read-in "
                        "(diagnostic for negative transfer).")
    p.add_argument("--mask-ratio", type=float, default=None,
                   help="Pass a single fixed --mask-ratio to finetune.py for every "
                        "run. Ignored when --mask-choices is set.")
    p.add_argument("--mask-choices", type=str, default=None,
                   help="Comma-separated list of mask ratios to cycle through across "
                        "configs, e.g. '0.0,0.25,0.5'. Config i gets "
                        "choices[i %% len(choices)]. Takes precedence over --mask-ratio.")
    p.add_argument("--wandb-project", default="bit_reproduction_sweep")
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers per run. Keep low (0-2) when running "
                        "many concurrent runs: each worker pins host memory, and "
                        "too many concurrent workers exhaust system pinned memory "
                        "→ async CUDA 'invalid argument' crashes even on free GPUs.")
    p.add_argument("--launch-stagger", type=float, default=8.0,
                   help="Seconds to wait between starting consecutive runs, so "
                        "many CUDA contexts don't initialise simultaneously "
                        "(a common trigger of transient CUDA errors).")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip the automatic val-PER eval after each run.")
    p.add_argument("--extra-kwargs", dest="extra_kwargs", nargs="*", default=None,
                   help="Extra kwargs appended to the --kwargs list of every finetune.py "
                        "call, e.g. 'training.resume_from=/path/to/LAST'. "
                        "Used by SLURM wrappers to inject resume paths.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sampled configs and commands, launch nothing.")
    p.add_argument("--emit-idx", type=int, default=None,
                   help="Print the shell command for a single config index and "
                        "exit. Used by the SLURM job-array wrapper: each array "
                        "task emits+runs its own config, so no local scheduler.")
    p.add_argument("--summary-only", action="store_true",
                   help="Re-print the ranked results table from existing logs.")
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None,
                   help="Root dir where checkpoint subdirs are saved. Passed as "
                        "dirs.checkpoint_dir to finetune.py. "
                        "Default: <here>/checkpoints.")
    return p.parse_args()


def main():
    args = parse_args()
    # "none" = train from scratch (BIT-TFS): random init, no pretrained encoder.
    if args.pretrained != "none" and not os.path.isabs(args.pretrained):
        args.pretrained = os.path.join(_HERE, args.pretrained)

    log_dir = os.path.join(_HERE, "logs", f"sweep_{args.tag}")
    os.makedirs(log_dir, exist_ok=True)

    # For --summary-only, load configs from the saved manifest (preserves mask_ratio).
    manifest = os.path.join(log_dir, "manifest.json")
    if args.summary_only and os.path.exists(manifest):
        configs = json.load(open(manifest))["configs"]
    else:
        batch_choices = [int(x) for x in str(args.batch_choices).split(",") if x.strip()]
        mask_choices = ([float(x) for x in str(args.mask_choices).split(",") if x.strip()]
                        if args.mask_choices else None)
        configs = sample_configs(args.n_runs, args.seed,
                                 lr_range=tuple(args.lr_range),
                                 wd_range=tuple(args.wd_range),
                                 batch_choices=batch_choices,
                                 mask_choices=mask_choices)
        json.dump({"args": vars(args), "configs": configs}, open(manifest, "w"), indent=2)

    if args.emit_idx is not None:
        by_idx = {c["idx"]: c for c in configs}
        if args.emit_idx not in by_idx:
            sys.exit(f"idx {args.emit_idx} not in sampled configs [0..{args.n_runs-1}]")
        cfg = by_idx[args.emit_idx]
        ss = savestring_for(args.tag, args.subject, cfg["idx"])
        log_path = os.path.join(log_dir, f"{ss}.log")
        # Emit only the command; the SLURM array task runs it. Nothing else on stdout.
        print(build_command(cfg, args, ss, log_path))
        return

    if args.summary_only:
        print_summary(configs, log_dir, args.tag, args.subject)
        return

    if args.dry_run:
        print(f"Sampled {len(configs)} configs (seed={args.seed}):")
        for cfg in configs:
            ss = savestring_for(args.tag, args.subject, cfg["idx"])
            print(f"  idx={cfg['idx']:02d} batch={cfg['batch_size']:>2} "
                  f"lr={cfg['lr']:.3e} wd={cfg['wd']:.3e} → {ss}")
        print(f"\nExample command (idx=0):\n")
        print(build_command(configs[0], args,
                            savestring_for(args.tag, args.subject, 0),
                            os.path.join(log_dir, "example.log")))
        return

    if args.pretrained != "none" and not os.path.exists(os.path.join(args.pretrained, "encoder.bin")):
        sys.exit(f"No encoder.bin under {args.pretrained!r}")

    run_scheduler(configs, args, log_dir)


if __name__ == "__main__":
    main()
