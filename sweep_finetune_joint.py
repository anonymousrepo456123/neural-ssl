"""Random hyperparameter sweep for JOINT phoneme finetuning on two datasets
of the *same* subject (e.g. card_t15 + kunz_t15, or willet_t12 + kunz_t12).

"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Reuse the EXACT sampling grid so joint idx K == single-sweep idx K.
from sweep_finetune_t15 import sample_configs, DEFAULT_PRETRAINED

_HERE = os.path.dirname(os.path.abspath(__file__))

# Same-subject pairs → (separate-mode feature bucket, shared-mode host subject).
JOINT_PRESETS = {
    frozenset({"card_t15", "kunz_t15"}):   ("t15_card_and_kunz", "card_t15"),
    frozenset({"willet_t12", "kunz_t12"}): ("t12_willett_and_kunz", "willet_t12"),
}


def savestring_for(tag: str, idx: int) -> str:
    return f"sweepjoint_{tag}_{idx:02d}"


def resolve_mode(sources, args):
    """Return (feature_bucket, shared_host) derived from sources unless overridden."""
    preset = JOINT_PRESETS.get(frozenset(sources))
    bucket = args.features or (preset[0] if preset else None)
    host = args.shared_as or (preset[1] if preset else sources[0])
    if args.mode == "separate" and bucket is None:
        sys.exit(f"No feature bucket known for sources {sources}; pass --features.")
    return bucket, host


def build_command(cfg, args, savestring, log_path, bucket, host, ckpt_base=None) -> str:
    py = sys.executable
    if args.mode == "separate":
        mode_flag = f"--features {bucket}"
    else:  # shared
        mode_flag = f"--shared_as {host}"
    # Per-config mask takes precedence over global --mask-ratio.
    _mr = cfg.get("mask_ratio")
    if _mr is None:
        _mr = getattr(args, "mask_ratio", None)
    mask_kw = f"mask_ratio={_mr} " if _mr is not None else ""
    cap = getattr(args, "per_subject_cap", None)
    cap_kw = f"training.train_per_subject_cap={cap} " if cap is not None else ""
    _extra_kw = (" ".join(getattr(args, "extra_kwargs", None) or []) + " ")
    _ckpt_base = ckpt_base or os.path.join(_HERE, "checkpoints")
    train = (
        f"{py} -u finetune_joint.py "
        f"--sources {args.sources} "
        f"--pretrained_ckpt {args.pretrained} "
        f"{mode_flag} "
        f"--savestring_suffix {savestring} "
        f"--kwargs "
        f"optimizer.lr={cfg['lr']:.6e} "
        f"optimizer.wd={cfg['wd']:.6e} "
        f"training.train_batch_size={cfg['batch_size']} "
        f"training.test_batch_size={cfg['batch_size']} "
        f"training.num_epochs={args.epochs} "
        f"training.num_workers={args.num_workers} "
        f"training.save_every=100000 "
        f"{mask_kw}"
        f"{cap_kw}"
        f"{_extra_kw}"
        f"dirs.checkpoint_dir={_ckpt_base} "
        f"wandb_project={args.wandb_project}"
    )
    # finetune_joint.py writes per_source_per.json itself → no separate eval step.
    return f"set -o pipefail; {{ {train}; }} 2>&1 | tee {log_path}"


def read_per_source(savestring: str, ckpt_base: str = None):
    """Return {source: per} from <ckpt_base>/<savestring>/per_source_per.json."""
    base = ckpt_base or os.path.join(_HERE, "checkpoints")
    path = os.path.join(base, savestring, "per_source_per.json")
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return {src: info.get("per") for src, info in d.items()}


def print_summary(configs, args, ckpt_base=None) -> None:
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows = []
    for cfg in configs:
        ss = savestring_for(args.tag, cfg["idx"])
        per = read_per_source(ss, ckpt_base)  # {src: per} or None
        mean = None
        if per and all(per.get(s) is not None for s in sources):
            mean = sum(per[s] for s in sources) / len(sources)
        rows.append((cfg, per, mean))
    rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 1e9))

    print("\n" + "=" * 84)
    print(f"JOINT SWEEP RESULTS  (tag={args.tag}, sources={args.sources}, mode={args.mode})")
    print("ranked by MEAN per-source validation PER")
    print("=" * 84)
    hdr = f"{'rank':>4} {'idx':>3} {'batch':>5} {'lr':>11} {'wd':>11} {'mean PER':>9}"
    for s in sources:
        hdr += f" {s[:14]:>14}"
    print(hdr)
    for rank, (cfg, per, mean) in enumerate(rows):
        m = f"{100*mean:.2f}%" if mean is not None else "  —  "
        line = (f"{rank:>4} {cfg['idx']:>3} {cfg['batch_size']:>5} "
                f"{cfg['lr']:>11.3e} {cfg['wd']:>11.3e} {m:>9}")
        for s in sources:
            v = per.get(s) if per else None
            line += f" {(f'{100*v:.2f}%' if v is not None else '—'):>14}"
        print(line)
    best = rows[0]
    if best[2] is not None:
        print("-" * 84)
        print(f"BEST: idx={best[0]['idx']} batch={best[0]['batch_size']} "
              f"lr={best[0]['lr']:.3e} wd={best[0]['wd']:.3e} "
              f"→ mean val PER {100*best[2]:.2f}%  "
              f"(checkpoints/{savestring_for(args.tag, best[0]['idx'])})")
    print("=" * 84 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default="card_t15,kunz_t15",
                   help="Comma-separated same-subject datasets, e.g. 'card_t15,kunz_t15' "
                        "or 'willet_t12,kunz_t12'.")
    p.add_argument("--mode", default="separate", choices=["separate", "shared"],
                   help="separate = per-source ReadIn (--features); "
                        "shared = one shared ReadIn (--shared_as).")
    p.add_argument("--features", default=None,
                   help="Override the separate-mode DATASET_CHANNELS bucket.")
    p.add_argument("--shared-as", dest="shared_as", default=None,
                   help="Override the shared-mode host subject.")
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                   help="Pretrained encoder dir (encoder.bin + encoder_config.pth).")
    p.add_argument("--n-runs", type=int, default=30)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--seed", type=int, default=0,
                   help="Same default as the single sweep → identical HP grid.")
    p.add_argument("--tag", default="t15joint_sep",
                   help="Namespaces savestrings/logs. Use a distinct tag per "
                        "sources+mode combo (e.g. t15joint_sep, t12joint_shared).")
    p.add_argument("--wandb-project", default="bit_reproduction_sweep")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--mask-ratio", dest="mask_ratio", type=float, default=None,
                   help="Single fixed mask ratio for all runs. Ignored if "
                        "--mask-choices is set.")
    p.add_argument("--mask-choices", type=str, default=None,
                   help="Comma-separated mask ratios to cycle across configs, "
                        "e.g. '0.0,0.25,0.5'. Config i → choices[i %% len(choices)].")
    p.add_argument("--emit-idx", type=int, default=None,
                   help="Print the shell command for one config index and exit "
                        "(used by the SLURM job-array wrapper).")
    p.add_argument("--per-subject-cap", dest="per_subject_cap", type=int, default=None,
                   help="Max trials per subject per epoch (GroupBatchSampler cap). "
                        "Useful for balancing very unequal dataset sizes.")
    p.add_argument("--lr-range", nargs=2, type=float, default=[5e-5, 1e-3],
                   metavar=("LR_MIN", "LR_MAX"),
                   help="LogUniform LR search range. Default: 5e-5 1e-3.")
    p.add_argument("--wd-range", nargs=2, type=float, default=[5e-5, 1e-1],
                   metavar=("WD_MIN", "WD_MAX"),
                   help="LogUniform WD search range. Default: 5e-5 1e-1.")
    p.add_argument("--extra-kwargs", dest="extra_kwargs", nargs="*", default=None,
                   help="Extra kwargs appended to the --kwargs list of every finetune_joint.py "
                        "call, e.g. 'training.resume_from=/path/to/LAST training.save_every=100'.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None,
                   help="Root dir where checkpoint subdirs are saved. Passed as "
                        "dirs.checkpoint_dir to finetune_joint.py. "
                        "Default: <here>/checkpoints.")
    return p.parse_args()


def main():
    args = parse_args()
    # "none" = train from scratch (random init, no pretrained encoder).
    if args.pretrained != "none" and not os.path.isabs(args.pretrained):
        args.pretrained = os.path.join(_HERE, args.pretrained)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    bucket, host = resolve_mode(sources, args)

    log_dir = os.path.join(_HERE, "logs", f"sweepjoint_{args.tag}")
    os.makedirs(log_dir, exist_ok=True)

    mask_choices = ([float(x) for x in str(args.mask_choices).split(",") if x.strip()]
                    if args.mask_choices else None)
    configs = sample_configs(args.n_runs, args.seed, mask_choices=mask_choices,
                             lr_range=tuple(args.lr_range),
                             wd_range=tuple(args.wd_range))

    manifest = os.path.join(log_dir, "manifest.json")
    json.dump({"args": vars(args), "sources": sources,
               "bucket": bucket, "host": host, "configs": configs},
              open(manifest, "w"), indent=2)

    ckpt_base = args.checkpoint_dir or os.path.join(_HERE, "checkpoints")

    if args.summary_only:
        print_summary(configs, args, ckpt_base)
        return

    if args.emit_idx is not None:
        by_idx = {c["idx"]: c for c in configs}
        if args.emit_idx not in by_idx:
            sys.exit(f"idx {args.emit_idx} not in [0..{args.n_runs-1}]")
        cfg = by_idx[args.emit_idx]
        ss = savestring_for(args.tag, cfg["idx"])
        log_path = os.path.join(log_dir, f"{ss}.log")
        print(build_command(cfg, args, ss, log_path, bucket, host, ckpt_base))
        return

    if args.dry_run:
        print(f"Joint sweep: sources={sources} mode={args.mode} "
              f"bucket={bucket} host={host} ckpt_base={ckpt_base}")
        for cfg in configs:
            ss = savestring_for(args.tag, cfg["idx"])
            print(f"  idx={cfg['idx']:02d} batch={cfg['batch_size']:>2} "
                  f"lr={cfg['lr']:.3e} wd={cfg['wd']:.3e} → {ss}")
        print("\nExample command (idx=0):\n")
        print(build_command(configs[0], args, savestring_for(args.tag, 0),
                            os.path.join(log_dir, "example.log"), bucket, host, ckpt_base))
        return

    sys.exit("This driver is SLURM-array based: use --emit-idx (via the sbatch), "
             "--dry-run, or --summary-only. No local scheduler.")


if __name__ == "__main__":
    main()
