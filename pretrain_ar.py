"""S3 -- Autoregressive ("GPT-brain") pretraining on the frozen study pools.


Usage
-----
    python pretrain_ar.py --regime h     # human-only pool
    python pretrain_ar.py --regime hm    # human + monkey pool

Writes ``checkpoints/bit_ar_{h,hm}/{BEST,LAST}/``.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.config import (
    ParseKwargs, update_config, default_trainer_config, config_from_kwargs,
)
from utils.eval_spike import eval_neurons_metric
from models.ndt import eval_rankme
from models.trainer import Trainer

from study_pretrain import build_study_dataset, apply_frozen_schedule, set_causal


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True, choices=["h", "hm"],
                   help="h = human-only pool, hm = human + monkey pool.")
    p.add_argument("--config", default="configs/pretrain/ndt/trainer_ar.yaml")
    p.add_argument("--savestring", default=None,
                   help="Defaults to bit_ar_<regime>.")
    p.add_argument("--smoke", action="store_true",
                   help="2-subject pool + 1 epoch: validates the code path only.")
    p.add_argument("--kwargs", nargs="*", action=ParseKwargs)
    return p.parse_args()


def main(args):
    os.chdir(_HERE)

    cfg_path = args.config if os.path.isabs(args.config) \
        else os.path.join(_HERE, args.config)
    config = update_config(default_trainer_config(), cfg_path)
    if args.kwargs is not None:
        config = update_config(config, config_from_kwargs(args.kwargs))

    # Frozen schedule + causal encoder (both asserted inside).
    apply_frozen_schedule(config)
    set_causal(config)

    config["savestring"] = args.savestring or f"bit_ar_{args.regime}"
    assert config.method.model_kwargs.method_name == "ar", \
        "trainer_ar.yaml must set method.model_kwargs.method_name: ar"
    print(f"[ar] savestring: {config.savestring}", flush=True)

    if args.smoke:
        from study_pretrain import SMOKE_POOL
        config["training"]["num_epochs"] = 1
        config["savestring"] = f"smoke_{config.savestring}"
    dataset, extra_model_kwargs = build_study_dataset(
        args.regime, config,
        subjects_override=(SMOKE_POOL if args.smoke else None))

    metric_name = "r2" if config.method.model_kwargs.loss == "mse" else "bps"
    metric_fns = {metric_name: eval_neurons_metric, "rankme": eval_rankme}

    trainer = Trainer(config, dataset=dataset, metric_fns=metric_fns,
                      extra_model_kwargs=extra_model_kwargs)
    trainer.train()


if __name__ == "__main__":
    main(parse_arguments())
