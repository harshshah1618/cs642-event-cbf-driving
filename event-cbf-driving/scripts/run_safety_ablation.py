"""Paired calibration/braking/intent ablations with fixed skills and seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from event_cbf_driving.config import load_config
from event_cbf_driving.eval.runner import run_evaluation
from event_cbf_driving.io_utils import atomic_write_json


def main():
    stages = (
        "legacy",
        "calibration_only",
        "braking_only",
        "calibration_braking",
        "calibration_braking_intent",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=stages, default=list(stages))
    args = parser.parse_args()
    records = []
    for stage in args.stages:
        cfg = load_config("configs/metadrive.yaml")
        if "calibration" in stage:
            cfg.checkpoint_dir = "checkpoints/metadrive-calibrated"
            cfg.stack.sigma_threshold_scale = 1.0
        cfg.shield.braking_guard = "braking" in stage
        cfg.shield.intent_corridor = "intent" in stage
        cfg.stack.rearm_on_skill_change = "intent" in stage
        cfg.eval.seeds = [0, 1, 2]
        cfg.eval.modes = ["proposed"]
        cfg.eval.figures = False
        cfg.eval.fail_on_regression = False
        result = run_evaluation(cfg, out_dir="runs/safety-ablation-final", run_id=stage)
        records.extend({"stage": stage, **e} for e in result.episodes)
        print(
            stage,
            [
                (r["scenario"], r["route_pct"], r["crashed"], r["infeasible_ticks"])
                for r in result.episodes
            ],
            flush=True,
        )
    atomic_write_json(Path("runs/safety-ablation-final/summary.json"), {"episodes": records})


if __name__ == "__main__":
    main()
