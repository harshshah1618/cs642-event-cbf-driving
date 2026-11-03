#!/usr/bin/env python3
"""Train CRUISE_RIGHT / OVERTAKE_LEFT / RETURN_RIGHT PPO skills and save checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_cbf_driving.env.skill_lane import LANE_Y
from event_cbf_driving.eval.figures import plot_training_curves
from event_cbf_driving.skills.ppo import (
    PPOTrainer,
    SkillMeta,
    calibrate_threshold,
    evaluate_skill,
    save_skill,
)


SKILL_SPECS = [
    ("CRUISE_RIGHT", 0, LANE_Y[0], 0),
    ("OVERTAKE_LEFT", 1, LANE_Y[1], 1),
    ("RETURN_RIGHT", 0, LANE_Y[0], 2),
]
# Seeds known to converge on CPU within ~160 updates (plus fallbacks).
SEED_TRIES = {
    "CRUISE_RIGHT": [0, 3, 5, 9],
    "OVERTAKE_LEFT": [1, 7, 11, 21, 42],
    "RETURN_RIGHT": [2, 4, 8, 12],
}


def main(argv=None):
    p = argparse.ArgumentParser(description="Train event-CBF driving skills")
    p.add_argument("--updates", type=int, default=160)
    p.add_argument("--seed", type=int, default=None, help="Force a single seed (skip sweep)")
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "media" / "full-stack",
    )
    p.add_argument("--skills", nargs="*", default=["CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT"])
    args = p.parse_args(argv)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    summary = {}
    for name, lane, y_t, offset in SKILL_SPECS:
        if name not in args.skills:
            continue
        seeds = [args.seed] if args.seed is not None else SEED_TRIES[name]
        best = None
        for seed in seeds:
            print(f"\n=== Training {name} (y*={y_t}, seed={seed}) ===")
            trainer = PPOTrainer(name=name, y_target=y_t, target_lane=lane, seed=seed)
            net = trainer.train(n_updates=args.updates, min_updates=40)
            diag = evaluate_skill(net, y_t)
            thr, mean, std, spread = calibrate_threshold(net, y_t, seed=seed + 99)
            meta = SkillMeta(
                name=name,
                target_lane=lane,
                y_target=y_t,
                threshold=float(thr),
                sigma_mean=float(mean),
                sigma_std=float(std),
                spread=float(spread),
            )
            cand = (diag, net, meta, trainer.history, seed)
            if best is None or (diag["learned"] and not best[0]["learned"]):
                best = cand
            elif diag["learned"] and best[0]["learned"]:
                if diag["hold"]["mean_abs_y"] < best[0]["hold"]["mean_abs_y"]:
                    best = cand
            if diag["learned"]:
                break
        diag, net, meta, history, seed = best
        path = args.ckpt_dir / f"{name}.pt"
        save_skill(path, net, meta, history=history)
        histories[name] = history
        summary[name] = {
            "path": str(path),
            "seed": seed,
            "meta": meta.__dict__,
            "diag": diag,
            "final_return": history[-1] if history else None,
        }
        print(
            f"  saved {path}  seed={seed} thr={meta.threshold:.3f} spread={meta.spread:.3f} "
            f"learned={diag['learned']} hold_y={diag['hold']['mean_abs_y']:.3f}"
        )

    plot_training_curves(histories, args.fig_dir / "training-curves.png")
    with open(args.ckpt_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote figures to {args.fig_dir / 'training-curves.png'}")
    print(f"Checkpoints in {args.ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
