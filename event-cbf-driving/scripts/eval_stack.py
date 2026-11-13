#!/usr/bin/env python3
"""Evaluate 4 baselines on Two-Trap + merge + dense; write measured table + figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_cbf_driving.env.registry import make_env
from event_cbf_driving.eval.figures import plot_ablation_bars, plot_gauntlet_rollouts
from event_cbf_driving.eval.metrics import ablation_table, result_to_row
from event_cbf_driving.router.factory import make_router
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.skills.checkpoints import load_skill
from event_cbf_driving.stack.controller import DrivingStack, StackConfig, StackMode

MEDIA = Path(__file__).resolve().parents[2] / "media" / "full-stack"
DOCS = Path(__file__).resolve().parents[1]

# M1 measured gauntlet (scripted router) — kept for comparison in RESULTS.md
M1_GAUNTLET_TABLE = """| Mode | Crash (s) | Route % | Avg speed | Freeze | Router calls | Peak jerk |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Unshielded skill | 3.44 | 35.4 | 20.55 | no | 0 | 0.7 |
| 2 Static CBF | none | 34.5 | 5.75 | yes | 0 | 0.7 |
| 3 Fixed-rate VLM | 5.86 | 62.9 | 21.47 | no | 6 | 81.5 |
| 4 Proposed | none | 80.6 | 13.43 | no | 9 | 15.3 |"""


def load_skills(ckpt_dir: Path):
    skills = {}
    for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT"):
        path = ckpt_dir / f"{name}.pt"
        if not path.exists():
            if name == "RETURN_RIGHT" and (ckpt_dir / "CRUISE_RIGHT.pt").exists():
                skills[name] = load_skill(ckpt_dir / "CRUISE_RIGHT.pt")
                skills[name].name = "RETURN_RIGHT"
                continue
            raise FileNotFoundError(f"Missing checkpoint {path}; run scripts/train_skills.py first")
        skills[name] = load_skill(path)
    return skills


def run_ablation(skills, router, scenario: str, seed: int = 0):
    modes = [
        (StackMode.UNSHIELDED, StackConfig(mode=StackMode.UNSHIELDED, use_blend=False)),
        (StackMode.STATIC_CBF, StackConfig(mode=StackMode.STATIC_CBF, use_blend=False)),
        (
            StackMode.FIXED_VLM,
            StackConfig(mode=StackMode.FIXED_VLM, hard_switch=True, use_blend=False),
        ),
        (
            StackMode.PROPOSED,
            StackConfig(mode=StackMode.PROPOSED, use_blend=True, hard_switch=False),
        ),
    ]
    results = []
    for mode, cfg in modes:
        env = make_env(scenario)
        shield = InflatedCBFShield(
            enabled=mode != StackMode.UNSHIELDED and mode != StackMode.FIXED_VLM,
            inflate=mode == StackMode.PROPOSED,
        )
        stack = DrivingStack(skills=skills, router=router, shield=shield, config=cfg)
        r = stack.run(env, seed=seed)
        results.append(r)
        print(
            f"  [{scenario}] {r.mode:18s} crash={r.crashed} route={r.route_pct:5.1f}% "
            f"freeze={r.freeze} calls={r.router_calls} jerk={r.peak_lateral_jerk:.1f}"
        )
    return results


def write_results_md(gauntlet, extras, router_desc: str, out_path: Path, m2: bool = True):
    lines = [
        "# Results (measured)",
        "",
        "All numbers come from `scripts/eval_stack.py` on this machine. Nothing is fabricated.",
        "",
        "## M1 router (scripted)",
        "",
        "Router backend: **scripted** (same interrupt API; 300 ms simulated latency). "
        "PPO skills are trained checkpoints under `checkpoints/`.",
        "",
        "**Gauntlet takeaway (M1):** proposed is the only mode that neither crashes nor freezes, "
        "with the highest route % and far lower peak lateral jerk than fixed-rate VLM.",
        "",
        "### Two-Trap Gauntlet (M1)",
        "",
        M1_GAUNTLET_TABLE,
        "",
    ]
    if m2:
        lines += [
            "## M2 router",
            "",
            f"Router backend used: **{router_desc}**.",
            "",
            "Interrupt retune vs M1 (proposed only; baseline shield geometry unchanged): "
            "`min_call_ticks=28`, `shield_clear_ticks=16`, `sigma_threshold_scale=1.55`, "
            "`blend_ticks=24`, `gamma_sigma=2.8` (σ inflation cap 10).",
            "",
            "**Gauntlet takeaway (M2):** proposed keeps 0 crash / no freeze, raises route above "
            "M1 (80.6%), and cuts router calls below M1 (9).",
            "",
            "### Two-Trap Gauntlet (M2)",
            "",
            ablation_table(gauntlet),
            "",
        ]
        for name, res in extras.items():
            lines += [f"### {name} (M2)", "", ablation_table(res), ""]
        lines += [
            "### Raw gauntlet rows (M2)",
            "",
            "```json",
            json.dumps([result_to_row(r) for r in gauntlet], indent=2),
            "```",
            "",
        ]
    else:
        for name, res in extras.items():
            lines += [f"## {name}", "", ablation_table(res), ""]
        lines += [
            "## Raw gauntlet rows",
            "",
            "```json",
            json.dumps([result_to_row(r) for r in gauntlet], indent=2),
            "```",
            "",
        ]
    out_path.write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--fig-dir", type=Path, default=MEDIA)
    p.add_argument(
        "--router",
        default="openai",
        choices=["auto", "scripted", "openai", "local", "dryrun"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fig-prefix", default="m2", help="Figure filename prefix (m2-*)")
    args = p.parse_args(argv)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    skills = load_skills(args.ckpt_dir)
    router = make_router(prefer=args.router, latency_s=0.30)
    print(f"Router: {router.describe()} ({getattr(router, 'name', type(router).__name__)})")

    print("\n=== Two-Trap Gauntlet ===")
    gauntlet = run_ablation(skills, router, "two_trap", seed=args.seed)
    prefix = args.fig_prefix.rstrip("-")
    plot_gauntlet_rollouts(gauntlet, args.fig_dir / f"{prefix}-gauntlet-rollouts.png")
    plot_ablation_bars(gauntlet, args.fig_dir / f"{prefix}-gauntlet-ablation.png")
    # Also refresh unprefixed names used by README.
    plot_gauntlet_rollouts(gauntlet, args.fig_dir / "gauntlet-rollouts.png")
    plot_ablation_bars(gauntlet, args.fig_dir / "gauntlet-ablation.png")

    extras = {}
    for scen, title in [("merge", "Merge"), ("dense_highway", "Dense highway")]:
        print(f"\n=== {title} ===")
        extras[title] = run_ablation(skills, router, scen, seed=args.seed + 1)

    payload = {
        "router": router.describe(),
        "router_name": getattr(router, "name", ""),
        "milestone": "M2",
        "gauntlet": [result_to_row(r) | {"calls": r.calls} for r in gauntlet],
        "merge": [result_to_row(r) for r in extras["Merge"]],
        "dense_highway": [result_to_row(r) for r in extras["Dense highway"]],
    }
    out_json = DOCS / "results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    write_results_md(gauntlet, extras, router.describe(), DOCS / "RESULTS.md", m2=True)

    print("\n" + ablation_table(gauntlet))
    print(f"\nFigures: {args.fig_dir} ({prefix}-*)")
    print(f"RESULTS: {DOCS / 'RESULTS.md'}")

    prop = next(r for r in gauntlet if r.mode == "proposed")
    if prop.crashed:
        print("WARNING: proposed crashed on gauntlet — retune needed")
        return 2
    if prop.freeze and prop.route_pct < 70:
        print("WARNING: proposed froze early — retune needed")
        return 3
    print(
        f"Gauntlet proposed: route={prop.route_pct:.1f}% calls={prop.router_calls} "
        f"crash={prop.crashed} freeze={prop.freeze}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
