#!/usr/bin/env python3
"""Run M3-style baselines (FASIONAD / CERT-Replan) on two_trap, merge, dense_highway.

Additive to the four-mode ablation in ``eval_stack.py``. Appends measured tables
to RESULTS.md under ``## M3-style baselines`` and writes ``media/full-stack/m3-*.png``.
Does not rewrite the existing M1/M2 tables.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_cbf_driving.env.registry import make_env
from event_cbf_driving.eval.m3_figures import (
    plot_m3_cross_scenario,
    plot_m3_gauntlet_rollouts,
    plot_m3_scenario_bars,
)
from event_cbf_driving.eval.metrics import ablation_table, result_to_row
from event_cbf_driving.router.factory import make_router
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.skills.checkpoints import load_skill
from event_cbf_driving.stack.baselines import BaselineMode, M3BaselineStack
from event_cbf_driving.stack.controller import DrivingStack, StackConfig, StackMode

MEDIA = Path(__file__).resolve().parents[2] / "media" / "full-stack"
DOCS = Path(__file__).resolve().parents[1]

M3_MARKER = "## M3-style baselines"

WHAT_IMPLEMENTED = """
### What was implemented (honest)

**5 FASIONAD-style (`fasionad_wakeup`):** same fast skill always (`CRUISE_RIGHT`).
When critic σ exceeds the skill's training threshold, a router call is submitted
(same interrupt API + 300 ms latency). The answer is **not** a discrete skill
switch (no CRUISE↔OVERTAKE bank swap). It is applied as a temporary lateral
bias / waypoint nudge (`ay += ±2.5` toward the suggested lane, hold 1.2 s then
decay 0.8 s) on top of the continuing cruise action. Static CBF on, no inflation.
Router calls counted.

**6 CERT-Replan-style (`cert_replan`):** no VLM/router. Static CBF on. When
barrier residual stays high (override with residual ≥ 0.4 m/s² or barrier
value `h < 0`) for 15 ticks (0.30 s), switch plan class:
`CRUISE` → `LANE_CHANGE` (PPO `OVERTAKE_LEFT`, min dwell 1.2 s) →
`CREEP_SAFE` (open-loop hold nearest lane, target ~3 m/s). Hard switch, no
blend. Router calls = 0.
""".strip()


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


def run_ref_modes(skills, router, scenario: str, seed: int):
    """Re-measure the four existing modes (same API) for side-by-side M3 tables."""
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


def run_m3_modes(skills, router, scenario: str, seed: int):
    results = []
    for mode in (BaselineMode.FASIONAD, BaselineMode.CERT_REPLAN):
        env = make_env(scenario)
        shield = InflatedCBFShield(enabled=True, inflate=False)
        stack = M3BaselineStack(skills=skills, router=router, shield=shield, mode=mode)
        r = stack.run(env, seed=seed)
        results.append(r)
        print(
            f"  [{scenario}] {r.mode:18s} crash={r.crashed} route={r.route_pct:5.1f}% "
            f"freeze={r.freeze} calls={r.router_calls} jerk={r.peak_lateral_jerk:.1f}"
        )
    return results


def append_results_md(by_scenario: dict, router_desc: str, out_path: Path):
    """Append (or replace) the M3 section without touching earlier M1/M2 tables."""
    section_lines = [
        M3_MARKER,
        "",
        "All numbers below come from `scripts/eval_m3_baselines.py` on this machine. "
        "Nothing is fabricated. Modes 1–4 are re-measured in the same run for a fair "
        "side-by-side; modes 5–6 are the new M3-style baselines.",
        "",
        f"Router backend used: **{router_desc}**.",
        "",
        WHAT_IMPLEMENTED,
        "",
    ]
    for title, results in by_scenario.items():
        section_lines += [f"### {title}", "", ablation_table(results), ""]

    section_lines += [
        "### Raw M3 rows (all scenarios)",
        "",
        "```json",
        json.dumps(
            {k: [result_to_row(r) for r in v] for k, v in by_scenario.items()},
            indent=2,
        ),
        "```",
        "",
    ]
    section = "\n".join(section_lines)

    existing = out_path.read_text() if out_path.exists() else "# Results (measured)\n\n"
    if M3_MARKER in existing:
        # Replace from marker to EOF (or next top-level sibling we don't have).
        existing = re.split(rf"(?m)^{re.escape(M3_MARKER)}\s*$", existing, maxsplit=1)[0].rstrip() + "\n\n"
    else:
        existing = existing.rstrip() + "\n\n"
    out_path.write_text(existing + section)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--fig-dir", type=Path, default=MEDIA)
    p.add_argument("--router", default="scripted", choices=["auto", "scripted", "openai", "local"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-ref", action="store_true", help="Only run modes 5–6 (faster)")
    args = p.parse_args(argv)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    skills = load_skills(args.ckpt_dir)
    router = make_router(prefer=args.router, latency_s=0.30)
    print(f"Router: {router.describe()} ({getattr(router, 'name', type(router).__name__)})")

    by_scenario = {}
    scenario_specs = [
        ("two_trap", "Two-Trap Gauntlet", args.seed),
        ("merge", "Merge", args.seed + 1),
        ("dense_highway", "Dense highway", args.seed + 1),
    ]

    for scen_key, title, seed in scenario_specs:
        print(f"\n=== {title} ===")
        if args.skip_ref:
            refs = []
        else:
            refs = run_ref_modes(skills, router, scen_key, seed=seed)
        m3 = run_m3_modes(skills, router, scen_key, seed=seed)
        by_scenario[title] = refs + m3

    # Figures
    gauntlet = by_scenario["Two-Trap Gauntlet"]
    m3_only = [r for r in gauntlet if r.mode in ("fasionad_wakeup", "cert_replan")]
    # Prefer proposed + two M3 for rollout panel when refs present
    rollout_set = [r for r in gauntlet if r.mode in ("proposed", "fasionad_wakeup", "cert_replan", "static_cbf")]
    if not rollout_set:
        rollout_set = m3_only

    plot_m3_gauntlet_rollouts(rollout_set, args.fig_dir / "m3-gauntlet-rollouts.png")
    plot_m3_scenario_bars(gauntlet, args.fig_dir / "m3-gauntlet-ablation.png", "M3 gauntlet (measured)")
    plot_m3_scenario_bars(
        by_scenario["Merge"], args.fig_dir / "m3-merge-ablation.png", "M3 merge (measured)"
    )
    plot_m3_scenario_bars(
        by_scenario["Dense highway"],
        args.fig_dir / "m3-dense-highway-ablation.png",
        "M3 dense highway (measured)",
    )
    plot_m3_cross_scenario(
        by_scenario,
        args.fig_dir / "m3-cross-scenario-route.png",
        mode_keys=["proposed", "fasionad_wakeup", "cert_replan", "static_cbf", "fixed_rate_vlm"],
    )

    # Persist JSON (additive file)
    payload = {
        "router": router.describe(),
        "router_name": getattr(router, "name", ""),
        "scenarios": {k: [result_to_row(r) | {"calls": r.calls} for r in v] for k, v in by_scenario.items()},
    }
    out_json = DOCS / "results_m3.json"
    out_json.write_text(json.dumps(payload, indent=2))

    append_results_md(by_scenario, router.describe(), DOCS / "RESULTS.md")

    print("\n--- M3 gauntlet ---")
    print(ablation_table(gauntlet))
    print(f"\nFigures: {args.fig_dir}/m3-*.png")
    print(f"RESULTS appended: {DOCS / 'RESULTS.md'}")
    print(f"JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
