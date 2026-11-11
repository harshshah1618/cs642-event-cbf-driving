"""MetaDrive closed loop with CERT-style position noise on other cars.

The shield and the skill picker see noisy positions. Bullet still moves the
cars and decides crashes from the true state. This is not nuScenes, CARLA
Town05, or the CERT-Replan obstacle course.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.env.types import WorldSnapshot
from event_cbf_driving.eval import runner
from event_cbf_driving.eval.runner import run_episode

_REAL_MAKE = runner.make_env


def _perturb(snap: WorldSnapshot, rng: np.random.Generator) -> WorldSnapshot:
    progress = float(snap.ego.x) / max(float(snap.route_length), 1.0)
    others = []
    for other in snap.others:
        noisy = other.copy()
        if progress < 0.4:
            dx = float(rng.normal(0.0, 0.05))
            dy = float(rng.normal(0.0, 0.05))
        elif rng.random() < 0.15:
            dx = float(np.clip(rng.laplace(0.0, 0.25), -0.75, 0.75))
            dy = float(np.clip(rng.laplace(0.0, 0.25), -0.75, 0.75))
        else:
            dx = float(np.clip(rng.normal(0.0, 0.15), -0.75, 0.75))
            dy = float(np.clip(rng.normal(0.0, 0.15), -0.75, 0.75))
        noisy.x += dx
        noisy.y += dy
        others.append(noisy)
    out = snap.copy()
    out.others = others
    return out


def _noisy_make(name, config):
    env = _REAL_MAKE(name, config)
    rng = np.random.default_rng(int(config.seed) + 90_001)
    original = env.snapshot

    def snapshot():
        return _perturb(original(), rng)

    env.snapshot = snapshot
    return env


def main() -> None:
    runner.make_env = _noisy_make
    cfg = load_config("configs/metadrive-study.yaml")
    cfg.stack.record_interactions = False
    cfg.eval.figures = False
    cfg.eval.record_episodes = False
    skills, _info = runner.load_skill_bank(cfg.checkpoint_dir)
    router = runner.create_router(cfg.router)
    scenarios = ["two_trap", "merge", "dense_highway"]
    modes = ["unshielded_skill", "static_cbf", "proposed"]
    seeds = list(range(20))
    rows = []
    try:
        for scenario in scenarios:
            spec = ScenarioConfig(
                scenario, simulator="metadrive", randomize=True, duration=20.0, params={}
            )
            for mode in modes:
                for seed in seeds:
                    result = run_episode(
                        mode,
                        spec,
                        seed,
                        skills,
                        router,
                        stack_config=cfg.stack,
                        shield_config=cfg.shield,
                    )
                    row = {
                        "scenario": scenario,
                        "mode": mode,
                        "seed": seed,
                        "route_pct": round(float(result.route_pct), 2),
                        "crashed": bool(result.crashed),
                        "freeze": bool(result.freeze),
                        "calls": int(result.router_calls),
                        "success": bool(
                            (not result.crashed) and float(result.route_pct) >= 90.0
                        ),
                    }
                    rows.append(row)
                    print(row, flush=True)
    finally:
        runner.make_env = _REAL_MAKE
        close = getattr(router, "close", None)
        if callable(close):
            close()

    summary = {}
    for mode in modes:
        group = [row for row in rows if row["mode"] == mode]
        n = len(group)
        crashes = sum(row["crashed"] for row in group)
        successes = sum(row["success"] for row in group)
        summary[mode] = {
            "n": n,
            "crash_pct": round(100.0 * crashes / n, 1),
            "success_pct": round(100.0 * successes / n, 1),
            "mean_route": round(sum(row["route_pct"] for row in group) / n, 1),
            "mean_calls": round(sum(row["calls"] for row in group) / n, 2),
        }
    brake = summary["static_cbf"]["crash_pct"]
    proposed = summary["proposed"]["crash_pct"]
    if brake > 0:
        reduction = round(100.0 * (brake - proposed) / brake, 1)
    else:
        reduction = None
    payload = {
        "noise": "CERT-like: N(0,0.05) then 15% Laplace(0.25) else N(0,0.15), clip 0.75 m",
        "note": "MetaDrive straight road. Not the CERT course, nuScenes, or CARLA Town05.",
        "summary": summary,
        "collision_reduction_vs_brake_only_pct": reduction,
        "episodes": rows,
    }
    out = Path("runs/official-bar/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "episodes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
