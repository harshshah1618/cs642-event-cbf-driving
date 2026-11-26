"""Run matched 800-episode local CPU controls for the unified 3-layer irregular suite.

Evaluates UnifiedEscalatingDriver (zero-peeking 259-dim LiDAR Layer 1 + PPO seed 0 Layer 2)
across all 8 irregular scenes x 50 fresh maps (1000-1049) for:
  - Unified Normal (trigger="none", initial_careful=False)
  - Unified Always-Careful (trigger="none", initial_careful=True)
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


SCENES = (
    "flagger",
    "billboard",
    "pedestrian-shoulder",
    "pedestrian-lane",
    "lane-closed",
    "wet-closed",
    "rain-clean",
    "clean",
)


def _run_shard(task: dict) -> dict:
    from driving.irregular.escalation import (
        EscalationConfig,
        irregular_config,
        run_episode,
        scene_placements,
    )
    from driving.irregular.scene import make_env

    scene = task["scene"]
    careful = task["careful"]
    seeds = task["seeds"]
    ckpt = task["ckpt"]
    env = make_env(irregular_config("irregular-test"))
    rows = []
    t0 = time.monotonic()
    try:
        for seed in seeds:
            cfg = EscalationConfig(trigger="none")
            row = run_episode(
                env,
                seed,
                scene_placements(scene, seed),
                cfg,
                driver_kwargs=dict(skill_checkpoint=ckpt, initial_careful=careful),
                max_steps=300 if scene in ("clean", "rain-clean") else 600,
                driver_kind="unified",
            )
            row.update(scene=scene, model=("unified-careful" if careful else "unified-normal"))
            rows.append(row)
    finally:
        env.close()

    crashes = sum(int(r["crash_vehicle"] or r["crash_object"] or r["crash_human"]) for r in rows)
    stops = sum(int(r["passed_stop"]) for r in rows)
    label = "careful" if careful else "normal"
    print(
        f"DONE unified-{label:7s} | {scene:19s} ({seeds[0]}-{seeds[-1]}) | "
        f"crashes={crashes:2d}/{len(rows)} | passed_stop={stops:2d}/{len(rows)} | "
        f"rc={np.mean([r['route_completion'] for r in rows])*100:4.1f}% ({time.monotonic()-t0:.1f}s)",
        flush=True,
    )
    return dict(scene=scene, careful=careful, rows=rows)


def main() -> None:
    ckpt = "runs/safemetadrive/skills-1m-v1/skills-seed0/ppo.zip"
    tasks = [
        dict(scene=scene, careful=careful, seeds=list(range(start, start + 25)), ckpt=ckpt)
        for careful in (False, True)
        for scene in SCENES
        for start in (1000, 1025)
    ]
    all_rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for res in pool.map(_run_shard, tasks):
            all_rows.extend(res["rows"])

    out = Path("runs/benchmarks/unified-irregular-controls-2026-11-25.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, default=str))

    print("\n=== UNIFIED IRREGULAR CONTROLS SUMMARY (800 episodes, maps 1000-1049) ===")
    print(f"{'Scene':20s} | {'Normal crash':>12s} | {'Careful crash':>13s} | {'Normal STOP':>11s} | {'Careful STOP':>12s}")
    print("-" * 80)
    for scene in SCENES:
        norm = [r for r in all_rows if r["scene"] == scene and r["model"] == "unified-normal"]
        care = [r for r in all_rows if r["scene"] == scene and r["model"] == "unified-careful"]
        nc = sum(int(r["crash_vehicle"] or r["crash_object"] or r["crash_human"]) for r in norm)
        cc = sum(int(r["crash_vehicle"] or r["crash_object"] or r["crash_human"]) for r in care)
        ns = sum(int(r["passed_stop"]) for r in norm)
        cs = sum(int(r["passed_stop"]) for r in care)
        print(f"{scene:20s} | {nc:2d}/50        | {cc:2d}/50         | {ns:2d}/50       | {cs:2d}/50")
    print(f"\nSaved 800 rows to {out}")


if __name__ == "__main__":
    main()
