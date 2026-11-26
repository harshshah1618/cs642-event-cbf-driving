"""No-VLM control arms for the irregular scenes on any map range (local CPU).

Arms: `normal` (never careful) and `careful` (careful from the first step).
Both use the same UnifiedEscalatingDriver and irregular-episode-v2 logger as
the VLM arm, so rows are directly comparable.

  PYTHONPATH=. .venv/bin/python tools/irregular_controls.py --seeds 1000 1050 \
      --tracker analytic --out runs/dev/irregular-guard.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCENES = ("flagger", "billboard", "pedestrian-shoulder", "pedestrian-lane", "lane-closed",
          "wet-closed", "rain-clean", "clean")
PPO = "runs/safemetadrive/skills-1m-v1/skills-seed{}/ppo.zip"


def env_config(first: int, count: int) -> dict:
    from driving.safemetadrive.protocol import environment_config
    config = environment_config("validation")
    config.update(start_seed=first, num_scenarios=count)
    return config


def arm_setup(arm: str, scene: str, latency: float):
    """(EscalationConfig, stand-in VLM, initial_careful) for a no-GPU arm."""
    from driving.irregular.escalation import EscalationConfig, FixedVLM, OracleVLM
    if arm in ("normal", "careful"):
        return EscalationConfig(trigger="none"), None, arm == "careful"
    cfg = EscalationConfig(trigger="lidar", verified_release=True)
    if arm == "oracle":
        return cfg, OracleVLM(scene, latency=latency), False
    if arm in ("lidar-careful", "lidar-stop"):
        return cfg, FixedVLM(arm.split("-")[1], latency=latency), False
    raise ValueError(arm)


def run_shard(task: dict) -> list[dict]:
    from driving.irregular.escalation import run_episode, scene_placements
    from driving.irregular.scene import make_env

    env = make_env(env_config(task["first"], task["count"]))
    rows = []
    try:
        for seed in task["seeds"]:
            cfg, vlm, careful = arm_setup(task["arm"], task["scene"], task["latency"])
            row = run_episode(
                env, seed, scene_placements(task["scene"], seed), cfg, vlm=vlm,
                driver_kwargs=dict(skill_checkpoint=task["ckpt"], initial_careful=careful,
                                   **task["system"]),
                max_steps=300 if task["scene"] in ("clean", "rain-clean") else 600,
                driver_kind="unified",
            )
            row.update(scene=task["scene"], arm=task["arm"], tracker=task["tracker"],
                       model=f"unified-{task['arm']}")
            rows.append(row)
    finally:
        env.close()
    return rows


def crashed(row: dict) -> bool:
    return bool(row["crash_vehicle"] or row["crash_object"] or row["crash_human"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs=2, type=int, required=True, metavar=("FIRST", "END"))
    ap.add_argument("--tracker", default="analytic", help="analytic | ppo0 | ppo1 | ppo2")
    ap.add_argument("--arms", nargs="+", default=["normal", "careful"],
                    choices=["normal", "careful", "oracle", "lidar-careful", "lidar-stop"])
    ap.add_argument("--latency", type=float, default=0.2, help="stand-in VLM latency (s)")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--system", default="{}", help="JSON overrides for UnifiedDrivingSystem")
    ap.add_argument("--shard", type=int, default=10)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    first, end = args.seeds
    ckpt = None if args.tracker == "analytic" else PPO.format(args.tracker[-1])
    system = json.loads(args.system)
    tasks = [dict(scene=scene, arm=arm, tracker=args.tracker, ckpt=ckpt, system=system, latency=args.latency,
                  first=first, count=end - first, seeds=list(range(s, min(end, s + args.shard))))
             for arm in args.arms for scene in args.scenes for s in range(first, end, args.shard)]
    t0 = time.monotonic()
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as pool:
        for shard in pool.map(run_shard, tasks):
            rows.extend(shard)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(seeds=[first, end], tracker=args.tracker, system=system,
                                        rows=rows), default=str))
    print(f"{len(rows)} episodes in {time.monotonic() - t0:.0f}s -> {args.out}")
    for arm in args.arms:
        for scene in args.scenes:
            sel = [r for r in rows if r["arm"] == arm and r["scene"] == scene]
            print(f"{arm:8s} {scene:20s} n={len(sel):3d} crash={sum(map(crashed, sel)):3d} "
                  f"stop_pass={sum(r['passed_stop'] for r in sel):3d} "
                  f"route={100 * sum(r['route_completion'] for r in sel) / len(sel):4.0f}% "
                  f"stopped_s={sum(r['seconds_stopped_near_object'] for r in sel) / len(sel):4.1f}")


if __name__ == "__main__":
    main()
