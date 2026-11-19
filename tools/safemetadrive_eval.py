"""Evaluate without altering upstream rewards, costs, collisions or termination."""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from driving.safemetadrive.policies import make_driver
from driving.safemetadrive.protocol import SPLITS, environment_config, manifest


def evaluate(policy="lane_caution_rule", parameters=None, split="validation", seeds=None,
             output=None):
    from metadrive import SafeMetaDriveEnv

    parameters = parameters or {}
    start, count = SPLITS[split]
    seeds = list(range(start, start + count)) if seeds is None else list(seeds)
    if not seeds or any(not start <= s < start + count for s in seeds):
        raise ValueError(f"Seeds must be in {split} split")
    report = manifest(split, policy, parameters)
    report.update(python=platform.python_version(), episodes=[], seeds=seeds)
    env = SafeMetaDriveEnv(environment_config(split))
    started = time.monotonic()
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            driver = make_driver(policy, env.agent, parameters, observation=observation)
            reward, cost = 0.0, 0.0
            crashes = set()
            brakes = 0
            steps = 0
            info = {}
            while True:
                action = driver.act()
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nonfinite action")
                observation, r, terminated, truncated, info = env.step(action)
                if hasattr(driver, "on_observation"):
                    driver.on_observation(observation)
                steps += 1
                reward += float(r)
                cost += float(info["cost"])
                brakes += int(action[1] < -0.8)
                for kind in ("crash_vehicle", "crash_object", "out_of_road", "crash_human"):
                    if info.get(kind, False):
                        crashes.add(kind)
                if terminated or truncated:
                    break
            success = bool(info.get("arrive_dest", False))
            row = dict(seed=seed, success=success, collision_free_success=success and cost == 0,
                       cost=cost, reward=reward, steps=steps,
                       route_completion=float(info.get("route_completion", 0)),
                       violations=sorted(crashes), hard_brake_steps=brakes,
                       lane_changes=getattr(driver, "changes", None),
                       interventions=getattr(driver, "interventions", None),
                       terminated=bool(terminated), truncated=bool(truncated))
            report["episodes"].append(row)
            print(json.dumps(row), flush=True)
            if hasattr(driver, "close"):
                driver.close()
            elif hasattr(driver, "destroy"):
                driver.destroy()
            if output:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text(json.dumps(report, indent=2))
    finally:
        env.close()
    rows = report["episodes"]
    report["summary"] = dict(
        episodes=len(rows), success_rate=float(np.mean([x["success"] for x in rows])),
        collision_free_success_rate=float(np.mean([x["collision_free_success"] for x in rows])),
        mean_cost=float(np.mean([x["cost"] for x in rows])),
        mean_reward=float(np.mean([x["reward"] for x in rows])),
        mean_route_completion=float(np.mean([x["route_completion"] for x in rows])),
        wall_seconds=time.monotonic() - started,
    )
    if output:
        Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"]), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="lane_caution_rule")
    parser.add_argument("--parameters", default="{}")
    parser.add_argument("--split", choices=list(SPLITS), default="validation")
    parser.add_argument("--seeds", help="Comma-separated map seeds inside the selected split")
    parser.add_argument("--output", default="runs/safemetadrive/evaluation.json")
    args = parser.parse_args()
    evaluate(args.policy, json.loads(args.parameters), args.split,
             [int(x) for x in args.seeds.split(",")] if args.seeds else None, args.output)
