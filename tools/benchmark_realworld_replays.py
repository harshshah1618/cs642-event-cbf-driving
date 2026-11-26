"""Benchmark #2: Real-world Waymo & nuScenes traffic replays (MetaDrive ScenarioEnv).

Strictly observation-only: the controller receives ONLY the standard ScenarioEnv
observation vector (12 side-detector rays + 7 ego floats + 22 waypoint floats +
120 LiDAR rays = 161 floats). Zero access to simulator object lists or future
ground-truth agent trajectories.

Compares on all 13 built-in real-world scenes (3 Waymo + 10 nuScenes):
1. Ours (Pure-Obs Waypoint Tracker + LiDAR Safety Filter + Speed/Reserve Cap)
2. Ablation: Unshielded Pure-Obs Waypoint Tracker (no LiDAR safety filter)
3. Naive Constant-Throttle Baseline (no steering/safety)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


_NUM_LIDAR = 120
_RAY_ANGLES = np.arange(_NUM_LIDAR, dtype=np.float64) * (2.0 * math.pi / _NUM_LIDAR)
_RAY_COS = np.cos(_RAY_ANGLES)
_RAY_SIN = np.sin(_RAY_ANGLES)


class ScenarioObsDriver:
    """Strictly observation-only (161-dim) driver for MetaDrive ScenarioEnv."""

    def __init__(self, *, protection: bool = True, max_speed_mps: float = 11.0):
        self.protection = bool(protection)
        self.max_speed_mps = float(max_speed_mps)
        self.prev_steer = 0.0
        self.interventions = 0

    def act(self, obs: np.ndarray) -> list[float]:
        arr = np.asarray(obs, dtype=np.float64)
        # Layout for default ScenarioEnv (161-dim):
        # [0:12]   side_detector (12 rays, scaled by 50 m)
        # [12]     heading_diff (0.5 = aligned)
        # [13]     speed_norm = (speed_kmh + 1) / 81.0
        # [14:19]  steering, last_act[0], last_act[1], yaw_rate, lateral_in_lane
        # [19:39]  10 future reference waypoints
        # [37]     lateral error to reference trajectory: (lat / max_lateral_dist + 1)/2 (<0.5 = left -> steer right -)
        # [38]     heading error to reference trajectory: (wrap(ref_h - ego_h) / pi + 1)/2 (<0.5 = pointing left -> steer right -)
        # [41:161] 120 LiDAR rays scaled by 50 m
        speed_mps = max(0.0, (float(arr[13]) * 81.0 - 1.0) / 3.6)
        lat_err = (float(arr[37]) * 2.0 - 1.0) * 10.0
        head_err = (float(arr[38]) * 2.0 - 1.0) * math.pi

        # Slow down naturally on sharper turns
        desired_speed = min(self.max_speed_mps, max(4.0, 10.5 - 8.0 * abs(head_err)))

        # LiDAR safety filter (strictly from obs[41:161])
        if self.protection:
            lidar_m = arr[41:161] * 50.0
            mask = lidar_m < 49.4
            if np.any(mask):
                r = lidar_m[mask]
                x = r * _RAY_COS[mask]
                y_left = r * _RAY_SIN[mask]
                look_s = max(8.0, speed_mps * 2.4 + 4.5)
                half_w = 1.25 + 0.04 * np.clip(x, 0.0, look_s)
                in_corridor = (x > 0.3) & (x < look_s) & (np.abs(y_left) < half_w)
                if np.any(in_corridor):
                    front_d = max(0.0, float(np.min(x[in_corridor])) - 2.3)
                    avail = max(0.0, front_d - 3.6)
                    brake_a = 4.2
                    headway = 1.2
                    cap = -brake_a * headway + math.sqrt((brake_a * headway) ** 2 + 2.0 * brake_a * avail)
                    if cap < desired_speed - 0.15:
                        self.interventions += 1
                    desired_speed = min(desired_speed, max(0.0, cap))
                    if front_d < max(3.2, speed_mps * 0.65):
                        desired_speed = 0.0

        # Stanley-style pure-obs trajectory tracking using obs[39] & obs[40]
        # head_err = ref_heading - ego_heading (>0 means steer left +)
        # lat_err > 0 means right of reference trajectory -> steer left (+)
        raw_steer = float(np.clip(1.6 * head_err + 0.30 * np.clip(lat_err, -4.0, 4.0), -1.0, 1.0))
        steer = float(np.clip(raw_steer, self.prev_steer - 0.35, self.prev_steer + 0.35))
        self.prev_steer = steer

        speed_err = desired_speed - speed_mps
        throttle = float(np.clip(0.5 * speed_err if speed_err >= 0 else 0.85 * speed_err, -1.0, 0.85))
        return [steer, throttle]


def run_dataset(name: str, num_scenarios: int, mode: str) -> dict:
    from metadrive.engine.asset_loader import AssetLoader
    from metadrive.envs.scenario_env import ScenarioEnv

    env = ScenarioEnv(dict(
        data_directory=AssetLoader.file_path(name, unix_style=False),
        num_scenarios=num_scenarios,
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_human_done=False,
        out_of_route_done=True,
        log_level=50,
    ))
    rows = []
    t0 = time.monotonic()
    latencies_ms = []
    try:
        for seed in range(num_scenarios):
            obs, _ = env.reset(seed=seed)
            drv = ScenarioObsDriver(protection=(mode == "ours"))
            crashes = 0
            out_of_road = False
            steps = 0
            info = {}
            while True:
                ts0 = time.perf_counter()
                if mode == "naive":
                    act = [0.0, 0.3]
                else:
                    act = drv.act(obs)
                latencies_ms.append((time.perf_counter() - ts0) * 1000.0)
                obs, _, term, trunc, info = env.step(act)
                steps += 1
                if info.get("crash_vehicle") or info.get("crash_object") or info.get("crash_human"):
                    crashes += 1
                if info.get("out_of_road"):
                    out_of_road = True
                if term or trunc or steps >= 400:
                    break
            rc = float(info.get("route_completion", 0.0))
            succ = bool(info.get("arrive_dest", False)) or (rc >= 0.90 and crashes == 0 and not out_of_road)
            rows.append(dict(
                dataset=name,
                seed=seed,
                success=succ,
                collision_free=(crashes == 0 and not out_of_road),
                crash_steps=crashes,
                out_of_road=out_of_road,
                route_completion=rc,
                steps=steps,
                interventions=drv.interventions,
            ))
    finally:
        env.close()

    return dict(
        dataset=name,
        mode=mode,
        episodes=len(rows),
        success_rate=float(np.mean([r["success"] for r in rows])),
        collision_free_rate=float(np.mean([r["collision_free"] for r in rows])),
        mean_route_completion=float(np.mean([r["route_completion"] for r in rows])),
        median_ms_per_step=float(np.median(latencies_ms)),
        wall_seconds=round(time.monotonic() - t0, 2),
        rows=rows,
    )


def main() -> None:
    results = []
    for mode in ("ours", "unshielded", "naive"):
        all_rows = []
        lat_list = []
        for ds, n in (("waymo", 3), ("nuscenes", 10)):
            res = run_dataset(ds, n, mode)
            results.append(res)
            all_rows.extend(res["rows"])
            lat_list.append(res["median_ms_per_step"])
            print(f"{ds:8s} | {mode:10s} | succ={res['success_rate']*100:5.1f}% | "
                  f"crash_free={res['collision_free_rate']*100:5.1f}% | "
                  f"rc={res['mean_route_completion']*100:5.1f}% | {res['median_ms_per_step']:.3f} ms/step", flush=True)
        print(f"COMBINED | {mode:10s} (n={len(all_rows)}) | "
              f"succ={np.mean([r['success'] for r in all_rows])*100:5.1f}% | "
              f"crash_free={np.mean([r['collision_free'] for r in all_rows])*100:5.1f}% | "
              f"rc={np.mean([r['route_completion'] for r in all_rows])*100:5.1f}%", flush=True)
        print("-" * 72, flush=True)

    out = Path("runs/benchmarks/realworld-waymo-nuscenes-2026-11-25.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
