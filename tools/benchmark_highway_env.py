"""Benchmark #3: highway-env (completely separate simulator: highway, merge, roundabout).

Evaluates 5 clean baselines across 50 seeds x 3 standard highway-env environments
(150 episodes per method = 750 episodes total) at 10 Hz control rate using ONLY
the standard Kinematics observation array (no simulator peeking):

1. Unshielded Greedy Driver (fast lane-picker, no safety filter)
2. Static Safety Filter (stays in lane, brakes safely, never replans)
3. Always-Ask Slow Planner (10 Hz polling, 1.5 s delay, no safety filter)
4. Event-Triggered + Safety Filter (1.5 s delay, NO option reserve)
5. Ours: Event-Triggered + Safety Filter + Option Reserve (1.5 s delay)
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np


# Standard highway-env DiscreteMetaAction indices:
# 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


class HighwayEnvStack:
    """Strictly observation-only controller for highway-env Kinematics obs.

    Kinematics obs (with absolute=False, normalize=False):
        row 0: [presence, ego_x, ego_y, ego_vx, ego_vy] (or relative zeros)
        rows 1..N-1: [presence, dx, dy, dvx, dvy] relative to ego!
    """

    def __init__(
        self,
        *,
        mode: str,
        delay_steps: int = 15,
        option_reserve_m: float = 12.0,
        lane_width: float = 4.0,
    ):
        self.mode = mode
        self.delay_steps = int(delay_steps)
        self.option_reserve_m = float(option_reserve_m)
        self.lane_width = float(lane_width)
        self.tick = 0
        self.calls = 0
        self.queue: deque[tuple[int, int]] = deque()
        self.last_call_tick = -100
        self.pending = False

    def _lane_gaps(self, obs: np.ndarray) -> tuple[float, float, float, float, float, float]:
        """Return (front_same, front_left, rear_left, front_right, rear_right, front_rel_vx)."""
        front_same, front_rel_vx = 100.0, 0.0
        front_left, rear_left = 100.0, 100.0
        front_right, rear_right = 100.0, 100.0
        for row in obs[1:]:
            if row[0] < 0.5:
                continue
            dx, dy, dvx = float(row[1]), float(row[2]), float(row[3])
            # Same lane: |dy| < 2.0 m
            if abs(dy) < 2.0:
                if dx > 0.0 and dx < front_same:
                    front_same = dx
                    front_rel_vx = dvx
            # Left lane: dy in [-6.0, -2.0] (highway-env +y is downward/right)
            elif -6.0 <= dy <= -2.0:
                if dx >= 0.0:
                    front_left = min(front_left, dx)
                else:
                    rear_left = min(rear_left, -dx)
            # Right lane: dy in [2.0, 6.0]
            elif 2.0 <= dy <= 6.0:
                if dx >= 0.0:
                    front_right = min(front_right, dx)
                else:
                    rear_right = min(rear_right, -dx)
        return front_same, front_left, rear_left, front_right, rear_right, front_rel_vx

    def _slow_planner_choice(self, obs: np.ndarray, ego_y: float, max_y: float) -> int:
        fs, fl, rl, fr, rr, _ = self._lane_gaps(obs)
        if fs > 32.0:
            return FASTER
        can_left = (ego_y > 1.5) and (fl > 22.0) and (rl > 14.0)
        can_right = (ego_y < max_y - 1.5) and (fr > 22.0) and (rr > 14.0)
        if can_left and fl > fs + 8.0 and (not can_right or fl >= fr):
            return LANE_LEFT
        if can_right and fr > fs + 8.0:
            return LANE_RIGHT
        return SLOWER if fs < 18.0 else IDLE

    def act(self, obs: np.ndarray, *, max_y: float = 12.0) -> int:
        ego_y = float(obs[0, 2])
        ego_vx = float(obs[0, 3])
        fs, fl, rl, fr, rr, f_dvx = self._lane_gaps(obs)

        # 1. Unshielded greedy fast driver
        if self.mode == "unshielded":
            self.tick += 1
            if fs < 22.0:
                if ego_y > 1.5 and fl > fs:
                    return LANE_LEFT
                if ego_y < max_y - 1.5 and fr > fs:
                    return LANE_RIGHT
            return FASTER

        # 2. Static safety filter (never changes lanes)
        if self.mode == "static_filter":
            self.tick += 1
            safe_d = max(14.0, ego_vx * 1.1)
            if fs < safe_d or (fs < safe_d + 8.0 and f_dvx < -2.0):
                return SLOWER
            return FASTER if fs > safe_d + 10.0 else IDLE

        # 3. Always-ask slow planner (10 Hz polling, delayed, NO fast safety filter)
        if self.mode == "always_ask_unshielded":
            self.calls += 1
            choice = self._slow_planner_choice(obs, ego_y, max_y)
            self.queue.append((self.tick + self.delay_steps, choice))
            out = FASTER
            while self.queue and self.queue[0][0] <= self.tick:
                _, out = self.queue.popleft()
            self.tick += 1
            return out

        # 4 & 5. Event-triggered + Fast Safety Filter (with or without Option Reserve)
        # Deliver ready slow-planner response with fresh gap check
        admitted_cmd = None
        if self.queue and self.queue[0][0] <= self.tick:
            _, cand = self.queue.popleft()
            self.pending = False
            if cand == LANE_LEFT and (ego_y > 1.5) and (fl > 18.0) and (rl > 12.0):
                admitted_cmd = LANE_LEFT
            elif cand == LANE_RIGHT and (ego_y < max_y - 1.5) and (fr > 18.0) and (rr > 12.0):
                admitted_cmd = LANE_RIGHT

        # Event-triggered wake-up: only call slow planner when blocked ahead
        blocked = fs < max(26.0, ego_vx * 1.4)
        if blocked and not self.pending and (self.tick - self.last_call_tick) >= max(self.delay_steps, 10):
            self.calls += 1
            self.last_call_tick = self.tick
            self.pending = True
            choice = self._slow_planner_choice(obs, ego_y, max_y)
            self.queue.append((self.tick + self.delay_steps, choice))

        # Fast safety filter + Option-Preserving Reserve while waiting!
        reserve = self.option_reserve_m if (self.mode == "ours_reserve" and self.pending) else 0.0
        safe_d = max(15.0, ego_vx * 1.15) + reserve

        self.tick += 1
        if admitted_cmd is not None and fs > 10.0:
            return admitted_cmd
        if fs < safe_d or (fs < safe_d + 8.0 and f_dvx < -2.0):
            return SLOWER
        if fs > safe_d + 10.0:
            return FASTER
        return IDLE


def run_env(env_id: str, mode: str, seeds: list[int]) -> dict:
    # Configure 10 Hz control and unnormalized Kinematics obs (exact same obs for all methods)
    cfg = dict(
        observation=dict(
            type="Kinematics",
            vehicles_count=10,
            features=["presence", "x", "y", "vx", "vy"],
            absolute=False,
            normalize=False,
            order="sorted",
        ),
        policy_frequency=10,
        simulation_frequency=15,
        duration=25,
    )
    if env_id == "highway-v0":
        cfg["lanes_count"] = 4
        cfg["vehicles_density"] = 1.4
        max_y = 12.0
    elif env_id == "merge-v1":
        max_y = 8.0
    else:
        max_y = 12.0

    env = gym.make(env_id, config=cfg)
    crashes = 0
    distances = []
    speeds = []
    calls_list = []
    latencies_ms = []
    t0 = time.monotonic()

    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            # In Kinematics(absolute=False), row 0 has ego absolute x, y, vx, vy!
            start_x = float(obs[0, 1])
            drv = HighwayEnvStack(mode=mode, delay_steps=15, option_reserve_m=12.0)
            ep_speeds = []
            crashed = False
            while True:
                ts0 = time.perf_counter()
                action = drv.act(obs, max_y=max_y)
                latencies_ms.append((time.perf_counter() - ts0) * 1000.0)
                obs, _, term, trunc, info = env.step(action)
                ep_speeds.append(float(math.hypot(obs[0, 3], obs[0, 4])))
                if info.get("crashed", False):
                    crashed = True
                if term or trunc:
                    break
            crashes += int(crashed)
            distances.append(max(0.0, float(obs[0, 1]) - start_x))
            speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
            calls_list.append(drv.calls)
    finally:
        env.close()

    return dict(
        env_id=env_id,
        mode=mode,
        episodes=len(seeds),
        crash_free_rate=1.0 - crashes / len(seeds),
        crashes=crashes,
        mean_speed_mps=float(np.mean(speeds)),
        mean_distance_m=float(np.mean(distances)),
        mean_calls=float(np.mean(calls_list)),
        median_ms_per_step=float(np.median(latencies_ms)),
        wall_seconds=round(time.monotonic() - t0, 2),
    )


def _run_task(args: tuple[str, str, list[int]]) -> dict:
    env_id, mode, seeds = args
    res = run_env(env_id, mode, seeds)
    print(f"DONE {env_id:14s} | {mode:22s} | crash_free={res['crash_free_rate']*100:5.1f}% | "
          f"avg_speed={res['mean_speed_mps']:4.1f} m/s | "
          f"calls={res['mean_calls']:5.1f} | {res['median_ms_per_step']:.3f} ms/step", flush=True)
    return res


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    seeds = list(range(100, 150))  # 50 seeds per scenario
    envs = ["highway-v0", "merge-v1", "roundabout-v1"]
    modes = [
        "unshielded",
        "static_filter",
        "always_ask_unshielded",
        "event_no_reserve",
        "ours_reserve",
    ]
    tasks = [(env_id, mode, seeds) for env_id in envs for mode in modes]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_run_task, tasks))

    # Aggregate across all 150 episodes per mode
    print("\n=== AGGREGATE ACROSS ALL 3 HIGHWAY-ENV SCENARIOS (150 episodes/mode) ===", flush=True)
    for mode in modes:
        sub = [r for r in results if r["mode"] == mode]
        print(f"  {mode:22s} | crash_free={np.mean([r['crash_free_rate'] for r in sub])*100:5.1f}% | "
              f"avg_speed={np.mean([r['mean_speed_mps'] for r in sub]):4.1f} m/s | "
              f"calls={np.mean([r['mean_calls'] for r in sub]):5.1f} | "
              f"{np.median([r['median_ms_per_step'] for r in sub]):.3f} ms/step", flush=True)

    out = Path("runs/benchmarks/highway-env-2026-11-25.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
