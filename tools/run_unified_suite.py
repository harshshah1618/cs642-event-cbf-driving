"""Run the canonical UnifiedDrivingSystem benchmark & one-switch-at-a-time ablations.

Every single row below executes the EXACT SAME 3-layer UnifiedDrivingSystem
(strictly 259-dim sensor input, zero simulator peeking):
  - Table A: 50 fresh untouched maps (600-649): PPO (3 seeds) & Analytic
  - Table B: 50 held-out maps (500-549): Full system & 6 single-component ablations
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


def _eval_group(task: dict) -> dict:
    from metadrive import SafeMetaDriveEnv
    from driving.safemetadrive.unified import UnifiedDrivingSystem

    start_seed = int(task["start_seed"])
    num_scenarios = int(task["num_scenarios"])
    params = task["parameters"]
    cfg = dict(
        start_seed=start_seed,
        num_scenarios=num_scenarios,
        map=3,
        horizon=1000,
        accident_prob=0.8,
        traffic_density=0.05,
        crash_vehicle_done=False,
        crash_object_done=False,
        use_render=False,
        log_level=50,
    )
    env = SafeMetaDriveEnv(cfg)
    t0 = time.monotonic()
    episodes = []
    latencies_ms = []
    try:
        for seed in range(start_seed, start_seed + num_scenarios):
            obs, _ = env.reset(seed=seed)
            drv = UnifiedDrivingSystem(obs, **params)
            reward = 0.0
            cost = 0.0
            steps = 0
            info = {}
            while True:
                ts0 = time.perf_counter()
                act = drv.act()
                latencies_ms.append((time.perf_counter() - ts0) * 1000.0)
                obs, r, term, trunc, info = env.step(act)
                drv.on_observation(obs)
                reward += float(r)
                cost += float(info["cost"])
                steps += 1
                if term or trunc:
                    break
            succ = bool(info.get("arrive_dest", False))
            episodes.append(dict(
                seed=seed,
                success=succ,
                zero_cost_success=succ and cost == 0.0,
                cost=cost,
                reward=reward,
                steps=steps,
                router_calls=drv.router_calls,
                lane_changes=drv.changes,
                mean_steer_delta=float(np.mean(drv.steer_deltas)) if drv.steer_deltas else 0.0,
                peak_steer_delta=float(np.max(drv.steer_deltas)) if drv.steer_deltas else 0.0,
            ))
    finally:
        env.close()

    summary = dict(
        split=task["split"],
        name=task["name"],
        episodes=len(episodes),
        success_rate=float(np.mean([e["success"] for e in episodes])),
        zero_cost_success_rate=float(np.mean([e["zero_cost_success"] for e in episodes])),
        mean_cost=float(np.mean([e["cost"] for e in episodes])),
        mean_reward=float(np.mean([e["reward"] for e in episodes])),
        mean_calls=float(np.mean([e["router_calls"] for e in episodes])),
        mean_steer_delta=float(np.mean([e["mean_steer_delta"] for e in episodes])),
        peak_steer_delta=float(np.mean([e["peak_steer_delta"] for e in episodes])),
        median_ms_per_step=float(np.median(latencies_ms)),
        wall_seconds=round(time.monotonic() - t0, 2),
    )
    print(
        f"DONE {summary['split']:13s} | {summary['name']:36s} | "
        f"succ={summary['success_rate']*100:5.1f}% zc={summary['zero_cost_success_rate']*100:5.1f}% "
        f"cost={summary['mean_cost']:5.3f} calls={summary['mean_calls']:6.1f} "
        f"dsteer={summary['peak_steer_delta']:.3f} ({summary['median_ms_per_step']:.3f}ms)",
        flush=True,
    )
    return dict(summary=summary, episodes=episodes)


def main() -> None:
    ppo0 = "runs/safemetadrive/skills-1m-v1/skills-seed0/ppo.zip"
    ppo1 = "runs/safemetadrive/skills-1m-v1/skills-seed1/ppo.zip"
    ppo2 = "runs/safemetadrive/skills-1m-v1/skills-seed2/ppo.zip"

    tasks = [
        # --- Table A: Fresh untouched maps (600-649) ---
        dict(split="fresh_600_649", start_seed=600, num_scenarios=50,
             name="Unified (PPO seed0)", parameters=dict(tracker="ppo", skill_checkpoint=ppo0)),
        dict(split="fresh_600_649", start_seed=600, num_scenarios=50,
             name="Unified (PPO seed1)", parameters=dict(tracker="ppo", skill_checkpoint=ppo1)),
        dict(split="fresh_600_649", start_seed=600, num_scenarios=50,
             name="Unified (PPO seed2)", parameters=dict(tracker="ppo", skill_checkpoint=ppo2)),
        dict(split="fresh_600_649", start_seed=600, num_scenarios=50,
             name="Unified (Analytic Layer 2)", parameters=dict(tracker="analytic")),

        # --- Table B: One-switch-at-a-time ablations on held-out maps (500-549) ---
        dict(split="ablation_500", start_seed=500, num_scenarios=50,
             name="Full Unified (PPO seed0)", parameters=dict(tracker="ppo", skill_checkpoint=ppo0)),
        dict(split="ablation_500", start_seed=500, num_scenarios=50,
             name="No Safety Filter (Layer 1 OFF)", parameters=dict(tracker="ppo", skill_checkpoint=ppo0, use_safety_filter=False)),
        dict(split="ablation_500", start_seed=500, num_scenarios=50,
             name="No Replanning (Layer 3 OFF)", parameters=dict(tracker="ppo", skill_checkpoint=ppo0, router="none", option_reserve_m=0.0)),
        dict(split="ablation_500", start_seed=500, num_scenarios=50,
             name="Always-Poll 10Hz (No Event Trigger)", parameters=dict(tracker="ppo", skill_checkpoint=ppo0, router="always_poll")),
        dict(split="ablation_500", start_seed=500, num_scenarios=50,
             name="Hard Switch (No Blending)", parameters=dict(tracker="ppo", skill_checkpoint=ppo0, blend_steps=0)),
        dict(split="ablation_500", start_seed=500, num_scenarios=50,
             name="Analytic Layer 2 (No PPO)", parameters=dict(tracker="analytic")),
        dict(split="ablation_500", start_seed=500, num_scenarios=50,
             name="4.0s Delay WITH Option Reserve", parameters=dict(tracker="ppo", skill_checkpoint=ppo0, decision_delay_steps=40, option_reserve_m=6.5)),
        dict(split="ablation_500", start_seed=500, num_scenarios=50,
             name="4.0s Delay WITHOUT Option Reserve", parameters=dict(tracker="ppo", skill_checkpoint=ppo0, decision_delay_steps=40, option_reserve_m=0.0)),
    ]

    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_eval_group, tasks))

    # Print 3-seed PPO average on fresh_600_649
    ppo_fresh = [r["summary"] for r in results if r["summary"]["split"] == "fresh_600_649" and "PPO" in r["summary"]["name"]]
    print("\n=== FRESH 600-649: UNIFIED (PPO 3-SEED MEAN, ZERO PEEKING) ===")
    print(f"  Success:   {np.mean([s['success_rate'] for s in ppo_fresh])*100:.1f}%")
    print(f"  Zero-cost: {np.mean([s['zero_cost_success_rate'] for s in ppo_fresh])*100:.1f}%")
    print(f"  Mean cost: {np.mean([s['mean_cost'] for s in ppo_fresh]):.3f}")
    print(f"  Calls/ep:  {np.mean([s['mean_calls'] for s in ppo_fresh]):.2f}")
    print(f"  Speed:     {np.median([s['median_ms_per_step'] for s in ppo_fresh]):.3f} ms/step")

    out = Path("runs/benchmarks/unified-system-suite-2026-11-25.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
