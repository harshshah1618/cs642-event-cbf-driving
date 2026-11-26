"""Run the frozen, strictly equal-observation SafeMetaDrive benchmark.

Evaluates:
1. Standard held-out test split (seeds 500-549, 50 maps):
   - Ours: LiDAR-Only + Replan + Option Reserve (0 s delay)
   - Ablation: LiDAR-Only Static Lane (no replan)
   - Ablation: LiDAR-Only + 1.5 s delay WITH Option Reserve
   - Ablation: LiDAR-Only + 1.5 s delay WITHOUT Option Reserve
   - Upstream PPO-Expert (same 259-dim obs)
2. Brand-new untouched fresh split (seeds 600-649, 50 maps):
   - Ours: LiDAR-Only + Replan + Option Reserve
   - IDM (upstream rule with simulator object access)
   - Upstream PPO-Expert (same 259-dim obs)
   - PPO-Lag (all 3 trained seeds, same 259-dim obs)
   - CPO (all 3 trained seeds, same 259-dim obs)
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


def _run_group(task: dict) -> dict:
    from metadrive import SafeMetaDriveEnv
    from driving.safemetadrive.policies import make_driver

    name = task["name"]
    policy = task["policy"]
    params = task.get("parameters", {})
    start_seed = int(task["start_seed"])
    num_scenarios = int(task["num_scenarios"])
    seeds = list(range(start_seed, start_seed + num_scenarios))

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
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            driver = make_driver(policy, env.agent, params, observation=obs, episode_seed=seed)
            reward = 0.0
            cost = 0.0
            crashes = set()
            steps = 0
            info = {}
            while True:
                action = driver.act()
                obs, r, term, trunc, info = env.step(action)
                if hasattr(driver, "on_observation"):
                    driver.on_observation(obs)
                steps += 1
                reward += float(r)
                cost += float(info["cost"])
                for kind in ("crash_vehicle", "crash_object", "out_of_road", "crash_human"):
                    if info.get(kind, False):
                        crashes.add(kind)
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
                route_completion=float(info.get("route_completion", 0.0)),
                violations=sorted(crashes),
                lane_changes=getattr(driver, "changes", None),
                reserve_steps=getattr(driver, "reserve_active_steps", None),
            ))
            if hasattr(driver, "close"):
                driver.close()
            elif hasattr(driver, "destroy"):
                driver.destroy()
    finally:
        env.close()

    summary = dict(
        name=name,
        split=task["split_label"],
        policy=policy,
        episodes=len(episodes),
        success_rate=float(np.mean([e["success"] for e in episodes])),
        zero_cost_success_rate=float(np.mean([e["zero_cost_success"] for e in episodes])),
        mean_cost=float(np.mean([e["cost"] for e in episodes])),
        mean_reward=float(np.mean([e["reward"] for e in episodes])),
        mean_route_completion=float(np.mean([e["route_completion"] for e in episodes])),
        wall_seconds=round(time.monotonic() - t0, 2),
    )
    print(f"DONE {task['split_label']} | {name}: succ={summary['success_rate']*100:.1f}% "
          f"zc={summary['zero_cost_success_rate']*100:.1f}% cost={summary['mean_cost']:.3f} "
          f"ret={summary['mean_reward']:.1f} ({summary['wall_seconds']}s)", flush=True)
    return dict(summary=summary, episodes=episodes)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "runs/safemetadrive/heldout-study-v1/artifacts"
    ppo_lag_ckpts = [
        str(art / "baselines-1m-v1/PPOLag-seed0/PPOLag-{CS642SafeMetaDrive-v0}/seed-000-2026-11-19-02-02-42/torch_save/epoch-250.pt"),
        str(art / "baselines-completion-v2/PPOLag-seed1/PPOLag-{CS642SafeMetaDrive-v0}/seed-001-2026-11-19-18-27-20/torch_save/epoch-250.pt"),
        str(art / "baselines-1m-v1/PPOLag-seed2/PPOLag-{CS642SafeMetaDrive-v0}/seed-002-2026-11-19-02-02-57/torch_save/epoch-250.pt"),
    ]
    cpo_ckpts = [
        str(art / "cpo-critic-lr-v1/CPO-seed0/CPO-{CS642SafeMetaDrive-v0}/seed-000-2026-11-19-19-28-02/torch_save/epoch-150.pt"),
        str(art / "cpo-critic-lr-v1/CPO-seed1/CPO-{CS642SafeMetaDrive-v0}/seed-001-2026-11-19-19-28-02/torch_save/epoch-150.pt"),
        str(art / "cpo-critic-lr-v1/CPO-seed2/CPO-{CS642SafeMetaDrive-v0}/seed-002-2026-11-19-19-28-03/torch_save/epoch-150.pt"),
    ]

    tasks = [
        # --- Standard held-out split (500-549) ---
        dict(split_label="test_500_549", start_seed=500, num_scenarios=50,
             name="Ours (LiDAR-only + Option Reserve)", policy="lidar_option_reserve",
             parameters=dict()),
        dict(split_label="test_500_549", start_seed=500, num_scenarios=50,
             name="Ablation: LiDAR-only Static Lane", policy="lidar_option_reserve",
             parameters=dict(allow_lane_change=False, option_reserve_m=0.0)),
        dict(split_label="test_500_549", start_seed=500, num_scenarios=50,
             name="Ours + 1.5s delay (WITH Reserve)", policy="lidar_option_reserve",
             parameters=dict(decision_delay_steps=15, option_reserve_m=6.5)),
        dict(split_label="test_500_549", start_seed=500, num_scenarios=50,
             name="Ablation: 1.5s delay (NO Reserve)", policy="lidar_option_reserve",
             parameters=dict(decision_delay_steps=15, option_reserve_m=0.0)),
        dict(split_label="test_500_549", start_seed=500, num_scenarios=50,
             name="Upstream PPO-Expert (259-dim obs)", policy="upstream_expert",
             parameters=dict()),
        # --- Brand-new untouched fresh split (600-649) ---
        dict(split_label="fresh_600_649", start_seed=600, num_scenarios=50,
             name="Ours (LiDAR-only + Option Reserve)", policy="lidar_option_reserve",
             parameters=dict()),
        dict(split_label="fresh_600_649", start_seed=600, num_scenarios=50,
             name="IDM (privileged object state)", policy="idm",
             parameters=dict()),
        dict(split_label="fresh_600_649", start_seed=600, num_scenarios=50,
             name="Upstream PPO-Expert (259-dim obs)", policy="upstream_expert",
             parameters=dict()),
    ]
    for i, ckpt in enumerate(ppo_lag_ckpts):
        if Path(ckpt).exists():
            tasks.append(dict(
                split_label="fresh_600_649", start_seed=600, num_scenarios=50,
                name=f"PPO-Lag seed{i}", policy="native_learned",
                parameters=dict(checkpoint=ckpt, deterministic=False, policy_seed=1000 * i),
            ))
    for i, ckpt in enumerate(cpo_ckpts):
        if Path(ckpt).exists():
            tasks.append(dict(
                split_label="fresh_600_649", start_seed=600, num_scenarios=50,
                name=f"CPO seed{i}", policy="native_learned",
                parameters=dict(checkpoint=ckpt, deterministic=False, policy_seed=2000 * i),
            ))

    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for res in pool.map(_run_group, tasks):
            results.append(res)

    out_path = root / "runs/safemetadrive/fair-lidar-benchmark-2026-11-25.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full benchmark results to {out_path}")


if __name__ == "__main__":
    main()
