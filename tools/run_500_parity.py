"""Run exact-same-runner parity baselines on held-out maps 500-549 (CPU, $0)."""

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


def _eval_one(task: dict) -> dict:
    from metadrive import SafeMetaDriveEnv
    from driving.safemetadrive.policies import make_driver

    cfg = dict(
        start_seed=500, num_scenarios=50, map=3, horizon=1000,
        accident_prob=0.8, traffic_density=0.05,
        crash_vehicle_done=False, crash_object_done=False,
        use_render=False, log_level=50,
    )
    env = SafeMetaDriveEnv(cfg)
    t0 = time.monotonic()
    episodes = []
    try:
        for seed in range(500, 550):
            obs, _ = env.reset(seed=seed)
            drv = make_driver(task["policy"], env.agent, task["parameters"], observation=obs, episode_seed=seed)
            reward, cost, steps, info = 0.0, 0.0, 0, {}
            while True:
                act = drv.act()
                obs, r, term, trunc, info = env.step(act)
                if hasattr(drv, "on_observation"):
                    drv.on_observation(obs)
                reward += float(r)
                cost += float(info["cost"])
                steps += 1
                if term or trunc:
                    break
            succ = bool(info.get("arrive_dest", False))
            episodes.append(dict(seed=seed, success=succ, zero_cost_success=succ and cost == 0.0,
                                 cost=cost, reward=reward, steps=steps))
            if hasattr(drv, "close"):
                drv.close()
            elif hasattr(drv, "destroy"):
                drv.destroy()
    finally:
        env.close()
    s = dict(
        name=task["name"], policy=task["policy"], episodes=len(episodes),
        success_rate=float(np.mean([e["success"] for e in episodes])),
        zero_cost_success_rate=float(np.mean([e["zero_cost_success"] for e in episodes])),
        mean_cost=float(np.mean([e["cost"] for e in episodes])),
        mean_reward=float(np.mean([e["reward"] for e in episodes])),
        wall_seconds=round(time.monotonic() - t0, 2),
    )
    print(f"DONE 500-549 | {s['name']:28s} | succ={s['success_rate']*100:5.1f}% "
          f"zc={s['zero_cost_success_rate']*100:5.1f}% cost={s['mean_cost']:6.3f} ret={s['mean_reward']:6.1f}",
          flush=True)
    return dict(summary=s, episodes=episodes)


def main() -> None:
    art = Path("runs/safemetadrive/heldout-study-v1/artifacts")
    ppo_lag = [
        str(art / "baselines-1m-v1/PPOLag-seed0/PPOLag-{CS642SafeMetaDrive-v0}/seed-000-2026-11-19-02-02-42/torch_save/epoch-250.pt"),
        str(art / "baselines-completion-v2/PPOLag-seed1/PPOLag-{CS642SafeMetaDrive-v0}/seed-001-2026-11-19-18-27-20/torch_save/epoch-250.pt"),
        str(art / "baselines-1m-v1/PPOLag-seed2/PPOLag-{CS642SafeMetaDrive-v0}/seed-002-2026-11-19-02-02-57/torch_save/epoch-250.pt"),
    ]
    cpo = [
        str(art / "cpo-critic-lr-v1/CPO-seed0/CPO-{CS642SafeMetaDrive-v0}/seed-000-2026-11-19-19-28-02/torch_save/epoch-150.pt"),
        str(art / "cpo-critic-lr-v1/CPO-seed1/CPO-{CS642SafeMetaDrive-v0}/seed-001-2026-11-19-19-28-02/torch_save/epoch-150.pt"),
        str(art / "cpo-critic-lr-v1/CPO-seed2/CPO-{CS642SafeMetaDrive-v0}/seed-002-2026-11-19-19-28-03/torch_save/epoch-150.pt"),
    ]
    tasks = [
        dict(name="Unified (PPO seed1)", policy="unified_system",
             parameters=dict(tracker="ppo", skill_checkpoint="runs/safemetadrive/skills-1m-v1/skills-seed1/ppo.zip")),
        dict(name="Unified (PPO seed2)", policy="unified_system",
             parameters=dict(tracker="ppo", skill_checkpoint="runs/safemetadrive/skills-1m-v1/skills-seed2/ppo.zip")),
        dict(name="IDM", policy="idm", parameters=dict()),
    ]
    for i, ckpt in enumerate(ppo_lag):
        tasks.append(dict(name=f"PPO-Lag seed{i}", policy="native_learned",
                          parameters=dict(checkpoint=ckpt, deterministic=False, policy_seed=1000 * i)))
    for i, ckpt in enumerate(cpo):
        tasks.append(dict(name=f"CPO seed{i}", policy="native_learned",
                          parameters=dict(checkpoint=ckpt, deterministic=False, policy_seed=2000 * i)))

    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_eval_one, tasks))

    out = Path("runs/benchmarks/parity-500-549-2026-11-25.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
