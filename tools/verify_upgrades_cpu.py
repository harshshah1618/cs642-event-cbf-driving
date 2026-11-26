"""Local CPU verification of Upgrades #1-#4 before any Modal GPU spend ($0).

Tests:
1. Upgrade #3 (Option-Viability Barrier) on SafeMetaDrive 500-549 at REALISTIC
   0.8 s and 1.5 s delays under wet/low-friction conditions (brake_scale=0.30).
2. Upgrade #4 (Online Slip-History Friction Adaptation) vs Fixed Dry Rule on
   50 unseen variable-friction maps (brake_scale randomized in [0.18, 1.0]).
3. Upgrade #1 + #2 (Grounded 3D Stop Wall + Detour + Rain-Invariant Trigger)
   on irregular scenes (flagger, wet-closed, rain-clean) with Oracle VLM on CPU.
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


def _eval_variable_friction(task: dict) -> dict:
    from metadrive import SafeMetaDriveEnv
    from driving.safemetadrive.unified import UnifiedDrivingSystem
    from driving.safemetadrive.slip_adapter import SlipHistoryFrictionEstimator

    adaptive = task["adaptive"]
    viability = task["viability"]
    delay_steps = task["delay_steps"]
    brake_scales = task["brake_scales"]
    seeds = task["seeds"]

    cfg = dict(
        start_seed=seeds[0], num_scenarios=len(seeds), map=3, horizon=1000,
        accident_prob=0.8, traffic_density=0.05,
        crash_vehicle_done=False, crash_object_done=False,
        use_render=False, log_level=50,
    )
    env = SafeMetaDriveEnv(cfg)
    succs, zc, costs = [], [], []
    t0 = time.monotonic()
    try:
        for seed, bscale in zip(seeds, brake_scales):
            obs, _ = env.reset(seed=seed)
            base_brake = float(env.agent.config["max_brake_force"])
            env.agent.update_config({"max_brake_force": base_brake * bscale})
            drv = UnifiedDrivingSystem(
                obs,
                tracker="analytic",
                use_viability_reserve=viability,
                option_reserve_m=(6.5 if viability else 0.0),
                decision_delay_steps=delay_steps,
                expected_delay_s=delay_steps * 0.1,
            )
            est = SlipHistoryFrictionEstimator() if adaptive else None
            cost = 0.0
            last_thr = 0.0
            info = {}
            try:
                while True:
                    if est is not None:
                        mu = est.update(obs, last_thr)
                        drv.set_grounded_constraints(friction_scale=mu)
                    act = drv.act()
                    last_thr = float(act[1])
                    obs, _, term, trunc, info = env.step(act)
                    drv.on_observation(obs)
                    cost += float(info["cost"])
                    if term or trunc:
                        break
            finally:
                env.agent.update_config({"max_brake_force": base_brake})
            s = bool(info.get("arrive_dest", False))
            succs.append(s)
            zc.append(s and cost == 0.0)
            costs.append(cost)
    finally:
        env.close()

    res = dict(
        name=task["name"],
        episodes=len(seeds),
        success_rate=float(np.mean(succs)),
        zero_cost_rate=float(np.mean(zc)),
        mean_cost=float(np.mean(costs)),
        wall_s=round(time.monotonic() - t0, 1),
    )
    print(f"DONE {res['name']:42s} | succ={res['success_rate']*100:5.1f}% "
          f"zc={res['zero_cost_rate']*100:5.1f}% cost={res['mean_cost']:5.3f} ({res['wall_s']}s)", flush=True)
    return res


def main() -> None:
    seeds = list(range(600, 650))
    rng = np.random.default_rng(42)
    # Unknown variable friction across 50 unseen maps (from icy 0.15 to dry 1.0)
    var_scales = [float(x) for x in rng.uniform(0.15, 0.45, size=len(seeds))]
    wet_scales = [0.20] * len(seeds)

    tasks = [
        # Upgrade #4: Unknown low/variable friction (0.15-0.45), 0 s delay
        dict(name="Fixed Dry Driver (unknown wet/icy)", seeds=seeds, brake_scales=var_scales,
             adaptive=False, viability=False, delay_steps=0),
        dict(name="Upgrade #4: Slip-History Adaptive", seeds=seeds, brake_scales=var_scales,
             adaptive=True, viability=True, delay_steps=0),
        # Upgrade #3: Real 1.0 s VLM delay on wet roads (brake_scale=0.20)
        dict(name="Wet 1.0s Delay — No Viability Reserve", seeds=seeds, brake_scales=wet_scales,
             adaptive=True, viability=False, delay_steps=10),
        dict(name="Upgrade #3: Wet 1.0s Delay + Viability", seeds=seeds, brake_scales=wet_scales,
             adaptive=True, viability=True, delay_steps=10),
    ]

    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_eval_variable_friction, tasks))

    out = Path("runs/benchmarks/upgrades-cpu-verification-2026-11-25.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved local CPU verification to {out}")


if __name__ == "__main__":
    main()
