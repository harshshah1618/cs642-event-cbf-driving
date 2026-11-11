"""Development probes for delay-induced physical stop/turn stalls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.eval.runner import load_skill_bank
from event_cbf_driving.router.scripted import ScriptedRouter
from event_cbf_driving.skills.classical import PDLanePolicy
from event_cbf_driving.stack.controller import DrivingStack
from event_cbf_driving.stack.maneuver_reserve import ManeuverReserveShield

ROOT = Path(__file__).resolve().parents[1]
BANK = None


class CorrectIntent(ScriptedRouter):
    name = "known_correct_delayed_overtake"

    def decide(self, req):
        return "OVERTAKE_LEFT"


@dataclass(frozen=True)
class Job:
    seed: int
    delay: float
    mode: str
    reserve: float = 8.0
    controller: str = "ppo"
    speed: float = 21.89514964766451
    gap: float = 68.22207213042444
    steering_deg: float = 40.0
    scale: float = 1.0

    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:24]


def initialize():
    import torch

    global BANK
    torch.set_num_threads(1)
    BANK, _ = load_skill_bank(ROOT / "checkpoints/metadrive-calibrated")


def execute(job, record=False):
    cfg = load_config(ROOT / "configs/metadrive-study.yaml")
    cfg.stack.mode = "fixed_rate_shielded"
    cfg.stack.fixed_rate_hz = 2
    cfg.stack.simulate_deadline_wait = True
    cfg.stack.max_pending_s = job.delay + 1
    cfg.stack.pending_margin = cfg.shield.pending_margin = False
    cfg.stack.record_interactions = record
    cfg.stack.skill_switch_guard = True
    cfg.shield.gamma_sigma = 0
    shield = ManeuverReserveShield(
        cfg.shield, mode=job.mode, reserve_m=job.reserve, scale=job.scale
    )
    bank = (
        BANK
        if job.controller == "ppo"
        else {
            name: PDLanePolicy(name) for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")
        }
    )
    stack = DrivingStack(bank, CorrectIntent(latency_s=job.delay), shield=shield, config=cfg.stack)
    spec = ScenarioConfig(
        "two_trap",
        simulator="metadrive",
        route_length=160,
        duration=60,
        params={
            "spawn_x": 0,
            "ego_vx": job.speed,
            "stalled_x": job.gap,
            "cut_in_trigger_x": 10000,
        },
    )
    env = make_env("two_trap", spec)
    snapshot = env.snapshot
    initial = {}

    def geometry_snapshot():
        snap = snapshot()
        if env.env is not None:
            vehicle = env.env.vehicle
            snap.info["maneuver_geometry"] = {
                "wheelbase": float(vehicle.FRONT_WHEELBASE + vehicle.REAR_WHEELBASE),
                "steering_limit_rad": math.radians(
                    float(vehicle.get_dynamics_parameters()["max_steering"])
                ),
            }
        return snap

    env.snapshot = geometry_snapshot
    reset = env.reset

    def initial_reset(*args, **kwargs):
        result = reset(*args, **kwargs)
        vehicle = env.env.vehicle
        vehicle.max_steering = job.steering_deg
        vehicle.config["max_steering"] = job.steering_deg
        snap = env.snapshot()
        initial.update(
            ego=asdict(snap.ego),
            others=[asdict(v) for v in snap.others],
            geometry=snap.info["maneuver_geometry"],
        )
        return result

    env.reset = initial_reset
    started = time.perf_counter()
    try:
        result = stack.run(env, seed=job.seed)
    finally:
        env.close()
    active = stack.last_shield
    output = {
        "id": job.key(),
        **asdict(job),
        **{
            k: result.to_dict()[k]
            for k in (
                "route_pct",
                "crashed",
                "freeze",
                "end_time",
                "end_reason",
                "router_calls",
                "calls_applied",
                "calls_rejected",
                "x",
                "y",
                "vx",
                "braking_infeasible_ticks",
            )
        },
        "success": not result.crashed and result.route_pct >= 99,
        "wall_s": time.perf_counter() - started,
        "reserve_events": active.events,
        "reserve_integral": active.reserve_integral,
        "reserve_peak": active.reserve_peak,
        "initial": initial,
    }
    if record:
        target = ROOT / "runs/maneuver-development/traces"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{job.key()}.json").write_text(json.dumps(result.to_dict(), allow_nan=False))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--record", action="store_true")
    p.add_argument(
        "--modes",
        nargs="+",
        choices=ManeuverReserveShield.MODES,
        default=["baseline", "static", "fixed_latched", "geometric_latched"],
    )
    p.add_argument("--output", type=Path, default=ROOT / "runs/maneuver-development/pilot-v2.jsonl")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    design = [Job(7101, delay, mode) for delay in (0.125, 12.0, 18.0) for mode in a.modes]
    existing = (
        {json.loads(line)["id"] for line in a.output.read_text().splitlines()}
        if a.output.exists()
        else set()
    )
    with a.output.open("a") as f, ProcessPoolExecutor(a.workers, initializer=initialize) as pool:
        futures = [pool.submit(execute, j, a.record) for j in design if j.key() not in existing]
        for future in as_completed(futures):
            row = future.result()
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            print(
                row["delay"],
                row["mode"],
                round(row["route_pct"], 2),
                row["crashed"],
                row["freeze"],
                flush=True,
            )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
