"""Upstream MetaDrive evaluation with shared fast controller and delayed planner.

No environment dynamics, collision flags, termination rules or traffic managers
are overridden. These are environment-based comparisons, not leaderboard claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from event_cbf_driving.stack.road_planning import (
    RoadController,
    plan_maneuver,
    road_request_due,
)

from research_freeze import verify

ROOT = Path(__file__).resolve().parents[1]
DELAYS = {
    "standard": (0.0, 0.0),
    "nominal": (0.3, 0.0),
    "slow": (1.0, 0.0),
    "long": (2.0, 0.0),
    "jitter": (0.8, 0.5),
}


@dataclass(frozen=True)
class Job:
    environment: str
    seed: int
    delay: str
    method: str
    value: float = 0.0

    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]


def initialize():
    import torch

    torch.set_num_threads(1)


def map_fingerprint(env):
    lanes = env.current_map.road_network.graph
    geometry = []
    for origin in sorted(lanes):
        for end in sorted(lanes[origin]):
            for lane in lanes[origin][end]:
                samples = [
                    np.asarray(lane.position(lane.length * t, 0)).round(5).tolist()
                    for t in (0.0, 0.25, 0.5, 0.75, 1.0)
                ]
                geometry.append([origin, end, lane.length, samples])
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def execute(job):
    from metadrive import MetaDriveEnv, SafeMetaDriveEnv
    from metadrive.policy.expert_policy import ExpertPolicy
    from metadrive.policy.idm_policy import IDMPolicy

    cls = SafeMetaDriveEnv if job.environment == "safe" else MetaDriveEnv
    config = dict(
        start_seed=job.seed,
        num_scenarios=1,
        use_render=False,
        log_level=50,
        show_terrain=False,
        show_skybox=False,
        preload_models=False,
    )
    env = cls(config)
    controller = reference = None
    rng = np.random.default_rng(job.seed + 90001)
    delay, jitter = DELAYS[job.delay]
    calls = rejected = changes = expired = 0
    planner_cpu = scheduler_cpu = 0.0
    planner_process = scheduler_process = control_process = 0.0
    controller_elapsed = 0.0
    obsolete_speed_ticks = 0
    collision = False
    first_collision = None
    safety_cost = 0.0
    pending = None
    last_call = -100.0
    intervention_streak = 0
    previous_velocity = previous_acceleration = None
    peak_jerk = 0.0
    wall_start = time.perf_counter()
    try:
        env.reset(seed=job.seed)
        fingerprint = map_fingerprint(env)
        resolved_config = json.loads(json.dumps(dict(env.config), default=str))
        if job.method in ("idm", "expert"):
            reference = (
                IDMPolicy(env.vehicle, job.seed)
                if job.method == "idm"
                else ExpertPolicy(env.vehicle, job.seed)
            )
            if job.method == "idm":
                reference.NORMAL_SPEED = 40.0
                reference.target_speed = 40.0
        else:
            controller = RoadController(env.vehicle, job.seed)
        dt = float(env.config["decision_repeat"] * env.config["physics_world_step_size"])
        for tick in range(int(env.config["horizon"])):
            t = tick * dt
            if reference is not None:
                started = time.perf_counter()
                cpu_started = time.process_time()
                action = np.clip(reference.act(), -1, 1).tolist()
                planner_cpu += time.perf_counter() - started
                planner_process += time.process_time() - cpu_started
            else:
                control_started = time.perf_counter()
                control_cpu_started = time.process_time()
                view = controller.observe()
                obsolete_speed_ticks += int(
                    view.speed_reference < 10.0
                    and view.lanes[view.current].front_gap > 50
                    and view.settled
                )
                road = tuple(controller.target_lane.index[:2])
                if pending is not None and t + 1e-9 >= pending[0]:
                    _, old_road, lane, speed = pending
                    if old_road != road or lane >= len(view.lanes):
                        rejected += 1
                        expired += 1
                    else:
                        old = view.current
                        if controller.accept(lane, speed, view):
                            changes += int(old != lane)
                        else:
                            rejected += 1
                    pending = None
                    view = controller.observe()
                started = time.perf_counter()
                cpu_started = time.process_time()
                if job.method in ("periodic", "periodic_conditioned", "periodic_gated"):
                    due = t - last_call >= 1 / job.value - 1e-9
                    if due and job.method == "periodic_gated":
                        due = road_request_due(view, "benefit", delay + jitter)
                        if not due:
                            last_call = t
                elif job.method == "shield":
                    due = intervention_streak >= 5 and t - last_call >= 1
                elif job.method == "never":
                    due = False
                else:
                    due = road_request_due(
                        view,
                        "no_latency" if job.method == "no_latency_matched" else job.method,
                        delay + jitter,
                        buffer=job.value
                        if job.method in ("deadline", "no_latency_matched")
                        else 0.5,
                    )
                scheduler_cpu += time.perf_counter() - started
                scheduler_process += time.process_time() - cpu_started
                rate_clear = (
                    job.method in ("periodic", "periodic_conditioned", "periodic_gated")
                    or t - last_call >= 0.5
                )
                if due and pending is None and rate_clear:
                    started = time.perf_counter()
                    cpu_started = time.process_time()
                    lane, speed = plan_maneuver(view)
                    planner_cpu += time.perf_counter() - started
                    planner_process += time.process_time() - cpu_started
                    calls += 1
                    last_call = t
                    latency = max(0, delay + rng.uniform(-jitter, jitter))
                    pending = (t + latency, road, lane, speed)
                    if latency == 0:
                        old = view.current
                        if controller.accept(lane, speed, view):
                            changes += int(old != lane)
                        else:
                            rejected += 1
                        pending = None
                action = controller.action(view)
                controller_elapsed += time.perf_counter() - control_started
                control_process += time.process_time() - control_cpu_started
                intervention_streak = intervention_streak + 1 if action[1] < 0 else 0
            _obs, reward, term, trunc, info = env.step(action)
            contact = any(
                bool(info.get(k, False))
                for k in (
                    "crash_vehicle",
                    "crash_object",
                    "crash_building",
                    "crash_human",
                    "crash_sidewalk",
                )
            )
            if contact and not collision:
                first_collision = {
                    "t": t,
                    "action": action,
                    "position": np.asarray(env.vehicle.position).tolist(),
                    "view": asdict(view) if controller is not None else None,
                    "flags": {
                        k: bool(info.get(k, False))
                        for k in ("crash_vehicle", "crash_object", "crash_sidewalk")
                    },
                }
            collision |= contact
            safety_cost += float(info.get("cost", 0))
            velocity = np.asarray(env.vehicle.velocity, dtype=float)
            if previous_velocity is not None:
                acceleration = (velocity - previous_velocity) / dt
                if previous_acceleration is not None:
                    peak_jerk = max(
                        peak_jerk, float(np.linalg.norm(acceleration - previous_acceleration) / dt)
                    )
                previous_acceleration = acceleration
            previous_velocity = velocity.copy()
            if term or trunc:
                break
        route = float(info.get("route_completion", 0)) * 100
        return {
            "id": job.key(),
            **asdict(job),
            "map_sha256": fingerprint,
            "success": bool(info.get("arrive_dest", False)) and not collision,
            "arrive_dest": bool(info.get("arrive_dest", False)),
            "crashed": collision,
            "route_pct": route,
            "safety_cost": safety_cost,
            "router_calls": calls,
            "calls_rejected": rejected,
            "road_expiries": expired,
            "lane_changes": changes,
            "planner_elapsed_s": planner_cpu,
            "scheduler_elapsed_s": scheduler_cpu,
            "planner_process_s": planner_process,
            "scheduler_process_s": scheduler_process,
            "control_process_s": control_process,
            "control_elapsed_s": controller_elapsed,
            "obsolete_speed_ticks": obsolete_speed_ticks,
            "peak_motion_jerk": peak_jerk,
            "steps": tick + 1,
            "end_time": (tick + 1) * dt,
            "out_of_road": bool(info.get("out_of_road", False)),
            "first_collision": first_collision,
            "wall_s": time.perf_counter() - wall_start,
            "dt": dt,
            "config": config,
            "resolved_config": resolved_config,
            "observation": "common surrounding-object sensor and road navigation state",
        }
    finally:
        if controller is not None:
            controller.close()
        if reference is not None:
            reference.destroy()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "validation", "test"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/research-roads")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = [("never", 0), ("shield", 0), ("ttc", 0), ("benefit", 0), ("no_latency", 0)]
    candidates += [("periodic", hz) for hz in (0.1, 0.25, 0.5, 1, 2, 10)]
    candidates += [("deadline", b) for b in (0, 0.5, 1)]
    if args.stage == "pilot":
        candidates = [
            ("idm", 0),
            ("expert", 0),
            ("never", 0),
            ("periodic", 1),
            ("benefit", 0),
            ("deadline", 0.5),
        ]
    if args.stage == "test":
        selection = verify("roads", args.output)
        candidates = [
            ("never", 0),
            ("shield", 0),
            ("ttc", 0),
            ("benefit", 0),
            ("no_latency_matched", selection["deadline"]),
            ("periodic", selection["periodic"]),
            ("periodic_gated", 10.0),
            ("deadline", selection["deadline"]),
        ]
    seeds = {"pilot": range(201, 203), "validation": range(201, 209), "test": range(2001, 2041)}[
        args.stage
    ]
    delays = list(DELAYS) if args.stage != "pilot" else ["standard", "long"]
    jobs = [
        Job(env, seed, delay, method, value)
        for env in ("generalization", "safe")
        for seed in seeds
        for delay in delays
        for method, value in candidates
    ]
    if args.stage == "test":
        validation = [
            json.loads(line) for line in (args.output / "validation.jsonl").read_text().splitlines()
        ]
        for environment in ("generalization", "safe"):
            for delay in delays:
                scores = []
                for rate in (0.1, 0.25, 0.5, 1, 2, 10):
                    group = [
                        r
                        for r in validation
                        if r["environment"] == environment
                        and r["delay"] == delay
                        and r["method"] == "periodic"
                        and r["value"] == rate
                    ]
                    scores.append(
                        (
                            (
                                sum(r["success"] for r in group),
                                np.mean([r["route_pct"] for r in group]),
                                -np.mean([r["router_calls"] for r in group]),
                            ),
                            rate,
                        )
                    )
                rate = max(scores)[1]
                if rate != selection["conditioned_periodic"][environment + "|" + delay]:
                    raise ValueError("conditioned polling selection changed after freeze")
                jobs.extend(
                    Job(environment, seed, delay, "periodic_conditioned", rate) for seed in seeds
                )
    if args.stage != "pilot":
        jobs += [
            Job(env, seed, "standard", method)
            for env in ("generalization", "safe")
            for seed in seeds
            for method in ("idm", "expert")
        ]
    target = args.output / f"{args.stage}.jsonl"
    completed = (
        {json.loads(line)["id"] for line in target.read_text().splitlines()}
        if target.exists()
        else set()
    )
    jobs = [j for j in jobs if j.key() not in completed]
    print(f"stage={args.stage}, pending={len(jobs)}, completed={len(completed)}", flush=True)
    with (
        target.open("a") as stream,
        ProcessPoolExecutor(args.workers, initializer=initialize) as pool,
    ):
        futures = {pool.submit(execute, j): j for j in jobs}
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if i % 10 == 0 or i == len(jobs):
                print(
                    f"{args.stage}: {i}/{len(jobs)} {row['environment']} {row['method']} "
                    f"route={row['route_pct']:.1f} crash={row['crashed']} calls={row['router_calls']}",
                    flush=True,
                )
    if args.stage == "validation":
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        selection = {}
        for method in ("periodic", "deadline"):
            scores = []
            for value in sorted({r["value"] for r in rows if r["method"] == method}):
                group = [r for r in rows if r["method"] == method and r["value"] == value]
                score = (
                    sum(r["success"] for r in group),
                    np.mean([r["route_pct"] for r in group]),
                    -np.mean([r["router_calls"] for r in group]),
                )
                scores.append((score, value))
            selection[method] = max(scores)[1]
        selection["validation_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(selection, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
