"""Paired physical factorial of controller, CBF, brake backup and lane guard.

Routers here are explicit ground-truth rules and synthetic faults, NOT VLMs.
The separately implemented VLM arm requires the actual teammate endpoint/model.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from event_cbf_driving import DT
from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.eval.runner import load_skill_bank
from event_cbf_driving.router.scripted import ScriptedRouter
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.skills.classical import PDLanePolicy
from event_cbf_driving.stack.controller import DrivingStack

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/components"
SCENES = ("two_trap", "dense_highway", "side_occupied", "lead_brakes")
ROUTERS = ("state_rule", "front_blind_rule", "copy_current")
CONTROLLERS = ("ppo", "pd")
DELAYS = (0.125, 1.0)
TEST_SEEDS = tuple(range(5001, 5013))
PILOT_SEEDS = (4901,)
BANK = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class RuleRouter(ScriptedRouter):
    """Teammate-style lane rule, with separately named controlled faults."""

    def __init__(self, policy, latency_s, seed):
        super().__init__(latency_s=latency_s, seed=seed)
        if policy not in ROUTERS:
            raise ValueError("unknown experimental router")
        self.policy = policy
        self.name = policy

    def decide(self, req):
        if self.policy == "copy_current":
            return req.current_skill
        ego = req.snapshot.ego
        others = req.snapshot.others
        if self.policy == "front_blind_rule":
            # Controlled information removal, not an estimate of VLM errors.
            others = [o for o in others if o.x - ego.x > 8.0]
        right_ahead = any(
            0 < o.x - ego.x < 80.0 and abs(o.y) < 1.85 and o.vx < 15.0 for o in others
        )
        left_occupied = any(abs(o.x - ego.x) < 20.0 and abs(o.y - 4.0) < 1.85 for o in others)
        if ego.y >= 2.0:
            return "OVERTAKE_LEFT" if right_ahead else "RETURN_RIGHT"
        return "OVERTAKE_LEFT" if right_ahead and not left_occupied else "CRUISE_RIGHT"


@dataclass(frozen=True)
class Job:
    scene: str
    seed: int
    latency_s: float
    router: str
    controller: str
    cbf: bool
    braking: bool
    admission: bool

    def key(self):
        return object_hash(asdict(self))[:24]


def jobs(stage):
    seeds = PILOT_SEEDS if stage == "pilot" else TEST_SEEDS
    delays = DELAYS[:1] if stage == "pilot" else DELAYS
    return [
        Job(*items)
        for items in itertools.product(
            SCENES,
            seeds,
            delays,
            ROUTERS,
            CONTROLLERS,
            (False, True),
            (False, True),
            (False, True),
        )
    ]


def scene_spec(job):
    if job.scene in ("two_trap", "dense_highway"):
        return ScenarioConfig(job.scene, simulator="metadrive", duration=20, randomize=True)
    rng = np.random.default_rng(job.seed + 87000)
    if job.scene == "side_occupied":
        traffic = [
            {"x": float(rng.uniform(35, 45)), "y": 0.0, "vx": float(rng.uniform(8, 11))},
            {"x": float(rng.uniform(-2, 2)), "y": 4.0, "vx": 20.0},
        ]
        events = []
    elif job.scene == "lead_brakes":
        traffic = [{"x": float(rng.uniform(32, 42)), "y": 0.0, "vx": 20.0}]
        events = [{"at": float(rng.uniform(0.8, 1.2)), "vehicle_index": 0, "target_speed": 0.0}]
    else:
        raise ValueError("unknown scene")
    return ScenarioConfig(
        "dense_highway",
        simulator="metadrive",
        duration=20,
        route_length=200,
        params={"initial_traffic": traffic, "traffic_events": events},
    )


def configs(job):
    cfg = load_config(ROOT / "configs/metadrive-study.yaml")
    cfg.stack.mode = "fixed_rate_shielded"
    cfg.stack.fixed_rate_hz = 2.0
    cfg.stack.polling_phase_s = float(np.random.default_rng(job.seed + 52000).uniform(0, 0.5))
    cfg.stack.polling_usefulness_gate = False
    cfg.stack.use_blend = True
    cfg.stack.blend_ticks = 24
    cfg.stack.router_async = False
    cfg.stack.realtime = False
    cfg.stack.pending_margin = cfg.shield.pending_margin = False
    cfg.stack.simulate_deadline_wait = True
    cfg.stack.max_pending_s = job.latency_s + 0.5
    cfg.stack.record_interactions = True
    cfg.stack.skill_switch_guard = job.admission
    cfg.shield.enabled = True
    cfg.shield.cbf_constraint_enabled = job.cbf
    cfg.shield.braking_guard = job.braking
    cfg.shield.gamma_sigma = 0.0
    cfg.shield.road_boundary_guard = True
    cfg.validate()
    return cfg


def initialize():
    import torch

    global BANK
    torch.set_num_threads(1)
    BANK, _ = load_skill_bank(ROOT / "checkpoints/metadrive-calibrated")


def reduce_trace(result):
    rows = result.interactions
    if not rows:
        raise ValueError("this study requires achieved-motion provenance")
    achieved = np.asarray([r["plant"]["achieved_acceleration"] for r in rows])
    jerk = np.diff(achieved[:, 1]) / DT
    warm = jerk[9:]  # Exclude first 0.2 s only for the separately named warm metric.
    n = len(rows)
    fields = {
        "cbf_override_fraction": sum(r["cbf_override"] for r in rows) / n,
        "braking_override_fraction": sum(r["braking_override"] for r in rows) / n,
        "boundary_override_fraction": sum(r["lateral_override"] for r in rows) / n,
        "monitor_emergency_ticks": sum(r["monitor"] == "emergency" for r in rows),
        "monitor_changed_action_ticks": sum(
            not np.allclose(r["applied_action"], r["shield_action"], atol=1e-6, rtol=0)
            for r in rows
        ),
        "admission_rejections": sum(
            c.get("error") == "skill switch blocked by fresh target-lane gap" for c in result.calls
        ),
        "achieved_lateral_jerk_rms": float(np.sqrt(np.mean(jerk**2))) if len(jerk) else None,
        "achieved_lateral_jerk_peak": float(np.max(np.abs(jerk))) if len(jerk) else None,
        "achieved_lateral_jerk_peak_after_0_2s": float(np.max(np.abs(warm))) if len(warm) else None,
        "ticks": n,
    }
    return fields


def execute(job, save_trace=False):
    cfg = configs(job)
    bank = (
        BANK
        if job.controller == "ppo"
        else {
            name: PDLanePolicy(name) for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")
        }
    )
    stack = DrivingStack(
        bank,
        RuleRouter(job.router, job.latency_s, job.seed),
        shield=InflatedCBFShield(config=cfg.shield),
        config=cfg.stack,
    )
    env = make_env("two_trap" if job.scene == "two_trap" else "dense_highway", scene_spec(job))
    initial = {}
    original_reset = env.reset

    def reset(*args, **kwargs):
        value = original_reset(*args, **kwargs)
        snap = env.snapshot()
        initial.update(
            ego=asdict(snap.ego),
            others=[asdict(v) for v in snap.others],
            route_length=snap.route_length,
        )
        return value

    env.reset = reset
    started = time.perf_counter()
    try:
        result = stack.run(env, seed=job.seed)
    finally:
        env.close()
    fields = reduce_trace(result)
    active = stack.last_shield
    assert active.enabled and active.cbf_constraint_enabled == job.cbf
    assert active.braking_guard == job.braking and active.road_boundary_guard
    assert active.gamma_sigma == 0 and not active.pending_margin
    assert stack.last_monitor is not None and not stack.last_monitor.latch_on_violation
    if not job.cbf:
        assert fields["cbf_override_fraction"] == 0
    if not job.braking:
        assert fields["braking_override_fraction"] == 0
    if not job.admission:
        assert fields["admission_rejections"] == 0
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
                "avg_speed",
                "router_calls",
                "calls_applied",
                "calls_rejected",
                "override_fraction",
                "infeasible_ticks",
                "braking_infeasible_ticks",
                "acceleration_tracking_rmse",
                "plant_peak_lateral_jerk",
                "peak_lateral_jerk",
                "router_live_calls",
                "router_fallbacks",
            )
        },
        **fields,
        "success": not result.crashed and result.route_pct >= 99,
        "initial_state_sha256": object_hash(initial),
        "initial_state": initial,
        "scene_config": asdict(scene_spec(job)),
        "wall_s": time.perf_counter() - started,
    }
    if save_trace:
        trace_dir = OUT / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{job.key()}.json").write_text(json.dumps(result.to_dict(), allow_nan=False))
    return output


def source_hashes():
    paths = sorted((ROOT / "event_cbf_driving").rglob("*.py"))
    paths += sorted((ROOT / "configs").glob("*.yaml"))
    paths += sorted((ROOT / "checkpoints/metadrive-calibrated").glob("*"))
    paths += [Path(__file__), ROOT.parent / "component-study/PROTOCOL.md"]
    return {str(p.relative_to(ROOT.parent)): digest(p) for p in paths if p.is_file()}


def read_rows(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    actual = {r["id"]: r for r in rows}
    design = {j.key(): asdict(j) for j in expected}
    if len(actual) != len(rows) or not set(actual) <= set(design):
        raise ValueError("duplicate or unexpected experiment cells")
    for key, row in actual.items():
        if {name: row[name] for name in design[key]} != design[key]:
            raise ValueError("row metadata differs from design")
    return rows


def verify():
    lock = json.loads((OUT / "lock.json").read_text())
    if lock["sources"] != source_hashes() or lock["jobs_sha256"] != object_hash(
        [asdict(j) for j in jobs("test")]
    ):
        raise ValueError("frozen source, checkpoint, protocol or job design changed")
    if lock["pilot_sha256"] != digest(OUT / "pilot.jsonl"):
        raise ValueError("pilot changed after freeze")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "freeze", "test", "verify"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze":
        if (OUT / "lock.json").exists():
            raise ValueError("refusing to overwrite experiment freeze")
        pilot = read_rows(OUT / "pilot.jsonl", jobs("pilot"))
        if len(pilot) != len(jobs("pilot")):
            raise ValueError("pilot incomplete")
        lock = {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "sources": source_hashes(),
            "jobs_sha256": object_hash([asdict(j) for j in jobs("test")]),
            "pilot_sha256": digest(OUT / "pilot.jsonl"),
            "test_rows": len(jobs("test")),
            "python": platform.python_version(),
            "versions": {p: version(p) for p in ("numpy", "torch", "metadrive-simulator")},
            "scope": "Rule routers and synthetic faults. No trained VLM or independent PPO training seeds.",
        }
        (OUT / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
        print(json.dumps(lock, indent=2))
        return
    if args.stage in ("test", "verify"):
        verify()
    if args.stage == "verify":
        rows = read_rows(OUT / "test.jsonl", jobs("test"))
        if len(rows) != len(jobs("test")):
            raise ValueError("test incomplete")
        print(f"Verified {len(rows)} exact frozen test cells")
        return
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in 1..8")
    design = jobs(args.stage)
    target = OUT / f"{args.stage}.jsonl"
    existing = read_rows(target, design)
    completed = {r["id"] for r in existing}
    pending = [j for j in design if j.key() not in completed]
    print(f"{args.stage}: {len(pending)} pending, {len(completed)} completed", flush=True)
    with (
        target.open("a") as stream,
        ProcessPoolExecutor(args.workers, initializer=initialize) as pool,
    ):
        futures = [pool.submit(execute, j) for j in pending]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if index % 48 == 0 or index == len(pending):
                print(f"{args.stage}: {index}/{len(pending)}", flush=True)
    if args.stage == "test":
        verify()
    print("Complete", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
