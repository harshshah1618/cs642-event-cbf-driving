"""Final proposal-aligned physical comparison with frozen source/design locks.

Controlled decisions are state rules, never described as trained VLM answers.
All policies share the same physical environment, PPO bank and metric code.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from event_cbf_driving import DT
from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.eval.runner import load_skill_bank
from event_cbf_driving.router.scripted import ScriptedRouter
from event_cbf_driving.stack.controller import DrivingStack
from event_cbf_driving.stack.maneuver_reserve import ManeuverReserveShield

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/final-integration"
BANK = None
SCENES = ("stall_ahead", "cut_in", "dense")
METHODS = (
    "unshielded_rl",
    "static_cbf",
    "timer_hard_unshielded",
    "uncertainty_wakeup",
    "barrier_replan_local",
    "original_proposal",
    "final_integrated",
    "matched_timer_hard",
    "high_rate_poll",
    "gated_poll",
    "no_reserve",
    "no_blend",
    "speed_delay_margin",
    "critic_margin",
    "immediate_benefit",
)
MODEL_METHODS = ("matched_timer_hard", "uncertainty_wakeup", "final_integrated")
LOCAL_METHODS = ("unshielded_rl", "static_cbf", "barrier_replan_local")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class PromptRule(ScriptedRouter):
    """Current geometry implements the teammate prompt's lane/occupancy rule.

    Ground-truth positions and speeds are available here. This deliberately
    removes perception errors; moving slow leads count as blockers (an extension
    of the original all-parked camera prompt, declared in the protocol).
    """

    name = "controlled_state_rule_not_vlm"

    def decide(self, req):
        ego, others = req.snapshot.ego, req.snapshot.others
        right_ahead = any(0 < o.x - ego.x < 100 and abs(o.y) < 1.85 and o.vx < 15 for o in others)
        left_occupied = any(abs(o.x - ego.x) < 20 and abs(o.y - 4) < 1.85 for o in others)
        if ego.y >= 2:
            return "OVERTAKE_LEFT" if right_ahead else "RETURN_RIGHT"
        return "OVERTAKE_LEFT" if right_ahead and not left_occupied else "CRUISE_RIGHT"


@dataclass(frozen=True)
class Job:
    scene: str
    seed: int
    delay: float
    method: str
    sigma_scale: float = 0.2

    def key(self):
        return object_hash(asdict(self))[:24]


def scene_spec(job, *, camera=False):
    rng = np.random.default_rng(job.seed + 185000)
    params = {"spawn_x": 0.0, "ego_vx": float(rng.uniform(18, 24)), "rgb_camera": camera}
    if job.scene in ("stall_ahead", "cut_in"):
        params.update(
            stalled_x=float(rng.uniform(58, 85)),
            cut_in_trigger_x=92.0 if job.scene == "cut_in" else 10000.0,
            cut_in_gap=float(rng.uniform(18, 30)),
            cut_in_speed=float(rng.uniform(8, 12)),
        )
        name = "two_trap"
    elif job.scene == "dense":
        name = "dense_highway"
        # This scenario owns its own randomized traffic and fixed ego start.
        params = {"rgb_camera": camera}
    else:
        raise ValueError(job.scene)
    return ScenarioConfig(
        name, simulator="metadrive", randomize=True, duration=60, route_length=200, params=params
    )


def configs(job, *, live=False):
    cfg = load_config(ROOT / "configs/metadrive-study.yaml")
    c, s, m = cfg.stack, cfg.shield, job.method
    c.mode = "proposed"
    c.trigger_policy = "deadline"
    c.trigger_latency_bound_s = job.delay
    c.trigger_transition_s, c.trigger_buffer_s = 2.0, 0.5
    c.min_call_ticks, c.shield_hold_ticks = 28, 10
    c.use_blend, c.hard_switch, c.blend_ticks = True, False, 24
    c.pending_margin = s.pending_margin = False
    c.pending_margin_on_all_calls = True
    c.max_pending_s = 45.0 if live else job.delay + 1.0
    c.simulate_deadline_wait = True
    c.record_interactions = True
    c.abort_on_emergency = False
    c.skill_switch_guard = c.rearm_on_skill_change = True
    c.router_async = c.realtime = live
    c.sigma_threshold_scale = job.sigma_scale
    c.route_success_pct = 99.0
    s.gamma_sigma, s.eta_latency, s.vlm_delay_s = 0.0, 1.0, job.delay
    s.cbf_constraint_enabled = s.braking_guard = s.road_boundary_guard = True
    reserve = "fixed_latched"
    if m == "unshielded_rl":
        c.mode = "unshielded_skill"
        c.skill_switch_guard = s.braking_guard = s.cbf_constraint_enabled = False
        reserve = "baseline"
    elif m == "static_cbf":
        c.mode = "static_cbf"
        c.skill_switch_guard = s.braking_guard = False
        reserve = "baseline"
    elif m == "timer_hard_unshielded":
        c.mode, c.fixed_rate_hz = "fixed_rate_vlm", 1.0
        c.hard_switch, c.use_blend = True, False
        c.skill_switch_guard = s.braking_guard = s.cbf_constraint_enabled = False
        reserve = "baseline"
    elif m == "uncertainty_wakeup":
        c.trigger_policy = "uncertainty"
    elif m == "barrier_replan_local":
        c.trigger_policy = "barrier"
    elif m == "original_proposal":
        c.trigger_policy, c.sigma_threshold_scale = "legacy", 1.0
        c.pending_margin = s.pending_margin = True
        s.gamma_sigma = 2.8
        reserve = "baseline"
    elif m == "matched_timer_hard":
        c.mode, c.fixed_rate_hz = "fixed_rate_shielded", 1.0
        c.hard_switch, c.use_blend = True, False
    elif m in ("high_rate_poll", "gated_poll"):
        c.mode, c.always_on_hz = "always_on_shielded", 10.0
        c.polling_usefulness_gate = m == "gated_poll"
    elif m == "no_reserve":
        reserve = "baseline"
    elif m == "no_blend":
        c.hard_switch, c.use_blend = True, False
    elif m == "speed_delay_margin":
        c.pending_margin = s.pending_margin = True
    elif m == "critic_margin":
        s.gamma_sigma = 2.8
    elif m == "immediate_benefit":
        c.trigger_policy = "benefit"
    elif m != "final_integrated":
        raise ValueError(m)
    # Polling phase is randomized once per physical seed. Both polling rates
    # use the same uniform fraction of a period; events are not timer shifted.
    if m in ("timer_hard_unshielded", "matched_timer_hard", "high_rate_poll", "gated_poll"):
        rate = c.always_on_hz if m in ("high_rate_poll", "gated_poll") else c.fixed_rate_hz
        c.polling_phase_s = float(np.random.default_rng(job.seed + 48100).uniform(0, 1 / rate))
    cfg.validate()
    return cfg, reserve


def initialize():
    import torch

    global BANK
    torch.set_num_threads(1)
    BANK, _ = load_skill_bank(ROOT / "checkpoints/metadrive-calibrated")


def trace_metrics(result):
    rows = result.interactions
    if not rows:
        raise ValueError("full physical trace required for metrics")
    achieved = np.asarray([r["plant"]["achieved_acceleration"] for r in rows])
    jerk = np.diff(achieved[:, 1]) / DT
    warm = jerk[9:]
    duration = len(rows) * DT
    stopped_s = sum(r["ego"][2] < 0.5 for r in rows) * DT
    # A sustained stop is not automatically failure: report a separate
    # terminal stall, while retaining the legacy ever-stopped flag too.
    terminal_stall = bool(result.vx < 0.5 and result.route_pct < 99 and not result.crashed)
    count0 = len(rows[0]["obstacles"])
    additions = [r for r in rows if len(r["obstacles"]) > count0]
    return {
        "achieved_jerk_rms": float(np.sqrt(np.mean(jerk**2))) if len(jerk) else 0.0,
        "achieved_jerk_peak": float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
        "achieved_jerk_peak_after_0_2s": float(np.max(np.abs(warm))) if len(warm) else 0.0,
        "stopped_s": stopped_s,
        "terminal_stall": terminal_stall,
        "tick_duration_s": duration,
        "cut_in_observed": bool(additions),
        "cut_in_first_seen_while_pending": bool(additions and additions[0]["router_pending"]),
        "cbf_override_fraction": sum(r["cbf_override"] for r in rows) / len(rows),
        "braking_override_fraction": sum(r["braking_override"] for r in rows) / len(rows),
    }


def execute(job, *, router=None, bank=None, target=None, live=False):
    cfg, reserve_mode = configs(job, live=live)
    effective_delay = 0.0 if job.method == "barrier_replan_local" else job.delay
    router = router or PromptRule(latency_s=effective_delay, seed=job.seed)
    shield = ManeuverReserveShield(cfg.shield, mode=reserve_mode, reserve_m=8.0)
    stack = DrivingStack(bank or BANK, router, shield=shield, config=cfg.stack)
    env = make_env(scene_spec(job).name, scene_spec(job, camera=live))
    original_reset = env.reset
    initial = {}

    def reset(*args, **kwargs):
        value = original_reset(*args, **kwargs)
        snap = env.snapshot()
        initial.update(ego=asdict(snap.ego), others=[asdict(o) for o in snap.others])
        return value

    env.reset = reset
    started = time.monotonic()
    try:
        result = stack.run(env, seed=job.seed)
    finally:
        env.close()
    payload = result.to_dict()
    out = {
        k: v
        for k, v in payload.items()
        if k not in ("interactions", "actions", "xs", "ys", "skills", "calls")
    }
    out.update(
        asdict(job),
        id=job.key(),
        initial=initial,
        wall_s=time.monotonic() - started,
        success=not result.crashed and result.route_pct >= 99,
        actual_model=live,
        decision_source="actual_trained_camera_vlm" if live else "controlled_state_rule",
        expensive_calls=result.router_calls if job.method != "barrier_replan_local" else 0,
        local_replans=result.router_calls if job.method == "barrier_replan_local" else 0,
        **trace_metrics(result),
    )
    active = stack.last_shield
    out["reserve_events"] = active.events
    out["effective_safety"] = {
        "enabled": active.enabled,
        "cbf": active.enabled and active.cbf_constraint_enabled,
        "braking": active.enabled and active.braking_guard,
        "boundary": active.enabled and active.road_boundary_guard,
        "admission": cfg.stack.skill_switch_guard,
        "inflation": active.inflate,
        "gamma_sigma": active.gamma_sigma,
        "pending_margin": active.pending_margin,
        "reserve_mode": active.reserve_mode,
        "base_clearance": active.base_clearance,
    }
    out["resolved_config"] = cfg.to_dict()
    out["calls"] = result.calls
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload, allow_nan=False).encode()
        path = target / f"{job.key()}.json.gz"
        path.write_bytes(gzip.compress(raw, mtime=0))
        out["trace_sha256"] = digest(path)
    return out


def jobs(stage, sigma_scale=0.2):
    if stage == "tune":
        return [
            Job(s, seed, d, "uncertainty_wakeup", scale)
            for s, seed, d, scale in itertools.product(
                SCENES, (8101, 8102), (0.125, 18.0), (0.2, 0.5, 1.0)
            )
        ]
    seeds = (8201,) if stage == "pilot" else tuple(range(9001, 9009))
    delays = (0.125, 18.0) if stage == "pilot" else (0.0, 0.125, 1.0, 18.0)
    return [
        Job(s, seed, d, m, sigma_scale)
        for s, seed, d, m in itertools.product(SCENES, seeds, delays, METHODS)
    ]


def source_hashes():
    paths = list((ROOT / "event_cbf_driving").rglob("*.py")) + [Path(__file__)]
    paths += list((ROOT / "configs").glob("*.yaml"))
    paths += [p for p in (ROOT / "checkpoints/metadrive-calibrated").rglob("*") if p.is_file()]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def verify_sources(hashes):
    for name, sha in hashes.items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"frozen source changed: {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("tune", "pilot", "test"))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    selection = ROOT.parent / "final-study/selection.json"
    scale = json.loads(selection.read_text())["sigma_scale"] if selection.exists() else 0.2
    if a.stage == "test" and not selection.exists():
        raise ValueError("freeze selection before held-out test")
    design = jobs(a.stage, scale)
    target = OUT / a.stage
    target.mkdir(parents=True, exist_ok=True)
    lock = {
        "stage": a.stage,
        "design": [asdict(j) for j in design],
        "sources": source_hashes(),
        "selection": json.loads(selection.read_text()) if selection.exists() else None,
    }
    encoded = json.dumps(lock, sort_keys=True, indent=2)
    path = target / "lock.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("design/source changed: use a fresh versioned output directory")
    path.write_text(encoded)
    sha = hashlib.sha256(encoded.encode()).hexdigest()
    output = target / "episodes.jsonl"
    rows = [json.loads(s) for s in output.read_text().splitlines()] if output.exists() else []
    seen = {r["id"] for r in rows}
    started = time.monotonic()
    with output.open("a") as f, ProcessPoolExecutor(a.workers, initializer=initialize) as pool:
        pending = [
            pool.submit(execute, j, target=target / "traces") for j in design if j.key() not in seen
        ]
        for future in as_completed(pending):
            row = future.result()
            row["source_lock_sha256"] = sha
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            rows.append(row)
            if len(rows) % 20 == 0:
                print(
                    f"{a.stage} {len(rows)}/{len(design)} episodes, {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    assert len(rows) == len(design) and len({r["id"] for r in rows}) == len(design)
    verify_sources(lock["sources"])
    print(f"COMPLETE {a.stage}: {len(rows)}; all source hashes stable", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
