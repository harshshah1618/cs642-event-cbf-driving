"""Exploratory actual-camera CPU pair; measured inference timing is not paired."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.eval.runner import load_skill_bank
from event_cbf_driving.stack.controller import DrivingStack
from event_cbf_driving.stack.maneuver_reserve import ManeuverReserveShield

from trained_vlm_cpu import TrainedVLMCPU
from trained_vlm_drive import ROOT, ActualCPURouter


def run(policy, target, mode, seed, reserve):
    target.mkdir()
    cfg = load_config(ROOT / "configs/metadrive-study.yaml")
    cfg.stack.mode = "fixed_rate_shielded"
    cfg.stack.fixed_rate_hz = 2.0
    cfg.stack.router_async = cfg.stack.realtime = True
    cfg.stack.max_pending_s = 40.0
    cfg.stack.pending_margin = cfg.shield.pending_margin = False
    cfg.stack.record_interactions = True
    cfg.shield.gamma_sigma = 0.0
    cfg.validate()
    bank, checkpoints = load_skill_bank(ROOT / "checkpoints/metadrive-calibrated")
    router = ActualCPURouter(policy, target)
    shield = ManeuverReserveShield(cfg.shield, mode=mode, reserve_m=reserve)
    stack = DrivingStack(bank, router, shield=shield, config=cfg.stack)
    env = make_env(
        "two_trap",
        ScenarioConfig(
            "two_trap",
            simulator="metadrive",
            duration=60,
            randomize=True,
            params={"rgb_camera": True},
        ),
    )
    original_reset = env.reset
    initial = {}

    def reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        from dataclasses import asdict

        snap = env.snapshot()
        initial.update(ego=asdict(snap.ego), others=[asdict(o) for o in snap.others])
        return result

    env.reset = reset
    started = time.monotonic()
    try:
        result = stack.run(env, seed=seed)
    finally:
        env.close()
    active = stack.last_shield
    summary = {
        "kind": "exploratory_actual_model_physical_cpu_pair",
        "actual_model": True,
        "seed": seed,
        "mode": mode,
        "reserve_m": reserve,
        "initial": initial,
        "route_pct": result.route_pct,
        "crashed": result.crashed,
        "freeze": result.freeze,
        "success": not result.crashed and result.route_pct >= 99,
        "end_time": result.end_time,
        "wall_s": time.monotonic() - started,
        "predictions": len(router.predictions),
        "fallbacks": result.router_fallbacks,
        "calls_applied": result.calls_applied,
        "calls_rejected": result.calls_rejected,
        "latencies": [r["latency_s"] for r in router.predictions],
        "letters": [r["letter"] for r in router.predictions],
        "reserve_events": active.events,
        "reserve_peak": active.reserve_peak,
        "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "adapter_audit": policy.audit,
        "checkpoints": checkpoints,
        "limitations": [
            "One development failure seed; not a representative VLM driving benchmark",
            "Same physical initial state; actual measured CPU/model/render timing varies across arms",
            "No conclusion about teammate GPU speed or population safety",
            "Asynchronous wall-clock timing and in-flight calls retained in full trace",
        ],
    }
    (target / "episode.json").write_text(json.dumps(result.to_dict(), allow_nan=False))
    (target / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "mode",
                    "route_pct",
                    "crashed",
                    "freeze",
                    "success",
                    "letters",
                    "latencies",
                )
            }
        ),
        flush=True,
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=6301)
    p.add_argument("--reserve", type=float, default=8.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT / "event_cbf_driving").rglob("*.py")) + [
        Path(__file__),
        ROOT / "scripts/trained_vlm_cpu.py",
        ROOT / "scripts/trained_vlm_drive.py",
        ROOT / "configs/metadrive-study.yaml",
        ROOT.parent / "component-study/teammate-source/vlm-router/prompts.py",
        ROOT.parent / "vlm-study/model/adapter/adapter_model.safetensors",
    ]
    hashes = {str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    lock = {
        "sources": hashes,
        "seed": a.seed,
        "reserve_m": a.reserve,
        "base": str(a.base),
        "base_revision": a.base.name,
        "frozen_before_model_load": True,
    }
    (a.output / "lock.json").write_text(json.dumps(lock, indent=2))
    policy = TrainedVLMCPU(
        a.base,
        ROOT.parent / "vlm-study/model/adapter",
        ROOT.parent / "component-study/teammate-source/vlm-router/prompts.py",
        threads=3,
    )
    results = [
        run(policy, a.output / mode, mode, a.seed, a.reserve)
        for mode in ("baseline", "fixed_latched")
    ]
    assert results[0]["initial"] == results[1]["initial"], "physical initial states differ"
    for filename, sha in hashes.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == sha, filename
    (a.output / "pair.json").write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
