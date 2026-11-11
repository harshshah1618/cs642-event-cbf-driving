"""Frozen-controller scheduling study with validation-only selection and raw rows."""

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
from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.eval.runner import load_skill_bank, run_episode
from event_cbf_driving.router.scripted import ScriptedRouter
from event_cbf_driving.skills.ppo import SkillMeta

ROOT = Path(__file__).resolve().parents[1]
DELAYS = {
    "fast": (0.05, 0.0),
    "nominal": (0.3, 0.0),
    "slow": (1.0, 0.0),
    "long": (2.0, 0.0),
    "jitter": (0.8, 0.5),
}
BANK = None


class PDPolicy:
    def __init__(self, name):
        self.name = name
        y = 4.0 if name == "OVERTAKE_LEFT" else 0.0
        self.meta = SkillMeta(name, int(y > 0), y, 1e6, 0.0, 0.0, 1.0)

    def act(self, y, vy, vx, deterministic=True):
        return (
            float(np.clip(1.2 * (20 - vx), -7.5, 3.5)),
            float(np.clip(1.6 * (self.meta.y_target - y) - 2.5 * vy, -5, 5)),
            0.0,
        )


def initialize():
    import torch

    global BANK
    torch.set_num_threads(1)
    BANK, _ = load_skill_bank(ROOT / "checkpoints/metadrive-calibrated")


@dataclass(frozen=True)
class Job:
    scenario: str
    seed: int
    delay: str
    method: str
    value: float = 0.0
    controller: str = "ppo"

    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]


def execute(job):
    cfg = load_config(ROOT / "configs/metadrive-study.yaml")
    cfg.stack.record_interactions = False
    cfg.stack.pending_margin = cfg.shield.pending_margin = False
    cfg.stack.simulate_deadline_wait = True
    latency, jitter = DELAYS[job.delay]
    cfg.stack.max_pending_s = latency + jitter + 1
    cfg.stack.trigger_latency_bound_s = latency + jitter
    cfg.stack.trigger_buffer_s = 0.5
    if job.method in ("periodic", "periodic_conditioned"):
        mode = "fixed_rate_shielded"
        cfg.stack.fixed_rate_hz = job.value
    else:
        mode = "proposed"
        cfg.stack.trigger_policy = job.method
        if job.method == "deadline":
            cfg.stack.trigger_buffer_s = job.value
    bank = (
        BANK
        if job.controller == "ppo"
        else {name: PDPolicy(name) for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")}
    )
    router = ScriptedRouter(latency_s=latency, latency_jitter_s=jitter, seed=job.seed)
    start = time.perf_counter()
    result = run_episode(
        mode,
        ScenarioConfig(job.scenario, simulator="metadrive", duration=20, randomize=True),
        job.seed,
        bank,
        router,
        stack_config=cfg.stack,
        shield_config=cfg.shield,
    )
    d = result.to_dict()
    keep = (
        "route_pct",
        "crashed",
        "freeze",
        "router_calls",
        "end_time",
        "end_reason",
        "infeasible_ticks",
        "override_fraction",
        "plant_peak_lateral_jerk",
        "peak_lateral_jerk",
        "calls_applied",
        "calls_rejected",
        "pending_survival",
    )
    return {
        "id": job.key(),
        **asdict(job),
        **{k: d.get(k) for k in keep},
        "success": not result.crashed and result.route_pct >= 99,
        "wall_s": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "validation", "test", "pd"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/research-isolation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = [("legacy", 0), ("ttc", 0), ("benefit", 0), ("no_latency", 0)]
    candidates += [("periodic", hz) for hz in (0.1, 0.25, 0.5, 1, 2, 10)]
    candidates += [("deadline", b) for b in (0, 0.5, 1)]
    if args.stage in ("test", "pd"):
        selected = json.loads((args.output / "selection.json").read_text())
        candidates = [
            ("legacy", 0),
            ("ttc", 0),
            ("benefit", 0),
            ("no_latency", 0),
            ("periodic", selected["periodic"]),
            ("deadline", selected["deadline"]),
        ]
    seeds = {
        "pilot": range(101, 103),
        "validation": range(101, 109),
        "test": range(1001, 1041),
        "pd": range(1001, 1011),
    }[args.stage]
    delays = list(DELAYS) if args.stage != "pilot" else ["nominal", "long"]
    jobs = [
        Job(s, seed, delay, method, value, "pd" if args.stage == "pd" else "ppo")
        for s in ("two_trap", "merge", "dense_highway")
        for seed in seeds
        for delay in delays
        for method, value in candidates
    ]
    if args.stage in ("test", "pd"):
        validation = [json.loads(line) for line in (args.output / "validation.jsonl").read_text().splitlines()]
        for scenario in ("two_trap", "merge", "dense_highway"):
            for delay in delays:
                scores = []
                for rate in (0.1, 0.25, 0.5, 1, 2, 10):
                    group = [r for r in validation if r["scenario"] == scenario and r["delay"] == delay
                             and r["method"] == "periodic" and r["value"] == rate]
                    scores.append(((sum(r["success"] for r in group),
                                    np.mean([r["route_pct"] for r in group]),
                                    -np.mean([r["router_calls"] for r in group])), rate))
                rate = max(scores)[1]
                jobs.extend(Job(scenario, seed, delay, "periodic_conditioned", rate,
                                "pd" if args.stage == "pd" else "ppo") for seed in seeds)
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
            if i % 20 == 0 or i == len(jobs):
                print(
                    f"{args.stage}: {i}/{len(jobs)}; {row['scenario']} {row['method']} "
                    f"route={row['route_pct']:.1f} calls={row['router_calls']}",
                    flush=True,
                )
    if args.stage == "validation":
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        selection = {}
        for method in ("periodic", "deadline"):
            values = sorted({r["value"] for r in rows if r["method"] == method})
            scores = []
            for value in values:
                group = [r for r in rows if r["method"] == method and r["value"] == value]
                score = (
                    sum(r["success"] for r in group),
                    np.mean([r["route_pct"] for r in group]),
                    -np.mean([r["router_calls"] for r in group]),
                )
                scores.append((score, value))
            selection[method] = max(scores)[1]
        selection["protocol"] = "research/PROTOCOL.md; selection uses validation only"
        selection["validation_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(selection, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
