"""Exploratory distance-aware usefulness revision, independently replicated.

No original experiment source is modified. Formula fixed before validation;
test seeds5001-5040 are disjoint from every primary experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from event_cbf_driving.stack import controller
from event_cbf_driving.stack.scheduling import LaneOpportunity

import research_isolation as base
from research_freeze import digest, verify

ROOT = base.ROOT
OUT = ROOT / "runs/research-spatial"
ORIGINAL = controller.lane_opportunity
LATENCY = 0.0
CONSTANTS = dict(desired_speed=20.0, transition_s=2.0, terminal_headway_s=2.0,
                 clearance_m=8.0, margin_m=1.0, braking_mps2=3.0,
                 min_speed_benefit=1.0, slow_speed=15.0, lookahead_m=200.0,
                 lane_tolerance_m=1.85, settled_y_m=0.8, settled_vy_mps=0.8)


def spatial_opportunity(snapshot, skill, *, transition_s=2.0, clearance=8.0,
                        prediction_margin=1.0):
    """Terminal-headway and stopping-distance achievable speed proxies.

    Fresh gap admission is identical to original usefulness gate; the existing
    controller independently rechecks lane admission when a delayed reply arrives.
    Forecasts use present observed position/velocity only, not scenario futures.
    """
    ego = snapshot.ego
    target = 4.0 if skill == "OVERTAKE_LEFT" else 0.0
    alternative = 0.0 if target == 4.0 else 4.0
    horizon = LATENCY + transition_s
    scores = {target: CONSTANTS["desired_speed"], alternative: CONSTANTS["desired_speed"]}
    threat, valid = False, True
    constraint, open_for = math.inf, math.inf
    for other in snapshot.others:
        dx = other.x - ego.x
        extent = (ego.length + other.length) / 2
        for lane in (target, alternative):
            if abs(other.y - lane) >= CONSTANTS["lane_tolerance_m"]:
                continue
            available = abs(dx) - extent - clearance - prediction_margin
            if lane == alternative:
                valid = valid and available > 0
                closing = (ego.vx - other.vx) if dx > 0 else (other.vx - ego.vx)
                if closing > 0:
                    open_for = min(open_for, max(0, available) / closing)
            if dx <= 0:
                continue
            speed = max(other.vx, 0)
            capacity = max(0, available)
            progress = (capacity + speed * horizon) / (horizon + CONSTANTS["terminal_headway_s"])
            stopping = math.sqrt(speed**2 + 2 * CONSTANTS["braking_mps2"] * capacity)
            scores[lane] = min(scores[lane], progress, stopping)
            if lane == target and dx <= CONSTANTS["lookahead_m"] and other.vx < CONSTANTS["slow_speed"]:
                threat = True
                constraint = min(constraint, capacity / max(ego.vx - other.vx, 0.1))
    settled = abs(ego.y - target) < CONSTANTS["settled_y_m"] and abs(ego.vy) < CONSTANTS["settled_vy_mps"]
    useful = settled and valid and threat and scores[alternative] > scores[target] + CONSTANTS["min_speed_benefit"]
    return LaneOpportunity(scores[target], scores[alternative], constraint, open_for, transition_s, useful)


def initialize():
    base.initialize()
    controller.lane_opportunity = ORIGINAL


def execute(job):
    global LATENCY
    controller.lane_opportunity = ORIGINAL
    try:
        if job.method == "spatial_gate":
            latency, jitter = base.DELAYS[job.delay]
            LATENCY = latency + jitter
            controller.lane_opportunity = spatial_opportunity
            row = base.execute(replace(job, method="benefit", value=0))
        else:
            row = base.execute(job)
        return {**row, **asdict(job), "id": job.key()}
    finally:
        controller.lane_opportunity = ORIGINAL


def jobs(stage):
    methods = [("spatial_gate", 0)] if stage == "validation" else [
        ("spatial_gate", 0), ("periodic", 0.1), ("periodic_gated", 10.0), ("ttc", 0)]
    seeds = range(101, 109) if stage == "validation" else range(5001, 5041)
    return [base.Job(scene, seed, delay, method, value)
            for scene in ("two_trap", "merge", "dense_highway")
            for seed in seeds for delay in base.DELAYS for method, value in methods]


def checked_rows(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {j.key() for j in expected}:
        raise ValueError("incomplete/duplicate/unexpected experiment rows")
    for job in expected:
        if any(by_id[job.key()][k] != v for k, v in asdict(job).items()):
            raise ValueError("job metadata mismatch")
    return rows


def summarize(rows):
    return {m: {"n": len(g), **{k: float(np.mean([r[k] for r in g]))
                                for k in ("success", "route_pct", "crashed", "router_calls", "end_time")}}
            for m in sorted({r["method"] for r in rows})
            if (g := [r for r in rows if r["method"] == m])}


def analyze():
    rows = checked_rows(OUT / "test.jsonl", jobs("test"))
    result = {"summary": summarize(rows), "paired": {}, "scope": "Exploratory revision, fresh independent scene seeds; no novelty claim."}
    def key(r):
        return (r["scenario"], r["seed"], r["delay"])
    spatial = {key(r): r for r in rows if r["method"] == "spatial_gate"}
    for method in ("periodic", "periodic_gated", "ttc"):
        other = {key(r): r for r in rows if r["method"] == method}
        comparison = {}
        for metric in ("success", "route_pct", "crashed", "router_calls", "end_time"):
            deltas = [np.mean([float(r[metric]) - float(other[k][metric])
                               for k, r in spatial.items() if r["seed"] == seed])
                      for seed in range(5001, 5041)]
            a = np.asarray(deltas)
            rng = np.random.default_rng(642)
            samples = a[rng.integers(0, len(a), (10000, len(a)))].mean(axis=1)
            comparison[metric] = {"difference": float(a.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}
        result["paired"][method] = comparison
    result["by_scenario"] = {s: summarize([r for r in rows if r["scenario"] == s]) for s in ("two_trap", "merge", "dense_highway")}
    result["by_delay"] = {d: summarize([r for r in rows if r["delay"] == d]) for d in base.DELAYS}
    result["test_sha256"] = digest(OUT / "test.jsonl")
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "summary.md").write_text("# Fresh spatial-gate replication\n\nExploratory revision; not a replacement for the negative primary study.\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "freeze", "test", "analyze"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    original_lock = ROOT / "runs/research-isolation/locked-selection.json"
    verify("isolation")
    if args.stage == "analyze":
        analyze()
        return
    if args.stage == "freeze":
        rows = checked_rows(OUT / "validation.jsonl", jobs("validation"))
        lock = {"script_sha256": digest(source), "original_lock_sha256": digest(original_lock),
                "constants": CONSTANTS, "validation_sha256": digest(OUT / "validation.jsonl"),
                "validation_summary": summarize(rows), "test_seeds": [5001, 5040],
                "test_methods": ["spatial_gate", "periodic", "periodic_gated", "ttc"],
                "formula": "min(20,(gap-clearance-margin+v_lead*(latency+transition))/(latency+transition+headway),sqrt(v_lead^2+2*b*available_gap)); difference>1m/s",
                "seed_clusters": "resample shared seed jointly across scenarios and delays",
                "interpretation": "Independent exploratory revision after negative primary study; no formal noninferiority or novelty claim"}
        with (OUT / "locked-design.json").open("x") as out:
            json.dump(lock, out, indent=2)
        (OUT / "source.py").write_bytes(source.read_bytes())
        (OUT / "original-locked-selection.json").write_bytes(original_lock.read_bytes())
        print(json.dumps(lock, indent=2), flush=True)
        return
    if args.stage == "test":
        lock = json.loads((OUT / "locked-design.json").read_text())
        if lock["script_sha256"] != digest(source) or lock["original_lock_sha256"] != digest(original_lock):
            raise ValueError("frozen source changed")
        if lock["constants"] != CONSTANTS or lock["validation_sha256"] != digest(OUT / "validation.jsonl"):
            raise ValueError("frozen design/validation changed")
    target = OUT / f"{args.stage}.jsonl"
    complete = {json.loads(line)["id"] for line in target.read_text().splitlines()} if target.exists() else set()
    pending = [j for j in jobs(args.stage) if j.key() not in complete]
    print(f"{args.stage}: pending={len(pending)}", flush=True)
    with target.open("a") as stream, ProcessPoolExecutor(args.workers, initializer=initialize) as pool:
        futures = [pool.submit(execute, j) for j in pending]
        for i, future in enumerate(as_completed(futures), 1):
            stream.write(json.dumps(future.result(), allow_nan=False) + "\n")
            stream.flush()
            if i % 40 == 0 or i == len(pending):
                print(f"{args.stage}: {i}/{len(pending)}", flush=True)
    rows = checked_rows(target, jobs(args.stage))
    print(json.dumps(summarize(rows), indent=2), flush=True)
    if args.stage == "test":
        analyze()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
