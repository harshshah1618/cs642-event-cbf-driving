"""Locked, random-clock-phase robustness test; no retuning of deadline policy.

Polling phase is sampled once per scene, independently of all observed traffic.
The same phase *fraction* is shared across rates; phase seconds = fraction/rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import research_isolation as base
from research_freeze import verify

FRACTION = 0.0
ORIGINAL_RUN = base.run_episode


def phase_run(*args, **kwargs):
    cfg = kwargs["stack_config"]
    if str(args[0]) == "fixed_rate_shielded":
        cfg.polling_phase_s = FRACTION / cfg.fixed_rate_hz
    return ORIGINAL_RUN(*args, **kwargs)


def initialize():
    base.initialize()
    base.run_episode = phase_run


def execute(job):
    global FRACTION
    FRACTION = float(np.random.default_rng(job.seed + 72001).uniform())
    return {
        **base.execute(job),
        "polling_phase_fraction": FRACTION,
        "polling_phase_s": FRACTION / job.value if job.method.startswith("periodic") else None,
    }


def score(group):
    return (
        sum(r["success"] for r in group),
        np.mean([r["route_pct"] for r in group]),
        -np.mean([r["router_calls"] for r in group]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["validation", "test"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, default=base.ROOT / "runs/research-phase")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    isolation_lock_path = base.ROOT / "runs/research-isolation/locked-selection.json"
    original_selection = json.loads(isolation_lock_path.read_text())
    delays = ["nominal", "long", "jitter"]
    candidates = [("periodic", r) for r in (0.1, 0.25, 0.5, 1, 2, 10)]
    seeds = range(111, 115)
    if args.stage == "test":
        selected = verify("phase", args.output)
        if (
            selected["isolation_lock_sha256"]
            != hashlib.sha256(isolation_lock_path.read_bytes()).hexdigest()
        ):
            raise ValueError("upstream isolation selection changed after phase freeze")
        seeds = range(1101, 1141)
        candidates = [
            ("legacy", 0),
            ("ttc", 0),
            ("benefit", 0),
            ("no_latency_matched", selected["deadline_buffer_s"]),
            ("deadline", selected["deadline_buffer_s"]),
            ("periodic", selected["periodic"]),
            ("periodic_gated", 10.0),
        ]
    jobs = [
        base.Job(scene, seed, delay, method, value)
        for scene in ("two_trap", "merge", "dense_highway")
        for seed in seeds
        for delay in delays
        for method, value in candidates
    ]
    if args.stage == "test":
        validation = [
            json.loads(line) for line in (args.output / "validation.jsonl").read_text().splitlines()
        ]
        for scene in ("two_trap", "merge", "dense_highway"):
            for delay in delays:
                values = [
                    (
                        score(
                            [
                                r
                                for r in validation
                                if r["scenario"] == scene
                                and r["delay"] == delay
                                and r["value"] == rate
                            ]
                        ),
                        rate,
                    )
                    for rate in (0.1, 0.25, 0.5, 1, 2, 10)
                ]
                if max(values)[1] != selected["conditioned_periodic"][scene + "|" + delay]:
                    raise ValueError("conditioned polling selection changed after freeze")
                jobs.extend(
                    base.Job(scene, seed, delay, "periodic_conditioned", max(values)[1])
                    for seed in seeds
                )
    target = args.output / f"{args.stage}.jsonl"
    completed = (
        {json.loads(line)["id"] for line in target.read_text().splitlines()}
        if target.exists()
        else set()
    )
    jobs = [j for j in jobs if j.key() not in completed]
    print(f"phase {args.stage}: pending={len(jobs)}", flush=True)
    with (
        target.open("a") as stream,
        ProcessPoolExecutor(args.workers, initializer=initialize) as pool,
    ):
        futures = [pool.submit(execute, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if i % 30 == 0 or i == len(jobs):
                print(f"phase {args.stage}: {i}/{len(jobs)}", flush=True)
    if args.stage == "validation":
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        scores = [
            (score([r for r in rows if r["value"] == rate]), rate)
            for rate in (0.1, 0.25, 0.5, 1, 2, 10)
        ]
        selected = {
            "periodic": max(scores)[1],
            "validation_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "deadline_buffer_s": original_selection["deadline"],
            "isolation_lock_sha256": hashlib.sha256(isolation_lock_path.read_bytes()).hexdigest(),
        }
        (args.output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
        print(selected, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
