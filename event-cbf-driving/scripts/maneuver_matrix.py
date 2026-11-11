"""Frozen condition-grid runner for the optional maneuver-reserve study."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from maneuver_probe import ROOT, Job, execute, initialize


def design(stage):
    jobs = []
    if stage == "development":
        conditions = itertools.product(
            ("ppo", "pd"),
            (15.0, 25.0, 40.0),
            ((14.0, 45.0), (26.0, 90.0)),
            (0.125, 12.0, 18.0),
        )
        policies = (
            [("baseline", 8.0, 1.0)]
            + [("fixed_latched", r, 1.0) for r in (2.0, 4.0, 8.0, 12.0, 20.0)]
            + [("geometric_latched", 8.0, 1.0)]
        )
        seed_base = 7200
    else:
        # Freeze before seeing any held-out outcomes. Fixed reserve is selected on development.
        conditions = itertools.product(
            ("ppo", "pd"),
            (18.0, 30.0, 40.0),
            ((18.0, 55.0), (24.0, 80.0)),
            (0.125, 6.0, 18.0, 30.0),
        )
        policies = [
            ("baseline", 8.0, 1.0),
            ("fixed_latched", 8.0, 1.0),
            ("geometric_latched", 8.0, 1.0),
            ("static", 8.0, 1.0),
        ]
        seed_base = 7400
    for i, (controller, steering, (speed, gap), delay) in enumerate(conditions):
        for mode, reserve, scale in policies:
            jobs.append(
                Job(seed_base + i, delay, mode, reserve, controller, speed, gap, steering, scale)
            )
    return jobs


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["reserve"])].append(r)
    return {
        f"{m}:{v:g}": {
            "n": len(rs),
            "success": sum(r["success"] for r in rs),
            "crashes": sum(r["crashed"] for r in rs),
            "freezes": sum(r["freeze"] for r in rs),
            "mean_route_pct": sum(r["route_pct"] for r in rs) / len(rs),
            "mean_end_time": sum(r["end_time"] for r in rs) / len(rs),
        }
        for (m, v), rs in sorted(groups.items())
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("development", "heldout"))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    target = ROOT / "runs/maneuver-matrix" / a.stage
    target.mkdir(parents=True, exist_ok=True)
    jobs = design(a.stage)
    sources = (
        list((ROOT / "event_cbf_driving").rglob("*.py"))
        + [
            Path(__file__),
            ROOT / "scripts/maneuver_probe.py",
            ROOT / "configs/metadrive-study.yaml",
        ]
        + list((ROOT / "checkpoints/metadrive-calibrated").rglob("*"))
    )
    hashes = {
        str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest()
        for s in sorted(sources)
        if s.is_file()
    }
    lock = {"stage": a.stage, "design": [asdict(j) for j in jobs], "sources": hashes}
    lock_json = json.dumps(lock, sort_keys=True, indent=2)
    lock_sha = hashlib.sha256(lock_json.encode()).hexdigest()
    lock_path = target / "lock.json"
    if lock_path.exists() and lock_path.read_text() != lock_json:
        raise ValueError("frozen source/design changed; choose a fresh study version")
    lock_path.write_text(lock_json)
    output = target / "episodes.jsonl"
    rows = [json.loads(s) for s in output.read_text().splitlines()] if output.exists() else []
    existing = {r["id"] for r in rows}
    started = time.monotonic()
    with output.open("a") as f, ProcessPoolExecutor(a.workers, initializer=initialize) as pool:
        futures = [pool.submit(execute, j) for j in jobs if j.key() not in existing]
        for future in as_completed(futures):
            row = future.result()
            row["lock_sha256"] = lock_sha
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            rows.append(row)
            if len(rows) % 20 == 0:
                print(
                    f"{a.stage}: {len(rows)}/{len(jobs)} episodes; {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    assert len(rows) == len(jobs) and len({r["id"] for r in rows}) == len(jobs)
    for name, sha in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    result = summarize(rows)
    (target / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
