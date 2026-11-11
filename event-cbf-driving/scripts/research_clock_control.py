"""Exploratory paired phase control: preselected rates, identical physical seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import research_isolation as base
from research_freeze import digest, verify

ROOT = base.ROOT
ORIGINAL_RUN = base.run_episode
FRACTION = 0.0
SCENES = ("two_trap", "merge", "dense_highway")
DELAYS = ("nominal", "long", "jitter")
CONDITIONS = (("zero", 0.1), ("zero", 1.0), ("uniform", 0.1))


@dataclass(frozen=True)
class ClockJob:
    scenario: str
    seed: int
    delay: str
    phase_mode: str
    value: float

    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]


def jobs():
    return [ClockJob(s, seed, d, mode, rate) for s in SCENES
            for seed in range(1101, 1141) for d in DELAYS for mode, rate in CONDITIONS]


def phase_run(*args, **kwargs):
    cfg = kwargs["stack_config"]
    assert not cfg.pending_margin and cfg.pending_margin_on_all_calls
    cfg.polling_phase_s = FRACTION / cfg.fixed_rate_hz
    return ORIGINAL_RUN(*args, **kwargs)


def initialize():
    base.initialize()
    base.run_episode = phase_run


def execute(job):
    global FRACTION
    FRACTION = (float(np.random.default_rng(job.seed + 72001).uniform())
                if job.phase_mode == "uniform" else 0.0)
    result = base.execute(base.Job(job.scenario, job.seed, job.delay, "periodic", job.value))
    return {**result, **asdict(job), "id": job.key(), "base_id": result["id"],
            "polling_phase_fraction": FRACTION, "polling_phase_s": FRACTION / job.value,
            "provenance": "new_clock_control"}


def lock(output):
    selected = verify("isolation")
    phase = verify("phase")
    assert selected["periodic"] == 0.1 and phase["periodic"] == 1
    expected = {
        "script_sha256": digest(Path(__file__)),
        "isolation_lock_sha256": digest(ROOT / "runs/research-isolation/locked-selection.json"),
        "phase_lock_sha256": digest(ROOT / "runs/research-phase/locked-selection.json"),
        "conditions": [list(x) for x in CONDITIONS], "seeds": [1101, 1140],
        "scenarios": list(SCENES), "delays": list(DELAYS), "new_episodes": 1080,
        "jobs_sha256": hashlib.sha256(json.dumps([asdict(j) for j in jobs()], sort_keys=True).encode()).hexdigest(),
        "analysis": "uniform minus zero; 20000 paired seed bootstrap draws, all scenarios/delays clustered by seed; no retuning",
    }
    path = output / "lock.json"
    if path.exists():
        existing = json.loads(path.read_text())
        assert all(existing[k] == v for k, v in expected.items()), "clock control lock changed"
    else:
        expected["frozen_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(output):
    lock(output)
    rows = read_rows(output / "test.jsonl")
    assert len(rows) == 1080 and {r["id"] for r in rows} == {j.key() for j in jobs()}
    phase_path = ROOT / "runs/research-phase/test.jsonl"
    phase_rows = read_rows(phase_path)
    assert len(phase_rows) == 2880 and len({r["id"] for r in phase_rows}) == 2880, "phase suite incomplete"
    reused = [r for r in phase_rows if r["method"] == "periodic" and r["value"] == 1]
    assert len(reused) == 360
    for r in reused:
        j = ClockJob(r["scenario"], r["seed"], r["delay"], "uniform", 1.0)
        expected = float(np.random.default_rng(j.seed + 72001).uniform())
        assert r["polling_phase_fraction"] == expected
        rows.append({**r, "id": j.key(), "base_id": r["id"], "phase_mode": "uniform",
                     "provenance": "existing_phase_test"})
    (output / "combined.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    metrics = ("success", "route_pct", "router_calls", "end_time", "wall_s", "crashed")
    rng = np.random.default_rng(84021)
    indices = rng.integers(0, 40, size=(20000, 40))
    summary = {"phase_test_sha256": digest(phase_path), "test_sha256": digest(output / "test.jsonl"),
               "metrics": {}, "groups": {}, "subgroups": {}}
    for rate in (0.1, 1.0):
        paired = {(r["scenario"], r["seed"], r["delay"], r["phase_mode"]): r
                  for r in rows if r["value"] == rate}
        for mode in ("zero", "uniform"):
            group = [r for r in rows if r["value"] == rate and r["phase_mode"] == mode]
            summary["groups"][f"{rate}|{mode}"] = {m: float(np.mean([r[m] for r in group])) for m in metrics}
        def differences(metric, scenes=SCENES, delays=DELAYS):
            return np.array([np.mean([
                float(paired[s, seed, d, "uniform"][metric]) - float(paired[s, seed, d, "zero"][metric])
                for s in scenes for d in delays]) for seed in range(1101, 1141)])
        for metric in metrics:
            delta = differences(metric)
            ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
            summary["metrics"][f"{rate}|{metric}"] = {"difference": float(delta.mean()), "ci95": ci.tolist()}
        for s in SCENES:
            for d in DELAYS:
                summary["subgroups"][f"{rate}|{s}|{d}"] = {m: float(differences(m, (s,), (d,)).mean()) for m in metrics}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["metrics"], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("run", "analyze"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", type=Path, default=ROOT / "runs/research-clock")
    args = p.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("clock control permits at most two workers")
    args.output.mkdir(parents=True, exist_ok=True)
    lock(args.output)
    if args.stage == "analyze":
        analyze(args.output)
        return
    target = args.output / "test.jsonl"
    existing = read_rows(target) if target.exists() else []
    assert len(existing) == len({r["id"] for r in existing})
    completed = {r["id"] for r in existing}
    pending = [j for j in jobs() if j.key() not in completed]
    print(f"clock: pending={len(pending)} completed={len(completed)}", flush=True)
    with target.open("a") as stream, ProcessPoolExecutor(args.workers, initializer=initialize) as pool:
        futures = [pool.submit(execute, j) for j in pending]
        for i, f in enumerate(as_completed(futures), 1):
            stream.write(json.dumps(f.result(), allow_nan=False) + "\n")
            stream.flush()
            if i % 30 == 0 or i == len(pending):
                print(f"clock: {i}/{len(pending)}", flush=True)
    lock(args.output)
    print("clock: complete; run analyze after all 2880 phase rows are available", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
