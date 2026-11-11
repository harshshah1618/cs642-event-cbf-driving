"""PPO versus fixed PD skills with identical sigma-free safety filtering.

Uses the prospective10-seed controller-ablation subset. Frozen PD gains come
from the original study driver, not held-out tuning. Isolates lane-control
actions by disabling critic-distance inflation for BOTH controller banks.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import research_isolation as base
from research_freeze import verify

ORIGINAL_CONFIG = base.load_config


def config(*args, **kwargs):
    cfg = ORIGINAL_CONFIG(*args, **kwargs)
    cfg.shield.gamma_sigma = 0.0
    return cfg


def initialize():
    base.initialize()
    base.load_config = config


def main():
    selected = verify("isolation")
    directory = base.ROOT / "runs/research-controller"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "test.jsonl"
    source = __file__
    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    lock_path = directory / "lock.json"
    lock = {
        "source_sha256": source_hash,
        "gamma_sigma": 0,
        "source_isolation_lock_sha256": hashlib.sha256(
            (base.ROOT / "runs/research-isolation/locked-selection.json").read_bytes()
        ).hexdigest(),
        "controllers": ["ppo", "pd"],
        "seeds": list(range(1001, 1011)),
        "methods": ["deadline", "periodic", "periodic_gated", "benefit"],
        "scope": "Prospective controller-ablation subset, independent training variance not estimated",
    }
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("controller-ablation implementation/config changed")
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    jobs = [
        base.Job(scene, seed, delay, method, value, controller)
        for scene in ("two_trap", "merge", "dense_highway")
        for seed in range(1001, 1011)
        for delay in base.DELAYS
        for controller in ("ppo", "pd")
        for method, value in (
            ("deadline", selected["deadline"]),
            ("periodic", selected["periodic"]),
            ("periodic_gated", 10.0),
            ("benefit", 0),
        )
    ]
    completed = (
        {json.loads(line)["id"] for line in target.read_text().splitlines()}
        if target.exists()
        else set()
    )
    jobs = [j for j in jobs if j.key() not in completed]
    print("controller ablation, pending", len(jobs), flush=True)
    with target.open("a") as stream, ProcessPoolExecutor(2, initializer=initialize) as pool:
        futures = [pool.submit(base.execute, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            row = {**future.result(), "gamma_sigma": 0.0}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if i % 50 == 0 or i == len(jobs):
                print("controller ablation", i, "/", len(jobs), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
