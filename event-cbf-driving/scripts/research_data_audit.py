"""Report-time cell and runtime audits; does not modify frozen experiments."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from importlib import metadata
from pathlib import Path

METADRIVE_COMMIT = "fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1"
ROOT = Path(__file__).resolve().parents[1]


def validate_test_cells(suite, rows, lock, isolation_lock=None):
    """Require every prospective cell exactly once, with its frozen value and ID."""
    if suite not in ("isolation", "roads", "phase"):
        raise ValueError(f"unknown suite: {suite}")
    roads = suite == "roads"
    field = "environment" if roads else "scenario"
    conditions = ("generalization", "safe") if roads else ("two_trap", "merge", "dense_highway")
    start = {"isolation": 1001, "roads": 2001, "phase": 1101}[suite]
    delays = ("nominal", "long", "jitter") if suite == "phase" else (
        "standard" if roads else "fast", "nominal", "slow", "long", "jitter"
    )
    deadline = lock["deadline_buffer_s"] if suite == "phase" else lock["deadline"]
    if suite == "phase" and isolation_lock is not None and deadline != isolation_lock["deadline"]:
        raise ValueError("phase deadline buffer differs from isolation lock")
    expected = {}
    for condition in conditions:
        for delay in delays:
            methods = {
                "ttc": 0, "benefit": 0, "no_latency_matched": deadline,
                "deadline": deadline, "periodic": lock["periodic"], "periodic_gated": 10,
                "periodic_conditioned": lock["conditioned_periodic"][condition + "|" + delay],
            }
            methods.update({"never": 0, "shield": 0} if roads else {"legacy": 0})
            if roads and delay == "standard":
                methods.update({"idm": 0, "expert": 0})
            for seed in range(start, start + 40):
                for method, value in methods.items():
                    expected[(condition, seed, delay, method)] = value
    seen, ids = set(), set()
    for row in rows:
        try:
            key = (row[field], row["seed"], row["delay"], row["method"])
            if key in seen:
                raise ValueError(f"duplicate cell: {key}")
            if key not in expected:
                raise ValueError(f"unexpected cell: {key}")
            value = row["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid parameter value: {key}")
            if value != expected[key]:
                raise ValueError(f"selection value mismatch: {key}: {value} != {expected[key]}")
            if type(row["seed"]) is not int:
                raise ValueError(f"noninteger scene seed: {key}")
            job = {name: row[name] for name in (field, "seed", "delay", "method", "value")}
            if not roads:
                if row["controller"] != "ppo":
                    raise ValueError(f"unexpected controller: {key}")
                job["controller"] = row["controller"]
            expected_id = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
            if row["id"] != expected_id:
                raise ValueError(f"job ID mismatch: {key}")
            if row["id"] in ids:
                raise ValueError(f"duplicate job ID: {row['id']}")
            seen.add(key)
            ids.add(row["id"])
        except KeyError as exc:
            raise ValueError(f"missing required row field: {exc}") from exc
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"missing {len(missing)} expected cells; example: {sorted(missing)[0]}")
    return {"suite": suite, "expected_cells": len(expected), "validated_cells": len(seen),
            "scene_seeds": 40, "exact_cells_verified": True,
            "selection_values_verified": True, "job_ids_verified": True}


def runtime_audit(locks):
    """Verify installed version metadata and simulator VCS pin against locks.

    This does not authenticate remote sources or hash installed runtime contents.
    Repository/checkpoint contents are checked separately by research_freeze.verify.
    """
    entries = locks.items() if isinstance(locks, dict) else enumerate(locks)
    current = {}
    for suite, lock in entries:
        inventory = {
            str(p.relative_to(ROOT))
            for p in (ROOT / "checkpoints/metadrive-calibrated").glob("*.pt")
        }
        if inventory != set(lock["checkpoint_sha256"]):
            raise ValueError(f"checkpoint inventory changed since {suite} freeze")
        if lock.get("python") != platform.python_version():
            raise ValueError(f"Python version changed since {suite} freeze")
        if not lock.get("packages"):
            raise ValueError(f"package versions missing from {suite} lock")
        for package, expected in lock["packages"].items():
            actual = metadata.version(package)
            if actual != expected:
                raise ValueError(f"runtime version mismatch: {suite}/{package}: {actual} != {expected}")
            current[package] = actual
    raw = metadata.distribution("metadrive-simulator").read_text("direct_url.json")
    if raw is None:
        raise ValueError("MetaDrive VCS provenance missing")
    provenance = json.loads(raw)
    vcs = provenance.get("vcs_info", {})
    if vcs.get("vcs") != "git" or vcs.get("commit_id") != METADRIVE_COMMIT:
        raise ValueError("MetaDrive installed commit differs from pinned simulator")
    return {"python": platform.python_version(), "packages": current,
            "metadrive_direct_url": provenance, "runtime_versions_verified": True,
            "metadrive_commit_metadata_verified": True,
            "checkpoint_inventory_verified": True,
            "scope": "Repository and checkpoint hashes are verified separately; runtime checks "
                     "verify installed version/VCS metadata, not runtime file integrity or remote authenticity."}
