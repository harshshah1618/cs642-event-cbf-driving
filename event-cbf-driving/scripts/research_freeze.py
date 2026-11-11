"""Freeze complete validation-only selections and experiment source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(group):
    return (
        sum(r["success"] for r in group),
        np.mean([r["route_pct"] for r in group]),
        -np.mean([r["router_calls"] for r in group]),
    )


def freeze(suite):
    directory = ROOT / "runs" / f"research-{suite}"
    rows_path = directory / "validation.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    expected = {"isolation": 1560, "roads": 1152, "phase": 216}[suite]
    if len(rows) != expected or len({r["id"] for r in rows}) != expected:
        raise ValueError(f"incomplete/duplicate validation: {len(rows)} != {expected}")
    selection_path = directory / "selection.json"
    selected = json.loads(selection_path.read_text())
    if selected["validation_sha256"] != digest(rows_path):
        raise ValueError("selection/validation hash mismatch")
    key = "environment" if suite == "roads" else "scenario"
    mapping = {}
    for condition in sorted({r[key] for r in rows}):
        for delay in sorted({r["delay"] for r in rows}):
            candidates = []
            for rate in (0.1, 0.25, 0.5, 1, 2, 10):
                group = [
                    r
                    for r in rows
                    if r[key] == condition
                    and r["delay"] == delay
                    and r["method"] == "periodic"
                    and r["value"] == rate
                ]
                if not group:
                    raise ValueError("missing polling cell")
                candidates.append((score(group), rate))
            mapping[condition + "|" + delay] = max(candidates)[1]
    selected["conditioned_periodic"] = mapping
    sources = list((ROOT / "event_cbf_driving").rglob("*.py"))
    if suite != "roads":
        sources = [p for p in sources if p.name != "road_planning.py"]
    sources += [
        ROOT / "scripts/research_freeze.py",
        ROOT / f"scripts/research_{suite if suite != 'isolation' else 'isolation'}.py",
    ]
    if suite == "phase":
        sources += [ROOT / "scripts/research_isolation.py"]
    sources += list((ROOT / "configs").glob("*.yaml"))
    selected["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources))}
    selected["checkpoint_sha256"] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (ROOT / "checkpoints/metadrive-calibrated").glob("*.pt")
    }
    selected["frozen_at"] = datetime.now(timezone.utc).isoformat()
    selected["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    selected["python"] = platform.python_version()
    selected["packages"] = {
        name: version(name) for name in ("numpy", "torch", "metadrive-simulator", "gymnasium")
    }
    lock_path = directory / "locked-selection.json"
    if lock_path.exists():
        raise ValueError("selection already frozen; do not overwrite an experiment lock")
    lock_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    print(lock_path)


def verify(suite, directory=None):
    directory = directory or ROOT / "runs" / f"research-{suite}"
    selected = json.loads((directory / "locked-selection.json").read_text())
    if selected["validation_sha256"] != digest(directory / "validation.jsonl"):
        raise ValueError("validation changed after freeze")
    for name, expected in {**selected["source_sha256"], **selected["checkpoint_sha256"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"experiment source/checkpoint changed after freeze: {name}")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["isolation", "roads", "phase"])
    freeze(parser.parse_args().suite)
