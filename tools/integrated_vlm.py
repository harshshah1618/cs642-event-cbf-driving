"""Small actual-trained-camera subset of the frozen final comparison.

Three queried policies and three no-model policies, three scenes, two seeds.
The model loads once; queried policy order is balanced across scene/seed cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

from driving.assets import asset_directory
from driving.evaluation import load_skill_bank

from integrated_eval import (
    LOCAL_METHODS,
    MODEL_METHODS,
    OUT,
    ROOT,
    SCENES,
    Job,
    digest,
    execute,
    source_hashes,
    verify_sources,
)
from trained_vlm_cpu import TrainedVLMCPU
from trained_vlm_drive import ActualCPURouter


def design(scale):
    jobs = []
    for i, (scene, seed) in enumerate((s, k) for s in SCENES for k in (9101, 9102)):
        order = list(LOCAL_METHODS)
        queried = [MODEL_METHODS[(i + j) % len(MODEL_METHODS)] for j in range(len(MODEL_METHODS))]
        order = order + queried if i % 2 == 0 else queried + order
        jobs.extend(Job(scene, seed, 30.0, m, scale) for m in order)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, default=OUT / "actual-vlm")
    a = p.parse_args()
    scale = json.loads((ROOT / "configs/study-selection.json").read_text())["sigma_scale"]
    jobs = design(scale)
    hashes = source_hashes()
    extras = [
        Path(__file__),
        ROOT / "tools/trained_vlm_cpu.py",
        ROOT / "tools/trained_vlm_drive.py",
        asset_directory(ROOT / "models/camera") / "prompts.py",
        asset_directory(ROOT / "models/camera") / "adapter_model.safetensors",
        asset_directory(ROOT / "models/camera") / "adapter_config.json",
        a.base / "config.json",
    ]
    external = {str(p.resolve()): digest(p) for p in extras}
    lock = {
        "design": [asdict(j) for j in jobs],
        "package_sources": hashes,
        "other_sources_and_models": external,
        "base_revision": a.base.name,
        "base_path": str(a.base),
        "threads": 3,
        "frozen_before_model_loading": True,
        "estimated_latency_bound_s": 30.0,
        "pending_deadline_s": 45.0,
        "limitations": [
            "Six scene/seed worlds; actual CPU timing varies",
            "Three queried arms only; not the full controlled matrix",
            "Original prompt assumes all parked and literal20m/s; scenes can violate these",
        ],
    }
    encoded = json.dumps(lock, sort_keys=True, indent=2)
    a.output.mkdir(parents=True, exist_ok=True)
    lock_path = a.output / "lock.json"
    if lock_path.exists() and lock_path.read_text() != encoded:
        raise ValueError("frozen live design/source changed")
    lock_path.write_text(encoded)
    sha = hashlib.sha256(encoded.encode()).hexdigest()
    with zipfile.ZipFile(a.output / "executed-sources.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(ROOT / name, "event-cbf-driving/" + name)
        for p in extras:
            if p.suffix == ".py" and p not in [ROOT / n for n in hashes]:
                z.write(p, "other/" + p.name)
    rows_path = a.output / "episodes.jsonl"
    rows = [json.loads(s) for s in rows_path.read_text().splitlines()] if rows_path.exists() else []
    existing = {r["id"] for r in rows}
    bank, checkpoint_info = load_skill_bank(ROOT / "models/physical")
    policy = TrainedVLMCPU(
        a.base,
        asset_directory(ROOT / "models/camera"),
        asset_directory(ROOT / "models/camera") / "prompts.py",
        threads=3,
    )
    (a.output / "model-audit.json").write_text(
        json.dumps({"adapter": policy.audit, "checkpoints": checkpoint_info}, indent=2)
    )
    started = time.monotonic()
    with rows_path.open("a") as f:
        for job in jobs:
            if job.key() in existing:
                continue
            target = a.output / "episodes" / job.key()
            target.mkdir(parents=True, exist_ok=False)
            live = job.method in MODEL_METHODS
            router = ActualCPURouter(policy, target) if live else None
            result = execute(job, router=router, bank=bank, target=target, live=live)
            result["live_study_lock_sha256"] = sha
            result["actual_model_predictions"] = len(router.predictions) if live else 0
            result["prediction_letters"] = [r["letter"] for r in router.predictions] if live else []
            result["prediction_latency_s"] = (
                [r["latency_s"] for r in router.predictions] if live else []
            )
            (target / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            f.write(json.dumps(result, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            rows.append(result)
            print(
                f"LIVE {len(rows)}/{len(jobs)} {job.scene} seed{job.seed} {job.method}: "
                f"route{result['route_pct']:.1f}% crash{result['crashed']} "
                f"predictions{result['actual_model_predictions']} total{time.monotonic() - started:.1f}s",
                flush=True,
            )
    assert len(rows) == len(jobs) and len({r["id"] for r in rows}) == len(jobs)
    verify_sources(hashes)
    for name, expected in external.items():
        assert digest(name) == expected, name
    (a.output / "COMPLETE.json").write_text(
        json.dumps(
            {
                "episodes": len(rows),
                "actual_model_episodes": sum(r["actual_model"] for r in rows),
                "actual_model_predictions": sum(r["actual_model_predictions"] for r in rows),
                "all_hashes_verified": True,
            },
            indent=2,
        )
    )
    print("COMPLETE actual-model subset; all source/model hashes verified", flush=True)


if __name__ == "__main__":
    main()
