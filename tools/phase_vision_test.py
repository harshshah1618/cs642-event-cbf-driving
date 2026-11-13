"""Eight rendered poses x three skill hints; actual trained-model image diagnostic.

No vehicle steps are taken. Expected letters are prespecified scene-design labels,
not independently annotated image ground truth. Outputs are exploratory only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from driving.assets import asset_directory

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")
CASES = (
    ("overtake", 0.0, 0.0, ((40.0, 0.0),), "B"),
    ("transitionRight", 1.0, 0.12, ((40.0, 0.0),), "B"),
    ("transitionLeft", 3.0, 0.12, ((40.0, 0.0),), "B"),
    ("settledLeft", 4.0, 0.0, ((40.0, 0.0),), "B"),
    ("returnLeft", 4.0, 0.0, ((-15.0, 0.0),), "C"),
    ("clearRight", 0.0, 0.0, (), "A"),
    ("blockedRight", 0.0, 0.0, ((40.0, 0.0), (45.0, 4.0)), "A"),
    ("earlyturnRight", 0.4, 0.08, ((40.0, 0.0),), "B"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize(rows):
    by_case = {}
    for name, _, _, _, expected in CASES:
        group = [r for r in rows if r["case"] == name]
        if len(group) != 3 or {r["skill"] for r in group} != set(SKILLS):
            raise ValueError("Need exactly three distinct skill hints for every image")
        if len({r["prediction"]["image_hash"] for r in group}) != 1:
            raise ValueError("Skill-hint comparison must use identical bytes")
        letters = {r["skill"]: r["prediction"]["letter"] for r in group}
        by_case[name] = {
            "expected": expected,
            "letters": letters,
            "text_flip": len(set(letters.values())) > 1,
            "design_label_matches": sum(x == expected for x in letters.values()),
        }
    return {
        "kind": "exploratory_8_pose_image_hint_diagnostic_not_driving",
        "predictions": len(rows),
        "distinct_images": len(by_case),
        "images_with_text_flip": sum(v["text_flip"] for v in by_case.values()),
        "design_label_matches": sum(v["design_label_matches"] for v in by_case.values()),
        "by_case": by_case,
        "latency_s": [r["prediction"]["latency_s"] for r in rows],
        "limitations": [
            "Eight selected poses; no population accuracy estimate",
            "Scene-design expected letters, not blinded visual annotations",
            "Changing hint is causal within a fixed image; pose comparisons also change pixels",
            "Literal 20m/s prompt; no temporal history or driving rollout",
        ],
    }


def capture(output):
    from driving.config import ScenarioConfig
    from driving.env_physics import MetaDriveAdapter

    frames = []
    for name, y, heading, traffic, expected in CASES:
        print(f"capture: {name}", flush=True)
        env = MetaDriveAdapter(
            ScenarioConfig(
                "dense_highway",
                simulator="metadrive",
                randomize=False,
                duration=10,
                route_length=200,
                params={
                    "rgb_camera": True,
                    "initial_y": y,
                    "initial_traffic": [{"x": x, "y": oy, "vx": 0.0} for x, oy in traffic],
                },
            )
        )
        try:
            env.reset(seed=7101)
            env.env.vehicle.set_heading_theta(heading)
            snap = env.snapshot()
            camera = env.capture_router_camera()
            path = output / "frames" / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(camera.image_bytes)
            frames.append(
                {
                    "case": name,
                    "expected": expected,
                    "path": str(path.resolve()),
                    "image_sha256": digest(path),
                    "lane": camera.lane,
                    "heading_requested": heading,
                    "snapshot": asdict(snap),
                    "width": camera.width,
                    "height": camera.height,
                    "captured_t": camera.captured_t,
                }
            )
        finally:
            env.close()
    return frames


def source_hashes(prompts):
    paths = [Path(__file__), ROOT / "tools/trained_vlm_cpu.py", prompts]
    paths += sorted((ROOT / "driving").rglob("*.py"))
    return {str(p.resolve()): digest(p) for p in paths}


def verify_lock(lock):
    for path, expected in lock["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for frame in lock["frames"]:
        if digest(frame["path"]) != frame["image_sha256"]:
            raise ValueError("Frozen frame changed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=asset_directory(ROOT / "models/camera"))
    p.add_argument(
        "--prompts",
        type=Path,
        default=asset_directory(ROOT / "models/camera") / "prompts.py",
    )
    p.add_argument("--output", type=Path, default=ROOT / "runs/maneuver-vision")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output directory already exists; do not overwrite evidence")
    args.output.mkdir(parents=True)
    frames = capture(args.output)
    lock = {
        "design": CASES,
        "skills": SKILLS,
        "frames": frames,
        "source_hashes": source_hashes(args.prompts),
        "base_path": str(args.base.resolve()),
        "base_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "adapter_sha256": digest(args.adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": digest(args.adapter / "adapter_config.json"),
        "base_config_sha256": digest(args.base / "config.json"),
        "threads": 3,
        "seed": 7101,
        "frozen_before_first_prediction": True,
        "created_unix": time.time(),
    }
    write_json(args.output / "lock.json", lock)
    print(f"frozen: {digest(args.output / 'lock.json')}", flush=True)
    verify_lock(lock)
    from trained_vlm_cpu import TrainedVLMCPU

    policy = TrainedVLMCPU(args.base, args.adapter, args.prompts, threads=3)
    rows = []
    with (args.output / "predictions.jsonl").open("x") as stream:
        for frame in frames:
            png = Path(frame["path"]).read_bytes()
            for skill in SKILLS:
                result = policy.predict(png, frame["lane"], skill)
                if result["image_hash"] != frame["image_sha256"]:
                    raise ValueError("Prediction image hash mismatch")
                row = {
                    "case": frame["case"],
                    "expected": frame["expected"],
                    "skill": skill,
                    "lane": frame["lane"],
                    "prediction": result,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(
                    f"prediction {len(rows)}/24: {frame['case']} {skill} -> {result['letter']}",
                    flush=True,
                )
    verify_lock(lock)
    summary = summarize(rows)
    summary["source_and_frame_locks_verified_after_predictions"] = True
    summary["lock_sha256"] = digest(args.output / "lock.json")
    write_json(args.output / "analysis.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
