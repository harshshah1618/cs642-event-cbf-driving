"""Audit historical episode collisions using the pinned native cost/termination rules.

No VLM inference or cloud jobs. This inference is valid only when departure and
human contact terminate immediately and the three charged native cost types
each cost one. Cost-one episodes without final contact remain ambiguous.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def collision_bounds(row):
    """Bounds conditional on verify_native_contract() and the published evaluator.

    Out-of-road can contribute at most one cost unit because the evaluator
    breaks at termination. More than one unit proves vehicle/object contact.
    Human contact terminates, so its final flag cannot disappear. Thus zero
    total cost and no final human flag exclude all three measured contact types.
    """
    cost = float(row["cost"])
    if not math.isfinite(cost) or cost < 0 or not cost.is_integer():
        raise ValueError("Expected a nonnegative integer sum of native unit costs")
    final_contact = any(row[key] for key in ("crash_vehicle", "crash_object", "crash_human"))
    if final_contact or cost > 1:
        return 1, 1
    if cost == 0:
        return 0, 0
    return 0, 1


def verify_native_contract():
    from importlib.metadata import distribution

    from driving.irregular.escalation import irregular_config
    from driving.safemetadrive.camera_planner import camera_config
    from driving.safemetadrive.protocol import METADRIVE_REVISION
    from metadrive import SafeMetaDriveEnv
    from metadrive.envs import metadrive_env

    package = distribution("metadrive-simulator")
    direct_url = json.loads(package.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != METADRIVE_REVISION:
        raise ValueError("Install the recorded pinned MetaDrive source before inferring counts")
    env = SafeMetaDriveEnv(camera_config(irregular_config("irregular-test")))
    expected = dict(out_of_road_done=True, crash_human_done=True, out_of_road_cost=1.,
                    crash_vehicle_cost=1., crash_object_cost=1.,
                    crash_vehicle_done=False, crash_object_done=False)
    try:
        actual = {key: env.config[key] for key in expected}
    finally:
        env.close()
    if actual != expected:
        raise ValueError(f"Native rules differ: {actual}")
    return dict(simulator_revision=METADRIVE_REVISION, config=actual,
                native_environment_source_sha256=hashlib.sha256(
                    Path(metadrive_env.__file__).read_bytes()).hexdigest(),
                assumption="Historical runs used the pinned published environment recipe; "
                           "their records do not contain independent runtime config dumps.")


def audit(rows, native_contract):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene"]].append(row)
    summaries = []
    for scene, rs in sorted(groups.items()):
        if len(rs) != 50 or {r["seed"] for r in rs} != set(range(1000, 1050)):
            raise ValueError(f"{scene}: expected exactly one episode per published map")
        bounds = [collision_bounds(r) for r in rs]
        lower, upper = sum(b[0] for b in bounds), sum(b[1] for b in bounds)
        summaries.append(dict(
            scene=scene, episodes=len(rs), collision_lower_bound=lower,
            collision_upper_bound=upper, collision_episodes=lower if lower == upper else None,
            collision_inference="native-cost-and-terminal-human-flag",
            legacy_final_step_collisions=sum(any(r[key] for key in
                ("crash_vehicle", "crash_object", "crash_human")) for r in rs),
            collision_seeds=[r["seed"] for r, b in zip(rs, bounds, strict=True) if b[0]],
            ambiguous_seeds=[r["seed"] for r, b in zip(rs, bounds, strict=True) if b[0] != b[1]],
            passed_stop=sum(r["passed_stop"] for r in rs),
            mean_cost=sum(r["cost"] for r in rs) / len(rs),
            mean_calls=sum(len(r["calls"]) for r in rs) / len(rs),
            mean_route_completion=sum(r["route_completion"] for r in rs) / len(rs),
        ))
    return dict(scope="Audit of saved historical Qwen episodes, not new inference or replay",
                native_contract=native_contract, episodes=len(rows), groups=summaries,
                no_cloud_jobs=True, no_new_inference=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = sorted(args.source.glob("*.json"))
    rows = [r for p in files for r in json.loads(p.read_text())]
    if len(rows) != 400 or len({r["scene"] for r in rows}) != 8:
        raise ValueError("Expected the complete eight-scene, 400-episode historical run")
    result = audit(rows, verify_native_contract())
    result["input_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for r in result["groups"]:
        print(r["scene"], "collisions", r["collision_episodes"], "STOP violations",r["passed_stop"],
              "mean cost", r["mean_cost"])


if __name__ == "__main__":
    main()
