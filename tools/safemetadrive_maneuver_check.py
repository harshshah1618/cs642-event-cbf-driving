"""Local development-only checks with saved planner decisions, no VLM or cloud.

Saved letters are fixed disturbance inputs after trajectories diverge; this is
not a fresh closed-loop camera evaluation. Test maps are explicitly prohibited.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driving.safemetadrive.camera_planner import PendingIntent
from driving.safemetadrive.policies import LaneCautionDriver
from driving.safemetadrive.protocol import environment_config
from tools.safemetadrive_replay import SavedTracker


def run(maps, replicas, output):
    if not maps or any(not 100 <= seed < 150 for seed in maps):
        raise ValueError("Only development maps 100–149 are allowed")
    if any(seed not in (0, 1, 2) for seed in replicas):
        raise ValueError("Unknown trained replica")
    import torch
    from metadrive import SafeMetaDriveEnv

    torch.set_num_threads(1)
    root = ROOT / "runs/safemetadrive"
    artifact_map = json.loads((root / "heldout-study-v1/artifact-map.json").read_text())
    result = dict(
        note="Development diagnostics with fixed saved planner letters/times. "
        "No camera inference, training or cloud. Not a camera benchmark after state divergence.",
        config=environment_config("validation"),
        maps=maps,
        replicas=replicas,
        episodes=[],
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "driving/safemetadrive").glob("*.py"))
        },
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = SafeMetaDriveEnv(environment_config("validation"))
    try:
        for replica in replicas:
            reports = [
                json.loads(p.read_text())
                for p in sorted(
                    (root / "camera-balanced-v2/development").glob(f"camera-skill{replica}-*.json")
                )
            ]
            for seed in maps:
                report = next(d for d in reports if seed in d["seeds"])
                original = next(r for r in report["episodes"] if r["seed"] == seed)
                calls = [c for c in report["calls"] if c["seed"] == seed]
                by_tick = {round(c["captured_t"] * 10): c for c in calls}
                path = artifact_map["files"][report["parameters"]["skill_checkpoint"]]["path"]
                for managed in (False, True):
                    env.reset(seed=seed)
                    driver = LaneCautionDriver(
                        env.agent, speed=30, planner="external", manage_maneuvers=managed
                    )
                    driver.skill = SavedTracker(path)
                    pending = None
                    cost = 0.0
                    statuses: Counter[str] = Counter()
                    rejection_reasons: Counter[str] = Counter()
                    tick = 0
                    applied_caution_on_rejection = 0
                    requests_after_original_end = 0
                    while True:
                        now = tick * 0.1
                        driver.synchronize_lane()
                        if pending:
                            status = pending.apply(now, driver)
                            if status != "pending":
                                statuses[status] += 1
                                if status == "rejected":
                                    rejection_reasons[pending.reason or "unspecified"] += 1
                                    applied_caution_on_rejection += driver.last_caution_applied
                                pending = None
                        call = by_tick.get(tick)
                        # If the diagnostic runs beyond the saved episode, keep
                        # the last intent; never invent a new camera answer.
                        if call is not None:
                            if pending is not None:
                                raise RuntimeError("Saved request schedule overlaps")
                            pending = PendingIntent(
                                call["letter"],
                                now,
                                call["ready_t"] - call["captured_t"],
                                driver.lane,
                            )
                        elif tick * 0.1 > calls[-1]["captured_t"]:
                            requests_after_original_end += 1
                        _, _, terminated, truncated, info = env.step(driver.act())
                        cost += float(info["cost"])
                        tick += 1
                        if terminated or truncated:
                            break
                    row = dict(
                        replica=replica,
                        seed=seed,
                        managed=managed,
                        success=bool(info.get("arrive_dest")),
                        cost=cost,
                        steps=tick,
                        route_completion=float(info.get("route_completion", 0)),
                        response_status=dict(statuses),
                        rejection_reasons=dict(rejection_reasons),
                        caution_on_rejected_request=int(applied_caution_on_rejection),
                        original_success=original["success"],
                        original_cost=original["cost"],
                        original_steps=original["steps"],
                        original_endpoint_matches=(
                            bool(info.get("arrive_dest")) == original["success"]
                            and cost == original["cost"]
                            and tick == original["steps"]
                        ),
                        steps_after_saved_requests_end=requests_after_original_end,
                        recovery_aborts=driver.maneuvers.abort_count if driver.maneuvers else 0,
                        events=list(driver.maneuvers.events) if driver.maneuvers else [],
                    )
                    result["episodes"].append(row)
                    output.write_text(json.dumps(result, indent=2))
                    print(json.dumps({k: v for k, v in row.items() if k != "events"}), flush=True)
                    driver.close()
    finally:
        env.close()
    result["summary"] = {
        str(managed): dict(
            episodes=len(rows := [r for r in result["episodes"] if r["managed"] == managed]),
            successes=sum(r["success"] for r in rows),
            cost=sum(r["cost"] for r in rows),
            aborts=sum(r["recovery_aborts"] for r in rows),
        )
        for managed in (False, True)
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", nargs="+", type=int, default=list(range(100, 110)))
    parser.add_argument("--replicas", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default="runs/safemetadrive/maneuver-development-v1.json")
    args = parser.parse_args()
    run(args.maps, args.replicas, args.output)
