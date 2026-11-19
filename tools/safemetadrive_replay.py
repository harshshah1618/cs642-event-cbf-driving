"""CPU-only diagnostic replay of recorded camera outputs; never runs a camera model.

A replay is evidence about an original drive only when recorded state, response
status and episode outcome checks pass. It never changes a controller or policy.
"""

import argparse
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from driving.safemetadrive.camera_planner import PendingIntent
from driving.safemetadrive.policies import LaneCautionDriver
from driving.safemetadrive.protocol import environment_config
from tools.safemetadrive_camera_data import teacher_recommendation


class SavedTracker:
    """Deterministic SB3 2.3.2 MLP actor: 6-64-64-2, Tanh hidden, clipped mean.

    Reads weights only, without importing SB3 or unpickling saved class metadata.
    This is a diagnostic loader, not a replacement benchmark implementation.
    """

    def __init__(self, path):
        with zipfile.ZipFile(path) as archive:
            state = torch.load(
                io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True
            )
            data = json.loads(archive.read("data"))
        assert data["use_sde"] is False
        self.layers = []
        for prefix, shape in [
            ("mlp_extractor.policy_net.0", (64, 6)),
            ("mlp_extractor.policy_net.2", (64, 64)),
            ("action_net", (2, 64)),
        ]:
            w, b = state[prefix + ".weight"], state[prefix + ".bias"]
            assert tuple(w.shape) == shape
            self.layers.append((w, b))

    def predict(self, features, deterministic=True):
        assert deterministic
        x = torch.as_tensor(features, dtype=torch.float32)
        with torch.inference_mode():
            for i, (weight, bias) in enumerate(self.layers):
                x = torch.nn.functional.linear(x, weight, bias)
                if i < 2:
                    x = torch.tanh(x)
            return x.clamp(-1, 1).numpy(), None


def rejection_reason(driver, pending):
    lanes = driver.vehicle.navigation.current_ref_lanes
    if pending.lane not in lanes or driver.lane is not pending.lane:
        return "stale_lane_context"
    target = lanes.index(pending.lane) + pending.delta
    if not 0 <= target < len(lanes):
        return "invalid_adjacent_lane"
    if pending.delta:
        lat = driver.lane.local_coordinates(driver.vehicle.position)[1]
        if abs(lat) > 0.65:
            return "transition_unfinished_or_off_center"
        f, r, _ = driver.lane_objects(
            lanes[target], driver.vehicle.lidar.get_surrounding_objects(driver.vehicle)
        )
        if driver.protection and (
            f < max(8, driver.vehicle.speed * 1.1) or r < max(10, driver.vehicle.speed * 0.8)
        ):
            return "front_or_rear_gap_too_small"
    return None


def replay(replica, seed, label="Camera-delay0"):
    study = ROOT / "runs/safemetadrive/heldout-study-v1"
    paths = json.loads((study / "groups-local.json").read_text())[label][replica]
    report = next(d for path in paths if seed in (d := json.loads(Path(path).read_text()))["seeds"])
    original = next(row for row in report["episodes"] if row["seed"] == seed)
    calls = [dict(row) for row in report["calls"] if row["seed"] == seed]
    manifest = json.loads((study / "artifact-map.json").read_text())
    checkpoint = manifest["files"][report["parameters"]["skill_checkpoint"]]["path"]
    torch.set_num_threads(1)
    from metadrive import SafeMetaDriveEnv

    env = SafeMetaDriveEnv(environment_config("test"))
    trace, requests, responses = [], [], []
    pending, pending_call, cost, reward = None, None, 0.0, 0.0
    captured = {round(row["captured_t"] * 10): row for row in calls}
    try:
        env.reset(seed=seed)
        driver = LaneCautionDriver(
            env.agent, speed=30, planner="external", protection=report["parameters"]["protection"]
        )
        driver.skill = SavedTracker(checkpoint)
        tick = 0
        while True:
            now = tick * 0.1
            driver.synchronize_lane()
            if pending:
                reason = rejection_reason(driver, pending) if now >= pending.ready_at else None
                status = pending.apply(now, driver)
                if status != "pending":
                    assert pending_call is not None
                    responses.append(
                        dict(
                            captured_t=pending.captured_at,
                            resolved_t=now,
                            status=status,
                            expected=pending_call.get("response_status"),
                            reason=reason,
                            letter=pending_call["letter"],
                        )
                    )
                    pending = None
            if tick in captured:
                call = captured[tick]
                assert pending is None
                lanes = env.agent.navigation.current_ref_lanes
                actual = dict(
                    speed_mps=float(env.agent.speed),
                    current_lane=int(env.agent.lane.index[-1]),
                    target_lane=int(driver.lane.index[-1]),
                    lane_count=len(lanes),
                )
                requests.append(
                    dict(
                        captured_t=now,
                        letter=call["letter"],
                        teacher=teacher_recommendation(driver),
                        actual=actual,
                        expected={k: call[k] for k in actual},
                        top_probability=max(call["probabilities"]),
                    )
                )
                pending = PendingIntent(
                    call["letter"], now, call["ready_t"] - call["captured_t"], driver.lane
                )
                pending_call = call
            action = driver.act()
            position = list(map(float, env.agent.position))
            heading = float(env.agent.heading_theta)
            assert driver.last_features is not None
            features = driver.last_features.tolist()
            stop_diagnostics = None
            if features[1] * 15 < 0.1 and features[2] * 15 < 0.1:
                objects = env.agent.lidar.get_surrounding_objects(env.agent)
                front, rear, front_speed = driver.lane_objects(driver.lane, objects)
                conflict = driver.conflict_time(objects)
                stop_diagnostics = dict(
                    front_gap_m=front,
                    rear_gap_m=rear,
                    front_speed_mps=front_speed,
                    conflict_seconds=conflict if math.isfinite(conflict) else None,
                )
            _, r, terminated, truncated, info = env.step(action)
            cost += float(info["cost"])
            reward += float(r)
            trace.append(
                dict(
                    t=now,
                    x=position[0],
                    y=position[1],
                    heading=heading,
                    speed_mps=features[1] * 15,
                    desired_speed_mps=features[2] * 15,
                    lateral_error_m=features[3] * 5,
                    careful=driver.careful,
                    target_lane=int(driver.lane.index[-1]),
                    action=list(map(float, action)),
                    cost=float(info["cost"]),
                    stop_diagnostics=stop_diagnostics,
                    route_completion=float(info.get("route_completion", 0)),
                    violations=[
                        key
                        for key in ("crash_vehicle", "crash_object", "out_of_road", "crash_human")
                        if info.get(key, False)
                    ],
                )
            )
            tick += 1
            if terminated or truncated:
                break
        actual = dict(
            success=bool(info.get("arrive_dest")),
            cost=cost,
            reward=reward,
            steps=tick,
            route_completion=float(info.get("route_completion", 0)),
            lane_changes=driver.changes,
            interventions=driver.interventions,
        )
        errors = [
            abs(row["actual"]["speed_mps"] - row["expected"]["speed_mps"]) for row in requests
        ]
        check = dict(
            requests_replayed=len(requests),
            original_requests=len(calls),
            max_capture_speed_error=max(errors, default=0),
            lane_metadata_mismatches=sum(
                any(
                    row["actual"][k] != row["expected"][k]
                    for k in ("current_lane", "target_lane", "lane_count")
                )
                for row in requests
            ),
            response_status_mismatches=sum(row["status"] != row["expected"] for row in responses),
            outcome_matches=all(
                actual[k] == original[k]
                for k in ("success", "cost", "steps", "lane_changes", "interventions")
            ),
            reward_absolute_error=abs(reward - original["reward"]),
        )
        check["faithful_within_tolerance"] = (
            len(requests) == len(calls)
            and max(errors, default=0) < 1e-4
            and check["lane_metadata_mismatches"] == 0
            and check["response_status_mismatches"] == 0
            and check["outcome_matches"]
            and check["reward_absolute_error"] < 1e-3
        )
        result = dict(
            replica=replica,
            seed=seed,
            label=label,
            source_report=report["parameter_sha256"],
            note="CPU diagnostic replay of saved decisions, no camera inference or tuning; "
            "telemetry describes original run only if faithfulness checks pass.",
            torch_version=torch.__version__,
            original=original,
            actual=actual,
            verification=check,
            requests=requests,
            responses=responses,
            trace=trace,
        )
        output = ROOT / f"runs/safemetadrive/local-assessment/replay-{replica}-{seed}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(
            json.dumps(dict(replica=replica, seed=seed, verification=check, actual=actual)),
            flush=True,
        )
        driver.close()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["0:524"])
    args = parser.parse_args()
    for case in args.cases:
        replica, seed = map(int, case.split(":"))
        replay(replica, seed)
