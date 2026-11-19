"""Closed-loop camera planning with simulated image-to-application latency."""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("P3D_DISPLAY", "p3headlessgl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate_camera(adapter, output, seeds, skill_checkpoint=None, extra_delay=0.0,
                    split="validation", planner_kind="camera", protection=True, commit=None,
                    manage_maneuvers=False):
    import numpy as np
    import torch
    from driving.safemetadrive.camera_planner import (
        PendingIntent,
        QwenCameraPlanner,
        camera_config,
        capture,
    )
    from driving.safemetadrive.policies import LaneCautionDriver
    from driving.safemetadrive.protocol import SPLITS, environment_config, manifest
    from metadrive import SafeMetaDriveEnv

    start, count = SPLITS[split]
    if not seeds or any(not start <= seed < start+count for seed in seeds):
        raise ValueError("Evaluation seeds are outside the chosen partition")
    torch.set_num_threads(2)
    camera = QwenCameraPlanner(adapter) if planner_kind == "camera" else None
    config = environment_config(split)
    env = SafeMetaDriveEnv(camera_config(config) if camera else config)
    result = manifest(split, planner_kind, dict(adapter=adapter, skill_checkpoint=skill_checkpoint,
                      extra_delay=extra_delay, protection=protection, speed=30.0,
                      lane_action_mask=True,
                      manage_maneuvers=manage_maneuvers,
                      delay_semantics="measured GPU inference time plus injected delay in simulation time"))
    result.update(episodes=[], seeds=seeds, calls=[])
    if camera:
        result["inference_hardware"] = dict(gpu=torch.cuda.get_device_name(0),
                                             torch=torch.__version__,cuda=torch.version.cuda,
                                             dtype=str(next(camera.model.parameters()).dtype))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        for seed in seeds:
            env.reset(seed=seed)
            dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]
            driver = LaneCautionDriver(env.agent, speed=30, planner="external" if camera else "rule",
                                       skill_checkpoint=skill_checkpoint, protection=protection,
                                       manage_maneuvers=manage_maneuvers, control_dt=dt)
            if planner_kind in ("normal", "careful"):
                driver.planner = "external"
                driver.careful = planner_kind == "careful"
            tick = 0
            cost = reward = 0.0
            pending = None
            next_request = 0.0
            previous_frame, previous_capture = None, 0.0
            statuses: Counter[str] = Counter()
            latencies = []
            violations: set[str] = set()
            hard_brakes = 0
            while True:
                now = tick*dt
                driver.synchronize_lane()
                if pending:
                    status = pending.apply(now, driver)
                    if status != "pending":
                        statuses[status] += 1
                        result["calls"][-1].update(response_status=status,resolved_t=now,
                                                   response_age=now-pending.captured_at,
                                                   response_reason=pending.reason,
                                                   caution_applied=driver.last_caution_applied)
                        pending = None
                if camera and pending is None and now >= next_request:
                    frame, state = capture(env, driver)
                    state["history_dt"] = now-previous_capture if previous_frame is not None else 0.0
                    letter, metadata = camera.predict(frame, state, previous_frame)
                    previous_frame, previous_capture = frame, now
                    latency = metadata["inference_seconds"]
                    latencies.append(latency)
                    pending = PendingIntent(letter, now, latency+extra_delay, driver.lane)
                    result["calls"].append(dict(seed=seed, captured_t=now, letter=letter,
                                                 ready_t=pending.ready_at, **metadata, **state))
                    next_request = now + 1.5
                # One fast control calculation per physical command tick,
                # including ticks that admit a newly returned planner answer.
                action = driver.act()
                _, r, terminated, truncated, info = env.step(action)
                hard_brakes += int(action[1] < -0.8)
                for kind in ("crash_vehicle","crash_object","out_of_road","crash_human"):
                    if info.get(kind,False):
                        violations.add(kind)
                cost += float(info["cost"])
                reward += float(r)
                tick += 1
                if terminated or truncated:
                    break
            statuses["pending_at_termination"] += int(pending is not None)
            row = dict(seed=seed, success=bool(info.get("arrive_dest")),
                       collision_free_success=bool(info.get("arrive_dest")) and cost == 0,
                       cost=cost, reward=reward, steps=tick,
                       route_completion=float(info.get("route_completion", 0)),
                       calls=len(latencies), response_status=dict(statuses),
                       mean_inference_seconds=float(np.mean(latencies)) if latencies else None,
                       lane_changes=driver.changes, interventions=driver.interventions,
                       violations=sorted(violations),hard_brake_steps=hard_brakes,
                       terminated=bool(terminated), truncated=bool(truncated))
            if driver.maneuvers is not None:
                row.update(maneuver_events=list(driver.maneuvers.events),
                           recovery_aborts=driver.maneuvers.abort_count,
                           completed_maneuvers=driver.maneuvers.completed_count)
            result["episodes"].append(row)
            output.write_text(json.dumps(result, indent=2))
            if commit and len(result["episodes"]) % 5 == 0:
                commit()
            print(json.dumps(row), flush=True)
            driver.close()
    finally:
        env.close()
    result["summary"] = dict(episodes=len(seeds),
        success_rate=float(np.mean([r["success"] for r in result["episodes"]])),
        collision_free_success_rate=float(np.mean([r["collision_free_success"] for r in result["episodes"]])),
        mean_cost=float(np.mean([r["cost"] for r in result["episodes"]])),
        wall_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result, indent=2))
    return result
