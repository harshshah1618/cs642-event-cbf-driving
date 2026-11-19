"""Render camera-only planner training examples along training-map trajectories."""

import json
import os
import sys
from collections import Counter
from copy import copy
from pathlib import Path

os.environ.setdefault("P3D_DISPLAY", "p3headlessgl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def teacher_recommendation(driver):
    """Query the rule now without changing the executing driver's target/timer.

    Data collection uses preview control, whose calculation does not mutate
    the shared vehicle/tracker. A shallow copy keeps only scalar planning state
    separate; no simulator step or control application occurs here.
    """
    from driving.safemetadrive.camera_planner import INTENTS, LETTERS

    if not driver.preview:
        raise ValueError("Teacher queries require side-effect-free preview control")
    oracle = copy(driver)
    oracle.planner, oracle.tick, oracle.skill = "rule", 0, None
    oracle.act()
    lanes = driver.vehicle.navigation.current_ref_lanes
    delta = lanes.index(oracle.lane)-lanes.index(driver.lane)
    return LETTERS[INTENTS.index((delta, oracle.last_caution_label))]


def generate_recovery(seeds, output, max_steps=1000, delay=2.0):
    """Label delayed teacher rollouts, including states after missed decisions."""
    from driving.safemetadrive.camera_planner import PendingIntent, camera_config, capture
    from driving.safemetadrive.policies import LaneCautionDriver
    from driving.safemetadrive.protocol import environment_config, manifest
    from metadrive import SafeMetaDriveEnv

    if not seeds or any(not 0 <= seed < 100 for seed in seeds):
        raise ValueError("Recovery collection is restricted to training maps")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = SafeMetaDriveEnv(camera_config(environment_config("train")))
    labels, episodes = [], []
    try:
        for seed in seeds:
            env.reset(seed=seed)
            driver = LaneCautionDriver(env.agent, speed=30.0, planner="external")
            dt = env.config["physics_world_step_size"]*env.config["decision_repeat"]
            pending, next_request = None, 0.0
            cost, steps = 0.0, 0
            statuses: Counter[str] = Counter()
            while steps < max_steps:
                now = steps*dt
                driver.synchronize_lane()
                if pending:
                    status = pending.apply(now, driver)
                    if status != "pending":
                        statuses[status] += 1
                        pending = None
                letter = None
                if steps % 5 == 0:
                    image, state = capture(env, driver)
                    letter = teacher_recommendation(driver)
                    name = f"map{seed:03d}-step{steps:04d}.jpg"
                    image.save(output/name, quality=90)
                    labels.append(dict(file=name, label=letter, map_seed=seed, step=steps, **state))
                if pending is None and now >= next_request:
                    letter = letter or teacher_recommendation(driver)
                    pending = PendingIntent(letter, now, delay, driver.lane)
                    next_request = now+1.5
                _, _, terminated, truncated, info = env.step(driver.act())
                cost += float(info["cost"])
                steps += 1
                if terminated or truncated:
                    break
            episodes.append(dict(seed=seed, steps=steps, cost=cost,
                                 success=bool(info.get("arrive_dest")),
                                 response_status=dict(statuses)))
            driver.close()
            (output/"labels.json").write_text(json.dumps(labels, indent=1))
            print(json.dumps(episodes[-1]), flush=True)
    finally:
        env.close()
    summary = manifest("train", "delayed_rule_camera_teacher", dict(speed=30.0, delay=delay))
    summary.update(seeds=seeds, episodes=episodes, examples=len(labels),
                   counts=dict(Counter(row["label"] for row in labels)),
                   teacher_query="Every captured frame, independent of execution timer",
                   teacher_privileges="Exact navigation geometry and simulator objects within 50 m",
                   input="front/rear RGB and declared ego/navigation state")
    (output/"manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def generate(seeds, output, max_steps=1000):
    from driving.safemetadrive.camera_planner import INTENTS, LETTERS, camera_config, capture
    from driving.safemetadrive.policies import LaneCautionDriver
    from driving.safemetadrive.protocol import environment_config, manifest
    from metadrive import SafeMetaDriveEnv

    if any(not 0 <= seed < 100 for seed in seeds):
        raise ValueError("Camera data collection is restricted to training maps")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = SafeMetaDriveEnv(camera_config(environment_config("train")))
    labels = []
    episodes = []
    try:
        for seed in seeds:
            env.reset(seed=seed)
            driver = LaneCautionDriver(env.agent, speed=30.0)
            episode_cost, steps = 0.0, 0
            while steps < max_steps:
                if steps % 5 == 0:
                    image, state = capture(env, driver)
                    previous_lane = driver.lane
                    action = driver.act()
                    lanes = env.agent.navigation.current_ref_lanes
                    delta = lanes.index(driver.lane)-lanes.index(previous_lane) if previous_lane in lanes else 0
                    if delta in (-1, 0, 1):
                        label = LETTERS[INTENTS.index((delta, driver.last_caution_label))]
                        name = f"map{seed:03d}-step{steps:04d}.jpg"
                        image.save(output / name, quality=90)
                        labels.append(dict(file=name, label=label, map_seed=seed, step=steps, **state))
                else:
                    action = driver.act()
                _, _, terminated, truncated, info = env.step(action)
                episode_cost += info["cost"]
                steps += 1
                if terminated or truncated:
                    break
            episodes.append(dict(seed=seed, steps=steps, cost=episode_cost,
                                 success=bool(info.get("arrive_dest"))))
            driver.close()
            (output / "labels.json").write_text(json.dumps(labels, indent=1))
            print(json.dumps(episodes[-1]), flush=True)
    finally:
        env.close()
    summary = manifest("train", "lane_caution_rule_camera_teacher", dict(speed=30.0))
    summary.update(seeds=seeds, episodes=episodes, examples=len(labels),
                   counts=dict(Counter(x["label"] for x in labels)),
                   input="front and rear RGB, ego speed, current/committed lane, route lane count",
                   teacher_privileges="navigation geometry and simulator objects within 50 m; labels may contain hidden information")
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary
