"""Render camera-only planner training examples along training-map trajectories."""

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("P3D_DISPLAY", "p3headlessgl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
