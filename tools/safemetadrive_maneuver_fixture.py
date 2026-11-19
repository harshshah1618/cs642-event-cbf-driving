"""Training-map fault injection for a newly blocked, unfinished lane change.

This is a controlled physics fixture, not a native benchmark episode. It places
the car partway through an admitted maneuver, then introduces physical cones.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driving.safemetadrive.policies import LaneCautionDriver
from driving.safemetadrive.protocol import environment_config


def fixture(managed=True, blocked_source=False, checkpoint=None, seed=0):
    if not 0 <= seed < 100:
        raise ValueError("Fault injection is restricted to training maps")
    from metadrive import SafeMetaDriveEnv
    from metadrive.component.static_object.traffic_object import TrafficCone

    config = environment_config("train")
    config.update(traffic_density=0.0, accident_prob=0.0)
    env = SafeMetaDriveEnv(config)
    objects = []
    trace = []
    driver = None
    try:
        env.reset(seed=seed)
        driver = LaneCautionDriver(
            env.agent, speed=30, planner="external", manage_maneuvers=managed
        )
        if checkpoint:
            from tools.safemetadrive_replay import SavedTracker

            driver.skill = SavedTracker(checkpoint)
        lanes = driver.synchronize_lane()
        source = driver.lane
        source_index = lanes.index(source)
        delta = -1 if source_index else 1
        target = lanes[source_index + delta]
        assert driver.apply_intent(delta, False, source)
        distance = source.local_coordinates(env.agent.position)[0] + 10
        start = 0.65 * source.position(distance, 0) + 0.35 * target.position(distance, 0)
        env.agent.set_position(start)
        env.agent.set_heading_theta(source.heading_theta_at(distance))
        env.agent.set_velocity([1, 0], 0)
        obstacle = env.engine.spawn_object(
            TrafficCone,
            position=target.position(distance + 7, 0),
            heading_theta=target.heading_theta_at(distance),
            lane=target,
            static=True,
            random_seed=101,
        )
        objects.append(obstacle.id)
        source_obstacle = None
        if blocked_source:
            source_obstacle = env.engine.spawn_object(
                TrafficCone,
                position=source.position(distance + 6, 0),
                heading_theta=source.heading_theta_at(distance),
                lane=source,
                static=True,
                random_seed=102,
            )
            objects.append(source_obstacle.id)
        # Settle the manually introduced physics state before issuing controls.
        env.step([0, 0])
        cost = 0.0
        for tick in range(160):
            if tick == 60 and source_obstacle is not None:
                object_id = source_obstacle.id
                env.engine.clear_objects([object_id], force_destroy=True)
                objects.remove(object_id)
                source_obstacle = None
            action = driver.act()
            assert driver.last_features is not None
            _, _, terminated, truncated, info = env.step(action)
            cost += float(info["cost"])
            trace.append(
                dict(
                    t=tick * 0.1,
                    speed=env.agent.speed,
                    target_lane=lanes.index(driver.lane) if driver.lane in lanes else None,
                    source_offset=abs(source.local_coordinates(env.agent.position)[1]),
                    progress=source.local_coordinates(env.agent.position)[0] - distance,
                    phase=driver.maneuvers.phase if driver.maneuvers else "legacy",
                    desired_speed=float(driver.last_features[2]) * 15,
                    cost=float(info["cost"]),
                )
            )
            if terminated or truncated:
                break
            if (
                driver.maneuvers is not None
                and driver.maneuvers.completed_count
                and trace[-1]["progress"] > 5
                and trace[-1]["source_offset"] < 0.35
            ):
                break
        events = list(driver.maneuvers.events) if driver.maneuvers else []
        return dict(
            managed=managed,
            source_blocked_until_seconds=6 if blocked_source else None,
            seed=seed,
            trained_tracker=bool(checkpoint),
            cost=cost,
            final_source_offset=trace[-1]["source_offset"],
            final_progress=trace[-1]["progress"],
            aborts=driver.maneuvers.abort_count if driver.maneuvers else 0,
            completions=driver.maneuvers.completed_count if driver.maneuvers else 0,
            events=events,
            trace=trace,
        )
    finally:
        try:
            if driver:
                driver.close()
            if objects:
                env.engine.clear_objects(objects, force_destroy=True)
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="runs/safemetadrive/maneuver-fixtures.json")
    args = parser.parse_args()
    rows = [
        fixture(managed, blocked, args.checkpoint)
        for managed in (False, True)
        for blocked in (False, True)
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
