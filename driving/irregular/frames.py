"""Render camera frames from SafeMetaDrive episodes, with optional placed objects.

Frames use Harsh's camera setup (front 448x252 over rear 448x126 mosaic) so the
novelty detector sees exactly what the camera planner sees. The ego drives with
the rule planner and protection (no learned weights needed).
"""

import json
import time
from pathlib import Path

from driving.irregular.scene import Placement, make_env, place, placement_record, relative


def collect(seeds, output, split, placements=None, every=5, max_steps=1000,
            stop_after_passing=True):
    """Drive each map once per placement (None = clean) and save frames + ground truth.

    `placements` is a list of Placement or None; each map is driven once per entry.
    """
    from driving.safemetadrive.camera_planner import camera_config, capture
    from driving.safemetadrive.policies import LaneCautionDriver
    from driving.safemetadrive.protocol import SPLITS, environment_config, manifest

    start, count = SPLITS[split]
    if any(not start <= seed < start + count for seed in seeds):
        raise ValueError(f"Seeds outside the {split} partition")
    placements = placements or [None]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_env(camera_config(environment_config(split)))
    rows, episodes = [], []
    started = time.monotonic()
    try:
        for seed in seeds:
            for placement in placements:
                env.reset(seed=seed)
                obj = place(env, placement) if placement else None
                driver = LaneCautionDriver(env.agent, speed=30.0)
                tag = "clean" if placement is None else "-".join(
                    x for x in (placement.kind, placement.face, placement.where) if x)
                cost, steps, frames = 0.0, 0, 0
                info = {}
                while steps < max_steps:
                    truth = relative(env.agent, obj) if obj is not None else None
                    if obj is not None and stop_after_passing and truth["forward"] < -5:
                        break
                    if steps % every == 0:
                        image, state = capture(env, driver)
                        name = f"map{seed:03d}-{tag}-step{steps:04d}.jpg"
                        image.save(output / name, quality=92)
                        rows.append(dict(file=name, map_seed=seed, step=steps, scene=tag,
                                         placement=placement_record(placement) if placement else None,
                                         truth=truth, **state))
                        frames += 1
                    _, _, terminated, truncated, info = env.step(driver.act())
                    cost += float(info["cost"])
                    steps += 1
                    if terminated or truncated:
                        break
                episodes.append(dict(seed=seed, scene=tag, steps=steps, frames=frames, cost=cost,
                                     crash_human=bool(info.get("crash_human", False)),
                                     success=bool(info.get("arrive_dest", False))))
                driver.close()
                print(json.dumps(episodes[-1]), flush=True)
            (output / "labels.json").write_text(json.dumps(rows, indent=1))
    finally:
        env.close()
    summary = manifest(split, "irregular_frames_rule_driver", dict(
        speed=30.0, every=every, max_steps=max_steps,
        placements=[placement_record(p) if p else None for p in placements]))
    summary.update(seeds=list(seeds), episodes=episodes, frames=len(rows),
                   wall_seconds=time.monotonic() - started)
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


# First evaluation set: built-in objects only. Pedestrians and cyclists never occur on
# the clean maps (novel); barriers and warning triangles do (familiar controls).
EVAL_PLACEMENTS = [
    None,
    Placement("pedestrian", 70, "shoulder"),
    Placement("pedestrian", 70, "lane"),
    Placement("cyclist", 70, "shoulder"),
    Placement("cyclist", 70, "lane"),
    Placement("barrier", 70, "lane"),
    Placement("warning", 70, "lane"),
]


# Custom-prop smoke set: flagger (STOP for 6 s, then SLOW), lane-closed sign, harmless board.
SIGN_PLACEMENTS = [
    Placement("flagger", 60, "shoulder", stop_seconds=6.0),
    Placement("sign", 60, "shoulder", face="lane_closed"),
    Placement("sign", 60, "shoulder", face="billboard"),
]

# Custom props on the development maps, same 70 m distance as EVAL_PLACEMENTS, with a
# clean run per map as negatives.
EVAL_SIGN_PLACEMENTS = [
    None,
    Placement("flagger", 70, "shoulder", stop_seconds=8.0),
    Placement("sign", 70, "shoulder", face="lane_closed"),
    Placement("sign", 70, "shoulder", face="billboard"),
]
