"""Render labelled router-decision frames from MetaDrive (front camera or top-down).

usage: python gen_frames.py --out DIR --kinds clear,overtake --n 50 --seed 7 [--view camera] [--variety 1]

Road: straight, two lanes in our direction. Lane y=3.5 is LEFT, y=0 is RIGHT. Other cars are parked.
The label is the correct router answer, computed by `rule()` from the ground truth:
  ego RIGHT: B (OVERTAKE_LEFT) if a car is ahead in the RIGHT lane and the LEFT lane is free, else A (CRUISE_RIGHT)
  ego LEFT:  B (stay left) if a car is still ahead in the RIGHT lane, else C (RETURN_RIGHT)

Scene kinds (KINDS below):
  clear       ego R, nothing                                   -> A
  left_only   ego R, a car ahead in the LEFT lane only         -> A   (distractor: a car that should NOT trigger overtake)
  overtake    ego R, car ahead in our (RIGHT) lane             -> B
  blocked     ego R, car ahead RIGHT and a car in LEFT lane    -> A   (wait)
  return      ego L, the passed car is behind us on the right  -> C   (variety: sometimes a car far ahead in the LEFT lane)
  keep_left   ego L, car still ahead in the RIGHT lane         -> B

--variety 0 reproduces the round-2 test distribution (5 kinds, DefaultVehicle, no jitter).
--variety 1 widens distances and adds vehicle types, lateral/heading jitter, ego position, and the left_only kind.
The front camera only sees ahead, so a car "in the LEFT lane" is always placed ahead of us.
"""
import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
if os.environ.get("P3D_DISPLAY"):  # force a panda3d display backend: p3headlessgl (EGL) or pandagl (X11, e.g. xvfb)
    from panda3d.core import loadPrcFileData

    loadPrcFileData("", "load-display " + os.environ["P3D_DISPLAY"])
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from metadrive.component.pgblock.first_block import FirstPGBlock  # noqa: E402
from metadrive.component.vehicle.vehicle_type import (DefaultVehicle, LVehicle, MVehicle, SVehicle,  # noqa: E402
                                                      XLVehicle)
from metadrive.envs.metadrive_env import MetaDriveEnv  # noqa: E402

L_LEFT, L_RIGHT = 3.5, 0.0
KINDS = ["clear", "left_only", "overtake", "blocked", "return", "keep_left"]
KIND_LABEL = {"clear": "A", "left_only": "A", "overtake": "B", "blocked": "A", "return": "C", "keep_left": "B"}
CARS = [DefaultVehicle, DefaultVehicle, SVehicle, MVehicle, LVehicle, XLVehicle]


def rule(ego_lane, right_ahead, left_car):
    if ego_lane == "RIGHT":
        return "B" if (right_ahead and not left_car) else "A"
    return "B" if right_ahead else "C"


def scene(kind, rng, variety, cam):
    """Returns (ego_lane_y, [(dx, lane_y), ...]) with dx = metres ahead of us (negative = behind)."""
    u = rng.uniform
    if not variety:  # round-2 distribution
        if kind == "clear":
            return L_RIGHT, []
        if kind == "overtake":
            return L_RIGHT, [(u(20, 60), L_RIGHT)]
        if kind == "blocked":
            return L_RIGHT, [(u(20, 60), L_RIGHT), (u(12, 35) if cam else u(-6, 8), L_LEFT)]
        if kind == "return":
            return L_LEFT, [(u(-35, -12), L_RIGHT)]
        if kind == "keep_left":
            return L_LEFT, [(u(8, 25) if cam else u(3, 20), L_RIGHT)]
        raise KeyError(kind)
    if kind == "clear":
        return L_RIGHT, []
    if kind == "left_only":
        return L_RIGHT, [(u(8, 60), L_LEFT)]
    if kind == "overtake":
        return L_RIGHT, [(u(12, 75), L_RIGHT)]
    if kind == "blocked":
        return L_RIGHT, [(u(15, 65), L_RIGHT), (u(8, 45) if cam else u(-6, 30), L_LEFT)]
    if kind == "return":
        obs = [(u(-40, -8), L_RIGHT)]
        if rng.random() < 0.4:
            obs.append((u(30, 70), L_LEFT))
        return L_LEFT, obs
    if kind == "keep_left":
        obs = [(u(5, 50), L_RIGHT)]
        if rng.random() < 0.3:
            obs.append((u(35, 70), L_LEFT))
        return L_LEFT, obs
    raise KeyError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=["topdown", "camera"], default="camera")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--n", type=int, default=10, help="frames per kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variety", type=int, default=1)
    ap.add_argument("--prefix", default="", help="file-name prefix (lets shards share one folder)")
    ap.add_argument("--flip", type=int, default=1, help="camera only: 1 = convert BGR->RGB")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    cam = a.view == "camera"

    cfg = dict(use_render=False, traffic_density=0.0, num_scenarios=1, start_seed=0, log_level=50,
               map_config=dict(type="block_sequence", config="SSSS", lane_num=2), random_spawn_lane_index=False,
               vehicle_config=dict(spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1)))
    if cam:
        from metadrive.component.sensors.rgb_camera import RGBCamera
        cfg.update(image_observation=True, norm_pixel=False, stack_size=1, sensors={"rgb_camera": (RGBCamera, 640, 360)})
        cfg["vehicle_config"]["image_source"] = "rgb_camera"
    env = MetaDriveEnv(cfg)

    labels = []
    need_reset, since_reset = True, 0
    for kind in a.kinds.split(","):
        for i in range(a.n):
            # reset() rebuilds the scene (~4 s even on a GPU), so only reset occasionally or after the episode ends;
            # otherwise just move the ego car and respawn the parked cars (~0.05 s per frame).
            if need_reset or since_reset >= 40:
                env.reset()
                need_reset, since_reset = False, 0
            since_reset += 1
            ego_y, obs = scene(kind, rng, a.variety, cam)
            ego_x = rng.uniform(45, 75) if a.variety else 60.0
            ego_dy = rng.uniform(-0.35, 0.35) if a.variety else 0.0
            env.agent.set_position((ego_x, ego_y + ego_dy))
            env.agent.set_heading_theta(rng.uniform(-0.03, 0.03) if a.variety else 0.0)
            env.agent.set_velocity([1.0, 0.0], 0.0)
            spawned = []
            for dx, ly in obs:
                car = CARS[rng.integers(len(CARS))] if a.variety else DefaultVehicle
                dy = rng.uniform(-0.3, 0.3) if a.variety else 0.0
                spawned.append(env.engine.spawn_object(car, vehicle_config={}, position=(ego_x + dx, ly + dy),
                                                       heading=float(rng.uniform(-0.05, 0.05)) if a.variety else 0.0))
            for _ in range(2):
                o, _, term, trunc, _ = env.step([0.0, 0.0])
                need_reset = need_reset or term or trunc
            if cam:
                img = np.asarray(o["image"][..., -1]).astype(np.uint8)
                if a.flip:
                    img = img[..., ::-1]
            else:
                W, H, S = 600, 1200, 6
                full = np.asarray(env.render(mode="topdown", window=False, screen_size=(W, H), scaling=S,
                                             target_agent_heading_up=True)).astype(np.uint8)
                cx, cy = W // 2, H // 2
                img = full[cy - 90 * S: cy + 40 * S, cx - 7 * S: cx + 13 * S]
            name = f"{a.prefix}{kind}_{i}.png"
            Image.fromarray(np.ascontiguousarray(img)).save(os.path.join(a.out, name))
            ego_lane = "RIGHT" if ego_y == L_RIGHT else "LEFT"
            right_ahead = any(ly == L_RIGHT and dx > 0 for dx, ly in obs)
            left_car = any(ly == L_LEFT for dx, ly in obs)
            label = rule(ego_lane, right_ahead, left_car)
            assert label == KIND_LABEL[kind], (kind, label, obs)
            labels.append(dict(file=name, kind=kind, label=label, ego_lane=ego_lane,
                               current_skill="CRUISE_RIGHT" if ego_lane == "RIGHT" else "OVERTAKE_LEFT",
                               right_ahead=right_ahead, left_car=left_car,
                               obstacles=[dict(dx=round(float(dx), 1), lane="RIGHT" if ly == L_RIGHT else "LEFT")
                                          for dx, ly in obs]))
            if spawned:
                env.engine.clear_objects([v.id for v in spawned])
        print(f"{a.view}: {kind} x{a.n}", flush=True)

    json.dump(labels, open(os.path.join(a.out, f"{a.prefix}labels.json"), "w"), indent=1)
    env.close()


if __name__ == "__main__":
    main()
