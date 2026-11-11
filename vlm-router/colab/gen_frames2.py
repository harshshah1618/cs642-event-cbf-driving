"""Render labelled router-decision frames from MetaDrive, in two views.

usage: python gen_frames2.py --view topdown|camera --out DIR --n N --seed S

Road: two lanes our direction. lane y=3.5 is LEFT, y=0 is RIGHT. Other cars are parked.
Classes and ground truth (label = correct router answer):
  clear      ego right lane, nothing ahead                         -> A
  overtake   ego right lane, parked car ahead right, left free      -> B
  blocked    ego right lane, car ahead right AND car in left lane   -> A (wait)
  return     ego left lane, right lane ahead clear (car passed)     -> C
  keep_left  ego left lane, car still ahead in right lane           -> B
Perception truth per frame: right_ahead (car ahead in RIGHT lane), left_car (another car in LEFT lane).
Camera only sees ahead, so in the camera view the 'blocked' left-lane car is placed ahead (12-35 m), not beside.
"""
import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
if os.environ.get("P3D_DISPLAY"):  # force a panda3d display backend, e.g. p3headlessgl (EGL) or pandagl (X11)
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "load-display " + os.environ["P3D_DISPLAY"])
import numpy as np
from PIL import Image
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.envs.metadrive_env import MetaDriveEnv

ap = argparse.ArgumentParser()
ap.add_argument("--view", choices=["topdown", "camera"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--flip", type=int, default=1, help="camera only: 1 = convert BGR->RGB")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)

L_LEFT, L_RIGHT, EGO_X = 3.5, 0.0, 60.0
cfg = dict(use_render=False, traffic_density=0.0, num_scenarios=1, start_seed=0, log_level=50,
           map_config=dict(type="block_sequence", config="SSSS", lane_num=2), random_spawn_lane_index=False,
           vehicle_config=dict(spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1)))
if a.view == "camera":
    from metadrive.component.sensors.rgb_camera import RGBCamera
    cfg.update(image_observation=True, norm_pixel=False, stack_size=1,
               sensors={"rgb_camera": (RGBCamera, 640, 360)})
    cfg["vehicle_config"]["image_source"] = "rgb_camera"
env = MetaDriveEnv(cfg)

CLASSES = {"clear": "A", "overtake": "B", "blocked": "A", "return": "C", "keep_left": "B"}


def scene(kind):
    cam = a.view == "camera"
    if kind == "clear":
        return L_RIGHT, []
    if kind == "overtake":
        return L_RIGHT, [(rng.uniform(20, 60), L_RIGHT)]
    if kind == "blocked":
        left_dx = rng.uniform(12, 35) if cam else rng.uniform(-6, 8)
        return L_RIGHT, [(rng.uniform(20, 60), L_RIGHT), (left_dx, L_LEFT)]
    if kind == "return":
        return L_LEFT, [(rng.uniform(-35, -12), L_RIGHT)]
    if kind == "keep_left":
        return L_LEFT, [(rng.uniform(8, 25) if cam else rng.uniform(3, 20), L_RIGHT)]
    raise KeyError(kind)


labels = []
for kind, label in CLASSES.items():
    for i in range(a.n):
        env.reset()
        ego_y, obs = scene(kind)
        env.agent.set_position((EGO_X, ego_y))
        env.agent.set_heading_theta(0.0)
        spawned = [env.engine.spawn_object(DefaultVehicle, vehicle_config={}, position=(EGO_X + dx, ly), heading=0.0)
                   for dx, ly in obs]
        for _ in range(2):
            o, *_ = env.step([0.0, 0.0])
        if a.view == "camera":
            img = np.asarray(o["image"][..., -1]).astype(np.uint8)
            if a.flip:
                img = img[..., ::-1]
        else:
            W, H, S = 600, 1200, 6
            full = np.asarray(env.render(mode="topdown", window=False, screen_size=(W, H), scaling=S,
                                         target_agent_heading_up=True)).astype(np.uint8)
            cx, cy = W // 2, H // 2
            img = full[cy - 90 * S: cy + 40 * S, cx - 7 * S: cx + 13 * S]
        name = f"{kind}_{i}.png"
        Image.fromarray(np.ascontiguousarray(img)).save(os.path.join(a.out, name))
        labels.append(dict(
            file=name, kind=kind, label=label,
            ego_lane="RIGHT" if ego_y == L_RIGHT else "LEFT",
            current_skill="CRUISE_RIGHT" if ego_y == L_RIGHT else "OVERTAKE_LEFT",
            right_ahead=any(ly == L_RIGHT and dx > 0 for dx, ly in obs),
            left_car=any(ly == L_LEFT for dx, ly in obs),
            obstacles=[dict(dx=round(float(dx), 1), lane="RIGHT" if ly == L_RIGHT else "LEFT") for dx, ly in obs]))
        if spawned:
            env.engine.clear_objects([v.id for v in spawned])
    print(f"{a.view}: {kind} x{a.n}", flush=True)

json.dump(labels, open(os.path.join(a.out, "labels.json"), "w"), indent=1)
env.close()
