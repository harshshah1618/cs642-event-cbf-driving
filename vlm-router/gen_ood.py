"""Harder, out-of-distribution test frames for the router (front camera). Never used for training.

usage: python gen_ood.py --cat curve --out DIR --n 50 --seed 1 [--view camera|topdown]

Cars are placed in lane coordinates (distance along the road), so the same code works on straight and curved roads.
Labels use the same rule as training (gen_frames.rule). Categories:
  curve       the 6 training scene kinds, on curved roads (SCSCS maps, varying curvature/direction)
  lanechange  ego partway between lanes, angled toward the other lane (mid-overtake / mid-return)
  oncoming    the 6 kinds plus 1-3 cars in the opposite-direction lanes (must be ignored)
  far         car in the RIGHT lane 75-130 m ahead (ego right -> B) or 55-110 m (ego left -> B)
  close       car in the RIGHT lane only 3-8 m ahead of us (ego right) or 1-5 m (ego left): partly out of view
  beside      'blocked' with the LEFT-lane car beside us (-5..+5 m): invisible to a front camera, label A
  multi       2-4 cars at random lanes/distances (ego in either lane)
"""
import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
if os.environ.get("P3D_DISPLAY"):
    from panda3d.core import loadPrcFileData

    loadPrcFileData("", "load-display " + os.environ["P3D_DISPLAY"])
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from metadrive.component.lane.circular_lane import CircularLane  # noqa: E402
from metadrive.component.pgblock.first_block import FirstPGBlock  # noqa: E402
from metadrive.envs.metadrive_env import MetaDriveEnv  # noqa: E402

from gen_frames import CARS, KIND_LABEL, KINDS, L_LEFT, L_RIGHT, rule, scene  # noqa: E402

CATS = ["curve", "lanechange", "oncoming", "far", "close", "beside", "multi"]
LEFT, RIGHT = 0, 1  # lane index: 0 = next to the centre line (LEFT), 1 = RIGHT


class Track:
    """Lanes along the ego route, indexed by distance s measured along the RIGHT lane."""

    def __init__(self, env):
        g = env.current_map.road_network.graph
        cps = env.agent.navigation.checkpoints
        self.segs, self.opp, self.start = [], [], []
        s = 0.0
        for a, b in zip(cps[:-1], cps[1:]):
            lanes = g[a][b]
            if len(lanes) < 2:
                continue
            opp = g.get("-" + b, {}).get("-" + a)
            self.segs.append((lanes[0], lanes[1]))
            self.opp.append(opp)
            self.start.append(s)
            s += lanes[1].length
        self.length = s

    def _seg(self, s):
        i = max(0, int(np.searchsorted(self.start, s, side="right")) - 1)
        f = (s - self.start[i]) / self.segs[i][1].length
        return i, min(max(f, 0.0), 1.0)

    def pose(self, k, s, lat=0.0):
        i, f = self._seg(s)
        lane = self.segs[i][k]
        lon = f * lane.length
        p = lane.position(lon, lat)
        return (float(p[0]), float(p[1])), float(lane.heading_theta_at(lon))

    def opp_pose(self, k, s):
        i, f = self._seg(s)
        if not self.opp[i]:
            return None
        lane = self.opp[i][k]
        lon = (1 - f) * lane.length
        p = lane.position(lon, 0.0)
        return (float(p[0]), float(p[1])), float(lane.heading_theta_at(lon))

    def curve_starts(self):
        return [self.start[i] for i, (l0, l1) in enumerate(self.segs) if isinstance(l1, CircularLane)]


def lane_idx(y):
    return LEFT if y == L_LEFT else RIGHT


def make_scene(cat, rng, cam=True):
    """Returns dict(kind, ego (lane, lat, dheading, skill), cars [(lane, dx)], opp [(lane, dx)])."""
    u = rng.uniform
    kind = KINDS[rng.integers(len(KINDS))]
    ego_lat, dh, skill, opp = 0.0, 0.0, None, []
    if cat in ("curve", "oncoming"):
        ego_y, obs = scene(kind, rng, 1, cam)
        cars = [(lane_idx(ly), dx) for dx, ly in obs]
        if cat == "oncoming":
            opp = [(int(rng.integers(2)), u(10, 90)) for _ in range(rng.integers(1, 4))]
    elif cat == "lanechange":
        ego_y, obs = scene(kind, rng, 1, cam)
        cars = [(lane_idx(ly), dx) for dx, ly in obs]
        # ego still nearer its own lane centre but moving toward the other lane:
        # right lane -> mid-overtake (heading left), left lane -> mid-return (heading right)
        toward_left = ego_y == L_RIGHT
        mag = u(0.6, 1.6)
        ego_lat = -mag if toward_left else mag  # MetaDrive lateral: positive = to the right of the lane
        dh = u(0.05, 0.2) * (1 if toward_left else -1)
        skill = "OVERTAKE_LEFT" if toward_left else "RETURN_RIGHT"
    elif cat == "far":
        if rng.random() < 0.5:
            kind, ego_y, cars = "overtake", L_RIGHT, [(RIGHT, u(75, 130))]
        else:
            kind, ego_y, cars = "keep_left", L_LEFT, [(RIGHT, u(55, 110))]
    elif cat == "close":
        if rng.random() < 0.5:
            kind, ego_y, cars = "overtake", L_RIGHT, [(RIGHT, u(6, 9))]
        else:
            kind, ego_y, cars = "keep_left", L_LEFT, [(RIGHT, u(1, 5))]
    elif cat == "beside":
        kind, ego_y, cars = "blocked", L_RIGHT, [(RIGHT, u(20, 60)), (LEFT, u(-5, 5))]
    elif cat == "multi":
        kind = "multi"
        ego_y = L_RIGHT if rng.random() < 0.5 else L_LEFT
        cars = []
        for _ in range(rng.integers(2, 5)):
            k = int(rng.integers(2))
            dx = u(8, 70) if k == LEFT else (u(10, 70) if rng.random() < 0.6 else u(-40, -10))
            if all(abs(dx - d) > 9 for kk, d in cars if kk == k):
                cars.append((k, dx))
    else:
        raise KeyError(cat)
    ego_lane = "LEFT" if ego_y == L_LEFT else "RIGHT"
    skill = skill or ("CRUISE_RIGHT" if ego_lane == "RIGHT" else "OVERTAKE_LEFT")
    return dict(kind=kind, ego_lane=ego_lane, ego_lat=ego_lat, dh=dh, skill=skill, cars=cars, opp=opp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", choices=CATS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", choices=["camera", "topdown"], default="camera")
    ap.add_argument("--flip", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    cam = a.view == "camera"
    curvy = a.cat == "curve"
    cfg = dict(use_render=False, traffic_density=0.0, num_scenarios=50 if curvy else 1, start_seed=1000 if curvy else 0,
               log_level=50, map_config=dict(type="block_sequence", config="SCSCS" if curvy else "SSSS", lane_num=2),
               random_spawn_lane_index=False,
               vehicle_config=dict(spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1)))
    if cam:
        from metadrive.component.sensors.rgb_camera import RGBCamera
        cfg.update(image_observation=True, norm_pixel=False, stack_size=1, sensors={"rgb_camera": (RGBCamera, 640, 360)})
        cfg["vehicle_config"]["image_source"] = "rgb_camera"
    env = MetaDriveEnv(cfg)

    labels, need_reset, since, track = [], True, 0, None
    for i in range(a.n):
        if need_reset or since >= (10 if curvy else 40):
            env.reset(seed=int(1000 + rng.integers(50)) if curvy else None)
            track, need_reset, since = Track(env), False, 0
        since += 1
        sc = make_scene(a.cat, rng, cam)
        ego_k = LEFT if sc["ego_lane"] == "LEFT" else RIGHT
        if curvy:
            c = track.curve_starts()
            s0 = float(rng.choice(c)) + rng.uniform(-25, 30) if c else rng.uniform(60, 120)
        else:
            s0 = rng.uniform(55, 80)
        s0 = float(np.clip(s0, 45, track.length - 140))
        (x, y), h = track.pose(ego_k, s0, sc["ego_lat"])
        env.agent.set_position((x, y))
        env.agent.set_heading_theta(h + sc["dh"])
        env.agent.set_velocity([1.0, 0.0], 0.0)
        spawned = []
        for k, dx in sc["cars"]:
            (cx, cy), ch = track.pose(k, s0 + dx, float(rng.uniform(-0.3, 0.3)))
            spawned.append(env.engine.spawn_object(CARS[rng.integers(len(CARS))], vehicle_config={}, position=(cx, cy),
                                                   heading=ch))
        for k, dx in sc["opp"]:
            p = track.opp_pose(k, s0 + dx)
            if p:
                spawned.append(env.engine.spawn_object(CARS[rng.integers(len(CARS))], vehicle_config={},
                                                       position=p[0], heading=p[1]))
        for _ in range(2):
            o, _, term, trunc, _ = env.step([0.0, 0.0])
            need_reset = need_reset or term or trunc
        if cam:
            img = np.asarray(o["image"][..., -1]).astype(np.uint8)
            img = img[..., ::-1] if a.flip else img
        else:
            W, H, S = 600, 1200, 6
            full = np.asarray(env.render(mode="topdown", window=False, screen_size=(W, H), scaling=S,
                                         target_agent_heading_up=True)).astype(np.uint8)
            img = full[H // 2 - 90 * S: H // 2 + 40 * S, W // 2 - 7 * S: W // 2 + 13 * S]
        name = f"{a.cat}_{i}.png"
        Image.fromarray(np.ascontiguousarray(img)).save(os.path.join(a.out, name))
        right_ahead = any(k == RIGHT and dx > 0 for k, dx in sc["cars"])
        left_car = any(k == LEFT for k, dx in sc["cars"])
        label = rule(sc["ego_lane"], right_ahead, left_car)
        if sc["kind"] in KIND_LABEL and a.cat not in ("multi",):
            assert label == KIND_LABEL[sc["kind"]], (a.cat, sc)
        labels.append(dict(file=name, cat=a.cat, kind=sc["kind"], label=label, ego_lane=sc["ego_lane"],
                           current_skill=sc["skill"], right_ahead=right_ahead, left_car=left_car,
                           ego_lat=round(sc["ego_lat"], 2), ego_dheading=round(sc["dh"], 3),
                           obstacles=[dict(dx=round(float(dx), 1), lane="LEFT" if k == LEFT else "RIGHT")
                                      for k, dx in sc["cars"]],
                           oncoming=[dict(dx=round(float(dx), 1)) for _, dx in sc["opp"]]))
        if spawned:
            env.engine.clear_objects([v.id for v in spawned])
    json.dump(labels, open(os.path.join(a.out, "labels.json"), "w"), indent=1)
    print(f"{a.cat}: {len(labels)} frames", flush=True)
    env.close()


if __name__ == "__main__":
    main()
