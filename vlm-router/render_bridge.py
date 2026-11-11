"""Render bridge: draw a point-mass world state as one MetaDrive front-camera frame.

Same road, camera and vehicle models as the training frames (gen_frames.py). The ego car is placed at a fixed
x on the straight SSSS map and every other car at its distance along the road, so the point-mass sims never need
MetaDrive physics. No reset per call (reset takes ~4 s); we only move the ego and respawn the other cars.

scene = dict(ego_y, ego_heading, cars=[dict(dx, y, heading, model)])   (MetaDrive metres; LEFT lane y=3.5)
"""
import io
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
if os.environ.get("P3D_DISPLAY"):
    from panda3d.core import loadPrcFileData

    loadPrcFileData("", "load-display " + os.environ["P3D_DISPLAY"])
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from metadrive.component.pgblock.first_block import FirstPGBlock  # noqa: E402
from metadrive.component.sensors.rgb_camera import RGBCamera  # noqa: E402
from metadrive.envs.metadrive_env import MetaDriveEnv  # noqa: E402

from gen_frames import CARS  # noqa: E402

EGO_X = 60.0


class Bridge:
    def __init__(self, reset_every: int = 40):
        cfg = dict(use_render=False, traffic_density=0.0, num_scenarios=1, start_seed=0, log_level=50,
                   map_config=dict(type="block_sequence", config="SSSS", lane_num=2), random_spawn_lane_index=False,
                   vehicle_config=dict(spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1),
                                       image_source="rgb_camera"),
                   image_observation=True, norm_pixel=False, stack_size=1,
                   sensors={"rgb_camera": (RGBCamera, 640, 360)})
        self.env = MetaDriveEnv(cfg)
        self.reset_every = reset_every
        self.need_reset, self.since = True, 0
        self.road_end = None

    def _reset(self):
        self.env.reset()
        self.need_reset, self.since = False, 0
        if self.road_end is None:
            from gen_ood import Track

            self.road_end = Track(self.env).length - 5.0

    def render(self, scene):
        """-> RGB uint8 array (360, 640, 3)."""
        if self.need_reset or self.since >= self.reset_every:
            self._reset()
        self.since += 1
        env = self.env
        env.agent.set_position((EGO_X, scene["ego_y"]))
        env.agent.set_heading_theta(scene["ego_heading"])
        env.agent.set_velocity([1.0, 0.0], 0.0)
        spawned = []
        for c in scene["cars"]:
            x = EGO_X + c["dx"]
            if not (3.0 <= x <= self.road_end):
                continue
            spawned.append(env.engine.spawn_object(CARS[c["model"] % len(CARS)], vehicle_config={},
                                                   position=(x, c["y"]), heading=c["heading"]))
        for _ in range(2):
            o, _, term, trunc, _ = env.step([0.0, 0.0])
            self.need_reset = self.need_reset or term or trunc
        img = np.ascontiguousarray(np.asarray(o["image"][..., -1]).astype(np.uint8)[..., ::-1])  # BGR -> RGB
        if spawned:
            env.engine.clear_objects([v.id for v in spawned])
        return img

    def render_png(self, scene):
        b = io.BytesIO()
        Image.fromarray(self.render(scene)).save(b, format="PNG")
        return b.getvalue()

    def close(self):
        self.env.close()
