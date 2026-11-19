"""Goal-conditioned lane/speed skill training on traffic-free training maps.

The observation has lane-relative preview, speed, target speed, lateral error,
heading error and curvature. It contains no traffic information. Tracking
rewards differ from the benchmark reward and are only used for skill training.
"""

from typing import Any

import gymnasium as gym
import numpy as np


class LaneSpeedSkillEnv(gym.Env):
    def __init__(self, seed=0):
        from metadrive import SafeMetaDriveEnv

        from driving.safemetadrive.protocol import environment_config

        config = environment_config("train")
        config.update(traffic_density=0.0, accident_prob=0.0)
        self.native = SafeMetaDriveEnv(config)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.driver: Any = None
        self.tick = 0

    def observation(self):
        self.driver.act()
        return self.driver.last_features.copy()

    def teacher_action(self):
        return self.driver.last_teacher_action.copy()

    def reset(self, *, seed=None, options=None):
        from driving.safemetadrive.policies import LaneCautionDriver

        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.driver is not None:
            self.driver.close()
        self.native.reset(seed=int(self.rng.integers(0, 100)))
        self.driver = LaneCautionDriver(self.native.agent, speed=float(self.rng.uniform(15, 40)),
                                       planner="external", prediction=False)
        self.tick = 0
        return self.observation(), {}

    def step(self, action):
        _, native_reward, terminated, truncated, info = self.native.step(action)
        self.tick += 1
        if self.tick % 100 == 0:
            self.driver.lane = self.rng.choice(self.native.agent.navigation.current_ref_lanes)
            self.driver.speed = float(self.rng.uniform(15, 40))
        obs = self.observation()
        reward = 1.0 + 0.05*float(native_reward) - 0.2*float(obs[3]*5)**2
        reward -= 0.04*float((obs[1]-obs[2])*15)**2 + 0.5*float(obs[4])**2
        if info.get("out_of_road"):
            reward -= 30
        if info.get("arrive_dest"):
            reward += 20
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None
        self.native.close()
