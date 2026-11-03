"""Empty-road skill training env (highway-env compatible kinematics, 50 Hz)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from event_cbf_driving import DT
from event_cbf_driving.env.types import (
    ACTION_HIGH,
    ACTION_LOW,
    OBS_DIM,
    VehicleState,
    WorldSnapshot,
    integrate_point_mass,
)

LANE_Y = {0: 0.0, 1: 4.0}  # right, left
TARGET_SPEED = 20.0


class SkillLaneEnv(gym.Env):
    """Train a single lane-keeping / lane-acquire skill on an empty two-lane road.

    Observation: [y, vy, vx]. Action: [ax, ay] m/s^2.
    Other vehicles are absent during skill training (skills are lane regulators).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_lane: int = 0,
        target_speed: float = TARGET_SPEED,
        episode_seconds: float = 8.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.target_lane = int(target_lane)
        self.target_y = LANE_Y[self.target_lane]
        self.target_speed = float(target_speed)
        self.episode_steps = int(episode_seconds / DT)
        self.observation_space = spaces.Box(
            low=np.array([-20.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 10.0, 40.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._step_i = 0
        self._ego = VehicleState(0.0, 0.0, TARGET_SPEED)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_i = 0
        r = self._rng.random()
        other_y = LANE_Y[1 - self.target_lane]
        if r < 0.55:
            y = self.target_y + self._rng.uniform(-1.5, 1.5)
            vx = self._rng.uniform(12.0, 22.0)
        elif r < 0.80:
            y = self.target_y + self._rng.uniform(-1.0, 1.0)
            vx = self._rng.uniform(0.0, 8.0)
        else:
            # Acquire from the other lane (router engage).
            y = other_y + self._rng.uniform(-0.5, 0.5)
            # Sometimes start mid-transition.
            if self._rng.random() < 0.4:
                y = self._rng.uniform(min(self.target_y, other_y), max(self.target_y, other_y))
            vx = self._rng.uniform(12.0, 22.0)
        vy = self._rng.uniform(-1.5, 1.5)
        self._ego = VehicleState(x=0.0, y=float(y), vx=float(vx), vy=float(vy))
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        e = self._ego
        return np.array([e.y, e.vy, e.vx], dtype=np.float32)

    def step(self, action: np.ndarray):
        ax, ay = float(action[0]), float(action[1])
        self._ego = integrate_point_mass(self._ego, ax, ay, DT)
        self._step_i += 1
        e = self._ego
        y_err = e.y - self.target_y
        v_err = e.vx - self.target_speed
        reward = (
            1.0
            - (y_err ** 2)
            - 0.05 * (v_err ** 2)
            - (0.02 * ax * ax + 0.35 * ay * ay)
            - 0.05 * (e.vy ** 2)
        )
        terminated = False
        truncated = self._step_i >= self.episode_steps
        return self._obs(), float(reward), terminated, truncated, {"ego": e}

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(t=self._step_i * DT, ego=self._ego, others=[], route_length=1e9)
