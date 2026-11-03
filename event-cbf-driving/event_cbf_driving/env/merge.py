"""Merge scenario: ego on highway with an on-ramp vehicle merging in.

Uses the same point-mass / lane kinematics and stack API as Two-Trap.
Optionally wraps highway-env merge-v0 observation for logging compatibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from event_cbf_driving import DT
from event_cbf_driving.env.skill_lane import LANE_Y, TARGET_SPEED
from event_cbf_driving.env.types import (
    ACTION_HIGH,
    ACTION_LOW,
    VehicleState,
    WorldSnapshot,
    collide,
    integrate_point_mass,
)


class MergeScenarioEnv(gym.Env):
    """Highway merge: slow traffic ahead + merging vehicle from on-ramp (y~-4)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        route_length: float = 180.0,
        duration: float = 14.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.route_length = float(route_length)
        self.duration = float(duration)
        self.max_steps = int(duration / DT)
        self.observation_space = spaces.Box(
            low=np.array([-20.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 10.0, 40.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._step_i = 0
        self._ego = VehicleState(0.0, LANE_Y[0], TARGET_SPEED)
        self._others: List[VehicleState] = []
        self._crashed = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_i = 0
        self._crashed = False
        self._ego = VehicleState(x=0.0, y=LANE_Y[0], vx=TARGET_SPEED, vy=0.0)
        # Lead vehicle in right lane (slow).
        lead = VehicleState(x=55.0, y=LANE_Y[0], vx=12.0, lane_id=0, speed_limit=12.0)
        # Left-lane traffic.
        left = VehicleState(x=40.0, y=LANE_Y[1], vx=18.0, lane_id=1, speed_limit=18.0)
        left2 = VehicleState(x=70.0, y=LANE_Y[1], vx=17.0, lane_id=1, speed_limit=17.0)
        # On-ramp merger (approaches right lane).
        merger = VehicleState(x=30.0, y=-3.5, vx=14.0, vy=0.6, lane_id=-1, speed_limit=14.0)
        self._others = [lead, left, left2, merger]
        return self._obs(), {"scenario": "merge"}

    def _obs(self) -> np.ndarray:
        e = self._ego
        return np.array([e.y, e.vy, e.vx], dtype=np.float32)

    def _propagate_others(self):
        new = []
        for o in self._others:
            ax, ay = 0.0, 0.0
            # Merger steers toward right lane then holds.
            if o.lane_id == -1:
                if o.y < LANE_Y[0] - 0.3:
                    ay = 1.2
                else:
                    ay = -0.3 * o.vy
                    o.lane_id = 0
                    o.y = min(o.y, LANE_Y[0] + 0.2)
            # Mild speed regulation.
            ax = float(np.clip(0.4 * (o.speed_limit - o.vx), -2.0, 1.5))
            o2 = integrate_point_mass(o, ax, ay, DT)
            if o.lane_id == 0 and o.speed_limit > 0.1 and abs(o2.y - LANE_Y[0]) > 0.01 and o.lane_id != -1:
                pass
            new.append(o2)
        self._others = new

    def step(self, action: np.ndarray):
        ax, ay = float(action[0]), float(action[1])
        self._ego = integrate_point_mass(self._ego, ax, ay, DT)
        # Soft lane bounds.
        self._ego.y = float(np.clip(self._ego.y, -1.0, 5.5))
        self._propagate_others()
        self._step_i += 1
        for o in self._others:
            if collide(self._ego, o):
                self._crashed = True
                break
        terminated = self._crashed or self._ego.x >= self.route_length
        truncated = self._step_i >= self.max_steps
        info = {
            "crashed": self._crashed,
            "route_pct": 100.0 * min(self._ego.x, self.route_length) / self.route_length,
            "snapshot": self.snapshot(),
        }
        return self._obs(), 0.0 if self._crashed else 1.0, terminated, truncated, info

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            t=self._step_i * DT,
            ego=self._ego,
            others=list(self._others),
            route_length=self.route_length,
            done=self._crashed or self._ego.x >= self.route_length or self._step_i >= self.max_steps,
            crashed=self._crashed,
        )
