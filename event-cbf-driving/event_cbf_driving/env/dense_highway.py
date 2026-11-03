"""Dense highway traffic scenario (multi-vehicle, same stack API).

Primary training/eval path uses shared kinematics. A highway-env factory is
also exposed for users who want the upstream MultiAgentRoad env directly.
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


class DenseHighwayEnv(gym.Env):
    """Dense two-lane traffic with several slower vehicles and a left-lane pack."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        route_length: float = 220.0,
        duration: float = 16.0,
        n_vehicles: int = 8,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.route_length = float(route_length)
        self.duration = float(duration)
        self.n_vehicles = int(n_vehicles)
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
        others: List[VehicleState] = []
        # Right-lane pack (slower, creates need to overtake).
        xs_right = [45.0, 75.0, 110.0, 145.0]
        for i, x in enumerate(xs_right):
            v = 11.0 + 1.5 * (i % 2)
            others.append(VehicleState(x=x, y=LANE_Y[0], vx=v, lane_id=0, speed_limit=v))
        # Left-lane denser faster pack with one slow poke.
        xs_left = [35.0, 60.0, 95.0, 130.0]
        speeds = [18.0, 10.5, 17.5, 16.0]  # 10.5 is a left-lane trap if you linger
        for x, v in zip(xs_left, speeds):
            others.append(VehicleState(x=x, y=LANE_Y[1], vx=v, lane_id=1, speed_limit=v))
        self._others = others[: self.n_vehicles]
        return self._obs(), {"scenario": "dense_highway"}

    def _obs(self) -> np.ndarray:
        e = self._ego
        return np.array([e.y, e.vy, e.vx], dtype=np.float32)

    def _propagate_others(self):
        new = []
        for o in self._others:
            ax = float(np.clip(0.5 * (o.speed_limit - o.vx), -2.5, 1.5))
            # Keep lane.
            ay = -1.5 * (o.y - LANE_Y[o.lane_id]) - 0.8 * o.vy
            new.append(integrate_point_mass(o, ax, ay, DT))
        self._others = new

    def step(self, action: np.ndarray):
        ax, ay = float(action[0]), float(action[1])
        self._ego = integrate_point_mass(self._ego, ax, ay, DT)
        self._ego.y = float(np.clip(self._ego.y, -0.5, 5.0))
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


def make_highway_env_continuous(vehicles_count: int = 20, duration: int = 30):
    """Optional upstream highway-env factory (ContinuousAction, 50 Hz)."""
    import gymnasium as gym

    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure(
        {
            "observation": {"type": "Kinematics", "vehicles_count": 5},
            "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
            "lanes_count": 2,
            "vehicles_count": vehicles_count,
            "duration": duration,
            "policy_frequency": 50,
            "simulation_frequency": 50,
            "initial_spacing": 2.0,
        }
    )
    return env
