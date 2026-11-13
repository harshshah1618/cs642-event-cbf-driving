"""Empty-road skill training env (highway-env compatible kinematics, 50 Hz)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from driving import DT
from driving.env_base import (
    detached_snapshot,
    require_non_negative,
    require_positive,
    validate_action,
)
from driving.env_types import (
    ACTION_HIGH,
    ACTION_LOW,
    VehicleState,
    WorldSnapshot,
    integrate_point_mass,
)
from driving.errors import ValidationError

LANE_Y = {0: 0.0, 1: 4.0}  # right, left
TARGET_SPEED = 20.0


class SkillLaneEnv(gym.Env):
    """Train a single lane-keeping / lane-acquire skill on an empty two-lane road.

    Observation: [y, vy, vx]. Action: [ax, ay] m/s^2.
    Other vehicles are absent during skill training (skills are lane regulators).

    ``target_speed`` is used both for the reset distribution and the reward
    target.  The default random reset distribution is unchanged from the
    original training setup (around the historical 20 m/s target); pass
    ``randomize=False`` for a deterministic reset exactly at the target lane
    and speed.
    """

    metadata = {"render_modes": []}
    SCENARIO_NAME = "skill_lane"
    ROUTE_LENGTH = 1e9

    def __init__(
        self,
        target_lane: int = 0,
        target_speed: float = TARGET_SPEED,
        episode_seconds: float = 8.0,
        seed: Optional[int] = None,
        randomize: bool = True,
    ):
        super().__init__()
        if int(target_lane) not in LANE_Y:
            raise ValidationError(
                f"target_lane: must be one of {sorted(LANE_Y)}, got {target_lane!r}"
            )
        self.target_lane = int(target_lane)
        self.target_y = LANE_Y[self.target_lane]
        self.target_speed = require_non_negative(target_speed, name="target_speed")
        self.episode_seconds = require_positive(episode_seconds, name="episode_seconds")
        self.episode_steps = int(self.episode_seconds / DT)
        self.randomize = bool(randomize)
        self.observation_space = spaces.Box(
            low=np.array([-20.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 10.0, 40.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._step_i = 0
        self._params: Dict[str, float] = {
            "target_lane": float(self.target_lane),
            "target_speed": self.target_speed,
            "y": self.target_y,
            "vx": self.target_speed,
            "vy": 0.0,
        }
        self._ego = VehicleState(0.0, self.target_y, self.target_speed)

    def _sample_params(self) -> Dict[str, float]:
        if not self.randomize:
            return {
                "target_lane": float(self.target_lane),
                "target_speed": self.target_speed,
                "y": self.target_y,
                "vx": self.target_speed,
                "vy": 0.0,
            }
        r = self._rng.random()
        other_y = LANE_Y[1 - self.target_lane]
        # Default target 20 m/s reproduces the original [12, 22] and [0, 8]
        # ranges exactly; other target speeds stay centred on the target.
        cruise_low = max(0.0, self.target_speed - 8.0)
        cruise_high = self.target_speed + 2.0
        slow_high = max(0.0, 0.4 * self.target_speed)
        if r < 0.55:
            y = self.target_y + self._rng.uniform(-1.5, 1.5)
            vx = self._rng.uniform(cruise_low, cruise_high)
        elif r < 0.80:
            y = self.target_y + self._rng.uniform(-1.0, 1.0)
            vx = self._rng.uniform(0.0, slow_high)
        else:
            # Acquire from the other lane (router engage).
            y = other_y + self._rng.uniform(-0.5, 0.5)
            # Sometimes start mid-transition.
            if self._rng.random() < 0.4:
                y = self._rng.uniform(min(self.target_y, other_y), max(self.target_y, other_y))
            vx = self._rng.uniform(cruise_low, cruise_high)
        vy = self._rng.uniform(-1.5, 1.5)
        return {
            "target_lane": float(self.target_lane),
            "target_speed": self.target_speed,
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        self._step_i = 0
        self._params = self._sample_params()
        self._ego = VehicleState(
            x=0.0,
            y=float(self._params["y"]),
            vx=float(self._params["vx"]),
            vy=float(self._params["vy"]),
        )
        snap = self.snapshot()
        info = {
            "scenario": self.scenario,
            "params": dict(self._params),
            "snapshot": snap,
            "crashed": False,
            "route_pct": self._route_pct(),
        }
        return self._obs(), info

    @property
    def scenario(self) -> str:
        return self.SCENARIO_NAME

    def _obs(self) -> np.ndarray:
        e = self._ego
        return np.array([e.y, e.vy, e.vx], dtype=np.float32)

    def _route_pct(self) -> float:
        return 100.0 * min(max(self._ego.x, 0.0), self.ROUTE_LENGTH) / self.ROUTE_LENGTH

    def _snapshot_info(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "params": dict(self._params),
            "sampled_params": dict(self._params),
        }

    def _step_info(self) -> Dict[str, Any]:
        return {
            "snapshot": self.snapshot(),
            "crashed": False,
            "route_pct": self._route_pct(),
            "scenario": self.scenario,
            "params": dict(self._params),
            "ego": self._ego.copy(),
        }

    def step(self, action: np.ndarray):
        action = validate_action(action)
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
        return self._obs(), float(reward), terminated, truncated, self._step_info()

    def snapshot(self) -> WorldSnapshot:
        return detached_snapshot(
            t=self._step_i * DT,
            ego=self._ego,
            others=[],
            route_length=self.ROUTE_LENGTH,
            done=self._step_i >= self.episode_steps,
            crashed=False,
            info=self._snapshot_info(),
        )

    def close(self) -> None:
        """No-op: this environment owns no external resources."""
