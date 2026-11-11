"""Two-Trap Gauntlet: stalled lane-blocker + cut-in during VLM delay.

Same stack API as merge/dense — not a one-off script.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


class TwoTrapGauntletEnv(gym.Env):
    """Mechanism-isolation scenario.

    Trap 1: stalled car in right lane at x=75.
    Trap 2: after ego passes x=92, a cut-in appears ~23 m ahead in the left lane
    at 9.5 m/s — timed to hit during a typical 300 ms VLM delay if the skill
    switches hard without inflation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        route_length: float = 200.0,
        duration: float = 12.0,
        seed: Optional[int] = None,
        randomize: bool = False,
    ):
        """randomize=False keeps the fixed M1 layout (stalled car at x=75, cut-in 17 m later, 23 m gap, 9.5 m/s).
        randomize=True draws that layout from the reset seed, so different seeds give different episodes."""
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
        self._cut_in_spawned = False
        self._crashed = False
        self.randomize = bool(randomize)
        self._layout = dict(stalled_x=75.0, cut_trigger=17.0, cut_gap=23.0, cut_speed=9.5)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_i = 0
        self._crashed = False
        self._cut_in_spawned = False
        self._ego = VehicleState(x=0.0, y=LANE_Y[0], vx=TARGET_SPEED, vy=0.0)
        if self.randomize:
            u = self._rng.uniform
            self._layout = dict(stalled_x=u(60.0, 90.0), cut_trigger=u(12.0, 22.0), cut_gap=u(18.0, 28.0),
                                cut_speed=u(8.0, 11.0))
        stalled = VehicleState(
            x=self._layout["stalled_x"], y=LANE_Y[0], vx=0.0, vy=0.0, speed_limit=0.01, lane_id=0
        )
        self._others = [stalled]
        return self._obs(), {"scenario": "two_trap", "layout": dict(self._layout)}

    def _obs(self) -> np.ndarray:
        e = self._ego
        return np.array([e.y, e.vy, e.vx], dtype=np.float32)

    def _maybe_spawn_cut_in(self):
        if self._cut_in_spawned:
            return
        L = self._layout
        if self._ego.x >= L["stalled_x"] + L["cut_trigger"]:
            cut = VehicleState(
                x=self._ego.x + L["cut_gap"],
                y=LANE_Y[1],
                vx=L["cut_speed"],
                vy=0.0,
                lane_id=1,
                speed_limit=L["cut_speed"],
            )
            self._others.append(cut)
            self._cut_in_spawned = True

    def _propagate_others(self):
        new = []
        for o in self._others:
            # Constant velocity (stalled stays put; cut-in holds 9.5 m/s).
            o2 = integrate_point_mass(o, ax=0.0, ay=0.0, dt=DT, vx_min=0.0)
            # Keep stalled at exactly 0.
            if o.speed_limit < 0.1 and o.vx < 0.1:
                o2.vx = 0.0
                o2.x = o.x
            new.append(o2)
        self._others = new

    def step(self, action: np.ndarray):
        ax, ay = float(action[0]), float(action[1])
        self._ego = integrate_point_mass(self._ego, ax, ay, DT)
        self._propagate_others()
        self._maybe_spawn_cut_in()
        self._step_i += 1

        for o in self._others:
            if collide(self._ego, o):
                self._crashed = True
                break

        terminated = self._crashed or self._ego.x >= self.route_length
        truncated = self._step_i >= self.max_steps
        reward = 0.0 if self._crashed else 1.0
        info = {
            "crashed": self._crashed,
            "route_pct": 100.0 * min(self._ego.x, self.route_length) / self.route_length,
            "snapshot": self.snapshot(),
        }
        return self._obs(), reward, terminated, truncated, info

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            t=self._step_i * DT,
            ego=self._ego,
            others=list(self._others),
            route_length=self.route_length,
            done=self._crashed or self._ego.x >= self.route_length or self._step_i >= self.max_steps,
            crashed=self._crashed,
            info={"cut_in_spawned": self._cut_in_spawned, "layout": dict(self._layout)},
        )
