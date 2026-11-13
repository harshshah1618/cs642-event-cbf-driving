"""Two-Trap Gauntlet: stalled lane-blocker + cut-in during VLM delay.

Same stack API as merge/dense — not a one-off script.  The published baseline
uses the fixed defaults below; randomized variants are opt-in through
``randomize=True`` or an explicit ``params`` mapping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from driving import DT
from driving.env_base import (
    detached_snapshot,
    normalize_range,
    require_positive,
    sample_range,
    validate_action,
    validate_lateral_bounds,
)
from driving.env_types import (
    ACTION_HIGH,
    ACTION_LOW,
    LANE_Y,
    TARGET_SPEED,
    VehicleState,
    WorldSnapshot,
    collide,
    integrate_point_mass,
)
from driving.errors import ValidationError


class TwoTrapGauntletEnv(gym.Env):
    """Mechanism-isolation scenario.

    Trap 1: stalled car in right lane at x=75.
    Trap 2: after ego passes x=92, a cut-in appears ~23 m ahead in the left lane
    at 9.5 m/s — timed to hit during a typical 300 ms VLM delay if the skill
    switches hard without inflation.

    Randomization is off by default.  With ``randomize=True`` the ranges in
    :attr:`RANDOM_RANGES` are sampled from the reset seed; explicit ``params``
    (or the matching keyword arguments) override individual ranges/scalars.
    """

    metadata = {"render_modes": []}
    SCENARIO_NAME = "two_trap"

    DEFAULT_PARAMS: Dict[str, float] = {
        "stalled_x": 75.0,
        "cut_in_gap": 23.0,
        "cut_in_speed": 9.5,
        "spawn_x": 0.0,
        "ego_vx": TARGET_SPEED,
        "cut_in_trigger_x": 92.0,
    }
    RANDOM_RANGES: Dict[str, Tuple[float, float]] = {
        "stalled_x": (70.0, 90.0),
        "cut_in_gap": (18.0, 30.0),
        "cut_in_speed": (8.0, 12.0),
        "spawn_x": (0.0, 12.0),
        "ego_vx": (16.0, 22.0),
        "cut_in_trigger_x": (92.0, 92.0),
    }
    _PARAM_KEYS = {
        "stalled_x": "stalled_x",
        "stalled_x_range": "stalled_x",
        "cut_in_gap": "cut_in_gap",
        "cut_in_gap_range": "cut_in_gap",
        "cut_in_speed": "cut_in_speed",
        "cut_in_speed_range": "cut_in_speed",
        "spawn_x": "spawn_x",
        "spawn_x_range": "spawn_x",
        "ego_vx": "ego_vx",
        "ego_vx_range": "ego_vx",
        "cut_in_trigger_x": "cut_in_trigger_x",
        "cut_in_trigger_x_range": "cut_in_trigger_x",
    }

    def __init__(
        self,
        route_length: float = 200.0,
        duration: float = 12.0,
        seed: Optional[int] = None,
        randomize: bool = False,
        params: Optional[Mapping[str, Any]] = None,
        *,
        stalled_x_range: Optional[Any] = None,
        cut_in_gap_range: Optional[Any] = None,
        cut_in_speed_range: Optional[Any] = None,
        spawn_x_range: Optional[Any] = None,
        ego_vx_range: Optional[Any] = None,
        cut_in_trigger_x: Optional[Any] = None,
        lateral_bounds: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.scenario = self.SCENARIO_NAME
        self.route_length = require_positive(route_length, name="route_length")
        self.duration = require_positive(duration, name="duration")
        self.max_steps = int(self.duration / DT)
        self.lateral_bounds = validate_lateral_bounds(lateral_bounds)
        self.observation_space = spaces.Box(
            low=np.array([-20.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 10.0, 40.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._step_i = 0
        self._ego = VehicleState(
            self.DEFAULT_PARAMS["spawn_x"],
            LANE_Y[0],
            self.DEFAULT_PARAMS["ego_vx"],
        )
        self._others: List[VehicleState] = []
        self._cut_in_spawned = False
        self._crashed = False

        explicit_ranges = {
            "stalled_x": stalled_x_range,
            "cut_in_gap": cut_in_gap_range,
            "cut_in_speed": cut_in_speed_range,
            "spawn_x": spawn_x_range,
            "ego_vx": ego_vx_range,
            "cut_in_trigger_x": cut_in_trigger_x,
        }
        supplied_params = dict(params or {})
        self.randomize = bool(
            randomize
            or supplied_params
            or any(value is not None for value in explicit_ranges.values())
        )
        specs: Dict[str, Tuple[float, float]] = {
            key: (float(value), float(value)) for key, value in self.DEFAULT_PARAMS.items()
        }
        if randomize:
            specs.update(self.RANDOM_RANGES)
        for key, value in explicit_ranges.items():
            if value is not None:
                specs[key] = normalize_range(value, name=f"{key}_range")
        for key, value in supplied_params.items():
            if key not in self._PARAM_KEYS:
                allowed = ", ".join(sorted(self._PARAM_KEYS))
                raise ValidationError(
                    f"two_trap params: unknown key {key!r}; allowed keys are {allowed}"
                )
            canonical = self._PARAM_KEYS[key]
            specs[canonical] = normalize_range(value, name=f"params.{key}")
        self.param_spec: Dict[str, Tuple[float, float]] = dict(specs)
        self._params: Dict[str, float] = dict(self.DEFAULT_PARAMS)

    def _sample_params(self) -> Dict[str, float]:
        # Fixed, documented order keeps sampling reproducible for a given seed.
        sampled = {
            key: sample_range(self._rng, self.param_spec[key])
            for key in (
                "spawn_x",
                "ego_vx",
                "stalled_x",
                "cut_in_gap",
                "cut_in_speed",
                "cut_in_trigger_x",
            )
        }
        if sampled["cut_in_speed"] < 0.0:
            raise ValidationError(
                f"two_trap params.cut_in_speed: must be >= 0, got {sampled['cut_in_speed']}"
            )
        return sampled

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        self._step_i = 0
        self._crashed = False
        self._cut_in_spawned = False
        self._params = self._sample_params()
        self._ego = VehicleState(
            x=self._params["spawn_x"],
            y=LANE_Y[0],
            vx=self._params["ego_vx"],
            vy=0.0,
        )
        stalled = VehicleState(
            x=self._params["stalled_x"],
            y=LANE_Y[0],
            vx=0.0,
            vy=0.0,
            speed_limit=0.01,
            lane_id=0,
        )
        self._others = [stalled]
        snap = self.snapshot()
        info = {
            "scenario": self.scenario,
            "params": dict(self._params),
            "snapshot": snap,
            "crashed": False,
            "route_pct": self._route_pct(),
            "cut_in_spawned": False,
        }
        return self._obs(), info

    def _obs(self) -> np.ndarray:
        e = self._ego
        return np.array([e.y, e.vy, e.vx], dtype=np.float32)

    def _route_pct(self) -> float:
        return 100.0 * min(max(self._ego.x, 0.0), self.route_length) / self.route_length

    def _maybe_spawn_cut_in(self) -> None:
        if self._cut_in_spawned:
            return
        if self._ego.x >= self._params["cut_in_trigger_x"]:
            cut = VehicleState(
                x=self._ego.x + self._params["cut_in_gap"],
                y=LANE_Y[1],
                vx=self._params["cut_in_speed"],
                vy=0.0,
                lane_id=1,
                speed_limit=self._params["cut_in_speed"],
            )
            self._others.append(cut)
            self._cut_in_spawned = True

    def _propagate_others(self) -> None:
        new = []
        for o in self._others:
            # Constant velocity (stalled stays put; cut-in holds its speed).
            o2 = integrate_point_mass(o, ax=0.0, ay=0.0, dt=DT, vx_min=0.0)
            # Keep stalled at exactly 0.
            if o.speed_limit < 0.1 and o.vx < 0.1:
                o2.vx = 0.0
                o2.x = o.x
            new.append(o2)
        self._others = new

    def _snapshot_info(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "params": dict(self._params),
            "sampled_params": dict(self._params),
            "cut_in_spawned": self._cut_in_spawned,
        }

    def _step_info(self) -> Dict[str, Any]:
        return {
            "snapshot": self.snapshot(),
            "crashed": self._crashed,
            "route_pct": self._route_pct(),
            "scenario": self.scenario,
            "params": dict(self._params),
            "cut_in_spawned": self._cut_in_spawned,
        }

    def step(self, action: np.ndarray):
        action = validate_action(action)
        ax, ay = float(action[0]), float(action[1])
        self._ego = integrate_point_mass(self._ego, ax, ay, DT)
        if self.lateral_bounds is not None:
            self._ego.y = float(np.clip(self._ego.y, *self.lateral_bounds))
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
        return self._obs(), reward, terminated, truncated, self._step_info()

    def snapshot(self) -> WorldSnapshot:
        done = self._crashed or self._ego.x >= self.route_length or self._step_i >= self.max_steps
        return detached_snapshot(
            t=self._step_i * DT,
            ego=self._ego,
            others=self._others,
            route_length=self.route_length,
            done=done,
            crashed=self._crashed,
            info=self._snapshot_info(),
        )

    def close(self) -> None:
        """No-op: this environment owns no external resources."""
