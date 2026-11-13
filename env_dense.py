"""Dense highway traffic scenario (multi-vehicle, same stack API).

Primary training/eval path uses shared kinematics.  A highway-env factory is
also exposed for users who want the upstream MultiAgentRoad env directly.

Defaults are the exact hard-coded pack used by the published baseline.
``randomize=True`` jitters spawn positions/speeds from the reset seed, while
explicit ``params`` can tune the jitter bounds and vehicle count.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from constants import DT
from env_base import (
    detached_snapshot,
    normalize_range,
    require_positive,
    sample_range,
    validate_action,
    validate_lateral_bounds,
)
from env_lane import LANE_Y, TARGET_SPEED
from env_types import (
    ACTION_HIGH,
    ACTION_LOW,
    VehicleState,
    WorldSnapshot,
    collide,
    integrate_point_mass,
)
from errors import ValidationError


def _jitter_bounds(value: Any, *, name: str) -> Tuple[float, float]:
    """Scalar jitter means +/- radius; a pair is an explicit offset range."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if np.isscalar(value) and not isinstance(value, (str, bytes)):
        radius = abs(float(cast(Any, value)))
        if not math.isfinite(radius):
            raise ValidationError(f"{name}: expected a finite jitter radius")
        return (-radius, radius)
    return normalize_range(value, name=name)


class DenseHighwayEnv(gym.Env):
    """Dense two-lane traffic with several slower vehicles and a left-lane pack.

    ``n_vehicles`` is explicit: the scenario currently defines eight fixed
    slots.  Asking for more than eight is capped with a :class:`UserWarning`
    rather than silently truncating.  Asking for fewer uses the first N slots
    (four right-lane vehicles, then four left-lane vehicles), preserving the
    historical ordering.
    """

    metadata = {"render_modes": []}
    SCENARIO_NAME = "dense_highway"
    DEFAULT_LATERAL_BOUNDS = (-0.5, 5.0)
    BASE_RIGHT_XS = (45.0, 75.0, 110.0, 145.0)
    BASE_RIGHT_SPEEDS = (11.0, 12.5, 11.0, 12.5)
    BASE_LEFT_XS = (35.0, 60.0, 95.0, 130.0)
    BASE_LEFT_SPEEDS = (18.0, 10.5, 17.5, 16.0)
    MAX_VEHICLES = len(BASE_RIGHT_XS) + len(BASE_LEFT_XS)
    DEFAULT_N_VEHICLES = MAX_VEHICLES
    DEFAULT_PARAMS: Dict[str, float] = {
        "position_jitter": 0.0,
        "speed_jitter": 0.0,
    }
    RANDOM_RANGES: Dict[str, Tuple[float, float]] = {
        "position_jitter": (-6.0, 6.0),
        "speed_jitter": (-1.5, 1.5),
    }
    _PARAM_KEYS = {
        "position_jitter": "position_jitter",
        "position_jitter_range": "position_jitter",
        "jitter_positions": "position_jitter",
        "jitter_positions_range": "position_jitter",
        "position_noise": "position_jitter",
        "position_noise_range": "position_jitter",
        "speed_jitter": "speed_jitter",
        "speed_jitter_range": "speed_jitter",
        "jitter_speeds": "speed_jitter",
        "jitter_speeds_range": "speed_jitter",
        "speed_noise": "speed_jitter",
        "speed_noise_range": "speed_jitter",
        "n_vehicles": "n_vehicles",
    }

    def __init__(
        self,
        route_length: float = 220.0,
        duration: float = 16.0,
        n_vehicles: int = DEFAULT_N_VEHICLES,
        seed: Optional[int] = None,
        randomize: bool = False,
        params: Optional[Mapping[str, Any]] = None,
        *,
        position_jitter: Optional[Any] = None,
        position_jitter_range: Optional[Any] = None,
        jitter_positions: Optional[Any] = None,
        speed_jitter: Optional[Any] = None,
        speed_jitter_range: Optional[Any] = None,
        jitter_speeds: Optional[Any] = None,
        lateral_bounds: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.scenario = self.SCENARIO_NAME
        self.route_length = require_positive(route_length, name="route_length")
        self.duration = require_positive(duration, name="duration")
        self.max_steps = int(self.duration / DT)
        self.lateral_bounds = validate_lateral_bounds(
            self.DEFAULT_LATERAL_BOUNDS if lateral_bounds is None else lateral_bounds
        )
        self.observation_space = spaces.Box(
            low=np.array([-20.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 10.0, 40.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._step_i = 0
        self._ego = VehicleState(0.0, LANE_Y[0], TARGET_SPEED)
        self._others: List[VehicleState] = []
        self._crashed = False

        supplied_params = dict(params or {})
        position_value = (
            position_jitter
            if position_jitter is not None
            else (position_jitter_range if position_jitter_range is not None else jitter_positions)
        )
        speed_value = (
            speed_jitter
            if speed_jitter is not None
            else (speed_jitter_range if speed_jitter_range is not None else jitter_speeds)
        )
        explicit_jitter = {
            "position_jitter": position_value,
            "speed_jitter": speed_value,
        }
        self.randomize = bool(
            randomize
            or supplied_params
            or any(value is not None for value in explicit_jitter.values())
        )

        specs = {
            key: _jitter_bounds(value, name=key)
            for key, value in self.DEFAULT_PARAMS.items()
        }
        if randomize:
            specs = {
                key: _jitter_bounds(value, name=key)
                for key, value in self.RANDOM_RANGES.items()
            }
        for key, value in explicit_jitter.items():
            if value is not None:
                specs[key] = _jitter_bounds(value, name=f"{key}_range")

        requested_n = n_vehicles
        for key, value in supplied_params.items():
            if key not in self._PARAM_KEYS:
                allowed = ", ".join(sorted(self._PARAM_KEYS))
                raise ValidationError(
                    f"dense_highway params: unknown key {key!r}; allowed keys are {allowed}"
                )
            canonical = self._PARAM_KEYS[key]
            if canonical == "n_vehicles":
                requested_n = value
            else:
                specs[canonical] = _jitter_bounds(value, name=f"params.{key}")
        self.param_spec: Dict[str, Tuple[float, float]] = dict(specs)

        if isinstance(requested_n, bool):
            raise ValidationError("n_vehicles: expected a non-negative integer, got a boolean")
        try:
            requested_n = int(requested_n)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"n_vehicles: expected a non-negative integer, got {requested_n!r}"
            ) from exc
        if requested_n < 0:
            raise ValidationError(f"n_vehicles: must be >= 0, got {requested_n}")
        self.requested_n_vehicles = requested_n
        if requested_n > self.MAX_VEHICLES:
            warnings.warn(
                f"DenseHighwayEnv defines {self.MAX_VEHICLES} fixed vehicle slots; "
                f"requested n_vehicles={requested_n}, capping to {self.MAX_VEHICLES}",
                UserWarning,
                stacklevel=2,
            )
        self.n_vehicles = min(requested_n, self.MAX_VEHICLES)
        self._params: Dict[str, Any] = {
            "n_vehicles": self.n_vehicles,
            "requested_n_vehicles": self.requested_n_vehicles,
            "position_jitter": [],
            "speed_jitter": [],
            "positions": [],
            "speeds": [],
        }

    def _base_slots(self) -> List[Tuple[float, float, int]]:
        slots: List[Tuple[float, float, int]] = []
        for x, v in zip(self.BASE_RIGHT_XS, self.BASE_RIGHT_SPEEDS, strict=False):
            slots.append((x, v, 0))
        for x, v in zip(self.BASE_LEFT_XS, self.BASE_LEFT_SPEEDS, strict=False):
            slots.append((x, v, 1))
        return slots[: self.n_vehicles]

    def _sample_params(self) -> Dict[str, Any]:
        slots = self._base_slots()
        pos_offsets = [
            sample_range(self._rng, self.param_spec["position_jitter"]) for _ in slots
        ]
        speed_offsets = [
            sample_range(self._rng, self.param_spec["speed_jitter"]) for _ in slots
        ]
        positions = [max(1.0, x + dx) for (x, _, _), dx in zip(slots, pos_offsets, strict=False)]
        speeds = [max(0.0, v + dv) for (_, v, _), dv in zip(slots, speed_offsets, strict=False)]
        lanes = [lane for _, _, lane in slots]
        return {
            "n_vehicles": self.n_vehicles,
            "requested_n_vehicles": self.requested_n_vehicles,
            "position_jitter": pos_offsets,
            "speed_jitter": speed_offsets,
            "position_jitter_bounds": list(self.param_spec["position_jitter"]),
            "speed_jitter_bounds": list(self.param_spec["speed_jitter"]),
            "positions": positions,
            "xs": list(positions),
            "speeds": speeds,
            "lanes": lanes,
        }

    def _params_copy(self) -> Dict[str, Any]:
        return copy.deepcopy(self._params)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        self._step_i = 0
        self._crashed = False
        self._params = self._sample_params()
        self._ego = VehicleState(x=0.0, y=LANE_Y[0], vx=TARGET_SPEED, vy=0.0)
        others: List[VehicleState] = []
        for x, v, lane in zip(
            self._params["positions"], self._params["speeds"], self._params["lanes"], strict=False
        ):
            others.append(
                VehicleState(
                    x=float(x),
                    y=LANE_Y[lane],
                    vx=float(v),
                    lane_id=lane,
                    speed_limit=float(v),
                )
            )
        self._others = others
        snap = self.snapshot()
        info = {
            "scenario": self.scenario,
            "params": self._params_copy(),
            "snapshot": snap,
            "crashed": False,
            "route_pct": self._route_pct(),
        }
        return self._obs(), info

    def _obs(self) -> np.ndarray:
        e = self._ego
        return np.array([e.y, e.vy, e.vx], dtype=np.float32)

    def _route_pct(self) -> float:
        return 100.0 * min(max(self._ego.x, 0.0), self.route_length) / self.route_length

    def _propagate_others(self) -> None:
        new = []
        for o in self._others:
            ax = float(np.clip(0.5 * (o.speed_limit - o.vx), -2.5, 1.5))
            # Keep lane.
            ay = -1.5 * (o.y - LANE_Y[o.lane_id]) - 0.8 * o.vy
            new.append(integrate_point_mass(o, ax, ay, DT))
        self._others = new

    def _snapshot_info(self) -> Dict[str, Any]:
        params = self._params_copy()
        return {
            "scenario": self.scenario,
            "params": params,
            "sampled_params": copy.deepcopy(params),
        }

    def _step_info(self) -> Dict[str, Any]:
        return {
            "snapshot": self.snapshot(),
            "crashed": self._crashed,
            "route_pct": self._route_pct(),
            "scenario": self.scenario,
            "params": self._params_copy(),
        }

    def step(self, action: np.ndarray):
        action = validate_action(action)
        ax, ay = float(action[0]), float(action[1])
        self._ego = integrate_point_mass(self._ego, ax, ay, DT)
        # Configurable road lateral bounds (legacy default: [-0.5, 5.0]).
        if self.lateral_bounds is not None:
            self._ego.y = float(np.clip(self._ego.y, *self.lateral_bounds))
        self._propagate_others()
        self._step_i += 1
        for o in self._others:
            if collide(self._ego, o):
                self._crashed = True
                break
        terminated = self._crashed or self._ego.x >= self.route_length
        truncated = self._step_i >= self.max_steps
        return self._obs(), 0.0 if self._crashed else 1.0, terminated, truncated, self._step_info()

    def snapshot(self) -> WorldSnapshot:
        done = (
            self._crashed
            or self._ego.x >= self.route_length
            or self._step_i >= self.max_steps
        )
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


def make_highway_env_continuous(vehicles_count: int = 20, duration: int = 30):
    """Optional upstream highway-env factory (ContinuousAction, 50 Hz).

    This import is intentionally local so the package works without the
    optional ``highway-env`` dependency installed.
    """

    import gymnasium as gym

    env = gym.make("highway-v0", render_mode=None)
    configure = getattr(env.unwrapped, "configure", None)
    if not callable(configure):
        raise ValidationError("highway-env environment does not expose configure()")
    configure(
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
