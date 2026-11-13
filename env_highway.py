"""Optional highway-env (Leurent) adapters.

Primary scenarios (`two_trap`, `merge`, `dense_highway`) share a 50 Hz
gymnasium API with continuous ``[ax, ay]`` actions so the stack, shield, and
skills stay simulator-agnostic.  Use these helpers when you want the upstream
MultiAgentRoad dynamics directly.

highway-env's ``ContinuousAction`` expects ``[acceleration, steering]`` in
``[-1, 1]``; it maps ``action[0]`` through its configured acceleration range
and ``action[1]`` through its steering range.  Our actions are physical
``[ax, ay]`` m/s^2, so :func:`accel_to_highway_action` scales each axis using
its *own asymmetric* lower/upper bounds rather than a single symmetric divisor.

Importing this module does not require ``highway-env`` to be installed; the
factory functions import/use it lazily.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import numpy as np

from config import ScenarioConfig
from constants import DT
from env_base import detached_snapshot, require_positive, validate_action
from env_types import AX_HIGH, AX_LOW, AY_HIGH, AY_LOW, VehicleState, WorldSnapshot
from errors import ValidationError


def configure_highway_continuous(env, *, lanes: int = 2, vehicles: int = 15, duration: int = 30):
    """Configure an existing highway-env env for ContinuousAction @ 50 Hz."""

    env.unwrapped.configure(
        {
            "observation": {"type": "Kinematics", "vehicles_count": 5},
            "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
            "lanes_count": lanes,
            "vehicles_count": vehicles,
            "duration": duration,
            "policy_frequency": 50,
            "simulation_frequency": 50,
        }
    )
    return env


def make_highway_v0(**kwargs: Any):
    import gymnasium as gym

    env = gym.make("highway-v0", render_mode=None)
    return configure_highway_continuous(env, **kwargs)


def make_merge_v0(**kwargs: Any):
    import gymnasium as gym

    env = gym.make("merge-v0", render_mode=None)
    return configure_highway_continuous(env, vehicles=kwargs.pop("vehicles", 8), **kwargs)


def _scale_to_unit(value: float, low: float, high: float) -> float:
    """Map ``[low, high]`` asymmetrically onto ``[-1, 1]``."""

    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"action value: expected a finite number, got {value!r}")
    if value >= 0.0:
        scaled = value / high if high > 0.0 else 0.0
    else:
        scaled = value / abs(low) if low < 0.0 else -1.0
    return float(np.clip(scaled, -1.0, 1.0))


def accel_to_highway_action(ax: float, ay: float) -> np.ndarray:
    """Map physical ``[ax, ay]`` m/s^2 to highway-env ContinuousAction.

    ``AX_LOW=-6.0`` / ``AX_HIGH=3.5`` and ``AY_LOW=-5.0`` / ``AY_HIGH=5.0``
    are asymmetric, so each side of zero must use its own bound.  The returned
    vector follows highway-env's order: ``[acceleration, steering]``.
    """

    acceleration = _scale_to_unit(ax, AX_LOW, AX_HIGH)
    steering = _scale_to_unit(ay, AY_LOW, AY_HIGH)
    return np.array([acceleration, steering], dtype=np.float32)


def _vehicle_number(vehicle: Any, names: Tuple[str, ...], default: float) -> float:
    for name in names:
        if hasattr(vehicle, name):
            value = getattr(vehicle, name)
            if value is not None:
                try:
                    out = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(out):
                    return out
    return float(default)


def _convert_vehicle(vehicle: Any) -> VehicleState:
    position = np.asarray(getattr(vehicle, "position", [0.0, 0.0]), dtype=float).reshape(-1)
    x = float(position[0]) if position.size >= 1 else 0.0
    y = float(position[1]) if position.size >= 2 else 0.0
    velocity = np.asarray(getattr(vehicle, "velocity", [0.0, 0.0]), dtype=float).reshape(-1)
    vel_x = float(velocity[0]) if velocity.size >= 1 else 0.0
    vel_y = float(velocity[1]) if velocity.size >= 2 else 0.0
    if hasattr(vehicle, "speed"):
        speed = _vehicle_number(vehicle, ("speed",), math.hypot(vel_x, vel_y))
        heading = _vehicle_number(vehicle, ("heading",), 0.0)
        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
    else:
        # Some lightweight test doubles expose only a velocity vector.
        vx, vy = vel_x, vel_y
    length = _vehicle_number(vehicle, ("LENGTH", "length"), 5.0)
    width = _vehicle_number(vehicle, ("WIDTH", "width"), 2.0)
    lane_id = 0
    lane_index = getattr(vehicle, "lane_index", None)
    try:
        if isinstance(lane_index, (tuple, list)) and len(lane_index) >= 3:
            lane_id = int(cast(Any, lane_index[2]))
        elif lane_index is not None:
            lane_id = int(cast(Any, lane_index))
    except (TypeError, ValueError):
        lane_id = 0
    return VehicleState(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        length=length,
        width=width,
        lane_id=lane_id,
        speed_limit=0.0,
    )


class HighwayEnvAdapter:
    """Wrap a highway-env continuous environment in the :class:`DrivingEnv` contract.

    The adapter reads every vehicle from ``env.unwrapped.road.vehicles`` and
    treats ``env.unwrapped.vehicle`` as the ego.  ``route_length`` comes from
    the supplied :class:`ScenarioConfig` (or the explicit argument).  ``crashed``
    is taken from the wrapped environment's ``info`` dictionary, as required by
    the highway-env wrapper contract.
    """

    def __init__(
        self,
        env: Any,
        config: Optional[ScenarioConfig] = None,
        *,
        route_length: float = 200.0,
        scenario: str = "highway_env",
        params: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        self.env = env
        self._seed = seed
        config_params: Dict[str, Any] = {}
        if config is not None:
            if isinstance(config, Mapping):
                config = ScenarioConfig(**dict(config))
            if not isinstance(config, ScenarioConfig):
                raise ValidationError(
                    f"config: expected ScenarioConfig, got {type(config).__name__}"
                )
            config.validate()
            if config.route_length is not None:
                route_length = config.route_length
            if config.duration is not None:
                self.duration = require_positive(config.duration, name="config.duration")
            else:
                self.duration = 0.0
            scenario = config.name
            self._seed = config.seed
            config_params.update(config.params)
        else:
            self.duration = 0.0
        if params:
            config_params.update(dict(params))
        self.route_length = require_positive(route_length, name="route_length")
        self.scenario = str(scenario)
        self._params: Dict[str, Any] = {"route_length": self.route_length}
        self._params.update(copy.deepcopy(config_params))
        self._step_i = 0
        self._terminated = False
        self._truncated = False
        self._crashed = False
        self._last_info: Dict[str, Any] = {}

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def _params_copy(self) -> Dict[str, Any]:
        return copy.deepcopy(self._params)

    def _build_snapshot(self) -> WorldSnapshot:
        unwrapped = self.unwrapped
        road = getattr(unwrapped, "road", None)
        vehicles = list(getattr(road, "vehicles", []) or [])
        ego_vehicle = getattr(unwrapped, "vehicle", None)
        if ego_vehicle is None and vehicles:
            ego_vehicle = vehicles[0]
        if ego_vehicle is None:
            raise ValidationError("HighwayEnvAdapter: underlying env exposes no ego vehicle")
        ego = _convert_vehicle(ego_vehicle)
        others = [_convert_vehicle(v) for v in vehicles if v is not ego_vehicle]
        try:
            t = float(getattr(unwrapped, "time", self._step_i * DT))
        except (TypeError, ValueError):
            t = self._step_i * DT
        return detached_snapshot(
            t=t,
            ego=ego,
            others=others,
            route_length=self.route_length,
            done=self._terminated or self._truncated,
            crashed=self._crashed,
            info={
                "scenario": self.scenario,
                "params": self._params_copy(),
                "sampled_params": self._params_copy(),
            },
        )

    def snapshot(self) -> WorldSnapshot:
        return self._build_snapshot()

    def _route_pct(self, snap: WorldSnapshot) -> float:
        return (
            100.0
            * min(max(snap.ego.x, 0.0), self.route_length)
            / self.route_length
        )

    def _reset_env(self, seed: Optional[int], options: Optional[Dict[str, Any]]):
        try:
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            # Older Gym/Gymnasium adapters do not accept ``options``.
            try:
                return self.env.reset(seed=seed)
            except TypeError:
                return self.env.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is None:
            seed = self._seed
        result = self._reset_env(seed, options)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            self._last_info = dict(info or {})
        else:  # very old Gym API
            obs, self._last_info = result, {}
        self._step_i = 0
        self._terminated = False
        self._truncated = False
        self._crashed = bool(self._last_info.get("crashed", False))
        snap = self.snapshot()
        info_out = dict(self._last_info)
        info_out.update(
            {
                "scenario": self.scenario,
                "params": self._params_copy(),
                "snapshot": snap,
                "crashed": self._crashed,
                "route_pct": self._route_pct(snap),
            }
        )
        return obs, info_out

    def step(self, action: np.ndarray):
        action = validate_action(action)
        highway_action = accel_to_highway_action(float(action[0]), float(action[1]))
        result = self.env.step(highway_action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif isinstance(result, tuple) and len(result) == 4:  # old Gym API
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            raise ValidationError(
                "HighwayEnvAdapter: wrapped env.step must return a 4- or 5-tuple"
            )
        self._step_i += 1
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._last_info = dict(info or {})
        self._crashed = bool(self._last_info.get("crashed", False))
        snap = self.snapshot()
        info_out = dict(self._last_info)
        info_out.update(
            {
                "scenario": self.scenario,
                "params": self._params_copy(),
                "snapshot": snap,
                "crashed": self._crashed,
                "route_pct": self._route_pct(snap),
            }
        )
        return obs, reward, self._terminated, self._truncated, info_out

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
