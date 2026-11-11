"""Merge scenario: ego on highway with an on-ramp vehicle merging in.

Uses the same point-mass / lane kinematics and stack API as Two-Trap.  The
defaults are the exact hard-coded values used for the published merge baseline;
randomization is opt-in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from event_cbf_driving import DT
from event_cbf_driving.env.base import (
    detached_snapshot,
    normalize_range,
    require_positive,
    sample_range,
    validate_action,
    validate_lateral_bounds,
)
from event_cbf_driving.env.skill_lane import LANE_Y, TARGET_SPEED
from event_cbf_driving.env.types import (
    ACTION_HIGH,
    ACTION_LOW,
    VehicleState,
    WorldSnapshot,
    collide,
    integrate_point_mass,
)
from event_cbf_driving.errors import ValidationError


def _is_scalar(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return bool(np.isscalar(value))


def _multi_ranges(value: Any, count: int, *, name: str, range_key: bool) -> List[Tuple[float, float]]:
    """Parse shared/per-vehicle ranges for the two left-lane vehicles."""

    if _is_scalar(value):
        bounds = normalize_range(value, name=name)
        return [bounds] * count
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes, Mapping)):
        raise ValidationError(f"{name}: expected a range or a sequence of ranges")
    seq = list(value)
    if len(seq) == 2 and all(_is_scalar(item) for item in seq):
        low = float(seq[0])
        high = float(seq[1])
        if range_key:
            bounds = normalize_range((low, high), name=name)
            return [bounds] * count
        return [
            normalize_range(low, name=f"{name}[0]"),
            normalize_range(high, name=f"{name}[1]"),
        ]
    if len(seq) == count:
        return [
            normalize_range(item, name=f"{name}[{i}]") for i, item in enumerate(seq)
        ]
    raise ValidationError(
        f"{name}: expected two bounds or {count} per-vehicle ranges, got {value!r}"
    )


class MergeScenarioEnv(gym.Env):
    """Highway merge: slow traffic ahead + merging vehicle from on-ramp (y~-4).

    Default parameters reproduce the baseline exactly: lead at x=55 at
    12 m/s, left vehicles at x=40/70 at 18/17 m/s, and a merger at x=30 at
    14 m/s drifting from y=-3.5.  ``randomize=True`` samples non-degenerate
    ranges from the reset seed; explicit ``params`` override individual fields.
    """

    metadata = {"render_modes": []}
    SCENARIO_NAME = "merge"
    DEFAULT_LATERAL_BOUNDS = (-1.0, 5.5)

    DEFAULT_PARAMS: Dict[str, Any] = {
        "lead_gap": 55.0,
        "lead_speed": 12.0,
        "left_gaps": (40.0, 70.0),
        "left_speeds": (18.0, 17.0),
        "merger_gap": 30.0,
        "merger_speed": 14.0,
        "merger_vy": 0.6,
        "ego_vx": TARGET_SPEED,
    }
    RANDOM_RANGES: Dict[str, Any] = {
        "lead_gap": (45.0, 70.0),
        "lead_speed": (10.0, 14.0),
        "left_gaps": ((30.0, 55.0), (60.0, 90.0)),
        "left_speeds": ((16.0, 20.0), (15.0, 19.0)),
        "merger_gap": (20.0, 45.0),
        "merger_speed": (12.0, 16.0),
        "merger_vy": (0.4, 0.8),
        "ego_vx": (TARGET_SPEED, TARGET_SPEED),
    }
    _PARAM_KEYS = {
        "lead_gap": ("lead_gap", False),
        "lead_gap_range": ("lead_gap", True),
        "lead_speed": ("lead_speed", False),
        "lead_speed_range": ("lead_speed", True),
        "left_gaps": ("left_gaps", False),
        "left_gaps_range": ("left_gaps", True),
        "left_gap_range": ("left_gaps", True),
        "left_vehicle_gaps": ("left_gaps", False),
        "left_vehicle_gaps_range": ("left_gaps", True),
        "left_speeds": ("left_speeds", False),
        "left_speeds_range": ("left_speeds", True),
        "left_speed_range": ("left_speeds", True),
        "left_vehicle_speeds": ("left_speeds", False),
        "left_vehicle_speeds_range": ("left_speeds", True),
        "merger_gap": ("merger_gap", False),
        "merger_gap_range": ("merger_gap", True),
        "merger_spawn_gap": ("merger_gap", False),
        "merger_spawn_gap_range": ("merger_gap", True),
        "merger_speed": ("merger_speed", False),
        "merger_speed_range": ("merger_speed", True),
        "merger_spawn_speed": ("merger_speed", False),
        "merger_spawn_speed_range": ("merger_speed", True),
        "merger_vy": ("merger_vy", False),
        "merger_vy_range": ("merger_vy", True),
        "ego_vx": ("ego_vx", False),
        "ego_vx_range": ("ego_vx", True),
    }

    def __init__(
        self,
        route_length: float = 180.0,
        duration: float = 14.0,
        seed: Optional[int] = None,
        randomize: bool = False,
        params: Optional[Mapping[str, Any]] = None,
        *,
        lead_gap_range: Optional[Any] = None,
        lead_speed_range: Optional[Any] = None,
        left_gaps_range: Optional[Any] = None,
        left_gap_range: Optional[Any] = None,
        left_speeds_range: Optional[Any] = None,
        left_speed_range: Optional[Any] = None,
        merger_gap_range: Optional[Any] = None,
        merger_spawn_gap_range: Optional[Any] = None,
        merger_speed_range: Optional[Any] = None,
        merger_spawn_speed_range: Optional[Any] = None,
        merger_vy_range: Optional[Any] = None,
        ego_vx_range: Optional[Any] = None,
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
        self._ego = VehicleState(
            x=0.0, y=LANE_Y[0], vx=self.DEFAULT_PARAMS["ego_vx"], vy=0.0
        )
        self._others: List[VehicleState] = []
        self._merger: Optional[VehicleState] = None
        self._crashed = False

        explicit_ranges = {
            "lead_gap": lead_gap_range,
            "lead_speed": lead_speed_range,
            "left_gaps": left_gaps_range if left_gaps_range is not None else left_gap_range,
            "left_speeds": (
                left_speeds_range if left_speeds_range is not None else left_speed_range
            ),
            "merger_gap": (
                merger_gap_range
                if merger_gap_range is not None
                else merger_spawn_gap_range
            ),
            "merger_speed": (
                merger_speed_range
                if merger_speed_range is not None
                else merger_spawn_speed_range
            ),
            "merger_vy": merger_vy_range,
            "ego_vx": ego_vx_range,
        }
        supplied_params = dict(params or {})
        self.randomize = bool(
            randomize
            or supplied_params
            or any(value is not None for value in explicit_ranges.values())
        )

        specs: Dict[str, Tuple[float, float]] = {
            "lead_gap": (float(self.DEFAULT_PARAMS["lead_gap"]),) * 2,
            "lead_speed": (float(self.DEFAULT_PARAMS["lead_speed"]),) * 2,
            "left_gap_0": (float(self.DEFAULT_PARAMS["left_gaps"][0]),) * 2,
            "left_gap_1": (float(self.DEFAULT_PARAMS["left_gaps"][1]),) * 2,
            "left_speed_0": (float(self.DEFAULT_PARAMS["left_speeds"][0]),) * 2,
            "left_speed_1": (float(self.DEFAULT_PARAMS["left_speeds"][1]),) * 2,
            "merger_gap": (float(self.DEFAULT_PARAMS["merger_gap"]),) * 2,
            "merger_speed": (float(self.DEFAULT_PARAMS["merger_speed"]),) * 2,
            "merger_vy": (float(self.DEFAULT_PARAMS["merger_vy"]),) * 2,
            "ego_vx": (float(self.DEFAULT_PARAMS["ego_vx"]),) * 2,
        }
        if randomize:
            specs["lead_gap"] = normalize_range(self.RANDOM_RANGES["lead_gap"], name="lead_gap")
            specs["lead_speed"] = normalize_range(
                self.RANDOM_RANGES["lead_speed"], name="lead_speed"
            )
            left_gaps = _multi_ranges(
                self.RANDOM_RANGES["left_gaps"], 2, name="left_gaps", range_key=True
            )
            left_speeds = _multi_ranges(
                self.RANDOM_RANGES["left_speeds"], 2, name="left_speeds", range_key=True
            )
            specs["left_gap_0"], specs["left_gap_1"] = left_gaps
            specs["left_speed_0"], specs["left_speed_1"] = left_speeds
            specs["merger_gap"] = normalize_range(
                self.RANDOM_RANGES["merger_gap"], name="merger_gap"
            )
            specs["merger_speed"] = normalize_range(
                self.RANDOM_RANGES["merger_speed"], name="merger_speed"
            )
            specs["merger_vy"] = normalize_range(
                self.RANDOM_RANGES["merger_vy"], name="merger_vy"
            )
            specs["ego_vx"] = normalize_range(self.RANDOM_RANGES["ego_vx"], name="ego_vx")

        for key, value in explicit_ranges.items():
            if value is None:
                continue
            if key in ("left_gaps", "left_speeds"):
                parsed = _multi_ranges(value, 2, name=f"{key}_range", range_key=True)
                if key == "left_gaps":
                    specs["left_gap_0"], specs["left_gap_1"] = parsed
                else:
                    specs["left_speed_0"], specs["left_speed_1"] = parsed
            else:
                specs[key] = normalize_range(value, name=f"{key}_range")

        for key, value in supplied_params.items():
            if key not in self._PARAM_KEYS:
                allowed = ", ".join(sorted(self._PARAM_KEYS))
                raise ValidationError(
                    f"merge params: unknown key {key!r}; allowed keys are {allowed}"
                )
            canonical, is_range = self._PARAM_KEYS[key]
            if canonical in ("left_gaps", "left_speeds"):
                parsed = _multi_ranges(value, 2, name=f"params.{key}", range_key=is_range)
                if canonical == "left_gaps":
                    specs["left_gap_0"], specs["left_gap_1"] = parsed
                else:
                    specs["left_speed_0"], specs["left_speed_1"] = parsed
            else:
                specs[canonical] = normalize_range(value, name=f"params.{key}")
        self.param_spec = dict(specs)
        self._params: Dict[str, Any] = {
            "lead_gap": self.DEFAULT_PARAMS["lead_gap"],
            "lead_speed": self.DEFAULT_PARAMS["lead_speed"],
            "left_gaps": list(self.DEFAULT_PARAMS["left_gaps"]),
            "left_speeds": list(self.DEFAULT_PARAMS["left_speeds"]),
            "merger_gap": self.DEFAULT_PARAMS["merger_gap"],
            "merger_speed": self.DEFAULT_PARAMS["merger_speed"],
            "merger_vy": self.DEFAULT_PARAMS["merger_vy"],
            "ego_vx": self.DEFAULT_PARAMS["ego_vx"],
        }

    def _sample_params(self) -> Dict[str, Any]:
        lead_gap = sample_range(self._rng, self.param_spec["lead_gap"])
        lead_speed = sample_range(self._rng, self.param_spec["lead_speed"])
        left_gap_0 = sample_range(self._rng, self.param_spec["left_gap_0"])
        left_gap_1 = sample_range(self._rng, self.param_spec["left_gap_1"])
        left_speed_0 = sample_range(self._rng, self.param_spec["left_speed_0"])
        left_speed_1 = sample_range(self._rng, self.param_spec["left_speed_1"])
        merger_gap = sample_range(self._rng, self.param_spec["merger_gap"])
        merger_speed = sample_range(self._rng, self.param_spec["merger_speed"])
        merger_vy = sample_range(self._rng, self.param_spec["merger_vy"])
        ego_vx = sample_range(self._rng, self.param_spec["ego_vx"])
        if min(left_speed_0, left_speed_1, merger_speed, lead_speed, ego_vx) < 0.0:
            raise ValidationError("merge params: vehicle speeds must be >= 0")
        params: Dict[str, Any] = {
            "lead_gap": lead_gap,
            "lead_speed": lead_speed,
            "left_gaps": [left_gap_0, left_gap_1],
            "left_speeds": [left_speed_0, left_speed_1],
            "left_gap_0": left_gap_0,
            "left_gap_1": left_gap_1,
            "left_speed_0": left_speed_0,
            "left_speed_1": left_speed_1,
            "merger_gap": merger_gap,
            "merger_speed": merger_speed,
            "merger_spawn_gap": merger_gap,
            "merger_spawn_speed": merger_speed,
            "merger_vy": merger_vy,
            "ego_vx": ego_vx,
        }
        return params

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        self._step_i = 0
        self._crashed = False
        self._params = self._sample_params()
        self._ego = VehicleState(
            x=0.0,
            y=LANE_Y[0],
            vx=self._params["ego_vx"],
            vy=0.0,
        )
        # Lead vehicle in right lane (slow).
        lead = VehicleState(
            x=self._params["lead_gap"],
            y=LANE_Y[0],
            vx=self._params["lead_speed"],
            lane_id=0,
            speed_limit=self._params["lead_speed"],
        )
        # Left-lane traffic.
        left = VehicleState(
            x=self._params["left_gaps"][0],
            y=LANE_Y[1],
            vx=self._params["left_speeds"][0],
            lane_id=1,
            speed_limit=self._params["left_speeds"][0],
        )
        left2 = VehicleState(
            x=self._params["left_gaps"][1],
            y=LANE_Y[1],
            vx=self._params["left_speeds"][1],
            lane_id=1,
            speed_limit=self._params["left_speeds"][1],
        )
        # On-ramp merger (approaches right lane).
        merger = VehicleState(
            x=self._params["merger_gap"],
            y=-3.5,
            vx=self._params["merger_speed"],
            vy=self._params["merger_vy"],
            lane_id=-1,
            speed_limit=self._params["merger_speed"],
        )
        self._others = [lead, left, left2, merger]
        self._merger = merger
        snap = self.snapshot()
        info = {
            "scenario": self.scenario,
            "params": self._params_copy(),
            "snapshot": snap,
            "crashed": False,
            "route_pct": self._route_pct(),
        }
        return self._obs(), info

    def _params_copy(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self._params.items():
            out[key] = list(value) if isinstance(value, list) else value
        return out

    def _obs(self) -> np.ndarray:
        e = self._ego
        return np.array([e.y, e.vy, e.vx], dtype=np.float32)

    def _route_pct(self) -> float:
        return 100.0 * min(max(self._ego.x, 0.0), self.route_length) / self.route_length

    def _propagate_others(self) -> None:
        new = []
        for o in self._others:
            ax = float(np.clip(0.4 * (o.speed_limit - o.vx), -2.0, 1.5))
            ay = 0.0
            if o is self._merger:
                if o.lane_id == -1:
                    if o.y < LANE_Y[0] - 0.3:
                        # Explicit on-ramp merge: steer toward the right lane.
                        ay = 1.2
                    else:
                        # Lane assignment completes with a lateral damper.  The
                        # legacy kinematics apply no further re-centering after
                        # this tick, which is required for baseline parity.
                        ay = -0.3 * o.vy
                        o.lane_id = 0
                        o.y = min(o.y, LANE_Y[0] + 0.2)
                # Else: merged; the legacy lateral damper is intentionally not
                # re-applied, preserving the published merge trajectory.
            else:
                # Explicit lane keeping for regular traffic.  The default
                # vehicles start exactly on their lane centres, so this is a
                # zero correction and leaves baseline motion unchanged.
                target_y = LANE_Y.get(o.lane_id)
                if target_y is not None:
                    ay = -1.5 * (o.y - target_y) - 0.8 * o.vy
            o2 = integrate_point_mass(o, ax, ay, DT)
            if o is self._merger:
                self._merger = o2
            new.append(o2)
        self._others = new

    def _snapshot_info(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "params": self._params_copy(),
            "sampled_params": self._params_copy(),
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
        # Configurable road lateral bounds (legacy default: [-1.0, 5.5]).
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
