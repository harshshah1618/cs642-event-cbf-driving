"""Shared state types for all scenarios and the control stack.

The dataclasses in this module are deliberately small value objects.  The
environment contract in :mod:`event_cbf_driving.env.base` owns validation and
the guarantee that snapshots handed to the router/shield are detached copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List

import numpy as np

# Action box used by skills and shield (m/s^2).
AX_LOW, AX_HIGH = -6.0, 3.5
AY_LOW, AY_HIGH = -5.0, 5.0
ACTION_LOW = np.array([AX_LOW, AY_LOW], dtype=np.float32)
ACTION_HIGH = np.array([AX_HIGH, AY_HIGH], dtype=np.float32)

# Skill observation: [y, vy, vx] — lane-relative continuous control features.
OBS_DIM = 3


@dataclass
class VehicleState:
    x: float
    y: float
    vx: float
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    length: float = 5.0
    width: float = 2.0
    lane_id: int = 0
    speed_limit: float = 0.0  # if >0 and vx below, treated as slow/blocker

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float64)

    def copy(self) -> "VehicleState":
        """Return an independent value copy (safe for snapshot consumers)."""

        return replace(self)


@dataclass
class WorldSnapshot:
    """What the stack sees each control tick."""

    t: float
    ego: VehicleState
    others: List[VehicleState] = field(default_factory=list)
    route_length: float = 200.0
    done: bool = False
    crashed: bool = False
    info: dict = field(default_factory=dict)

    def skill_obs(self) -> np.ndarray:
        e = self.ego
        return np.array([e.y, e.vy, e.vx], dtype=np.float32)

    def obstacles_ahead(self, lane_half_width: float = 1.85) -> List[VehicleState]:
        e = self.ego
        out = []
        for o in self.others:
            if o.x <= e.x:
                continue
            if abs(o.y - e.y) < lane_half_width:
                out.append(o)
        out.sort(key=lambda v: v.x)
        return out

    def copy(self) -> "WorldSnapshot":
        """Detached deep-enough copy of the snapshot payload."""

        from copy import deepcopy

        return WorldSnapshot(
            t=float(self.t),
            ego=self.ego.copy(),
            others=[o.copy() for o in self.others],
            route_length=float(self.route_length),
            done=bool(self.done),
            crashed=bool(self.crashed),
            info=deepcopy(self.info),
        )


def collide(
    a: VehicleState,
    b: VehicleState,
    dx_thresh: float = 4.5,
    dy_thresh: float = 1.6,
    use_footprints: bool = False,
) -> bool:
    """Axis-aligned collision test.

    The defaults reproduce the published research runs exactly (4.5 m
    longitudinal and 1.6 m lateral thresholds).  Passing
    ``use_footprints=True`` instead derives the thresholds from the two
    vehicles' ``length``/``width`` fields.
    """

    if use_footprints:
        dx_thresh = 0.5 * (float(a.length) + float(b.length))
        dy_thresh = 0.5 * (float(a.width) + float(b.width))
    return abs(a.x - b.x) < dx_thresh and abs(a.y - b.y) < dy_thresh


def validate_vehicle(state, **kwargs):
    """Lazy re-export of :func:`event_cbf_driving.env.base.validate_vehicle`."""

    from event_cbf_driving.env.base import validate_vehicle as _validate

    return _validate(state, **kwargs)


def validate_action(action, **kwargs):
    """Lazy re-export of :func:`event_cbf_driving.env.base.validate_action`."""

    from event_cbf_driving.env.base import validate_action as _validate

    return _validate(action, **kwargs)


def validate_snapshot(snapshot, **kwargs):
    """Lazy re-export of :func:`event_cbf_driving.env.base.validate_snapshot`."""

    from event_cbf_driving.env.base import validate_snapshot as _validate

    return _validate(snapshot, **kwargs)


def integrate_point_mass(
    state: VehicleState,
    ax: float,
    ay: float,
    dt: float,
    vx_min: float = 0.0,
    *,
    ax_min: float = AX_LOW - 1.5,
    ax_max: float = AX_HIGH,
    ay_min: float = AY_LOW,
    ay_max: float = AY_HIGH,
) -> VehicleState:
    """Integrate a point-mass vehicle for one tick.

    The default acceleration clips are unchanged from the original code.  The
    keyword-only bounds let callers model a different actuator box without
    changing the legacy call signature.
    """

    ax = float(np.clip(ax, ax_min, ax_max))
    ay = float(np.clip(ay, ay_min, ay_max))
    x = state.x + state.vx * dt + 0.5 * ax * dt * dt
    y = state.y + state.vy * dt + 0.5 * ay * dt * dt
    vx = max(vx_min, state.vx + ax * dt)
    vy = state.vy + ay * dt
    return VehicleState(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        ax=ax,
        ay=ay,
        length=state.length,
        width=state.width,
        lane_id=state.lane_id,
        speed_limit=state.speed_limit,
    )
