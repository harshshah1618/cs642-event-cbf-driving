"""Shared state types for all scenarios and the control stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

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


def collide(a: VehicleState, b: VehicleState, dx_thresh: float = 4.5, dy_thresh: float = 1.6) -> bool:
    return abs(a.x - b.x) < dx_thresh and abs(a.y - b.y) < dy_thresh


def integrate_point_mass(
    state: VehicleState,
    ax: float,
    ay: float,
    dt: float,
    vx_min: float = 0.0,
) -> VehicleState:
    ax = float(np.clip(ax, AX_LOW - 1.5, AX_HIGH))
    ay = float(np.clip(ay, AY_LOW, AY_HIGH))
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
