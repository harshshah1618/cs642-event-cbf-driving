"""Experimental extra stopping room retained until a maneuver is admitted.

The ordinary collision/braking constraints remain active. The optional reserve
is consumed on a changed target after the stack's fresh admission check. The
geometric estimate is a heuristic primitive, not a reachability certificate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from driving.config import ShieldConfig
from driving.env_types import WorldSnapshot
from driving.safety import InflatedCBFShield, ShieldResult


def geometric_reserve(
    wheelbase: float,
    steering_limit_rad: float,
    lateral_distance: float = 4.0,
    scale: float = 1.0,
) -> float:
    if min(wheelbase, steering_limit_rad, lateral_distance, scale) <= 0:
        raise ValueError("reserve geometry must be positive")
    if not all(math.isfinite(x) for x in (wheelbase, steering_limit_rad, lateral_distance, scale)):
        raise ValueError("reserve geometry must be finite")
    if steering_limit_rad >= math.pi / 2:
        raise ValueError("steering angle must be below pi/2")
    radius = wheelbase / math.tan(steering_limit_rad)
    if radius <= 0 or lateral_distance > 2 * radius:
        raise ValueError("two-arc primitive is outside its geometric domain")
    return scale * math.sqrt(4 * radius * lateral_distance - lateral_distance**2)


class ManeuverReserveShield(InflatedCBFShield):
    """Study wrapper; callers must keep fresh skill-switch admission enabled."""

    MODES = (
        "baseline",
        "static",
        "fixed_latched",
        "geometric_latched",
        "speed_scaled",
        "pending_only",
    )

    def __init__(
        self,
        config: ShieldConfig,
        *,
        mode: str = "baseline",
        reserve_m: float = 8.0,
        scale: float = 1.0,
    ):
        super().__init__(config=config)
        if mode not in self.MODES:
            raise ValueError("unknown reserve policy")
        if any(not math.isfinite(v) or v <= 0 for v in (reserve_m, scale)):
            raise ValueError("reserve and scale must be finite and positive")
        self.reserve_mode, self.reserve_m, self.reserve_scale = mode, float(reserve_m), float(scale)
        self.base_clearance = self.braking_clearance
        self.armed = False
        self.origin_target: float | None = None
        self.transition = False
        self.held_reserve = 0.0
        self.arm_speed = 0.0
        self.last_filter_t: float | None = None
        self.events: list[dict[str, Any]] = []
        self.reserve_integral = 0.0
        self.reserve_peak = 0.0

    def _has_lead(self, snap: WorldSnapshot, target: float) -> bool:
        return any(o.x > snap.ego.x and abs(o.y - target) < 1.85 and o.vx < 15 for o in snap.others)

    def filter(
        self,
        action: np.ndarray,
        snap: WorldSnapshot,
        sigma: float = 0.0,
        vlm_pending: bool = False,
    ) -> ShieldResult:
        target = float(snap.info.get("target_y", 0.0))
        if self.transition and abs(snap.ego.y - target) < 0.35 and abs(snap.ego.vy) < 0.3:
            self.transition = False
        if self.armed and self.origin_target is not None and abs(target - self.origin_target) > 0.1:
            # target_y is changed only after the DrivingStack admission gate.
            self.events.append(
                {
                    "t": snap.t,
                    "event": "release_on_changed_admitted_target",
                    "target": target,
                    "reserve": self.held_reserve,
                }
            )
            self.armed = False
            self.transition = True
            self.held_reserve = 0.0
        if (
            self.reserve_mode not in ("baseline", "static")
            and not self.armed
            and not self.transition
            and vlm_pending
            and self._has_lead(snap, target)
        ):
            value = self.reserve_m
            if self.reserve_mode == "geometric_latched":
                geometry = snap.info["maneuver_geometry"]
                value = geometric_reserve(
                    geometry["wheelbase"], geometry["steering_limit_rad"], scale=self.reserve_scale
                )
            self.held_reserve = value
            self.origin_target = target
            self.arm_speed = max(snap.ego.vx, 1e-3)
            self.armed = True
            self.events.append({"t": snap.t, "event": "arm", "reserve": value, "target": target})
        extra = self.held_reserve if self.armed else 0.0
        if self.reserve_mode == "static":
            extra = self.reserve_m
        elif self.reserve_mode == "speed_scaled" and self.armed:
            extra *= max(0.0, min(1.0, snap.ego.vx / self.arm_speed))
        elif self.reserve_mode == "pending_only" and not vlm_pending:
            extra = 0.0
        self.braking_clearance = self.base_clearance + extra
        if self.last_filter_t is not None:
            self.reserve_integral += max(0.0, snap.t - self.last_filter_t) * extra
        self.last_filter_t = snap.t
        self.reserve_peak = max(self.reserve_peak, extra)
        return super().filter(action, snap, sigma=sigma, vlm_pending=vlm_pending)
