"""Latency- and uncertainty-inflated CBF shield at control rate.

The default configuration reproduces the original research filter:

    h = (x_obs - x - standoff) - headway*(vx - v_obs) - inflation
    inflation = gamma * sigma + eta * I(VLM pending) * v * delta_t_vlm

with obstacle selection restricted to the current lane corridor and no
lateral lookahead.  The optional footprint/lookahead modes are conservative
extensions used by the production profile.

Lateral lookahead approximation
-------------------------------
Future ego lateral motion is approximated as constant-velocity for
``lookahead_s`` seconds.  The ego is treated as sweeping the whole lateral
interval between its current and predicted ``y`` positions (with ``vy``
clamped to a physically plausible range).  This ignores lateral acceleration,
tracking lag, and road curvature, so it is a heuristic horizon check rather
than a guarantee; it is deliberately conservative about entering a target
lane while changing lanes.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np

from config import ShieldConfig
from env_types import WorldSnapshot
from errors import SafetyError
from logging_utils import get_logger, log_event

logger = get_logger(__name__)

# Predicted lateral motion is linear in vy; clamp the estimate so a noisy
# velocity cannot sweep the predicted interval across the entire road.
_MAX_LOOKAHEAD_VY = 3.0  # m/s

# Floating-point tolerance for "h is non-negative" and override detection.
_H_EPS = 1e-9
_OVERRIDE_EPS = 1e-3

_UNSET = object()


@dataclass
class ShieldResult:
    """Outcome of one safety-filter tick.

    The first five fields preserve the original positional contract.  The
    remaining fields make the result observable and fail-closed without
    breaking callers that only know about the legacy fields.
    """

    action: np.ndarray
    residual: float  # how much ax was reduced (override magnitude)
    overridden: bool
    inflation: float
    h: float
    feasible: bool = True
    safe: bool = True
    reason: str = "ok"
    obstacle_count: int = 0
    emergency: bool = False
    lateral_overridden: bool = False
    lateral_h: Optional[float] = None
    braking_h: Optional[float] = None
    braking_feasible: bool = True
    cbf_overridden: bool = False
    braking_overridden: bool = False


class InflatedCBFShield:
    """Longitudinal CBF on obstacles ahead in the ego's lateral corridor.

    Parameters
    ----------
    config:
        Optional :class:`ShieldConfig`.  Validated on construction (on a
        copy, so the caller's object is not mutated).
    **overrides:
        Any legacy or :class:`ShieldConfig` field may be supplied explicitly;
        explicit values take precedence over values from ``config``.
    """

    def __init__(
        self,
        standoff: Any = _UNSET,
        headway_s: Any = _UNSET,
        class_k: Any = _UNSET,
        lane_half_width: Any = _UNSET,
        gamma_sigma: Any = _UNSET,
        eta_latency: Any = _UNSET,
        vlm_delay_s: Any = _UNSET,
        ax_min: Any = _UNSET,
        ax_max: Any = _UNSET,
        enabled: Any = _UNSET,
        inflate: Any = _UNSET,
        config: Optional[ShieldConfig] = None,
        *,
        sigma_cap: Any = _UNSET,
        pending_margin: Any = _UNSET,
        finite_checks: Any = _UNSET,
        fail_closed: Any = _UNSET,
        lateral_mode: Any = _UNSET,
        lateral_margin: Any = _UNSET,
        lookahead_s: Any = _UNSET,
        emergency_decel: Any = _UNSET,
        min_gap_m: Any = _UNSET,
    ):
        # Allow InflatedCBFShield(config) as well as config=... while keeping
        # the historical positional arguments intact.
        if isinstance(standoff, ShieldConfig):
            if config is not None:
                raise SafetyError("shield config supplied both positionally and by keyword")
            config = standoff
            standoff = _UNSET

        cfg = copy.deepcopy(config) if config is not None else ShieldConfig()
        overrides = {
            "standoff": standoff,
            "headway_s": headway_s,
            "class_k": class_k,
            "lane_half_width": lane_half_width,
            "gamma_sigma": gamma_sigma,
            "sigma_cap": sigma_cap,
            "eta_latency": eta_latency,
            "vlm_delay_s": vlm_delay_s,
            "ax_min": ax_min,
            "ax_max": ax_max,
            "enabled": enabled,
            "inflate": inflate,
            "pending_margin": pending_margin,
            "finite_checks": finite_checks,
            "fail_closed": fail_closed,
            "lateral_mode": lateral_mode,
            "lateral_margin": lateral_margin,
            "lookahead_s": lookahead_s,
            "emergency_decel": emergency_decel,
            "min_gap_m": min_gap_m,
        }
        for name, value in overrides.items():
            if value is not _UNSET:
                setattr(cfg, name, value)
        cfg.validate("shield")

        self._config = cfg
        # Explicit typed assignments keep the public field surface visible to
        # type checkers and IDEs (the legacy implementation used setattr in a
        # loop over dataclass fields).
        self.standoff: float = cfg.standoff
        self.headway_s: float = cfg.headway_s
        self.class_k: float = cfg.class_k
        self.lane_half_width: float = cfg.lane_half_width
        self.gamma_sigma: float = cfg.gamma_sigma
        self.sigma_cap: float = cfg.sigma_cap
        self.eta_latency: float = cfg.eta_latency
        self.vlm_delay_s: float = cfg.vlm_delay_s
        self.ax_min: float = cfg.ax_min
        self.ax_max: float = cfg.ax_max
        self.enabled: bool = cfg.enabled
        self.cbf_constraint_enabled: bool = cfg.cbf_constraint_enabled
        self.inflate: bool = cfg.inflate
        self.pending_margin: bool = cfg.pending_margin
        self.finite_checks: bool = cfg.finite_checks
        self.fail_closed: bool = cfg.fail_closed
        self.lateral_mode: str = cfg.lateral_mode
        self.lateral_margin: float = cfg.lateral_margin
        self.lookahead_s: float = cfg.lookahead_s
        self.emergency_decel: float = cfg.emergency_decel
        self.min_gap_m: float = cfg.min_gap_m
        self.road_boundary_guard = cfg.road_boundary_guard
        self.boundary_gain = cfg.boundary_gain
        self.boundary_margin = cfg.boundary_margin
        self.braking_guard = cfg.braking_guard
        self.braking_decel = cfg.braking_decel
        self.braking_clearance = cfg.braking_clearance
        self.braking_response_s = cfg.braking_response_s
        self.intent_corridor = cfg.intent_corridor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def inflation(self, sigma: float, vlm_pending: bool, vx: float) -> float:
        """Return the total safety inflation for this tick."""

        if not self.inflate:
            return 0.0
        # Cap the sigma term so a brief OOD critic spike cannot demand an
        # unreachable standoff.
        sig = min(self.gamma_sigma * max(float(sigma), 0.0), self.sigma_cap)
        latency = (
            self.eta_latency
            * (1.0 if (vlm_pending and self.pending_margin) else 0.0)
            * max(float(vx), 0.0)
            * self.vlm_delay_s
        )
        return float(sig + latency)

    def filter(
        self,
        action: np.ndarray,
        snap: WorldSnapshot,
        sigma: float = 0.0,
        vlm_pending: bool = False,
    ) -> ShieldResult:
        if self.intent_corridor:
            if not isinstance(getattr(snap, "info", None), dict):
                return self._invalid("invalid_skill_intent")
            target = snap.info.get("target_y")
            if target is not None and not self._is_finite(target):
                return self._invalid("invalid_skill_intent")
        result = self._longitudinal_filter(action, snap, sigma, vlm_pending)
        result.cbf_overridden = bool(
            self.enabled and self.cbf_constraint_enabled and result.overridden
            and not result.emergency
        )
        if self.enabled and self.braking_guard and not result.emergency:
            result = self._braking_filter(result, snap)
        if not self.enabled or not self.road_boundary_guard or result.emergency:
            return result
        try:
            bounds = np.asarray(snap.info["road_bounds"], dtype=float)
            heading = float(snap.info["heading"])
            if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not math.isfinite(heading):
                return self._invalid("invalid_road_bounds")
            ego = snap.ego
            extent = 0.5 * (ego.width * abs(math.cos(heading))
                            + ego.length * abs(math.sin(heading))) + self.boundary_margin
            lower, upper = float(bounds[0]) + extent, float(bounds[1]) - extent
            if lower >= upper:
                return self._invalid("invalid_road_bounds")
            k = self.boundary_gain
            # Relative-degree-two boundary conditions: h''+2*k*h'+k²*h >= 0.
            lo = -2 * k * ego.vy - k*k*(ego.y-lower)
            hi = -2 * k * ego.vy + k*k*(upper-ego.y)
            feasible = lo <= hi and lo <= 5.0 and hi >= -5.0
            old_ay = float(result.action[1])
            if feasible:
                ay = float(np.clip(old_ay, max(-5.0, lo), min(5.0, hi)))
            else:
                ay = 5.0 if ego.y < (lower+upper)/2 else -5.0
            result.lateral_h = min(ego.y-lower, upper-ego.y)
            result.lateral_overridden = abs(ay-old_ay) > _OVERRIDE_EPS
            result.action[1] = ay
            result.residual = math.hypot(result.residual, ay-old_ay)
            result.overridden = result.overridden or result.lateral_overridden
            result.feasible = result.feasible and feasible
            result.safe = result.safe and feasible and result.lateral_h >= 0
            if not feasible:
                result.reason = "lateral_boundary_infeasible"
            elif result.lateral_overridden and result.reason == "ok":
                result.reason = "lateral_boundary"
            return result
        except (KeyError, TypeError, ValueError, AttributeError):
            return self._invalid("invalid_road_bounds")

    def _braking_filter(self, result: ShieldResult, snap: WorldSnapshot) -> ShieldResult:
        """Discrete stopping-envelope backup, separate from the soft margin.

        Assumes ego can achieve braking_decel and the lead can brake at that
        same rate. Includes a tracking/response allowance and full footprints.
        These are testable model assumptions, not a real-vehicle guarantee.
        The original inflated h/feasibility remain visible in the result.
        """
        from constants import DT

        ego = snap.ego
        if any(not self._is_finite(getattr(v, field, None)) or getattr(v, field) <= 0
               for v in [ego, *snap.others] for field in ("length", "width")):
            return self._invalid("invalid_braking_footprint")
        b = self.braking_decel
        interval = self._snapshot_lateral_interval(snap)
        cap = self.ax_max
        minimum = float("inf")
        feasible = True
        for obs in snap.others:
            if obs.x <= ego.x or not self._laterally_relevant(ego, obs, interval):
                continue
            extent = 0.5 * (math.hypot(ego.length, ego.width) + math.hypot(obs.length, obs.width))
            gap = obs.x - ego.x - extent - self.braking_clearance
            ve, vo = max(ego.vx, 0.0), max(obs.vx, 0.0)

            def barrier(distance, ev, ov):
                return distance - max(0.0, (ev*ev - ov*ov)/(2*b)) - self.braking_response_s*ev

            h = barrier(gap, ve, vo)
            minimum = min(minimum, h)
            target = (1.0 - self.class_k*DT)*h
            obstacle_dt = min(DT, vo/b)
            obstacle_dx = vo*obstacle_dt - 0.5*b*obstacle_dt**2

            def allowed(ax, ve=ve, vo=vo, gap=gap, obstacle_dx=obstacle_dx, target=target):
                ego_dt = min(DT, ve/abs(ax)) if ax < 0 else DT
                ego_dx = ve*ego_dt + 0.5*ax*ego_dt**2
                next_h = barrier(gap + obstacle_dx - ego_dx, max(0.0, ve+ax*DT), max(0.0, vo-b*DT))
                return next_h >= target - 1e-9

            if not allowed(self.ax_min):
                feasible = False
                cap = self.ax_min
            elif not allowed(self.ax_max):
                low, high = self.ax_min, self.ax_max
                for _ in range(24):
                    middle = 0.5*(low+high)
                    if allowed(middle):
                        low = middle
                    else:
                        high = middle
                cap = min(cap, low)
        if math.isfinite(minimum):
            old_ax = float(result.action[0])
            result.action[0] = min(old_ax, cap)
            result.braking_overridden = old_ax - float(result.action[0]) > _OVERRIDE_EPS
            result.residual = math.hypot(result.residual, old_ax-float(result.action[0]))
            result.overridden = result.overridden or old_ax-float(result.action[0]) > _OVERRIDE_EPS
            result.braking_h = minimum
            result.braking_feasible = feasible
            result.feasible = result.feasible and feasible
            result.safe = result.safe and feasible and minimum >= 0
            if not feasible:
                result.reason = "braking_infeasible"
            elif minimum < 0:
                result.reason = "braking_envelope_violated"
        return result

    def _snapshot_lateral_interval(self, snap):
        interval = self._ego_lateral_interval(snap.ego)
        target = snap.info.get("target_y")
        if self.intent_corridor and target is not None:
            if not self._is_finite(target):
                raise SafetyError("non-finite skill intent")
            half_width = snap.ego.width / 2 + self.lateral_margin
            if interval is None:
                interval = (snap.ego.y-half_width, snap.ego.y+half_width)
            interval = (min(interval[0], float(target)-half_width),
                        max(interval[1], float(target)+half_width))
        return interval

    def _longitudinal_filter(
        self, action: np.ndarray, snap: WorldSnapshot, sigma: float = 0.0,
        vlm_pending: bool = False,
    ) -> ShieldResult:
        """Apply the CBF and return a fail-closed, observable result."""

        # --- Action shape/finiteness (always check shape; finite_checks
        # controls the numerical checks for performance-sensitive ablations).
        try:
            requested = np.asarray(action, dtype=float)
        except (TypeError, ValueError):
            return self._invalid("invalid_action")
        if requested.shape != (2,):
            return self._invalid("invalid_action")
        if self.finite_checks and not bool(np.all(np.isfinite(requested))):
            return self._invalid("invalid_action")
        if not self.finite_checks:
            # Clipping below still requires usable scalars.
            try:
                ax_rl, ay = float(requested[0]), float(requested[1])
            except (TypeError, ValueError, IndexError):
                return self._invalid("invalid_action")
        else:
            ax_rl, ay = float(requested[0]), float(requested[1])

        if not self.enabled:
            return ShieldResult(
                action=np.array([ax_rl, ay], dtype=np.float32),
                residual=0.0,
                overridden=False,
                inflation=0.0,
                h=float("inf"),
                feasible=True,
                safe=True,
                reason="disabled",
                obstacle_count=0,
                emergency=False,
            )

        # --- Ego state.
        ego = getattr(snap, "ego", None)
        if ego is None or not self._valid_ego(ego):
            return self._invalid("invalid_ego")

        # --- Critic uncertainty.
        if self.finite_checks and not self._is_finite(sigma):
            return self._invalid("invalid_sigma")
        try:
            sigma_value = float(sigma)
        except (TypeError, ValueError):
            return self._invalid("invalid_sigma")

        infl = (
            self.inflation(sigma_value, bool(vlm_pending), float(ego.vx))
            if self.cbf_constraint_enabled else 0.0
        )

        # --- Obstacles: validate before lateral filtering so a non-finite
        # obstacle that might be relevant is never silently skipped.
        try:
            others: List[Any] = list(getattr(snap, "others", []) or [])
        except TypeError:
            return self._invalid("invalid_obstacle")

        lateral_interval = self._snapshot_lateral_interval(snap)
        h_min = float("inf")
        ax_cap = float(self.ax_max)
        obstacle_count = 0

        for obs in others:
            obs_x = getattr(obs, "x", None)
            if self.finite_checks and not self._is_finite(obs_x):
                # Cannot decide whether this obstacle is ahead, so fail closed.
                return self._invalid("invalid_obstacle")
            try:
                ox = float(cast(Any, obs_x))
            except (TypeError, ValueError):
                return self._invalid("invalid_obstacle")
            if not ox > float(ego.x):
                continue  # only obstacles ahead matter

            # Lateral selection itself needs finite y (and width in footprint
            # mode); a non-finite value here could be relevant, so fail closed.
            if self.finite_checks:
                if not self._is_finite(getattr(obs, "y", None)):
                    return self._invalid("invalid_obstacle")
                if self.lateral_mode == "footprint":
                    width = getattr(obs, "width", None)
                    if not self._is_finite(width) or float(cast(Any, width)) <= 0.0:
                        return self._invalid("invalid_obstacle")
            if not self._laterally_relevant(ego, obs, lateral_interval):
                continue

            # Only obstacles that are actually considered must have a usable
            # closing speed; irrelevant adjacent-lane data is ignored.
            if self.finite_checks and not self._is_finite(getattr(obs, "vx", None)):
                return self._invalid("invalid_obstacle")

            obstacle_count += 1
            dx = ox - float(ego.x) - self.standoff
            h = dx - self.headway_s * (float(ego.vx) - float(obs.vx)) - infl
            h_min = min(h_min, h)

            # hdot = (v_obs - vx) - headway*ax (zero obstacle acceleration)
            # Require hdot + class_k*h >= 0, solve for the ax upper bound.
            ax_safe = (
                (float(obs.vx) - float(ego.vx)) + self.class_k * h
            ) / self.headway_s
            ax_cap = min(ax_cap, ax_safe)

        if not self.cbf_constraint_enabled:
            # Keep validation and actuator limits while removing ONLY this
            # constraint. Braking and boundary guards still run in filter().
            # safe=True here means this disabled stage did not flag a fault;
            # it is not an assertion that the physical scene is safe.
            ax = float(np.clip(ax_rl, self.ax_min, self.ax_max))
            return ShieldResult(
                action=np.array([ax, ay], dtype=np.float32),
                residual=max(0.0, ax_rl-ax),
                overridden=abs(ax-ax_rl) > _OVERRIDE_EPS,
                inflation=0.0,
                h=float(h_min),
                reason="cbf_disabled",
                obstacle_count=obstacle_count,
            )

        if obstacle_count == 0:
            ax = float(np.clip(min(ax_rl, self.ax_max), self.ax_min, self.ax_max))
            residual = max(0.0, ax_rl - ax)
            return ShieldResult(
                action=np.array([ax, ay], dtype=np.float32),
                residual=residual,
                overridden=abs(ax - ax_rl) > _OVERRIDE_EPS,
                inflation=infl,
                h=float("inf"),
                feasible=True,
                safe=True,
                reason="ok",
                obstacle_count=0,
                emergency=False,
            )

        # --- Feasibility.  Do not saturate silently and call the result safe.
        feasible = ax_cap >= self.ax_min
        if feasible:
            ax = float(np.clip(min(ax_rl, ax_cap), self.ax_min, self.ax_max))
        else:
            # The barrier needs more deceleration than the actuator can give.
            ax = float(self.ax_min)

        residual = max(0.0, ax_rl - ax)
        overridden = abs(ax - ax_rl) > _OVERRIDE_EPS
        # Did the barrier constraint ask for a reduction relative to the
        # requested action?  This distinguishes an active correction from a
        # request that was already inside the safe set.
        override_needed = ax_cap < ax_rl - _OVERRIDE_EPS

        safe = feasible and h_min >= -_H_EPS
        reason = "ok"
        if not feasible:
            safe = False
            reason = "infeasible"
            # Expected for aggressive baselines that saturate at ax_min; keep
            # this at debug level so a long episode does not spam the console.
            log_event(
                logger,
                "shield.infeasible",
                logging.DEBUG,
                h=h_min,
                ax_cap=ax_cap,
                ax_min=self.ax_min,
            )
        elif h_min < -_H_EPS:
            safe = False
            reason = "barrier_violated"
        elif h_min < self.min_gap_m and not override_needed:
            # Inside the configured margin and the requested action was not
            # already being corrected by the barrier: do not claim safety.
            safe = False
            reason = "barrier_violated"

        return ShieldResult(
            action=np.array([ax, ay], dtype=np.float32),
            residual=residual,
            overridden=overridden,
            inflation=infl,
            h=float(h_min),
            feasible=bool(feasible),
            safe=bool(safe),
            reason=reason,
            obstacle_count=obstacle_count,
            emergency=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _valid_ego(self, ego: Any) -> bool:
        if not self.finite_checks:
            return True
        for name in ("x", "y", "vx", "vy"):
            if not self._is_finite(getattr(ego, name, None)):
                return False
        if self.lateral_mode == "footprint":
            width = getattr(ego, "width", None)
            if not self._is_finite(width) or float(cast(Any, width)) <= 0.0:
                return False
        return True

    def _ego_lateral_interval(self, ego: Any) -> Tuple[float, float]:
        """Lateral interval swept by the ego over the lookahead horizon."""

        y0 = float(ego.y)
        if self.lookahead_s <= 0.0:
            return (y0, y0)
        vy = float(np.clip(float(ego.vy), -_MAX_LOOKAHEAD_VY, _MAX_LOOKAHEAD_VY))
        y1 = y0 + vy * self.lookahead_s
        return (min(y0, y1), max(y0, y1))

    def _laterally_relevant(
        self, ego: Any, obs: Any, ego_interval: Tuple[float, float]
    ) -> bool:
        """Lane-corridor (legacy) or footprint-aware lateral overlap test."""

        low, high = ego_interval
        oy = float(obs.y)
        if low <= oy <= high:
            distance = 0.0
        else:
            distance = min(abs(oy - low), abs(oy - high))

        if self.lateral_mode == "footprint":
            threshold = (
                0.5 * (float(ego.width) + float(obs.width)) + self.lateral_margin
            )
        else:
            threshold = self.lane_half_width
        return distance < threshold

    def _emergency_action(self) -> np.ndarray:
        ax = float(np.clip(self.emergency_decel, self.ax_min, self.ax_max))
        # Lateral demand is discarded in an emergency: hold the wheel and
        # brake longitudinally.
        return np.array([ax, 0.0], dtype=np.float32)

    def _invalid(self, reason: str, obstacle_count: int = 0) -> ShieldResult:
        if not self.fail_closed:
            raise SafetyError(f"shield failed validation: {reason}")
        log_event(logger, "shield.fail_closed", logging.WARNING, reason=reason)
        return ShieldResult(
            action=self._emergency_action(),
            residual=0.0,
            overridden=True,
            inflation=0.0,
            h=float("-inf"),
            feasible=False,
            safe=False,
            reason=reason,
            obstacle_count=obstacle_count,
            emergency=True,
        )
