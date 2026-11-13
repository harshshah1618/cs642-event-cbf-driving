"""Final safety gate and observability for the CBF shield.

``SafetyMonitor`` consumes one :class:`ShieldResult` per control tick.  A
short run of unsafe results is reported as ``DEGRADED``; repeated *invalid*
measurements (malformed actions, non-finite state, internal errors) latch
``EMERGENCY`` until an explicit ``reset()``.  The controller can use
:meth:`safe_action` as the last stage after the CBF so even a malformed shield
result cannot reach the plant.

Why valid-but-infeasible results do **not** latch by default
------------------------------------------------------------
Infeasibility means the requested plan cannot be made safe with the available
deceleration; the shield already applies ``ax_min`` in that case.  It is a
normal (if undesirable) outcome for aggressive baselines and for transients
such as a cut-in at close range.  Treating a run of infeasible ticks as an
irreversible emergency stop transforms a recoverable hazard into a permanent
freeze, which is itself unsafe in traffic.  Set ``latch_on_violation=True``
for a strictly conservative monitor that latches on any sustained violation.
"""

from __future__ import annotations

import copy
import logging
import math
from enum import Enum
from typing import Any, Optional

import numpy as np

from driving.config import ShieldConfig
from driving.errors import SafetyError
from driving.safety import ShieldResult

logger = logging.getLogger(__name__)


class SafetyStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def _coerce_action(action: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(action, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (2,) or not bool(np.all(np.isfinite(arr))):
        return None
    return arr.astype(np.float32, copy=True)


class SafetyMonitor:
    """Latching, fail-safe monitor over shield results.

    Parameters
    ----------
    violation_threshold:
        Number of consecutive unsafe shield results that latch EMERGENCY.
        Only consulted when ``latch_on_violation=True``.
    max_invalid_ticks:
        Number of consecutive invalid results (bad reason/action/state) that
        latch EMERGENCY.  Always active: malformed safety data is not
        recoverable by continuing to drive.
    latch_on_violation:
        Opt-in conservative policy.  When true, a run of ``violation_threshold``
        consecutive valid-but-unsafe results also latches EMERGENCY.  The
        default is false because normal infeasibility is handled by the shield
        (``ax_min``) and must not freeze the vehicle permanently.
    config:
        Optional :class:`ShieldConfig`, used for the actuator/emergency
        deceleration bounds.  It is copied and validated.
    """

    def __init__(
        self,
        violation_threshold: int = 3,
        max_invalid_ticks: int = 5,
        config: Optional[ShieldConfig] = None,
        *,
        latch_on_violation: bool = False,
        consecutive_violation_threshold: Optional[int] = None,
        ax_min: Optional[float] = None,
        ax_max: Optional[float] = None,
        emergency_decel: Optional[float] = None,
    ):
        if consecutive_violation_threshold is not None:
            violation_threshold = consecutive_violation_threshold
        try:
            self.violation_threshold = int(violation_threshold)
            self.max_invalid_ticks = int(max_invalid_ticks)
        except (TypeError, ValueError) as exc:
            raise SafetyError(f"invalid SafetyMonitor thresholds: {exc}") from exc
        if self.violation_threshold < 1:
            raise SafetyError("violation_threshold must be >= 1")
        if self.max_invalid_ticks < 1:
            raise SafetyError("max_invalid_ticks must be >= 1")
        self.latch_on_violation = bool(latch_on_violation)

        cfg = copy.deepcopy(config) if config is not None else ShieldConfig()
        cfg.validate("shield")

        self.ax_min = float(cfg.ax_min if ax_min is None else ax_min)
        self.ax_max = float(cfg.ax_max if ax_max is None else ax_max)
        self.emergency_decel = float(
            cfg.emergency_decel if emergency_decel is None else emergency_decel
        )
        for name, value in (
            ("ax_min", self.ax_min),
            ("ax_max", self.ax_max),
            ("emergency_decel", self.emergency_decel),
        ):
            if not math.isfinite(value):
                raise SafetyError(f"SafetyMonitor {name} must be finite")
        if self.ax_min >= self.ax_max:
            raise SafetyError("SafetyMonitor requires ax_min < ax_max")

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> SafetyStatus:
        """Clear all counters and return to OK."""

        self.status = SafetyStatus.OK
        self.total_ticks = 0
        self.ok_count = 0
        self.degraded_count = 0
        self.emergency_count = 0
        self.violations = 0
        self.invalid_ticks = 0
        self._violation_streak = 0
        self._invalid_streak = 0
        self.last_reason = ""
        self.last_h: Optional[float] = None
        return self.status

    def observe(self, result: ShieldResult) -> SafetyStatus:
        """Record one shield result and return the resulting status."""

        self.total_ticks += 1
        reason, invalid, unsafe = self._classify(result)
        self.last_reason = reason
        self.last_h = self._finite_or_none(getattr(result, "h", None))

        if unsafe:
            self.violations += 1
            self._violation_streak += 1
        else:
            self._violation_streak = 0

        if invalid:
            self.invalid_ticks += 1
            self._invalid_streak += 1
        else:
            self._invalid_streak = 0

        if self.status == SafetyStatus.EMERGENCY:
            # Safety latching: only reset() can leave EMERGENCY.
            self.emergency_count += 1
            return self.status

        violation_latch = (
            self.latch_on_violation and self._violation_streak >= self.violation_threshold
        )
        invalid_latch = self._invalid_streak >= self.max_invalid_ticks
        if violation_latch or invalid_latch:
            self.status = SafetyStatus.EMERGENCY
            self.emergency_count += 1
            logger.warning(
                "shield.monitor_emergency reason=%s violation_streak=%s invalid_streak=%s latch_on_violation=%s",
                reason,
                self._violation_streak,
                self._invalid_streak,
                self.latch_on_violation,
            )
        elif unsafe:
            # Valid but infeasible/unsafe: the shield's action (ax_min) is
            # still applied, but a transient does not cause a permanent stop.
            self.status = SafetyStatus.DEGRADED
            self.degraded_count += 1
        else:
            self.status = SafetyStatus.OK
            self.ok_count += 1
        return self.status

    def safe_action(self, requested_action: np.ndarray, shield_result: ShieldResult) -> np.ndarray:
        """Final gate: full emergency braking when latched, else shield action.

        ``requested_action`` is accepted for API symmetry/traceability; once a
        result has gone through the shield, the shield's action (not the raw
        request) is authoritative.
        """

        if self.status == SafetyStatus.EMERGENCY:
            return self._emergency_action()
        action = _coerce_action(getattr(shield_result, "action", None))
        if action is None:
            # A malformed shield output must never reach the plant.
            return self._emergency_action()
        return action

    def summary(self) -> dict:
        """Counters plus the last reason, suitable for logs/manifests."""

        return {
            "status": self.status.value,
            "last_reason": self.last_reason,
            "last_h": self.last_h,
            "counts": {
                "total": self.total_ticks,
                "ok": self.ok_count,
                "degraded": self.degraded_count,
                "emergency": self.emergency_count,
            },
            "total": self.total_ticks,
            "ok": self.ok_count,
            "degraded": self.degraded_count,
            "emergency": self.emergency_count,
            "violations": self.violations,
            "invalid_ticks": self.invalid_ticks,
            "consecutive_violations": self._violation_streak,
            "consecutive_invalid": self._invalid_streak,
            "violation_threshold": self.violation_threshold,
            "max_invalid_ticks": self.max_invalid_ticks,
            "latch_on_violation": self.latch_on_violation,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _finite_or_none(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    @staticmethod
    def _classify(result: ShieldResult):
        if result is None:
            return "invalid_result", True, True

        reason = str(getattr(result, "reason", "") or "")
        action = _coerce_action(getattr(result, "action", None))
        invalid = (
            action is None
            or reason.startswith("invalid")
            or bool(getattr(result, "emergency", False))
        )
        unsafe = (
            invalid
            or not bool(getattr(result, "feasible", False))
            or not bool(getattr(result, "safe", False))
        )
        if not reason:
            reason = "invalid_result" if invalid else "ok"
        return reason, invalid, unsafe

    def _emergency_action(self) -> np.ndarray:
        ax = float(np.clip(self.emergency_decel, self.ax_min, self.ax_max))
        return np.array([ax, 0.0], dtype=np.float32)
