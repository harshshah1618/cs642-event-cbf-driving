"""Baseline stacks sharing the scenario / metrics API.

Honest labels (these are implementations *in the style of* the papers, not
their original code):

1. ``fasionad_wakeup`` — uncertainty wake-up that keeps the same fast skill
   running. A router comment is applied as a temporary lateral nudge, not as a
   discrete skill switch. Static CBF only.
2. ``cert_replan`` — barrier-risk plan-class switching without a VLM. The
   plan classes are ``CRUISE``, ``LANE_CHANGE`` and ``CREEP_SAFE``.

Both use the same :class:`RolloutTracker` as the proposed stack, so crash,
freeze, jerk, pending, and safety metrics are directly comparable.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from driving import DT
from driving.config import BaselineMode, StackConfig
from driving.env_lane import LANE_Y
from driving.env_types import WorldSnapshot
from driving.errors import RouterError
from driving.pending import PendingCall
from driving.policy import SkillPolicy
from driving.result import RolloutResult, RolloutTracker
from driving.router_base import (
    RouterBackend,
    RouterRequest,
    response_record_fields,
    validate_response,
)
from driving.safety import InflatedCBFShield

__all__ = [
    "BaselineMode",
    "FasionadConfig",
    "CertReplanConfig",
    "M3BaselineStack",
]


@dataclass
class FasionadConfig:
    """Uncertainty wake-up → comment as lateral nudge (same skill)."""

    nudge_ay: float = 2.5
    nudge_hold_s: float = 1.2
    nudge_decay_s: float = 0.8
    min_call_ticks: int = 10
    skill_name: str = "CRUISE_RIGHT"


@dataclass
class CertReplanConfig:
    """Barrier-risk plan-class switch without a VLM."""

    residual_thresh: float = 0.4
    risk_hold_ticks: int = 15
    clear_ticks: int = 25
    lane_change_min_ticks: int = 60
    creep_target_vx: float = 3.0
    creep_ay_gain: float = 2.0
    creep_ax_gain: float = 1.5


class M3BaselineStack:
    """Runs FASIONAD- or CERT-Replan-style baselines; returns ``RolloutResult``."""

    def __init__(
        self,
        skills: Dict[str, SkillPolicy],
        router: RouterBackend,
        shield: Optional[InflatedCBFShield] = None,
        mode: BaselineMode | str = BaselineMode.FASIONAD,
        fasionad: Optional[FasionadConfig] = None,
        cert: Optional[CertReplanConfig] = None,
        config: Optional[StackConfig] = None,
    ):
        self.skills = dict(skills)
        self.alias_used = False
        if "RETURN_RIGHT" not in self.skills and "CRUISE_RIGHT" in self.skills:
            self.skills["RETURN_RIGHT"] = self.skills["CRUISE_RIGHT"]
            self.alias_used = True
        self.router = router
        self.shield = shield or InflatedCBFShield(enabled=True, inflate=False)
        self.mode = BaselineMode(mode) if not isinstance(mode, BaselineMode) else mode
        self.fasionad = fasionad or FasionadConfig()
        self.cert = cert or CertReplanConfig()
        self.config = config or StackConfig()
        self.config.validate("stack")
        self.last_shield: Optional[InflatedCBFShield] = None

    def run(self, env, seed: Optional[int] = None) -> RolloutResult:
        if self.mode == BaselineMode.FASIONAD:
            return self._run_fasionad(env, seed=seed)
        return self._run_cert(env, seed=seed)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _skill_action(self, name: str, snap: WorldSnapshot):
        if name not in self.skills:
            raise KeyError(f"skill {name!r} is not loaded")
        ax, ay, sigma = self.skills[name].act(
            snap.ego.y, snap.ego.vy, snap.ego.vx, deterministic=True
        )
        action = np.array([ax, ay], dtype=np.float32)
        if not np.all(np.isfinite(action)) or not math.isfinite(float(sigma)):
            raise ValueError(f"skill {name!r} produced a non-finite action or sigma")
        return action, float(sigma), self.skills[name]

    def _creep_action(self, snap: WorldSnapshot) -> np.ndarray:
        ego = snap.ego
        lane_y = LANE_Y[0] if abs(ego.y - LANE_Y[0]) <= abs(ego.y - LANE_Y[1]) else LANE_Y[1]
        ay = float(np.clip(self.cert.creep_ay_gain * (lane_y - ego.y), -3.0, 3.0))
        ax = float(
            np.clip(self.cert.creep_ax_gain * (self.cert.creep_target_vx - ego.vx), -6.0, 2.0)
        )
        return np.array([ax, ay], dtype=np.float32)

    def _lane_nudge_sign(self, suggested_skill: str, snap: WorldSnapshot) -> float:
        target = LANE_Y[1] if suggested_skill == "OVERTAKE_LEFT" else LANE_Y[0]
        dy = target - snap.ego.y
        if abs(dy) < 0.15:
            return 0.0
        return 1.0 if dy > 0 else -1.0

    @staticmethod
    def _end_reason(crashed: bool, term: bool, trunc: bool, snap2: WorldSnapshot) -> str:
        if crashed:
            return "crash"
        if term:
            return (
                "route_complete"
                if float(snap2.ego.x) >= float(snap2.route_length)
                else "terminated"
            )
        if trunc:
            return "timeout"
        return "unknown"

    # ------------------------------------------------------------------
    # FASIONAD-style
    # ------------------------------------------------------------------
    def _run_fasionad(self, env, seed: Optional[int] = None) -> RolloutResult:
        cfg = self.fasionad
        shield = copy.deepcopy(self.shield)
        shield.enabled = True
        shield.inflate = False
        tracker = RolloutTracker(BaselineMode.FASIONAD.value, self.config, scenario="", seed=seed)

        _obs, reset_info = env.reset(seed=seed)
        snap = env.snapshot()
        tracker.scenario = str(snap.info.get("scenario", ""))
        if isinstance(reset_info, dict) and reset_info.get("params"):
            tracker.params = dict(reset_info["params"])

        skill = cfg.skill_name
        pending: Optional[PendingCall] = None
        since_call = 10_000
        sigma_latch = False
        nudge_ay = 0.0
        nudge_age = 0.0
        crashed = False
        crash_time: Optional[float] = None
        end_reason = "timeout"
        run_token = "fasionad"

        while True:
            snap = env.snapshot()
            t = float(snap.t)

            if pending is not None and pending.is_ready(t):
                sign = self._lane_nudge_sign(pending.response.skill, snap)
                nudge_ay = cfg.nudge_ay * sign
                nudge_age = 0.0
                pending.record["applied"] = True
                pending.record["applied_as"] = "lateral_nudge"
                tracker.calls_applied += 1
                pending = None
            elif pending is not None and pending.is_expired(t):
                pending.record["rejected"] = True
                pending.record["error"] = "pending call expired"
                tracker.calls_rejected += 1
                pending = None

            if abs(nudge_ay) > 1e-6:
                nudge_age += DT
                if nudge_age > cfg.nudge_hold_s:
                    rem = nudge_age - cfg.nudge_hold_s
                    if rem >= cfg.nudge_decay_s or cfg.nudge_decay_s <= 0:
                        nudge_ay = 0.0
                    else:
                        scale = 1.0 - rem / cfg.nudge_decay_s
                        nudge_ay = np.sign(nudge_ay) * cfg.nudge_ay * scale

            a_rl, sigma, pol = self._skill_action(skill, snap)
            a_rl = a_rl.copy()
            a_rl[1] = float(np.clip(a_rl[1] + nudge_ay, -5.0, 5.0))
            sr = shield.filter(a_rl, snap, sigma=0.0, vlm_pending=False)

            thr = pol.meta.threshold
            sigma_high = sigma > thr
            can_call = pending is None and since_call >= cfg.min_call_ticks
            if can_call and sigma_high and not sigma_latch:
                request = RouterRequest(
                    t=t,
                    current_skill=skill,
                    snapshot=snap,
                    sigma=sigma,
                    reason="fasionad_sigma",
                    request_id=f"{run_token}-{tracker.call_attempts + 1:03d}",
                )
                record = {
                    "t": t,
                    "from": skill,
                    "to": skill,
                    "reason": "fasionad_sigma_nudge",
                    "sigma": sigma,
                    "backend": getattr(self.router, "name", "unknown"),
                    "ready_at": None,
                    "applied": False,
                    "rejected": False,
                    "fallback_used": False,
                    "wall_latency_ms": None,
                    "prompt_hash": "",
                    "error": None,
                    "request_id": request.request_id,
                    "applied_as": "lateral_nudge",
                }
                try:
                    response = self.router.submit(request)
                    validate_response(response, request)
                except RouterError as exc:
                    record["rejected"] = True
                    record["error"] = str(exc)
                    tracker.add_call(record)
                else:
                    record["to"] = response.skill
                    record.update(response_record_fields(response))
                    tracker.add_call(record)
                    pending = PendingCall(
                        request=request,
                        response=response,
                        submitted_t=t,
                        deadline_t=t + self.config.max_pending_s,
                        record=record,
                    )
                since_call = 0
                sigma_latch = True
            if not sigma_high:
                sigma_latch = False

            tracker.record_tick(
                snap,
                sr.action,
                sr,
                skill,
                sigma=sigma,
                vlm_pending=False,
            )
            _obs, _reward, term, trunc, info = env.step(sr.action)
            snap2 = info.get("snapshot") or env.snapshot()
            since_call += 1
            if snap2.crashed or info.get("crashed"):
                crashed = True
                crash_time = float(snap2.t)
            end_reason = self._end_reason(crashed, term, trunc, snap2)
            if crashed or term or trunc:
                break

        result = tracker.finalize(
            env, crashed=crashed, crash_time=crash_time, end_reason=end_reason
        )
        result.scenario = tracker.scenario
        result.seed = seed
        result.params = dict(tracker.params)
        self.last_shield = shield
        return result

    # ------------------------------------------------------------------
    # CERT-Replan-style
    # ------------------------------------------------------------------
    def _run_cert(self, env, seed: Optional[int] = None) -> RolloutResult:
        cfg = self.cert
        shield = copy.deepcopy(self.shield)
        shield.enabled = True
        shield.inflate = False
        tracker = RolloutTracker(BaselineMode.CERT.value, self.config, scenario="", seed=seed)

        _obs, reset_info = env.reset(seed=seed)
        snap = env.snapshot()
        tracker.scenario = str(snap.info.get("scenario", ""))
        if isinstance(reset_info, dict) and reset_info.get("params"):
            tracker.params = dict(reset_info["params"])

        plan = "CRUISE"
        plan_age = 0
        risk_streak = 0
        clear_streak = 0
        plan_switches: List[dict] = []
        crashed = False
        crash_time: Optional[float] = None
        end_reason = "timeout"

        while True:
            snap = env.snapshot()
            t = float(snap.t)
            plan_age += 1

            if plan == "CREEP_SAFE":
                a_rl = self._creep_action(snap)
                sigma = 0.0
                skill_label = "CREEP_SAFE"
            elif plan == "LANE_CHANGE":
                a_rl, sigma, _ = self._skill_action("OVERTAKE_LEFT", snap)
                skill_label = "OVERTAKE_LEFT"
            else:
                a_rl, sigma, _ = self._skill_action("CRUISE_RIGHT", snap)
                skill_label = "CRUISE_RIGHT"

            sr = shield.filter(a_rl, snap, sigma=0.0, vlm_pending=False)
            risky = (sr.overridden and sr.residual >= cfg.residual_thresh) or sr.h < 0.0
            if risky:
                risk_streak += 1
                clear_streak = 0
            else:
                clear_streak += 1
                if clear_streak >= cfg.clear_ticks:
                    risk_streak = 0

            if risk_streak >= cfg.risk_hold_ticks:
                old = plan
                if plan == "CRUISE":
                    plan = "LANE_CHANGE"
                elif plan == "LANE_CHANGE" and plan_age >= cfg.lane_change_min_ticks:
                    plan = "CREEP_SAFE"
                if plan != old:
                    plan_switches.append(
                        {
                            "t": t,
                            "from": old,
                            "to": plan,
                            "reason": "barrier_risk",
                            "residual": sr.residual,
                            "h": sr.h if math.isfinite(sr.h) else None,
                        }
                    )
                    plan_age = 0
                risk_streak = 0
                clear_streak = 0

            if plan == "CREEP_SAFE" and clear_streak >= cfg.clear_ticks * 2 and not risky:
                plan_switches.append(
                    {
                        "t": t,
                        "from": "CREEP_SAFE",
                        "to": "CRUISE",
                        "reason": "risk_cleared",
                        "residual": sr.residual,
                        "h": sr.h if math.isfinite(sr.h) else None,
                    }
                )
                plan = "CRUISE"
                plan_age = 0
                clear_streak = 0

            tracker.record_tick(snap, sr.action, sr, skill_label, sigma=sigma, vlm_pending=False)
            _obs, _reward, term, trunc, info = env.step(sr.action)
            snap2 = info.get("snapshot") or env.snapshot()
            if snap2.crashed or info.get("crashed"):
                crashed = True
                crash_time = float(snap2.t)
            end_reason = self._end_reason(crashed, term, trunc, snap2)
            if crashed or term or trunc:
                break

        result = tracker.finalize(
            env, crashed=crashed, crash_time=crash_time, end_reason=end_reason
        )
        result.scenario = tracker.scenario
        result.seed = seed
        result.params = dict(tracker.params)
        result.calls = plan_switches  # diagnostic plan switches, not VLM calls
        self.last_shield = shield
        return result
