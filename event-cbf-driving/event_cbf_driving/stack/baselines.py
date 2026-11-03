"""M3-style baselines sharing the DrivingStack scenario / metrics API.

Honest labels (not the original papers' code):

1. **FASIONAD-style** (`fasionad_wakeup`)
   Uncertainty wake-up that keeps the *same* fast skill bank entry running
   (always ``CRUISE_RIGHT``). When critic σ crosses the skill's training
   threshold, a router call is submitted (same interrupt API + simulated
   latency). When the answer arrives it is **not** applied as a discrete
   CRUISE/OVERTAKE skill switch. Instead the suggested lane is turned into a
   **temporary lateral bias / waypoint nudge** added on top of the continuing
   cruise action. Static CBF is on (no σ/latency inflation). Router calls are
   counted.

2. **CERT-Replan-style** (`cert_replan`)
   Pure barrier-risk replan with **no VLM/router**. Static CBF stays on. When
   barrier residual stays high (sustained override with residual above a
   threshold, or barrier value ``h`` below zero) for a hold window, the
   nominal plan class switches among:
   - ``CRUISE`` → PPO ``CRUISE_RIGHT``
   - ``LANE_CHANGE`` → PPO ``OVERTAKE_LEFT`` (min dwell ~1.2 s before CREEP)
   - ``CREEP_SAFE`` → open-loop creep (hold current lane center, target ~3 m/s)
   No router calls. Hard plan-class switch (no sigmoid blend).

These modes are additive; they do not alter the four M1/M2 ablation modes in
``controller.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from event_cbf_driving import DT
from event_cbf_driving.env.skill_lane import LANE_Y
from event_cbf_driving.env.types import WorldSnapshot
from event_cbf_driving.router.base import RouterBackend, RouterRequest, RouterResponse
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.skills.ppo import SkillPolicy
from event_cbf_driving.stack.controller import RolloutResult


class BaselineMode(str, Enum):
    FASIONAD = "fasionad_wakeup"
    CERT_REPLAN = "cert_replan"


@dataclass
class FasionadConfig:
    """Uncertainty wake-up → comment as lateral nudge (same skill)."""

    nudge_ay: float = 2.5  # m/s^2 added toward suggested lane
    nudge_hold_s: float = 1.2  # how long the nudge stays at full strength
    nudge_decay_s: float = 0.8  # linear decay after hold
    min_call_ticks: int = 10
    skill_name: str = "CRUISE_RIGHT"


@dataclass
class CertReplanConfig:
    """Barrier-risk plan-class switch without a VLM."""

    residual_thresh: float = 0.4  # m/s^2 ax clipped away by CBF
    risk_hold_ticks: int = 15  # 0.30 s at 50 Hz before first escalate
    clear_ticks: int = 25  # risk must clear this long before escalating again
    # Hold LANE_CHANGE at least this long before escalating to CREEP_SAFE so the
    # corridor change has time to act (still pure barrier-risk, no VLM).
    lane_change_min_ticks: int = 60  # 1.2 s
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
        mode: BaselineMode = BaselineMode.FASIONAD,
        fasionad: Optional[FasionadConfig] = None,
        cert: Optional[CertReplanConfig] = None,
    ):
        self.skills = skills
        self.router = router
        self.shield = shield or InflatedCBFShield(enabled=True, inflate=False)
        self.mode = mode
        self.fasionad = fasionad or FasionadConfig()
        self.cert = cert or CertReplanConfig()
        if "RETURN_RIGHT" not in self.skills and "CRUISE_RIGHT" in self.skills:
            self.skills["RETURN_RIGHT"] = self.skills["CRUISE_RIGHT"]

    def _skill_action(self, name: str, snap: WorldSnapshot):
        pol = self.skills[name]
        ax, ay, sigma = pol.act(snap.ego.y, snap.ego.vy, snap.ego.vx, deterministic=True)
        return np.array([ax, ay], dtype=np.float32), float(sigma), pol

    def _creep_action(self, snap: WorldSnapshot) -> np.ndarray:
        """Open-loop CREEP_SAFE plan class: slow + hold nearest lane center."""
        ego = snap.ego
        # Nearest lane center.
        lane_y = LANE_Y[0] if abs(ego.y - LANE_Y[0]) <= abs(ego.y - LANE_Y[1]) else LANE_Y[1]
        ay = float(np.clip(self.cert.creep_ay_gain * (lane_y - ego.y), -3.0, 3.0))
        ax = float(np.clip(self.cert.creep_ax_gain * (self.cert.creep_target_vx - ego.vx), -6.0, 2.0))
        return np.array([ax, ay], dtype=np.float32)

    def _lane_nudge_sign(self, suggested_skill: str, snap: WorldSnapshot) -> float:
        """Map router 'comment' (suggested skill) → signed lateral bias."""
        if suggested_skill == "OVERTAKE_LEFT":
            target = LANE_Y[1]
        else:
            target = LANE_Y[0]
        dy = target - snap.ego.y
        if abs(dy) < 0.15:
            return 0.0
        return 1.0 if dy > 0 else -1.0

    def run(self, env, seed: Optional[int] = None) -> RolloutResult:
        if self.mode == BaselineMode.FASIONAD:
            return self._run_fasionad(env, seed=seed)
        return self._run_cert(env, seed=seed)

    def _run_fasionad(self, env, seed: Optional[int] = None) -> RolloutResult:
        cfg = self.fasionad
        obs, _ = env.reset(seed=seed)
        snap: WorldSnapshot = env.snapshot()

        skill = cfg.skill_name
        pending: Optional[RouterResponse] = None
        since_call = 10_000
        sigma_latch = False
        nudge_ay = 0.0
        nudge_age = 0.0
        last_ay = 0.0
        peak_jerk = 0.0
        peak_sigma = 0.0
        calls: List[dict] = []
        xs, ys, skill_trace = [], [], []
        crashed = False
        crash_time = None
        dist = 0.0
        x0 = snap.ego.x

        # Static CBF, no inflation; same skill always.
        self.shield.enabled = True
        self.shield.inflate = False

        while True:
            snap = env.snapshot()
            t = snap.t

            # Deliver pending "comment" as lateral nudge — never switch skill.
            if pending is not None and t >= pending.ready_at:
                sign = self._lane_nudge_sign(pending.skill, snap)
                nudge_ay = cfg.nudge_ay * sign
                nudge_age = 0.0
                pending = None

            # Decay nudge after hold window.
            if abs(nudge_ay) > 1e-6:
                nudge_age += DT
                if nudge_age > cfg.nudge_hold_s:
                    rem = nudge_age - cfg.nudge_hold_s
                    if rem >= cfg.nudge_decay_s or cfg.nudge_decay_s <= 0:
                        nudge_ay = 0.0
                    else:
                        scale = 1.0 - rem / cfg.nudge_decay_s
                        # Keep sign; magnitude from initial peak.
                        peak = cfg.nudge_ay
                        nudge_ay = np.sign(nudge_ay) * peak * scale

            a_rl, sigma, pol = self._skill_action(skill, snap)
            a_rl = a_rl.copy()
            a_rl[1] = float(np.clip(a_rl[1] + nudge_ay, -5.0, 5.0))
            peak_sigma = max(peak_sigma, sigma)

            sr = self.shield.filter(a_rl, snap, sigma=0.0, vlm_pending=False)
            action = sr.action

            # Uncertainty wake-up only (FASIONAD-style); no shield-triggered skill bank.
            thr = pol.meta.threshold
            sigma_high = sigma > thr
            can_call = pending is None and since_call >= cfg.min_call_ticks
            if can_call and sigma_high and not sigma_latch:
                req = RouterRequest(
                    t=t,
                    current_skill=skill,
                    snapshot=snap,
                    sigma=sigma,
                    reason="fasionad_sigma",
                )
                resp = self.router.submit(req)
                since_call = 0
                sigma_latch = True
                calls.append(
                    {
                        "t": t,
                        "from": skill,
                        "to": resp.skill,
                        "reason": "fasionad_sigma_nudge",
                        "sigma": sigma,
                        "backend": resp.backend,
                        "ready_at": resp.ready_at,
                        "applied_as": "lateral_nudge",
                    }
                )
                pending = resp
            if not sigma_high:
                sigma_latch = False
            since_call += 1

            jerk = abs(float(action[1]) - last_ay) / DT
            peak_jerk = max(peak_jerk, jerk)
            last_ay = float(action[1])

            xs.append(snap.ego.x)
            ys.append(snap.ego.y)
            skill_trace.append(skill)

            obs, reward, term, trunc, info = env.step(action)
            snap2 = info.get("snapshot") or env.snapshot()
            dist = snap2.ego.x - x0
            if snap2.crashed or info.get("crashed"):
                crashed = True
                crash_time = snap2.t
                xs.append(snap2.ego.x)
                ys.append(snap2.ego.y)
                break
            if term or trunc:
                xs.append(snap2.ego.x)
                ys.append(snap2.ego.y)
                break

        return self._finalize(
            env,
            mode=BaselineMode.FASIONAD.value,
            crashed=crashed,
            crash_time=crash_time,
            dist=dist,
            x0=x0,
            peak_jerk=peak_jerk,
            peak_sigma=peak_sigma,
            calls=calls,
            xs=xs,
            ys=ys,
            skills=skill_trace,
        )

    def _run_cert(self, env, seed: Optional[int] = None) -> RolloutResult:
        cfg = self.cert
        obs, _ = env.reset(seed=seed)
        snap: WorldSnapshot = env.snapshot()

        # Plan classes (not VLM skills): cruise / lane_change / creep_safe
        plan = "CRUISE"
        plan_age = 0
        risk_streak = 0
        clear_streak = 0
        last_ay = 0.0
        peak_jerk = 0.0
        peak_sigma = 0.0
        calls: List[dict] = []  # no VLM; log plan switches instead as calls=0
        plan_switches: List[dict] = []
        xs, ys, skill_trace = [], [], []
        crashed = False
        crash_time = None
        dist = 0.0
        x0 = snap.ego.x

        self.shield.enabled = True
        self.shield.inflate = False

        while True:
            snap = env.snapshot()
            t = snap.t
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

            peak_sigma = max(peak_sigma, sigma)
            sr = self.shield.filter(a_rl, snap, sigma=0.0, vlm_pending=False)
            action = sr.action

            # Barrier risk: sustained override with material residual, or h < 0.
            risky = (sr.overridden and sr.residual >= cfg.residual_thresh) or (sr.h < 0.0)
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
                elif plan == "CREEP_SAFE":
                    plan = "CREEP_SAFE"
                # else: still in LANE_CHANGE min dwell — keep trying corridor.
                if plan != old:
                    plan_switches.append(
                        {
                            "t": t,
                            "from": old,
                            "to": plan,
                            "reason": "barrier_risk",
                            "residual": sr.residual,
                            "h": sr.h,
                        }
                    )
                    plan_age = 0
                risk_streak = 0
                clear_streak = 0

            # Optional recovery: if creeping and risk has been clear, return to cruise.
            if plan == "CREEP_SAFE" and clear_streak >= cfg.clear_ticks * 2 and not risky:
                plan_switches.append(
                    {
                        "t": t,
                        "from": "CREEP_SAFE",
                        "to": "CRUISE",
                        "reason": "risk_cleared",
                        "residual": sr.residual,
                        "h": sr.h,
                    }
                )
                plan = "CRUISE"
                plan_age = 0
                clear_streak = 0

            jerk = abs(float(action[1]) - last_ay) / DT
            peak_jerk = max(peak_jerk, jerk)
            last_ay = float(action[1])

            xs.append(snap.ego.x)
            ys.append(snap.ego.y)
            skill_trace.append(skill_label)

            obs, reward, term, trunc, info = env.step(action)
            snap2 = info.get("snapshot") or env.snapshot()
            dist = snap2.ego.x - x0
            if snap2.crashed or info.get("crashed"):
                crashed = True
                crash_time = snap2.t
                xs.append(snap2.ego.x)
                ys.append(snap2.ego.y)
                break
            if term or trunc:
                xs.append(snap2.ego.x)
                ys.append(snap2.ego.y)
                break

        # Store plan switches in calls list for debugging, but router_calls = 0.
        result = self._finalize(
            env,
            mode=BaselineMode.CERT_REPLAN.value,
            crashed=crashed,
            crash_time=crash_time,
            dist=dist,
            x0=x0,
            peak_jerk=peak_jerk,
            peak_sigma=peak_sigma,
            calls=[],  # no VLM
            xs=xs,
            ys=ys,
            skills=skill_trace,
        )
        # Attach plan switches without counting them as router calls.
        result.calls = plan_switches
        return result

    def _finalize(
        self,
        env,
        *,
        mode: str,
        crashed: bool,
        crash_time: Optional[float],
        dist: float,
        x0: float,
        peak_jerk: float,
        peak_sigma: float,
        calls: List[dict],
        xs: List[float],
        ys: List[float],
        skills: List[str],
    ) -> RolloutResult:
        final = env.snapshot()
        ego = final.ego
        end_t = max(final.t, DT)
        route_pct = 100.0 * min(ego.x, final.route_length) / final.route_length
        avg_speed = float(dist / end_t) if end_t > 0 else 0.0
        freeze = (not crashed) and ego.vx < 0.5 and route_pct < 90.0
        return RolloutResult(
            mode=mode,
            crashed=crashed,
            crash_time=crash_time,
            route_pct=route_pct,
            avg_speed=avg_speed,
            router_calls=len(calls),
            peak_lateral_jerk=peak_jerk,
            peak_sigma=peak_sigma,
            freeze=freeze,
            x=ego.x,
            y=ego.y,
            vx=ego.vx,
            end_time=end_t,
            calls=calls,
            xs=xs,
            ys=ys,
            skills=skills,
        )
