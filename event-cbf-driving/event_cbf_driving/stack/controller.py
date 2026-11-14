"""3-tier driving stack: event router + PPO skills + inflated CBF."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from event_cbf_driving import DT
from event_cbf_driving.env.types import WorldSnapshot
from event_cbf_driving.router.base import RouterBackend, RouterRequest, RouterResponse
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.skills.ppo import SkillPolicy
from event_cbf_driving.stack.blend import blend_actions, sigmoid_blend_alpha


class StackMode(str, Enum):
    UNSHIELDED = "unshielded_skill"
    STATIC_CBF = "static_cbf"
    FIXED_VLM = "fixed_rate_vlm"
    PROPOSED = "proposed"


@dataclass
class StackConfig:
    mode: StackMode = StackMode.PROPOSED
    initial_skill: str = "CRUISE_RIGHT"
    blend_ticks: int = 24
    fixed_rate_hz: float = 1.0
    # Rising-edge shield interrupt: require this many clear ticks before edge.
    shield_clear_ticks: int = 16
    # Min ticks between router submits (50 Hz → 28 ticks ≈ 0.56 s).
    min_call_ticks: int = 28
    # Multiply skill meta.threshold for interrupt (higher → fewer σ calls).
    sigma_threshold_scale: float = 1.55
    use_blend: bool = True
    hard_switch: bool = False
    # Ablation (PROPOSED mode only): if set, call the router on a timer at this rate instead of on sigma/shield
    # events. Shield, inflation and blending stay as in PROPOSED. None = event-triggered (default).
    periodic_hz: Optional[float] = None


@dataclass
class RolloutResult:
    mode: str
    crashed: bool
    crash_time: Optional[float]
    route_pct: float
    avg_speed: float
    router_calls: int
    peak_lateral_jerk: float
    peak_sigma: float
    freeze: bool
    x: float
    y: float
    vx: float
    end_time: float
    calls: List[dict] = field(default_factory=list)
    xs: List[float] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)


class DrivingStack:
    def __init__(
        self,
        skills: Dict[str, SkillPolicy],
        router: RouterBackend,
        shield: Optional[InflatedCBFShield] = None,
        config: Optional[StackConfig] = None,
    ):
        self.skills = skills
        self.router = router
        self.shield = shield or InflatedCBFShield()
        self.config = config or StackConfig()
        if "RETURN_RIGHT" not in self.skills and "CRUISE_RIGHT" in self.skills:
            self.skills["RETURN_RIGHT"] = self.skills["CRUISE_RIGHT"]

    def _skill_action(self, name: str, snap: WorldSnapshot):
        pol = self.skills[name]
        ax, ay, sigma = pol.act(snap.ego.y, snap.ego.vy, snap.ego.vx, deterministic=True)
        return np.array([ax, ay], dtype=np.float32), float(sigma), pol

    def run(self, env, seed: Optional[int] = None) -> RolloutResult:
        cfg = self.config
        mode = cfg.mode
        obs, _ = env.reset(seed=seed)
        snap: WorldSnapshot = env.snapshot()

        skill = cfg.initial_skill
        pending: Optional[RouterResponse] = None
        blend_from: Optional[str] = None
        blend_tick = 0
        blending = False
        override_off = 0
        since_call = 10_000
        sigma_latch = False
        last_fixed_t = -1e9
        last_ay = 0.0
        peak_jerk = 0.0
        peak_sigma = 0.0
        calls: List[dict] = []
        xs, ys, skill_trace = [], [], []
        crashed = False
        crash_time = None
        dist = 0.0
        x0 = snap.ego.x

        if mode == StackMode.UNSHIELDED:
            self.shield.enabled = False
            self.shield.inflate = False
        elif mode == StackMode.STATIC_CBF:
            self.shield.enabled = True
            self.shield.inflate = False
        elif mode == StackMode.FIXED_VLM:
            self.shield.enabled = False
            self.shield.inflate = False
        else:
            self.shield.enabled = True
            self.shield.inflate = True

        while True:
            snap = env.snapshot()
            t = snap.t
            vlm_pending = pending is not None and t < pending.ready_at

            # Deliver pending router response.
            if pending is not None and t >= pending.ready_at:
                new_skill = pending.skill
                if new_skill != skill:
                    if cfg.hard_switch or mode == StackMode.FIXED_VLM or not cfg.use_blend:
                        skill = new_skill
                        blending = False
                        blend_from = None
                    else:
                        blend_from = skill
                        skill = new_skill
                        blend_tick = 0
                        blending = True
                pending = None

            # Active skill action (+ optional blend).
            a_new, sigma, pol = self._skill_action(skill, snap)
            sigma_skill = skill
            if blending and blend_from is not None:
                a_old, s_old, _ = self._skill_action(blend_from, snap)
                alpha = sigmoid_blend_alpha(blend_tick, cfg.blend_ticks)
                a_rl = blend_actions(a_old, a_new, alpha)
                if alpha >= 0.5:
                    sigma, sigma_skill = sigma, skill
                else:
                    sigma, sigma_skill = s_old, blend_from
                    pol = self.skills[blend_from]
                blend_tick += 1
                if blend_tick >= cfg.blend_ticks:
                    blending = False
                    blend_from = None
            else:
                a_rl = a_new

            peak_sigma = max(peak_sigma, sigma)

            sr = self.shield.filter(
                a_rl,
                snap,
                sigma=sigma if mode == StackMode.PROPOSED else 0.0,
                vlm_pending=vlm_pending,
            )
            action = sr.action
            overriding = sr.overridden

            # Router
            if mode == StackMode.FIXED_VLM:
                if t - last_fixed_t >= 1.0 / cfg.fixed_rate_hz - 1e-9 and pending is None and not blending:
                    last_fixed_t = t
                    req = RouterRequest(
                        t=t, current_skill=skill, snapshot=snap, sigma=sigma, reason="fixed_rate"
                    )
                    resp = self.router.submit(req)
                    calls.append(
                        {
                            "t": t,
                            "from": skill,
                            "to": resp.skill,
                            "reason": "fixed_rate",
                            "sigma": sigma,
                            "backend": resp.backend,
                            "ready_at": resp.ready_at,
                        }
                    )
                    if resp.skill != skill:
                        pending = resp
            elif mode == StackMode.PROPOSED:
                thr = self.skills[sigma_skill].meta.threshold * cfg.sigma_threshold_scale
                sigma_high = sigma > thr
                shield_rise = overriding and override_off >= cfg.shield_clear_ticks
                can_call = (
                    pending is None
                    and not blending
                    and since_call >= cfg.min_call_ticks
                )
                if cfg.periodic_hz:
                    fire = pending is None and not blending and t - last_fixed_t >= 1.0 / cfg.periodic_hz - 1e-9
                else:
                    fire = can_call and (shield_rise or (sigma_high and not sigma_latch))
                if fire:
                    if cfg.periodic_hz:
                        last_fixed_t = t
                    reason = (
                        "periodic"
                        if cfg.periodic_hz
                        else "shield+sigma"
                        if (overriding and sigma_high)
                        else ("shield" if overriding else "sigma")
                    )
                    req = RouterRequest(
                        t=t, current_skill=skill, snapshot=snap, sigma=sigma, reason=reason
                    )
                    resp = self.router.submit(req)
                    since_call = 0
                    calls.append(
                        {
                            "t": t,
                            "from": skill,
                            "to": resp.skill,
                            "reason": reason,
                            "sigma": sigma,
                            "backend": resp.backend,
                            "ready_at": resp.ready_at,
                        }
                    )
                    if resp.skill != skill:
                        pending = resp
                    elif sigma_high and not shield_rise:
                        sigma_latch = True
                if not sigma_high:
                    sigma_latch = False
                override_off = 0 if overriding else override_off + 1
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

        final = env.snapshot()
        ego = final.ego
        end_t = max(final.t, DT)
        route_pct = 100.0 * min(ego.x, final.route_length) / final.route_length
        avg_speed = float(dist / end_t) if end_t > 0 else 0.0
        freeze = (not crashed) and ego.vx < 0.5 and route_pct < 90.0

        return RolloutResult(
            mode=mode.value,
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
            skills=skill_trace,
        )
