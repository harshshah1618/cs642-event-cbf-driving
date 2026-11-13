"""Rollout result schema and the single metrics accumulator.

Both the proposed stack and the baseline stacks use this class, so every
reported number is computed the same way. Metrics are recorded per tick and
finalized once, instead of being re-derived from ad-hoc local variables in
each controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config import StackConfig
from constants import DT
from env_types import WorldSnapshot
from safety import ShieldResult


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
    # Per-tick [ax, ay] actions, recorded for deterministic open-loop replay.
    actions: List[List[float]] = field(default_factory=list)
    interactions: List[dict] = field(default_factory=list)
    plant_peak_lateral_jerk: Optional[float] = None
    acceleration_tracking_rmse: Optional[float] = None
    actuator_saturation_fraction: Optional[float] = None
    # Production observability (all optional so old consumers keep working).
    mean_lateral_jerk: float = 0.0
    peak_lateral_jerk_after_warmup: float = 0.0
    override_fraction: float = 0.0
    pending_ticks: int = 0
    pending_survival: bool = True
    call_attempts: int = 0
    calls_applied: int = 0
    calls_rejected: int = 0
    calls_suppressed: int = 0
    router_fallbacks: int = 0
    router_prompt_tokens: int = 0
    router_completion_tokens: int = 0
    router_tokens: int = 0
    router_live_calls: int = 0
    router_models: List[str] = field(default_factory=list)
    max_pending_s: float = 0.0
    safety_events: int = 0
    infeasible_ticks: int = 0
    min_barrier_h: float = float("inf")
    min_braking_h: Optional[float] = None
    braking_infeasible_ticks: int = 0
    wall_latency_ms: List[float] = field(default_factory=list)
    end_reason: str = "unknown"
    seed: Optional[int] = None
    scenario: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["min_barrier_h"] = (
            None if not math.isfinite(float(self.min_barrier_h)) else float(self.min_barrier_h)
        )
        return payload


class RolloutTracker:
    """Accumulates per-tick metrics and produces one :class:`RolloutResult`."""

    def __init__(self, mode: str, config: StackConfig, *, scenario: str = "", seed: Optional[int] = None):
        self.mode = mode
        self.config = config
        self.scenario = scenario
        self.seed = seed
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.skills: List[str] = []
        self.actions: List[List[float]] = []
        self.interactions: List[dict] = []
        self.calls: List[dict] = []
        self.wall_latency_ms: List[float] = []
        self.params: Dict[str, Any] = {}

        self.last_ay = 0.0
        self.peak_jerk = 0.0
        self.peak_jerk_warmup = 0.0
        self.jerk_sum = 0.0
        self.jerk_count = 0
        self.peak_sigma = 0.0
        self.ticks = 0
        self.override_ticks = 0
        self.pending_ticks = 0
        self.safety_events = 0
        self.infeasible_ticks = 0
        self.min_h = float("inf")
        self.min_braking_h: Optional[float] = None
        self.braking_infeasible_ticks = 0
        self.pending_survival = True
        self.low_speed_streak = 0
        self.freeze_seen = False
        self.max_pending_s = 0.0
        self.call_attempts = 0
        self.calls_applied = 0
        self.calls_rejected = 0
        self.calls_suppressed = 0
        self.router_fallbacks = 0
        self.router_prompt_tokens = 0
        self.router_completion_tokens = 0
        self.router_tokens = 0
        self.router_live_calls = 0
        self.router_models: List[str] = []
        self.start_x: Optional[float] = None

    def add_call(self, record: dict) -> None:
        self.calls.append(record)
        self.call_attempts += 1
        if record.get("applied"):
            self.calls_applied += 1
        if record.get("rejected"):
            self.calls_rejected += 1
        self.record_router_response(record)

    def record_router_response(self, record: dict) -> None:
        """Account for response telemetry once, including asynchronously completed calls."""
        if record.get("fallback_used"):
            self.router_fallbacks += 1
        if record.get("live"):
            self.router_live_calls += 1
        model = record.get("model")
        if model and str(model) not in self.router_models:
            self.router_models.append(str(model))
        for key, attribute in (
            ("prompt_tokens", "router_prompt_tokens"),
            ("completion_tokens", "router_completion_tokens"),
            ("total_tokens", "router_tokens"),
        ):
            value = record.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number >= 0:
                setattr(self, attribute, getattr(self, attribute) + number)
        latency = record.get("wall_latency_ms")
        if latency is not None and math.isfinite(float(latency)):
            self.wall_latency_ms.append(float(latency))

    def suppress_call(self, record: dict) -> None:
        self.calls_suppressed += 1
        self.calls.append({**record, "suppressed": True, "applied": False})

    def record_tick(
        self,
        snap: WorldSnapshot,
        action: np.ndarray,
        shield_result: ShieldResult,
        skill: str,
        *,
        sigma: float = 0.0,
        vlm_pending: bool = False,
        pending_age_s: float = 0.0,
    ) -> None:
        self.ticks += 1
        if self.start_x is None:
            self.start_x = float(snap.ego.x)
        self.xs.append(float(snap.ego.x))
        self.ys.append(float(snap.ego.y))
        self.skills.append(skill)
        self.actions.append([float(action[0]), float(action[1])])

        ay = float(action[1])
        jerk = abs(ay - self.last_ay) / DT
        self.peak_jerk = max(self.peak_jerk, jerk)
        if self.ticks > 5:
            self.peak_jerk_warmup = max(self.peak_jerk_warmup, jerk)
        if self.ticks > 1:
            self.jerk_sum += jerk
            self.jerk_count += 1
        self.last_ay = ay

        self.peak_sigma = max(self.peak_sigma, float(sigma))
        if shield_result.overridden:
            self.override_ticks += 1
        if not shield_result.safe or shield_result.emergency:
            self.safety_events += 1
        if not shield_result.feasible:
            self.infeasible_ticks += 1
        if math.isfinite(float(shield_result.h)):
            self.min_h = min(self.min_h, float(shield_result.h))
        braking_h = getattr(shield_result, "braking_h", None)
        if braking_h is not None and math.isfinite(float(braking_h)):
            self.min_braking_h = float(braking_h) if self.min_braking_h is None else min(self.min_braking_h, float(braking_h))
        if not getattr(shield_result, "braking_feasible", True):
            self.braking_infeasible_ticks += 1

        if vlm_pending:
            self.pending_ticks += 1
            self.max_pending_s = max(self.max_pending_s, float(pending_age_s))

        if float(snap.ego.vx) < self.config.freeze_speed:
            self.low_speed_streak += 1
            if self.low_speed_streak * DT >= self.config.freeze_window_s:
                self.freeze_seen = True
        else:
            self.low_speed_streak = 0

        if snap.info.get("params"):
            self.params = dict(snap.info["params"]) if isinstance(snap.info["params"], dict) else self.params

    def finalize(
        self,
        env,
        *,
        crashed: bool,
        crash_time: Optional[float],
        end_reason: str,
    ) -> RolloutResult:
        final = env.snapshot()
        ego = final.ego
        end_t = max(float(final.t), DT)
        route_length = max(float(final.route_length), 1e-6)
        route_pct = 100.0 * min(float(ego.x), route_length) / route_length
        dist = float(ego.x) - (self.start_x if self.start_x is not None else float(ego.x))
        avg_speed = float(dist / end_t) if end_t > 0 else 0.0

        # Sustained low speed with progress left, not a single stopped tick.
        freeze = (not crashed) and self.freeze_seen and route_pct < self.config.route_success_pct
        override_fraction = self.override_ticks / self.ticks if self.ticks else 0.0
        mean_jerk = self.jerk_sum / self.jerk_count if self.jerk_count else 0.0
        plant = [row["plant"] for row in self.interactions
                 if row.get("plant", {}).get("achieved_acceleration") is not None]
        plant_jerk = tracking_rmse = saturation = None
        if plant:
            achieved = np.asarray([row["achieved_acceleration"] for row in plant], dtype=float)
            requested = np.asarray([row["requested_acceleration"] for row in plant], dtype=float)
            plant_jerk = float(np.max(np.abs(np.diff(achieved[:, 1]))) / DT) if len(plant) > 1 else 0.0
            tracking_rmse = float(np.sqrt(np.mean((achieved - requested)**2)))
            saturation = float(np.mean([row["saturated"] for row in plant]))

        return RolloutResult(
            mode=self.mode,
            crashed=crashed,
            crash_time=crash_time,
            route_pct=route_pct,
            avg_speed=avg_speed,
            router_calls=self.call_attempts,
            peak_lateral_jerk=self.peak_jerk,
            peak_sigma=self.peak_sigma,
            freeze=freeze,
            x=float(ego.x),
            y=float(ego.y),
            vx=float(ego.vx),
            end_time=end_t,
            calls=self.calls,
            xs=self.xs,
            ys=self.ys,
            skills=self.skills,
            actions=self.actions,
            interactions=self.interactions,
            plant_peak_lateral_jerk=plant_jerk,
            acceleration_tracking_rmse=tracking_rmse,
            actuator_saturation_fraction=saturation,
            mean_lateral_jerk=mean_jerk,
            peak_lateral_jerk_after_warmup=self.peak_jerk_warmup,
            override_fraction=override_fraction,
            pending_ticks=self.pending_ticks,
            pending_survival=self.pending_survival,
            call_attempts=self.call_attempts,
            calls_applied=self.calls_applied,
            calls_rejected=self.calls_rejected,
            calls_suppressed=self.calls_suppressed,
            router_fallbacks=self.router_fallbacks,
            router_prompt_tokens=self.router_prompt_tokens,
            router_completion_tokens=self.router_completion_tokens,
            router_tokens=self.router_tokens,
            router_live_calls=self.router_live_calls,
            router_models=list(self.router_models),
            max_pending_s=self.max_pending_s,
            safety_events=self.safety_events,
            infeasible_ticks=self.infeasible_ticks,
            min_barrier_h=self.min_h,
            min_braking_h=self.min_braking_h,
            braking_infeasible_ticks=self.braking_infeasible_ticks,
            wall_latency_ms=self.wall_latency_ms,
            end_reason=end_reason,
            seed=self.seed,
            scenario=self.scenario,
            params=self.params,
        )
