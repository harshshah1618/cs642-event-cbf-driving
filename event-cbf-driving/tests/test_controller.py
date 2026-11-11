"""Controller integration tests: pending calls, triggers, blending, safety.

These tests use a deterministic toy plant and duck-typed policies/routers so
they exercise the real :class:`DrivingStack` state machine without depending on
checkpoint quality or scenario geometry.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import pytest
from event_cbf_driving import DT
from event_cbf_driving.config import StackConfig
from event_cbf_driving.env.types import VehicleState, WorldSnapshot
from event_cbf_driving.router.base import RouterBackend, RouterRequest, RouterResponse
from event_cbf_driving.shield.cbf import ShieldResult
from event_cbf_driving.stack.controller import DrivingStack
from event_cbf_driving.stack.pending import PendingCall

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeSkill:
    def __init__(self, name: str, *, ay: float = 0.0, sigma: float = 0.0, threshold: float = 1.0):
        self.name = name
        self.ay = float(ay)
        self.sigma = float(sigma)
        self.meta = type("Meta", (), {"threshold": float(threshold)})()

    def act(self, y, vy, vx, deterministic=True):
        return 0.0, self.ay, self.sigma


class FakeEnv:
    """Point-mass plant with a snapshot/step contract."""

    def __init__(self, *, duration: float = 1.0, route_length: float = 100.0, vx: float = 10.0):
        self.duration = float(duration)
        self.route_length = float(route_length)
        self.initial_vx = float(vx)
        self.t = 0.0
        self.ego = VehicleState(x=0.0, y=0.0, vx=self.initial_vx, vy=0.0)
        self.crashed = False
        self.closed = False

    def reset(self, *, seed: Optional[int] = None, options=None):
        self.t = 0.0
        self.ego = VehicleState(x=0.0, y=0.0, vx=self.initial_vx, vy=0.0)
        self.crashed = False
        return self.snapshot(), {
            "scenario": "fake",
            "params": {"seed": seed},
            "snapshot": self.snapshot(),
            "crashed": False,
            "route_pct": 0.0,
        }

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            t=self.t,
            ego=copy.deepcopy(self.ego),
            others=[],
            route_length=self.route_length,
            done=False,
            crashed=self.crashed,
            info={"scenario": "fake", "params": {}},
        )

    def step(self, action):
        ax, ay = float(action[0]), float(action[1])
        self.ego.x += self.ego.vx * DT + 0.5 * ax * DT * DT
        self.ego.y += self.ego.vy * DT + 0.5 * ay * DT * DT
        self.ego.vx = max(0.0, self.ego.vx + ax * DT)
        self.ego.vy += ay * DT
        self.t += DT
        terminated = self.ego.x >= self.route_length
        truncated = self.t >= self.duration - 1e-12
        snap = self.snapshot()
        return (
            np.array([self.ego.y, self.ego.vy, self.ego.vx], dtype=np.float32),
            0.0,
            bool(terminated),
            bool(truncated),
            {
                "snapshot": snap,
                "scenario": "fake",
                "params": {},
                "crashed": False,
                "route_pct": 100.0 * self.ego.x / self.route_length,
            },
        )

    def close(self):
        self.closed = True


class FakeRouter(RouterBackend):
    """Returns a configurable response after a simulated latency."""

    name = "fake"

    def __init__(
        self,
        *,
        skill: str = "CRUISE_RIGHT",
        latency_s: float = 0.10,
        ready_at_offset: Optional[float] = None,
        bad_skill: bool = False,
        nan_ready: bool = False,
        error: Optional[str] = None,
        live: bool = False,
        model: str = "",
        usage: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.skill = skill
        self.latency_s = float(latency_s)
        self.ready_at_offset = ready_at_offset
        self.bad_skill = bad_skill
        self.nan_ready = nan_ready
        self.error = error
        self.live = live
        self.model = model
        self.usage = dict(usage or {})
        self.requests: List[RouterRequest] = []
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.requests.clear()

    def submit(self, request: RouterRequest) -> RouterResponse:
        self.requests.append(copy.deepcopy(request))
        offset = self.latency_s if self.ready_at_offset is None else self.ready_at_offset
        ready = float(request.t) + offset
        if self.nan_ready:
            ready = float("nan")
        return RouterResponse(
            skill="BOGUS" if self.bad_skill else self.skill,
            ready_at=ready,
            backend="fake",
            latency_s=self.latency_s,
            request_id=request.request_id,
            submitted_t=request.t,
            wall_latency_s=self.latency_s,
            error=self.error,
            live=self.live,
            model=self.model,
            usage=dict(self.usage),
        )

    def describe(self):
        return "fake-router"


class ScriptedShield:
    """Small duck-typed shield for trigger/abort tests."""

    ax_min = -7.5
    ax_max = 3.5
    emergency_decel = -7.5
    enabled = True
    inflate = True
    pending_margin = True

    def __init__(self, *, mode: str = "override_edges", invalid: bool = False):
        self.mode = mode
        self.invalid = invalid
        self.calls = 0

    def filter(self, action, snap, sigma=0.0, vlm_pending=False):
        self.calls += 1
        if self.invalid:
            return ShieldResult(
                action=np.array([-7.5, 0.0], dtype=np.float32),
                residual=0.0,
                overridden=True,
                inflation=0.0,
                h=float("-inf"),
                feasible=False,
                safe=False,
                reason="invalid_action",
                obstacle_count=0,
                emergency=True,
            )
        if self.mode == "never_override":
            override = False
        elif self.mode == "always_override":
            override = True
        else:
            # Alternate 20 clear ticks / 20 override ticks.  The clear window
            # is long enough to re-arm the shield trigger.
            phase = int(round(float(snap.t) / DT)) % 40
            override = phase >= 20
        if override:
            return ShieldResult(
                action=np.array([-2.0, 0.0], dtype=np.float32),
                residual=1.0,
                overridden=True,
                inflation=0.0,
                h=2.0,
                feasible=True,
                safe=True,
                reason="ok",
                obstacle_count=1,
                emergency=False,
            )
        return ShieldResult(
            action=np.array([float(action[0]), float(action[1])], dtype=np.float32),
            residual=0.0,
            overridden=False,
            inflation=0.0,
            h=10.0,
            feasible=True,
            safe=True,
            reason="ok",
            obstacle_count=0,
            emergency=False,
        )


def _skills(**overrides) -> Dict[str, FakeSkill]:
    return {
        "CRUISE_RIGHT": overrides.get("cruise", FakeSkill("CRUISE_RIGHT", ay=0.0)),
        "OVERTAKE_LEFT": overrides.get("overtake", FakeSkill("OVERTAKE_LEFT", ay=4.0)),
        "RETURN_RIGHT": overrides.get("returning", FakeSkill("RETURN_RIGHT", ay=0.0)),
    }


def _stack(
    *,
    router: FakeRouter,
    skills: Optional[Dict[str, FakeSkill]] = None,
    shield=None,
    **cfg_overrides,
) -> DrivingStack:
    defaults = dict(
        mode="proposed",
        min_call_ticks=0,
        shield_clear_ticks=0,
        shield_hold_ticks=1,
        blend_ticks=4,
        max_pending_s=2.0,
        sigma_threshold_scale=1.0,
    )
    defaults.update(cfg_overrides)
    return DrivingStack(
        skills=skills or _skills(),
        router=router,
        shield=shield if shield is not None else ScriptedShield(mode="never_override"),
        config=StackConfig(**defaults),
    )


# ---------------------------------------------------------------------------
# Pending-call state machine
# ---------------------------------------------------------------------------


def test_pending_call_ready_expiry_and_delivery():
    request = RouterRequest(t=1.0, current_skill="CRUISE_RIGHT", snapshot=None, sigma=0.0, reason="test")
    response = RouterResponse(skill="OVERTAKE_LEFT", ready_at=1.3, backend="fake")
    pending = PendingCall(request=request, response=response, submitted_t=1.0, deadline_t=2.0)

    assert not pending.is_ready(1.29)
    assert pending.is_ready(1.3)
    assert not pending.is_expired(2.0)
    assert pending.is_expired(2.5)
    assert pending.age_s(1.15) == pytest.approx(0.15)
    assert pending.age_s(2.0) == pytest.approx(0.3)  # capped at ready_at
    assert pending.deliver("CRUISE_RIGHT") == "OVERTAKE_LEFT"
    assert pending.applied is True
    assert pending.deliver("OVERTAKE_LEFT") is None


# ---------------------------------------------------------------------------
# Delivery, same-skill latency, rejection
# ---------------------------------------------------------------------------


def test_router_answer_switches_skill_after_latency():
    router = FakeRouter(skill="OVERTAKE_LEFT", latency_s=0.2)
    skills = _skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=10.0))
    stack = _stack(router=router, skills=skills)
    result = stack.run(FakeEnv(duration=1.0, route_length=1000.0), seed=0)

    assert len(router.requests) == 1
    assert router.resets == 1
    assert result.router_calls == 1
    assert result.calls_applied == 1
    assert result.calls[0]["switched"] is True
    assert result.calls[0]["ready_at"] == pytest.approx(0.2)
    assert "OVERTAKE_LEFT" in result.skills
    assert result.skills[-1] == "OVERTAKE_LEFT"


def test_router_telemetry_flows_into_rollout_and_call_record():
    router = FakeRouter(
        skill="OVERTAKE_LEFT",
        latency_s=0.2,
        live=True,
        model="test-model",
        usage={"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
    )
    skills = _skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=10.0))
    result = _stack(router=router, skills=skills).run(
        FakeEnv(duration=1.0, route_length=1000.0), seed=0
    )

    assert result.router_live_calls == 1
    assert result.router_prompt_tokens == 9
    assert result.router_completion_tokens == 1
    assert result.router_tokens == 10
    assert result.router_models == ["test-model"]
    call = result.calls[0]
    assert call["live"] is True
    assert call["model"] == "test-model"
    assert call["total_tokens"] == 10


def test_same_skill_answer_keeps_delay_margin_when_configured():
    router = FakeRouter(skill="CRUISE_RIGHT", latency_s=0.2)
    skills = _skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=10.0))

    with_margin = _stack(
        router=router, skills=skills, pending_margin_on_all_calls=True
    ).run(FakeEnv(duration=1.0, route_length=1000.0), seed=0)
    assert with_margin.pending_ticks >= 5
    assert with_margin.calls_applied == 1
    assert with_margin.calls[0]["switched"] is False

    router_legacy = FakeRouter(skill="CRUISE_RIGHT", latency_s=0.2)
    without_margin = _stack(
        router=router_legacy, skills=skills, pending_margin_on_all_calls=False
    ).run(FakeEnv(duration=1.0, route_length=1000.0), seed=0)
    assert without_margin.pending_ticks == 0
    assert without_margin.calls_applied == 1
    assert without_margin.calls[0]["applied"] is True


def test_late_answer_is_rejected_without_blocking_the_run():
    # ready_at=1.0 s but the controller only waits 0.2 s.
    router = FakeRouter(skill="OVERTAKE_LEFT", latency_s=1.0)
    skills = _skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=10.0))
    stack = _stack(router=router, skills=skills, max_pending_s=0.2)
    result = stack.run(FakeEnv(duration=0.5, route_length=1000.0), seed=0)

    assert result.calls_rejected == 1
    assert result.calls[0]["rejected"] is True
    assert "deadline" in result.calls[0]["error"]
    assert result.pending_ticks == 0
    assert result.skills[-1] == "CRUISE_RIGHT"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bad_skill": True},
        {"nan_ready": True},
        {"ready_at_offset": -1.0},
    ],
)
def test_invalid_router_responses_are_rejected(kwargs):
    router = FakeRouter(skill="OVERTAKE_LEFT", **kwargs)
    skills = _skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=10.0))
    result = _stack(router=router, skills=skills).run(
        FakeEnv(duration=0.4, route_length=1000.0), seed=0
    )

    assert result.calls_rejected == 1
    assert result.calls_applied == 0
    assert result.calls[0]["rejected"] is True
    assert result.skills[-1] == "CRUISE_RIGHT"


# ---------------------------------------------------------------------------
# Trigger hysteresis and latched edges
# ---------------------------------------------------------------------------


def test_shield_trigger_requires_clear_window_and_latches_suppressed_edge():
    shield = ScriptedShield(mode="override_edges")
    router = FakeRouter(skill="CRUISE_RIGHT", latency_s=0.1)
    skills = _skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=0.0))
    # 20 clear ticks between override bursts; min_call_ticks is long enough
    # that a burst edge can be suppressed while the stack waits.
    stack = _stack(
        router=router,
        skills=skills,
        shield=shield,
        shield_clear_ticks=10,
        min_call_ticks=60,
    )
    result = stack.run(FakeEnv(duration=3.0, route_length=10000.0), seed=0)

    # The first override burst triggers a call.  Later bursts are separated by
    # a clear window, so the trigger re-arms instead of firing every tick.
    assert len(router.requests) >= 2
    assert result.calls_suppressed >= 1
    submitted = [call for call in result.calls if not call.get("suppressed")]
    assert result.router_calls == len(submitted)
    assert result.calls_applied + result.calls_rejected == result.router_calls


def test_shield_hold_guard_fires_once_per_override_episode():
    # Regression: shield_hold_ticks >= 2 used to consume the only armed tick,
    # so a sustained override never produced a call (and the episode froze).
    shield = ScriptedShield(mode="always_override")
    router = FakeRouter(skill="OVERTAKE_LEFT", latency_s=0.02)
    skills = _skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=0.0))
    stack = _stack(
        router=router,
        skills=skills,
        shield=shield,
        shield_clear_ticks=10,
        shield_hold_ticks=5,
    )
    result = stack.run(FakeEnv(duration=1.0, route_length=10000.0), seed=0)

    assert result.router_calls == 1
    assert result.calls_applied == 1
    assert result.skills[-1] == "OVERTAKE_LEFT"


def test_always_on_mode_calls_at_configured_rate():
    router = FakeRouter(skill="CRUISE_RIGHT", latency_s=0.01)
    stack = _stack(router=router, mode="always_on_vlm", always_on_hz=10.0)
    result = stack.run(FakeEnv(duration=1.0, route_length=10000.0), seed=0)

    assert result.mode == "always_on_vlm"
    # 1 s at 10 Hz, allowing for the final tick and pending windows.
    assert 7 <= result.router_calls <= 11
    assert result.calls_applied >= 5


# ---------------------------------------------------------------------------
# Blending and mode variants
# ---------------------------------------------------------------------------


def test_hard_switch_has_higher_lateral_jerk_than_blend():
    skills = {
        "CRUISE_RIGHT": FakeSkill("CRUISE_RIGHT", ay=0.0, sigma=10.0),
        "OVERTAKE_LEFT": FakeSkill("OVERTAKE_LEFT", ay=4.0),
        "RETURN_RIGHT": FakeSkill("RETURN_RIGHT", ay=0.0),
    }
    blend_result = _stack(
        router=FakeRouter(skill="OVERTAKE_LEFT", latency_s=0.1),
        skills=skills,
        use_blend=True,
        hard_switch=False,
        blend_ticks=20,
    ).run(FakeEnv(duration=1.2, route_length=10000.0), seed=0)
    hard_result = _stack(
        router=FakeRouter(skill="OVERTAKE_LEFT", latency_s=0.1),
        skills=skills,
        use_blend=False,
        hard_switch=True,
    ).run(FakeEnv(duration=1.2, route_length=10000.0), seed=0)

    assert blend_result.peak_lateral_jerk < hard_result.peak_lateral_jerk


def test_variant_modes_disable_margin_or_blend():
    skills = _skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=10.0))
    router = FakeRouter(skill="CRUISE_RIGHT", latency_s=0.2)
    no_margin_stack = _stack(
        router=router,
        skills=skills,
        mode="proposed_no_margin",
        pending_margin_on_all_calls=True,
    )
    no_margin = no_margin_stack.run(FakeEnv(duration=1.0, route_length=10000.0), seed=0)
    # The call is still in flight (pending_ticks records that), but the shield
    # must not add the latency margin in the no-margin ablation.
    assert no_margin.pending_ticks > 0
    assert no_margin.mode == "proposed_no_margin"
    assert no_margin_stack.last_shield.pending_margin is False

    router2 = FakeRouter(skill="OVERTAKE_LEFT", latency_s=0.1)
    no_blend = _stack(
        router=router2,
        skills=_skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=10.0)),
        mode="proposed_no_blend",
    ).run(FakeEnv(duration=1.0, route_length=10000.0), seed=0)
    assert no_blend.mode == "proposed_no_blend"
    assert no_blend.peak_lateral_jerk > 20.0


# ---------------------------------------------------------------------------
# Safety monitor integration
# ---------------------------------------------------------------------------


def test_emergency_abort_stops_the_run_when_configured():
    stack = _stack(
        router=FakeRouter(),
        shield=ScriptedShield(invalid=True),
        abort_on_emergency=True,
    )
    result = stack.run(FakeEnv(duration=5.0, route_length=10000.0), seed=0)

    assert result.end_reason == "safety_abort"
    assert result.end_time < 0.2  # five invalid ticks at 20 ms
    assert stack.last_monitor is not None
    assert stack.last_monitor.summary()["status"] == "emergency"


def test_emergency_is_monitored_but_not_irreversible_by_default():
    stack = _stack(
        router=FakeRouter(),
        shield=ScriptedShield(invalid=True),
        abort_on_emergency=False,
    )
    result = stack.run(FakeEnv(duration=0.3, route_length=10000.0), seed=0)

    assert result.end_reason == "timeout"
    assert result.end_time == pytest.approx(0.3, abs=DT * 2)
    assert stack.last_monitor.summary()["status"] == "emergency"
