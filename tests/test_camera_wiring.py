"""Controller camera timing contracts using toy plant and mock router only."""

from __future__ import annotations

import threading
from concurrent.futures import Future

import pytest
from driving.errors import ConfigError, SafetyError
from driving.router_base import CameraObservation

from test_controller import FakeEnv, FakeRouter, FakeSkill, _skills, _stack


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class CameraEnv(FakeEnv):
    def __init__(self, clock, capture_s=0.25, failure=None):
        super().__init__(duration=0.8, route_length=1000.0)
        self.clock, self.capture_s, self.failure = clock, capture_s, failure
        self.capture_threads = []

    def capture_router_camera(self):
        self.capture_threads.append(threading.get_ident())
        self.clock.now += self.capture_s
        if self.failure:
            raise self.failure
        return CameraObservation(
            b"detached mock image", "image/png", self.t, "RIGHT", self.ego.vx, 1, 1
        )


class CameraRouter(FakeRouter):
    requires_camera = True


class ImmediateExecutor:
    """Deterministic future completion exercises async delivery without races."""

    def submit(self, fn):
        f = Future()
        try:
            f.set_result(fn())
        except Exception as exc:
            f.set_exception(exc)
        return f


def build(router):
    return _stack(
        router=router,
        skills=_skills(cruise=FakeSkill("CRUISE_RIGHT", sigma=10.0)),
        max_pending_s=0.6,
        pending_margin_on_all_calls=True,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_capture_is_in_total_latency_and_subtracted_from_model_budget(monkeypatch, asynchronous):
    clock = Clock()
    monkeypatch.setattr("driving.controller.time.perf_counter", clock)
    router = CameraRouter(skill="OVERTAKE_LEFT", latency_s=0.1)
    env = CameraEnv(clock)
    stack = build(router)
    result = stack._run(env, seed=0, executor=ImmediateExecutor() if asynchronous else None)
    assert len(router.requests) == 1
    req = router.requests[0]
    assert req.camera_capture_wall_s == 0.25
    assert req.deadline_s == pytest.approx(0.35)
    assert req.camera.captured_t == req.t
    assert env.capture_threads == [threading.get_ident()]
    assert result.calls_applied == 1
    record = result.calls[0]
    assert record["camera_capture_wall_ms"] == 250.0
    assert record["ready_at"] == pytest.approx(req.t + 0.35)
    assert record["latency_s"] == pytest.approx(0.35)
    assert record["wall_latency_ms"] == pytest.approx(100.0)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("capture_s", [0.6, 0.7])
def test_capture_exhausting_deadline_never_submits_router(monkeypatch, asynchronous, capture_s):
    clock = Clock()
    monkeypatch.setattr("driving.controller.time.perf_counter", clock)
    router = CameraRouter(skill="OVERTAKE_LEFT", latency_s=0.1)
    stack = build(router)
    result = stack._run(
        CameraEnv(clock, capture_s=capture_s),
        seed=0,
        executor=ImmediateExecutor() if asynchronous else None,
    )
    assert router.requests == []
    assert result.calls_applied == 0 and result.calls_rejected >= 1
    assert all(r["rejected"] and "camera capture" in r["error"] for r in result.calls)


def test_missing_camera_hook_fails_explicitly_without_router_call():
    router = CameraRouter()
    with pytest.raises(SafetyError, match="environment camera"):
        build(router).run(FakeEnv())
    assert router.requests == []


def test_capture_failure_propagates_without_router_or_scripted_substitution(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("driving.controller.time.perf_counter", clock)
    router = CameraRouter()
    env = CameraEnv(clock, failure=ConfigError("camera sensor unavailable"))
    with pytest.raises(ConfigError, match="camera sensor unavailable"):
        build(router).run(env)
    assert router.requests == []
