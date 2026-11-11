"""Offline VLM contract tests; no real-model performance is measured."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace

import pytest
from event_cbf_driving.config import RouterConfig
from event_cbf_driving.env.types import VehicleState, WorldSnapshot
from event_cbf_driving.errors import RouterError
from event_cbf_driving.router.base import CameraObservation, RouterRequest, response_record_fields
from event_cbf_driving.router.openai_compat import TransportError, TransportResponse
from event_cbf_driving.router.vlm import CHOICES, VLMRouter

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeTransport:
    def __init__(self, clock, results, duration=0.125):
        self.clock, self.results, self.duration = clock, list(results), duration
        self.calls = []
        self.closed = False

    def request(self, url, **kwargs):
        self.calls.append((url, kwargs))
        self.clock.now += self.duration
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


def completion(letter="B", finish="stop", usage=None, **message_fields):
    return TransportResponse(
        200,
        json.dumps(
            {
                "model": "test-vlm",
                "choices": [
                    {"message": {"content": letter, **message_fields}, "finish_reason": finish}
                ],
                "usage": usage
                if usage is not None
                else {"prompt_tokens": 305, "completion_tokens": 1},
            }
        ).encode(),
        {"Content-Type": "application/json"},
    )


def request(**kwargs):
    camera = CameraObservation(
        image_bytes=PNG,
        mime_type="image/png",
        captured_t=1.9,
        lane="RIGHT",
        speed_mps=20.0,
        width=1,
        height=1,
    )
    values = dict(
        t=2.0,
        current_skill="RETURN_RIGHT",
        sigma=777.0,
        reason="SECRET_REASON",
        snapshot=WorldSnapshot(
            2.0,
            VehicleState(12345.0, 0.0, 20.0, 0.0),
            [VehicleState(98765.0, 0.0, 0.0, 0.0)],
            200.0,
        ),
        request_id="test-contract",
        camera=camera,
    )
    values.update(kwargs)
    return RouterRequest(**values)


def router(results=None, duration=0.125, **kwargs):
    clock = Clock()
    transport = FakeTransport(clock, results or [completion()], duration)
    config = RouterConfig(
        model="test-vlm",
        base_url="http://127.0.0.1:8000/v1",
        timeout_s=1.0,
        max_retries=2,
        backoff_s=0.1,
        allow_fallback=True,
        fail_on_fallback=False,
    )
    config = replace(config, **kwargs)
    return VLMRouter(config, transport=transport, clock=clock, sleep=clock.sleep), transport, clock


def test_payload_excludes_privileged_state_and_records_metadata():
    r, transport, _ = router()
    resp = r.submit(request())
    payload = json.loads(transport.calls[0][1]["body"])
    assert payload["structured_outputs"] == {"choice": ["A", "B", "C"]}
    assert payload["max_tokens"] == 1 and payload["temperature"] == 0
    for forbidden in ("RETURN_RIGHT", "SECRET_REASON", "98765", "12345", "777", "current_skill"):
        assert forbidden not in json.dumps(payload)
    blocks = payload["messages"][1]["content"]
    assert (
        blocks[0]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    )
    assert blocks[1]["text"] == "Our car is in the RIGHT lane. Ego speed: 20.000 m/s."
    assert resp.skill == "OVERTAKE_LEFT" and not resp.fallback_used and not resp.live
    assert resp.ready_at == 2.125 and resp.wall_latency_s == resp.latency_s == 0.125
    assert resp.image_hash == hashlib.sha256(PNG).hexdigest()
    assert resp.image_age_s == pytest.approx(0.1) and resp.image_captured_t == 1.9
    assert resp.request_attempts == 1 and resp.usage["total_tokens"] == 306
    record = response_record_fields(resp)
    assert record["image_hash"] == resp.image_hash and record["request_attempts"] == 1
    assert r.health().total_calls == 1 and r.health().total_tokens == 306


@pytest.mark.parametrize("letter,skill", CHOICES.items())
@pytest.mark.parametrize("finish", ["stop", "length"])
def test_exact_letter_mapping_and_one_token_length_completion(letter, skill, finish):
    r, _, _ = router([completion(letter, finish)])
    assert r.submit(request()).skill == skill


@pytest.mark.parametrize(
    "text,finish",
    [
        ("", "length"),
        ("BB", "length"),
        ("B because safe", "stop"),
        ("OVERTAKE_LEFT", "stop"),
        ("a", "stop"),
        ("B", "content_filter"),
        ("B", "tool_calls"),
    ],
)
def test_invalid_output_never_falls_back_or_retries(text, finish):
    r, transport, _ = router([completion(text, finish)])
    with pytest.raises(RouterError):
        r.submit(request())
    assert len(transport.calls) == 1 and r.health().total_failures == 1


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"image_bytes": b""},
        {"image_bytes": b"not PNG"},
        {"mime_type": "image/jpeg"},
        {"captured_t": 1.0},
        {"captured_t": 2.1},
        {"speed_mps": float("nan")},
        {"lane": "UNKNOWN"},
        {"width": 0},
    ],
)
def test_missing_invalid_or_stale_image_fails_before_transport(change):
    r, transport, _ = router()
    req = request()
    req.camera = None if change is None else replace(req.camera, **change)
    with pytest.raises(RouterError):
        r.submit(req)
    assert transport.calls == [] and r.health().total_failures == 1


def test_retry_shares_total_budget_and_records_attempts():
    r, transport, _ = router([TransportResponse(503), completion()])
    resp = r.submit(request(deadline_s=0.5))
    assert resp.request_attempts == 2 and resp.latency_s == pytest.approx(0.35)
    assert transport.calls[0][1]["timeout"] == 0.5
    assert transport.calls[1][1]["timeout"] == pytest.approx(0.275)
    assert r.health().total_calls == 1 and r.health().total_failures == 0


def test_overbudget_completion_rejected():
    r, transport, _ = router(duration=0.6)
    with pytest.raises(RouterError, match="deadline"):
        r.submit(request(deadline_s=0.5))
    assert len(transport.calls) == 1 and r.health().total_failures == 1


def test_backoff_cannot_exhaust_budget_and_start_another_request():
    r, transport, _ = router([TransportResponse(429)], backoff_s=0.5)
    with pytest.raises(RouterError, match="deadline"):
        r.submit(request(deadline_s=0.5))
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "failure", [TransportResponse(401), TransportError("private", transient=False)]
)
def test_permanent_failure_not_retried_or_leaked(failure):
    r, transport, _ = router([failure])
    with pytest.raises(RouterError) as error:
        r.submit(request())
    assert "private" not in str(error.value) and len(transport.calls) == 1


def test_circuit_failure_and_reset():
    r, transport, _ = router([TransportResponse(400), completion()], max_consecutive_failures=1)
    with pytest.raises(RouterError):
        r.submit(request())
    assert r.health().failed
    with pytest.raises(RouterError, match="circuit"):
        r.submit(request())
    assert len(transport.calls) == 1
    r.reset()
    assert r.submit(request()).skill == "OVERTAKE_LEFT" and r.health().healthy
    r.close()
    assert transport.closed


def test_model_endpoint_and_optional_auth_are_caller_controlled(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVENT_CBF_API_KEY", raising=False)
    r, transport, _ = router()
    r.submit(request())
    assert "Authorization" not in transport.calls[0][1]["headers"]
    assert transport.calls[0][0] == "http://127.0.0.1:8000/v1/chat/completions"
    with pytest.raises(RouterError):
        VLMRouter(RouterConfig(), model="")
    with pytest.raises(RouterError):
        VLMRouter(RouterConfig(), base_url="http://user:password@host/v1")


def test_prompt_hash_ignores_privileged_state_and_changes_with_camera():
    r, _, _ = router()
    req = request()
    a = r._payload(r._camera(req))[1]
    req.current_skill = "CRUISE_RIGHT"
    req.snapshot.others.clear()
    assert a == r._payload(r._camera(req))[1]
    req.camera = replace(req.camera, image_bytes=PNG + b"extra")
    assert a != r._payload(r._camera(req))[1]


@pytest.mark.parametrize(
    "reply",
    [
        TransportResponse(200, b"not JSON"),
        TransportResponse(200, b'{"choices":[]}'),
        completion(tool_calls=[{"id": "bad"}]),
        completion(usage={"completion_tokens": 2}),
    ],
)
def test_malformed_response_is_terminal(reply):
    r, transport, _ = router([reply])
    with pytest.raises(RouterError):
        r.submit(request())
    assert len(transport.calls) == 1


def test_factory_live_gate_and_camera_requirement():
    from event_cbf_driving.errors import ConfigError
    from event_cbf_driving.router.factory import create_router

    config = RouterConfig(backend="vlm", model="test-vlm", base_url="http://localhost:8000/v1")
    with pytest.raises(ConfigError, match="live calls are disabled"):
        create_router(config)
    clock = Clock()
    transport = FakeTransport(clock, [completion()])
    result = create_router(
        config, allow_live=True, transport=transport, clock=clock, sleep=clock.sleep
    )
    assert isinstance(result, VLMRouter) and result.requires_camera
    assert not result.submit(request()).live


def test_transient_exhaustion_and_timeout_retry_are_bounded():
    r, transport, _ = router([TimeoutError(), TransportResponse(503)], max_retries=1)
    with pytest.raises(RouterError, match="retries exhausted"):
        r.submit(request())
    assert len(transport.calls) == 2 and r.health().total_failures == 1
