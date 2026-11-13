"""Opt-in v1 prompt contract checks with mock transport, not model evaluation."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from driving.config import RouterConfig
from driving.errors import RouterError

from test_vlm_router import Clock, FakeTransport, completion, request

PATH = Path(__file__).resolve().parents[1] / "tools/teammate_vlm_backend.py"
spec = importlib.util.spec_from_file_location("teammate_backend_tests", PATH)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def build(results=None):
    clock = Clock()
    transport = FakeTransport(clock, results or [completion()])
    config = RouterConfig(model="teammate-v1", base_url="http://localhost:8000/v1", timeout_s=2)
    router = backend.TeammateV1VLMRouter(
        config, transport=transport, clock=clock, sleep=clock.sleep
    )
    return router, transport


def test_exact_pinned_messages_and_processor_limits():
    router, transport = build()
    req = request()
    req.camera = replace(req.camera, speed_mps=3.2)
    response = router.submit(req)
    data = json.loads(transport.calls[0][1]["body"])
    image = data["messages"][1]["content"][0]["image_url"]["url"]
    expected = router._prompts.messages(
        image, {"ego_lane": "RIGHT", "current_skill": "RETURN_RIGHT"}
    )
    assert data["messages"] == expected
    assert data["messages"][0]["content"] == router._prompts.SYSTEM
    assert data["messages"][1]["content"][1]["text"] == (
        "Our car is in the RIGHT lane. Current skill: RETURN_RIGHT. Ego speed: 20 m/s."
    )
    assert data["mm_processor_kwargs"] == {"min_pixels": 784, "max_pixels": 1254400}
    assert data["structured_outputs"] == {"choice": ["A", "B", "C"]}
    assert data["max_tokens"] == 1 and not response.live
    assert response.backend == "vlm_teammate_v1"
    assert router.prompt_source_sha256 == backend.PROMPTS_SHA256
    assert "98765" not in json.dumps(data) and "SECRET_REASON" not in json.dumps(data)


def test_context_is_reset_after_failures_and_new_calls():
    router, transport = build([completion("invalid"), completion()])
    with pytest.raises(RouterError):
        router.submit(request(current_skill="CRUISE_RIGHT"))
    assert router._request_skill.get() is None
    router.submit(request(current_skill="OVERTAKE_LEFT"))
    text = json.loads(transport.calls[1][1]["body"])["messages"][1]["content"][1]["text"]
    assert "Current skill: OVERTAKE_LEFT" in text
    assert router._request_skill.get() is None


def test_reentrant_concurrent_submit_is_refused_without_context_corruption():
    router, transport = build()
    original = transport.request

    def nested(url, **kwargs):
        with pytest.raises(RouterError, match="concurrent"):
            router.submit(request(current_skill="CRUISE_RIGHT"))
        assert router._request_skill.get() == "RETURN_RIGHT"
        return original(url, **kwargs)

    transport.request = nested
    router.submit(request())
    assert len(transport.calls) == 1 and router._request_skill.get() is None


def test_unpinned_or_missing_source_is_rejected(tmp_path):
    config = RouterConfig(model="teammate-v1", base_url="http://localhost:8000/v1")
    with pytest.raises(RouterError, match="unavailable"):
        backend.TeammateV1VLMRouter(config, prompts_path=tmp_path / "missing.py")
    altered = tmp_path / "prompts.py"
    altered.write_text("raise RuntimeError('must never execute unpinned source')")
    with pytest.raises(RouterError, match="hash mismatch"):
        backend.TeammateV1VLMRouter(config, prompts_path=altered)
