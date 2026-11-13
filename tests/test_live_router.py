"""Offline tests for the live-LLM path: gating, telemetry, and checks.

No real API key, daemon, or model download is used.  A fake transport exercises
the real OpenAI-compatible request/JSON/parse path; a fake ``urlopen`` does the
same for Ollama.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from driving.config import RouterConfig, StackConfig
from driving.errors import ConfigError, RouterError
from driving.metrics import aggregate_rollouts, result_to_episode_record
from driving.planner import ScriptedRouter
from driving.result import RolloutTracker
from driving.router_base import RouterBackend, RouterRequest, RouterResponse
from driving.router_check import check_router, sample_request
from driving.router_factory import create_router
from driving.router_http import OpenAICompatRouter, TransportResponse
from driving.router_local import LocalLLMRouter


class _CompletionTransport:
    def __init__(
        self,
        *,
        content: str = "OVERTAKE_LEFT",
        status: int = 200,
        usage: object = None,
        model: str = "provider-model",
        finish_reason: str = "stop",
    ):
        self.content = content
        self.status = status
        self.usage = (
            {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}
            if usage is None
            else usage
        )
        self.model = model
        self.finish_reason = finish_reason
        self.calls = 0

    def request(self, url, *, method="POST", body, headers, timeout):
        self.calls += 1
        payload = {
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": self.usage,
        }
        return TransportResponse(
            status=self.status,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )


def _openai_router(transport: _CompletionTransport, **kwargs) -> OpenAICompatRouter:
    config = RouterConfig(backend="openai", allow_live=True, max_retries=0, backoff_s=0.0)
    return OpenAICompatRouter(
        config=config,
        api_key="test-key",
        transport=transport,
        sleep=lambda _delay: None,
        **kwargs,
    )


def test_live_profiles_extend_transitively():
    from driving.config import load_config

    repo = os.path.dirname(os.path.dirname(__file__))
    for name, backend, local_backend in (
        ("live-openai", "openai", "auto"),
        ("live-ollama", "local", "ollama"),
        ("live-local", "local", "transformers"),
    ):
        config = load_config(os.path.join(repo, "configs", f"{name}.yaml"))
        assert config.router.backend == backend
        assert config.router.allow_live is True
        assert config.router.local_backend == local_backend
        # The production parent still contributes the hardened shield config.
        assert config.shield.lateral_mode == "footprint"


def test_live_backends_require_explicit_opt_in():
    with pytest.raises(ConfigError, match="allow_live"):
        create_router(RouterConfig(backend="openai"))
    with pytest.raises(ConfigError, match="allow_live"):
        create_router(RouterConfig(backend="local"))

    # An unloaded local router is a safe offline diagnostic object.
    router = create_router(RouterConfig(backend="local"), load_model=False)
    assert isinstance(router, LocalLLMRouter)

    router = create_router(
        RouterConfig(backend="openai", allow_live=True),
        api_key="test-key",
        transport=_CompletionTransport(),
    )
    assert isinstance(router, OpenAICompatRouter)


def test_openai_live_response_carries_model_usage_and_health():
    transport = _CompletionTransport()
    router = _openai_router(transport, latency_s=0.2)

    response = router.submit(sample_request())

    assert response.skill == "OVERTAKE_LEFT"
    assert response.live is True
    assert response.fallback_used is False
    assert response.model == "provider-model"
    assert response.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "total_tokens": 13,
    }
    assert response.finish_reason == "stop"

    health = router.health()
    assert health.total_calls == 1
    assert health.total_prompt_tokens == 11
    assert health.total_completion_tokens == 2
    assert health.total_tokens == 13


def test_openai_failure_is_visible_and_never_reports_fake_tokens():
    transport = _CompletionTransport(status=503)
    router = _openai_router(transport, allow_fallback=True)

    response = router.submit(sample_request())

    assert response.fallback_used is True
    assert response.live is False
    assert response.usage == {}
    assert response.error and "transient" in response.error
    assert response.backend == "scripted_fallback"
    health = router.health()
    assert health.total_tokens == 0
    assert health.total_failures == 1


def test_fail_on_fallback_raises_instead_of_silently_routing():
    transport = _CompletionTransport(status=503)
    router = _openai_router(transport, allow_fallback=False)

    with pytest.raises(RouterError, match="503"):
        router.submit(sample_request())


def test_malformed_usage_is_ignored_not_fatal():
    transport = _CompletionTransport(
        usage={"prompt_tokens": "bad", "completion_tokens": -3, "total_tokens": None}
    )
    router = _openai_router(transport)

    response = router.submit(sample_request())

    assert response.skill == "OVERTAKE_LEFT"
    assert response.usage == {}
    assert router.health().total_tokens == 0


def test_router_check_reports_live_and_offline_backends():
    class FakeLiveRouter(RouterBackend):
        name = "fake_live"
        is_live_capable = True

        def submit(self, req: RouterRequest) -> RouterResponse:
            return RouterResponse(
                skill="OVERTAKE_LEFT",
                ready_at=req.t,
                backend=self.name,
                live=True,
                model="fake-model",
                usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            )

        def describe(self) -> str:
            return "fake_live(model=fake-model)"

    live = check_router(
        RouterConfig(backend="scripted"),
        router=FakeLiveRouter(),
        require_live=True,
    )
    assert live.ok is True
    assert live.live is True
    assert live.usage["total_tokens"] == 4

    offline = check_router(
        RouterConfig(backend="scripted"), router=ScriptedRouter(), require_live=True
    )
    assert offline.ok is False
    assert offline.live is False
    assert "no live model answer" in (offline.error or "")


def test_router_check_reports_fallback_instead_of_claiming_live():
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "", "EVENT_CBF_API_KEY": ""},
        clear=False,
    ):
        check = check_router(
            RouterConfig(
                backend="openai",
                allow_live=True,
                allow_fallback=True,
                max_retries=0,
            )
        )

    assert check.ok is False
    assert check.live is False
    assert check.fallback_used is True
    assert check.error == "no_api_key"


class _FakeHTTPResponse:
    status = 200

    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _limit=-1):
        return self._raw


def test_ollama_usage_and_model_are_parsed():
    config = RouterConfig(
        backend="local",
        allow_live=True,
        local_backend="ollama",
        max_retries=0,
    )
    router = LocalLLMRouter(config=config, backend="none", load_model=False)
    router._mode = "ollama"  # bypass the daemon probe; generation is faked below

    payload = {
        "model": "qwen-test",
        "response": "OVERTAKE_LEFT",
        "prompt_eval_count": 17,
        "eval_count": 2,
        "done_reason": "stop",
    }
    with patch(
        "driving.router_local.urllib.request.urlopen",
        return_value=_FakeHTTPResponse(payload),
    ):
        response = router.submit(sample_request())

    assert response.skill == "OVERTAKE_LEFT"
    assert response.live is True
    assert response.backend == "local_ollama"
    assert response.model == "qwen-test"
    assert response.usage == {
        "prompt_tokens": 17,
        "completion_tokens": 2,
        "total_tokens": 19,
    }
    assert response.finish_reason == "stop"
    assert router.health().total_tokens == 19


def test_call_records_and_aggregates_include_live_telemetry():
    tracker = RolloutTracker("proposed", StackConfig())
    tracker.add_call(
        {
            "applied": True,
            "live": True,
            "model": "provider-model",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "wall_latency_ms": 5.0,
        }
    )
    tracker.add_call(
        {
            "applied": False,
            "fallback_used": True,
            "live": False,
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "wall_latency_ms": 7.0,
        }
    )

    assert tracker.router_live_calls == 1
    assert tracker.router_tokens == 12
    assert tracker.router_models == ["provider-model"]

    result = type(
        "Result",
        (),
        {
            "mode": "proposed",
            "scenario": "two_trap",
            "seed": 0,
            "repeat": 0,
            "crashed": False,
            "crash_time": None,
            "route_pct": 100.0,
            "avg_speed": 20.0,
            "router_calls": 2,
            "call_attempts": 2,
            "calls_applied": 1,
            "calls_rejected": 0,
            "calls_suppressed": 0,
            "router_fallbacks": 1,
            "router_prompt_tokens": 10,
            "router_completion_tokens": 2,
            "router_tokens": 12,
            "router_live_calls": 1,
            "router_models": ["provider-model"],
            "peak_lateral_jerk": 0.0,
            "mean_lateral_jerk": 0.0,
            "peak_sigma": 0.0,
            "override_fraction": 0.0,
            "pending_ticks": 0,
            "pending_survival": True,
            "safety_events": 0,
            "infeasible_ticks": 0,
            "min_barrier_h": 10.0,
            "max_pending_s": 0.0,
            "end_reason": "route_complete",
            "x": 200.0,
            "y": 0.0,
            "vx": 20.0,
            "end_time": 10.0,
            "params": {},
        },
    )()

    record = result_to_episode_record(result)
    assert record["router_tokens"] == 12
    assert record["router_live_calls"] == 1
    assert record["router_models"] == ["provider-model"]

    aggregates = aggregate_rollouts([record, record])
    metrics = aggregates["two_trap"]["proposed"]
    assert metrics["router_tokens"]["mean"] == 12.0
    assert metrics["router_live_calls"]["mean"] == 1.0
    assert metrics["router_live_fraction"] == 0.5
    assert metrics["router_models"] == ["provider-model"]
