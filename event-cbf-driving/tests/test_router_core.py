"""Core router contract, parsing, retry, fallback, and factory tests.

All tests are offline.  Live HTTP is replaced with an injected transport.
"""

from __future__ import annotations

import json
import math
import os
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from event_cbf_driving.config import RouterConfig
from event_cbf_driving.env.types import VehicleState, WorldSnapshot
from event_cbf_driving.errors import ConfigError, RouterError
from event_cbf_driving.router.base import (
    RouterRequest,
    RouterResponse,
    validate_response,
)
from event_cbf_driving.router.dry_run import DryRunOpenAIRouter
from event_cbf_driving.router.factory import create_router, make_router
from event_cbf_driving.router.local_llm import LocalLLMRouter
from event_cbf_driving.router.openai_compat import (
    OpenAICompatRouter,
    TransportResponse,
)
from event_cbf_driving.router.prompt import (
    SYSTEM_PROMPT,
    build_messages,
    build_prompt,
    build_user_prompt,
    parse_skill,
    prompt_hash,
)
from event_cbf_driving.router.scripted import ScriptedRouter


def _snap(ego_x=60.0, others=None):
    return WorldSnapshot(
        t=1.0,
        ego=VehicleState(x=ego_x, y=0.0, vx=20.0, vy=0.0),
        others=others
        if others is not None
        else [VehicleState(x=75.0, y=0.0, vx=0.0, vy=0.0, speed_limit=0.01)],
        route_length=200.0,
    )


def _req(**overrides):
    values = dict(
        t=2.0,
        current_skill="CRUISE_RIGHT",
        snapshot=_snap(),
        sigma=4.0,
        reason="sigma",
        request_id="req-123",
    )
    values.update(overrides)
    return RouterRequest(**values)


def _completion(content="OVERTAKE_LEFT", *, status=200, finish_reason="stop", headers=None, body=None):
    if body is None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ]
            }
        ).encode("utf-8")
    return TransportResponse(
        status=status,
        body=body,
        headers=headers if headers is not None else {"Content-Type": "application/json"},
    )


class QueueTransport:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []
        self.closed = False

    def request(self, url, *, method="POST", body, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "method": method,
                "body": body,
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        if not self.items:
            raise AssertionError("transport called more times than expected")
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


class ParseSkillTests(unittest.TestCase):
    def test_exact_whole_tokens(self):
        self.assertEqual(parse_skill("OVERTAKE_LEFT"), "OVERTAKE_LEFT")
        self.assertEqual(parse_skill("overtake left"), "OVERTAKE_LEFT")
        self.assertEqual(parse_skill("overtake-left"), "OVERTAKE_LEFT")
        self.assertEqual(parse_skill(" Return Right. "), "RETURN_RIGHT")

    def test_explicit_answer_and_last_answer_wins(self):
        self.assertEqual(
            parse_skill("I choose: overtake left now."), "OVERTAKE_LEFT"
        )
        self.assertEqual(
            parse_skill("Answer: CRUISE_RIGHT\nFinal answer: RETURN_RIGHT"),
            "RETURN_RIGHT",
        )
        self.assertEqual(parse_skill("skill: cruise"), "CRUISE_RIGHT")
        self.assertEqual(parse_skill("Final answer:\nOVERTAKE_LEFT"), "OVERTAKE_LEFT")

    def test_negation_is_rejected(self):
        self.assertIsNone(parse_skill("Do not choose OVERTAKE_LEFT"))
        self.assertIsNone(parse_skill("avoid RETURN_RIGHT"))
        self.assertEqual(
            parse_skill("not required, choose OVERTAKE_LEFT"), "OVERTAKE_LEFT"
        )

    def test_multiple_conflicting_tokens_are_rejected(self):
        self.assertIsNone(parse_skill("CRUISE_RIGHT or OVERTAKE_LEFT"))
        self.assertIsNone(parse_skill("OVERTAKE_LEFT RETURN_RIGHT"))

    def test_prompt_echo_is_rejected(self):
        echo = build_prompt("CRUISE_RIGHT", _snap(), "sigma")
        self.assertIsNone(parse_skill(echo))

    def test_fallback_only_for_unparseable_text(self):
        self.assertEqual(parse_skill("???", fallback="CRUISE_RIGHT"), "CRUISE_RIGHT")
        self.assertIsNone(parse_skill("not OVERTAKE_LEFT", fallback="CRUISE_RIGHT"))


class PromptTests(unittest.TestCase):
    def test_prompt_contains_router_context_and_obstacles(self):
        far_but_urgent = VehicleState(x=160.0, y=0.0, vx=0.0)
        near_but_slow_closing = VehicleState(x=70.0, y=0.0, vx=19.0)
        snap = _snap(
            ego_x=60.0, others=[near_but_slow_closing, far_but_urgent]
        )
        req = _req(
            snapshot=snap,
            sigma=4.25,
            residual=0.125,
            shield_h=0.75,
            vlm_pending=True,
        )
        prompt = build_user_prompt(
            req.current_skill, req.snapshot, req.reason, request=req
        )
        self.assertIn("current_skill=CRUISE_RIGHT", prompt)
        self.assertIn("trigger_reason=sigma", prompt)
        self.assertIn("sigma=4.2500", prompt)
        self.assertIn("shield_residual=0.1250", prompt)
        self.assertIn("barrier_h=0.7500", prompt)
        self.assertIn("vlm_pending=yes", prompt)
        self.assertIn("route_progress=30.0%", prompt)
        self.assertIn("right_threat=yes", prompt)
        # TTC ordering: 100 m at 20 m/s closing is more urgent than 10 m at 1 m/s.
        self.assertLess(prompt.index("dx=100.00"), prompt.index("dx=10.00"))
        self.assertNotIn("VehicleState(", prompt)
        self.assertNotIn("WorldSnapshot(", prompt)

    def test_chat_messages_have_system_user_split(self):
        messages = build_messages("CRUISE_RIGHT", _snap(), "sigma")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("EGO_STATE", messages[1]["content"])

    def test_prompt_hash_is_stable(self):
        first = prompt_hash("hello")
        self.assertEqual(first, prompt_hash("hello"))
        self.assertNotEqual(first, prompt_hash("goodbye"))
        self.assertEqual(len(first), 64)


class ResponseValidationTests(unittest.TestCase):
    def test_valid_response(self):
        req = _req()
        resp = RouterResponse("OVERTAKE_LEFT", 2.3, "test")
        self.assertIs(validate_response(resp, req), resp)

    def test_rejects_nan_ready_at(self):
        with self.assertRaises(RouterError):
            validate_response(
                RouterResponse("OVERTAKE_LEFT", math.nan, "test"), _req()
            )

    def test_rejects_invalid_skill(self):
        with self.assertRaises(RouterError):
            validate_response(RouterResponse("FLY", 2.3, "test"), _req())

    def test_rejects_ready_at_in_the_past(self):
        with self.assertRaises(RouterError):
            validate_response(RouterResponse("OVERTAKE_LEFT", 1.0, "test"), _req(t=2.0))

    def test_rejects_missing_backend(self):
        with self.assertRaises(RouterError):
            validate_response(RouterResponse("OVERTAKE_LEFT", 2.3, ""), _req())


class ScriptedRouterTests(unittest.TestCase):
    def test_rule_behavior_and_metadata(self):
        router = ScriptedRouter(latency_s=0.4)
        req = _req()
        resp = router.submit(req)
        self.assertEqual(resp.skill, "OVERTAKE_LEFT")
        self.assertEqual(resp.backend, "scripted")
        self.assertEqual(resp.request_id, "req-123")
        self.assertEqual(resp.submitted_t, 2.0)
        self.assertAlmostEqual(resp.ready_at, 2.4)
        self.assertFalse(resp.fallback_used)
        self.assertTrue(resp.prompt_hash)
        self.assertEqual(router.health().total_calls, 1)
        self.assertTrue(router.health().healthy)

    def test_no_mutation_and_describe(self):
        snap = _snap()
        before = (snap.ego.x, snap.ego.y, snap.ego.vx, len(snap.others))
        router = ScriptedRouter()
        router.submit(_req(snapshot=snap))
        after = (snap.ego.x, snap.ego.y, snap.ego.vx, len(snap.others))
        self.assertEqual(before, after)
        self.assertIn("scripted(", router.describe())

    def test_rejects_invalid_request(self):
        router = ScriptedRouter()
        with self.assertRaises(RouterError):
            router.submit(_req(current_skill="FLY"))
        self.assertEqual(router.health().total_failures, 1)


class OpenAICompatTests(unittest.TestCase):
    def _config(self, **overrides):
        values = dict(
            backend="openai",
            latency_s=0.3,
            timeout_s=1.0,
            max_retries=2,
            backoff_s=0.01,
            allow_fallback=True,
            fail_on_fallback=False,
            max_consecutive_failures=5,
            max_tokens=8,
        )
        values.update(overrides)
        return RouterConfig(**values)

    def test_transient_failures_retry_then_succeed(self):
        transport = QueueTransport(
            [
                TransportResponse(500, b"server error", {}),
                _completion("RETURN_RIGHT", status=429),
                _completion("OVERTAKE_LEFT"),
            ]
        )
        sleeps = []
        router = OpenAICompatRouter(
            config=self._config(),
            api_key="test-key",
            transport=transport,
            sleep=sleeps.append,
        )
        resp = router.submit(_req())
        self.assertEqual(resp.skill, "OVERTAKE_LEFT")
        self.assertEqual(resp.backend, "openai_compat")
        self.assertFalse(resp.fallback_used)
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(resp.request_id, "req-123")
        self.assertTrue(resp.prompt_hash)
        self.assertGreaterEqual(resp.wall_latency_s, 0.0)
        self.assertTrue(router.health().healthy)

    def test_timeout_is_transient(self):
        transport = QueueTransport([TimeoutError("slow"), _completion("OVERTAKE_LEFT")])
        router = OpenAICompatRouter(
            config=self._config(max_retries=1),
            api_key="test-key",
            transport=transport,
            sleep=lambda _: None,
        )
        resp = router.submit(_req())
        self.assertEqual(resp.skill, "OVERTAKE_LEFT")
        self.assertEqual(len(transport.calls), 2)

    def test_401_does_not_retry_and_falls_back(self):
        transport = QueueTransport([_completion("", status=401, body=b"denied")])
        router = OpenAICompatRouter(
            config=self._config(max_retries=5),
            api_key="bad-key",
            transport=transport,
            sleep=lambda _: None,
        )
        resp = router.submit(_req())
        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(resp.fallback_used)
        self.assertEqual(resp.backend, "scripted_fallback")
        self.assertIn("401", resp.error)
        self.assertEqual(resp.skill, "OVERTAKE_LEFT")

    def test_401_raises_without_fallback(self):
        transport = QueueTransport([_completion("", status=401, body=b"denied")])
        router = OpenAICompatRouter(
            config=self._config(max_retries=5, allow_fallback=False),
            api_key="bad-key",
            transport=transport,
            sleep=lambda _: None,
        )
        with self.assertRaises(RouterError):
            router.submit(_req())
        self.assertEqual(len(transport.calls), 1)

    def test_injected_httperror_401_does_not_retry(self):
        transport = QueueTransport([urllib.error.HTTPError(
            "http://example.test", 401, "unauthorized", {}, None
        )])
        router = OpenAICompatRouter(
            config=self._config(max_retries=5, allow_fallback=False),
            api_key="bad-key",
            transport=transport,
            sleep=lambda _: None,
        )
        with self.assertRaises(RouterError):
            router.submit(_req())
        self.assertEqual(len(transport.calls), 1)

    def test_transient_exhaustion_falls_back(self):
        transport = QueueTransport(
            [_completion("", status=503) for _ in range(3)]
        )
        router = OpenAICompatRouter(
            config=self._config(max_retries=2),
            api_key="test-key",
            transport=transport,
            sleep=lambda _: None,
        )
        resp = router.submit(_req())
        self.assertEqual(len(transport.calls), 3)
        self.assertTrue(resp.fallback_used)
        self.assertIn("transient", resp.error)

    def test_contract_error_does_not_retry(self):
        transport = QueueTransport(
            [_completion("", body=b"not json", headers={"Content-Type": "application/json"})]
        )
        router = OpenAICompatRouter(
            config=self._config(max_retries=5),
            api_key="test-key",
            transport=transport,
            sleep=lambda _: None,
        )
        resp = router.submit(_req())
        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(resp.fallback_used)
        self.assertIn("contract", resp.error)

    def test_content_type_is_validated(self):
        transport = QueueTransport(
            [_completion("OVERTAKE_LEFT", headers={"Content-Type": "text/html"})]
        )
        router = OpenAICompatRouter(
            config=self._config(),
            api_key="test-key",
            transport=transport,
            sleep=lambda _: None,
        )
        resp = router.submit(_req())
        self.assertTrue(resp.fallback_used)
        self.assertIn("content_type", resp.error)

    def test_finish_reason_is_supported(self):
        transport = QueueTransport(
            [_completion("OVERTAKE_LEFT", finish_reason="length")]
        )
        router = OpenAICompatRouter(
            config=self._config(),
            api_key="test-key",
            transport=transport,
            sleep=lambda _: None,
        )
        resp = router.submit(_req())
        self.assertFalse(resp.fallback_used)
        self.assertEqual(router.last_finish_reason, "length")

        filtered = QueueTransport(
            [_completion("OVERTAKE_LEFT", finish_reason="content_filter")]
        )
        router = OpenAICompatRouter(
            config=self._config(),
            api_key="test-key",
            transport=filtered,
            sleep=lambda _: None,
        )
        resp = router.submit(_req())
        self.assertTrue(resp.fallback_used)

    def test_circuit_breaker_opens_and_fails_fast(self):
        transport = QueueTransport(
            [_completion("", status=500) for _ in range(2)]
        )
        router = OpenAICompatRouter(
            config=self._config(
                max_retries=0,
                max_consecutive_failures=2,
                allow_fallback=False,
            ),
            api_key="test-key",
            transport=transport,
            sleep=lambda _: None,
        )
        for _ in range(2):
            with self.assertRaises(RouterError):
                router.submit(_req())
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(router.health().status, "failed")
        with self.assertRaises(RouterError) as ctx:
            router.submit(_req())
        self.assertIn("circuit_open", str(ctx.exception))
        self.assertEqual(len(transport.calls), 2)

    def test_circuit_breaker_can_use_explicit_fallback(self):
        transport = QueueTransport(
            [_completion("", status=500) for _ in range(2)]
        )
        router = OpenAICompatRouter(
            config=self._config(
                max_retries=0,
                max_consecutive_failures=2,
                allow_fallback=True,
            ),
            api_key="test-key",
            transport=transport,
            sleep=lambda _: None,
        )
        for _ in range(2):
            self.assertTrue(router.submit(_req()).fallback_used)
        resp = router.submit(_req())
        self.assertEqual(len(transport.calls), 2)
        self.assertTrue(resp.fallback_used)
        self.assertIn("circuit_open", resp.error)

    def test_no_key_falls_back_without_network(self):
        transport = QueueTransport([])
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "EVENT_CBF_API_KEY": ""},
            clear=False,
        ):
            router = OpenAICompatRouter(
                config=self._config(), transport=transport, sleep=lambda _: None
            )
        resp = router.submit(_req())
        self.assertEqual(transport.calls, [])
        self.assertTrue(resp.fallback_used)
        self.assertEqual(resp.error, "no_api_key")
        self.assertTrue(resp.raw_text.startswith("no_api_key:"))

    def test_close_releases_transport(self):
        transport = QueueTransport([])
        router = OpenAICompatRouter(
            config=self._config(), api_key="test-key", transport=transport
        )
        router.close()
        self.assertTrue(transport.closed)


class DryRunTests(unittest.TestCase):
    def test_dry_run_uses_scripted_answer(self):
        router = DryRunOpenAIRouter(latency_s=0.25)
        resp = router.submit(_req())
        self.assertEqual(resp.backend, "openai_compat_dryrun")
        self.assertEqual(resp.skill, "OVERTAKE_LEFT")
        self.assertTrue(resp.raw_text.startswith("dryrun:"))
        self.assertAlmostEqual(resp.ready_at, 2.25)
        self.assertFalse(resp.fallback_used)

    def test_dry_run_never_touches_urllib(self):
        router = DryRunOpenAIRouter(latency_s=0.1)
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("dry-run must not call urllib"),
        ):
            resp = router.submit(_req())
        self.assertEqual(resp.skill, "OVERTAKE_LEFT")

    def test_dry_run_is_thread_safe(self):
        router = DryRunOpenAIRouter(latency_s=0.1)

        def run(has_blocker):
            snap = _snap(
                others=(
                    [VehicleState(x=80.0, y=0.0, vx=0.0)]
                    if has_blocker
                    else [VehicleState(x=80.0, y=4.0, vx=20.0)]
                )
            )
            return router.submit(_req(snapshot=snap)).skill

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, [True, False] * 8))
        self.assertEqual(results, ["OVERTAKE_LEFT", "CRUISE_RIGHT"] * 8)


class FactoryTests(unittest.TestCase):
    def test_scripted_and_dryrun_choices(self):
        scripted = create_router(RouterConfig(backend="scripted", latency_s=0.75))
        self.assertIsInstance(scripted, ScriptedRouter)
        self.assertAlmostEqual(scripted.latency_s, 0.75)
        dry = make_router("dryrun")
        self.assertIsInstance(dry, DryRunOpenAIRouter)

    def test_unknown_preference_raises(self):
        with self.assertRaises(ConfigError):
            make_router("magic")
        with self.assertRaises(ConfigError):
            create_router(RouterConfig(backend="magic"))

    def test_auto_is_offline_unless_live_is_allowed(self):
        env = {"OPENAI_API_KEY": "test-key", "EVENT_CBF_API_KEY": ""}
        with patch.dict(os.environ, env, clear=False):
            auto = make_router("auto")
            self.assertIsInstance(auto, DryRunOpenAIRouter)
            live = make_router("auto", allow_live=True)
            self.assertIsInstance(live, OpenAICompatRouter)

    def test_explicit_openai_means_live_backend_even_without_key(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "EVENT_CBF_API_KEY": ""},
            clear=False,
        ):
            router = make_router("openai")
        self.assertIsInstance(router, OpenAICompatRouter)

    def test_explicit_api_key_counts_as_live_opt_in(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "EVENT_CBF_API_KEY": ""},
            clear=False,
        ):
            router = make_router("auto", allow_live=True, api_key="test-key")
        self.assertIsInstance(router, OpenAICompatRouter)

    def test_create_router_local_is_explicit(self):
        router = create_router(
            RouterConfig(backend="local"), load_model=False
        )
        self.assertIsInstance(router, LocalLLMRouter)

    def test_make_router_latency_override(self):
        router = make_router("scripted", latency_s=1.25)
        self.assertAlmostEqual(router.latency_s, 1.25)


class LocalFallbackTests(unittest.TestCase):
    def test_unloaded_local_backend_reports_fallback(self):
        router = LocalLLMRouter(
            config=RouterConfig(backend="local"), load_model=False
        )
        resp = router.submit(_req())
        self.assertTrue(resp.fallback_used)
        self.assertEqual(resp.backend, "scripted_fallback")
        self.assertIn("load_model_disabled", resp.error)
        self.assertEqual(resp.skill, "OVERTAKE_LEFT")
        self.assertFalse(router.health().healthy)

    def test_close_unloads_and_submit_falls_back(self):
        router = LocalLLMRouter(
            config=RouterConfig(backend="local"), load_model=False
        )
        router.close()
        resp = router.submit(_req())
        self.assertTrue(resp.fallback_used)
        self.assertIn("closed", resp.error)


if __name__ == "__main__":
    unittest.main()
