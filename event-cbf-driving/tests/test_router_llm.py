"""Dry-run unit tests: prompt -> mock HTTP/local -> parse -> skill token.

No live API key or model download required.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch

from event_cbf_driving.env.types import VehicleState, WorldSnapshot
from event_cbf_driving.router.base import RouterRequest
from event_cbf_driving.router.openai_compat import (
    OpenAICompatRouter,
    TransportResponse,
)
from event_cbf_driving.router.prompt import (
    build_chat_prompt,
    build_messages,
    build_prompt,
    parse_skill,
)


def _snap(ego_x=60.0, others=None):
    return WorldSnapshot(
        t=1.0,
        ego=VehicleState(x=ego_x, y=0.0, vx=20.0, vy=0.0),
        others=others
        or [VehicleState(x=75.0, y=0.0, vx=0.0, vy=0.0, speed_limit=0.01)],
        route_length=200.0,
    )


class ParseSkillTests(unittest.TestCase):
    def test_exact_token(self):
        self.assertEqual(parse_skill("OVERTAKE_LEFT"), "OVERTAKE_LEFT")

    def test_noisy_text(self):
        self.assertEqual(parse_skill("I choose: overtake left now."), "OVERTAKE_LEFT")

    def test_fallback(self):
        self.assertEqual(parse_skill("???", fallback="CRUISE_RIGHT"), "CRUISE_RIGHT")


class RecordingTransport:
    def __init__(self, content="OVERTAKE_LEFT"):
        self.content = content
        self.calls = []

    def request(self, url, *, method="POST", body, headers, timeout):
        payload = json.loads(body.decode("utf-8"))
        self.calls.append(
            {"url": url, "method": method, "payload": payload, "headers": dict(headers)}
        )
        completion = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": self.content},
                    "finish_reason": "stop",
                }
            ]
        }
        return TransportResponse(
            status=200,
            body=json.dumps(completion).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )


class OpenAIMockHTTPTests(unittest.TestCase):
    def test_prompt_parse_skill_path_with_injected_transport(self):
        transport = RecordingTransport("OVERTAKE_LEFT\n")
        router = OpenAICompatRouter(
            api_key="test-key", latency_s=0.30, transport=transport
        )
        snap = _snap()
        req = RouterRequest(
            t=2.0,
            current_skill="CRUISE_RIGHT",
            snapshot=snap,
            sigma=4.0,
            reason="sigma",
        )
        prompt = build_chat_prompt(req.current_skill, req.snapshot, req.reason)
        self.assertIn("CRUISE_RIGHT", prompt)
        self.assertIn("OVERTAKE_LEFT", prompt)
        stem = build_prompt(req.current_skill, req.snapshot, req.reason)
        self.assertIn("right_threat=", stem)

        resp = router.submit(req)

        self.assertEqual(resp.skill, "OVERTAKE_LEFT")
        self.assertEqual(resp.backend, "openai_compat")
        self.assertAlmostEqual(resp.ready_at, 2.0 + 0.30)
        self.assertIn("OVERTAKE", resp.raw_text)
        self.assertFalse(resp.fallback_used)

        self.assertEqual(len(transport.calls), 1)
        call = transport.calls[0]
        self.assertIn("chat/completions", call["url"])
        self.assertTrue(call["headers"]["Authorization"].startswith("Bearer "))
        self.assertEqual(
            call["payload"]["messages"],
            build_messages("CRUISE_RIGHT", snap, "sigma", request=req),
        )
        self.assertEqual(call["payload"]["messages"][0]["role"], "system")
        self.assertEqual(call["payload"]["messages"][1]["role"], "user")

    def test_default_urllib_transport_is_still_supported(self):
        router = OpenAICompatRouter(api_key="test-key", latency_s=0.30)
        req = RouterRequest(
            t=2.0,
            current_skill="CRUISE_RIGHT",
            snapshot=_snap(),
            sigma=4.0,
            reason="sigma",
        )
        payload = {"choices": [{"message": {"content": "OVERTAKE_LEFT\n"}}]}
        raw = json.dumps(payload).encode("utf-8")

        class FakeResp(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=None):
            self.assertIn("chat/completions", request.full_url)
            self.assertTrue(request.get_header("Authorization").startswith("Bearer "))
            return FakeResp(raw)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            resp = router.submit(req)

        self.assertEqual(resp.skill, "OVERTAKE_LEFT")
        self.assertEqual(resp.backend, "openai_compat")

    def test_no_key_falls_back_scripted(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "EVENT_CBF_API_KEY": ""},
            clear=False,
        ):
            router = OpenAICompatRouter(api_key=None, latency_s=0.30)
        req = RouterRequest(
            t=0.0,
            current_skill="CRUISE_RIGHT",
            snapshot=_snap(),
            sigma=1.0,
            reason="sigma",
        )
        resp = router.submit(req)
        self.assertEqual(resp.backend, "scripted_fallback")
        self.assertTrue(resp.fallback_used)
        self.assertEqual(resp.error, "no_api_key")
        self.assertEqual(resp.skill, "OVERTAKE_LEFT")  # stalled ahead

    def test_dryrun_openai_router(self):
        from event_cbf_driving.router.dry_run import DryRunOpenAIRouter

        router = DryRunOpenAIRouter(latency_s=0.30)
        req = RouterRequest(
            t=1.0,
            current_skill="CRUISE_RIGHT",
            snapshot=_snap(),
            sigma=4.0,
            reason="sigma",
        )
        resp = router.submit(req)
        self.assertEqual(resp.backend, "openai_compat_dryrun")
        self.assertEqual(resp.skill, "OVERTAKE_LEFT")
        self.assertTrue(resp.raw_text.startswith("dryrun:"))
        self.assertAlmostEqual(resp.ready_at, 1.0 + 0.30)


if __name__ == "__main__":
    unittest.main()
