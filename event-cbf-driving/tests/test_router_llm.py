"""Dry-run unit tests: prompt → mock HTTP/local → parse → skill token.

No live API key or model download required.
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from event_cbf_driving.env.types import VehicleState, WorldSnapshot
from event_cbf_driving.router.openai_compat import OpenAICompatRouter
from event_cbf_driving.router.prompt import build_chat_prompt, build_prompt, parse_skill
from event_cbf_driving.router.base import RouterRequest


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


class OpenAIMockHTTPTests(unittest.TestCase):
    def test_prompt_parse_skill_path(self):
        router = OpenAICompatRouter(api_key="test-key", latency_s=0.30)
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

        payload = {
            "choices": [
                {"message": {"content": "OVERTAKE_LEFT\n"}},
            ]
        }
        raw = json.dumps(payload).encode("utf-8")

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(request, timeout=None):
            self.assertIn("chat/completions", request.full_url)
            self.assertTrue(request.get_header("Authorization").startswith("Bearer "))
            body = json.loads(request.data.decode())
            self.assertEqual(body["messages"][0]["content"], prompt)
            return FakeResp(raw)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            resp = router.submit(req)

        self.assertEqual(resp.skill, "OVERTAKE_LEFT")
        self.assertEqual(resp.backend, "openai_compat")
        self.assertAlmostEqual(resp.ready_at, 2.0 + 0.30)
        self.assertIn("OVERTAKE", resp.raw_text)

    def test_no_key_falls_back_scripted(self):
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
