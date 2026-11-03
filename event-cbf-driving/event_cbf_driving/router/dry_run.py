"""Dry-run OpenAI-compatible router: real prompt → mock HTTP → parse_skill.

Used when no API key is present but M2 still needs to exercise the LLM
contract end-to-end. The mock response body is filled by ScriptedRouter.decide
(same interrupt timing / latency as live backends).
"""

from __future__ import annotations

import io
import json
from typing import Optional
from unittest.mock import patch

from event_cbf_driving.router.base import RouterBackend, RouterRequest, RouterResponse
from event_cbf_driving.router.openai_compat import OpenAICompatRouter
from event_cbf_driving.router.prompt import build_chat_prompt, parse_skill
from event_cbf_driving.router.scripted import ScriptedRouter


class DryRunOpenAIRouter(RouterBackend):
    """OpenAICompatRouter with urllib mocked to a local JSON completion."""

    name = "openai_compat_dryrun"

    def __init__(self, latency_s: float = 0.30, model: str = "dry-run-skill-router"):
        self.latency_s = float(latency_s)
        self.model = model
        self._inner = OpenAICompatRouter(
            api_key="dry-run-key",
            base_url="http://dry-run.local/v1",
            model=model,
            latency_s=latency_s,
        )
        self._teacher = ScriptedRouter(latency_s=latency_s)

    def describe(self) -> str:
        return f"openai_compat_dryrun({self.model})"

    def submit(self, req: RouterRequest) -> RouterResponse:
        prompt = build_chat_prompt(req.current_skill, req.snapshot, req.reason)
        teacher_skill = self._teacher.decide(req)
        payload = {
            "choices": [{"message": {"content": teacher_skill}}],
            "model": self.model,
        }
        raw = json.dumps(payload).encode("utf-8")

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(request, timeout=None):
            assert "chat/completions" in request.full_url
            body = json.loads(request.data.decode())
            assert body["messages"][0]["content"] == prompt
            return FakeResp(raw)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            resp = self._inner.submit(req)

        # Ensure parse path produced a valid skill.
        skill = parse_skill(resp.raw_text) or teacher_skill
        return RouterResponse(
            skill=skill,
            ready_at=req.t + self.latency_s,
            backend=self.name,
            raw_text=f"dryrun:{resp.raw_text}",
            latency_s=self.latency_s,
        )
