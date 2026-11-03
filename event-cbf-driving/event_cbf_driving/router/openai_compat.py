"""OpenAI-compatible Tier-1 router (Chat Completions over HTTP).

Env vars:
  OPENAI_API_KEY or EVENT_CBF_API_KEY  — required for live calls
  OPENAI_BASE_URL or EVENT_CBF_BASE_URL — optional (default https://api.openai.com/v1)
  EVENT_CBF_OPENAI_MODEL — optional (default gpt-4o-mini)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from event_cbf_driving.router.base import RouterBackend, RouterRequest, RouterResponse
from event_cbf_driving.router.prompt import PROMPT, build_chat_prompt, build_prompt, parse_skill
from event_cbf_driving.router.scripted import ScriptedRouter

# Re-export for older imports / tests.
__all__ = ["OpenAICompatRouter", "PROMPT", "build_prompt", "build_chat_prompt", "parse_skill"]


class OpenAICompatRouter(RouterBackend):
    name = "openai_compat"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        latency_s: float = 0.30,
        timeout_s: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "EVENT_CBF_API_KEY"
        )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("EVENT_CBF_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("EVENT_CBF_OPENAI_MODEL") or "gpt-4o-mini"
        self.latency_s = float(latency_s)
        self.timeout_s = float(timeout_s)
        self._fallback = ScriptedRouter(latency_s=latency_s)

    def describe(self) -> str:
        if not self.api_key:
            return "openai_compat(no_key→scripted_fallback)"
        return f"openai_compat({self.model}@{self.base_url})"

    def _chat(self, prompt: str) -> str:
        """POST /chat/completions. Isolated for unit-test HTTP mocks."""
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
                "temperature": 0.0,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()

    def submit(self, req: RouterRequest) -> RouterResponse:
        if not self.api_key:
            r = self._fallback.submit(req)
            r.backend = "scripted_fallback"
            r.raw_text = "no_api_key:" + r.raw_text
            return r
        prompt = build_chat_prompt(req.current_skill, req.snapshot, req.reason)
        try:
            text = self._chat(prompt)
            skill = parse_skill(text) or self._fallback.decide(req)
            return RouterResponse(
                skill=skill,
                ready_at=req.t + self.latency_s,
                backend=self.name,
                raw_text=text[:200],
                latency_s=self.latency_s,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            r = self._fallback.submit(req)
            r.backend = "scripted_fallback"
            r.raw_text = f"api_error:{e}"
            return r
        except Exception as e:  # pragma: no cover — unexpected SDK/transport errors
            r = self._fallback.submit(req)
            r.backend = "scripted_fallback"
            r.raw_text = f"api_error:{e}"
            return r
