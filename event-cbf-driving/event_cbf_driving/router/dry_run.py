"""Dry-run OpenAI-compatible router.

The public class name is unchanged, but the implementation no longer patches
``urllib`` globally.  It is a normal :class:`OpenAICompatRouter` with an
injected, thread-local mock transport that answers with the scripted teacher's
decision.  This exercises the real prompt/HTTP/JSON/parse path while remaining
offline, deterministic, and thread-safe.
"""

from __future__ import annotations

import copy
import json
import threading
from typing import Optional

from event_cbf_driving.config import RouterConfig
from event_cbf_driving.errors import RouterError
from event_cbf_driving.router.base import (
    RouterBackend,
    RouterHealth,
    RouterRequest,
    RouterResponse,
    validate_response,
)
from event_cbf_driving.router.openai_compat import (
    OpenAICompatRouter,
    TransportResponse,
)
from event_cbf_driving.router.scripted import ScriptedRouter


class _ScriptedCompletionTransport:
    """Mock HTTP transport that renders the scripted teacher answer as JSON.

    The request is bound per thread so concurrent submits cannot mix their
    snapshots.
    """

    def __init__(self, teacher: ScriptedRouter, model: str):
        self._teacher = teacher
        self.model = model
        self._local = threading.local()

    def bind(self, request: RouterRequest) -> None:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        stack.append(request)

    def unbind(self) -> None:
        stack = getattr(self._local, "stack", None)
        if stack:
            stack.pop()

    def request(
        self,
        url: str,
        *,
        method: str = "POST",
        body: bytes,
        headers,
        timeout: float,
    ) -> TransportResponse:
        stack = getattr(self._local, "stack", None)
        if not stack:
            raise RuntimeError("dry-run transport used without a bound request")
        request = stack[-1]

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"dry-run transport received invalid JSON: {exc}") from exc
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise RuntimeError("dry-run transport expected system+user messages")

        skill = self._teacher.decide(request)
        completion = {
            "id": "chatcmpl-dryrun",
            "object": "chat.completion",
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": skill},
                    "finish_reason": "stop",
                }
            ],
            # Dry-run tokens are not real provider usage; keep them at zero so
            # canonical offline manifests never report invented telemetry.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return TransportResponse(
            status=200,
            body=json.dumps(completion).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    def close(self) -> None:
        return None


class DryRunOpenAIRouter(RouterBackend):
    """OpenAI-compatible router backed by an injected scripted transport."""

    name = "openai_compat_dryrun"

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        *,
        latency_s: Optional[float] = None,
        model: Optional[str] = None,
        **overrides,
    ):
        super().__init__()
        cfg = copy.deepcopy(config) if config is not None else RouterConfig()
        if latency_s is not None:
            cfg.latency_s = latency_s
        cfg.backend = "dryrun"
        # A dry run must not silently fall back; failures should surface.
        cfg.allow_fallback = False
        cfg.fail_on_fallback = False
        cfg.max_retries = 0
        cfg.backoff_s = 0.0
        self.latency_s = float(cfg.latency_s)
        self.model = model or "dry-run-skill-router"
        # These are owned by the dry-run harness itself; ignore user overrides
        # that would bypass the injected transport or re-enable fallback.
        for owned in (
            "api_key",
            "base_url",
            "model",
            "latency_s",
            "allow_fallback",
            "fail_on_fallback",
            "max_retries",
            "backoff_s",
            "transport",
            "sleep",
        ):
            overrides.pop(owned, None)
        self._teacher = ScriptedRouter(latency_s=self.latency_s)
        self._transport = _ScriptedCompletionTransport(self._teacher, self.model)
        self._inner = OpenAICompatRouter(
            config=cfg,
            api_key="dry-run-key",
            base_url="http://dry-run.local/v1",
            model=self.model,
            latency_s=self.latency_s,
            allow_fallback=False,
            max_retries=0,
            transport=self._transport,
            sleep=lambda _delay: None,
            **overrides,
        )

    def describe(self) -> str:
        return f"openai_compat_dryrun({self.model})"

    def submit(self, req: RouterRequest) -> RouterResponse:
        self._transport.bind(req)
        try:
            response = self._inner.submit(req)
        finally:
            self._transport.unbind()

        if response.fallback_used:  # Defensive: dry-run should not fall back.
            raise RouterError(f"dry-run router unexpectedly fell back: {response.error}")
        response.backend = self.name
        response.raw_text = f"dryrun:{response.raw_text}"
        response.latency_s = self.latency_s
        response.ready_at = req.t + self.latency_s
        response.live = False
        response.model = self.model
        response.usage = {}
        response.finish_reason = "stop"
        validate_response(response, req)
        return response

    def health(self) -> RouterHealth:
        return self._inner.health()

    def reset(self) -> None:
        self._inner.reset()
        super().reset()

    def close(self) -> None:
        self._transport.close()
        self._inner.close()


# Friendly alias for new code.
DryRunRouter = DryRunOpenAIRouter

__all__ = ["DryRunOpenAIRouter", "DryRunRouter"]
