"""OpenAI-compatible Tier-1 router (Chat Completions over HTTP).

Env vars:
  OPENAI_API_KEY or EVENT_CBF_API_KEY  - required for live calls
  OPENAI_BASE_URL or EVENT_CBF_BASE_URL - optional (default https://api.openai.com/v1)
  EVENT_CBF_OPENAI_MODEL - optional (default gpt-4o-mini)

Only transient failures (timeouts, 408, 429, 5xx) are retried.  Authentication
and validation failures (other 4xx) are never retried.  After
``max_consecutive_failures`` the circuit opens and calls fail fast or use the
scripted fallback, depending on ``allow_fallback``/``fail_on_fallback``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol

from event_cbf_driving.config import RouterConfig
from event_cbf_driving.errors import RouterError
from event_cbf_driving.logging_utils import get_logger
from event_cbf_driving.router.base import (
    RouterBackend,
    RouterRequest,
    RouterResponse,
    normalise_usage,
    validate_response,
)
from event_cbf_driving.router.prompt import (
    PROMPT,
    SYSTEM_PROMPT,
    build_chat_prompt,
    build_messages,
    build_prompt,
    parse_skill,
    prompt_hash,
)
from event_cbf_driving.router.scripted import ScriptedRouter

# Re-export for older imports / tests.
__all__ = [
    "MAX_RESPONSE_BYTES",
    "OpenAICompatRouter",
    "PROMPT",
    "SYSTEM_PROMPT",
    "Transport",
    "TransportError",
    "TransportResponse",
    "UrllibTransport",
    "build_chat_prompt",
    "build_messages",
    "build_prompt",
    "parse_skill",
    "prompt_hash",
]

logger = get_logger(__name__)

MAX_RESPONSE_BYTES = 1_000_000
_TRANSIENT_STATUS = frozenset({408, 429})


class TransportError(Exception):
    """Raised by an injected transport.  ``transient`` controls retries."""

    def __init__(self, message: str, *, transient: bool = True):
        super().__init__(message)
        self.transient = bool(transient)


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)


class Transport(Protocol):
    def request(
        self,
        url: str,
        *,
        method: str = "POST",
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:  # pragma: no cover - protocol definition
        ...


class UrllibTransport:
    """Default stdlib transport.  HTTP error responses become normal results."""

    def __init__(self, max_response_bytes: int = MAX_RESPONSE_BYTES):
        self.max_response_bytes = int(max_response_bytes)

    def request(
        self,
        url: str,
        *,
        method: str = "POST",
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
                return TransportResponse(
                    status=int(getattr(response, "status", 200)),
                    body=raw,
                    headers=dict(getattr(response, "headers", None) or {}),
                )
        except urllib.error.HTTPError as exc:
            raw = b""
            try:
                raw = exc.read(self.max_response_bytes + 1)
            except OSError:
                pass
            return TransportResponse(
                status=int(exc.code),
                body=raw,
                headers=dict(exc.headers or {}),
            )

    def close(self) -> None:
        return None


class _TransientFailure(Exception):
    pass


class _PermanentFailure(Exception):
    pass


class _ContractFailure(Exception):
    pass


class OpenAICompatRouter(RouterBackend):
    name = "openai_compat"
    is_live_capable = True

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        *,
        # Backward-compatible keyword arguments.
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        latency_s: Optional[float] = None,
        timeout_s: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_s: Optional[float] = None,
        allow_fallback: Optional[bool] = None,
        fail_on_fallback: Optional[bool] = None,
        max_consecutive_failures: Optional[int] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        transport: Optional[Transport] = None,
        sleep: Optional[Callable[[float], None]] = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ):
        super().__init__()
        cfg = copy.deepcopy(config) if config is not None else RouterConfig()
        self.allow_live = bool(cfg.allow_live)

        self.latency_s = float(cfg.latency_s if latency_s is None else latency_s)
        self.timeout_s = float(cfg.timeout_s if timeout_s is None else timeout_s)
        self.max_retries = int(cfg.max_retries if max_retries is None else max_retries)
        self.backoff_s = float(cfg.backoff_s if backoff_s is None else backoff_s)
        self.allow_fallback = (
            bool(cfg.allow_fallback) if allow_fallback is None else bool(allow_fallback)
        )
        self.fail_on_fallback = (
            bool(cfg.fail_on_fallback) if fail_on_fallback is None else bool(fail_on_fallback)
        )
        self.max_consecutive_failures = int(
            cfg.max_consecutive_failures
            if max_consecutive_failures is None
            else max_consecutive_failures
        )
        self.seed = int(cfg.seed if seed is None else seed)
        self.temperature = float(cfg.temperature if temperature is None else temperature)
        self.max_tokens = int(cfg.max_tokens if max_tokens is None else max_tokens)
        self.max_response_bytes = int(max_response_bytes)

        self.api_key = (
            api_key
            or os.environ.get(cfg.api_key_env)
            or os.environ.get(cfg.api_key_env_fallback)
        )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("EVENT_CBF_BASE_URL")
            or cfg.base_url
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("EVENT_CBF_OPENAI_MODEL")
            or cfg.model
        )

        if self.max_retries < 0:
            raise RouterError(f"invalid max_retries {self.max_retries!r}")
        if self.max_consecutive_failures < 1:
            raise RouterError(
                f"invalid max_consecutive_failures {self.max_consecutive_failures!r}"
            )
        if self.timeout_s <= 0.0:
            raise RouterError(f"invalid timeout_s {self.timeout_s!r}")
        if not math.isfinite(self.latency_s) or self.latency_s < 0.0:
            raise RouterError(f"invalid latency_s {self.latency_s!r}")
        if not math.isfinite(self.backoff_s) or self.backoff_s < 0.0:
            raise RouterError(f"invalid backoff_s {self.backoff_s!r}")
        if self.max_tokens < 1:
            raise RouterError(f"invalid max_tokens {self.max_tokens!r}")

        self._fallback = ScriptedRouter(latency_s=self.latency_s)
        self._transport: Any = transport if transport is not None else UrllibTransport(
            max_response_bytes=self.max_response_bytes
        )
        self._sleep = sleep or time.sleep
        self.last_finish_reason: Optional[str] = None
        self.last_usage: dict[str, int] = {}
        self.last_model: str = self.model

    # -- description / lifecycle ----------------------------------------
    def describe(self) -> str:
        fallback = "on" if self._fallback_allowed() else "off"
        if not self.api_key:
            return (
                f"openai_compat(no_key->scripted_fallback, fallback={fallback}, "
                f"retries={self.max_retries})"
            )
        return (
            f"openai_compat({self.model}@{self.base_url}, retries={self.max_retries}, "
            f"fallback={fallback})"
        )

    def close(self) -> None:
        closer = getattr(self._transport, "close", None)
        if callable(closer):
            closer()

    # -- failure / fallback policy --------------------------------------
    def _fallback_allowed(self) -> bool:
        return bool(self.allow_fallback and not self.fail_on_fallback)

    def _circuit_is_open(self) -> bool:
        state = self.health()
        return (
            state.status == "failed"
            and state.consecutive_failures >= self.max_consecutive_failures
        )

    def _fallback_response(
        self,
        req: RouterRequest,
        reason: str,
        *,
        prompt_digest: str,
        started: float,
    ) -> RouterResponse:
        elapsed = time.perf_counter() - started
        if not self._fallback_allowed():
            self._record_failure(
                reason,
                latency_s=elapsed,
                failure_threshold=self.max_consecutive_failures,
            )
            raise RouterError(f"{self.name}: {self._short_error(reason)}")

        self._record_failure(
            reason,
            latency_s=elapsed,
            failure_threshold=self.max_consecutive_failures,
        )
        teacher = self._fallback.submit(req)
        response = RouterResponse(
            skill=teacher.skill,
            ready_at=req.t + self.latency_s,
            backend="scripted_fallback",
            raw_text=f"{reason}:{teacher.raw_text}",
            latency_s=self.latency_s,
            request_id=self._request_id(req),
            submitted_t=req.t,
            wall_latency_s=time.perf_counter() - started,
            fallback_used=True,
            error=self._short_error(reason),
            prompt_hash=prompt_digest,
            health=self.health(),
            live=False,
            model="",
            usage={},
            finish_reason=None,
        )
        validate_response(response, req)
        logger.warning(
            "router fallback %s -> scripted (%s)",
            self.name,
            self._short_error(reason),
            extra={"event": "router_fallback", "backend": self.name, "reason": response.error},
        )
        return response

    def _raise_or_fallback(
        self, req: RouterRequest, reason: str, *, prompt_digest: str, started: float
    ) -> RouterResponse:
        return self._fallback_response(
            req, reason, prompt_digest=prompt_digest, started=started
        )

    # -- HTTP ------------------------------------------------------------
    @staticmethod
    def _normalise_transport_result(result: Any) -> TransportResponse:
        if isinstance(result, TransportResponse):
            return result
        if isinstance(result, tuple) and len(result) == 3:
            status, headers, body = result
            return TransportResponse(status=int(status), body=body, headers=headers or {})
        if hasattr(result, "status"):
            body = getattr(result, "body", None)
            if body is None and hasattr(result, "read"):
                body = result.read(MAX_RESPONSE_BYTES + 1)
            if body is not None:
                return TransportResponse(
                    status=int(getattr(result, "status", 200)),
                    body=body,
                    headers=dict(getattr(result, "headers", None) or {}),
                )
        # An injected transport returning the wrong type is a programming
        # error, not a model failure: let it surface.
        raise TypeError(
            "transport must return TransportResponse or (status, headers, body), "
            f"got {type(result).__name__}"
        )

    def _transport_request(
        self,
        url: str,
        *,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        transport = self._transport
        if hasattr(transport, "request"):
            result = transport.request(
                url, method="POST", body=body, headers=headers, timeout=timeout
            )
        elif callable(transport):
            result = transport(url, body=body, headers=headers, timeout=timeout)
        else:
            raise TypeError("transport must provide request(...) or be callable")
        return self._normalise_transport_result(result)

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.backoff_s <= 0.0:
            return
        delay = self.backoff_s * (2.0**attempt)
        # Small jitter avoids synchronised retry storms.
        delay += random.uniform(0.0, min(0.05, self.backoff_s * 0.25))
        self._sleep(delay)

    def _post(self, messages: list[dict]) -> TransportResponse:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return self._transport_request(
            url, body=body, headers=headers, timeout=self.timeout_s
        )

    def _decode_completion(self, result: TransportResponse) -> str:
        headers = {
            str(key).lower(): str(value)
            for key, value in dict(result.headers or {}).items()
        }
        content_type = headers.get("content-type", "")
        if content_type and "json" not in content_type.lower():
            raise _ContractFailure(f"unexpected_content_type:{content_type[:40]}")

        body = result.body
        if isinstance(body, str):
            body = body.encode("utf-8")
        elif isinstance(body, (bytearray, memoryview)):
            body = bytes(body)
        if not isinstance(body, bytes):
            raise TypeError(
                f"transport body must be bytes, got {type(body).__name__}"
            )
        if len(body) > self.max_response_bytes:
            raise _ContractFailure("response_too_large")
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _ContractFailure("response_not_utf8") from exc
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise _ContractFailure("response_not_json") from exc
        if not isinstance(payload, dict):
            raise _ContractFailure("response_not_object")

        self.last_usage = normalise_usage(payload.get("usage"))
        provider_model = payload.get("model")
        self.last_model = (
            provider_model
            if isinstance(provider_model, str) and provider_model.strip()
            else self.model
        )

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _ContractFailure("missing_choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise _ContractFailure("choice_not_object")

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise _ContractFailure("invalid_finish_reason")
        if finish_reason == "content_filter":
            raise _ContractFailure("content_filtered")
        self.last_finish_reason = finish_reason

        message = choice.get("message")
        if not isinstance(message, dict):
            raise _ContractFailure("missing_message")
        content = message.get("content")
        if not isinstance(content, str):
            raise _ContractFailure("content_not_string")
        return content.strip()

    def _complete(self, messages: list[dict]) -> str:
        attempts = self.max_retries + 1
        last_error: Optional[str] = None
        for attempt in range(attempts):
            try:
                result = self._post(messages)
            except TransportError as exc:
                last_error = self._short_error(exc)
                if not exc.transient:
                    raise _PermanentFailure(last_error) from exc
            except urllib.error.HTTPError as exc:
                # Some injected transports raise HTTPError instead of
                # returning a status; classify it exactly like a response.
                last_error = f"http_{exc.code}"
                if exc.code not in _TRANSIENT_STATUS and not 500 <= exc.code <= 599:
                    raise _PermanentFailure(last_error) from exc
            except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
                last_error = self._short_error(exc) or type(exc).__name__
            else:
                status = int(result.status)
                if status in _TRANSIENT_STATUS or 500 <= status <= 599:
                    last_error = f"http_{status}"
                elif status < 200 or status >= 300:
                    snippet = ""
                    try:
                        snippet = result.body[:120].decode("utf-8", errors="replace")
                    except Exception:
                        snippet = ""
                    raise _PermanentFailure(
                        self._short_error(f"http_{status}:{snippet}".rstrip(":"))
                    )
                else:
                    return self._decode_completion(result)

            if attempt + 1 < attempts:
                self._sleep_before_retry(attempt)
                continue
            raise _TransientFailure(last_error or "transport_error")
        raise _TransientFailure(last_error or "transport_error")

    def _chat(self, prompt: str) -> str:
        """Legacy single-turn helper used by older callers/tests."""
        return self._complete([{"role": "user", "content": prompt}])

    # -- public API ------------------------------------------------------
    def submit(self, req: RouterRequest) -> RouterResponse:
        started = time.perf_counter()
        messages = build_messages(
            req.current_skill, req.snapshot, req.reason, request=req
        )
        digest = prompt_hash(
            messages[0]["content"] + "\n\n" + messages[1]["content"]
        )

        if not self.api_key:
            return self._raise_or_fallback(
                req, "no_api_key", prompt_digest=digest, started=started
            )

        if self._circuit_is_open():
            state = self.health()
            return self._raise_or_fallback(
                req,
                f"circuit_open:{state.last_error or 'backend_failed'}",
                prompt_digest=digest,
                started=started,
            )

        try:
            text = self._complete(messages)
        except _TransientFailure as exc:
            return self._raise_or_fallback(
                req, f"transient:{exc}", prompt_digest=digest, started=started
            )
        except _PermanentFailure as exc:
            return self._raise_or_fallback(
                req, f"permanent:{exc}", prompt_digest=digest, started=started
            )
        except _ContractFailure as exc:
            return self._raise_or_fallback(
                req, f"contract:{exc}", prompt_digest=digest, started=started
            )

        skill = parse_skill(text)
        if skill is None:
            return self._raise_or_fallback(
                req,
                f"unparseable_skill:{self._short_error(text, 60)}",
                prompt_digest=digest,
                started=started,
            )

        wall = time.perf_counter() - started
        response = RouterResponse(
            skill=skill,
            ready_at=req.t + self.latency_s,
            backend=self.name,
            raw_text=text[:500],
            latency_s=self.latency_s,
            request_id=self._request_id(req),
            submitted_t=req.t,
            wall_latency_s=wall,
            fallback_used=False,
            error=None,
            prompt_hash=digest,
            live=True,
            model=self.last_model,
            usage=dict(self.last_usage),
            finish_reason=self.last_finish_reason,
        )
        validate_response(response, req)
        self._record_success(wall, usage=self.last_usage)
        response.health = self.health()
        return response
