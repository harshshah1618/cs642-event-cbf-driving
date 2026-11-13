"""OpenCode-hosted model router.

OpenCode exposes a local HTTP server (``opencode serve``) with a stateless
text-generation endpoint.  That endpoint can route to the free DeepSeek Flash
model installed in OpenCode without requiring a separate provider key.

Connection details are read from the service state file written by OpenCode
(``~/.local/state/opencode/service.json`` by default).  The file contains the
server URL and a short-lived local password; neither is logged or serialized.
An explicit URL/password or ``OPENCODE_SERVER_PASSWORD`` can override it.

The endpoint returns only text, not token usage.  Live responses therefore
carry ``live=True`` with empty usage rather than invented token counts.  As with
the other live backends, fallback behavior is explicit and audited.
"""

from __future__ import annotations

import base64
import copy
import json
import math
import os
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from config import RouterConfig
from errors import RouterError
from logging_utils import get_logger
from planner import ScriptedRouter
from router_base import RouterBackend, RouterRequest, RouterResponse, validate_response
from router_http import MAX_RESPONSE_BYTES, Transport, TransportResponse, UrllibTransport
from router_prompt import build_prompt, parse_skill, prompt_hash

DEFAULT_STATE_FILE = "~/.local/state/opencode/service.json"
DEFAULT_PROVIDER = "opencode"
DEFAULT_MODEL = "deepseek-v4.1-flash-free"
GENERATE_PATH = "/api/experimental/generate/text"

# A provider that asks us to wait longer than this should fail immediately
# rather than burning retries against a quota window.
_MAX_INLINE_RETRY_AFTER_S = 60.0

logger = get_logger(__name__)


class _OpenCodeFailure(Exception):
    def __init__(
        self,
        message: str,
        *,
        transient: bool = False,
        retry_after_s: Optional[float] = None,
    ):
        super().__init__(message)
        self.transient = bool(transient)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class _Service:
    url: str
    password: str


class OpenCodeRouter(RouterBackend):
    """Route skill decisions through a locally running OpenCode server."""

    name = "opencode"
    is_live_capable = True

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        variant: Optional[str] = None,
        url: Optional[str] = None,
        password: Optional[str] = None,
        state_file: Optional[str] = None,
        password_env: Optional[str] = None,
        latency_s: Optional[float] = None,
        timeout_s: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_s: Optional[float] = None,
        allow_fallback: Optional[bool] = None,
        fail_on_fallback: Optional[bool] = None,
        max_consecutive_failures: Optional[int] = None,
        transport: Optional[Transport] = None,
        sleep: Optional[Callable[[float], None]] = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ):
        super().__init__()
        cfg = copy.deepcopy(config) if config is not None else RouterConfig()
        self.allow_live = bool(cfg.allow_live)
        if not self.allow_live:
            raise RouterError(
                "opencode router requires live calls to be explicitly enabled "
                "(set router.allow_live=true or pass --allow-live)"
            )

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
        self.max_response_bytes = int(max_response_bytes)

        self.model = str(model or cfg.model).strip()
        self.provider = str(provider or cfg.opencode_provider or DEFAULT_PROVIDER).strip()
        self.variant = (
            str(variant).strip()
            if variant is not None
            else (cfg.opencode_variant.strip() if cfg.opencode_variant else None)
        ) or None
        self.url = (url if url is not None else cfg.opencode_url) or None
        self.state_file = (
            state_file if state_file is not None else cfg.opencode_state_file
        )
        self.password_env = str(
            password_env or cfg.opencode_password_env or "OPENCODE_SERVER_PASSWORD"
        ).strip()
        self._explicit_password = password

        if not self.model:
            raise RouterError("opencode router requires a non-empty model")
        if not self.provider:
            raise RouterError("opencode router requires a non-empty provider")
        if self.max_retries < 0:
            raise RouterError(f"invalid max_retries {self.max_retries!r}")
        if self.max_consecutive_failures < 1:
            raise RouterError(
                f"invalid max_consecutive_failures {self.max_consecutive_failures!r}"
            )
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise RouterError(f"invalid timeout_s {self.timeout_s!r}")
        if not math.isfinite(self.latency_s) or self.latency_s < 0.0:
            raise RouterError(f"invalid latency_s {self.latency_s!r}")
        if not math.isfinite(self.backoff_s) or self.backoff_s < 0.0:
            raise RouterError(f"invalid backoff_s {self.backoff_s!r}")

        self._fallback = ScriptedRouter(latency_s=self.latency_s)
        self._transport: Any = transport if transport is not None else UrllibTransport(
            max_response_bytes=self.max_response_bytes
        )
        self._sleep = sleep or time.sleep
        self._service: Optional[_Service] = None
        self._service_error = ""
        self._closed = False

    # -- service discovery ----------------------------------------------
    def _service_info(self) -> _Service:
        if self._service is not None:
            return self._service

        url = (self.url or "").strip().rstrip("/")
        password = (self._explicit_password or "").strip()
        password = password or os.environ.get(self.password_env, "").strip()

        if not url or not password:
            path = Path(self.state_file).expanduser() if self.state_file else None
            if path is None or not path.is_file():
                self._service_error = (
                    f"service_state_missing:{path}" if path is not None else "service_state_unset"
                )
                raise _OpenCodeFailure(self._service_error)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except OSError as exc:
                self._service_error = (
                    f"service_state_unreadable:{self._short_error(exc)}"
                )
                raise _OpenCodeFailure(self._service_error) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._service_error = f"service_state_bad_json:{self._short_error(exc)}"
                raise _OpenCodeFailure(self._service_error) from exc
            if not isinstance(payload, dict):
                self._service_error = "service_state_not_object"
                raise _OpenCodeFailure(self._service_error)

            file_url = payload.get("url")
            file_password = payload.get("password")
            if not url:
                url = file_url.strip().rstrip("/") if isinstance(file_url, str) else ""
            if not password:
                password = file_password.strip() if isinstance(file_password, str) else ""

        if not url or not password:
            self._service_error = "service_state_incomplete"
            raise _OpenCodeFailure(self._service_error)
        if not url.startswith(("http://", "https://")):
            self._service_error = "service_url_invalid"
            raise _OpenCodeFailure(self._service_error)

        self._service = _Service(url=url, password=password)
        self._service_error = ""
        return self._service

    # -- description / lifecycle ----------------------------------------
    def describe(self) -> str:
        fallback = "on" if self._fallback_allowed() else "off"
        location = self.url or self.state_file or "service-auto"
        detail = f", service={self._service_error}" if self._service_error else ""
        return (
            f"opencode({self.model}@{self.provider}{detail}, "
            f"endpoint={location}, retries={self.max_retries}, fallback={fallback})"
        )

    def close(self) -> None:
        self._closed = True
        closer = getattr(self._transport, "close", None)
        if callable(closer):
            closer()

    def reset(self) -> None:
        self._service = None
        self._service_error = ""
        super().reset()

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

    # -- HTTP ------------------------------------------------------------
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
        if isinstance(result, TransportResponse):
            return result
        if isinstance(result, tuple) and len(result) == 3:
            status, response_headers, response_body = result
            return TransportResponse(
                status=int(status), body=response_body, headers=response_headers or {}
            )
        if hasattr(result, "status"):
            response_body = getattr(result, "body", None)
            if response_body is None and hasattr(result, "read"):
                response_body = result.read(MAX_RESPONSE_BYTES + 1)
            if response_body is not None:
                return TransportResponse(
                    status=int(getattr(result, "status", 200)),
                    body=response_body,
                    headers=dict(getattr(result, "headers", None) or {}),
                )
        raise TypeError(
            "transport must return TransportResponse or (status, headers, body), "
            f"got {type(result).__name__}"
        )

    def _post(self, prompt: str, service: _Service) -> TransportResponse:
        model: dict[str, Any] = {"id": self.model, "providerID": self.provider}
        if self.variant:
            model["variant"] = self.variant
        payload = {"prompt": prompt, "model": model}
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        token = base64.b64encode(
            f"opencode:{service.password}".encode("utf-8")
        ).decode("ascii")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {token}",
        }
        return self._transport_request(
            service.url + GENERATE_PATH,
            body=body,
            headers=headers,
            timeout=self.timeout_s,
        )

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.backoff_s > 0.0:
            self._sleep(self.backoff_s * (2.0**attempt))

    def _body_text(self, result: TransportResponse) -> str:
        body = result.body
        if isinstance(body, str):
            return body
        if isinstance(body, (bytearray, memoryview)):
            body = bytes(body)
        if not isinstance(body, bytes):
            raise TypeError(f"transport body must be bytes, got {type(body).__name__}")
        if len(body) > self.max_response_bytes:
            raise _OpenCodeFailure("response_too_large")
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _OpenCodeFailure("response_not_utf8") from exc

    def _error_detail(
        self, text: str
    ) -> tuple[str, Optional[int], Optional[float]]:
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return self._short_error(text, 160), None, None
        if not isinstance(payload, dict):
            return self._short_error(text, 160), None, None

        parts = []
        reason = payload.get("reason")
        message = payload.get("message")
        tag = payload.get("_tag")
        if isinstance(reason, str) and reason.strip():
            parts.append(reason.strip())
        if isinstance(message, str) and message.strip():
            parts.append(message.strip())
        elif isinstance(tag, str) and tag.strip():
            parts.append(tag.strip())

        raw_status = payload.get("status")
        status: Optional[int] = None
        if isinstance(raw_status, int) and not isinstance(raw_status, bool):
            status = raw_status
        raw_retry = payload.get("retryAfterMs")
        retry_after_s: Optional[float] = None
        if isinstance(raw_retry, (int, float)) and not isinstance(raw_retry, bool):
            if math.isfinite(float(raw_retry)) and raw_retry >= 0:
                retry_after_s = float(raw_retry) / 1000.0
        detail = ":".join(parts) if parts else self._short_error(text, 160)
        return detail, status, retry_after_s

    def _decode_text(self, result: TransportResponse) -> str:
        text = self._body_text(result)
        http_status = int(result.status)
        detail, provider_status, retry_after_s = self._error_detail(text)
        effective_status = provider_status if provider_status is not None else http_status

        if http_status < 200 or http_status >= 300 or effective_status >= 400:
            transient = effective_status in (408, 429) or 500 <= effective_status <= 599
            if retry_after_s is not None and retry_after_s > _MAX_INLINE_RETRY_AFTER_S:
                transient = False
            if effective_status == 429:
                prefix = "opencode_rate_limit"
            elif effective_status in (401, 403):
                prefix = "opencode_auth"
            elif effective_status == 404:
                prefix = "opencode_not_found"
            elif effective_status >= 500:
                prefix = "opencode_server_error"
            else:
                prefix = f"opencode_http_{effective_status}"
            message = f"{prefix}:{detail}" if detail else prefix
            if retry_after_s is not None:
                message += f":retry_after_s={retry_after_s:.0f}"
            if effective_status in (401, 403):
                self._service = None
            raise _OpenCodeFailure(
                message, transient=transient, retry_after_s=retry_after_s
            )

        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise _OpenCodeFailure("response_not_json") from exc
        if not isinstance(payload, dict):
            raise _OpenCodeFailure("response_not_object")
        if "_tag" in payload:
            detail, _, _ = self._error_detail(text)
            raise _OpenCodeFailure(f"opencode_error:{detail}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise _OpenCodeFailure("response_missing_data")
        response_text = data.get("text")
        if not isinstance(response_text, str):
            raise _OpenCodeFailure("response_missing_text")
        response_text = response_text.strip()
        if not response_text:
            raise _OpenCodeFailure("response_empty_text")
        return response_text

    def _complete(self, prompt: str) -> str:
        attempts = self.max_retries + 1
        last_error = "opencode_error"
        for attempt in range(attempts):
            try:
                service = self._service_info()
                result = self._post(prompt, service)
                return self._decode_text(result)
            except _OpenCodeFailure as exc:
                last_error = str(exc)
                if not exc.transient or attempt + 1 >= attempts:
                    raise
            except urllib.error.HTTPError as exc:
                # Injected transports may raise instead of returning a status.
                last_error = f"opencode_http_{exc.code}"
                transient = exc.code in (408, 429) or 500 <= exc.code <= 599
                if not transient or attempt + 1 >= attempts:
                    raise _OpenCodeFailure(last_error, transient=transient) from exc
            except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
                last_error = f"opencode_transport:{self._short_error(exc)}"
                if attempt + 1 >= attempts:
                    raise _OpenCodeFailure(last_error, transient=True) from exc
            if attempt + 1 < attempts:
                self._sleep_before_retry(attempt)
        raise _OpenCodeFailure(last_error, transient=True)

    # -- public API ------------------------------------------------------
    def submit(self, req: RouterRequest) -> RouterResponse:
        started = time.perf_counter()
        prompt = build_prompt(
            req.current_skill, req.snapshot, req.reason, request=req
        )
        digest = prompt_hash(prompt)

        if self._closed:
            return self._fallback_response(
                req, "closed", prompt_digest=digest, started=started
            )
        if self._circuit_is_open():
            state = self.health()
            return self._fallback_response(
                req,
                f"circuit_open:{state.last_error or 'backend_failed'}",
                prompt_digest=digest,
                started=started,
            )

        try:
            text = self._complete(prompt)
        except _OpenCodeFailure as exc:
            return self._fallback_response(
                req, str(exc), prompt_digest=digest, started=started
            )

        skill = parse_skill(text)
        if skill is None:
            return self._fallback_response(
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
            model=self.model,
            usage={},
            finish_reason=None,
        )
        validate_response(response, req)
        self._record_success(wall, usage={})
        response.health = self.health()
        return response


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "DEFAULT_STATE_FILE",
    "GENERATE_PATH",
    "OpenCodeRouter",
]
