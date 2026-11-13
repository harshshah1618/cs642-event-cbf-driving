"""Strict image-and-ego-state router for a caller-configured vLLM server.

No scene-object text, current-skill hint, text-only path or scripted fallback is
sent/used. Injected transports are contract tests (live=False) unless the caller
explicitly identifies them as a live transport. HTTP timeouts and retries share
one wall-time budget; a late response is rejected, never applied as timely.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
import urllib.error
from typing import Callable, Optional
from urllib.parse import urlsplit

from config import RouterConfig
from errors import RouterError
from router_base import (
    CameraObservation,
    RouterBackend,
    RouterRequest,
    RouterResponse,
    normalise_usage,
    validate_response,
)
from router_http import (
    MAX_RESPONSE_BYTES,
    Transport,
    TransportError,
    TransportResponse,
    UrllibTransport,
)

CHOICES = {"A": "CRUISE_RIGHT", "B": "OVERTAKE_LEFT", "C": "RETURN_RIGHT"}
SYSTEM_PROMPT = (
    "Choose the next driving maneuver from the front camera image and ego state. "
    "A = cruise in the right lane; B = overtake or continue in the left lane; "
    "C = return to the right lane. Respond with exactly one letter: A, B, or C. "
    "Do not add explanations."
)


class VLMRouter(RouterBackend):
    name = "vlm"
    is_live_capable = True
    requires_camera = True

    def __init__(
        self,
        config: RouterConfig,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        transport: Optional[Transport] = None,
        transport_is_live: bool = False,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_image_bytes: int = 8_000_000,
        max_image_age_s: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_url = (config.base_url if base_url is None else base_url).rstrip("/")
        self.model = config.model if model is None else model
        parts = urlsplit(self.base_url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise RouterError(
                "vlm: configure an HTTP(S) base URL without credentials/query/fragment"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise RouterError("vlm: an explicit nonempty served model is required")
        self.timeout_s = float(config.timeout_s)
        self.max_retries = int(config.max_retries)
        self.backoff_s = float(config.backoff_s)
        self.max_consecutive_failures = int(config.max_consecutive_failures)
        self.max_image_age_s = float(max_image_age_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise RouterError("vlm: timeout_s must be finite and positive")
        if not math.isfinite(self.backoff_s) or self.backoff_s < 0:
            raise RouterError("vlm: backoff_s must be finite and nonnegative")
        if self.max_retries < 0 or self.max_consecutive_failures < 1:
            raise RouterError("vlm: invalid retry/circuit limit")
        if not math.isfinite(self.max_image_age_s) or self.max_image_age_s < 0:
            raise RouterError("vlm: invalid maximum image age")
        if max_response_bytes < 1 or max_image_bytes < 1:
            raise RouterError("vlm: byte limits must be positive")
        self.max_response_bytes = int(max_response_bytes)
        self.max_image_bytes = int(max_image_bytes)
        self.api_key = (
            api_key
            if api_key is not None
            else os.environ.get(config.api_key_env) or os.environ.get(config.api_key_env_fallback)
        )
        self._transport = (
            transport if transport is not None else UrllibTransport(max_response_bytes)
        )
        self._live = transport is None or transport_is_live
        self._sleep = sleep or time.sleep
        self._clock = clock or time.perf_counter
        self.seed = config.seed
        self.latency_s = 0.0  # Responses use measured elapsed time, never this placeholder.

    def describe(self) -> str:
        return f"vlm({self.model}@{self.base_url}, strict_image, fallback=off)"

    def close(self) -> None:
        closer = getattr(self._transport, "close", None)
        if callable(closer):
            closer()

    def _camera(self, req: RouterRequest) -> CameraObservation:
        camera = req.camera
        if not isinstance(camera, CameraObservation):
            raise RouterError("vlm: camera observation required")
        if not isinstance(camera.image_bytes, bytes) or not camera.image_bytes:
            raise RouterError("vlm: encoded camera bytes required")
        if len(camera.image_bytes) > self.max_image_bytes:
            raise RouterError("vlm: image exceeds byte limit")
        signatures = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff"}
        if camera.mime_type not in signatures or not camera.image_bytes.startswith(
            signatures[camera.mime_type]
        ):
            raise RouterError("vlm: unsupported image type or mismatched signature")
        if (
            not isinstance(camera.width, int)
            or isinstance(camera.width, bool)
            or not isinstance(camera.height, int)
            or isinstance(camera.height, bool)
            or camera.width < 1
            or camera.height < 1
        ):
            raise RouterError("vlm: positive image dimensions required")
        if camera.lane not in ("LEFT", "RIGHT"):
            raise RouterError("vlm: camera ego lane must be LEFT or RIGHT")
        if not math.isfinite(camera.speed_mps) or camera.speed_mps < 0:
            raise RouterError("vlm: invalid ego speed")
        if not math.isfinite(req.t) or not math.isfinite(camera.captured_t):
            raise RouterError("vlm: invalid camera/request timestamp")
        age = req.t - camera.captured_t
        if age < -1e-6 or age > self.max_image_age_s + 1e-6:
            raise RouterError("vlm: camera observation is future-dated or stale")
        return camera

    def _payload(self, camera: CameraObservation) -> tuple[bytes, str]:
        encoded = base64.b64encode(camera.image_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{camera.mime_type};base64,{encoded}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Our car is in the {camera.lane} lane. "
                            f"Ego speed: {camera.speed_mps:.3f} m/s."
                        ),
                    },
                ],
            },
        ]
        prompt = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1,
                "seed": self.seed,
                "stream": False,
                "structured_outputs": {"choice": list(CHOICES)},
            },
            allow_nan=False,
        ).encode("utf-8")
        return body, hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _decode(self, result: TransportResponse) -> tuple[str, dict[str, int], str, str]:
        content_type = next(
            (str(v) for k, v in result.headers.items() if str(k).lower() == "content-type"), ""
        )
        if content_type and "json" not in content_type.lower():
            raise RouterError("vlm: response is not JSON content type")
        if not isinstance(result.body, bytes) or len(result.body) > self.max_response_bytes:
            raise RouterError("vlm: invalid or oversized response body")
        try:
            obj = json.loads(result.body.decode("utf-8"))
            choices = obj["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("expected one choice")
            choice = choices[0]
            message = choice["message"]
            text = message["content"]
            finish = choice["finish_reason"]
            if (
                not isinstance(text, str)
                or text.strip() not in CHOICES
                or finish not in ("stop", "length")
                or message.get("tool_calls")
                or message.get("refusal")
            ):
                raise ValueError("invalid constrained answer")
            usage = normalise_usage(obj.get("usage"))
            if "completion_tokens" in usage and usage["completion_tokens"] != 1:
                raise ValueError("completion was not one token")
            model = obj.get("model", self.model)
            if not isinstance(model, str) or not model.strip():
                raise ValueError("invalid response model")
            return text.strip(), usage, model, finish
        except (UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise RouterError("vlm: malformed or non-single-letter completion") from exc

    def submit(self, req: RouterRequest) -> RouterResponse:
        started = self._clock()
        try:
            if self.health().failed:
                raise RouterError("vlm: circuit open; reset required")
            camera = self._camera(req)
            budget = self.timeout_s
            if req.deadline_s is not None:
                if not math.isfinite(req.deadline_s) or req.deadline_s <= 0:
                    raise RouterError("vlm: invalid request deadline")
                budget = min(budget, req.deadline_s)
            image_digest = hashlib.sha256(camera.image_bytes).hexdigest()
            body, prompt_digest = self._payload(camera)
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            for attempt in range(self.max_retries + 1):
                remaining = budget - (self._clock() - started)
                if remaining <= 0:
                    raise RouterError("vlm: request deadline exhausted")
                retry = False
                try:
                    result = self._transport.request(
                        f"{self.base_url}/chat/completions",
                        method="POST",
                        body=body,
                        headers=headers,
                        timeout=remaining,
                    )
                except TransportError as exc:
                    if not exc.transient:
                        raise RouterError("vlm: permanent transport failure") from exc
                    retry = True
                except urllib.error.HTTPError as exc:
                    if exc.code not in (408, 429) and not 500 <= exc.code <= 599:
                        raise RouterError(f"vlm: HTTP {exc.code}") from exc
                    retry = True
                except (TimeoutError, urllib.error.URLError, ConnectionError):
                    retry = True
                if self._clock() - started >= budget:
                    raise RouterError("vlm: response exceeded request deadline")
                if not retry:
                    if not isinstance(result, TransportResponse):
                        raise RouterError("vlm: transport violated response contract")
                    status = result.status
                    if status in (408, 429) or 500 <= status <= 599:
                        retry = True
                    elif not 200 <= status < 300:
                        raise RouterError(f"vlm: HTTP {status}")
                    else:
                        letter, usage, model, finish = self._decode(result)
                        wall = self._clock() - started
                        if wall >= budget:
                            raise RouterError("vlm: decoding exceeded request deadline")
                        response = RouterResponse(
                            skill=CHOICES[letter],
                            ready_at=req.t + wall,
                            backend=self.name,
                            raw_text=letter,
                            latency_s=wall,
                            request_id=self._request_id(req),
                            submitted_t=req.t,
                            wall_latency_s=wall,
                            fallback_used=False,
                            prompt_hash=prompt_digest,
                            live=self._live,
                            model=model,
                            usage=usage,
                            finish_reason=finish,
                            image_hash=image_digest,
                            image_captured_t=camera.captured_t,
                            image_age_s=max(0.0, req.t - camera.captured_t),
                            request_attempts=attempt + 1,
                        )
                        validate_response(response, req)
                        self._record_success(wall, usage=usage)
                        response.health = self.health()
                        return response
                if attempt == self.max_retries:
                    raise RouterError("vlm: transient request retries exhausted")
                remaining = budget - (self._clock() - started)
                pause = self.backoff_s * (2.0 ** min(attempt, 30))
                if pause >= remaining:
                    raise RouterError("vlm: retry would exceed request deadline")
                if pause:
                    self._sleep(pause)
            raise RouterError("vlm: request failed")  # defensive, loop always exits above
        except RouterError as exc:
            self._record_failure(
                exc,
                latency_s=max(0.0, self._clock() - started),
                failure_threshold=self.max_consecutive_failures,
            )
            raise
