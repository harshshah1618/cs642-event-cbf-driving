"""Pluggable Tier-1 skill router interface.

Called ONLY on interrupt (critic sigma above threshold OR sustained CBF
residual).  Backends may be OpenAI-compatible, local transformers/ollama, or
scripted.

The contract is intentionally small: every backend returns a
:class:`RouterResponse` whose ``ready_at`` is expressed in simulation time.
Backends own their own retry/fallback policy, but the controller can rely on
:func:`validate_response` to reject malformed or time-travelling answers.
"""

from __future__ import annotations

import copy
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from driving.config import SKILL_NAMES
from driving.env_types import WorldSnapshot
from driving.errors import RouterError
from driving.io_utils import stable_hash

# Response times earlier than this are treated as clock noise rather than a
# contract violation (simulation clocks are often float accumulators).
_TIME_EPSILON = 1e-6

_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def normalise_usage(value: Any) -> Dict[str, int]:
    """Return strict, non-negative token counts from a provider usage object.

    Missing or malformed fields are ignored rather than turning a valid skill
    answer into a routing failure.  ``total_tokens`` is derived when a provider
    supplies only the prompt/completion counts.
    """

    if not isinstance(value, Mapping):
        return {}
    usage: Dict[str, int] = {}
    for key in _USAGE_KEYS:
        raw = value.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            number = int(raw)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            usage[key] = number
    if "total_tokens" not in usage and (
        "prompt_tokens" in usage or "completion_tokens" in usage
    ):
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
            "completion_tokens", 0
        )
    return usage


@dataclass(frozen=True)
class CameraObservation:
    """Detached camera evidence; no neighboring-object simulator state."""

    image_bytes: bytes
    mime_type: str
    captured_t: float
    lane: str
    speed_mps: float
    width: int
    height: int


@dataclass
class RouterRequest:
    t: float
    current_skill: str
    snapshot: WorldSnapshot
    sigma: float
    reason: str  # "sigma" | "shield" | "sigma+shield" | "fixed_rate"
    request_id: str = ""
    residual: Optional[float] = None
    shield_h: Optional[float] = None
    vlm_pending: bool = False
    deadline_s: Optional[float] = None
    camera: Optional[CameraObservation] = None
    camera_capture_wall_s: float = 0.0


@dataclass
class RouterHealth:
    """Snapshot of a backend's operational state.

    ``status`` is one of ``"healthy"``, ``"degraded"`` or ``"failed"``.
    The boolean properties make the common checks readable while keeping the
    serialized form compact.
    """

    status: str = "healthy"
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    last_error: Optional[str] = None
    last_latency_s: Optional[float] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.status not in ("healthy", "degraded", "failed"):
            raise ValueError(f"invalid router health status {self.status!r}")

    @property
    def healthy(self) -> bool:
        return self.status == "healthy"

    @property
    def degraded(self) -> bool:
        return self.status == "degraded"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "healthy": self.healthy,
            "degraded": self.degraded,
            "failed": self.failed,
            "consecutive_failures": self.consecutive_failures,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "last_error": self.last_error,
            "last_latency_s": self.last_latency_s,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class RouterResponse:
    # First three fields keep the historical positional signature.
    skill: str
    ready_at: float  # simulation time when answer becomes available
    backend: str
    raw_text: str = ""
    latency_s: float = 0.30
    # Observable request/response metadata.
    request_id: str = ""
    submitted_t: Optional[float] = None
    wall_latency_s: float = 0.0
    fallback_used: bool = False
    error: Optional[str] = None
    prompt_hash: str = ""
    health: Optional[RouterHealth] = None
    # Live-model telemetry.  ``live`` is true only when a real model (not a
    # scripted/dry-run fallback) produced the answer.  Canonical offline runs
    # therefore carry ``live=False`` and empty usage.
    live: bool = False
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    image_hash: str = ""
    image_captured_t: Optional[float] = None
    image_age_s: Optional[float] = None
    request_attempts: int = 0


def response_record_fields(resp: RouterResponse) -> Dict[str, Any]:
    """Flatten the observable parts of a response into a call record."""

    usage = normalise_usage(resp.usage)
    return {
        "backend": resp.backend,
        "ready_at": float(resp.ready_at),
        "latency_s": float(resp.latency_s),
        "fallback_used": bool(resp.fallback_used),
        "wall_latency_ms": float(resp.wall_latency_s * 1000.0),
        "prompt_hash": resp.prompt_hash,
        "live": bool(resp.live),
        "model": str(resp.model or ""),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "finish_reason": resp.finish_reason,
        "error": resp.error,
        "image_hash": resp.image_hash,
        "image_captured_t": resp.image_captured_t,
        "image_age_s": resp.image_age_s,
        "request_attempts": resp.request_attempts,
    }


def validate_response(resp: RouterResponse, request: RouterRequest) -> RouterResponse:
    """Check the router response contract or raise :class:`RouterError`.

    This is deliberately strict about *safety-relevant* fields and lenient
    about optional metadata so old callers that construct minimal responses
    keep working.
    """

    if not isinstance(resp, RouterResponse):
        raise RouterError(f"router returned {type(resp).__name__}, expected RouterResponse")
    if not isinstance(resp.backend, str) or not resp.backend.strip():
        raise RouterError("router response is missing a backend label")
    if resp.skill not in SKILL_NAMES:
        raise RouterError(f"router returned invalid skill {resp.skill!r}")

    try:
        ready_at = float(resp.ready_at)
    except (TypeError, ValueError) as exc:
        raise RouterError(f"router response has non-numeric ready_at {resp.ready_at!r}") from exc
    if not math.isfinite(ready_at):
        raise RouterError(f"router response has non-finite ready_at {resp.ready_at!r}")

    try:
        submitted_t = float(request.t if resp.submitted_t is None else resp.submitted_t)
    except (TypeError, ValueError) as exc:
        raise RouterError("router request has non-numeric t") from exc
    if not math.isfinite(submitted_t):
        raise RouterError("router request has non-finite t")
    if ready_at < submitted_t - _TIME_EPSILON:
        raise RouterError(
            f"router response ready_at={ready_at!r} precedes request t={submitted_t!r}"
        )

    for name in ("latency_s", "wall_latency_s"):
        value = getattr(resp, name)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RouterError(f"router response has non-numeric {name} {value!r}") from exc
        if not math.isfinite(number) or number < 0.0:
            raise RouterError(f"router response has invalid {name} {value!r}")

    return resp


class RouterBackend(ABC):
    name: str = "base"
    latency_s: float = 0.30

    #: True only for backends that can call a real external/local model.
    is_live_capable: bool = False

    def __init__(self) -> None:
        self._health = RouterHealth()
        self._health_lock = threading.Lock()

    # -- health plumbing -------------------------------------------------
    def _health_state(self) -> RouterHealth:
        state = getattr(self, "_health", None)
        if state is None:  # Subclasses written before this base existed.
            state = RouterHealth()
            self._health = state
        lock = getattr(self, "_health_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._health_lock = lock
        return state

    def _lock_health(self) -> threading.Lock:
        self._health_state()
        return self._health_lock  # type: ignore[return-value]

    @staticmethod
    def _short_error(error: object, limit: int = 160) -> str:
        text = " ".join(str(error).split())
        return text[:limit] if len(text) > limit else text

    def _record_success(
        self, latency_s: float, usage: Optional[Mapping[str, Any]] = None
    ) -> None:
        with self._lock_health():
            state = self._health_state()
            state.total_calls += 1
            state.consecutive_failures = 0
            state.status = "healthy"
            state.last_error = None
            state.last_latency_s = float(latency_s)
            counts = normalise_usage(usage)
            state.total_prompt_tokens += int(counts.get("prompt_tokens", 0))
            state.total_completion_tokens += int(counts.get("completion_tokens", 0))
            state.total_tokens += int(counts.get("total_tokens", 0))

    def _record_failure(
        self,
        error: object,
        *,
        latency_s: Optional[float] = None,
        failure_threshold: Optional[int] = None,
    ) -> None:
        with self._lock_health():
            state = self._health_state()
            state.total_calls += 1
            state.total_failures += 1
            state.consecutive_failures += 1
            state.last_error = self._short_error(error)
            if latency_s is not None:
                state.last_latency_s = float(latency_s)
            if failure_threshold is not None and state.consecutive_failures >= int(
                failure_threshold
            ):
                state.status = "failed"
            else:
                state.status = "degraded"

    def _request_id(self, req: RouterRequest) -> str:
        if req.request_id:
            return req.request_id
        return stable_hash(
            {
                "t": req.t,
                "skill": req.current_skill,
                "reason": req.reason,
                "sigma": req.sigma,
            }
        )[:16]

    # -- public interface ------------------------------------------------
    @abstractmethod
    def submit(self, req: RouterRequest) -> RouterResponse:
        """Start a skill query. May return immediately with ready_at = t + latency."""

    def describe(self) -> str:
        return self.name

    def health(self) -> RouterHealth:
        """Return a copy so callers cannot mutate backend state."""
        with self._lock_health():
            return copy.copy(self._health_state())

    def reset(self) -> None:
        """Reset health/circuit-breaker state (does not reload models)."""
        with self._lock_health():
            self._health = RouterHealth()

    def close(self) -> None:
        """Release backend resources. Safe to call more than once."""
        # Most backends hold no external resources.
        return None


# Backwards-compatible alias used by some older imports.
RouterHealthSnapshot = RouterHealth

__all__ = [
    "CameraObservation",
    "SKILL_NAMES",
    "RouterBackend",
    "RouterHealth",
    "RouterRequest",
    "RouterResponse",
    "normalise_usage",
    "response_record_fields",
    "validate_response",
]
