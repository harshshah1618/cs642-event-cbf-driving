"""Router connectivity and sample-request checks.

``event-cbf router-check`` is deliberately separate from evaluation: it sends
one synthetic interrupt and reports whether a real model answered, which
fallback/error was used, the provider-reported token usage, and current health.
It never writes run artifacts and never affects canonical results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from driving.config import RouterConfig
from driving.env_types import VehicleState, WorldSnapshot
from driving.errors import ConfigError, RouterError
from driving.router_base import (
    RouterBackend,
    RouterRequest,
    normalise_usage,
    response_record_fields,
    validate_response,
)
from driving.router_factory import create_router


def sample_request(
    t: float = 0.0, *, request_id: str = "router-check"
) -> RouterRequest:
    """A deterministic, safety-relevant interrupt for a connectivity probe."""

    snapshot = WorldSnapshot(
        t=float(t),
        ego=VehicleState(x=60.0, y=0.0, vx=20.0, vy=0.0),
        others=[
            VehicleState(x=75.0, y=0.0, vx=0.0, vy=0.0, speed_limit=0.01),
            VehicleState(x=95.0, y=4.0, vx=24.0, vy=0.0),
        ],
        route_length=200.0,
        info={"scenario": "router-check"},
    )
    return RouterRequest(
        t=float(t),
        current_skill="CRUISE_RIGHT",
        snapshot=snapshot,
        sigma=4.0,
        reason="sigma",
        request_id=request_id,
    )


@dataclass
class RouterCheckResult:
    ok: bool
    live: bool
    configured_backend: str
    name: str
    describe: str
    live_capable: bool = False
    model: str = ""
    skill: Optional[str] = None
    fallback_used: bool = False
    error: Optional[str] = None
    wall_latency_s: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    health: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "live": bool(self.live),
            "configured_backend": self.configured_backend,
            "name": self.name,
            "describe": self.describe,
            "live_capable": bool(self.live_capable),
            "model": self.model,
            "skill": self.skill,
            "fallback_used": bool(self.fallback_used),
            "error": self.error,
            "wall_latency_s": float(self.wall_latency_s),
            "usage": dict(self.usage),
            "finish_reason": self.finish_reason,
            "health": dict(self.health),
        }


def _health_dict(router: Any) -> Dict[str, Any]:
    health = getattr(router, "health", None)
    if not callable(health):
        return {}
    try:
        state = health()
        return state.to_dict() if hasattr(state, "to_dict") else dict(state)
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"{type(exc).__name__}: {exc}"}


def check_router(
    config: RouterConfig,
    *,
    allow_live: Optional[bool] = None,
    require_live: bool = False,
    router: Optional[RouterBackend] = None,
    request: Optional[RouterRequest] = None,
) -> RouterCheckResult:
    """Build (or accept) a router, send one sample, and report observables.

    ``require_live`` only changes the pass/fail decision: a valid scripted
    fallback is still reported with its reason.  Configuration errors (for
    example a live backend without the live opt-in) propagate as
    :class:`ConfigError`; runtime failures become an ``ok=False`` result.
    """

    if not isinstance(config, RouterConfig):
        raise ConfigError(
            f"check_router expected RouterConfig, got {type(config).__name__}"
        )
    config.validate("router")
    effective_allow = bool(config.allow_live if allow_live is None else allow_live)
    configured_backend = config.backend
    created = router is None

    if router is None:
        try:
            router = create_router(config, allow_live=effective_allow)
        except ConfigError:
            raise
        except Exception as exc:
            return RouterCheckResult(
                ok=False,
                live=False,
                configured_backend=configured_backend,
                name="unavailable",
                describe="",
                error=f"{type(exc).__name__}: {exc}",
            )

    name = getattr(router, "name", type(router).__name__)
    describe = ""
    try:
        describer = getattr(router, "describe", None)
        describe = describer() if callable(describer) else str(describer or name)
    except Exception as exc:  # pragma: no cover - defensive
        describe = f"{name} (describe failed: {exc})"
    live_capable = bool(getattr(router, "is_live_capable", False))

    req = request if request is not None else sample_request()
    started = time.perf_counter()
    response = None
    error: Optional[str] = None
    try:
        response = router.submit(req)
        validate_response(response, req)
    except RouterError as exc:
        error = str(exc)
    except Exception as exc:  # pragma: no cover - unexpected backend bug
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    health = _health_dict(router)

    try:
        if created:
            closer = getattr(router, "close", None)
            if callable(closer):
                closer()
    except Exception:  # pragma: no cover - close is best-effort
        pass

    if response is None:
        return RouterCheckResult(
            ok=False,
            live=False,
            configured_backend=configured_backend,
            name=str(name),
            describe=describe,
            live_capable=live_capable,
            error=error or "router produced no response",
            wall_latency_s=elapsed,
            health=health,
        )

    live = bool(response.live and not response.fallback_used)
    if live_capable:
        ok = live
    else:
        ok = response.error is None
    if require_live:
        ok = ok and live
        if not live and error is None:
            error = "no live model answer (offline backend, fallback, or live disabled)"
    if error is not None:
        ok = False

    fields = response_record_fields(response)
    return RouterCheckResult(
        ok=bool(ok),
        live=live,
        configured_backend=configured_backend,
        name=str(name),
        describe=describe,
        live_capable=live_capable,
        model=str(fields.get("model") or ""),
        skill=str(response.skill),
        fallback_used=bool(fields.get("fallback_used")),
        error=error or response.error,
        wall_latency_s=float(fields.get("wall_latency_ms") or 0.0) / 1000.0,
        usage=normalise_usage(response.usage),
        finish_reason=response.finish_reason,
        health=health,
    )


__all__ = ["RouterCheckResult", "check_router", "sample_request"]
