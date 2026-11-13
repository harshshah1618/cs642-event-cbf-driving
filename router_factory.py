"""Router backend selection with explicit, testable policy.

Modern API::

    create_router(RouterConfig(backend="scripted"))

Legacy API (kept working)::

    make_router("auto", latency_s=0.30)

``auto`` is conservative: it uses the live OpenAI-compatible backend only when
an API key is present **and** the caller explicitly opts in with
``allow_live=True``.  Otherwise it uses the offline dry-run backend.
``prefer="openai"`` explicitly requests the live backend (which still has its
own scripted fallback when no key is configured).
"""

from __future__ import annotations

import copy
import os
from typing import Optional

from config import RouterBackend as RouterBackendKind
from config import RouterConfig
from errors import ConfigError
from planner import ScriptedRouter
from router_base import RouterBackend
from router_dry_run import DryRunOpenAIRouter
from router_http import OpenAICompatRouter
from router_local import LocalLLMRouter
from router_opencode import OpenCodeRouter

_VALID_BACKENDS = tuple(kind.value for kind in RouterBackendKind)


def _normalise_backend(value: object) -> str:
    if isinstance(value, RouterBackendKind):
        text = value.value
    else:
        text = str(value or "auto").strip().lower()
    if text not in _VALID_BACKENDS:
        raise ConfigError(
            f"unknown router preference {text!r}; choose from {list(_VALID_BACKENDS)}"
        )
    return text


def _api_key(config: RouterConfig) -> Optional[str]:
    return os.environ.get(config.api_key_env) or os.environ.get(
        config.api_key_env_fallback
    )


def create_router(config: RouterConfig, **overrides) -> RouterBackend:
    """Build a router from a :class:`RouterConfig`.

    Non-config keyword arguments (for example ``transport`` or ``load_model``)
    are passed through to the concrete backend constructor.  Unknown backend
    names raise :class:`ConfigError`; nothing is silently substituted.
    """

    if not isinstance(config, RouterConfig):
        raise ConfigError(
            f"create_router expected RouterConfig, got {type(config).__name__}"
        )
    cfg = copy.deepcopy(config)
    allow_live = bool(overrides.pop("allow_live", False)) or bool(cfg.allow_live)
    cfg.allow_live = allow_live

    constructor_overrides = {}
    for key, value in list(overrides.items()):
        if key == "backend":
            cfg.backend = _normalise_backend(value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            constructor_overrides[key] = value

    cfg.backend = _normalise_backend(cfg.backend)
    cfg.validate("router")
    backend = cfg.backend

    if backend == "auto":
        explicit_key = bool(constructor_overrides.get("api_key"))
        backend = "openai" if (allow_live and (_api_key(cfg) or explicit_key)) else "dryrun"

    # A live backend must be requested explicitly.  ``local`` with model
    # loading disabled is a safe offline fallback object and remains available
    # for diagnostics/tests without the opt-in.
    offline_local = (
        backend == "local"
        and (constructor_overrides.get("load_model") is False or cfg.local_backend == "none")
    )
    if backend in ("openai", "local", "opencode", "vlm") and not allow_live and not offline_local:
        raise ConfigError(
            f"router.backend={backend!r} is a live backend but live calls are disabled. "
            "Set router.allow_live=true in the profile (or pass --allow-live) to opt in, "
            "or use backend='dryrun' for offline evaluation."
        )

    if backend == "vlm":
        from camera import VLMRouter

        return VLMRouter(config=cfg, **constructor_overrides)
    if backend == "scripted":
        return ScriptedRouter(latency_s=cfg.latency_s, latency_jitter_s=cfg.latency_jitter_s,
                              seed=cfg.seed, latency_schedule_s=cfg.latency_schedule_s, **constructor_overrides)
    if backend == "dryrun":
        return DryRunOpenAIRouter(config=cfg, **constructor_overrides)
    if backend == "openai":
        return OpenAICompatRouter(config=cfg, **constructor_overrides)
    if backend == "opencode":
        return OpenCodeRouter(config=cfg, **constructor_overrides)
    if backend == "local":
        return LocalLLMRouter(
            config=cfg, backend=cfg.local_backend, **constructor_overrides
        )
    raise ConfigError(f"unsupported router backend {backend!r}")  # pragma: no cover


def make_router(
    prefer: str = "auto",
    latency_s: float = 0.30,
    *,
    allow_live: bool = False,
    config: Optional[RouterConfig] = None,
    **overrides,
) -> RouterBackend:
    """Backward-compatible router factory.

    Preferences:
      - ``scripted``: lane-threat rule (M1 baseline).
      - ``dryrun``: forced offline OpenAI-compatible dry run.
      - ``openai``: explicitly live OpenAI-compatible backend. If no key is
        present the backend uses its visible scripted fallback.
      - ``opencode``: explicitly live local OpenCode server backend.
      - ``local``: try transformers, then Ollama, then the dry-run backend.
      - ``auto``: dry-run unless ``allow_live=True`` and an API key is set.
        This conservative default prevents accidental paid/live calls.
    """

    backend = _normalise_backend(prefer)
    cfg = copy.deepcopy(config) if config is not None else RouterConfig()
    cfg.backend = backend
    if latency_s is not None:
        cfg.latency_s = latency_s

    constructor_overrides = {}
    for key, value in list(overrides.items()):
        if key != "backend" and hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            constructor_overrides[key] = value

    # Legacy local behavior: prefer real transformers, then Ollama, then the
    # offline dry-run contract.  ``create_router(backend='local')`` keeps an
    # explicit LocalLLMRouter so its fallback accounting is observable.
    if backend == "local":
        load_model = constructor_overrides.pop("load_model", True)
        if not load_model:
            return LocalLLMRouter(
                config=cfg,
                backend="transformers",
                load_model=False,
                **constructor_overrides,
            )
        local = LocalLLMRouter(
            config=cfg,
            backend="transformers",
            load_model=True,
            **constructor_overrides,
        )
        if local.mode != "scripted":
            return local
        local = LocalLLMRouter(
            config=cfg,
            backend="ollama",
            load_model=True,
            **constructor_overrides,
        )
        if local.mode != "scripted":
            return local
        return DryRunOpenAIRouter(config=cfg)

    # This legacy function treats an explicit ``prefer="openai"`` as the
    # opt-in itself.  ``auto`` still requires ``allow_live=True``.
    return create_router(
        cfg,
        allow_live=allow_live or backend in ("openai", "local", "opencode"),
        **constructor_overrides,
    )


__all__ = ["create_router", "make_router"]
