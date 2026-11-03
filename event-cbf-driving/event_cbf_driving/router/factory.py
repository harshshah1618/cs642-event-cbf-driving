"""Pick the best available router backend."""

from __future__ import annotations

import os

from event_cbf_driving.router.base import RouterBackend
from event_cbf_driving.router.dry_run import DryRunOpenAIRouter
from event_cbf_driving.router.local_llm import LocalLLMRouter
from event_cbf_driving.router.openai_compat import OpenAICompatRouter
from event_cbf_driving.router.scripted import ScriptedRouter


def make_router(prefer: str = "auto", latency_s: float = 0.30) -> RouterBackend:
    """
    prefer:
      - scripted: lane-threat rule (M1 baseline)
      - openai: live OpenAI-compat if OPENAI_API_KEY / EVENT_CBF_API_KEY set;
                otherwise DryRunOpenAIRouter (prompt→mock HTTP→parse)
      - local: real transformers (EVENT_CBF_LOCAL_MODEL) or ollama; else dry-run
      - dryrun: forced OpenAI-compat dry-run
      - auto: API key → openai; else dry-run (stable closed-loop). Use --router local
              explicitly to force on-box transformers.
    """
    prefer = (prefer or "auto").lower()
    if prefer == "scripted":
        return ScriptedRouter(latency_s=latency_s)
    if prefer == "dryrun":
        return DryRunOpenAIRouter(latency_s=latency_s)
    if prefer == "openai":
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("EVENT_CBF_API_KEY"):
            return OpenAICompatRouter(latency_s=latency_s)
        return DryRunOpenAIRouter(latency_s=latency_s)
    if prefer == "local":
        local = LocalLLMRouter(backend="transformers", latency_s=latency_s)
        if getattr(local, "_mode", "scripted") != "scripted":
            return local
        local = LocalLLMRouter(backend="ollama", latency_s=latency_s)
        if getattr(local, "_mode", "scripted") != "scripted":
            return local
        return DryRunOpenAIRouter(latency_s=latency_s)
    # auto — prefer stable LLM contract without a key
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("EVENT_CBF_API_KEY"):
        return OpenAICompatRouter(latency_s=latency_s)
    return DryRunOpenAIRouter(latency_s=latency_s)
