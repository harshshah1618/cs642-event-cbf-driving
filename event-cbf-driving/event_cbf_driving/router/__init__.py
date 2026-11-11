"""Stable router interface and backend implementations."""

from .base import (
    SKILL_NAMES,
    RouterBackend,
    RouterHealth,
    RouterRequest,
    RouterResponse,
    normalise_usage,
    response_record_fields,
    validate_response,
)
from .check import RouterCheckResult, check_router, sample_request
from .dry_run import DryRunOpenAIRouter, DryRunRouter
from .factory import create_router, make_router
from .local_llm import LocalLLMRouter
from .openai_compat import (
    OpenAICompatRouter,
    Transport,
    TransportError,
    TransportResponse,
    UrllibTransport,
)
from .prompt import (
    PROMPT,
    SYSTEM_PROMPT,
    build_chat_prompt,
    build_messages,
    build_prompt,
    build_user_prompt,
    parse_skill,
    prompt_hash,
)
from .scripted import ScriptedRouter

__all__ = [
    "SKILL_NAMES",
    "DryRunOpenAIRouter",
    "DryRunRouter",
    "LocalLLMRouter",
    "OpenAICompatRouter",
    "PROMPT",
    "RouterBackend",
    "RouterCheckResult",
    "RouterHealth",
    "RouterRequest",
    "RouterResponse",
    "SYSTEM_PROMPT",
    "ScriptedRouter",
    "Transport",
    "TransportError",
    "TransportResponse",
    "UrllibTransport",
    "build_chat_prompt",
    "build_messages",
    "build_prompt",
    "build_user_prompt",
    "check_router",
    "create_router",
    "make_router",
    "normalise_usage",
    "parse_skill",
    "prompt_hash",
    "response_record_fields",
    "sample_request",
    "validate_response",
]
