"""Local transformers / Ollama Tier-1 router.

Default path (no API key): load a tiny instruct model via transformers and
generate a single skill token, preferably on CPU.

Environment overrides:
  EVENT_CBF_LOCAL_MODEL   - HF model id (default from RouterConfig)
  EVENT_CBF_LOCAL_DEVICE  - cpu | cuda | mps | auto
  EVENT_CBF_OLLAMA_URL    - Ollama base (default http://127.0.0.1:11434)
  EVENT_CBF_OLLAMA_MODEL  - Ollama tag (default from RouterConfig)

Security/model-loading notes: ``trust_remote_code`` is off by default and
``local_revision`` is passed through when configured.  This backend never
silently replaces a live model with the scripted rule; every fallback carries
``fallback_used=True`` and a short ``error`` reason.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Set

from driving.config import RouterConfig
from driving.errors import RouterError
from driving.logging_utils import get_logger
from driving.planner import ScriptedRouter
from driving.router_base import (
    SKILL_NAMES,
    RouterBackend,
    RouterHealth,
    RouterRequest,
    RouterResponse,
    normalise_usage,
    validate_response,
)
from driving.router_prompt import build_messages, build_prompt, parse_skill, prompt_hash

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

logger = get_logger(__name__)


class _LocalBackendError(Exception):
    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = bool(transient)


class LocalLLMRouter(RouterBackend):
    name = "local_llm"
    is_live_capable = True

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        *,
        backend: str = "auto",  # auto | ollama | transformers | none
        model: Optional[str] = None,
        hf_model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        latency_s: Optional[float] = None,
        load_model: bool = True,
        probe_timeout_s: float = 1.0,
    ):
        super().__init__()
        cfg = copy.deepcopy(config) if config is not None else RouterConfig()

        mode = (backend or "auto").lower()
        if mode not in ("auto", "ollama", "transformers", "none"):
            raise RouterError(f"unknown local backend mode {backend!r}")
        self.backend = mode
        self.config = cfg
        self.local_dtype = cfg.local_dtype
        self.allow_live = bool(cfg.allow_live)

        self.latency_s = float(cfg.latency_s if latency_s is None else latency_s)
        self.timeout_s = float(cfg.timeout_s)
        self.max_retries = int(cfg.max_retries)
        self.backoff_s = float(cfg.backoff_s)
        self.allow_fallback = bool(cfg.allow_fallback)
        self.fail_on_fallback = bool(cfg.fail_on_fallback)
        self.max_consecutive_failures = int(cfg.max_consecutive_failures)

        self.hf_model = (
            hf_model
            or os.environ.get("EVENT_CBF_LOCAL_MODEL")
            or cfg.local_model
            or DEFAULT_HF_MODEL
        )
        self.model = (
            model
            or os.environ.get("EVENT_CBF_OLLAMA_MODEL")
            or cfg.ollama_model
        )
        self.ollama_url = (
            ollama_url
            or os.environ.get("EVENT_CBF_OLLAMA_URL")
            or cfg.ollama_url
        ).rstrip("/")
        self.local_revision = cfg.local_revision
        self.trust_remote_code = bool(cfg.trust_remote_code)
        self.local_device = (
            os.environ.get("EVENT_CBF_LOCAL_DEVICE") or cfg.local_device or "cpu"
        )
        self.probe_timeout_s = float(probe_timeout_s)

        if not math.isfinite(self.latency_s) or self.latency_s < 0.0:
            raise RouterError(f"invalid local latency_s {self.latency_s!r}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise RouterError(f"invalid local timeout_s {self.timeout_s!r}")

        self._fallback = ScriptedRouter(latency_s=self.latency_s)
        self._mode = "scripted"
        self._tokenizer = None
        self._model: Any = None
        self._allowed_token_ids: Optional[Set[int]] = None
        self._load_error = ""
        self._device = "cpu"
        self._closed = False
        self._last_usage: Dict[str, int] = {}
        self._last_model = ""
        self._last_finish_reason: Optional[str] = None
        self._generation_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()

        if load_model:
            self._resolve(mode)
        else:
            self._load_error = "load_model_disabled"

    # -- introspection ---------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def load_error(self) -> str:
        return self._load_error

    def describe(self) -> str:
        if self._mode == "transformers":
            return (
                f"local_transformers({self.hf_model}@{self.local_revision or 'default'}, "
                f"device={self._device})"
            )
        if self._mode == "ollama":
            return f"local_ollama({self.model}@{self.ollama_url})"
        detail = self._short_error(self._load_error) if self._load_error else "local_unavailable"
        return f"local_llm(scripted_fallback:{detail})"

    # -- resolution / loading -------------------------------------------
    def _resolve(self, backend: str) -> None:
        if backend in ("auto", "ollama"):
            if self._probe_ollama():
                self._mode = "ollama"
                return
        if backend in ("auto", "transformers"):
            if self._load_transformers():
                self._mode = "transformers"
                return
        self._mode = "scripted"

    def _probe_ollama(self) -> bool:
        url = self.ollama_url + "/api/tags"
        timeout = min(self.probe_timeout_s, self.timeout_s)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read(2_000_000)
        except urllib.error.HTTPError as exc:
            self._load_error = f"ollama_probe_http_{exc.code}"
            return False
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                self._load_error = "ollama_probe_timeout"
            else:
                self._load_error = f"ollama_probe_unreachable:{self._short_error(reason)}"
            return False
        except TimeoutError as exc:
            self._load_error = f"ollama_probe_timeout:{self._short_error(exc)}"
            return False
        except OSError as exc:
            self._load_error = f"ollama_probe_os_error:{self._short_error(exc)}"
            return False

        if status != 200:
            self._load_error = f"ollama_probe_http_{status}"
            return False
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._load_error = f"ollama_probe_bad_json:{self._short_error(exc)}"
            return False
        if not isinstance(payload, dict):
            self._load_error = "ollama_probe_bad_shape"
            return False
        self._load_error = ""
        return True

    def _resolve_device(self, torch) -> str:
        requested = (self.local_device or "cpu").lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("requested CUDA is unavailable; falling back to CPU")
            return "cpu"
        if requested == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                logger.warning("requested MPS is unavailable; falling back to CPU")
                return "cpu"
        return requested

    def _load_transformers(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except (ImportError, OSError) as exc:
            self._load_error = f"import_error:{self._short_error(exc)}"
            return False

        load_kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        dtype = getattr(torch, getattr(self, "local_dtype", "float32"))
        if self.local_revision:
            load_kwargs["revision"] = self.local_revision

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model, **load_kwargs)
            model_factory: Any = AutoModelForCausalLM
            try:
                model = model_factory.from_pretrained(
                    self.hf_model, torch_dtype=dtype, **load_kwargs
                )
            except TypeError as exc:
                # transformers renamed torch_dtype -> dtype in newer releases.
                message = str(exc)
                if "torch_dtype" not in message and "dtype" not in message:
                    raise
                model = model_factory.from_pretrained(
                    self.hf_model, dtype=dtype, **load_kwargs
                )
            device = self._resolve_device(torch)
            model.to(device)
            model.eval()

            allowed: Set[int] = set()
            for skill in SKILL_NAMES:
                ids = tokenizer.encode(skill, add_special_tokens=False)
                if ids:
                    allowed.add(ids[0])
                ids_sp = tokenizer.encode(" " + skill, add_special_tokens=False)
                if ids_sp:
                    allowed.add(ids_sp[0])

            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._allowed_token_ids = allowed or None
            self._load_error = ""
            return True
        except Exception as exc:  # model loading is an external failure surface
            self._load_error = f"load_error:{type(exc).__name__}:{self._short_error(exc)}"
            self._tokenizer = None
            self._model = None
            return False

    # -- generation ------------------------------------------------------
    def _prompt_ids(self, messages: list[dict], prompt: str):
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise _LocalBackendError("tokenizer_unloaded")
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
                return ids.to(self._device) if hasattr(ids, "to") else ids
            except (TypeError, ValueError, AttributeError, RuntimeError):
                # Tokenizer has no usable chat template; use the flat prompt.
                pass
        encoded = tokenizer(prompt, return_tensors="pt")
        return encoded.input_ids.to(self._device)

    def _generate_transformers(self, messages: list[dict], prompt: str) -> str:
        import torch

        with self._generation_lock:
            if self._model is None or self._tokenizer is None:
                raise _LocalBackendError("model_unloaded")
            if self._closed:
                raise _LocalBackendError("closed")
            try:
                input_ids = self._prompt_ids(messages, prompt)
                allowed = self._allowed_token_ids
                vocab = list(range(len(self._tokenizer)))
                prompt_len = int(input_ids.shape[-1])

                def prefix_allowed_tokens_fn(batch_id, sentence):
                    # Constrain only the first generated token to a skill start.
                    if int(sentence.shape[-1]) == prompt_len:
                        return list(allowed) if allowed else vocab
                    return vocab

                generation_kwargs = dict(
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(input_ids),
                )
                if allowed:
                    generation_kwargs["prefix_allowed_tokens_fn"] = (
                        prefix_allowed_tokens_fn
                    )
                with torch.inference_mode():
                    output = self._model.generate(input_ids, **generation_kwargs)
                new_tokens = output[0, input_ids.shape[-1] :]
                generated = int(new_tokens.shape[-1]) if hasattr(new_tokens, "shape") else 0
                self._last_usage = normalise_usage(
                    {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": generated,
                        "total_tokens": prompt_len + generated,
                    }
                )
                self._last_model = self.hf_model
                self._last_finish_reason = "stop"
                return self._tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()
            except _LocalBackendError:
                raise
            except (RuntimeError, OSError) as exc:
                raise _LocalBackendError(
                    f"transformers_generate:{type(exc).__name__}:{self._short_error(exc)}"
                ) from exc

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.backoff_s > 0.0:
            time.sleep(self.backoff_s * (2.0**attempt))

    def _generate_ollama(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 8},
            }
        ).encode("utf-8")
        url = self.ollama_url + "/api/generate"
        attempts = self.max_retries + 1
        last_error = "ollama_error"

        for attempt in range(attempts):
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    status = int(getattr(response, "status", 200))
                    raw = response.read(2_000_000)
                if status != 200:
                    raise _LocalBackendError(
                        f"ollama_http_{status}",
                        transient=status in (408, 429) or status >= 500,
                    )
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise _LocalBackendError(
                        f"ollama_bad_json:{self._short_error(exc)}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise _LocalBackendError("ollama_bad_shape")
                text = payload.get("response")
                if not isinstance(text, str):
                    raise _LocalBackendError("ollama_missing_response")
                self._last_usage = normalise_usage(
                    {
                        "prompt_tokens": payload.get("prompt_eval_count"),
                        "completion_tokens": payload.get("eval_count"),
                    }
                )
                provider_model = payload.get("model")
                self._last_model = (
                    provider_model
                    if isinstance(provider_model, str) and provider_model.strip()
                    else self.model
                )
                done_reason = payload.get("done_reason")
                self._last_finish_reason = (
                    done_reason if isinstance(done_reason, str) and done_reason else None
                )
                return text
            except urllib.error.HTTPError as exc:
                transient = exc.code in (408, 429) or exc.code >= 500
                last_error = f"ollama_http_{exc.code}"
                if transient and attempt + 1 < attempts:
                    self._sleep_before_retry(attempt)
                    continue
                raise _LocalBackendError(last_error, transient=transient) from exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                last_error = f"ollama_unreachable:{self._short_error(reason)}"
            except TimeoutError as exc:
                last_error = f"ollama_timeout:{self._short_error(exc)}"
            except OSError as exc:
                last_error = f"ollama_os_error:{self._short_error(exc)}"
            except _LocalBackendError as exc:
                last_error = str(exc)
                if exc.transient and attempt + 1 < attempts:
                    self._sleep_before_retry(attempt)
                    continue
                raise
            if attempt + 1 < attempts:
                self._sleep_before_retry(attempt)
                continue
            raise _LocalBackendError(last_error, transient=True)
        raise _LocalBackendError(last_error, transient=True)

    # -- fallback --------------------------------------------------------
    def _fallback_allowed(self) -> bool:
        return bool(self.allow_fallback and not self.fail_on_fallback)

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
            "router fallback %s(%s) -> scripted (%s)",
            self.name,
            self._mode,
            self._short_error(reason),
            extra={"event": "router_fallback", "backend": self.name, "reason": response.error},
        )
        return response

    # -- public API ------------------------------------------------------
    def submit(self, req: RouterRequest) -> RouterResponse:
        started = time.perf_counter()
        messages = build_messages(
            req.current_skill, req.snapshot, req.reason, request=req
        )
        prompt = build_prompt(
            req.current_skill, req.snapshot, req.reason, request=req
        )
        digest = prompt_hash(prompt)

        if self._closed:
            return self._fallback_response(
                req, "closed", prompt_digest=digest, started=started
            )
        if self._mode == "scripted":
            return self._fallback_response(
                req,
                self._load_error or "local_unavailable",
                prompt_digest=digest,
                started=started,
            )

        try:
            if self._mode == "ollama":
                text = self._generate_ollama(prompt)
            else:
                text = self._generate_transformers(messages, prompt)
        except _LocalBackendError as exc:
            return self._fallback_response(
                req,
                f"local_{self._mode}:{exc}",
                prompt_digest=digest,
                started=started,
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
            backend=f"local_{self._mode}",
            raw_text=text[:500],
            latency_s=self.latency_s,
            request_id=self._request_id(req),
            submitted_t=req.t,
            wall_latency_s=wall,
            fallback_used=False,
            error=None,
            prompt_hash=digest,
            live=True,
            model=self._last_model or (self.hf_model if self._mode == "transformers" else self.model),
            usage=dict(self._last_usage),
            finish_reason=self._last_finish_reason,
        )
        validate_response(response, req)
        self._record_success(wall, usage=self._last_usage)
        response.health = self.health()
        return response

    def health(self) -> RouterHealth:
        return super().health()

    def reset(self) -> None:
        super().reset()

    def close(self) -> None:
        with self._lifecycle_lock, self._generation_lock:
            self._closed = True
            self._tokenizer = None
            self._model = None
            self._allowed_token_ids = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, OSError, RuntimeError):
            pass


__all__ = ["DEFAULT_HF_MODEL", "LocalLLMRouter"]
