"""Local transformers / Ollama Tier-1 router.

Default path (no API key): download a tiny instruct model via transformers
and generate a single skill token on CPU.

Env vars:
  EVENT_CBF_LOCAL_MODEL — HF model id (default Qwen/Qwen2.5-0.5B-Instruct)
  EVENT_CBF_OLLAMA_URL  — Ollama base (default http://127.0.0.1:11434)
  EVENT_CBF_OLLAMA_MODEL — Ollama tag (default llama3.2:1b)
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional, Set

from event_cbf_driving.router.base import (
    SKILL_NAMES,
    RouterBackend,
    RouterRequest,
    RouterResponse,
)
from event_cbf_driving.router.prompt import build_prompt, parse_skill
from event_cbf_driving.router.scripted import ScriptedRouter

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class LocalLLMRouter(RouterBackend):
    name = "local_llm"

    def __init__(
        self,
        backend: str = "auto",  # auto | ollama | transformers | none
        model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        latency_s: float = 0.30,
        load_model: bool = True,
    ):
        self.backend = backend
        self.model = model or os.environ.get("EVENT_CBF_OLLAMA_MODEL") or "llama3.2:1b"
        self.hf_model = os.environ.get("EVENT_CBF_LOCAL_MODEL") or DEFAULT_HF_MODEL
        self.ollama_url = (
            ollama_url or os.environ.get("EVENT_CBF_OLLAMA_URL") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.latency_s = float(latency_s)
        self._fallback = ScriptedRouter(latency_s=latency_s)
        self._mode = "scripted"
        self._tokenizer = None
        self._model = None
        self._allowed_token_ids: Optional[Set[int]] = None
        self._load_error = ""
        if load_model:
            self._resolve(backend)

    def describe(self) -> str:
        if self._mode == "transformers":
            return f"local_transformers({self.hf_model})"
        if self._mode == "ollama":
            return f"local_ollama({self.model})"
        return "local_llm(scripted_fallback)"

    def _resolve(self, backend: str):
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
        try:
            with urllib.request.urlopen(self.ollama_url + "/api/tags", timeout=0.5) as r:
                return r.status == 200
        except Exception:
            return False

    def _load_transformers(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            self._load_error = f"import_error:{e}"
            return False
        try:
            tok = AutoTokenizer.from_pretrained(self.hf_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model,
                dtype=torch.float32,
                trust_remote_code=True,
            )
            model.eval()
            self._tokenizer = tok
            self._model = model
            # First-token allowlist for constrained decoding.
            allowed: Set[int] = set()
            for skill in SKILL_NAMES:
                ids = tok.encode(skill, add_special_tokens=False)
                if ids:
                    allowed.add(ids[0])
                ids_sp = tok.encode(" " + skill, add_special_tokens=False)
                if ids_sp:
                    allowed.add(ids_sp[0])
            self._allowed_token_ids = allowed or None
            return True
        except Exception as e:
            self._load_error = str(e)
            self._tokenizer = None
            self._model = None
            return False

    def _prompt_ids(self, prompt: str):
        assert self._tokenizer is not None
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                enc = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if hasattr(enc, "input_ids"):
                    return enc.input_ids
                return enc
            except Exception:
                pass
        return self._tokenizer(prompt, return_tensors="pt").input_ids

    def _generate_transformers(self, prompt: str) -> str:
        import torch

        assert self._tokenizer is not None and self._model is not None
        input_ids = self._prompt_ids(prompt)
        allowed = self._allowed_token_ids

        vocab = list(range(len(self._tokenizer)))

        def prefix_allowed_tokens_fn(batch_id, sentence):
            # Constrain only the first generated token to a skill-name start.
            if int(sentence.shape[0]) == int(input_ids.shape[1]):
                return list(allowed) if allowed else vocab
            return vocab

        gen_kwargs = dict(
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if allowed:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        with torch.no_grad():
            out = self._model.generate(input_ids, **gen_kwargs)
        new_tokens = out[0, input_ids.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_ollama(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 8},
            }
        ).encode()
        req_http = urllib.request.Request(
            self.ollama_url + "/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_http, timeout=60) as r:
            return json.loads(r.read().decode()).get("response", "")

    def submit(self, req: RouterRequest) -> RouterResponse:
        if self._mode == "scripted":
            r = self._fallback.submit(req)
            r.backend = "scripted_fallback"
            err = self._load_error or "local_unavailable"
            r.raw_text = f"{err}:" + r.raw_text
            return r
        prompt = build_prompt(req.current_skill, req.snapshot, req.reason)
        text = ""
        try:
            if self._mode == "ollama":
                text = self._generate_ollama(prompt)
            elif self._mode == "transformers":
                text = self._generate_transformers(prompt)
        except Exception as e:
            r = self._fallback.submit(req)
            r.backend = "scripted_fallback"
            r.raw_text = f"local_error:{e}"
            return r
        skill = parse_skill(text) or self._fallback.decide(req)
        return RouterResponse(
            skill=skill,
            ready_at=req.t + self.latency_s,
            backend=f"local_{self._mode}",
            raw_text=text[:200],
            latency_s=self.latency_s,
        )
