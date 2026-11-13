"""Opt-in teammate-v1 prompt diagnostic, outside the frozen package.

This deliberately restores the v1 current-skill cue and literal 20 m/s text.
It is not the clean visual-router profile and does not inherit v1 accuracy.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from contextvars import ContextVar
from pathlib import Path

from assets import asset_directory
from camera import CHOICES, VLMRouter
from config import RouterConfig
from errors import RouterError
from router_base import CameraObservation, RouterRequest, RouterResponse

SOURCE_COMMIT = "1f16b7ff59aae20c2a9adfbeb4943762d4b1f5b4"
PROMPTS_SHA256 = "0e6ea6d72a9492dc97c12ae71818aaf24d8cec3f656c8633148c8f0b913ebc5e"
DEFAULT_PROMPTS = (
    asset_directory(Path(__file__).resolve().parent / "camera-adapter.zip") / "prompts.py"
)


class TeammateV1VLMRouter(VLMRouter):
    """Exact v1 messages with current HTTP choice decoding and strict failures.

    The serving endpoint must allow per-request mm_processor_kwargs, or reject
    the request explicitly. The endpoint must be configured to accept per-request processor settings.
    """

    name = "vlm_teammate_v1"

    def __init__(self, config: RouterConfig, *, prompts_path: Path | None = None, **vlm_kwargs):
        super().__init__(config, **vlm_kwargs)
        path = Path(prompts_path) if prompts_path is not None else DEFAULT_PROMPTS
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RouterError("teammate-v1: pinned prompts.py unavailable") from exc
        self.prompt_source_sha256 = hashlib.sha256(raw).hexdigest()
        if self.prompt_source_sha256 != PROMPTS_SHA256:
            raise RouterError("teammate-v1: pinned prompt source hash mismatch")
        spec = importlib.util.spec_from_file_location("component_teammate_v1_prompts", path)
        if spec is None or spec.loader is None:
            raise RouterError("teammate-v1: cannot import pinned prompt source")
        self._prompts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._prompts)
        self._submit_lock = threading.Lock()
        self._request_skill: ContextVar[str | None] = ContextVar("teammate_v1_skill", default=None)

    def describe(self) -> str:
        return (
            f"teammate-v1 diagnostic({self.model}@{self.base_url}, current-skill cue, fallback=off)"
        )

    def _payload(self, camera: CameraObservation) -> tuple[bytes, str]:
        skill = self._request_skill.get()
        if skill not in CHOICES.values():
            raise RouterError("teammate-v1: valid request skill context required")
        raw, _ = super()._payload(camera)
        payload = json.loads(raw)
        image_uri = payload["messages"][1]["content"][0]["image_url"]["url"]
        messages = self._prompts.messages(
            image_uri,
            {
                "ego_lane": camera.lane,
                "current_skill": skill,
            },
        )
        payload["messages"] = messages
        payload["mm_processor_kwargs"] = {
            "min_pixels": self._prompts.MIN_PIXELS,
            "max_pixels": self._prompts.MAX_PIXELS,
        }
        prompt = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        return (
            json.dumps(payload, allow_nan=False).encode("utf-8"),
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )

    def submit(self, req: RouterRequest) -> RouterResponse:
        if not self._submit_lock.acquire(blocking=False):
            raise RouterError("teammate-v1: concurrent submit refused")
        token = self._request_skill.set(req.current_skill)
        try:
            return super().submit(req)
        finally:
            self._request_skill.reset(token)
            self._submit_lock.release()
