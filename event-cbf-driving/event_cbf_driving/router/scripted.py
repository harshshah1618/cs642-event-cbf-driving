"""Scripted interrupt router: deterministic lane-threat rule.

This backend is the always-available safety net.  It shares the router
contract with the LLM backends but never performs I/O and never mutates the
snapshot it is given.
"""

from __future__ import annotations

import math
import random
import time

from event_cbf_driving.config import SKILL_NAMES
from event_cbf_driving.env.skill_lane import LANE_Y
from event_cbf_driving.errors import RouterError
from event_cbf_driving.router.base import (
    RouterBackend,
    RouterRequest,
    RouterResponse,
    validate_response,
)
from event_cbf_driving.router.prompt import prompt_hash


class ScriptedRouter(RouterBackend):
    """Lane-threat rule with simulated VLM latency (default 300 ms).

    If a vehicle ahead in the active skill's target lane travels under
    ``slow_speed``, switch to the other lane skill.
    """

    name = "scripted"

    def __init__(
        self,
        latency_s: float = 0.30,
        slow_speed: float = 15.0,
        look_ahead: float = 200.0,
        latency_jitter_s: float = 0.0,
        seed: int = 0,
        latency_schedule_s=None,
    ):
        super().__init__()
        self.latency_s = float(latency_s)
        self.slow_speed = float(slow_speed)
        self.look_ahead = float(look_ahead)
        self.latency_jitter_s = float(latency_jitter_s)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.latency_schedule_s = list(latency_schedule_s or [])
        self._delay_index = 0
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in self.latency_schedule_s):
            raise RouterError("invalid recorded latency schedule")
        if not math.isfinite(self.latency_jitter_s) or self.latency_jitter_s < 0:
            raise RouterError("invalid latency jitter")
        if not math.isfinite(self.latency_s) or self.latency_s < 0.0:
            raise RouterError(f"invalid scripted latency_s {latency_s!r}")
        if not math.isfinite(self.slow_speed) or self.slow_speed < 0.0:
            raise RouterError(f"invalid scripted slow_speed {slow_speed!r}")
        if not math.isfinite(self.look_ahead) or self.look_ahead <= 0.0:
            raise RouterError(f"invalid scripted look_ahead {look_ahead!r}")

    def _lane_y(self, skill: str) -> float:
        if skill == "OVERTAKE_LEFT":
            return LANE_Y[1]
        return LANE_Y[0]

    def decide(self, req: RouterRequest) -> str:
        """Pure rule evaluation.  Does not touch router health or I/O."""
        skill = req.current_skill
        lane = self._lane_y(skill)
        ego = req.snapshot.ego
        threat = False
        for o in req.snapshot.others:
            dx = o.x - ego.x
            if dx <= 0.0 or dx > self.look_ahead:
                continue
            if abs(o.y - lane) < 1.85 and o.vx < self.slow_speed:
                threat = True
                break
        if skill in ("CRUISE_RIGHT", "RETURN_RIGHT") and threat:
            return "OVERTAKE_LEFT"
        if skill == "OVERTAKE_LEFT" and threat:
            return "RETURN_RIGHT"
        # No threat: prefer cruise/return on right.
        if skill == "OVERTAKE_LEFT" and not threat:
            # Keep overtake unless explicitly returning - during fixed-rate
            # polling this holds lane until a left-lane threat appears.
            return skill
        return skill if skill in SKILL_NAMES else "CRUISE_RIGHT"

    def _validate_request(self, req: RouterRequest) -> None:
        if not isinstance(req, RouterRequest):
            raise RouterError(f"scripted router expected RouterRequest, got {type(req).__name__}")
        if not isinstance(req.current_skill, str) or req.current_skill not in SKILL_NAMES:
            raise RouterError(f"scripted router got invalid current_skill {req.current_skill!r}")
        if not isinstance(req.reason, str) or not req.reason:
            raise RouterError("scripted router got an empty trigger reason")
        try:
            t = float(req.t)
        except (TypeError, ValueError) as exc:
            raise RouterError(f"scripted router got non-numeric t {req.t!r}") from exc
        if not math.isfinite(t):
            raise RouterError(f"scripted router got non-finite t {req.t!r}")
        try:
            sigma = float(req.sigma)
        except (TypeError, ValueError) as exc:
            raise RouterError(f"scripted router got non-numeric sigma {req.sigma!r}") from exc
        if not math.isfinite(sigma):
            raise RouterError(f"scripted router got non-finite sigma {req.sigma!r}")

    def submit(self, req: RouterRequest) -> RouterResponse:
        started = time.perf_counter()
        try:
            self._validate_request(req)
            skill = self.decide(req)
        except RouterError as exc:
            self._record_failure(exc)
            raise

        wall = time.perf_counter() - started
        raw = f"scripted:{req.current_skill}->{skill} reason={req.reason}"
        delay = max(0.0, self.latency_s + self._rng.uniform(-self.latency_jitter_s, self.latency_jitter_s)) if self.latency_jitter_s else self.latency_s
        if self.latency_schedule_s:
            delay = self.latency_schedule_s[min(self._delay_index, len(self.latency_schedule_s)-1)]
            self._delay_index += 1
        resp = RouterResponse(
            skill=skill,
            ready_at=req.t + delay,
            backend=self.name,
            raw_text=raw,
            latency_s=delay,
            request_id=self._request_id(req),
            submitted_t=req.t,
            wall_latency_s=wall,
            fallback_used=False,
            error=None,
            prompt_hash=prompt_hash(raw),
            live=False,
            model="scripted",
            usage={},
            finish_reason=None,
        )
        validate_response(resp, req)
        self._record_success(wall)
        resp.health = self.health()
        return resp

    def describe(self) -> str:
        return (
            f"scripted(slow_speed={self.slow_speed:g}, look_ahead={self.look_ahead:g}, "
            f"latency_s={self.latency_s:g}, jitter_s={self.latency_jitter_s:g})"
        )

    def reset(self):
        super().reset()
        self._rng.seed(self.seed)
        self._delay_index = 0


__all__ = ["ScriptedRouter"]
