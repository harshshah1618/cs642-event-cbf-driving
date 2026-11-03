"""Scripted interrupt router — same API as LLM backends, labeled clearly."""

from __future__ import annotations

from event_cbf_driving.env.skill_lane import LANE_Y
from event_cbf_driving.router.base import RouterBackend, RouterRequest, RouterResponse, SKILL_NAMES


class ScriptedRouter(RouterBackend):
    """Lane-threat rule with simulated VLM latency (default 300 ms).

    If a vehicle ahead in the active skill's target lane travels under
    ``slow_speed``, switch to the other lane skill.
    """

    name = "scripted"

    def __init__(self, latency_s: float = 0.30, slow_speed: float = 15.0, look_ahead: float = 200.0):
        self.latency_s = float(latency_s)
        self.slow_speed = float(slow_speed)
        self.look_ahead = float(look_ahead)

    def _lane_y(self, skill: str) -> float:
        if skill == "OVERTAKE_LEFT":
            return LANE_Y[1]
        return LANE_Y[0]

    def decide(self, req: RouterRequest) -> str:
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
            # Keep overtake unless explicitly returning — during fixed-rate
            # polling this holds lane until a left-lane threat appears.
            return skill
        return skill if skill in SKILL_NAMES else "CRUISE_RIGHT"

    def submit(self, req: RouterRequest) -> RouterResponse:
        skill = self.decide(req)
        return RouterResponse(
            skill=skill,
            ready_at=req.t + self.latency_s,
            backend=self.name,
            raw_text=f"scripted:{req.current_skill}->{skill} reason={req.reason}",
            latency_s=self.latency_s,
        )
