"""Image-based Tier-1 router (VLM) and the rule it was trained to imitate.

The VLM was fine-tuned on MetaDrive front-camera frames (see ../vlm-router/). The point-mass scenarios have no
camera, so `VLMRouter` uses a render bridge: on every call it copies the point-mass car positions into a MetaDrive
scene, renders one front frame and sends it to the model. The skills and the shield stay unchanged.

Answer letters (same meaning as in training):  A = CRUISE_RIGHT, B = OVERTAKE_LEFT, C = RETURN_RIGHT.

`LabelRuleRouter` applies the labelling rule used to make the training data, but to simulator ground truth.
With latency 0 it is the "instant rule" baseline; with the VLM's measured latency it isolates the effect of the
delay; the gap between it and `VLMRouter` is the effect of perception errors.

The heavy parts (MetaDrive, vLLM) are passed in as plain callables, so this module only needs numpy:
  renderer(scene: dict) -> PNG bytes              e.g. vlm-router/render_bridge.py (MetaDrive, GPU)
  policy(png: bytes, ego_lane: str, current_skill: str) -> (letter, seconds)   e.g. vlm-router/vlm_policy.py
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from event_cbf_driving.env.skill_lane import LANE_Y
from event_cbf_driving.env.types import WorldSnapshot
from event_cbf_driving.router.base import RouterBackend, RouterRequest, RouterResponse

LETTER_TO_SKILL = {"A": "CRUISE_RIGHT", "B": "OVERTAKE_LEFT", "C": "RETURN_RIGHT"}
SKILL_TO_LETTER = {v: k for k, v in LETTER_TO_SKILL.items()}

# Point-mass lanes: right y=0, left y=4. MetaDrive lanes (straight SSSS map): right y=0, left y=3.5.
MD_LANE_SCALE = 3.5 / (LANE_Y[1] - LANE_Y[0])
LANE_MID = 0.5 * (LANE_Y[0] + LANE_Y[1])
LANE_HALF = 0.5 * (LANE_Y[1] - LANE_Y[0])
# Training frames showed cars up to 75 m ahead; the rule uses the same range so it asks the same question.
LOOK_AHEAD = 75.0
CAR_TYPES = 6  # number of vehicle models the renderer cycles through


def ego_lane(snapshot: WorldSnapshot) -> str:
    return "LEFT" if snapshot.ego.y > LANE_MID else "RIGHT"


def label_rule(snapshot: WorldSnapshot, look_ahead: float = LOOK_AHEAD) -> str:
    """The training-label rule (vlm-router/gen_frames.py:rule) on ground truth. Returns a letter.

    ego RIGHT: B if a car is ahead in the RIGHT lane and the LEFT lane has no car ahead, else A
    ego LEFT:  B if a car is still ahead in the RIGHT lane, else C
    Only cars ahead count: the front camera cannot see beside or behind us.
    """
    e = snapshot.ego
    right_ahead = left_car = False
    for o in snapshot.others:
        dx = o.x - e.x
        if dx <= 0.0 or dx > look_ahead:
            continue
        if abs(o.y - LANE_Y[0]) < LANE_HALF:
            right_ahead = True
        elif abs(o.y - LANE_Y[1]) < LANE_HALF:
            left_car = True
    if ego_lane(snapshot) == "RIGHT":
        return "B" if (right_ahead and not left_car) else "A"
    return "B" if right_ahead else "C"


def snapshot_to_scene(snapshot: WorldSnapshot, car_seed: int = 0, max_ahead: float = 150.0,
                      max_behind: float = 50.0) -> dict:
    """Point-mass state -> MetaDrive scene, relative to the ego car (dx along the road, y in MetaDrive metres)."""
    e = snapshot.ego
    cars = []
    for i, o in enumerate(snapshot.others):
        dx = o.x - e.x
        if -max_behind <= dx <= max_ahead:
            cars.append(dict(dx=float(dx), y=float(o.y * MD_LANE_SCALE),
                             heading=float(math.atan2(o.vy * MD_LANE_SCALE, max(o.vx, 1.0))) if o.vx > 0.5 else 0.0,
                             model=int((car_seed * 7 + i * 3) % CAR_TYPES)))
    return dict(ego_y=float(e.y * MD_LANE_SCALE),
                ego_heading=float(math.atan2(e.vy * MD_LANE_SCALE, max(e.vx, 1.0))), cars=cars)


class LabelRuleRouter(RouterBackend):
    """Ground-truth version of the VLM's task. latency_s may be a number or a list to cycle through."""

    name = "label_rule"

    def __init__(self, latency_s: float | Sequence[float] = 0.0):
        self._lat = list(latency_s) if isinstance(latency_s, (list, tuple)) else [float(latency_s)]
        self._i = 0
        self.latency_s = float(sorted(self._lat)[len(self._lat) // 2])

    def submit(self, req: RouterRequest) -> RouterResponse:
        lat = self._lat[self._i % len(self._lat)]
        self._i += 1
        letter = label_rule(req.snapshot)
        return RouterResponse(skill=LETTER_TO_SKILL[letter], ready_at=req.t + lat, backend=self.name,
                              raw_text=f"rule:{letter}", latency_s=lat)


class VLMRouter(RouterBackend):
    """Render bridge + VLM. ready_at = t + the model's measured wall-clock latency for this call.

    Render time is recorded but not added: a real car gets its camera frame for free, the render is only a
    stand-in for the camera. Every call is logged in `self.log` (with the rule's answer for comparison).
    """

    name = "vlm"

    def __init__(self, renderer: Callable[[dict], bytes], policy: Callable[[bytes, str, str], Tuple[str, float]],
                 frame_sink: Optional[Callable[[str, bytes], None]] = None, right_lane_skill: Optional[str] = None):
        """right_lane_skill: DIAGNOSTIC PATCH, not a fix. If set, this skill text is sent instead of the real
        current skill whenever the ego is in the RIGHT lane. Training only ever paired RIGHT with CRUISE_RIGHT
        (gen_frames.py), so right_lane_skill="CRUISE_RIGHT" tests how much of the error is that shortcut."""
        self.right_lane_skill = right_lane_skill
        self.renderer = renderer
        self.policy = policy
        self.frame_sink = frame_sink
        self.car_seed = 0
        self.episode = ""
        self.log: List[Dict] = []
        self.latency_s = 0.125

    def start_episode(self, tag: str, car_seed: int):
        self.episode, self.car_seed = tag, int(car_seed)

    def submit(self, req: RouterRequest) -> RouterResponse:
        import time

        snap = req.snapshot
        lane = ego_lane(snap)
        t0 = time.perf_counter()
        png = self.renderer(snapshot_to_scene(snap, self.car_seed))
        render_s = time.perf_counter() - t0
        sent = self.right_lane_skill if (self.right_lane_skill and lane == "RIGHT") else req.current_skill
        letter, infer_s = self.policy(png, lane, sent)
        skill = LETTER_TO_SKILL.get(letter, req.current_skill)  # unparseable answer: keep the current skill
        want = label_rule(snap)
        frame = f"{self.episode}_t{req.t:05.2f}.png"
        if self.frame_sink:
            self.frame_sink(frame, png)
        self.log.append(dict(episode=self.episode, t=round(req.t, 2), reason=req.reason, ego_lane=lane,
                             current=req.current_skill, sent_skill=sent, letter=letter, rule=want, ok=letter == want,
                             infer_ms=round(infer_s * 1000, 1), render_ms=round(render_s * 1000, 1),
                             ego_y=round(snap.ego.y, 2), frame=frame,
                             cars=[dict(dx=round(o.x - snap.ego.x, 1), y=round(o.y, 2), vx=round(o.vx, 1))
                                   for o in snap.others]))
        self.latency_s = infer_s
        return RouterResponse(skill=skill, ready_at=req.t + infer_s, backend=self.name,
                              raw_text=f"vlm:{letter} rule:{want}", latency_s=infer_s)
