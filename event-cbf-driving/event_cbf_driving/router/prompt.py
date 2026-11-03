"""Shared Tier-1 prompt + skill-token parse (used by OpenAI-compat and local LLM)."""

from __future__ import annotations

from typing import Optional, Tuple

from event_cbf_driving.env.skill_lane import LANE_Y
from event_cbf_driving.env.types import WorldSnapshot
from event_cbf_driving.router.base import SKILL_NAMES

PROMPT = """Skill router. Output exactly one token: CRUISE_RIGHT or OVERTAKE_LEFT or RETURN_RIGHT.

right_threat={right_threat}
left_threat={left_threat}
current={skill}
reason={reason}
ego_x={x:.0f} ego_y={y:.1f} ego_vx={vx:.0f}
others={others}

Examples:
right_threat=yes left_threat=no current=CRUISE_RIGHT -> OVERTAKE_LEFT
right_threat=no left_threat=yes current=OVERTAKE_LEFT -> RETURN_RIGHT
right_threat=no left_threat=no current=OVERTAKE_LEFT -> OVERTAKE_LEFT
right_threat=no left_threat=no current=CRUISE_RIGHT -> CRUISE_RIGHT

Answer:"""


def _lane_threat(
    snapshot: WorldSnapshot,
    lane_y: float,
    slow_speed: float = 15.0,
    look_ahead: float = 200.0,
    lane_half: float = 1.85,
) -> bool:
    ego = snapshot.ego
    for o in snapshot.others:
        dx = o.x - ego.x
        if dx <= 0.0 or dx > look_ahead:
            continue
        if abs(o.y - lane_y) < lane_half and o.vx < slow_speed:
            return True
    return False


def threat_flags(snapshot: WorldSnapshot) -> Tuple[str, str]:
    right = "yes" if _lane_threat(snapshot, LANE_Y[0]) else "no"
    left = "yes" if _lane_threat(snapshot, LANE_Y[1]) else "no"
    return right, left


def format_others(snapshot: WorldSnapshot, limit: int = 8) -> str:
    if not snapshot.others:
        return "none"
    parts = []
    for o in snapshot.others[:limit]:
        parts.append(f"({o.x:.0f},{o.y:.1f},{o.vx:.1f})")
    return ",".join(parts)


def build_prompt(skill: str, snapshot: WorldSnapshot, reason: str) -> str:
    right_threat, left_threat = threat_flags(snapshot)
    return PROMPT.format(
        skill=skill,
        x=snapshot.ego.x,
        y=snapshot.ego.y,
        vx=snapshot.ego.vx,
        others=format_others(snapshot),
        reason=reason,
        right_threat=right_threat,
        left_threat=left_threat,
    )


def build_chat_prompt(skill: str, snapshot: WorldSnapshot, reason: str) -> str:
    """OpenAI-compatible chat prompt (same facts, chat-oriented)."""
    return build_prompt(skill, snapshot, reason)


def parse_skill(text: str, fallback: Optional[str] = None) -> Optional[str]:
    """Extract a skill token from model text. Prefers first explicit match."""
    if not text:
        return fallback if fallback in SKILL_NAMES else None
    up = text.upper().replace("-", "_").replace(" ", "_")
    best = None
    best_idx = 10**9
    for s in SKILL_NAMES:
        idx = up.find(s)
        if 0 <= idx < best_idx:
            best = s
            best_idx = idx
    if best is not None:
        return best
    aliases = {
        "CRUISE": "CRUISE_RIGHT",
        "OVERTAKE": "OVERTAKE_LEFT",
        "RETURN": "RETURN_RIGHT",
    }
    for key, skill_name in aliases.items():
        if key in up:
            return skill_name
    return fallback if fallback in SKILL_NAMES else None
