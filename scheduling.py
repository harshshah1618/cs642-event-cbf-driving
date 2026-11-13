"""Cheap, explicit scheduling for delayed maneuver selection.

This module predicts *opportunity loss*, not collision probability. Its constant
velocity estimates do not constitute a safety certificate. Every accepted plan
still needs fresh admission checks and an independent control-rate backup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from env_types import WorldSnapshot


@dataclass(frozen=True)
class LaneOpportunity:
    current_speed: float
    alternative_speed: float
    time_to_constraint: float
    alternative_open_for: float
    transition_s: float
    useful: bool


def lane_opportunity(
    snapshot: WorldSnapshot,
    skill: str,
    *,
    transition_s: float = 2.0,
    clearance: float = 8.0,
    prediction_margin: float = 1.0,
) -> LaneOpportunity:
    """Two-lane diagnostic shared by threshold and deadline ablations.

    Reads current observed geometry only, never scenario parameters/future
    events. Finite lane-change opportunities include fast rear approaches.
    """
    ego = snapshot.ego
    target = 4.0 if skill == "OVERTAKE_LEFT" else 0.0
    alternative = 0.0 if target == 4.0 else 4.0
    speed = max(ego.vx, 0.0)
    current_speed = 20.0
    alternative_speed = 20.0
    time_to_constraint = math.inf
    open_for = math.inf
    valid_alternative = True
    for other in snapshot.others:
        dx = other.x - ego.x
        extent = 0.5 * (ego.length + other.length)
        if abs(other.y - target) < 1.85 and 0 < dx < 200 and other.vx < 15:
            current_speed = min(current_speed, max(other.vx, 0.0))
            closing = max(speed - other.vx, 0.1)
            time_to_constraint = min(
                time_to_constraint, max(0.0, dx - extent - clearance) / closing
            )
        if abs(other.y - alternative) < 1.85:
            gap = abs(dx) - extent - clearance - prediction_margin
            if gap <= 0:
                valid_alternative = False
            if dx > 0:
                if dx < 100:
                    alternative_speed = min(alternative_speed, max(other.vx, 0.0))
                closing = speed - other.vx
            else:
                closing = other.vx - speed
            if closing > 0:
                open_for = min(open_for, max(0.0, gap) / closing)
    # Do not schedule an opposite maneuver while still acquiring this lane.
    settled = abs(ego.y - target) < 0.8 and abs(ego.vy) < 0.8
    useful = (
        settled
        and valid_alternative
        and current_speed < 15
        and alternative_speed > current_speed + 1.0
    )
    return LaneOpportunity(
        current_speed, alternative_speed, time_to_constraint, open_for, transition_s, useful
    )


def opportunity_due(
    opportunity: LaneOpportunity,
    policy: str,
    latency_bound_s: float,
    buffer_s: float = 0.5,
    threshold_s: float = 3.0,
) -> bool:
    """Compare scheduling policies with the *same* opportunity information.

    deadline: ask before the active corridor runs out of progress reserve or
    before an alternative's admission window closes. no_latency uses the same
    rule with zero estimated latency. benefit asks immediately when useful.
    """
    if not opportunity.useful:
        return False
    if policy == "benefit":
        return True
    if policy == "ttc":
        return opportunity.time_to_constraint <= threshold_s
    latency = 0.0 if policy == "no_latency" else latency_bound_s
    required = latency + opportunity.transition_s + buffer_s
    return min(opportunity.time_to_constraint, opportunity.alternative_open_for) <= required
