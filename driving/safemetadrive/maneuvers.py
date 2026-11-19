"""Opt-in maneuver supervision; forecasts remain heuristic, not certificates.

Recovery may return to the source of an unfinished lane change. It never chooses
an unrelated lane, releases a collision stop, or disables the longitudinal cap.
"""

import math
from collections import deque
from copy import copy

import numpy as np


def transition_reason(driver, lane, objects):
    """Check a proposed target without changing the executing driver's state."""
    vehicle = driver.vehicle
    lanes = vehicle.navigation.current_ref_lanes
    if lane not in lanes:
        return "lane_context_changed"
    front, rear, _ = driver.lane_objects(lane, objects)
    if driver.protection:
        if front < max(8, vehicle.speed * 1.1):
            return "front_gap_too_small"
        if rear < max(10, vehicle.speed * 0.8):
            return "rear_gap_too_small"
    following = getattr(vehicle.navigation, "next_ref_lanes", None)
    distance, _ = lane.local_coordinates(vehicle.position)
    if following and lane.length - distance < max(15, vehicle.speed * 3):
        network = vehicle.navigation.map.road_network
        if not any(network.has_connection(lane.index, successor.index) for successor in following):
            return "route_lane_ends"
    if driver.protection and driver.prediction:
        probe = copy(driver)
        probe.lane = lane
        # Rear traffic is intentionally ignored by emergency braking, but must
        # be considered before entering or returning to another lane.
        if not math.isinf(probe.conflict_time(objects, include_rear=True)):
            return "transition_forecast_conflict"
    return None


class ManeuverSupervisor:
    """Track an admitted maneuver and attempt a checked abort if it gets stuck."""

    def __init__(self, dt=0.1, blocked_after=2.0, retry_after=5.0):
        if not all(math.isfinite(x) and x > 0 for x in (dt, blocked_after, retry_after)):
            raise ValueError("Maneuver times must be positive and finite")
        self.dt, self.blocked_after, self.retry_after = dt, blocked_after, retry_after
        self.time = 0.0
        self.phase = "idle"
        self.source = self.target = None
        self.last_progress = 0.0
        self.anchor = None
        self.last_block_check = -math.inf
        self.cooldowns: dict[int, float] = {}
        self.events: deque[dict] = deque(maxlen=128)
        self.abort_count = 0
        self.completed_count = 0
        self.recovery_block_reason = "none"

    @property
    def active(self):
        return self.phase in ("executing", "blocked", "returning")

    def event(self, phase, reason):
        self.phase = phase
        self.events.append(dict(t=round(self.time, 6), phase=phase, reason=reason))

    def admission_reason(self, driver, lane, objects):
        if self.active:
            return "maneuver_in_progress"
        if self.cooldowns.get(id(lane), -math.inf) > self.time:
            return "recently_aborted_target"
        return transition_reason(driver, lane, objects)

    def start(self, source, target, position):
        self.source, self.target = source, target
        self.anchor = np.array(position, dtype=float)
        self.last_progress = self.time
        self.event("executing", "lane_change_admitted")

    def reset_road(self):
        if self.active:
            self.event("idle", "navigation_road_changed")
        self.source = self.target = None
        self.anchor = None
        self.cooldowns.clear()
        self.recovery_block_reason = "none"

    def update(self, driver, objects):
        self.time += self.dt
        if not self.active:
            return
        lanes = driver.vehicle.navigation.current_ref_lanes
        lane = self.target
        if lane is None or lane not in lanes or driver.lane is not lane:
            self.reset_road()
            return
        position = np.asarray(driver.vehicle.position)
        _, lateral = lane.local_coordinates(position)
        if abs(lateral) <= 0.35:
            heading = lane.heading_theta_at(lane.local_coordinates(position)[0])
            error = math.atan2(
                math.sin(heading - driver.vehicle.heading_theta),
                math.cos(heading - driver.vehicle.heading_theta),
            )
            if abs(error) <= 0.15:
                self.completed_count += 1
                self.event("completed", "lane_center_reached")
                self.source = self.target = None
                self.anchor = None
                return
        if self.anchor is None or np.linalg.norm(position - self.anchor) >= 0.5:
            self.anchor = position.copy()
            self.last_progress = self.time
            if self.phase == "blocked":
                self.event("executing", "motion_resumed")
                self.recovery_block_reason = "none"
        if driver.vehicle.speed >= 0.5 or self.time - self.last_progress < self.blocked_after:
            return
        if self.phase == "returning":
            # Never oscillate between two failed targets or override a new stop.
            return
        if self.phase != "blocked":
            self.event("blocked", "unfinished_maneuver_without_progress")
        if not driver.protection or not driver.prediction:
            self.recovery_block_reason = "protection_unavailable"
            return
        if self.time - self.last_block_check < 0.5:
            return
        self.last_block_check = self.time
        if self.source not in lanes:
            return
        reason = transition_reason(driver, self.source, objects)
        if reason is not None:
            self.recovery_block_reason = reason
            return
        self.recovery_block_reason = "none"
        abandoned = self.target
        driver.lane = self.source
        driver.careful = True
        driver.changes += 1
        self.target = self.source
        self.source = None
        self.cooldowns[id(abandoned)] = self.time + self.retry_after
        self.abort_count += 1
        self.anchor = position.copy()
        self.last_progress = self.time
        self.event("returning", "checked_return_to_source_lane")

    def feedback(self):
        return dict(
            maneuver_state=self.phase,
            no_progress_seconds=round(self.time - self.last_progress, 1) if self.active else 0.0,
            recovery_aborts=self.abort_count,
            recovery_block_reason=self.recovery_block_reason,
        )
