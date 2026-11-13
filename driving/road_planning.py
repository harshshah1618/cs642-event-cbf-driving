"""Road-relative maneuver planning on upstream procedural MetaDrive roads.

All scheduling policies share this finite-horizon constant-velocity planner,
fresh PID/IDM controller, and lane-change admission guard. It uses simulator
objects returned by the same surrounding-object sensor for every method.
The predictor/guard are engineering approximations, not certified MPC/CBF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LaneView:
    index: int
    front_gap: float = 150.0
    front_speed: float = 25.0
    rear_gap: float = 150.0
    rear_speed: float = 0.0
    remaining: float = 150.0
    route_valid: bool = True


@dataclass(frozen=True)
class RoadView:
    speed: float
    current: int
    lanes: tuple[LaneView, ...]
    settled: bool = True
    speed_reference: float = 11.11


def switch_admissible(
    view: RoadView, lane: int, delay: float = 0.0, transition: float = 2.0
) -> bool:
    if lane == view.current:
        return True
    if abs(lane - view.current) != 1 or not view.settled:
        return False
    target = view.lanes[lane]
    times = np.linspace(delay, delay + transition, 9)
    front = target.front_gap + (target.front_speed - view.speed) * times
    rear = target.rear_gap + (view.speed - target.rear_speed) * times
    return bool(np.min(front) > 7.0 and np.min(rear) > 7.0 and target.route_valid)


def lane_speed(lane: LaneView, desired: float) -> float:
    # Continuous progress proxy, with a long lookahead so we do not wait until
    # emergency braking to notice the benefit of a different lane.
    return min(desired, lane.front_speed + max(0.0, lane.front_gap - 12.0) / 4.0)


def road_request_due(
    view: RoadView,
    policy: str,
    latency: float,
    *,
    transition: float = 2.0,
    buffer: float = 0.5,
    desired: float = 11.11,
    threshold: float = 3.0,
) -> bool:
    current = view.lanes[view.current]
    latency = 0.0 if policy == "no_latency" else latency
    # The maneuver planner chooses speed as well as lane. A cleared blocker
    # invalidates an old low-speed/stop plan even when there is only one lane.
    if view.speed_reference < lane_speed(current, desired) - 0.8:
        return True
    if not current.route_valid and current.remaining < max(
        25, view.speed * (latency + transition + 1)
    ):
        return True
    if not view.settled:
        return False
    current_progress = lane_speed(current, desired)
    choices = [
        lane
        for lane in view.lanes
        if lane.index != view.current
        and switch_admissible(view, lane.index)
        and lane_speed(lane, desired) > current_progress + 0.8
    ]
    if not choices:
        return False
    if policy == "benefit":
        return True
    closing = view.speed - current.front_speed
    t_front = max(0.0, current.front_gap - 8.0) / max(closing, 0.1)
    if policy == "ttc":
        return t_front <= threshold
    expiry = math.inf
    for lane in choices:
        if lane.rear_speed > view.speed:
            expiry = min(expiry, max(0, lane.rear_gap - 7) / (lane.rear_speed - view.speed))
    required = latency + transition + buffer
    return min(t_front, expiry) <= required


def plan_maneuver(view: RoadView, desired: float = 11.11) -> tuple[int, float]:
    """Enumerate lane/speed candidates with explicit finite-horizon checks.

    Candidate scores reward progress and penalize small predicted gaps and
    maneuver changes. The scalar speed reference is passed to the same fresh
    IDM loop; it is not a committed open-loop actuation sequence.
    """
    times = np.linspace(0.2, 4.0, 20)
    scores = []
    for lane in view.lanes:
        if abs(lane.index - view.current) > 1:
            continue
        if not switch_admissible(view, lane.index):
            continue
        for speed in (0.0, desired * 0.55, desired * 0.8, desired):
            acceleration = float(np.clip((speed - view.speed) / 2.0, -3.0, 1.5))
            duration = abs(speed - view.speed) / max(abs(acceleration), 1e-6)
            accel_time = np.minimum(times, duration)
            position = (
                view.speed * accel_time
                + 0.5 * acceleration * accel_time**2
                + np.maximum(0.0, times - duration) * speed
            )
            front = lane.front_gap + lane.front_speed * times - position
            penalty = np.maximum(0.0, 5.0 + 0.8 * speed - front)
            cost = float(position[-1] - 10 * np.mean(penalty**2))
            if lane.index != view.current:
                cost -= 2.0
                source = view.lanes[view.current]
                early = times < 1.0
                old_gap = source.front_gap + source.front_speed * times[early] - position[early]
                cost -= float(10 * np.sum(np.maximum(0, 5 - old_gap) ** 2))
            if not lane.route_valid:
                cost -= 100 * max(0.0, 1.0 - lane.remaining / 60)
            scores.append((cost, -abs(lane.index - view.current), speed, lane.index))
    if not scores:
        return view.current, 0.0
    _, _, speed, selected_lane = max(scores)
    return selected_lane, speed


class RoadController:
    """Upstream PID steering + IDM acceleration; geometry is refreshed each tick."""

    def __init__(self, vehicle, seed=0, desired=11.11):
        from metadrive.policy.idm_policy import IDMPolicy

        self.vehicle = vehicle
        self.policy = IDMPolicy(vehicle, seed)
        self.target_lane = vehicle.lane
        self.desired = desired
        self.speed_reference = desired
        self.front_objects = []
        self.reference_lanes = []

    def observe(self):
        refs = self.vehicle.navigation.current_ref_lanes
        if self.target_lane not in refs:
            # Road-topology bookkeeping runs locally for every scheduler. A
            # stale answer from a previous road is rejected by the caller.
            idx = min(self.target_lane.index[-1], len(refs) - 1)
            self.target_lane = refs[idx]
        self.reference_lanes = refs
        objects = self.vehicle.lidar.get_surrounding_objects(self.vehicle)
        self.surrounding_objects = objects
        next_lanes = self.vehicle.navigation.next_ref_lanes
        self.front_objects = []
        views = []
        for i, lane in enumerate(refs):
            longitudinal, _ = lane.local_coordinates(self.vehicle.position)
            front = None
            front_gap = rear_gap = 150.0
            # Lane labels can lag a physical cut-in. Project observed footprints
            # rather than waiting for an object's discrete lane identity to
            # change. Use only current state and a short velocity extrapolation.
            front_speed, rear_speed = 25.0, 0.0
            for obj in objects:
                other_s, lateral = lane.local_coordinates(obj.position)
                heading = lane.heading_theta_at(other_s)
                tangent = np.array([math.cos(heading), math.sin(heading)])
                normal = np.array([-tangent[1], tangent[0]])
                velocity = np.asarray(getattr(obj, "velocity", [0.0, 0.0]))
                future_lat = lateral + 1.0 * float(velocity @ normal)
                crossing = lateral * future_lat <= 0
                clearance = 0.5 * (self.vehicle.WIDTH + float(getattr(obj, "WIDTH", 2.0))) + 0.5
                if not crossing and min(abs(lateral), abs(future_lat)) > clearance:
                    continue
                distance = other_s - longitudinal
                extent = 0.5 * (self.vehicle.LENGTH + float(getattr(obj, "LENGTH", 4.5)))
                gap = max(0.0, abs(distance) - extent)
                if distance >= 0 and gap < front_gap:
                    front, front_gap = obj, gap
                    front_speed = max(0.0, float(velocity @ tangent))
                elif distance < 0 and gap < rear_gap:
                    rear_gap = gap
                    rear_speed = max(0.0, float(velocity @ tangent))
            self.front_objects.append(front)
            valid = next_lanes is None or any(lane.is_previous_lane_of(n) for n in next_lanes)
            views.append(
                LaneView(
                    i,
                    front_gap,
                    front_speed,
                    rear_gap,
                    rear_speed,
                    lane.length - longitudinal,
                    valid,
                )
            )
        index = refs.index(self.target_lane)
        _, lateral = self.target_lane.local_coordinates(self.vehicle.position)
        heading = self.target_lane.heading_theta_at(
            self.target_lane.local_coordinates(self.vehicle.position)[0]
        )
        heading_error = math.atan2(
            math.sin(self.vehicle.heading_theta - heading),
            math.cos(self.vehicle.heading_theta - heading),
        )
        return RoadView(
            float(self.vehicle.speed),
            index,
            tuple(views),
            abs(lateral) < 0.6 and abs(heading_error) < 0.12,
            self.speed_reference,
        )

    def accept(self, lane, speed, view):
        if not switch_admissible(view, lane):
            return False
        self.target_lane = self.reference_lanes[lane]
        self.speed_reference = speed
        return True

    def action(self, view):
        target = self.reference_lanes.index(self.target_lane)
        # Steering follows the committed maneuver. Braking checks both the
        # occupied and intended lane during transition, using fresh observations.
        longitudinal, lateral = self.target_lane.local_coordinates(self.vehicle.position)
        h0 = self.target_lane.heading_theta_at(longitudinal)
        h1 = self.target_lane.heading_theta_at(min(self.target_lane.length, longitudinal + 5))
        curvature = abs(math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))) / 5
        turn_speed = math.sqrt(2.5 / max(curvature, 1e-6))
        self.policy.target_speed = max(0.1, min(self.speed_reference, turn_speed) * 3.6)
        choices = [target]
        physical = min(
            range(len(self.reference_lanes)),
            key=lambda i: abs(self.reference_lanes[i].local_coordinates(self.vehicle.position)[1]),
        )
        if physical not in choices:
            choices.append(physical)
        accelerations = [
            self.policy.acceleration(self.front_objects[i], max(0.1, view.lanes[i].front_gap))
            for i in choices
        ]
        throttle = min(accelerations)
        # A front/rear lane abstraction misses crossing traffic and objects on
        # the next road segment. Use a short world-coordinate footprint forecast
        # as a common fresh-state brake backup for every scheduler. It assumes
        # constant obstacle velocity; no invariance guarantee is implied.
        for future in np.linspace(0.1, 1.5, 8):
            distance = longitudinal + view.speed * future
            heading = self.target_lane.heading_theta_at(distance)
            tangent = np.array([math.cos(heading), math.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            ego_position = np.asarray(
                self.target_lane.position(distance, lateral * max(0.0, 1 - future / 2))
            )
            for other in self.surrounding_objects:
                velocity = np.asarray(getattr(other, "velocity", [0.0, 0.0]))
                relative = np.asarray(other.position) + future * velocity - ego_position
                yaw = float(getattr(other, "heading_theta", heading)) - heading
                length = float(getattr(other, "LENGTH", 4.5))
                width = float(getattr(other, "WIDTH", 2.0))
                along = (
                    0.5
                    * (
                        self.vehicle.LENGTH
                        + length * abs(math.cos(yaw))
                        + width * abs(math.sin(yaw))
                    )
                    + 1.0
                )
                across = (
                    0.5
                    * (
                        self.vehicle.WIDTH
                        + length * abs(math.sin(yaw))
                        + width * abs(math.cos(yaw))
                    )
                    + 0.15
                )
                if (
                    abs(float(relative @ tangent)) < along
                    and abs(float(relative @ normal)) < across
                ):
                    throttle = min(throttle, -1.0)
        steer = self.policy.steering_control(self.target_lane)
        return np.clip([steer, throttle], -1, 1).tolist()

    def close(self):
        self.policy.destroy()
