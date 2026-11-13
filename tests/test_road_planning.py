from types import SimpleNamespace

import numpy as np
from driving.road_planning import (
    LaneView,
    RoadController,
    RoadView,
    plan_maneuver,
    road_request_due,
    switch_admissible,
)


def test_rear_approach_blocks_an_otherwise_open_lane():
    scene = RoadView(10, 0, (LaneView(0, 20, 0), LaneView(1, rear_gap=15, rear_speed=20)))
    assert not switch_admissible(scene, 1)
    assert plan_maneuver(scene)[0] == 0


def test_planner_prefers_open_adjacent_lane_over_stalled_lead():
    scene = RoadView(10, 0, (LaneView(0, 40, 0), LaneView(1)))
    assert plan_maneuver(scene)[0] == 1
    assert road_request_due(scene, "deadline", 2)
    assert not road_request_due(scene, "no_latency", 2)


def test_no_chatter_when_already_switching_or_no_alternative_has_benefit():
    scene = RoadView(10, 0, (LaneView(0), LaneView(1)))
    assert not road_request_due(scene, "benefit", 1)
    blocked = RoadView(10, 0, (LaneView(0, 20, 0), LaneView(1)), settled=False)
    assert not road_request_due(blocked, "benefit", 1)


def test_invalid_route_is_replanned_before_lane_ends():
    scene = RoadView(10, 0, (LaneView(0, remaining=15, route_valid=False), LaneView(1)))
    assert road_request_due(scene, "deadline", 1)
    assert plan_maneuver(scene)[0] == 1


def test_observation_detects_cut_in_before_discrete_lane_label_changes():
    class Lane:
        length = 100

        def __init__(self, index, y):
            self.index = ("a", "b", index)
            self.y = y

        def local_coordinates(self, position):
            return float(position[0]), float(position[1] - self.y)

        def heading_theta_at(self, _s):
            return 0.0

    lanes = [Lane(0, 0), Lane(1, 4)]
    other = SimpleNamespace(
        position=np.array([25.0, 3.0]),
        velocity=np.array([8.0, -1.0]),
        WIDTH=2,
        LENGTH=4.5,
        lane=lanes[1],
    )
    vehicle = SimpleNamespace(
        navigation=SimpleNamespace(current_ref_lanes=lanes, next_ref_lanes=None),
        position=np.array([0.0, 0.0]),
        heading_theta=0.0,
        speed=10.0,
        WIDTH=2.0,
        LENGTH=4.5,
        lidar=SimpleNamespace(get_surrounding_objects=lambda _v: [other]),
    )
    controller = RoadController.__new__(RoadController)
    controller.vehicle = vehicle
    controller.target_lane = lanes[0]
    controller.speed_reference = 11.11
    view = controller.observe()
    assert view.lanes[0].front_gap == 20.5
    assert controller.front_objects[0] is other


def test_single_lane_stop_plan_is_invalidated_when_blocker_clears():
    stopped = RoadView(8, 0, (LaneView(0, 8, 0),))
    assert plan_maneuver(stopped) == (0, 0.0)
    cleared = RoadView(0, 0, (LaneView(0),), speed_reference=0.0)
    for policy in ("deadline", "no_latency", "benefit", "ttc"):
        assert road_request_due(cleared, policy, 1.0)
    assert plan_maneuver(cleared)[1] > 10
    mid_transition = RoadView(0, 0, (LaneView(0),), settled=False, speed_reference=0)
    for policy in ("deadline", "no_latency", "benefit", "ttc"):
        assert road_request_due(mid_transition, policy, 1.0)
