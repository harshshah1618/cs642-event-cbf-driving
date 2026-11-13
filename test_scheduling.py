import math

from env_types import VehicleState, WorldSnapshot
from scheduling import LaneOpportunity, lane_opportunity, opportunity_due


def test_latency_advances_request_without_equating_it_with_collision():
    opportunity = LaneOpportunity(0, 20, 4.0, math.inf, 2.0, True)
    assert not opportunity_due(opportunity, "deadline", 0.3)
    assert opportunity_due(opportunity, "deadline", 2.0)
    assert not opportunity_due(opportunity, "no_latency", 2.0)


def test_closing_alternative_window_can_trigger_before_current_lane_constraint():
    opportunity = LaneOpportunity(8, 20, 8.0, 3.0, 2.0, True)
    assert opportunity_due(opportunity, "deadline", 1.0)
    assert not opportunity_due(opportunity, "ttc", 1.0)


def test_scene_geometry_changes_trigger_when_ego_state_is_identical():
    empty = WorldSnapshot(0, VehicleState(0, 0, 20), [], 200)
    blocked = empty.copy()
    blocked.others = [VehicleState(55, 0, 0)]
    assert not lane_opportunity(empty, "CRUISE_RIGHT").useful
    assert lane_opportunity(blocked, "CRUISE_RIGHT").useful
    assert opportunity_due(lane_opportunity(blocked, "CRUISE_RIGHT"), "deadline", 1)


def test_blocked_alternative_and_in_progress_switch_do_not_chatter():
    snap = WorldSnapshot(
        0, VehicleState(0, 0, 20), [VehicleState(45, 0, 0), VehicleState(0, 4, 20)], 200
    )
    assert not lane_opportunity(snap, "CRUISE_RIGHT").useful
    snap.others.pop()
    snap.ego.y = 2
    assert not lane_opportunity(snap, "OVERTAKE_LEFT").useful
