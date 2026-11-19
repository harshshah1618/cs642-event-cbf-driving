"""Maneuver admission/recovery checks independent of learned camera outputs."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from driving.safemetadrive.camera_planner import messages
from driving.safemetadrive.maneuvers import ManeuverSupervisor
from driving.safemetadrive.policies import LaneCautionDriver

spec = importlib.util.spec_from_file_location(
    "safemetadrive_camera_data",
    Path(__file__).resolve().parents[1] / "tools/safemetadrive_camera_data.py",
)
camera_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_data)
teacher_recommendation = camera_data.teacher_recommendation


class Lane:
    length = 100

    def __init__(self, index):
        self.index = ("road", "end", index)
        self.center = -3.5 * index

    def local_coordinates(self, position):
        return position[0], self.center - position[1]

    def heading_theta_at(self, distance):
        return 0.0

    def position(self, distance, lateral):
        return np.array([distance, self.center - lateral])


def driver_fixture(monkeypatch):
    lanes = [Lane(i) for i in range(3)]
    driver = object.__new__(LaneCautionDriver)
    driver.vehicle = SimpleNamespace(
        navigation=SimpleNamespace(current_ref_lanes=lanes, next_ref_lanes=[]),
        position=np.array([0.0, 0.0]),
        speed=0.0,
        heading_theta=0.0,
        lidar=SimpleNamespace(get_surrounding_objects=lambda vehicle: []),
        LENGTH=4.5,
        WIDTH=1.8,
        FRONT_WHEELBASE=1.3,
        REAR_WHEELBASE=1.3,
        get_dynamics_parameters=lambda: {"max_steering": 35},
    )
    driver.lane, driver.careful, driver.changes = lanes[0], False, 0
    driver.preview, driver.protection, driver.prediction = True, True, True
    driver.speed, driver.headway, driver.lateral_accel = 30.0, 1.0, 2.5
    driver.clearance, driver.change_period, driver.tick = 3.0, 15, 0
    driver.planner, driver.skill, driver.interventions = "external", None, 0
    driver.maneuvers = ManeuverSupervisor(dt=0.1, blocked_after=0.2)
    monkeypatch.setattr(LaneCautionDriver, "lane_objects", lambda self, lane, objects: (80, 80, 0))
    monkeypatch.setattr(
        LaneCautionDriver, "conflict_time", lambda self, objects, **kwargs: math.inf
    )
    return driver, lanes


def test_future_transition_conflict_rejects_target_without_changing_lane(monkeypatch):
    driver, lanes = driver_fixture(monkeypatch)

    def forecast(probe, objects, **kwargs):
        assert probe.lane is lanes[1]
        assert driver.lane is lanes[0]
        return 2.0

    monkeypatch.setattr(LaneCautionDriver, "conflict_time", forecast)
    assert not driver.apply_intent(1, True, lanes[0])
    assert driver.lane is lanes[0]
    assert driver.last_intent_reason == "transition_forecast_conflict"
    assert driver.careful and driver.last_caution_applied
    assert not driver.maneuvers.active


def test_rejected_lane_change_can_only_increase_current_context_caution(monkeypatch):
    driver, lanes = driver_fixture(monkeypatch)
    driver.vehicle.position[1] = -1.0
    assert not driver.apply_intent(1, True, lanes[0])
    assert driver.careful
    assert not driver.apply_intent(1, False, lanes[0])
    assert driver.careful
    driver.careful = False
    assert not driver.apply_intent(0, True, lanes[1])
    assert not driver.careful


def test_stay_requests_do_not_reset_watchdog_and_checked_abort_has_cooldown(monkeypatch):
    driver, lanes = driver_fixture(monkeypatch)
    assert driver.apply_intent(1, False, lanes[0])
    driver.vehicle.position[1] = -1.0
    driver.maneuvers.update(driver, [])
    assert driver.apply_intent(0, True, lanes[1])
    for _ in range(3):
        driver.maneuvers.update(driver, [])
    assert driver.lane is lanes[0]
    assert driver.maneuvers.phase == "returning"
    assert driver.maneuvers.abort_count == 1
    assert driver.careful and driver.protection
    driver.vehicle.position[1] = 0.0
    driver.maneuvers.update(driver, [])
    assert driver.maneuvers.phase == "completed"
    assert not driver.apply_intent(1, False, lanes[0])
    assert driver.last_intent_reason == "recently_aborted_target"


def test_blocked_source_does_not_trigger_unchecked_abort_or_third_lane(monkeypatch):
    driver, lanes = driver_fixture(monkeypatch)
    assert driver.apply_intent(1, False, lanes[0])
    driver.vehicle.position[1] = -1.0
    monkeypatch.setattr(
        LaneCautionDriver,
        "lane_objects",
        lambda self, lane, objects: (2 if lane is lanes[0] else 80, 80, 0),
    )
    for _ in range(30):
        driver.maneuvers.update(driver, [])
    assert driver.lane is lanes[1]
    assert driver.maneuvers.phase == "blocked"
    assert driver.maneuvers.abort_count == 0
    assert driver.protection


def test_recovery_does_not_oscillate_if_return_itself_is_blocked(monkeypatch):
    driver, lanes = driver_fixture(monkeypatch)
    assert driver.apply_intent(1, False, lanes[0])
    driver.vehicle.position[1] = -1.0
    for _ in range(4):
        driver.maneuvers.update(driver, [])
    assert driver.maneuvers.phase == "returning"
    monkeypatch.setattr(LaneCautionDriver, "conflict_time", lambda self, objects, **kwargs: 0.2)
    for _ in range(50):
        driver.maneuvers.update(driver, [])
    assert driver.lane is lanes[0]
    assert driver.maneuvers.abort_count == 1
    assert driver.maneuvers.phase == "returning"


def test_recovery_is_disabled_when_collision_checks_are_unavailable(monkeypatch):
    driver, lanes = driver_fixture(monkeypatch)
    assert driver.apply_intent(1, False, lanes[0])
    driver.vehicle.position[1] = -1.0
    driver.protection = False
    for _ in range(30):
        driver.maneuvers.update(driver, [])
    assert driver.lane is lanes[1]
    assert driver.maneuvers.abort_count == 0
    assert driver.maneuvers.recovery_block_reason == "protection_unavailable"


def test_returning_keeps_caution_and_still_brakes_for_new_conflict(monkeypatch):
    driver, lanes = driver_fixture(monkeypatch)
    assert driver.apply_intent(1, False, lanes[0])
    driver.vehicle.position[1] = -1.0
    for _ in range(4):
        driver.maneuvers.update(driver, [])
    assert driver.lane is lanes[0]
    assert driver.apply_intent(0, False, lanes[0])
    assert driver.careful and not driver.last_caution_applied
    monkeypatch.setattr(LaneCautionDriver, "conflict_time", lambda self, objects, **kwargs: 0.2)
    action = driver.act()
    assert driver.last_features[2] == 0.0
    assert action[1] <= 0.0
    assert driver.maneuvers.phase == "returning"


def test_admission_rejects_lane_that_ends_before_available_route(monkeypatch):
    driver, lanes = driver_fixture(monkeypatch)
    lanes[1].length = 10
    driver.vehicle.navigation.next_ref_lanes = [Lane(0)]
    driver.vehicle.navigation.map = SimpleNamespace(
        road_network=SimpleNamespace(has_connection=lambda left, right: False)
    )
    assert not driver.apply_intent(1, False, lanes[0])
    assert driver.last_intent_reason == "route_lane_ends"
    assert driver.lane is lanes[0]


def test_lane_transition_forecast_includes_fast_approaching_rear_vehicle():
    driver = object.__new__(LaneCautionDriver)
    driver.vehicle = SimpleNamespace(
        position=np.array([0.0, 0.0]), speed=4.0, heading_theta=0.0, LENGTH=4.5, WIDTH=1.8
    )
    driver.lane = SimpleNamespace(local_coordinates=lambda position: (0, -3.5))
    driver.path_point = lambda distance: (np.array([distance, -3.5]), 0.0)
    rear_car = SimpleNamespace(
        position=np.array([-12.0, -3.5]),
        velocity=np.array([12.0, 0.0]),
        heading_theta=0.0,
        LENGTH=4.5,
        WIDTH=1.8,
    )
    assert math.isinf(driver.conflict_time([rear_car]))
    assert driver.conflict_time([rear_car], include_rear=True) < 3


def test_new_camera_instruction_cannot_interrupt_unfinished_maneuver(monkeypatch):
    driver, lanes = driver_fixture(monkeypatch)
    assert driver.apply_intent(1, False, lanes[0])
    driver.vehicle.position[1] = -3.0  # within old admission threshold, not yet completed
    assert not driver.apply_intent(1, False, lanes[1])
    assert driver.last_intent_reason == "maneuver_in_progress"
    assert driver.lane is lanes[1]


def test_teacher_query_does_not_advance_shared_supervisor(monkeypatch):
    driver, _ = driver_fixture(monkeypatch)
    monkeypatch.setattr(
        LaneCautionDriver, "act", lambda oracle: setattr(oracle, "last_caution_label", False)
    )
    before = driver.maneuvers.feedback().copy()
    assert teacher_recommendation(driver) == "A"
    assert driver.maneuvers.feedback() == before
    assert not driver.maneuvers.events


def test_feedback_is_optional_and_reveals_rejection_without_object_records():
    state = dict(speed_mps=0, current_lane=0, target_lane=1, lane_count=3)
    original = messages(None, state)[0]["content"][-1]["text"]
    assert "Execution status" not in original
    state.update(
        maneuver_state="blocked",
        no_progress_seconds=3.0,
        last_intent_reason="transition_unfinished_or_off_center",
        caution_applied=True,
    )
    managed = messages(None, state)[0]["content"][-1]["text"]
    assert managed.startswith(original)
    assert "blocked" in managed and "transition_unfinished_or_off_center" in managed


def physics_fixture():
    pytest.importorskip("metadrive")
    path = Path(__file__).resolve().parents[1] / "tools/safemetadrive_maneuver_fixture.py"
    module_spec = importlib.util.spec_from_file_location("maneuver_physics_fixture", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.fixture


@pytest.mark.metadrive
def test_physics_unfinished_lane_change_can_return_without_disabling_protection():
    fixture = physics_fixture()
    original = fixture(managed=False)
    managed = fixture(managed=True)
    assert original["final_progress"] < 0.2
    assert managed["aborts"] == managed["completions"] == 1
    assert managed["cost"] == 0
    assert managed["final_progress"] > 5
    assert managed["final_source_offset"] < 0.35


@pytest.mark.metadrive
def test_physics_blocked_return_waits_for_obstruction_to_clear():
    managed = physics_fixture()(managed=True, blocked_source=True)
    returns = [event for event in managed["events"] if event["phase"] == "returning"]
    assert len(returns) == 1
    assert returns[0]["t"] >= 6
    assert managed["cost"] == 0
    assert managed["completions"] == 1
