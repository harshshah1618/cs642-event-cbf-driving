"""Independent component-mask contracts for the prospective physical factorial."""

import itertools

import numpy as np
import pytest

from config import ShieldConfig, StackConfig, StackMode
from controller import DrivingStack
from env_types import VehicleState, WorldSnapshot
from safety_exports import InflatedCBFShield, SafetyMonitor, SafetyStatus


def shield(cbf, braking, boundary=False):
    return InflatedCBFShield(config=ShieldConfig(
        cbf_constraint_enabled=cbf, braking_guard=braking,
        road_boundary_guard=boundary, gamma_sigma=0, pending_margin=False,
        finite_checks=True, fail_closed=True,
    ))


def snapshot(distance=27, *, y=0, vy=0):
    return WorldSnapshot(
        0, VehicleState(0, y, 20, vy), [VehicleState(distance, y, 0)], 200,
        info={"road_bounds": [-2, 6], "heading": 0},
    )


def test_cbf_off_braking_off_preserves_valid_requested_action():
    result = shield(False, False).filter(np.array([2., .25]), snapshot())
    # A dangerous scene remains dangerous; neither hidden CBF nor monitor brakes.
    assert result.h < 0
    np.testing.assert_allclose(result.action, [2, .25])
    assert not result.cbf_overridden and not result.braking_overridden
    assert result.braking_h is None
    monitor = SafetyMonitor()
    assert monitor.observe(result) == SafetyStatus.OK
    np.testing.assert_allclose(monitor.safe_action(result.action, result), [2, .25])


def test_stopping_backup_still_brakes_when_cbf_constraint_disabled():
    result = shield(False, True).filter(np.array([2., .25]), snapshot())
    assert result.action[0] < 0
    assert result.action[1] == pytest.approx(.25)
    assert result.braking_h is not None
    assert result.braking_overridden
    assert not result.cbf_overridden


def test_cbf_still_brakes_when_stopping_backup_disabled():
    result = shield(True, False).filter(np.array([2., .25]), snapshot())
    assert result.action[0] < 0
    assert result.cbf_overridden
    assert not result.braking_overridden
    assert result.braking_h is None


@pytest.mark.parametrize("cbf,braking", list(itertools.product([False, True], repeat=2)))
def test_common_boundary_guard_survives_all_longitudinal_masks(cbf, braking):
    scene = snapshot(y=-.8, vy=-1)
    requested = np.array([2., -2.])
    without = shield(cbf, braking).filter(requested, scene)
    with_boundary = shield(cbf, braking, boundary=True).filter(requested, scene)
    assert with_boundary.action[0] == pytest.approx(without.action[0])
    assert with_boundary.action[1] > without.action[1]
    assert with_boundary.lateral_overridden
    assert with_boundary.lateral_h is not None


@pytest.mark.parametrize("cbf,braking", list(itertools.product([False, True], repeat=2)))
def test_invalid_action_fail_closed_is_common_to_all_masks(cbf, braking):
    result = shield(cbf, braking, boundary=True).filter(np.array([np.nan, 0]), snapshot())
    assert result.emergency
    assert result.reason.startswith("invalid")
    monitor = SafetyMonitor(max_invalid_ticks=1)
    assert monitor.observe(result) == SafetyStatus.EMERGENCY
    np.testing.assert_allclose(monitor.safe_action(result.action, result), [-7.5, 0])


def test_valid_infeasibility_does_not_latch_extra_monitor_braking():
    result = shield(True, False).filter(np.array([2., .25]), snapshot(distance=10))
    assert not result.feasible and not result.emergency
    monitor = SafetyMonitor(violation_threshold=1)
    for _ in range(10):
        assert monitor.observe(result) == SafetyStatus.DEGRADED
        np.testing.assert_allclose(monitor.safe_action(result.action, result), result.action)
    assert monitor.emergency_count == 0


@pytest.mark.parametrize("cbf,braking", list(itertools.product([False, True], repeat=2)))
def test_matched_polling_mode_preserves_independent_component_masks(cbf, braking):
    instance = shield(cbf, braking, boundary=True)
    config = StackConfig(mode="fixed_rate_shielded", pending_margin=False)
    DrivingStack._configure_mode(instance, StackMode.FIXED_VLM, config,
                                 variant="fixed_rate_shielded")
    assert instance.enabled
    assert instance.cbf_constraint_enabled is cbf
    assert instance.braking_guard is braking
    assert instance.road_boundary_guard
    assert instance.inflation(100, True, 20) == 0
    assert config.use_blend and not config.hard_switch
