"""Reserve lifecycle and base-braking contracts; no claim of reachability."""

import math

import numpy as np
import pytest

from config import ShieldConfig
from env_types import VehicleState, WorldSnapshot
from maneuver_reserve import ManeuverReserveShield, geometric_reserve
from safety import InflatedCBFShield


def snap(t=0.0, speed=20.0, target=0.0):
    return WorldSnapshot(
        t=t,
        ego=VehicleState(0.0, 0.0, speed),
        others=[VehicleState(70.0, 0.0, 0.0)],
        info={
            "target_y": target,
            "maneuver_geometry": {
                "wheelbase": 2.7,
                "steering_limit_rad": math.radians(40),
            },
        },
    )


@pytest.mark.parametrize("mode", ["fixed_latched", "geometric_latched"])
def test_reserve_survives_stopping_and_same_target_then_releases(mode):
    shield = ManeuverReserveShield(ShieldConfig(), mode=mode)
    action = np.array([2.0, 1.0])
    shield.filter(action, snap(), vlm_pending=True)
    held = shield.braking_clearance
    assert held > shield.base_clearance
    shield.filter(action, snap(1.0, speed=0.0), vlm_pending=False)
    assert shield.braking_clearance == held
    shield.filter(action, snap(2.0, speed=0.0, target=4.0), vlm_pending=True)
    assert shield.braking_clearance == shield.base_clearance
    assert shield.transition and not shield.armed
    assert [e["event"] for e in shield.events] == ["arm", "release_on_changed_admitted_target"]


def test_speed_scaled_initial_unarmed_tick_and_stopped_tick():
    shield = ManeuverReserveShield(ShieldConfig(), mode="speed_scaled")
    action = np.zeros(2)
    shield.filter(action, snap(), vlm_pending=False)
    shield.filter(action, snap(1.0), vlm_pending=True)
    shield.filter(action, snap(2.0, speed=0.0), vlm_pending=True)
    assert shield.braking_clearance == shield.base_clearance


def test_releasing_extra_keeps_original_collision_braking():
    shield = ManeuverReserveShield(ShieldConfig(braking_guard=True), mode="fixed_latched")
    shield.filter(np.array([3.0, 0.0]), snap(), vlm_pending=True)
    s = snap(1.0, speed=10.0, target=4.0)
    s.others[0].x = 7.0
    result = shield.filter(np.array([3.0, 0.0]), s, vlm_pending=True)
    assert shield.braking_clearance == 2.0
    base = InflatedCBFShield(config=ShieldConfig(braking_guard=True))
    ordinary = base.filter(np.array([3.0, 0.0]), s, vlm_pending=True)
    assert result.action[0] < 0 and not result.braking_feasible
    np.testing.assert_array_equal(result.action, ordinary.action)
    assert result.braking_h == ordinary.braking_h


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_reserve_rejected(value):
    with pytest.raises(ValueError):
        ManeuverReserveShield(ShieldConfig(), reserve_m=value)
    with pytest.raises(ValueError):
        geometric_reserve(2.7, math.radians(40), scale=value)


def test_steering_changes_geometry_and_invalid_primitive_rejected():
    assert geometric_reserve(2.7, math.radians(20)) > geometric_reserve(2.7, math.radians(40))
    with pytest.raises(ValueError):
        geometric_reserve(2.7, math.pi)
    with pytest.raises(ValueError):
        geometric_reserve(2.7, math.radians(40), lateral_distance=100.0)
