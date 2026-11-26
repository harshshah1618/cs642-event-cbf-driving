"""Tests for strictly observation-only SafeMetaDrive LiDAR driver."""

import inspect
import numpy as np
import pytest

from driving.safemetadrive.lidar_driver import (
    LidarOptionReserveDriver,
    decode_observation,
)


def _make_obs(
    *,
    dist_left: float = 5.25,
    dist_right: float = 5.25,
    heading_err: float = 0.0,
    speed_kmh: float = 25.0,
    lidar_fill: float = 1.0,
) -> np.ndarray:
    obs = np.zeros(259, dtype=np.float32)
    obs[0] = dist_left / 18.0
    obs[1] = dist_right / 18.0
    obs[2] = 0.5 + heading_err / np.pi
    obs[3] = (speed_kmh + 1.0) / 81.0
    obs[4] = 0.5
    obs[5] = 0.5
    obs[6] = 0.5
    obs[7] = 0.0
    obs[8] = 0.5
    # Straight road checkpoints
    obs[9] = 0.8
    obs[10] = 0.5
    obs[11] = 0.0
    obs[12] = 0.5
    obs[13] = 0.5
    obs[14] = 1.0
    obs[15] = 0.5
    obs[16] = 0.0
    obs[17] = 0.5
    obs[18] = 0.5
    obs[19:259] = lidar_fill
    return obs


def test_zero_simulator_peeking_contract():
    """Driver constructor must accept only an observation array, never vehicle/env."""
    sig = inspect.signature(LidarOptionReserveDriver.__init__)
    params = set(sig.parameters.keys())
    assert "vehicle" not in params
    assert "env" not in params
    obs = _make_obs()
    drv = LidarOptionReserveDriver(obs)
    assert not hasattr(drv, "vehicle")
    assert not hasattr(drv, "env")
    action = drv.act()
    assert len(action) == 2
    assert all(np.isfinite(x) and -1.0 <= x <= 1.0 for x in action)


def test_decode_observation_rejects_wrong_shape():
    with pytest.raises(ValueError):
        decode_observation(np.zeros(10, dtype=np.float32))


def test_road_boundary_guard_steers_away_from_edges():
    # Very close to left border -> must steer right (negative)
    obs_left = _make_obs(dist_left=1.1, dist_right=9.4, speed_kmh=25.0)
    drv_left = LidarOptionReserveDriver(obs_left)
    steer_left, _ = drv_left.act()
    assert steer_left < -0.2

    # Very close to right border -> must steer left (positive)
    obs_right = _make_obs(dist_left=9.4, dist_right=1.1, speed_kmh=25.0)
    drv_right = LidarOptionReserveDriver(obs_right)
    steer_right, _ = drv_right.act()
    assert steer_right > 0.2


def test_lidar_brakes_for_close_front_obstacle():
    obs = _make_obs(speed_kmh=35.0, lidar_fill=1.0)
    # Place obstacle 6 m straight ahead (ray 0 = 0 deg)
    obs[19] = 6.0 / 50.0
    drv = LidarOptionReserveDriver(obs, allow_lane_change=False)
    _, throttle = drv.act()
    assert throttle < -0.5


def test_option_reserve_activates_while_pending_delay():
    obs = _make_obs(speed_kmh=25.0, lidar_fill=1.0)
    # Obstacle 18 m ahead in middle lane (ray 0)
    obs[19] = 18.0 / 50.0
    drv = LidarOptionReserveDriver(obs, decision_delay_steps=15, option_reserve_m=6.5)
    drv.act()
    assert drv.pending_target is not None
    assert drv.reserve_active_steps == 1
