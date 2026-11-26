"""Unit tests covering all 4 next-level upgrades (strictly local CPU)."""

import numpy as np

from driving.irregular.patchcore import local_contrast_metrics
from driving.safemetadrive.slip_adapter import SlipHistoryFrictionEstimator
from driving.safemetadrive.unified import UnifiedDrivingSystem
from driving.safemetadrive.viability import (
    option_viability_reserve_m,
    project_patch_to_lidar_distance,
)
from tests.test_lidar_driver import _make_obs


def test_upgrade3_viability_reserve_grows_with_speed_delay_and_wetness():
    r_slow_dry = option_viability_reserve_m(8.0, delay_s=0.3, friction_scale=1.0)
    r_fast_dry = option_viability_reserve_m(25.0, delay_s=0.8, friction_scale=1.0)
    r_fast_wet = option_viability_reserve_m(18.0, delay_s=0.8, friction_scale=0.30)
    assert r_fast_dry > r_slow_dry
    assert r_fast_wet >= r_slow_dry


def test_upgrade2_local_contrast_separates_rain_from_local_hazard():
    # Uniform rain: high everywhere -> low contrast ratio (~1.05)
    rain_map = np.full((18, 32), 38.0, dtype=np.float64)
    rain_map[5, 10] = 40.0
    m_rain = local_contrast_metrics(rain_map)
    assert m_rain["contrast_ratio"] < 1.2

    # Localized flagger on dry road: low median (~18), sharp local peak (48) -> high contrast (>2.5)
    flagger_map = np.full((18, 32), 18.0, dtype=np.float64)
    flagger_map[9, 24] = 48.0
    m_flag = local_contrast_metrics(flagger_map)
    assert m_flag["contrast_ratio"] > 2.5
    assert 0.70 < m_flag["cx_norm"] < 0.85


def test_upgrade1_patch_to_lidar_projection_and_stop_wall():
    obs = _make_obs(speed_kmh=30.0, lidar_fill=1.0)
    # Place obstacle straight ahead (ray 0 = 0 deg) at 20 m
    obs[19] = 20.0 / 50.0
    dist = project_patch_to_lidar_distance(obs, cx_norm=0.5)
    assert abs(dist - 20.0) < 0.5

    # Stop wall at 5 m ahead must force strong braking
    drv = UnifiedDrivingSystem(obs)
    drv.set_grounded_constraints(stop_wall_s=5.0)
    _, throttle = drv.act()
    assert throttle < -0.5


def test_upgrade4_slip_history_detects_slippery_road():
    est = SlipHistoryFrictionEstimator()
    # Simulate commanding full brake (-1.0) but only decelerating at 1.2 m/s^2 (wet/icy!)
    v_kmh = 36.0
    for _ in range(8):
        obs = _make_obs(speed_kmh=v_kmh)
        mu = est.update(obs, last_throttle=-1.0)
        v_kmh -= 0.12 * 3.6  # only 1.2 m/s^2 decel instead of ~5.5 m/s^2
    assert mu < 0.45
