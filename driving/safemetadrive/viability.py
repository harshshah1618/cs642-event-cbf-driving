"""Closed-form Option-Viability Barrier for asynchronous slow-planner driving.

Problem:
    Standard delayed-CBF margins inflate stopping distance to prevent collisions
    under sensor lag. Under asynchronous VLM planning (fresh LiDAR, delayed
    high-level maneuver), braking safely to a standstill right behind an obstacle
    causes *unaware drift into an unrecoverable state*: the vehicle has zero
    collisions, yet is kinematically pinned too close to steer into an adjacent lane.

Closed-form Option-Viability Distance:
    Given speed v, expected planner latency tau, lane width dy, lateral accel
    limit a_lat, max dry deceleration b_max, and estimated friction scale mu:

        b_eff       = max(0.8, b_max * mu)
        L_stop      = v^2 / (2 * b_eff)
        L_swerve    = v * sqrt(2 * dy / a_lat) + car_length
        d_viable    = v * tau + max(L_stop, L_swerve)
        reserve_m   = max(min_turn_gap, d_viable - L_stop)

    Maintained while a slow decision is pending (or lane is blocked without an
    admitted detour), and released immediately once a maneuver is accepted.
"""

from __future__ import annotations

import math


def swerve_distance_m(
    speed_mps: float,
    *,
    lane_width_m: float = 3.5,
    lateral_accel: float = 2.2,
    car_length_m: float = 4.5,
) -> float:
    """Minimum longitudinal distance to execute a lane-change arc of width lane_width_m."""
    v = max(0.0, float(speed_mps))
    t_lat = math.sqrt(2.0 * max(0.5, lane_width_m) / max(0.5, lateral_accel))
    return v * t_lat + car_length_m


def stopping_distance_m(
    speed_mps: float,
    *,
    max_brake: float = 4.2,
    friction_scale: float = 1.0,
) -> float:
    """Kinematic stopping distance under effective road friction scale mu."""
    v = max(0.0, float(speed_mps))
    b_eff = max(0.8, float(max_brake) * max(0.15, float(friction_scale)))
    return (v * v) / (2.0 * b_eff)


def option_viability_reserve_m(
    speed_mps: float,
    delay_s: float,
    *,
    lane_width_m: float = 3.5,
    lateral_accel: float = 2.2,
    max_brake: float = 4.2,
    friction_scale: float = 1.0,
    car_length_m: float = 4.5,
    min_turn_gap_m: float = 6.0,
    max_reserve_m: float = 28.0,
) -> float:
    """Closed-form extra longitudinal clearance beyond bare stopping distance.

    Guarantees that at reply arrival time t + delay_s, BOTH stopping safely AND
    initiating a lane-change swerve remain kinematically feasible.
    """
    v = max(0.0, float(speed_mps))
    tau = max(0.0, float(delay_s))
    mu = max(0.15, float(friction_scale))
    a_lat_eff = max(0.8, float(lateral_accel) * math.sqrt(mu))

    l_stop = stopping_distance_m(v, max_brake=max_brake, friction_scale=mu)
    l_swerve = swerve_distance_m(
        v, lane_width_m=lane_width_m, lateral_accel=a_lat_eff, car_length_m=car_length_m
    )
    # Distance needed so at t + tau we still have room to swerve OR finish braking
    d_viable = v * min(tau, 3.0) + max(l_stop, l_swerve)
    extra = d_viable - l_stop
    # Always preserve at least min_turn_gap_m when a decision is pending/blocked
    # so the vehicle never pins itself bumper-to-bumper even at v -> 0.
    return float(min(max_reserve_m, max(min_turn_gap_m, extra)))


def project_patch_to_lidar_distance(
    obs_259,
    cx_norm: float,
    *,
    camera_fov_deg: float = 65.0,
    window_deg: float = 10.0,
    default_distance_m: float = 32.0,
) -> float:
    """Project a normalized horizontal image coordinate cx_norm in [0, 1] onto 240 LiDAR rays.

    Camera convention: cx_norm = 0.0 is left edge (+FOV/2 CCW), cx_norm = 1.0 is right edge (-FOV/2).
    MetaDrive LiDAR convention: 240 rays, 1.5 deg step CCW from ego heading, range 50 m.
    """
    import numpy as np

    arr = np.asarray(obs_259, dtype=np.float64)
    if arr.shape != (259,):
        return float(default_distance_m)
    lidar_m = arr[19:259] * 50.0
    # Yaw angle in degrees CCW from ego forward (+ = left, - = right)
    yaw_deg = (0.5 - float(np.clip(cx_norm, 0.0, 1.0))) * float(camera_fov_deg)
    angles_deg = (np.arange(240, dtype=np.float64) * 1.5 + 180.0) % 360.0 - 180.0
    in_cone = np.abs(angles_deg - yaw_deg) <= float(window_deg)
    hits = lidar_m[in_cone]
    valid = hits[hits < 49.0]
    if valid.size == 0:
        return float(default_distance_m)
    return float(np.min(valid))
