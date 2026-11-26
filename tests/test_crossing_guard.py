"""Crossing-traffic guard, LiDAR backup trigger and verified hold release."""

import math

import numpy as np
from driving.safemetadrive.lidar_driver import (
    LIDAR_RANGE,
    NUM_LIDAR,
    CrossingTrafficGuard,
    decode_observation,
    lidar_clusters,
)


def _obs(points, speed_mps=5.0):
    """259-dim observation with LiDAR returns at the given ego-frame points (x fwd, y left)."""
    obs = np.zeros(259, dtype=np.float32)
    obs[0] = obs[1] = 5.25 / 18.0
    obs[2] = 0.5
    obs[3] = (speed_mps * 3.6 + 1.0) / 81.0
    obs[4] = 0.5
    obs[8] = 0.5
    obs[9] = obs[14] = 0.8
    obs[10] = obs[15] = 0.5
    lidar = np.ones(NUM_LIDAR)
    for x, y in points:
        for dy in np.linspace(-0.9, 0.9, 10):  # an object about 2 m wide
            py = y + dy
            ang = math.atan2(py, x) % (2 * math.pi)
            i = int(round(ang / (2 * math.pi / NUM_LIDAR))) % NUM_LIDAR
            lidar[i] = min(lidar[i], math.hypot(x, py) / LIDAR_RANGE)
    obs[19:] = lidar
    return obs


def test_clusters_find_objects():
    cents = lidar_clusters(decode_observation(_obs([(15.0, 6.0), (20.0, -8.0)])))
    assert len(cents) == 2


def test_guard_yields_to_converging_car_from_left():
    guard = CrossingTrafficGuard()
    v, dt = 5.0, 0.1
    yielded = False
    # A car 16 m ahead, 14 m left, driving toward our path at 6 m/s (ground frame).
    for k in range(10):
        t = k * dt
        x, y = 16.0 - v * t, 14.0 - 6.0 * t
        yielded = guard.update(decode_observation(_obs([(x, y)], speed_mps=v))) or yielded
    assert yielded


def test_guard_ignores_static_roadside_object():
    guard = CrossingTrafficGuard()
    v, dt = 5.0, 0.1
    for k in range(10):
        x = 25.0 - v * k * dt
        assert not guard.update(decode_observation(_obs([(x, 6.0)], speed_mps=v)))
    assert guard.static_objects(), "a parked object should be tracked as static"


def test_guard_ignores_parallel_traffic():
    guard = CrossingTrafficGuard()
    for _ in range(10):  # same speed, adjacent lane: constant relative position
        assert not guard.update(decode_observation(_obs([(6.0, 3.5)], speed_mps=8.0)))


def test_verified_release_needs_confident_proceed():
    from driving.irregular.escalation import Escalation, EscalationConfig

    class FakeVLM:
        def __init__(self, keep):
            self.keep = keep

            self.asked = 0

        def __call__(self, *a):
            self.asked += 1
            return ("stop" if self.asked == 1 else "continue"), 0.1

        def verify_hold(self, *a):
            return self.keep, 0.1

    class Driver:
        hold = careful = False
        planner = "rule"
        interventions = 0

        def apply_intent(self, *a):
            return True

    for keep, released in ((0.6, False), (0.4, True)):
        esc = Escalation(EscalationConfig(trigger="poll", verified_release=True, poll_period=0.3),
                         vlm=FakeVLM(keep))
        drv = Driver()
        cap = lambda: (None, dict(speed_mps=0.0, lane_count=2, current_lane=1))  # noqa: E731
        for step in range(200):
            esc.step(step * 0.1, drv, None, cap, step % 3 == 0)
        assert drv.hold is not released, (keep, [(c["action"], c.get("applied")) for c in esc.calls])
