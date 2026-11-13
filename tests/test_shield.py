"""Safety-filter contract tests.

These tests pin the legacy research math, the production footprint/lookahead
extensions, fail-closed behavior, and the final SafetyMonitor gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from driving.config import ShieldConfig
from driving.env_types import VehicleState, WorldSnapshot
from driving.errors import ConfigError
from driving.safety_exports import InflatedCBFShield, SafetyMonitor, SafetyStatus, ShieldResult


def _snap(ego=None, others=None):
    return WorldSnapshot(
        t=0.0,
        ego=ego or VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0),
        others=list(others or []),
        route_length=200.0,
    )


def _action(ax=1.0, ay=0.2):
    return np.array([ax, ay], dtype=np.float32)


def _legacy_filter(
    action,
    snap,
    *,
    standoff=6.0,
    headway_s=1.1,
    class_k=2.5,
    lane_half_width=1.85,
    gamma_sigma=2.8,
    eta_latency=1.0,
    vlm_delay_s=0.30,
    ax_min=-7.5,
    ax_max=3.5,
    sigma=0.0,
    vlm_pending=False,
):
    """Literal copy of the pre-hardening filter, for regression comparison."""

    ax_rl, ay = float(action[0]), float(action[1])
    lat = (
        eta_latency
        * (1.0 if vlm_pending else 0.0)
        * max(snap.ego.vx, 0.0)
        * vlm_delay_s
    )
    sig = min(gamma_sigma * max(sigma, 0.0), 10.0)
    infl = sig + lat
    ax_cap = ax_max
    h_min = 1e9
    for obs in snap.obstacles_ahead(lane_half_width):
        dx = obs.x - snap.ego.x - standoff
        h = dx - headway_s * (snap.ego.vx - obs.vx) - infl
        h_min = min(h_min, h)
        ax_safe = ((obs.vx - snap.ego.vx) + class_k * h) / headway_s
        ax_cap = min(ax_cap, ax_safe)
    ax = float(np.clip(min(ax_rl, ax_cap), ax_min, ax_max))
    residual = max(0.0, ax_rl - ax)
    return {
        "action": np.array([ax, ay], dtype=np.float32),
        "residual": residual,
        "overridden": ax < ax_rl - 1e-3,
        "inflation": infl,
        "h": float(h_min),
    }


def test_legacy_barrier_math_matches_hand_computed_value():
    # h = (50 - 0 - 6) - 1.1*(20 - 10) = 33
    # ax_safe = ((10 - 20) + 2.5*33) / 1.1 = 65.909...
    shield = InflatedCBFShield()
    snap = _snap(
        ego=VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0),
        others=[VehicleState(x=50.0, y=0.0, vx=10.0, vy=0.0)],
    )
    result = shield.filter(_action(ax=1.0, ay=0.25), snap)

    assert result.h == pytest.approx(33.0)
    assert result.inflation == pytest.approx(0.0)
    assert result.action[0] == pytest.approx(1.0)
    assert result.action[1] == pytest.approx(0.25)
    assert result.residual == pytest.approx(0.0)
    assert result.overridden is False
    assert result.feasible is True
    assert result.safe is True
    assert result.reason == "ok"
    assert result.obstacle_count == 1
    assert result.emergency is False


def test_default_behavior_matches_old_filter_formula():
    shield = InflatedCBFShield()
    snap = _snap(
        ego=VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0),
        others=[
            VehicleState(x=50.0, y=0.0, vx=15.0, vy=0.0),
            VehicleState(x=70.0, y=0.5, vx=12.0, vy=0.0),
        ],
    )
    action = _action(ax=3.0, ay=0.3)
    result = shield.filter(action, snap, sigma=0.5, vlm_pending=True)
    old = _legacy_filter(action, snap, sigma=0.5, vlm_pending=True)

    assert np.allclose(result.action, old["action"])
    assert result.residual == pytest.approx(old["residual"])
    assert result.overridden is old["overridden"]
    assert result.inflation == pytest.approx(old["inflation"])
    assert result.h == pytest.approx(old["h"])
    assert result.feasible is True
    assert result.safe is True

    # Also compare a case where the barrier actually overrides the request.
    close_snap = _snap(
        ego=VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0),
        others=[VehicleState(x=6.44, y=0.0, vx=20.0, vy=0.0)],
    )
    close_action = _action(ax=3.0, ay=0.0)
    close_result = shield.filter(close_action, close_snap)
    close_old = _legacy_filter(close_action, close_snap)
    assert np.allclose(close_result.action, close_old["action"])
    assert close_result.overridden is True
    assert close_result.residual == pytest.approx(close_old["residual"])
    assert close_result.h == pytest.approx(close_old["h"])


def test_lane_vs_footprint_obstacle_selection():
    snap = _snap(
        ego=VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0, width=2.0),
        others=[VehicleState(x=40.0, y=2.0, vx=20.0, vy=0.0, width=2.0)],
    )

    lane = InflatedCBFShield(
        config=ShieldConfig(lateral_mode="lane", lane_half_width=1.85)
    )
    lane_result = lane.filter(_action(), snap)
    assert lane_result.obstacle_count == 0
    assert math.isinf(lane_result.h)
    assert lane_result.safe is True

    footprint = InflatedCBFShield(
        config=ShieldConfig(lateral_mode="footprint", lateral_margin=0.35)
    )
    footprint_result = footprint.filter(_action(), snap)
    # (2.0 + 2.0)/2 + 0.35 = 2.35 > 2.0 -> lateral overlap.
    assert footprint_result.obstacle_count == 1
    assert math.isfinite(footprint_result.h)
    assert footprint_result.h == pytest.approx(
        40.0 - 6.0 - 1.1 * (20.0 - 20.0)
    )


def test_lookahead_includes_target_lane_obstacle_during_lane_change():
    ego = VehicleState(x=0.0, y=0.0, vx=20.0, vy=2.0)
    snap = _snap(
        ego=ego,
        others=[VehicleState(x=50.0, y=2.5, vx=20.0, vy=0.0)],
    )

    no_lookahead = InflatedCBFShield(
        config=ShieldConfig(lateral_mode="lane", lookahead_s=0.0)
    )
    assert no_lookahead.filter(_action(), snap).obstacle_count == 0

    lookahead = InflatedCBFShield(
        config=ShieldConfig(lateral_mode="lane", lookahead_s=0.6)
    )
    # Predicted y = 0 + clamp(2.0)*0.6 = 1.2; |2.5 - 1.2| = 1.3 < 1.85.
    result = lookahead.filter(_action(), snap)
    assert result.obstacle_count == 1
    assert math.isfinite(result.h)


def _assert_fail_closed(result, reason):
    assert result.emergency is True
    assert result.feasible is False
    assert result.safe is False
    assert result.reason == reason
    assert result.action[0] == pytest.approx(-7.5)
    assert result.action[1] == pytest.approx(0.0)


def test_nan_action_fails_closed():
    shield = InflatedCBFShield()
    snap = _snap(others=[VehicleState(x=50.0, y=0.0, vx=10.0)])
    result = shield.filter(np.array([np.nan, 0.0]), snap)
    _assert_fail_closed(result, "invalid_action")


def test_invalid_action_shape_fails_closed():
    shield = InflatedCBFShield()
    snap = _snap(others=[VehicleState(x=50.0, y=0.0, vx=10.0)])
    _assert_fail_closed(shield.filter(np.array([1.0]), snap), "invalid_action")
    _assert_fail_closed(
        shield.filter(np.array([[1.0, 0.0]]), snap), "invalid_action"
    )


def test_nan_sigma_fails_closed():
    shield = InflatedCBFShield()
    snap = _snap(others=[VehicleState(x=50.0, y=0.0, vx=10.0)])
    result = shield.filter(_action(), snap, sigma=float("nan"))
    _assert_fail_closed(result, "invalid_sigma")


def test_nan_ego_fails_closed():
    shield = InflatedCBFShield()
    snap = WorldSnapshot(
        t=0.0,
        ego=VehicleState(x=float("nan"), y=0.0, vx=20.0, vy=0.0),
        others=[],
        route_length=200.0,
    )
    result = shield.filter(_action(), snap)
    _assert_fail_closed(result, "invalid_ego")


def test_nan_obstacle_fails_closed():
    shield = InflatedCBFShield()
    snap = _snap(
        others=[VehicleState(x=50.0, y=0.0, vx=float("nan"), vy=0.0)]
    )
    result = shield.filter(_action(), snap)
    _assert_fail_closed(result, "invalid_obstacle")


def test_infeasible_barrier_applies_ax_min_without_claiming_safe():
    shield = InflatedCBFShield(ax_min=-7.5)
    snap = _snap(
        ego=VehicleState(x=0.0, y=0.0, vx=30.0, vy=0.0),
        others=[VehicleState(x=5.0, y=0.0, vx=0.0, vy=0.0)],
    )
    result = shield.filter(_action(ax=3.0), snap)

    assert result.feasible is False
    assert result.safe is False
    assert result.reason == "infeasible"
    assert result.emergency is False
    assert result.action[0] == pytest.approx(-7.5)
    assert result.residual >= 0.0


def test_min_gap_marks_uncovered_barrier_violation_unsafe():
    shield = InflatedCBFShield(min_gap_m=40.0)
    snap = _snap(
        ego=VehicleState(x=0.0, y=0.0, vx=20.0, vy=0.0),
        others=[VehicleState(x=50.0, y=0.0, vx=10.0, vy=0.0)],
    )
    # h = 33 < min_gap_m = 40; the requested ax is already inside the safe
    # set, so there is no active override to justify claiming safety.
    result = shield.filter(_action(ax=1.0), snap)
    assert result.h == pytest.approx(33.0)
    assert result.overridden is False
    assert result.feasible is True
    assert result.safe is False
    assert result.reason == "barrier_violated"


def test_pending_margin_false_removes_latency_term():
    pending = InflatedCBFShield(
        config=ShieldConfig(pending_margin=True, sigma_cap=10.0)
    )
    no_margin = InflatedCBFShield(
        config=ShieldConfig(pending_margin=False, sigma_cap=10.0)
    )
    assert pending.inflation(0.0, True, 20.0) == pytest.approx(
        1.0 * 20.0 * 0.30
    )
    assert no_margin.inflation(0.0, True, 20.0) == pytest.approx(0.0)

    snap = _snap(others=[VehicleState(x=50.0, y=0.0, vx=20.0)])
    result = no_margin.filter(_action(), snap, sigma=0.0, vlm_pending=True)
    assert result.inflation == pytest.approx(0.0)


def test_no_obstacles_returns_infinite_h_and_safe():
    shield = InflatedCBFShield()
    result = shield.filter(_action(ax=1.5, ay=-0.3), _snap(others=[]))

    assert math.isinf(result.h)
    assert result.obstacle_count == 0
    assert result.feasible is True
    assert result.safe is True
    assert result.reason == "ok"
    assert result.action[0] == pytest.approx(1.5)
    assert result.action[1] == pytest.approx(-0.3)
    assert result.residual == pytest.approx(0.0)
    assert result.overridden is False


def test_constructor_validates_config_and_allows_overrides():
    cfg = ShieldConfig(
        lateral_mode="footprint",
        lateral_margin=0.35,
        lookahead_s=0.6,
        standoff=7.0,
    )
    shield = InflatedCBFShield(config=cfg, standoff=8.0, lookahead_s=0.2)
    assert shield.lateral_mode == "footprint"
    assert shield.lateral_margin == pytest.approx(0.35)
    assert shield.lookahead_s == pytest.approx(0.2)
    assert shield.standoff == pytest.approx(8.0)
    # The caller's config is not mutated by construction/validation.
    assert cfg.standoff == pytest.approx(7.0)

    # Config can also be passed positionally.
    positional = InflatedCBFShield(cfg)
    assert positional.lateral_mode == "footprint"

    with pytest.raises(ConfigError):
        InflatedCBFShield(gamma_sigma=-0.1)
    with pytest.raises(ConfigError):
        InflatedCBFShield(config=ShieldConfig(eta_latency=-1.0))
    with pytest.raises(ConfigError):
        InflatedCBFShield(config=ShieldConfig(lateral_mode="bogus"))


def _unsafe_result(reason="barrier_violated"):
    return ShieldResult(
        action=np.array([-5.0, 0.0], dtype=np.float32),
        residual=2.0,
        overridden=True,
        inflation=0.0,
        h=-1.0,
        feasible=True,
        safe=False,
        reason=reason,
        obstacle_count=1,
        emergency=False,
    )


def _safe_result():
    return ShieldResult(
        action=np.array([1.0, 0.2], dtype=np.float32),
        residual=0.0,
        overridden=False,
        inflation=0.0,
        h=10.0,
        feasible=True,
        safe=True,
        reason="ok",
        obstacle_count=1,
        emergency=False,
    )


def test_monitor_does_not_latch_on_valid_infeasibility_by_default():
    """A run of infeasible-but-valid results must not freeze the vehicle."""

    monitor = SafetyMonitor(violation_threshold=3, max_invalid_ticks=10)
    safe = _safe_result()
    unsafe = _unsafe_result()

    assert monitor.observe(safe) == SafetyStatus.OK
    for _ in range(20):
        assert monitor.observe(unsafe) == SafetyStatus.DEGRADED

    # Degraded never escalates to the irreversible emergency stop unless
    # explicitly requested. The shield's own action is still authoritative.
    assert monitor.status == SafetyStatus.DEGRADED
    assert np.allclose(
        monitor.safe_action(np.array([2.0, 0.0]), unsafe),
        unsafe.action,
    )

    summary = monitor.summary()
    assert summary["status"] == "degraded"
    assert summary["last_reason"] == "barrier_violated"
    assert summary["counts"]["total"] == 21
    assert summary["counts"]["emergency"] == 0
    assert summary["violations"] == 20
    assert summary["latch_on_violation"] is False

    monitor.reset()
    assert monitor.status == SafetyStatus.OK
    assert monitor.summary()["counts"]["total"] == 0

    assert monitor.observe(safe) == SafetyStatus.OK
    assert np.allclose(
        monitor.safe_action(np.array([2.0, 0.0]), safe),
        safe.action,
    )


def test_monitor_opt_in_violation_latch_and_reset():
    monitor = SafetyMonitor(
        violation_threshold=3, max_invalid_ticks=10, latch_on_violation=True
    )
    safe = _safe_result()
    unsafe = _unsafe_result()

    assert monitor.observe(safe) == SafetyStatus.OK
    assert monitor.observe(unsafe) == SafetyStatus.DEGRADED
    assert monitor.observe(unsafe) == SafetyStatus.DEGRADED
    assert monitor.observe(unsafe) == SafetyStatus.EMERGENCY

    gated = monitor.safe_action(np.array([2.0, 0.0]), unsafe)
    assert np.allclose(gated, np.array([-7.5, 0.0]))

    summary = monitor.summary()
    assert summary["status"] == "emergency"
    assert summary["last_reason"] == "barrier_violated"
    assert summary["counts"]["total"] == 4
    assert summary["counts"]["emergency"] >= 1
    assert summary["latch_on_violation"] is True

    monitor.reset()
    assert monitor.status == SafetyStatus.OK


def test_monitor_invalid_tick_limit_latches_emergency():
    monitor = SafetyMonitor(violation_threshold=100, max_invalid_ticks=2)
    invalid = ShieldResult(
        action=np.array([-7.5, 0.0], dtype=np.float32),
        residual=0.0,
        overridden=True,
        inflation=0.0,
        h=float("-inf"),
        feasible=False,
        safe=False,
        reason="invalid_action",
        obstacle_count=0,
        emergency=True,
    )
    assert monitor.observe(invalid) == SafetyStatus.DEGRADED
    assert monitor.observe(invalid) == SafetyStatus.EMERGENCY
    assert np.allclose(
        monitor.safe_action(np.array([0.0, 0.0]), invalid),
        np.array([-7.5, 0.0]),
    )
