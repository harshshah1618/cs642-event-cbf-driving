"""Physical-model assumptions, delay jitter and event/calibration contracts."""

from __future__ import annotations

import numpy as np
import pytest
from driving.assets import asset_directory
from driving.config import RouterConfig, ScenarioConfig, ShieldConfig
from driving.env_types import VehicleState, WorldSnapshot
from driving.errors import ConfigError
from driving.router_check import sample_request
from driving.router_factory import create_router
from driving.safety import InflatedCBFShield


def test_jitter_reproducible_bounded_and_not_a_fixed_delay():
    router = create_router(
        RouterConfig(backend="scripted", latency_s=0.8, latency_jitter_s=0.3, seed=17)
    )
    first = [router.submit(sample_request()).latency_s for _ in range(20)]
    router.reset()
    second = [router.submit(sample_request()).latency_s for _ in range(20)]
    assert first == second and len(set(first)) > 10
    assert min(first) >= 0.5 and max(first) <= 1.1


def test_stopping_backup_starts_braking_before_short_headway_filter():
    snap = WorldSnapshot(0, VehicleState(0, 0, 25), [VehicleState(48, 0, 0)])
    old = InflatedCBFShield(config=ShieldConfig(inflate=False))
    new = InflatedCBFShield(config=ShieldConfig(inflate=False, braking_guard=True))
    requested = np.array([3.0, 0.0])
    a, b = old.filter(requested, snap), new.filter(requested, snap)
    assert a.action[0] > 0
    assert b.action[0] < 0 and b.braking_h < 0
    # An unsafe initial condition is reported, not relabelled safe.
    assert not b.safe


def test_stopping_backup_reports_unrecoverable_close_obstacle():
    snap = WorldSnapshot(0, VehicleState(0, 0, 25), [VehicleState(10, 0, 0)])
    shield = InflatedCBFShield(config=ShieldConfig(braking_guard=True))
    result = shield.filter(np.array([3.0, 0.0]), snap)
    assert result.braking_h < 0 and not result.braking_feasible
    assert result.action[0] == -7.5


def test_instant_extra_margin_cannot_be_claimed_physically_safe():
    snap = WorldSnapshot(0, VehicleState(0, 0, 20), [VehicleState(45, 0, 0)])
    shield = InflatedCBFShield(config=ShieldConfig(braking_guard=True, vlm_delay_s=2.0))
    result = shield.filter(np.array([1.0, 0.0]), snap, sigma=0, vlm_pending=True)
    assert result.h < 0 and not result.feasible
    assert result.braking_h > 0  # Separate physical envelope, no hidden margin violation.


def test_stopping_assumption_cannot_exceed_available_actuator():
    with pytest.raises(ConfigError, match="actuator"):
        ShieldConfig(braking_decel=9).validate()


def test_physical_events_validated_before_loading_engine():
    from driving.env_physics import MetaDriveAdapter

    for event in (
        {"at": -1, "vehicle_index": 0, "target_speed": 0},
        {"at": 1, "vehicle_index": 0, "target_y": 100},
    ):
        with pytest.raises(ConfigError):
            MetaDriveAdapter(
                ScenarioConfig("merge", simulator="metadrive", params={"traffic_events": [event]})
            )


def test_intended_lane_is_checked_before_lateral_motion_begins():
    snap = WorldSnapshot(
        0, VehicleState(0, 0, 20), [VehicleState(20, 4, 0)], info={"target_y": 4.0}
    )
    a = np.array([2.0, 1.0])
    without = InflatedCBFShield(config=ShieldConfig(lateral_mode="footprint"))
    with_intent = InflatedCBFShield(
        config=ShieldConfig(lateral_mode="footprint", intent_corridor=True)
    )
    assert without.filter(a, snap).action[0] > 0
    assert with_intent.filter(a, snap).action[0] < 0
    snap.info["target_y"] = float("nan")
    assert with_intent.filter(a, snap).emergency


@pytest.mark.metadrive
def test_traffic_event_changes_physical_motion_without_teleporting():
    pytest.importorskip("metadrive")
    from driving.env_registry import make_env

    cfg = ScenarioConfig(
        "merge",
        simulator="metadrive",
        duration=1.0,
        params={"traffic_events": [{"at": 0.1, "vehicle_index": 1, "target_y": 0.0}]},
    )
    env = make_env("merge", cfg)
    try:
        env.reset(seed=0)
        before = env.snapshot().others[1]
        for _ in range(30):
            _, _, _, _, info = env.step([0, 0])
        after = env.snapshot().others[1]
        assert info["plant"]["traffic_events_applied"] == [0]
        assert 0 < after.y < before.y
        assert after.x > before.x
    finally:
        env.close()


def test_new_intent_rearms_shield_once_without_same_skill_feedback_loop():
    from driving.config import StackConfig
    from driving.controller import DrivingStack

    from test_controller import FakeEnv, FakeRouter, FakeSkill, ScriptedShield

    bank = {n: FakeSkill(n) for n in ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")}
    config = StackConfig(rearm_on_skill_change=True, min_call_ticks=28, blend_ticks=24)
    result = DrivingStack(
        bank,
        FakeRouter(skill="OVERTAKE_LEFT", latency_s=0.1),
        ScriptedShield(mode="always_override"),
        config,
    ).run(FakeEnv(duration=1.5))
    assert result.router_calls == 2
    assert result.calls_applied == 2


def test_recorded_delays_repeat_last_and_reset():
    r = create_router(RouterConfig(latency_schedule_s=[0.2, 0.5]))
    assert [r.submit(sample_request()).latency_s for _ in range(3)] == [0.2, 0.5, 0.5]
    r.reset()
    assert r.submit(sample_request()).latency_s == 0.2
    with pytest.raises(ConfigError):
        RouterConfig(latency_schedule_s=[0.2], latency_jitter_s=0.1).validate()


def test_recorded_late_reply_waits_for_deadline_and_keeps_worker_busy():
    from driving.config import StackConfig
    from driving.controller import DrivingStack

    from test_controller import FakeEnv, FakeRouter, FakeSkill

    bank = {n: FakeSkill(n, sigma=10) for n in ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")}
    c = StackConfig(
        simulate_deadline_wait=True,
        max_pending_s=0.08,
        min_call_ticks=0,
        pending_margin_on_all_calls=True,
    )
    result = DrivingStack(bank, FakeRouter(skill="OVERTAKE_LEFT", latency_s=0.25), config=c).run(
        FakeEnv(duration=0.5)
    )
    assert result.pending_ticks > 0 and result.calls_applied == 0
    attempts = [x for x in result.calls if not x.get("suppressed")]
    assert len(attempts) == 2 and attempts[1]["t"] >= 0.25


@pytest.mark.metadrive
def test_braking_target_lane_regression_recovers_instead_of_colliding_or_freezing():
    pytest.importorskip("metadrive")
    from pathlib import Path

    from driving.config import load_config
    from driving.evaluation_runner import load_skill_bank, run_episode

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/metadrive-study.yaml")
    bank, _ = load_skill_bank(root / cfg.checkpoint_dir)
    cfg.router.latency_jitter_s = 0.2
    scenario = ScenarioConfig(
        "dense_highway",
        simulator="metadrive",
        randomize=True,
        duration=20,
        params={"traffic_events": [{"at": 1.0, "vehicle_index": 4, "target_speed": 0.0}]},
    )
    result = run_episode(
        "proposed",
        scenario,
        307,
        bank,
        create_router(cfg.router),
        stack_config=cfg.stack,
        shield_config=cfg.shield,
    )
    assert not result.crashed and not result.freeze
    assert result.route_pct > 90


def test_calibration_artifact_keeps_weights_fixed_and_validation_seeds_held_out():
    from pathlib import Path

    from driving.io_utils import read_json
    from driving.ppo_compat import load_skill, state_dict_fingerprint

    root = asset_directory(Path(__file__).resolve().parents[1] / "models/native")
    physical = asset_directory(Path(__file__).resolve().parents[1] / "models/physical")
    artifact = read_json(physical / "calibration.json")
    assert not set(artifact["train_seeds"]) & set(artifact["validation_seeds"])
    for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT"):
        original = load_skill(root / f"{name}.pt")
        calibrated = load_skill(physical / f"{name}.pt", strict=True)
        assert state_dict_fingerprint(original.net.state_dict()) == state_dict_fingerprint(
            calibrated.net.state_dict()
        )
