"""Real Bullet integration tests; install .[metadrive] to run these locally/CI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from driving.config import ScenarioConfig, load_config
from driving.env_base import DrivingEnv, validate_snapshot
from driving.env_registry import make_env
from driving.errors import ValidationError
from driving.evaluation import load_skill_bank, run_episode
from driving.io_utils import atomic_write_json
from driving.planner import ScriptedRouter
from driving.runtime import load_episode, replay_open_loop

pytestmark = [
    pytest.mark.metadrive,
    pytest.mark.skipif(
        importlib.util.find_spec("metadrive") is None, reason="optional MetaDrive not installed"
    ),
]
ROOT = Path(__file__).resolve().parents[1]


def config(name="two_trap", **kwargs):
    return ScenarioConfig(name, simulator="metadrive", **kwargs)


@pytest.mark.parametrize("name", ["two_trap", "merge", "dense_highway"])
def test_physical_environment_contract_lifecycle_and_seeded_initial_conditions(name):
    env = make_env(name, config(name, duration=0.3, randomize=True))
    assert isinstance(env, DrivingEnv)
    try:
        with pytest.raises(ValidationError, match="reset"):
            env.snapshot()
        first, info = env.reset(seed=2)
        params = info["params"]
        initial = env.snapshot()
        validate_snapshot(initial)
        initial.ego.x = 999
        assert env.snapshot().ego.x != 999
        assert env.env.config["decision_repeat"] == 1
        assert len(initial.others) >= 1
        for _ in range(10):
            obs, reward, term, trunc, info = env.step([0.0, 0.0])
            validate_snapshot(info["snapshot"])
            assert obs.shape == first.shape == (3,)
            assert len(info["plant"]["command"]) == 2
            assert np.isfinite(reward)
        assert env.snapshot().t == pytest.approx(0.2)
        assert env.snapshot().ego.x > 1
        with pytest.raises(ValidationError):
            env.step([np.nan, 0])
        env.reset(seed=2)
        assert env.snapshot().info["params"] == params
        env.reset(seed=3)
        assert env.snapshot().info["params"] != params
        for seed in (11, 23, 47):
            env.reset(seed=seed)
            validate_snapshot(env.snapshot())
    finally:
        env.close()
        env.close()


def test_requested_braking_is_realized_without_resetting_vehicle_state():
    env = make_env("two_trap", config(duration=1.2))
    try:
        env.reset(seed=0)
        for _ in range(10):
            env.step([0.0, 0.0])
        start = env.snapshot().ego.vx
        achieved = []
        for _ in range(40):
            _, _, _, _, info = env.step([-3.0, 0.0])
            achieved.append(info["plant"]["achieved_acceleration"][0])
        assert env.snapshot().ego.vx < start - 1.5
        assert -4.0 < np.mean(achieved[10:]) < -2.0
    finally:
        env.close()


def test_unshielded_vehicle_physically_collides_with_stalled_car():
    env = make_env("two_trap", config(duration=8))
    try:
        env.reset(seed=0)
        for _ in range(300):
            _, _, terminated, truncated, info = env.step([0.0, 0.0])
            if terminated or truncated:
                break
        assert info["crashed"]
        assert env.snapshot().crashed
    finally:
        env.close()


def test_lateral_acceleration_tracks_physical_request_without_speed_amplification():
    env = make_env("two_trap", config(duration=1.2))
    try:
        env.reset(seed=0)
        for _ in range(10):
            env.step([0.0, 0.0])
        achieved = []
        for _ in range(30):
            _, _, _, _, info = env.step([0.0, 1.0])
            achieved.append(info["plant"]["achieved_acceleration"][1])
        assert 0.6 < np.mean(achieved[10:]) < 1.5
    finally:
        env.close()


def test_three_layers_interact_in_real_physics_and_episode_replays(tmp_path):
    cfg = load_config(ROOT / "configs/metadrive.yaml")
    bank, _ = load_skill_bank(ROOT / "models/native")
    scenario = config(duration=20)
    result = run_episode(
        "proposed",
        scenario,
        0,
        bank,
        ScriptedRouter(),
        stack_config=cfg.stack,
        shield_config=cfg.shield,
    )
    assert not result.crashed and not result.freeze
    assert result.route_pct == 100
    assert result.calls_applied >= 2
    assert {"CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT"} <= set(result.skills)
    assert any(row["shield_override"] for row in result.interactions)
    assert any(row["router_pending"] for row in result.interactions)
    assert any(row["router_trigger"] == "shield" for row in result.interactions)
    assert any(0 < row["blend_alpha"] < 1 for row in result.interactions)
    assert result.plant_peak_lateral_jerk is not None
    assert result.acceleration_tracking_rmse is not None
    payload = result.to_dict()
    payload["scenario_config"] = {"name": "two_trap", "simulator": "metadrive", "duration": 20}
    path = tmp_path / "episode.json"
    atomic_write_json(path, payload)
    report = replay_open_loop(load_episode(path), tolerance=1e-4)
    assert report.ok, report.errors
