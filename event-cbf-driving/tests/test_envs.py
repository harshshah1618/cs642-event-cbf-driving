"""Environment contract, determinism, and baseline-parity tests."""

from __future__ import annotations

import math
from typing import Any, List

import numpy as np
import pytest
from event_cbf_driving.config import SCENARIO_NAMES, ScenarioConfig
from event_cbf_driving.env import (
    AX_HIGH,
    AY_LOW,
    CarlaAdapterStub,
    DenseHighwayEnv,
    DrivingEnv,
    HighwayEnvAdapter,
    MockCarlaAdapter,
    SkillLaneEnv,
    TwoTrapGauntletEnv,
    VehicleState,
    WorldSnapshot,
    accel_to_highway_action,
    list_scenarios,
    make_env,
    scenario_defaults,
    validate_action,
    validate_snapshot,
    validate_vehicle,
)
from event_cbf_driving.env.skill_lane import LANE_Y
from event_cbf_driving.errors import ConfigError, ValidationError
from gymnasium import spaces


def _factory(name: str):
    return lambda: make_env(name)


ENV_FACTORIES = [
    pytest.param(_factory("two_trap"), id="two_trap"),
    pytest.param(_factory("merge"), id="merge"),
    pytest.param(_factory("dense_highway"), id="dense_highway"),
    pytest.param(lambda: SkillLaneEnv(randomize=False), id="skill_lane"),
]


def _trajectory(env, steps: int = 6) -> List[Any]:
    trace = []
    action = np.array([0.0, 0.0], dtype=np.float32)
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)
        snap = info["snapshot"]
        trace.append(
            (
                round(float(obs[0]), 9),
                round(float(obs[1]), 9),
                round(float(obs[2]), 9),
                round(float(snap.ego.x), 9),
                round(float(snap.ego.y), 9),
                tuple((round(float(o.x), 9), round(float(o.vx), 9)) for o in snap.others),
            )
        )
        if terminated or truncated:
            break
    return trace


def test_registry_is_strict_and_supports_scenario_config():
    assert list_scenarios() == list(SCENARIO_NAMES)
    assert scenario_defaults("two_trap")["route_length"] == pytest.approx(200.0)
    assert scenario_defaults("merge")["route_length"] == pytest.approx(180.0)
    assert scenario_defaults("dense_highway")["n_vehicles"] == 8

    for name in SCENARIO_NAMES:
        env = make_env(name)
        assert isinstance(env, DrivingEnv)
        assert env.SCENARIO_NAME == name
        env.close()

    for bad in ("gauntlet", "two", "dense", "highway", "skill", "two_trap_gauntlet", "unknown"):
        with pytest.raises(KeyError):
            make_env(bad)

    cfg = ScenarioConfig(
        name="two_trap",
        route_length=123.0,
        duration=5.0,
        seed=7,
        randomize=True,
        params={"stalled_x_range": (60.0, 80.0)},
    )
    env = make_env("two_trap", cfg)
    assert env.route_length == pytest.approx(123.0)
    assert env.duration == pytest.approx(5.0)
    _, info = env.reset(seed=7)
    assert 60.0 <= info["params"]["stalled_x"] <= 80.0

    with pytest.raises(ConfigError):
        make_env("merge", ScenarioConfig(name="two_trap"))
    with pytest.raises(ConfigError):
        make_env("two_trap", config="not a config")


@pytest.mark.parametrize("factory", ENV_FACTORIES)
def test_reset_step_snapshot_contract(factory):
    env = factory()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray) and obs.shape == (3,)
    assert np.all(np.isfinite(obs))
    for key in ("scenario", "params", "snapshot", "crashed", "route_pct"):
        assert key in info
    assert isinstance(info["snapshot"], WorldSnapshot)
    validate_snapshot(info["snapshot"])

    snap = env.snapshot()
    assert isinstance(snap, WorldSnapshot)
    assert validate_snapshot(snap) is snap
    assert snap.info["params"] == info["params"]

    out = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert isinstance(out, tuple) and len(out) == 5
    obs2, reward, terminated, truncated, info2 = out
    assert obs2.shape == (3,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    for key in ("snapshot", "crashed", "route_pct", "scenario", "params"):
        assert key in info2
    validate_snapshot(info2["snapshot"])
    assert info2["scenario"] == env.SCENARIO_NAME
    assert math.isfinite(info2["route_pct"])
    env.close()  # no-op must be safe


@pytest.mark.parametrize("factory", ENV_FACTORIES)
@pytest.mark.parametrize(
    "bad_action",
    [
        np.array([np.nan, 0.0]),
        np.array([0.0, np.inf]),
        np.array([1.0]),
        np.zeros((2, 2)),
        [[0.0, 0.0, 0.0]],
    ],
)
def test_action_validation_rejects_bad_actions(factory, bad_action):
    env = factory()
    env.reset(seed=0)
    before = env.snapshot()
    with pytest.raises(ValidationError):
        env.step(bad_action)
    after = env.snapshot()
    assert after.t == pytest.approx(before.t)
    assert after.ego.x == pytest.approx(before.ego.x)
    assert after.ego.y == pytest.approx(before.ego.y)

    with pytest.raises(ValidationError):
        validate_action(bad_action)


def test_validation_helpers_reject_bad_snapshots_and_vehicles():
    bad_vehicle = VehicleState(x=float("nan"), y=0.0, vx=1.0)
    with pytest.raises(ValidationError):
        validate_vehicle(bad_vehicle)
    with pytest.raises(ValidationError):
        validate_vehicle({"x": 0.0})
    bad_snapshot = WorldSnapshot(
        t=0.0,
        ego=bad_vehicle,
        others=[],
        route_length=200.0,
    )
    with pytest.raises(ValidationError):
        validate_snapshot(bad_snapshot)

    action = validate_action([1, -2])
    assert action.shape == (2,)
    assert action[0] == pytest.approx(1.0)
    assert action[1] == pytest.approx(-2.0)


@pytest.mark.parametrize("name", ["two_trap", "merge", "dense_highway"])
def test_deterministic_randomization_same_seed_and_different_seed(name):
    cfg = ScenarioConfig(name=name, randomize=True, seed=11)

    env_a = make_env(name, cfg)
    _, info_a = env_a.reset(seed=123)
    params_a = info_a["params"]
    trace_a = _trajectory(env_a)

    env_b = make_env(name, cfg)
    _, info_b = env_b.reset(seed=123)
    params_b = info_b["params"]
    trace_b = _trajectory(env_b)

    assert params_a == params_b
    assert trace_a == trace_b

    # Repeating the same seed on the *same* env must also reproduce params.
    _, info_a2 = env_a.reset(seed=123)
    assert info_a2["params"] == params_a

    env_c = make_env(name, cfg)
    _, info_c = env_c.reset(seed=124)
    assert info_c["params"] != params_a


def test_randomize_false_preserves_published_two_trap_defaults():
    env = TwoTrapGauntletEnv()
    _, info = env.reset(seed=0)
    params = info["params"]
    assert params["stalled_x"] == pytest.approx(75.0)
    assert params["cut_in_gap"] == pytest.approx(23.0)
    assert params["cut_in_speed"] == pytest.approx(9.5)
    assert params["spawn_x"] == pytest.approx(0.0)
    assert params["ego_vx"] == pytest.approx(20.0)

    snap = env.snapshot()
    assert snap.ego.x == pytest.approx(0.0)
    assert snap.ego.y == pytest.approx(LANE_Y[0])
    assert snap.ego.vx == pytest.approx(20.0)
    assert len(snap.others) == 1
    assert snap.others[0].x == pytest.approx(75.0)
    assert snap.others[0].vx == pytest.approx(0.0)

    # Cut-in trigger is x >= 92 and spawns exactly 23 m ahead at 9.5 m/s.
    env._ego = VehicleState(x=91.0, y=LANE_Y[1], vx=20.0)
    env.step(np.array([0.0, 0.0]))
    assert env._cut_in_spawned is False  # x = 91.4
    env.step(np.array([0.0, 0.0]))
    env.step(np.array([0.0, 0.0]))  # x >= 92
    assert env._cut_in_spawned is True
    cut = env.snapshot().others[-1]
    assert env.snapshot().ego.x >= 92.0
    assert cut.x - env.snapshot().ego.x == pytest.approx(23.0)
    assert cut.vx == pytest.approx(9.5)
    assert cut.y == pytest.approx(LANE_Y[1])


def test_default_merge_and_dense_are_unchanged():
    merge = make_env("merge")
    _, info = merge.reset(seed=0)
    snap = merge.snapshot()
    assert info["params"]["lead_gap"] == pytest.approx(55.0)
    assert info["params"]["lead_speed"] == pytest.approx(12.0)
    assert [o.x for o in snap.others] == pytest.approx([55.0, 40.0, 70.0, 30.0])
    assert [o.vx for o in snap.others] == pytest.approx([12.0, 18.0, 17.0, 14.0])
    assert [o.y for o in snap.others] == pytest.approx(
        [LANE_Y[0], LANE_Y[1], LANE_Y[1], -3.5]
    )

    dense = make_env("dense_highway")
    _, info = dense.reset(seed=0)
    snap = dense.snapshot()
    assert len(snap.others) == 8
    assert [o.x for o in snap.others] == pytest.approx(
        [45.0, 75.0, 110.0, 145.0, 35.0, 60.0, 95.0, 130.0]
    )
    assert [o.vx for o in snap.others] == pytest.approx(
        [11.0, 12.5, 11.0, 12.5, 18.0, 10.5, 17.5, 16.0]
    )
    assert info["params"]["positions"] == pytest.approx(
        [45.0, 75.0, 110.0, 145.0, 35.0, 60.0, 95.0, 130.0]
    )


@pytest.mark.parametrize("factory", ENV_FACTORIES)
def test_snapshot_is_detached_from_live_state(factory):
    env = factory()
    env.reset(seed=0)
    snap = env.snapshot()
    live = env.snapshot()
    snap.ego.x = 1e9
    snap.ego.y = -1e9
    snap.ego.vx = float("nan")
    if snap.others:
        snap.others[0].x = -1e9
        snap.others[0].vx = -1e9
        snap.others.append(VehicleState(x=1e9, y=1e9, vx=1e9))
    if isinstance(snap.info.get("params"), dict):
        snap.info["params"]["mutated"] = True

    fresh = env.snapshot()
    assert fresh.ego.x == pytest.approx(live.ego.x)
    assert fresh.ego.y == pytest.approx(live.ego.y)
    assert fresh.ego.vx == pytest.approx(live.ego.vx)
    assert len(fresh.others) == len(live.others)
    for got, expected in zip(fresh.others, live.others, strict=False):
        assert got.x == pytest.approx(expected.x)
        assert got.vx == pytest.approx(expected.vx)
    assert "mutated" not in fresh.info.get("params", {})


def test_crash_termination_and_route_metrics():
    crash_env = TwoTrapGauntletEnv()
    crash_env.reset(seed=0)
    crash_env._ego = VehicleState(x=74.9, y=LANE_Y[0], vx=0.0)
    _, reward, terminated, truncated, info = crash_env.step(np.array([0.0, 0.0]))
    assert info["crashed"] is True
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info["snapshot"].crashed is True
    assert 0.0 <= info["route_pct"] <= 100.0

    for name in ("two_trap", "merge", "dense_highway"):
        env = make_env(name, route_length=0.2)
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
        assert terminated is True
        assert truncated is False
        assert info["crashed"] is False
        assert info["route_pct"] == pytest.approx(100.0)
        assert reward == pytest.approx(1.0)

    skill = SkillLaneEnv(episode_seconds=0.06, randomize=False)
    skill.reset(seed=0)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, info = skill.step(np.array([0.0, 0.0]))
        assert terminated is False
    assert truncated is True
    assert math.isfinite(info["route_pct"])


def test_dense_n_vehicles_is_explicit_and_capped_with_warning():
    with pytest.warns(UserWarning, match="capping"):
        env = DenseHighwayEnv(n_vehicles=10)
    assert env.requested_n_vehicles == 10
    assert env.n_vehicles == DenseHighwayEnv.MAX_VEHICLES
    env.reset(seed=0)
    assert len(env.snapshot().others) == DenseHighwayEnv.MAX_VEHICLES

    small = DenseHighwayEnv(n_vehicles=3)
    small.reset(seed=0)
    assert small.n_vehicles == 3
    assert [o.x for o in small.snapshot().others] == pytest.approx([45.0, 75.0, 110.0])

    empty = DenseHighwayEnv(n_vehicles=0)
    empty.reset(seed=0)
    assert empty.snapshot().others == []
    with pytest.raises(ValidationError):
        DenseHighwayEnv(n_vehicles=-1)


def test_skill_lane_uses_constructor_target_speed():
    env = SkillLaneEnv(target_lane=1, target_speed=15.0, randomize=False)
    _, info = env.reset(seed=0)
    assert env.snapshot().ego.y == pytest.approx(LANE_Y[1])
    assert env.snapshot().ego.vx == pytest.approx(15.0)
    assert info["params"]["target_speed"] == pytest.approx(15.0)

    random_env = SkillLaneEnv(target_lane=0, target_speed=15.0, randomize=True)
    _, info = random_env.reset(seed=3)
    assert info["params"]["target_speed"] == pytest.approx(15.0)
    # Cruising samples are centred on the constructor target.
    assert 7.0 <= info["params"]["vx"] <= 17.0
    _, _, _, _, step_info = random_env.step(np.array([0.0, 0.0]))
    assert step_info["scenario"] == "skill_lane"
    assert "snapshot" in step_info


def test_highway_scaling_uses_asymmetric_bounds_and_mapping_order():
    high = accel_to_highway_action(AX_HIGH, AY_LOW)
    assert high == pytest.approx([1.0, -1.0])
    low = accel_to_highway_action(-6.0, 5.0)
    assert low == pytest.approx([-1.0, 1.0])
    zero = accel_to_highway_action(0.0, 0.0)
    assert zero == pytest.approx([0.0, 0.0])
    # Positive ax reaches full scale at AX_HIGH, not at the old symmetric 6.0.
    assert accel_to_highway_action(AX_HIGH, 0.0)[0] == pytest.approx(1.0)
    assert accel_to_highway_action(AX_HIGH / 2.0, 0.0)[0] == pytest.approx(0.5)
    assert accel_to_highway_action(-3.0, 0.0)[0] == pytest.approx(-0.5)


class _FakeVehicle:
    LENGTH = 5.0
    WIDTH = 2.0

    def __init__(self, x, y, speed, lane):
        self.position = np.array([x, y], dtype=float)
        self.speed = float(speed)
        self.heading = 0.0
        self.lane_index = ("a", "b", lane)


class _FakeRoad:
    def __init__(self, vehicles):
        self.vehicles = vehicles


class _FakeHighwayEnv:
    def __init__(self):
        self.ego = _FakeVehicle(10.0, 0.0, 20.0, 0)
        self.other = _FakeVehicle(40.0, 4.0, 15.0, 1)
        self.road = _FakeRoad([self.ego, self.other])
        self.unwrapped = self
        self.vehicle = self.ego
        self.time = 0.0
        self.last_action = None
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.time = 0.0
        return np.zeros(3, dtype=np.float32), {"crashed": False}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.time += 0.02
        self.ego.position = self.ego.position + np.array([0.4, 0.0])
        return np.zeros(3, dtype=np.float32), 1.0, False, False, {"crashed": False}

    def close(self):
        self.closed = True


def test_highway_adapter_contract_and_snapshot_from_road():
    native = _FakeHighwayEnv()
    cfg = ScenarioConfig(name="merge", route_length=123.0, duration=10.0)
    adapter = HighwayEnvAdapter(native, cfg)
    assert isinstance(adapter, DrivingEnv)
    assert adapter.route_length == pytest.approx(123.0)

    obs, info = adapter.reset(seed=0)
    assert obs.shape == (3,)
    for key in ("scenario", "params", "snapshot", "crashed", "route_pct"):
        assert key in info
    snap = adapter.snapshot()
    assert snap.ego.x == pytest.approx(10.0)
    assert len(snap.others) == 1
    assert snap.others[0].x == pytest.approx(40.0)
    assert snap.others[0].y == pytest.approx(4.0)
    assert snap.route_length == pytest.approx(123.0)

    out = adapter.step(np.array([AX_HIGH, AY_LOW], dtype=np.float32))
    assert len(out) == 5
    assert native.last_action == pytest.approx([1.0, -1.0])
    obs2, reward, terminated, truncated, info2 = out
    assert info2["snapshot"].ego.x == pytest.approx(10.4)
    assert info2["scenario"] == "merge"
    assert info2["crashed"] is False
    adapter.close()
    assert native.closed is True


def test_mock_carla_adapter_round_trip_and_real_stub_does_not_fake_carla():
    assert CarlaAdapterStub.IS_REAL_CARLA is False
    assert CarlaAdapterStub.REQUIRES_SERVER is True
    assert CarlaAdapterStub.REQUIRES_CARLA_PYTHON_API is True
    assert MockCarlaAdapter.IS_REAL_CARLA is False
    assert MockCarlaAdapter.IS_MOCK is True
    assert MockCarlaAdapter.REQUIRES_SERVER is False

    stub = CarlaAdapterStub()
    with pytest.raises(NotImplementedError, match="connect"):
        stub.reset(seed=0)
    with pytest.raises(NotImplementedError, match="connect"):
        stub.step(np.array([0.0, 0.0]))

    native = make_env("two_trap")
    mock = MockCarlaAdapter(native)
    obs, info = mock.reset(seed=0)
    assert info["scenario"] == "two_trap"
    assert isinstance(info["snapshot"], WorldSnapshot)
    out = mock.step(np.array([0.0, 0.0], dtype=np.float32))
    assert len(out) == 5
    assert validate_snapshot(mock.snapshot()) is not None
    # Snapshot detachment still holds through the adapter.
    snap = mock.snapshot()
    snap.ego.x = 1e9
    assert mock.snapshot().ego.x != pytest.approx(1e9)
    mock.close()
