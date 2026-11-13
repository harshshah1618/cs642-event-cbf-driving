"""Episode recording, validation, and open-loop replay tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from driving import DT
from driving.assets import asset_directory
from driving.config import RouterConfig, ScenarioConfig
from driving.errors import ValidationError
from driving.evaluation_runner import load_skill_bank, run_episode
from driving.io_utils import read_json
from driving.router_factory import create_router
from driving.runtime import Episode, load_episode, replay_open_loop, validate_episode, write_episode

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = asset_directory(ROOT / "models/native")
SCENARIO = ScenarioConfig(name="two_trap", seed=0)


@pytest.fixture(scope="module")
def episode_path(tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("runtime") / "two-trap-proposed.json"
    skills, _info = load_skill_bank(CHECKPOINTS)
    router = create_router(RouterConfig(backend="scripted", latency_s=0.3))
    result = run_episode("proposed", SCENARIO, 0, skills, router)
    return write_episode(target, result, scenario=SCENARIO)


def test_round_trip_is_strict_and_preserves_actions(episode_path: Path):
    raw = read_json(episode_path)  # strict JSON: rejects NaN/Infinity
    assert raw["recorded"] is True
    assert raw["scenario_config"]["name"] == "two_trap"
    assert len(raw["actions"]) == len(raw["xs"]) == len(raw["ys"])

    episode = load_episode(episode_path)
    assert isinstance(episode, Episode)
    assert episode.source == episode_path
    assert episode.ticks == len(episode.actions)
    assert episode.ticks > 0
    assert episode.scenario.name == "two_trap"
    assert episode.scenario.seed == 0
    assert episode.result.route_pct > 0.0


def test_open_loop_replay_reproduces_the_recorded_final_state(episode_path: Path):
    report = replay_open_loop(episode_path)

    assert report.ok, report.errors
    assert report.ticks == report.expected_ticks
    assert report.final_state_error <= 1e-6
    assert report.replayed_crashed == report.expected_crashed


def test_corrupt_action_log_is_rejected(episode_path: Path):
    payload = read_json(episode_path)
    payload["actions"] = payload["actions"][:-3]

    problems = validate_episode(payload)
    assert any("actions length" in problem for problem in problems)
    with pytest.raises(ValidationError, match="actions length"):
        Episode.from_payload(payload)


def test_legacy_episode_without_actions_loads_but_cannot_be_replayed(episode_path: Path):
    payload = read_json(episode_path)
    del payload["actions"]

    episode = Episode.from_payload(payload)
    assert episode.actions == []
    assert episode.ticks == len(episode.result.xs)
    with pytest.raises(ValidationError, match="no action log"):
        replay_open_loop(episode)


def test_non_finite_trace_is_rejected(episode_path: Path):
    payload = read_json(episode_path)
    payload["xs"][10] = float("nan")

    problems = validate_episode(payload)
    assert any("xs[10]" in problem for problem in problems)


def test_unknown_keys_are_rejected():
    payload = {
        "mode": "proposed",
        "scenario": "two_trap",
        "crashed": False,
        "route_pct": 100.0,
        "avg_speed": 20.0,
        "router_calls": 0,
        "peak_lateral_jerk": 1.0,
        "peak_sigma": 1.0,
        "freeze": False,
        "x": 1.0,
        "y": 0.0,
        "vx": 20.0,
        "end_time": 1.0,
        "xs": [0.0],
        "ys": [0.0],
        "skills": ["CRUISE_RIGHT"],
        "mystery": True,
    }
    problems = validate_episode(payload)
    assert any("unknown keys" in problem for problem in problems)


def test_samples_include_time_skill_and_estimated_speed(episode_path: Path):
    episode = load_episode(episode_path)
    samples = list(episode.samples())

    assert len(samples) == episode.ticks
    assert samples[0].t == pytest.approx(0.0)
    assert samples[3].t == pytest.approx(3 * DT)
    assert samples[0].skill
    assert samples[0].vx is not None
    assert math.isfinite(samples[0].vx)
    assert samples[-1].vx is not None


def test_write_episode_rejects_plain_dicts(tmp_path: Path):
    with pytest.raises(ValidationError, match="RolloutResult"):
        write_episode(tmp_path / "bad.json", {"mode": "proposed"})  # type: ignore[arg-type]
