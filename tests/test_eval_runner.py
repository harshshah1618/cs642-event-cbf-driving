"""Production evaluation runner tests (offline, real checkpoints)."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from driving.assets import asset_directory
from driving.config import EVAL_MODES, EvalConfig, ProjectConfig, RouterConfig, ScenarioConfig
from driving.errors import ConfigError
from driving.evaluation_runner import (
    check_regressions,
    load_skill_bank,
    run_episode,
    run_evaluation,
)
from driving.io_utils import read_json
from driving.planner import ScriptedRouter
from driving.provenance import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = asset_directory(ROOT / "models/native")


@pytest.fixture(scope="module")
def skill_bank():
    skills, _info = load_skill_bank(CHECKPOINTS)
    return skills


def _config(
    tmp_path: Path,
    *,
    modes,
    seeds,
    repeats: int = 1,
    record_episodes: bool = False,
    checkpoint_dir: Path | None = None,
) -> ProjectConfig:
    config = ProjectConfig()
    config.checkpoint_dir = str(checkpoint_dir or CHECKPOINTS)
    config.router = RouterConfig(backend="scripted", latency_s=0.3, fail_on_fallback=True)
    config.eval = EvalConfig(
        scenarios=[ScenarioConfig("two_trap", seed=0, randomize=False, params={})],
        modes=list(modes),
        seeds=list(seeds),
        repeats=repeats,
        out_dir=str(tmp_path / "runs"),
        figures=False,
        update_docs=False,
        record_episodes=record_episodes,
        run_id=None,
    )
    return config.validate()


def _assert_finite_numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_numbers(item)
    elif isinstance(value, float):
        assert math.isfinite(value), value


def test_ci_run_writes_strict_artifacts(tmp_path):
    config = _config(
        tmp_path,
        modes=["unshielded_skill", "static_cbf", "proposed"],
        seeds=[0],
    )
    result = run_evaluation(
        config,
        out_dir=tmp_path / "ci-runs",
        run_id="ci-test",
        router=ScriptedRouter(latency_s=0.3),
        write=True,
    )

    run_dir = Path(result.run_dir)
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "figures").is_dir()
    assert (run_dir / "episodes").is_dir()

    results = read_json(run_dir / "results.json")
    _assert_finite_numbers(results)
    raw_results = (run_dir / "results.json").read_text(encoding="utf-8")
    assert "NaN" not in raw_results
    assert "Infinity" not in raw_results
    assert results["schema_version"] == 1
    assert set(results["aggregates"]) == {"two_trap"}
    assert {"unshielded_skill", "static_cbf", "proposed"} <= set(
        results["aggregates"]["two_trap"]
    )
    proposed = results["aggregates"]["two_trap"]["proposed"]
    assert proposed["n"] == 1
    for key in (
        "route_pct",
        "avg_speed",
        "router_calls",
        "call_attempts",
        "calls_applied",
        "calls_rejected",
        "calls_suppressed",
        "router_fallbacks",
        "peak_lateral_jerk",
        "mean_lateral_jerk",
        "peak_sigma",
        "override_fraction",
        "pending_ticks",
        "safety_events",
        "infeasible_ticks",
    ):
        assert key in proposed
        assert "mean" in proposed[key]
        assert "std" in proposed[key]
        assert "ci95" in proposed[key]
    assert "crash_rate" in proposed
    assert "freeze_rate" in proposed
    assert "pending_survival_rate" in proposed
    assert "end_reason" in proposed

    manifest = read_json(run_dir / "manifest.json")
    assert validate_manifest(manifest) == []
    assert manifest["extra"]["modes"] == ["unshielded_skill", "static_cbf", "proposed"]
    assert manifest["extra"]["scenarios"] == ["two_trap"]

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "two_trap" in report
    assert "Per-scenario results" in report


def test_multi_seed_aggregation_is_deterministic(tmp_path):
    modes = ["proposed"]
    seeds = [0, 1]
    config_one = _config(tmp_path, modes=modes, seeds=seeds)
    first = run_evaluation(
        config_one,
        out_dir=tmp_path / "first",
        run_id="first",
        router=ScriptedRouter(latency_s=0.3),
    )

    config_two = _config(tmp_path, modes=modes, seeds=seeds)
    second = run_evaluation(
        config_two,
        out_dir=tmp_path / "second",
        run_id="second",
        router=ScriptedRouter(latency_s=0.3),
    )

    aggregate = first.aggregates["two_trap"]["proposed"]
    assert aggregate["n"] == 2
    for key in ("route_pct", "avg_speed", "router_calls"):
        summary = aggregate[key]
        assert summary["n"] == 2
        assert summary["std"] is not None and math.isfinite(summary["std"])
        assert summary["ci95"] is not None and math.isfinite(summary["ci95"])
    _assert_finite_numbers(first.aggregates)

    first_routes = [record["route_pct"] for record in first.episodes if record["mode"] == "proposed"]
    second_routes = [record["route_pct"] for record in second.episodes if record["mode"] == "proposed"]
    assert len(first_routes) == 2
    assert [round(value, 6) for value in first_routes] == [
        round(value, 6) for value in second_routes
    ]
    assert round(first_routes[0], 6) == round(second_routes[0], 6)


@pytest.mark.parametrize("mode", EVAL_MODES)
def test_all_eval_modes_are_accepted(mode, skill_bank):
    scenario = ScenarioConfig("two_trap", seed=0)
    result = run_episode(mode, scenario, 0, skill_bank, ScriptedRouter(latency_s=0.3))
    assert result.mode == mode
    assert result.scenario == "two_trap"
    assert result.seed == 0
    assert math.isfinite(result.route_pct)
    # The caller's scenario is not mutated by the seed override.
    assert scenario.seed == 0


def test_unknown_mode_raises_config_error(skill_bank):
    with pytest.raises(ConfigError):
        run_episode(
            "not_a_mode",
            ScenarioConfig("two_trap", seed=0),
            0,
            skill_bank,
            ScriptedRouter(),
        )


def test_missing_return_right_checkpoint_is_aliased(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    shutil.copy2(CHECKPOINTS / "CRUISE_RIGHT.pt", checkpoint_dir / "CRUISE_RIGHT.pt")
    shutil.copy2(CHECKPOINTS / "OVERTAKE_LEFT.pt", checkpoint_dir / "OVERTAKE_LEFT.pt")

    config = _config(
        tmp_path,
        modes=["proposed"],
        seeds=[0],
        checkpoint_dir=checkpoint_dir,
    )
    result = run_evaluation(
        config,
        checkpoint_dir=checkpoint_dir,
        out_dir=tmp_path / "alias-runs",
        run_id="alias-test",
        router=ScriptedRouter(),
    )

    manifest = read_json(Path(result.run_dir) / "manifest.json")
    assert validate_manifest(manifest) == []
    assert manifest["extra"]["aliases"]["RETURN_RIGHT"] == "CRUISE_RIGHT"
    assert any("RETURN_RIGHT" in warning for warning in result.warnings)
    report = (Path(result.run_dir) / "report.md").read_text(encoding="utf-8")
    assert "RETURN_RIGHT" in report and "CRUISE_RIGHT" in report


def test_regression_gate_reports_failures():
    fake = SimpleNamespace(
        aggregates={
            "two_trap": {
                "proposed": {
                    "n": 2,
                    "route_pct": {"mean": 10.0, "std": 0.0, "ci95": 0.0},
                    "router_calls": {"mean": 1000.0, "std": 0.0, "ci95": 0.0},
                    "crash_rate": 1.0,
                    "freeze_rate": 1.0,
                }
            }
        }
    )
    failures = check_regressions(fake)
    text = " ".join(failures)
    assert any("crash" in failure for failure in failures)
    assert any("froze" in failure for failure in failures)
    assert "route" in text
    assert "router calls" in text


def test_regression_gate_passes_clean_aggregate():
    fake = SimpleNamespace(
        aggregates={
            "two_trap": {
                "proposed": {
                    "n": 5,
                    "route_pct": {"mean": 95.0, "std": 1.0, "ci95": 0.5},
                    "router_calls": {"mean": 5.0, "std": 1.0, "ci95": 0.5},
                    "crash_rate": 0.0,
                    "freeze_rate": 0.0,
                },
                "unshielded_skill": {
                    "n": 5,
                    "route_pct": {"mean": 35.0},
                    "router_calls": {"mean": 0.0},
                    "crash_rate": 1.0,
                    "freeze_rate": 0.0,
                },
            }
        }
    )
    assert check_regressions(fake) == []


def test_record_episodes_writes_full_rollout(tmp_path):
    config = _config(
        tmp_path,
        modes=["proposed"],
        seeds=[0],
        record_episodes=True,
    )
    result = run_evaluation(
        config,
        out_dir=tmp_path / "record-runs",
        run_id="record-test",
        router=ScriptedRouter(),
    )
    episode_path = Path(result.run_dir) / "episodes" / "two_trap-proposed-seed0.json"
    assert episode_path.is_file()
    payload = read_json(episode_path)
    assert payload["xs"]
    assert payload["ys"]
    assert "calls" in payload
    assert payload["repeat"] == 0

    results = read_json(Path(result.run_dir) / "results.json")
    assert "xs" not in results["episodes"][0]
    assert "ys" not in results["episodes"][0]
