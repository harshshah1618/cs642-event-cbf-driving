"""CLI contract tests (offline, no network)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from driving import __version__
from driving.assets import asset_directory
from driving.cli import build_parser, main
from driving.io_utils import atomic_write_text, strict_json_loads
from driving.result import RolloutResult
from driving.runtime import write_episode

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = asset_directory(ROOT / "models/native")


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_parser_requires_a_command():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


def test_unknown_config_is_a_config_error(capsys):
    rc = main(["eval", "--config", "/definitely/not/here.yaml"])
    assert rc == 2
    assert "config file not found" in capsys.readouterr().err


def test_eval_ci_profile_runs_offline(tmp_path, capsys):
    rc = main(
        [
            "eval",
            "--profile",
            "ci",
            "--router",
            "scripted",
            "--ckpt-dir",
            str(CHECKPOINTS),
            "--no-write",
            "--no-fail-on-regression",
            "--seeds",
            "0",
        ]
    )
    output = capsys.readouterr().out
    assert rc == 0
    assert "two_trap" in output
    assert "proposed" in output


def test_eval_json_output_is_strict(tmp_path, capsys):
    rc = main(
        [
            "eval",
            "--profile",
            "ci",
            "--router",
            "scripted",
            "--ckpt-dir",
            str(CHECKPOINTS),
            "--no-write",
            "--no-fail-on-regression",
            "--json",
        ]
    )
    assert rc == 0
    payload = strict_json_loads(capsys.readouterr().out)
    assert payload["run_id"]
    assert "two_trap" in payload["aggregates"]


def test_verify_writes_validation_to_requested_directory(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT"):
        shutil.copy2(CHECKPOINTS / f"{name}.pt", ckpt_dir / f"{name}.pt")

    rc = main(
        [
            "verify",
            "--profile",
            "ci",
            "--ckpt-dir",
            str(ckpt_dir),
            "--min-learned-skills",
            "0",
        ]
    )
    assert rc == 0
    assert (ckpt_dir / "validation.json").is_file()


def test_replay_validate_only_accepts_a_well_formed_episode(tmp_path, capsys):
    result = RolloutResult(
        mode="proposed",
        crashed=False,
        crash_time=None,
        route_pct=100.0,
        avg_speed=20.0,
        router_calls=0,
        peak_lateral_jerk=0.0,
        peak_sigma=0.0,
        freeze=False,
        x=1.0,
        y=0.0,
        vx=20.0,
        end_time=0.02,
        xs=[0.0],
        ys=[0.0],
        skills=["CRUISE_RIGHT"],
    )
    path = write_episode(tmp_path / "episode.json", result)

    rc = main(["replay", str(path), "--validate-only", "--json"])
    payload = strict_json_loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["ticks"] == 1
    assert payload["has_actions"] is False
    assert payload["scenario"] == "two_trap"


def test_replay_rejects_a_malformed_episode(tmp_path, capsys):
    path = tmp_path / "broken.json"
    atomic_write_text(path, '{"mode": "proposed"}\n')

    rc = main(["replay", str(path), "--validate-only"])

    assert rc == 2
    assert "invalid episode" in capsys.readouterr().err


def test_doctor_json_reports_ok(capsys):
    rc = main(["doctor", "--json"])
    assert rc == 0
    payload = strict_json_loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["checkpoints"]["aliases"]["RETURN_RIGHT"] == "CRUISE_RIGHT"
    assert payload["imports"]["numpy"]["ok"] is True
    assert payload["router"]["live_capable"] is False


def test_router_check_offline_backend_is_explicitly_not_live(capsys):
    rc = main(["router-check", "--router", "dryrun", "--json"])
    assert rc == 0
    payload = strict_json_loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["live"] is False
    assert payload["live_capable"] is False
    assert payload["skill"] in ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")


def test_router_check_require_live_rejects_scripted(capsys):
    rc = main(["router-check", "--router", "scripted", "--require-live", "--json"])
    assert rc == 4
    payload = strict_json_loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["live"] is False
    assert "no live model answer" in payload["error"]


def test_router_check_live_backend_requires_opt_in(capsys):
    rc = main(["router-check", "--router", "openai"])
    assert rc == 2
    assert "allow_live" in capsys.readouterr().err


def test_eval_require_live_rejects_offline_router(capsys):
    rc = main(
        [
            "eval",
            "--profile",
            "ci",
            "--router",
            "scripted",
            "--require-live",
            "--no-write",
            "--no-fail-on-regression",
        ]
    )
    assert rc == 4
    assert "live preflight failed" in capsys.readouterr().err


def test_live_profiles_load_and_opt_in(capsys):
    for name, backend in (
        ("live-openai", "openai"),
        ("live-ollama", "local"),
        ("live-local", "local"),
    ):
        rc = main(["router-check", "--profile", name, "--json"])
        # Loading succeeds; the actual live probe fails because this test
        # environment has no key/daemon/model. A config error would return 2.
        assert rc in (0, 4)
        payload = strict_json_loads(capsys.readouterr().out)
        assert payload["configured_backend"] == backend
        assert payload["live_capable"] is True
