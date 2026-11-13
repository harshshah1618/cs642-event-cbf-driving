"""The maintained commands must run with the real saved driver bank."""

import json

import pytest
from driving.cli import main
from driving.config import RouterConfig, load_config
from driving.errors import ConfigError
from driving.runtime import replay_open_loop


def test_eval_writes_replayable_episode(tmp_path, capsys):
    assert (
        main(
            [
                "eval",
                "--profile",
                "ci",
                "--modes",
                "proposed",
                "--seeds",
                "0",
                "--out-dir",
                str(tmp_path),
                "--run-id",
                "smoke",
            ]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1 and rows[0]["mode"] == "proposed"
    assert json.loads((tmp_path / "smoke/results.json").read_text()) == rows
    replay = replay_open_loop(tmp_path / "smoke/two_trap-proposed-0.json")
    assert replay.ok, replay.errors


@pytest.mark.parametrize("command", ["verify", "doctor"])
def test_saved_models_validate(command, capsys):
    assert main([command, "--profile", "ci"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"]
    if command == "verify":
        assert all(row["learned"] for row in report["diagnostics"].values())


def test_bad_config_returns_error(tmp_path, capsys):
    config = tmp_path / "bad.yaml"
    config.write_text("router:\n  backend: imaginary\n")
    assert main(["eval", "--config", str(config)]) == 2
    assert "router.backend" in capsys.readouterr().out


def test_retired_config_keys_are_rejected(tmp_path):
    config = tmp_path / "old.yaml"
    config.write_text("router:\n  allow_fallback: true\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(config)
    with pytest.raises(ConfigError):
        RouterConfig(backend="ollama").validate()


def test_train_help_reaches_current_trainer(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["train", "--help"])
    assert stopped.value.code == 0
    assert "--num-envs" in capsys.readouterr().out
