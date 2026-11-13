"""Checkpoint format, provenance, alias, and validation contract tests.

These tests are deliberately fast: tiny default ``ActorCritic`` instances only,
no PPO training, and all artifacts under ``tmp_path``.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict

import pytest
import torch
from driving.checkpoints import (
    _load_torch_checkpoint,
    inspect_checkpoint,
    load_skill,
    save_skill,
    state_dict_fingerprint,
)
from driving.errors import ArtifactError, ValidationError
from driving.io_utils import read_json, strict_json_dumps
from driving.model_loader import load_skill_bank, validate_checkpoint_dir
from driving.network import ActorCritic
from driving.policy import SkillMeta


def _meta(name: str = "CRUISE_RIGHT", lane: int = 0, y_target: float = 0.0) -> SkillMeta:
    return SkillMeta(
        name=name,
        target_lane=lane,
        y_target=y_target,
        threshold=1.5,
        sigma_mean=0.8,
        sigma_std=0.2,
        spread=0.5,
    )


def _net(seed: int = 0, bias: float = 0.0) -> ActorCritic:
    torch.manual_seed(seed)
    net = ActorCritic()
    if bias:
        with torch.no_grad():
            net.mu.bias.add_(bias)
    return net


def _write_legacy(path, net: ActorCritic, meta: SkillMeta, history=(1.0, 2.0)):
    torch.save(
        {
            "state_dict": net.state_dict(),
            "meta": asdict(meta),
            "history": list(history),
        },
        path,
    )


def _write_distinct_pair(directory):
    (directory / "CRUISE_RIGHT.pt").parent.mkdir(parents=True, exist_ok=True)
    cruise_info = save_skill(
        directory / "CRUISE_RIGHT.pt",
        _net(seed=1),
        _meta("CRUISE_RIGHT", 0, 0.0),
        history=[1.0],
        provenance={"seed": 1, "updates": 1, "agent": "test"},
    )
    overtake_info = save_skill(
        directory / "OVERTAKE_LEFT.pt",
        _net(seed=2),
        _meta("OVERTAKE_LEFT", 1, 4.0),
        history=[2.0],
        provenance={"seed": 2, "updates": 1, "agent": "test"},
    )
    return cruise_info, overtake_info


def test_checkpoint_loader_prefers_restricted_deserialization(monkeypatch, tmp_path):
    calls = []

    def fake_load(path, *, map_location=None, weights_only=False):
        calls.append(weights_only)
        return {"state_dict": {}}

    monkeypatch.setattr("driving.checkpoints.torch.load", fake_load)

    payload = _load_torch_checkpoint(tmp_path / "safe.pt", map_location="cpu")

    assert payload == {"state_dict": {}}
    assert calls == [True]


def test_checkpoint_loader_warns_before_legacy_fallback(monkeypatch, caplog, tmp_path):
    calls = []

    def fake_load(path, *, map_location=None, weights_only=False):
        calls.append(weights_only)
        if weights_only:
            raise RuntimeError("unsupported global")
        return {"state_dict": {}}

    monkeypatch.setattr("driving.checkpoints.torch.load", fake_load)

    with caplog.at_level(logging.WARNING):
        payload = _load_torch_checkpoint(tmp_path / "legacy.pt", map_location="cpu")

    assert payload == {"state_dict": {}}
    assert calls == [True, False]
    assert any("falling back to full deserialization" in r.message for r in caplog.records)


def test_save_load_roundtrip_with_provenance(tmp_path):
    path = tmp_path / "CRUISE_RIGHT.pt"
    net = _net(seed=0)
    meta = _meta()
    expected_fingerprint = state_dict_fingerprint(net.state_dict())

    info = save_skill(
        path,
        net,
        meta,
        history=[1.0, 2.0, 3.0],
        provenance={"seed": 7, "updates": 3, "agent": "test_checkpoints"},
    )

    assert info["fingerprint"] == expected_fingerprint
    assert path.is_file()
    assert path.with_suffix(".json").is_file()

    sidecar = read_json(path.with_suffix(".json"))
    assert sidecar["schema_version"] == 2
    assert sidecar["fingerprint"] == expected_fingerprint
    assert sidecar["history_len"] == 3
    assert sidecar["meta"] == asdict(meta)

    policy = load_skill(path, strict=True)
    assert policy.name == "CRUISE_RIGHT"
    assert policy.meta == meta
    assert policy.checkpoint_info["fingerprint"] == expected_fingerprint
    assert policy.checkpoint_info["legacy"] is False
    assert policy.checkpoint_info["provenance"]["seed"] == 7
    assert policy.checkpoint_info["provenance"]["updates"] == 3
    assert policy.checkpoint_info["provenance"]["state_dict_sha256"] == expected_fingerprint
    assert policy.checkpoint_info["provenance"]["agent"] == "test_checkpoints"

    summary = inspect_checkpoint(path)
    assert summary["fingerprint"] == expected_fingerprint
    assert summary["fingerprint_matches"] is True
    assert summary["legacy"] is False
    assert summary["history_len"] == 3
    strict_json_dumps(summary)


def test_legacy_checkpoint_warns_non_strict_and_fails_strict(tmp_path, caplog):
    path = tmp_path / "CRUISE_RIGHT.pt"
    _write_legacy(path, _net(seed=0), _meta())

    with caplog.at_level(logging.WARNING, logger="driving.checkpoints"):
        policy = load_skill(path)
    assert policy.checkpoint_info["legacy"] is True
    assert any("legacy checkpoint" in record.getMessage() for record in caplog.records)

    summary = inspect_checkpoint(path)
    assert summary["legacy"] is True
    strict_json_dumps(summary)

    with pytest.raises(ArtifactError, match="legacy checkpoint"):
        load_skill(path, strict=True)


def test_corrupt_state_dict_fingerprint_is_rejected(tmp_path):
    path = tmp_path / "CRUISE_RIGHT.pt"
    save_skill(path, _net(seed=0), _meta(), history=[1.0], provenance={"seed": 0})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state_dict_sha256"] = "0" * 64
    payload["provenance"]["state_dict_sha256"] = "0" * 64
    torch.save(payload, path)

    with pytest.raises(ArtifactError, match="fingerprint mismatch"):
        load_skill(path)


@pytest.mark.parametrize("mutation", ["missing_key", "zero_spread", "nonfinite_threshold"])
def test_malformed_meta_is_rejected(tmp_path, mutation):
    path = tmp_path / "CRUISE_RIGHT.pt"
    meta = asdict(_meta())
    if mutation == "missing_key":
        del meta["threshold"]
    elif mutation == "zero_spread":
        meta["spread"] = 0.0
    else:
        meta["threshold"] = float("nan")
    torch.save(
        {
            "state_dict": _net(seed=0).state_dict(),
            "meta": meta,
            "history": [1.0],
        },
        path,
    )

    with pytest.raises(ValidationError):
        load_skill(path)


def test_load_skill_bank_aliases_return_right_when_missing(tmp_path, caplog):
    cruise, _ = _write_distinct_pair(tmp_path)

    with caplog.at_level(logging.WARNING, logger="model_loader"):
        bank = load_skill_bank(tmp_path)

    assert set(bank) == {"CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT"}
    assert bank["RETURN_RIGHT"].name == "RETURN_RIGHT"
    assert bank["RETURN_RIGHT"].meta.name == "CRUISE_RIGHT"
    assert bank["RETURN_RIGHT"].checkpoint_info["alias_of"] == "CRUISE_RIGHT"
    assert bank["RETURN_RIGHT"].checkpoint_info["fingerprint"] == cruise["fingerprint"]
    assert bank.metadata["aliases"] == {"RETURN_RIGHT": "CRUISE_RIGHT"}
    assert bank.metadata["alias_used"] is True
    assert bank.metadata["duplicates"] == []
    assert any("documented alias" in record.getMessage() for record in caplog.records)


def test_load_skill_bank_prefers_real_return_right(tmp_path):
    _write_distinct_pair(tmp_path)
    return_info = save_skill(
        tmp_path / "RETURN_RIGHT.pt",
        _net(seed=9, bias=0.25),
        _meta("RETURN_RIGHT", 0, 0.0),
        history=[3.0],
        provenance={"seed": 9, "updates": 1, "agent": "test"},
    )

    bank = load_skill_bank(tmp_path)

    assert "alias_of" not in bank["RETURN_RIGHT"].checkpoint_info
    assert bank["RETURN_RIGHT"].checkpoint_info["fingerprint"] == return_info["fingerprint"]
    assert (
        bank["RETURN_RIGHT"].checkpoint_info["fingerprint"]
        != bank["CRUISE_RIGHT"].checkpoint_info["fingerprint"]
    )
    assert bank.metadata["aliases"] == {}


def test_load_skill_bank_missing_cruise_raises(tmp_path):
    save_skill(
        tmp_path / "OVERTAKE_LEFT.pt",
        _net(seed=2),
        _meta("OVERTAKE_LEFT", 1, 4.0),
        history=[2.0],
        provenance={"seed": 2},
    )

    with pytest.raises(FileNotFoundError, match="CRUISE_RIGHT"):
        load_skill_bank(tmp_path)


def test_duplicate_checkpoint_is_flagged(tmp_path):
    _write_distinct_pair(tmp_path)
    shutil.copyfile(tmp_path / "CRUISE_RIGHT.pt", tmp_path / "RETURN_RIGHT.pt")

    report = validate_checkpoint_dir(tmp_path)
    duplicate_pairs = [set(item["skills"]) for item in report["duplicates"]]
    assert {"CRUISE_RIGHT", "RETURN_RIGHT"} in duplicate_pairs
    assert not report["valid"]
    assert any("duplicate state_dict fingerprint" in problem for problem in report["problems"])
    strict_json_dumps(report)

    bank = load_skill_bank(tmp_path)
    assert bank.metadata["duplicates"]
    assert any(
        {"CRUISE_RIGHT", "RETURN_RIGHT"} == set(item["skills"])
        for item in bank.metadata["duplicates"]
    )


def test_inspect_and_validate_are_strict_json_safe(tmp_path):
    _write_distinct_pair(tmp_path)
    report = validate_checkpoint_dir(tmp_path)

    assert report["valid"] is True
    assert report["missing_required"] == []
    assert report["duplicates"] == []
    assert report["aliases"] == {"RETURN_RIGHT": "CRUISE_RIGHT"}
    assert report["alias_used"] is True
    assert set(report["fingerprints"]) == {"CRUISE_RIGHT", "OVERTAKE_LEFT"}

    serialized = strict_json_dumps(report)
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
