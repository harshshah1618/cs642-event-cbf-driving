import hashlib
import importlib.util
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "research_data_audit", Path(__file__).resolve().parents[1] / "scripts/research_data_audit.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row_id(row):
    names = ["scenario", "seed", "delay", "method", "value", "controller"]
    row["id"] = hashlib.sha256(
        json.dumps({k: row[k] for k in names}, sort_keys=True).encode()
    ).hexdigest()[:20]
    return row


@pytest.fixture
def native():
    scenes = ["two_trap", "merge", "dense_highway"]
    delays = ["fast", "nominal", "slow", "long", "jitter"]
    lock = {"deadline": 1, "periodic": 0.1,
            "conditioned_periodic": {f"{s}|{d}": .25 for s in scenes for d in delays}}
    rows = []
    for seed in range(1001, 1041):
        for scene in scenes:
            for delay in delays:
                for method, value in [("legacy", 0), ("ttc", 0), ("benefit", 0),
                                      ("deadline", 1), ("no_latency_matched", 1),
                                      ("periodic", .1), ("periodic_gated", 10.),
                                      ("periodic_conditioned", .25)]:
                    rows.append(row_id(dict(scenario=scene, seed=seed, delay=delay,
                                            method=method, value=value, controller="ppo")))
    return rows, lock


def test_complete_native_grid(native):
    rows, lock = native
    assert audit.validate_test_cells("isolation", rows, lock)["validated_cells"] == 4800


def test_duplicate_cell_with_different_value_and_unique_id_is_rejected(native):
    rows, lock = native
    duplicate = row_id(dict(rows[0], value=9))
    assert duplicate["id"] != rows[0]["id"]
    with pytest.raises(ValueError, match="duplicate cell"):
        audit.validate_test_cells("isolation", rows + [duplicate], lock)


def test_missing_and_equal_count_extra_cells_are_rejected(native):
    rows, lock = native
    with pytest.raises(ValueError, match="missing 1 expected"):
        audit.validate_test_cells("isolation", rows[:-1], lock)
    rows[-1] = row_id(dict(rows[-1], seed=9999))
    with pytest.raises(ValueError, match="unexpected cell"):
        audit.validate_test_cells("isolation", rows, lock)


def test_corrupt_id_rejected(native):
    rows, lock = native
    rows[0]["id"] = "not-the-job-hash"
    with pytest.raises(ValueError, match="job ID mismatch"):
        audit.validate_test_cells("isolation", rows, lock)


def test_changed_selection_cannot_relabel_old_results(native):
    rows, lock = native
    lock["periodic"] = .5
    with pytest.raises(ValueError, match="selection value mismatch"):
        audit.validate_test_cells("isolation", rows, lock)


def test_phase_inherited_deadline_must_match():
    with pytest.raises(ValueError, match="phase deadline buffer"):
        audit.validate_test_cells("phase", [], {"deadline_buffer_s": 1}, {"deadline": .5})


def test_complete_phase_grid_and_inherited_buffer(native):
    native_rows, lock = native
    rows = [row_id(dict(r, seed=r["seed"] + 100)) for r in native_rows
            if r["delay"] in ("nominal", "long", "jitter")]
    phase_lock = dict(lock, deadline_buffer_s=1)
    assert audit.validate_test_cells("phase", rows, phase_lock, lock)["validated_cells"] == 2880


def test_road_grid_requires_references_only_at_standard_delay():
    conditions = ["generalization", "safe"]
    delays = ["standard", "nominal", "slow", "long", "jitter"]
    lock = {"deadline": .5, "periodic": 1,
            "conditioned_periodic": {f"{c}|{d}": 2 for c in conditions for d in delays}}
    rows = []
    for condition in conditions:
        for seed in range(2001, 2041):
            for delay in delays:
                candidates = [("never", 0), ("shield", 0), ("ttc", 0), ("benefit", 0),
                              ("deadline", .5), ("no_latency_matched", .5),
                              ("periodic", 1), ("periodic_gated", 10.), ("periodic_conditioned", 2)]
                if delay == "standard":
                    candidates += [("idm", 0), ("expert", 0)]
                for method, value in candidates:
                    row = dict(environment=condition, seed=seed, delay=delay, method=method, value=value)
                    row["id"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
                    rows.append(row)
    assert audit.validate_test_cells("roads", rows, lock)["validated_cells"] == 3760
    reference = next(r for r in rows if r["method"] == "expert")
    reference["delay"] = "long"
    with pytest.raises(ValueError, match="unexpected cell"):
        audit.validate_test_cells("roads", rows, lock)


def test_runtime_version_and_commit_checks(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    lock = {"python": platform.python_version(), "packages": {"numpy": "test-version"},
            "checkpoint_sha256": {}}
    provenance = {"url": "https://github.com/metadriverse/metadrive.git",
                  "vcs_info": {"vcs": "git", "commit_id": audit.METADRIVE_COMMIT}}
    monkeypatch.setattr(audit.metadata, "version", lambda _: "test-version")
    monkeypatch.setattr(audit.metadata, "distribution", lambda _: SimpleNamespace(
        read_text=lambda _: json.dumps(provenance)))
    assert audit.runtime_audit({"native": lock})["runtime_versions_verified"]
    provenance["vcs_info"]["commit_id"] = "different"
    with pytest.raises(ValueError, match="installed commit"):
        audit.runtime_audit({"native": lock})
    monkeypatch.setattr(audit.metadata, "version", lambda _: "changed-version")
    with pytest.raises(ValueError, match="runtime version mismatch"):
        audit.runtime_audit({"native": lock})


def test_new_optional_checkpoint_is_detected(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    checkpoint_dir = tmp_path / "checkpoints/metadrive-calibrated"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "RETURN_RIGHT.pt").write_bytes(b"new optional model")
    with pytest.raises(ValueError, match="checkpoint inventory changed"):
        audit.runtime_audit({"native": {"checkpoint_sha256": {}}})
