"""Artifact recovery must be relocatable, checksum-verified and non-destructive."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "restore_safemetadrive", Path(__file__).resolve().parents[1] / "tools/restore_safemetadrive.py"
)
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def archive_fixture(tmp_path, name="runs/example.json", payload=b'{"complete":true}'):
    archive = tmp_path / "records.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    package = dict(
        sha256=restore.digest(archive),
        files=[dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())],
    )
    return archive, package


def test_restore_checks_contents_and_is_idempotent(tmp_path):
    archive, package = archive_fixture(tmp_path)
    destination = tmp_path / "fresh-checkout"
    assert restore.restore_package(archive, package, destination) == 1
    assert restore.restore_package(archive, package, destination) == 1
    assert json.loads((destination / "runs/example.json").read_text()) == {"complete": True}


def test_corrupt_archive_and_differing_existing_files_are_not_accepted(tmp_path):
    archive, package = archive_fixture(tmp_path)
    destination = tmp_path / "checkout"
    restore.restore_package(archive, package, destination)
    target = destination / "runs/example.json"
    target.write_text("local work")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        restore.restore_package(archive, package, destination)
    assert target.read_text() == "local work"
    archive.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        restore.restore_package(archive, package, destination)


def test_archive_cannot_escape_destination_or_replace_through_symlink(tmp_path):
    archive, package = archive_fixture(tmp_path, "../escaped")
    with pytest.raises(ValueError, match="Unsafe"):
        restore.restore_package(archive, package, tmp_path / "destination")
    assert not (tmp_path / "escaped").exists()
    destination = tmp_path / "checkout"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "runs").symlink_to(outside, target_is_directory=True)
    archive, package = archive_fixture(tmp_path)
    with pytest.raises(ValueError, match="leaves destination"):
        restore.restore_package(archive, package, destination)
    assert not (outside / "example.json").exists()


def test_rebase_derives_paths_and_aliases_without_rewriting_frozen_plan(tmp_path):
    root = tmp_path / "relocated-checkout"
    study = root / "runs/safemetadrive/heldout-study-v1"
    study.mkdir(parents=True)
    plan = dict(groups={"camera": [["/results/heldout-study-v1/episodes/camera.json"]]})
    frozen = study / "frozen-plan.json"
    frozen.write_text(json.dumps(plan))
    before = frozen.read_bytes()
    model = root / "runs/safemetadrive/skills-1m-v1/skills-seed0/ppo.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"trained checkpoint")
    remote = "/results/skills-1m-v1/skills-seed0/ppo.zip"
    original = dict(
        files={remote: dict(path="/old-machine/model.zip", sha256=restore.digest(model))}
    )
    (study / "artifact-map.original.json").write_text(json.dumps(original))
    restore.rebase_paths(root)
    groups = json.loads((study / "groups-local.json").read_text())
    assert groups["camera"][0][0] == str(study / "episodes/camera.json")
    mapping = json.loads((study / "artifact-map.json").read_text())
    assert mapping["files"][remote]["path"] == str(model)
    assert (
        study / "artifacts/skills-1m-v1/skills-seed0/ppo.zip"
    ).read_bytes() == model.read_bytes()
    assert frozen.read_bytes() == before
