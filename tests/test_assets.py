"""Packed assets preserve bytes and do not extract into the working folder."""

import os
from zipfile import ZipFile

import pytest
from driving.assets import asset_directory


def test_archive_reuses_external_cache_and_preserves_bytes(tmp_path):
    path = tmp_path / "models.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("weights.bin", b"original model bytes")
    extracted = asset_directory(path)
    assert extracted != tmp_path
    assert (extracted / "weights.bin").read_bytes() == b"original model bytes"
    assert asset_directory(path) == extracted
    assert not (tmp_path / "weights.bin").exists()


def test_changed_archive_does_not_reuse_old_assets(tmp_path):
    path = tmp_path / "models.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("model", b"first")
    old = asset_directory(path)
    stamp = path.stat().st_mtime_ns
    with ZipFile(path, "w") as archive:
        archive.writestr("model", b"other")
    os.utime(path, ns=(stamp + 10_000_000, stamp + 10_000_000))
    new = asset_directory(path)
    assert new != old
    assert (old / "model").read_bytes() == b"first"
    assert (new / "model").read_bytes() == b"other"


def test_parent_paths_are_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("../escape", b"not allowed")
    with pytest.raises(ValueError, match="Invalid asset archive path"):
        asset_directory(path)
