"""Run manifests: what code, config, environment, and weights produced a result."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from constants import __version__
from io_utils import sha256_file, stable_hash, utc_now

PACKAGE_NAME = "event-cbf-driving"
_TRACKED_PACKAGES = (
    "numpy",
    "torch",
    "gymnasium",
    "matplotlib",
    "pyyaml",
    "highway-env",
    "metadrive-simulator",
    "panda3d",
    "transformers",
    "accelerate",
    "safetensors",
)


def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info(cwd: Optional[Path] = None) -> Dict[str, Any]:
    root = (cwd or Path.cwd()).resolve()
    sha = _run_git(["rev-parse", "HEAD"], root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    return {
        "commit": sha,
        "branch": branch,
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def _version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info() -> Dict[str, Any]:
    metadrive_commit = None
    try:
        direct_url = importlib.metadata.distribution("metadrive-simulator").read_text("direct_url.json")
        if direct_url:
            metadrive_commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        pass
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_version": __version__,
        "packages": {name: _version(name) for name in _TRACKED_PACKAGES},
        "metadrive_source_commit": metadrive_commit,
    }


def checkpoint_hashes(paths: Iterable[Path | str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw in sorted(str(p) for p in paths):
        path = Path(raw)
        if path.exists():
            out[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        else:
            out[path.name] = {"missing": True}
    return out


def checkpoint_hashes_in_dir(directory: Path | str) -> Dict[str, Dict[str, Any]]:
    root = Path(directory)
    if not root.exists():
        return {}
    return checkpoint_hashes(sorted(root.glob("*.pt")))


def build_manifest(
    *,
    config: Any,
    command: str,
    run_id: str,
    checkpoint_dir: Optional[Path | str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "command": command,
        "config": config_dict,
        "config_sha256": stable_hash(config_dict),
        "environment": environment_info(),
        "git": git_info(),
    }
    if checkpoint_dir is not None:
        manifest["checkpoints"] = checkpoint_hashes_in_dir(checkpoint_dir)
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable problems (empty means valid)."""

    problems: list[str] = []
    if manifest.get("schema_version") != 1:
        problems.append(f"unsupported manifest schema_version: {manifest.get('schema_version')!r}")
    for key in ("run_id", "created_at", "config", "config_sha256", "environment", "git"):
        if key not in manifest:
            problems.append(f"missing manifest key: {key}")
    if "config" in manifest and "config_sha256" in manifest:
        if stable_hash(manifest["config"]) != manifest["config_sha256"]:
            problems.append("config hash does not match the recorded config")
    return problems
