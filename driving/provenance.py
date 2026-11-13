"""Record source and environment details when saving a trained model."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from driving import __version__

_TRACKED_PACKAGES = (
    "numpy",
    "torch",
    "gymnasium",
    "matplotlib",
    "pyyaml",
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
        direct_url = importlib.metadata.distribution("metadrive-simulator").read_text(
            "direct_url.json"
        )
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
