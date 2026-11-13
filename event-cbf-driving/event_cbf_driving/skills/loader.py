"""Checkpoint-bank loading, documented aliases, and directory validation.

Canonical deployments contain two *distinct* trained skills:
``CRUISE_RIGHT`` and ``OVERTAKE_LEFT``. ``RETURN_RIGHT`` targets the same right
lane as ``CRUISE_RIGHT``; when no distinct ``RETURN_RIGHT.pt`` exists it is
loaded as a documented alias rather than pretending a separate policy exists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from event_cbf_driving.io_utils import utc_now
from event_cbf_driving.skills.checkpoints import (
    find_duplicate_fingerprints,
    inspect_checkpoint,
    load_skill,
)

logger = logging.getLogger(__name__)

DEFAULT_ALIASES: Dict[str, str] = {"RETURN_RIGHT": "CRUISE_RIGHT"}
REQUIRED_SKILLS = ("CRUISE_RIGHT", "OVERTAKE_LEFT")
VALIDATION_SCHEMA_VERSION = 1


class SkillBank(dict):
    """A ``dict[str, SkillPolicy]`` with loader metadata attached."""

    def __init__(self, *args: Any, metadata: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.metadata: Dict[str, Any] = dict(metadata or {})


def _expected_path(checkpoint_dir: Path, name: str) -> Path:
    return checkpoint_dir / f"{name}.pt"


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def load_skill_bank(
    checkpoint_dir: Path,
    *,
    require: Sequence[str] = REQUIRED_SKILLS,
    aliases: Optional[Mapping[str, str]] = None,
    strict: bool = False,
) -> SkillBank:
    """Load required skills plus documented aliases from a checkpoint directory.

    The default alias map is ``{"RETURN_RIGHT": "CRUISE_RIGHT"}``. A real
    ``RETURN_RIGHT.pt`` is always preferred when present; otherwise the alias
    target is loaded and the returned policy is renamed to the requested
    skill. Missing *required distinct* skills raise ``FileNotFoundError`` with
    the expected paths.

    Duplicate fingerprints across distinct loaded skill names are reported in
    ``bank.metadata["duplicates"]`` (and logged).
    """

    root = Path(checkpoint_dir)
    alias_map = dict(DEFAULT_ALIASES if aliases is None else aliases)
    required = tuple(require)
    requested = _ordered_unique([*required, *alias_map.keys()])
    bank = SkillBank()
    alias_used: Dict[str, str] = {}
    loaded_real: Dict[str, Optional[str]] = {}
    missing: list[Path] = []

    for name in requested:
        path = _expected_path(root, name)
        if path.is_file():
            policy = load_skill(path, strict=strict)
            bank[name] = policy
            loaded_real[name] = policy.checkpoint_info.get("fingerprint")
            continue

        target = alias_map.get(name)
        if target and target != name:
            target_path = _expected_path(root, target)
            if target_path.is_file():
                policy = load_skill(target_path, strict=strict)
                policy.name = name
                info = dict(policy.checkpoint_info)
                info.update(
                    {
                        "alias_of": target,
                        "requested_name": name,
                        "requested_path": str(path),
                        "path": str(target_path),
                        "alias_used": True,
                    }
                )
                policy.checkpoint_info = info
                bank[name] = policy
                alias_used[name] = target
                logger.warning(
                    "%s: no distinct checkpoint at %s; using documented alias of %s",
                    name,
                    path,
                    target,
                )
                continue

        if name in required:
            missing.append(path)

    if missing:
        expected = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing required skill checkpoint(s): {expected}")

    duplicates = find_duplicate_fingerprints(loaded_real)
    for duplicate in duplicates:
        logger.warning(
            "duplicate state_dict fingerprint across distinct skills %s (%s)",
            duplicate["skills"],
            duplicate["fingerprint"],
        )

    bank.metadata = {
        "checkpoint_dir": str(root),
        "required": list(required),
        "alias_map": alias_map,
        "aliases": alias_used,
        "alias_used": bool(alias_used),
        "requested": requested,
        "fingerprints": loaded_real,
        "duplicates": duplicates,
        "strict": bool(strict),
    }
    return bank


def _aggregate_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def validate_checkpoint_dir(
    checkpoint_dir: Path,
    *,
    require: Sequence[str] = REQUIRED_SKILLS,
    aliases: Optional[Mapping[str, str]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Inspect and validate every top-level checkpoint in a directory.

    Returns a strict-JSON-safe report containing per-file fingerprints/meta,
    duplicate groups, missing required skills, alias use, and sidecar warnings.
    Invalid/missing/fingerprint-mismatching/duplicated checkpoints make
    ``valid`` false; stale sidecar metadata is reported as a warning.
    """

    root = Path(checkpoint_dir)
    alias_map = dict(DEFAULT_ALIASES if aliases is None else aliases)
    required = tuple(require)
    result: Dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "created_at": utc_now(),
        "checkpoint_dir": str(root),
        "required": list(required),
        "alias_map": alias_map,
        "skills": {},
        "fingerprints": {},
        "duplicates": [],
        "aliases": {},
        "alias_used": False,
        "missing_required": [],
        "missing_required_paths": [],
        "fingerprint_mismatches": [],
        "warnings": [],
        "sidecar_warnings": [],
        "errors": [],
        "problems": [],
        "valid": False,
    }

    paths = sorted(root.glob("*.pt")) if root.is_dir() else []
    for path in paths:
        info = inspect_checkpoint(path)
        name = path.stem
        result["skills"][name] = info
        fingerprint = info.get("fingerprint")
        if fingerprint:
            result["fingerprints"][name] = fingerprint
        for warning in info.get("warnings", []):
            result["warnings"].append(str(warning))
        for error in info.get("errors", []):
            result["errors"].append(str(error))
        if info.get("fingerprint_matches") is False:
            result["fingerprint_mismatches"].append(
                {
                    "skill": name,
                    "path": str(path),
                    "recorded": info.get("recorded_fingerprints"),
                    "computed": fingerprint,
                }
            )
        sidecar = info.get("sidecar") or {}
        for warning in sidecar.get("warnings", []):
            result["sidecar_warnings"].append(str(warning))
        try:
            load_skill(path, strict=strict)
        except Exception as exc:
            result["errors"].append(str(exc))

    for name in required:
        path = _expected_path(root, name)
        if not path.is_file():
            result["missing_required"].append(name)
            result["missing_required_paths"].append(str(path))

    if "RETURN_RIGHT" not in result["fingerprints"]:
        cruise = _expected_path(root, "CRUISE_RIGHT")
        target = alias_map.get("RETURN_RIGHT")
        if target == "CRUISE_RIGHT" and cruise.is_file():
            result["aliases"]["RETURN_RIGHT"] = "CRUISE_RIGHT"

    result["alias_used"] = bool(result["aliases"])
    result["duplicates"] = find_duplicate_fingerprints(result["fingerprints"])

    problems: list[str] = []
    if result["missing_required"]:
        problems.append(
            "missing required skill(s): "
            + ", ".join(
                f"{name} ({path})"
                for name, path in zip(
                    result["missing_required"], result["missing_required_paths"], strict=False
                )
            )
        )
    if result["errors"]:
        problems.extend(result["errors"])
    for mismatch in result["fingerprint_mismatches"]:
        problems.append(f"fingerprint mismatch for {mismatch['skill']} at {mismatch['path']}")
    for duplicate in result["duplicates"]:
        problems.append(
            "duplicate state_dict fingerprint across distinct skills: "
            + ", ".join(duplicate["skills"])
        )
    result["problems"] = _aggregate_unique(problems)
    result["errors"] = _aggregate_unique(result["errors"])
    result["warnings"] = _aggregate_unique(result["warnings"])
    result["sidecar_warnings"] = _aggregate_unique(result["sidecar_warnings"])
    result["valid"] = not result["problems"]
    return result
