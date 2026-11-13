"""Checkpoint storage and validation, separate from PPO learning."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import numbers
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from driving.errors import ArtifactError, ValidationError
from driving.io_utils import atomic_write_bytes, atomic_write_json, read_json, utc_now
from driving.network import DEVICE, ActorCritic, CleanRLAgent
from driving.policy import SkillMeta, SkillPolicy
from driving.provenance import environment_info, git_info

CHECKPOINT_SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
META_FIELDS = ("name", "target_lane", "y_target", "threshold", "sigma_mean", "sigma_std", "spread")
_PROVENANCE_REQUIRED = (
    "created_at",
    "package_version",
    "python",
    "torch",
    "numpy",
    "git",
    "state_dict_sha256",
    "agent",
)
logger = logging.getLogger(__name__)


def _load_torch_checkpoint(path: Path, *, map_location: Any) -> Any:
    """Load a checkpoint with the restricted unpickler when possible.

    ``weights_only=True`` blocks arbitrary code execution during
    deserialization.  Some older checkpoints contain custom classes and cannot
    be read that way; those fall back to full deserialization with an explicit
    warning.  Only load checkpoint files from sources you trust.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as restricted_exc:
        logger.warning(
            "%s: restricted checkpoint load failed (%s); falling back to full "
            "deserialization, only load trusted checkpoint files",
            path,
            type(restricted_exc).__name__,
        )
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ArtifactError(f"{path}: cannot read checkpoint: {exc}") from exc


def _json_safe(value: Any) -> Any:
    """Convert arbitrary provenance values into strict-JSON-safe primitives."""

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        out = float(value)
        return out if math.isfinite(out) else None
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    try:
        return tensor.numpy().tobytes()
    except (TypeError, RuntimeError):
        flat = tensor.reshape(-1)
        if flat.dtype == torch.bool:
            flat = flat.to(torch.uint8)
        return flat.view(torch.uint8).numpy().tobytes()


def state_dict_fingerprint(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Stable SHA-256 over sorted state-dict names, dtypes, shapes, and bytes."""

    if not isinstance(state_dict, Mapping):
        raise ValidationError("state_dict must be a mapping of tensor names to tensors")
    digest = hashlib.sha256(b"event-cbf-state-dict-v1\0")
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            raise ValidationError(f"state_dict[{key!r}] is not a torch tensor")
        prepared = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(prepared.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(prepared.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(prepared))
        digest.update(b"\0")
    return digest.hexdigest()


def _finite_float(value: Any, field: str, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{path}: meta[{field!r}] must be a finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValidationError(f"{path}: meta[{field!r}] must be finite, got {value!r}")
    return out


def validate_skill_meta(meta: Any, *, path: str) -> SkillMeta:
    """Validate required meta keys/types and return a normalized ``SkillMeta``."""

    if isinstance(meta, SkillMeta):
        data: Dict[str, Any] = asdict(meta)
    elif isinstance(meta, Mapping):
        data = dict(meta)
    else:
        raise ValidationError(f"{path}: meta must be a mapping, got {type(meta).__name__}")
    missing = [key for key in META_FIELDS if key not in data]
    if missing:
        raise ValidationError(f"{path}: meta is missing required keys {missing}")
    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(f"{path}: meta['name'] must be a non-empty string, got {name!r}")
    lane = data["target_lane"]
    if isinstance(lane, bool) or not isinstance(lane, numbers.Integral):
        raise ValidationError(f"{path}: meta['target_lane'] must be an integer, got {lane!r}")
    values = {
        field: _finite_float(data[field], field, path)
        for field in ("y_target", "threshold", "sigma_mean", "sigma_std", "spread")
    }
    if values["spread"] <= 0.0:
        raise ValidationError(f"{path}: meta['spread'] must be > 0, got {values['spread']}")
    if values["sigma_std"] < 0.0:
        raise ValidationError(f"{path}: meta['sigma_std'] must be >= 0, got {values['sigma_std']}")
    return SkillMeta(
        name=name.strip(),
        target_lane=int(lane),
        y_target=values["y_target"],
        threshold=values["threshold"],
        sigma_mean=values["sigma_mean"],
        sigma_std=values["sigma_std"],
        spread=values["spread"],
    )


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _recorded_fingerprints(payload: Mapping[str, Any], provenance: Any) -> List[Any]:
    values: List[Any] = []
    for key in ("state_dict_sha256", "fingerprint"):
        value = payload.get(key)
        if value is not None:
            values.append(value)
    if isinstance(provenance, Mapping):
        value = provenance.get("state_dict_sha256")
        if value is not None:
            values.append(value)
    return values


def _validate_provenance(provenance: Any, *, path: str) -> List[str]:
    if not isinstance(provenance, Mapping):
        return [f"{path}: provenance must be a mapping, got {type(provenance).__name__}"]
    problems: List[str] = []
    missing = [key for key in _PROVENANCE_REQUIRED if key not in provenance]
    if missing:
        problems.append(f"{path}: provenance is missing required keys {missing}")
    if "created_at" in provenance and not isinstance(provenance["created_at"], str):
        problems.append(f"{path}: provenance['created_at'] must be a string")
    for key in ("package_version", "python", "torch", "numpy"):
        if key in provenance and not isinstance(provenance[key], str):
            problems.append(f"{path}: provenance[{key!r}] must be a string")
    git = provenance.get("git")
    if not isinstance(git, Mapping):
        problems.append(f"{path}: provenance['git'] must be a mapping")
    else:
        for key in ("commit", "branch", "dirty"):
            if key not in git:
                problems.append(f"{path}: provenance['git'] is missing {key!r}")
        if "dirty" in git and not isinstance(git["dirty"], bool):
            problems.append(f"{path}: provenance['git']['dirty'] must be a boolean")
    fingerprint = provenance.get("state_dict_sha256")
    if fingerprint is not None and not _is_sha256_hex(fingerprint):
        problems.append(f"{path}: provenance['state_dict_sha256'] must be a 64-char hex digest")
    if "agent" in provenance and (
        not isinstance(provenance["agent"], str) or not provenance["agent"].strip()
    ):
        problems.append(f"{path}: provenance['agent'] must be a non-empty string")
    for key in ("seed", "updates"):
        if key in provenance and provenance[key] is not None:
            value = provenance[key]
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                problems.append(f"{path}: provenance[{key!r}] must be an integer")
    return problems


def build_checkpoint_provenance(
    *,
    meta: SkillMeta,
    fingerprint: str,
    history: Optional[List[float]] = None,
    supplied: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the provenance block embedded in a v2 checkpoint."""

    environment = environment_info()
    packages = environment.get("packages") or {}
    git = git_info()
    history_list = list(history or [])
    values = dict(supplied or {})
    agent = values.pop("agent", None)
    caller = values.pop("caller", None)
    seed = values.pop("seed", None)
    updates = values.pop("updates", None)
    created_at = values.pop("created_at", None)
    provenance: Dict[str, Any] = {
        "created_at": created_at or utc_now(),
        "package_version": environment.get("package_version"),
        "python": environment.get("python"),
        "torch": packages.get("torch") or getattr(torch, "__version__", None),
        "numpy": packages.get("numpy") or getattr(np, "__version__", None),
        "git": {
            "commit": git.get("commit"),
            "branch": git.get("branch"),
            "dirty": bool(git.get("dirty")),
            "dirty_files": int(git.get("dirty_files") or 0),
        },
        "history_len": len(history_list),
        "final_return": history_list[-1] if history_list else None,
        "state_dict_sha256": fingerprint,
        "agent": agent or caller or "driving.checkpoints.save_skill",
        "meta_name": meta.name,
    }
    if seed is not None:
        provenance["seed"] = seed
    if updates is not None:
        provenance["updates"] = updates
    protected = set(provenance)
    for key, value in values.items():
        if key not in protected:
            provenance[key] = value
    if extra:
        provenance["extra_keys"] = sorted(str(k) for k in extra)
    return _json_safe(provenance)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _meta_mismatch(side_meta: Any, meta: Mapping[str, Any]) -> List[str]:
    if not isinstance(side_meta, Mapping):
        return ["sidecar meta is not a mapping"]
    mismatch: List[str] = []
    for key in META_FIELDS:
        if key not in side_meta:
            continue
        left = side_meta[key]
        right: Any = meta.get(key)
        try:
            if key in ("name",):
                equal = str(left) == str(right)
            elif key == "target_lane":
                equal = int(left) == int(right)
            else:
                equal = math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            equal = False
        if not equal:
            mismatch.append(key)
    return mismatch


def _inspect_sidecar(
    path: Path,
    *,
    meta: Mapping[str, Any],
    fingerprint: Optional[str],
    history_len: int,
    legacy: bool,
) -> Dict[str, Any]:
    """Summarize the JSON sidecar without raising.

    ``errors`` are always fatal (fingerprint mismatch). ``strict_errors`` are
    fatal only when the caller requested strict provenance validation.
    """

    sidecar_path = _sidecar_path(path)
    info: Dict[str, Any] = {
        "path": str(sidecar_path),
        "present": sidecar_path.is_file(),
        "format": None,
        "fingerprint": None,
        "fingerprint_matches": None,
        "meta_matches": None,
        "history_len": None,
        "history_len_matches": None,
        "warnings": [],
        "errors": [],
        "strict_errors": [],
    }
    if not sidecar_path.is_file():
        info["warnings"].append(f"{path}: JSON sidecar {sidecar_path.name} is missing")
        return info
    try:
        data = read_json(sidecar_path)
    except Exception as exc:  # pragma: no cover - message depends on JSON parser
        msg = f"{path}: cannot read sidecar {sidecar_path.name}: {exc}"
        info["warnings"].append(msg)
        info["strict_errors"].append(msg)
        return info
    if not isinstance(data, Mapping):
        msg = f"{path}: sidecar {sidecar_path.name} must contain a JSON object"
        info["warnings"].append(msg)
        info["strict_errors"].append(msg)
        return info

    if isinstance(data.get("meta"), Mapping):
        info["format"] = "nested"
        side_meta = data["meta"]
        side_fingerprint = data.get("fingerprint") or data.get("state_dict_sha256")
        side_history_len = data.get("history_len")
    elif "name" in data:
        info["format"] = "legacy_flat"
        side_meta = {key: data[key] for key in META_FIELDS if key in data}
        side_fingerprint = data.get("fingerprint") or data.get("state_dict_sha256")
        side_history_len = data.get("history_len")
    else:
        msg = f"{path}: sidecar {sidecar_path.name} has neither nested meta nor legacy fields"
        info["warnings"].append(msg)
        info["strict_errors"].append(msg)
        return info

    info["fingerprint"] = side_fingerprint
    if side_fingerprint is not None:
        if not _is_sha256_hex(side_fingerprint):
            msg = f"{path}: sidecar fingerprint is not a 64-char hex digest"
            info["warnings"].append(msg)
            info["strict_errors"].append(msg)
        elif fingerprint is not None:
            info["fingerprint_matches"] = side_fingerprint == fingerprint
            if not info["fingerprint_matches"]:
                msg = (
                    f"{path}: sidecar fingerprint mismatch "
                    f"(recorded {side_fingerprint}, computed {fingerprint})"
                )
                info["warnings"].append(msg)
                info["errors"].append(msg)
    elif not legacy:
        info["warnings"].append(f"{path}: sidecar has no state_dict fingerprint")

    mismatch = _meta_mismatch(side_meta, meta)
    info["meta_matches"] = not mismatch
    if mismatch:
        msg = f"{path}: sidecar meta differs from checkpoint meta for {mismatch}"
        info["warnings"].append(msg)
        info["strict_errors"].append(msg)

    if side_history_len is not None:
        info["history_len"] = side_history_len
        info["history_len_matches"] = (
            isinstance(side_history_len, numbers.Integral)
            and not isinstance(side_history_len, bool)
            and int(side_history_len) == history_len
        )
        if not info["history_len_matches"]:
            msg = (
                f"{path}: sidecar history_len={side_history_len!r} does not match "
                f"checkpoint history length {history_len}"
            )
            info["warnings"].append(msg)
            info["strict_errors"].append(msg)
    return info


def save_skill(
    path: Path,
    net: ActorCritic | CleanRLAgent,
    meta: SkillMeta,
    history: Optional[List[float]] = None,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Atomically save a v2 skill checkpoint and its JSON sidecar.

    The first four positional arguments match the legacy API. ``provenance``
    is merged into the automatically collected environment/git provenance;
    ``extra`` is stored verbatim in the ``.pt`` payload (and only its keys are
    listed in provenance).
    """

    path = Path(path)
    if not isinstance(net, (ActorCritic, CleanRLAgent)):
        raise ValidationError(f"{path}: unsupported policy network {type(net).__name__}")
    meta_obj = validate_skill_meta(meta, path=str(path))
    try:
        history_list = [float(value) for value in (history or [])]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{path}: history must be a sequence of numbers: {exc}") from exc
    state_dict = net.state_dict()
    fingerprint = state_dict_fingerprint(state_dict)
    provenance_dict = build_checkpoint_provenance(
        meta=meta_obj,
        fingerprint=fingerprint,
        history=history_list,
        supplied=provenance,
        extra=extra,
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state_dict": state_dict,
        "meta": asdict(meta_obj),
        "history": history_list,
        "provenance": provenance_dict,
        "state_dict_sha256": fingerprint,
        "extra": _json_safe(dict(extra or {})),
    }
    if isinstance(net, CleanRLAgent):
        payload["architecture"] = "cleanrl_ppo"
    buffer = io.BytesIO()
    try:
        torch.save(payload, buffer)
    except Exception as exc:  # pragma: no cover - torch serialization guard
        raise ArtifactError(f"{path}: cannot serialize checkpoint: {exc}") from exc
    atomic_write_bytes(path, buffer.getvalue())

    sidecar = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": provenance_dict["created_at"],
        "name": meta_obj.name,
        "meta": asdict(meta_obj),
        "history_len": len(history_list),
        "fingerprint": fingerprint,
        "state_dict_sha256": fingerprint,
        "provenance": provenance_dict,
    }
    atomic_write_json(_sidecar_path(path), sidecar)
    return {
        "path": str(path),
        "sidecar": str(_sidecar_path(path)),
        "fingerprint": fingerprint,
        "provenance": provenance_dict,
    }


def load_skill(
    path: Path,
    device: torch.device = DEVICE,
    *,
    strict: bool = False,
) -> SkillPolicy:
    """Load and validate a skill checkpoint.

    Legacy v1 payloads (no provenance) are accepted with a warning unless
    ``strict=True``. A recorded state-dict fingerprint is always verified,
    even in non-strict mode, because a mismatch means corruption or tampering.
    """

    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"{path}: checkpoint file not found")
    checkpoint = _load_torch_checkpoint(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ArtifactError(
            f"{path}: checkpoint payload must be a mapping, got {type(checkpoint).__name__}"
        )

    warnings: List[str] = []
    schema_version = checkpoint.get("schema_version")
    if schema_version is None:
        schema_version = LEGACY_SCHEMA_VERSION
    else:
        if isinstance(schema_version, bool) or not isinstance(schema_version, numbers.Integral):
            raise ArtifactError(
                f"{path}: schema_version must be an integer, got {schema_version!r}"
            )
        schema_version = int(schema_version)
    if schema_version > CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactError(
            f"{path}: unsupported checkpoint schema_version {schema_version}; "
            f"this loader supports <= {CHECKPOINT_SCHEMA_VERSION}"
        )

    provenance = checkpoint.get("provenance")
    legacy = (
        schema_version < CHECKPOINT_SCHEMA_VERSION
        or not isinstance(provenance, Mapping)
        or not provenance
    )
    if legacy:
        message = (
            f"{path}: legacy checkpoint (schema_version={schema_version}) has no "
            "v2 provenance block; loading in compatibility mode"
        )
        logger.warning(message)
        warnings.append(message)
        if strict:
            raise ArtifactError(message)
    else:
        provenance_problems = _validate_provenance(provenance, path=str(path))
        if provenance_problems:
            warnings.extend(provenance_problems)
            logger.warning("%s: %s", path, "; ".join(provenance_problems))
            if strict:
                raise ArtifactError(f"{path}: invalid provenance: {provenance_problems[0]}")

    if "state_dict" not in checkpoint:
        raise ArtifactError(f"{path}: missing required payload key 'state_dict'")
    if "meta" not in checkpoint:
        raise ArtifactError(f"{path}: missing required payload key 'meta'")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ArtifactError(
            f"{path}: state_dict must be a mapping, got {type(state_dict).__name__}"
        )
    meta = validate_skill_meta(checkpoint["meta"], path=str(path))
    try:
        fingerprint = state_dict_fingerprint(state_dict)
    except ValidationError as exc:
        raise ArtifactError(f"{path}: invalid state_dict: {exc}") from exc

    recorded = _recorded_fingerprints(checkpoint, provenance)
    for value in recorded:
        if not _is_sha256_hex(value):
            raise ArtifactError(
                f"{path}: recorded state_dict fingerprint must be a 64-char hex digest, "
                f"got {value!r}"
            )
    unique_recorded = set(str(value) for value in recorded)
    if len(unique_recorded) > 1:
        raise ArtifactError(
            f"{path}: conflicting recorded state_dict fingerprints {sorted(unique_recorded)}"
        )
    if unique_recorded and fingerprint not in unique_recorded:
        raise ArtifactError(
            f"{path}: state_dict fingerprint mismatch "
            f"(recorded {sorted(unique_recorded)[0]}, computed {fingerprint})"
        )

    architecture = checkpoint.get("architecture", "dual_critic")
    net: ActorCritic | CleanRLAgent
    if architecture == "dual_critic":
        net = ActorCritic().to(device)
    elif architecture == "cleanrl_ppo":
        net = CleanRLAgent().to(device)
    else:
        raise ArtifactError(f"{path}: unsupported policy architecture {architecture!r}")
    try:
        net.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ArtifactError(
            f"{path}: state_dict is incompatible with {architecture}: {exc}"
        ) from exc

    history = checkpoint.get("history", [])
    if not isinstance(history, (list, tuple)):
        warnings.append(f"{path}: history is not a sequence; treating it as empty")
        history_len = 0
    else:
        history_len = len(history)

    sidecar = _inspect_sidecar(
        path,
        meta=asdict(meta),
        fingerprint=fingerprint,
        history_len=history_len,
        legacy=legacy,
    )
    warnings.extend(sidecar.get("warnings", []))
    if sidecar.get("errors"):
        raise ArtifactError(str(sidecar["errors"][0]))
    if strict and sidecar.get("strict_errors"):
        raise ArtifactError(str(sidecar["strict_errors"][0]))
    for message in sidecar.get("warnings", []):
        logger.warning(message)

    info = {
        "path": str(path),
        "schema_version": schema_version,
        "legacy": legacy,
        "fingerprint": fingerprint,
        "state_dict_sha256": fingerprint,
        "recorded_fingerprints": [str(value) for value in recorded],
        "provenance": _json_safe(provenance) if isinstance(provenance, Mapping) else None,
        "history_len": history_len,
        "sidecar": sidecar,
        "warnings": warnings,
        "strict": bool(strict),
        "meta_name": meta.name,
    }
    return SkillPolicy(meta.name, net, meta, checkpoint_info=info)


def find_duplicate_fingerprints(fingerprints: Mapping[str, Optional[str]]) -> List[Dict[str, Any]]:
    """Group distinct skill names that share a state-dict fingerprint."""

    groups: Dict[str, List[str]] = {}
    for name, fingerprint in fingerprints.items():
        if fingerprint:
            groups.setdefault(str(fingerprint), []).append(str(name))
    duplicates: List[Dict[str, Any]] = []
    for fingerprint, names in sorted(groups.items()):
        if len(names) > 1:
            duplicates.append(
                {
                    "fingerprint": fingerprint,
                    "skills": sorted(names),
                    "copies": len(names),
                }
            )
    return duplicates


def inspect_checkpoint(path: Path) -> Dict[str, Any]:
    """Return a strict-JSON-safe summary of a checkpoint without loading a net."""

    path = Path(path)
    summary: Dict[str, Any] = {
        "path": str(path),
        "skill": path.stem,
        "exists": path.is_file(),
        "schema_version": None,
        "legacy": None,
        "keys": [],
        "state_dict_keys": [],
        "meta": None,
        "name": None,
        "fingerprint": None,
        "recorded_fingerprints": [],
        "fingerprint_matches": None,
        "provenance": None,
        "history_len": 0,
        "learned": None,
        "sidecar": {"present": _sidecar_path(path).is_file()},
        "warnings": [],
        "errors": [],
    }
    if not path.is_file():
        summary["errors"].append(f"{path}: checkpoint file not found")
        return _json_safe(summary)
    try:
        checkpoint = _load_torch_checkpoint(path, map_location="cpu")
    except ArtifactError as exc:
        summary["errors"].append(str(exc))
        return _json_safe(summary)
    if not isinstance(checkpoint, Mapping):
        summary["errors"].append(
            f"{path}: checkpoint payload must be a mapping, got {type(checkpoint).__name__}"
        )
        return _json_safe(summary)

    summary["keys"] = sorted(str(key) for key in checkpoint)
    schema_version = checkpoint.get("schema_version")
    if schema_version is None:
        schema_version = LEGACY_SCHEMA_VERSION
    summary["schema_version"] = _json_safe(schema_version)
    provenance = checkpoint.get("provenance")
    summary["provenance"] = _json_safe(provenance) if isinstance(provenance, Mapping) else None
    summary["legacy"] = (
        not isinstance(schema_version, numbers.Integral)
        or int(schema_version) < CHECKPOINT_SCHEMA_VERSION
        or summary["provenance"] is None
    )

    meta_raw = checkpoint.get("meta")
    if isinstance(meta_raw, Mapping):
        summary["meta"] = _json_safe(meta_raw)
        summary["name"] = meta_raw.get("name")
    try:
        meta = validate_skill_meta(meta_raw, path=str(path))
        summary["meta"] = asdict(meta)
        summary["name"] = meta.name
        if meta.name != path.stem:
            summary["warnings"].append(
                f"{path}: meta name {meta.name!r} differs from file stem {path.stem!r}"
            )
    except ValidationError as exc:
        summary["errors"].append(str(exc))
        meta = None
    if summary["legacy"]:
        summary["warnings"].append(f"{path}: legacy checkpoint without v2 provenance")

    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, Mapping):
        summary["state_dict_keys"] = sorted(str(key) for key in state_dict)
        try:
            summary["fingerprint"] = state_dict_fingerprint(state_dict)
        except ValidationError as exc:
            summary["errors"].append(f"{path}: {exc}")
    else:
        summary["errors"].append(f"{path}: missing or invalid state_dict")

    recorded = _recorded_fingerprints(checkpoint, provenance)
    summary["recorded_fingerprints"] = _json_safe(recorded)
    if recorded and summary["fingerprint"] is not None:
        valid_recorded = [value for value in recorded if _is_sha256_hex(value)]
        if len(valid_recorded) != len(recorded):
            summary["errors"].append(f"{path}: invalid recorded fingerprint value")
        summary["fingerprint_matches"] = all(
            value == summary["fingerprint"] for value in valid_recorded
        ) and len(valid_recorded) == len(recorded)
        if not summary["fingerprint_matches"]:
            summary["errors"].append(
                f"{path}: state_dict fingerprint mismatch (recorded {recorded!r}, "
                f"computed {summary['fingerprint']})"
            )

    history = checkpoint.get("history")
    if isinstance(history, (list, tuple)):
        summary["history_len"] = len(history)
    elif history is not None:
        summary["warnings"].append(f"{path}: history is not a sequence")
    if isinstance(provenance, Mapping):
        learned = provenance.get("learned")
        diagnostics = provenance.get("diagnostics")
        if isinstance(learned, bool):
            summary["learned"] = learned
        elif isinstance(diagnostics, Mapping) and isinstance(diagnostics.get("learned"), bool):
            summary["learned"] = diagnostics["learned"]
    else:
        extra = checkpoint.get("extra")
        if isinstance(extra, Mapping):
            diagnostics = extra.get("diagnostics")
            if isinstance(diagnostics, Mapping) and isinstance(diagnostics.get("learned"), bool):
                summary["learned"] = diagnostics["learned"]

    sidecar = _inspect_sidecar(
        path,
        meta=asdict(meta) if isinstance(meta, SkillMeta) else (summary["meta"] or {}),
        fingerprint=summary["fingerprint"],
        history_len=int(summary["history_len"]),
        legacy=bool(summary["legacy"]),
    )
    summary["sidecar"] = sidecar
    summary["warnings"].extend(sidecar.get("warnings", []))
    summary["errors"].extend(sidecar.get("errors", []))
    return _json_safe(summary)
