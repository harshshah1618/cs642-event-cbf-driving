"""Dual-critic PPO skills with epistemic uncertainty for interrupt gating.

Architecture mirrors the validated toy trainer: relative (y−y*) encoding,
tanh-Gaussian with 0→0 accel map, linear lateral skip, bootstrap critics.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import numbers
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from torch.distributions import Normal

from event_cbf_driving import DT
from event_cbf_driving.env.types import AX_HIGH, AX_LOW, AY_HIGH, AY_LOW
from event_cbf_driving.errors import ArtifactError, ValidationError
from event_cbf_driving.io_utils import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    utc_now,
)
from event_cbf_driving.provenance import environment_info, git_info

DEVICE = torch.device("cpu")
TARGET_SPEED = 20.0


def encode(y: ArrayLike, vy: ArrayLike, vx: ArrayLike, y_target: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    vy = np.asarray(vy, dtype=np.float32)
    vx = np.asarray(vx, dtype=np.float32)
    return np.stack(
        [(y - np.float32(y_target)) / np.float32(2.0), vy / np.float32(2.0), (vx - np.float32(20.0)) / np.float32(5.0)],
        axis=-1,
    ).astype(np.float32)


def env_action_np(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map tanh actions with 0 at the origin, then clip to the accel box."""
    ax = np.clip(6.0 * a[..., 0], AX_LOW, AX_HIGH)
    ay = np.clip(5.0 * a[..., 1], AY_LOW, AY_HIGH)
    return ax, ay


def _mlp(in_dim, hidden, out_dim, out_gain):
    net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.414)
            nn.init.zeros_(m.bias)
    output_layer = cast(nn.Linear, net[-1])
    nn.init.orthogonal_(output_layer.weight, gain=out_gain)
    nn.init.zeros_(output_layer.bias)
    return net


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, 2)
        self.k_ax = nn.Parameter(torch.zeros(1))
        self.k_e = nn.Parameter(torch.zeros(1))
        self.k_v = nn.Parameter(torch.zeros(1))
        self.log_std = nn.Parameter(torch.full((2,), -1.6))
        self.v1 = _mlp(3, hidden, 1, 1.0)
        self.v2 = _mlp(3, hidden, 1, 1.0)
        for m in (self.fc1, self.fc2):
            nn.init.orthogonal_(m.weight, gain=1.414)
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)

    def _std(self):
        return self.log_std.clamp(-2.8, -1.2).exp()

    def _mu(self, obs):
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        base = self.mu(h)
        ax = base[:, 0] + self.k_ax * obs[:, 2]
        ay = base[:, 1] + self.k_e * obs[:, 0] + self.k_v * obs[:, 1]
        return torch.stack([ax, ay], dim=-1)

    def _dist(self, obs):
        return Normal(self._mu(obs), self._std())

    def values(self, obs):
        return self.v1(obs).squeeze(-1), self.v2(obs).squeeze(-1)

    def act(self, obs, deterministic=False):
        dist = self._dist(obs)
        u = dist.mean if deterministic else dist.sample()
        a = torch.tanh(u)
        logp = dist.log_prob(u) - torch.log(1.0 - a.pow(2) + 1e-6)
        v1, v2 = self.values(obs)
        return a, logp.sum(-1), v1, v2

    def log_prob(self, obs, a):
        dist = self._dist(obs)
        a = a.clamp(-0.999999, 0.999999)
        u = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        logp = dist.log_prob(u) - torch.log(1.0 - a.pow(2) + 1e-6)
        return logp.sum(-1), dist.entropy().sum(-1)


def pi_parameters(net: ActorCritic):
    return (
        list(net.fc1.parameters())
        + list(net.fc2.parameters())
        + list(net.mu.parameters())
        + [net.k_ax, net.k_e, net.k_v, net.log_std]
    )


def v_parameters(net: ActorCritic):
    return list(net.v1.parameters()) + list(net.v2.parameters())


@dataclass
class SkillMeta:
    name: str
    target_lane: int
    y_target: float
    threshold: float
    sigma_mean: float
    sigma_std: float
    spread: float


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


class SkillPolicy:
    def __init__(
        self,
        name: str,
        net: ActorCritic,
        meta: SkillMeta,
        checkpoint_info: Optional[Mapping[str, Any]] = None,
    ):
        self.name = name
        self.net = net
        self.meta = meta
        self.checkpoint_info: Dict[str, Any] = dict(checkpoint_info or {})
        self.source_path = self.checkpoint_info.get("path")
        self.net.eval()

    def act(self, y: float, vy: float, vx: float, deterministic: bool = True):
        o = torch.from_numpy(encode(y, vy, vx, self.meta.y_target)).view(1, 3)
        with torch.no_grad():
            a, _, v1, v2 = self.net.act(o, deterministic=deterministic)
        ax, ay = env_action_np(a.numpy()[0])
        raw = abs(float(v1) - float(v2))
        sigma = raw / max(self.meta.spread, 1e-6)
        return float(ax), float(ay), float(sigma)

    def interrupt(self, sigma: float) -> bool:
        return sigma > self.meta.threshold


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
        raise ValidationError(
            f"{path}: meta['target_lane'] must be an integer, got {lane!r}"
        )
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
        "agent": agent or caller or "event_cbf_driving.skills.ppo.save_skill",
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
    net: ActorCritic,
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
    if not isinstance(net, ActorCritic):
        raise ValidationError(f"{path}: net must be an ActorCritic, got {type(net).__name__}")
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
    legacy = schema_version < CHECKPOINT_SCHEMA_VERSION or not isinstance(provenance, Mapping) or not provenance
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

    net = ActorCritic().to(device)
    try:
        net.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ArtifactError(f"{path}: state_dict is incompatible with ActorCritic: {exc}") from exc

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


# ---------------------------------------------------------------------------
# Vectorized empty-road trainer
# ---------------------------------------------------------------------------


class EmptyRoadVec:
    def __init__(self, n: int, y_target: float, rng: np.random.Generator):
        self.n = n
        self.yt = float(y_target)
        self.rng = rng
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        self.steps = np.zeros(n, np.int32)
        self.ep_ret = np.zeros(n)
        self.ep_steps = int(8.0 / DT)
        self.reset_mask(np.ones(n, dtype=bool))

    def obs(self):
        return encode(self.y, self.vy, self.vx, self.yt)

    def reset_mask(self, mask):
        idx = np.flatnonzero(mask)
        k = idx.size
        if k == 0:
            return
        draw = self.rng.random(k)
        y = np.empty(k)
        vx = np.empty(k)
        near = draw < 0.55
        slow = (draw >= 0.55) & (draw < 0.80)
        wide = ~near & ~slow
        y[near] = self.yt + self.rng.uniform(-1.5, 1.5, int(near.sum()))
        vx[near] = self.rng.uniform(12.0, 22.0, int(near.sum()))
        y[slow] = self.yt + self.rng.uniform(-1.0, 1.0, int(slow.sum()))
        vx[slow] = self.rng.uniform(0.0, 8.0, int(slow.sum()))
        y[wide] = self.yt + self.rng.uniform(-4.5, 4.5, int(wide.sum()))
        vx[wide] = self.rng.uniform(12.0, 22.0, int(wide.sum()))
        self.y[idx] = y
        self.vy[idx] = self.rng.uniform(-1.5, 1.5, k)
        self.vx[idx] = vx
        self.x[idx] = 0.0
        self.steps[idx] = 0
        self.ep_ret[idx] = 0.0

    def step(self, ax, ay):
        ax = np.clip(np.asarray(ax, np.float64), AX_LOW, AX_HIGH)
        ay = np.clip(np.asarray(ay, np.float64), AY_LOW, AY_HIGH)
        self.x = self.x + self.vx * DT + 0.5 * ax * DT * DT
        self.y = self.y + self.vy * DT + 0.5 * ay * DT * DT
        self.vx = np.maximum(0.0, self.vx + ax * DT)
        self.vy = self.vy + ay * DT
        self.steps += 1
        err = self.y - self.yt
        rew = 1.0 - err**2 - 0.05 * (self.vx - 20.0) ** 2 - 0.02 * ax**2 - 0.35 * ay**2 - 0.05 * self.vy**2
        fail = np.abs(err) > 7.0
        timeout = self.steps >= self.ep_steps
        done = fail | timeout
        rew = np.where(fail, rew - 5.0, rew)
        # Bootstrap features from the pre-reset terminal state (matches validated toy).
        boot = encode(self.y, self.vy, self.vx, self.yt)
        self.ep_ret += rew
        finished = self.ep_ret[done].copy()
        self.reset_mask(done)
        return (
            self.obs(),
            rew.astype(np.float32),
            done.astype(np.float32),
            fail.astype(np.float32),
            boot,
            finished,
        )


def compute_gae(rew, val, v_boot, done, gamma=0.99, lam=0.95):
    t_steps, n = rew.shape
    adv = np.zeros((t_steps, n), np.float32)
    last = np.zeros(n, np.float32)
    for t in range(t_steps - 1, -1, -1):
        delta = rew[t] + gamma * v_boot[t] - val[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + val


class PPOTrainer:
    def __init__(
        self,
        name: str,
        y_target: float,
        target_lane: int,
        n_envs: int = 16,
        seed: int = 0,
        pi_lr: float = 3e-4,
        v_lr: float = 1e-3,
    ):
        self.name = name
        self.y_target = float(y_target)
        self.target_lane = int(target_lane)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.env = EmptyRoadVec(n_envs, y_target, np.random.default_rng(seed + 17))
        self.net = ActorCritic().to(DEVICE)
        self.opt_pi = torch.optim.Adam(pi_parameters(self.net), lr=pi_lr)
        self.opt_v = torch.optim.Adam(v_parameters(self.net), lr=v_lr)
        self.history: List[float] = []
        self.rollout_steps = 400
        self.ppo_epochs = 8
        self.minibatch = 512
        self.clip = 0.2
        self.ent_coef = 0.0

    def collect(self):
        n = self.env.n
        T = self.rollout_steps
        obs_buf = np.zeros((T, n, 3), np.float32)
        act_buf = np.zeros((T, n, 2), np.float32)
        logp_buf = np.zeros((T, n), np.float32)
        rew_buf = np.zeros((T, n), np.float32)
        done_buf = np.zeros((T, n), np.float32)
        val_buf = np.zeros((T, n), np.float32)
        boot_buf = np.zeros((T, n), np.float32)
        returns: List[float] = []
        obs = self.env.obs()
        self.net.eval()
        for t in range(T):
            obs_buf[t] = obs
            o = torch.from_numpy(obs)
            with torch.no_grad():
                a, logp, v1, v2 = self.net.act(o, deterministic=False)
                v = 0.5 * (v1 + v2)
            act_buf[t] = a.numpy()
            logp_buf[t] = logp.numpy()
            val_buf[t] = v.numpy()
            ax, ay = env_action_np(act_buf[t])
            obs, rew, done, fail, boot, finished = self.env.step(ax, ay)
            rew_buf[t] = rew
            done_buf[t] = done
            with torch.no_grad():
                bv1, bv2 = self.net.values(torch.from_numpy(boot))
                bv = (0.5 * (bv1 + bv2)).numpy()
            boot_buf[t] = np.where(fail > 0.5, 0.0, bv).astype(np.float32)
            if finished.size:
                returns.extend(float(r) for r in finished)
        adv, ret = compute_gae(rew_buf, val_buf, boot_buf, done_buf)
        mean_ret = float(np.mean(returns)) if returns else float("nan")
        self.history.append(mean_ret)
        return {
            "obs": obs_buf.reshape(-1, 3),
            "act": act_buf.reshape(-1, 2),
            "logp": logp_buf.reshape(-1),
            "adv": adv.reshape(-1),
            "ret": ret.reshape(-1),
            "mean_return": mean_ret,
        }

    def update(self, batch):
        obs = torch.from_numpy(batch["obs"])
        act = torch.from_numpy(batch["act"])
        old_logp = torch.from_numpy(batch["logp"])
        adv = torch.from_numpy(batch["adv"])
        ret = torch.from_numpy(batch["ret"])
        adv = (adv - adv.mean()) / (adv.std().clamp(min=1e-6))
        n = obs.shape[0]
        self.net.train()
        pi_params = pi_parameters(self.net)
        v_params = v_parameters(self.net)
        for _ in range(self.ppo_epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, self.minibatch):
                j = perm[s : s + self.minibatch]
                if j.size < 64:
                    continue
                jj = torch.from_numpy(j.copy())
                logp, entropy = self.net.log_prob(obs[jj], act[jj])
                ratio = torch.exp(logp - old_logp[jj])
                unclipped = ratio * adv[jj]
                clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[jj]
                loss_pi = -torch.min(unclipped, clipped).mean() - self.ent_coef * entropy.mean()
                self.opt_pi.zero_grad(set_to_none=True)
                loss_pi.backward()
                nn.utils.clip_grad_norm_(pi_params, 0.5)
                self.opt_pi.step()

                v1, v2 = self.net.values(obs[jj])
                m1 = torch.bernoulli(torch.full((jj.shape[0],), 0.8))
                m2 = torch.bernoulli(torch.full((jj.shape[0],), 0.8))
                loss_v1 = ((v1 - ret[jj]).pow(2) * m1).sum() / m1.sum().clamp(min=1.0)
                loss_v2 = ((v2 - ret[jj]).pow(2) * m2).sum() / m2.sum().clamp(min=1.0)
                loss_v = 0.5 * 0.5 * (loss_v1 + loss_v2)
                self.opt_v.zero_grad(set_to_none=True)
                loss_v.backward()
                nn.utils.clip_grad_norm_(v_params, 0.5)
                self.opt_v.step()

    def train(self, n_updates: int = 140, min_updates: int = 40) -> ActorCritic:
        for i in range(n_updates):
            batch = self.collect()
            self.update(batch)
            if (i + 1) % 10 == 0 or (i + 1) == n_updates:
                diag = evaluate_skill(self.net, self.y_target)
                ok = bool(diag.get("learned"))
                print(
                    f"  [{self.name}] update {i+1}/{n_updates}  return={batch['mean_return']:.1f}  "
                    f"hold|y|={diag['hold']['mean_abs_y']:.3f}  acq|y|={diag['acquire']['mean_abs_y']:.3f}  "
                    f"{'OK' if ok else '...'}"
                )
                if ok and (i + 1) >= min_updates:
                    break
        return self.net


@torch.no_grad()
def calibrate_threshold(net: ActorCritic, y_target: float, seed: int = 0, n_steps: int = 4000, k_std: float = 3.0):
    rng = np.random.default_rng(seed)
    env = EmptyRoadVec(8, y_target, rng)
    spreads = []
    obs = env.obs()
    net.eval()
    for _ in range(n_steps // 8):
        o = torch.from_numpy(obs)
        a, _, v1, v2 = net.act(o, deterministic=False)
        spreads.append(torch.abs(v1 - v2).numpy())
        ax, ay = env_action_np(a.numpy())
        obs, _, _, _, _, _ = env.step(ax, ay)
    spread_array = np.concatenate(spreads)
    spread = float(max(np.std(spread_array), 1e-3))
    sigmas = spread_array / spread
    mean = float(sigmas.mean())
    std = float(sigmas.std() + 1e-8)
    threshold = mean + k_std * std
    return threshold, mean, std, spread


@torch.no_grad()
def evaluate_skill(net: ActorCritic, y_target: float) -> Dict:
    def rollout(y0, vy0, vx0, steps=500):
        y, vy, vx, x = float(y0), float(vy0), float(vx0), 0.0
        ys, vxs = [], []
        t16 = None
        other = 0.0 if y_target > 2.0 else 4.0
        for i in range(steps):
            o = torch.from_numpy(encode(y, vy, vx, y_target)).view(1, 3)
            a, _, _, _ = net.act(o, deterministic=True)
            ax, ay = env_action_np(a.numpy()[0])
            x = x + vx * DT + 0.5 * float(ax) * DT * DT
            y = y + vy * DT + 0.5 * float(ay) * DT * DT
            vx = max(0.0, vx + float(ax) * DT)
            vy = vy + float(ay) * DT
            ys.append(y)
            vxs.append(vx)
            if t16 is None and abs(y - other) >= 1.6:
                t16 = (i + 1) * DT
        ys_array = np.asarray(ys)
        return {
            "mean_abs_y": float(np.mean(np.abs(ys_array - y_target))),
            "mean_abs_v": float(np.mean(np.abs(np.asarray(vxs) - 20.0))),
            "final_y": float(ys_array[-1]),
            "final_vx": float(vxs[-1]),
            "t16": t16,
        }

    other = 0.0 if y_target > 2.0 else 4.0
    hold = rollout(y_target, 0.0, 20.0)
    acq = rollout(other, 0.0, 20.0)
    # Probe lateral authority from the other lane.
    o = torch.from_numpy(encode(other, 0.0, 20.0, y_target)).view(1, 3)
    a, _, _, _ = net.act(o, deterministic=True)
    _, ay_other = env_action_np(a.numpy()[0])
    learned = (
        hold["mean_abs_y"] < 0.30
        and hold["mean_abs_v"] < 1.0
        and abs(hold["final_y"] - y_target) < 0.35
        and 18.5 <= hold["final_vx"] <= 21.5
        and abs(acq["final_y"] - y_target) < 0.5
        and (acq["t16"] is not None and 1.0 <= acq["t16"] <= 2.4)
        and abs(float(ay_other)) >= 1.0
    )
    return {"hold": hold, "acquire": acq, "learned": learned, "ay_at_other": float(ay_other)}
