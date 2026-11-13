"""Atomic, strict, provenance-friendly artifact I/O.

Research scripts commonly write JSON and Markdown directly. That is fine until
a run is interrupted (leaving half a file) or a NaN leaks into a results
artifact (which strict JSON consumers reject). All outputs in this package go
through the helpers here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from driving.errors import ArtifactError

PathLike = Union[str, Path]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if hasattr(value, "value") and hasattr(value, "name"):  # Enum
        return value.value
    if hasattr(value, "tolist"):  # numpy scalar/array
        return value.tolist()
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def strict_json_dumps(obj: Any, *, indent: Optional[int] = 2, sort_keys: bool = False) -> str:
    """Serialize JSON with NaN/Infinity rejected and enums/numpy handled."""

    try:
        return json.dumps(
            obj,
            indent=indent,
            sort_keys=sort_keys,
            allow_nan=False,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"refusing to write non-strict JSON: {exc}") from exc


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def strict_json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant)


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text via a same-directory temp file and an atomic rename."""

    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding=encoding) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
            try:
                dir_fd = os.open(str(target.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    except OSError as exc:
        raise ArtifactError(f"cannot write {target}: {exc}") from exc
    return target


def atomic_write_bytes(path: PathLike, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except OSError as exc:
        raise ArtifactError(f"cannot write {target}: {exc}") from exc
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return target


def atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    indent: Optional[int] = 2,
    sort_keys: bool = True,
) -> Path:
    return atomic_write_text(
        path, strict_json_dumps(obj, indent=indent, sort_keys=sort_keys) + "\n"
    )


def read_json(path: PathLike) -> Any:
    target = Path(path)
    try:
        return strict_json_loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactError(f"cannot read {target}: {exc}") from exc
    except ValueError as exc:
        raise ArtifactError(f"invalid JSON in {target}: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    """Stable SHA-256 of a JSON-serializable object (order-independent maps)."""

    return sha256_bytes(strict_json_dumps(obj, indent=None, sort_keys=True).encode("utf-8"))


def json_safe(value: Any) -> Any:
    """Recursively convert a value to something ``strict_json_dumps`` accepts."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Enums, numpy scalars, and similar objects.
    if hasattr(value, "value") and hasattr(value, "name"):
        return json_safe(value.value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None
