"""Atomic, strict, provenance-friendly artifact I/O.

Research scripts commonly write JSON and Markdown directly. That is fine until
a run is interrupted (leaving half a file) or a NaN leaks into a results
artifact (which strict JSON consumers reject). All outputs in this package go
through the helpers here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

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
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
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
    return atomic_write_text(path, strict_json_dumps(obj, indent=indent, sort_keys=sort_keys) + "\n")


def read_json(path: PathLike) -> Any:
    target = Path(path)
    try:
        return strict_json_loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactError(f"cannot read {target}: {exc}") from exc
    except ValueError as exc:
        raise ArtifactError(f"invalid JSON in {target}: {exc}") from exc


def append_jsonl(path: PathLike, records: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(strict_json_dumps(record, indent=None) + "\n")
    except OSError as exc:
        raise ArtifactError(f"cannot append {target}: {exc}") from exc
    return target


def read_jsonl(path: PathLike) -> list[Dict[str, Any]]:
    target = Path(path)
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactError(f"cannot read {target}: {exc}") from exc
    out: list[Dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = strict_json_loads(line)
        except ValueError as exc:
            raise ArtifactError(f"invalid JSONL in {target}:{i}: {exc}") from exc
        if not isinstance(value, dict):
            raise ArtifactError(f"{target}:{i}: expected an object")
        out.append(value)
    return out


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    """Stable SHA-256 of a JSON-serializable object (order-independent maps)."""

    return sha256_bytes(strict_json_dumps(obj, indent=None, sort_keys=True).encode("utf-8"))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "run"


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slugify(prefix)}-{stamp}-{secrets.token_hex(3)}"


@dataclass
class RunDirectory:
    """A self-contained artifact directory for one experiment.

    Layout::

        <root>/<run_id>/
            manifest.json
            results.json
            report.md
            logs/run.log
            figures/
            episodes/
    """

    root: Path
    run_id: str

    @classmethod
    def create(cls, root: PathLike = "runs", run_id: Optional[str] = None) -> "RunDirectory":
        run_id = run_id or new_run_id()
        base = Path(root) / run_id
        try:
            for sub in ("logs", "figures", "episodes"):
                (base / sub).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactError(f"cannot create run directory {base}: {exc}") from exc
        return cls(root=base, run_id=run_id)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def results_path(self) -> Path:
        return self.root / "results.json"

    @property
    def report_path(self) -> Path:
        return self.root / "report.md"

    @property
    def log_path(self) -> Path:
        return self.root / "logs" / "run.log"

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        return atomic_write_json(self.manifest_path, dict(manifest))

    def write_results(self, results: Mapping[str, Any]) -> Path:
        return atomic_write_json(self.results_path, dict(results))

    def write_report(self, markdown: str) -> Path:
        return atomic_write_text(self.report_path, markdown)

    def figure_path(self, name: str) -> Path:
        return self.root / "figures" / name

    def episode_path(self, name: str) -> Path:
        return self.root / "episodes" / name
