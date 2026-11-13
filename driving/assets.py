"""Read model directories and cached research archives."""

import hashlib
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile


@lru_cache(maxsize=16)
def _unpack(filename: str, modified_ns: int, size: int) -> Path:
    source = Path(filename)
    hasher = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    cache = Path(tempfile.gettempdir()) / "event-cbf-assets"
    destination = cache / digest
    if (destination / ".complete").is_file():
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        folder = Path(temporary)
        with ZipFile(source) as archive:
            for name in archive.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError(f"Invalid asset archive path: {name}")
            archive.extractall(folder)
        (folder / ".complete").touch()
        try:
            os.rename(folder, destination)
        except OSError:
            if not (destination / ".complete").is_file():
                raise
    return destination


def asset_directory(path: str | Path) -> Path:
    """Accept a normal directory or transparently unpack an immutable ZIP."""
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix == ".zip":
        stat = path.stat()
        return _unpack(str(path), stat.st_mtime_ns, stat.st_size)
    return path
