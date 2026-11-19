"""Class balancing and atomic, versioned checkpoints for cloud training."""

import json
import math
import shutil
import tempfile
import uuid
from collections import Counter
from pathlib import Path


def sampling_weights(labels, scheme="sqrt"):
    counts = Counter(labels)
    if not counts:
        raise ValueError("No training labels")
    if scheme == "balanced":
        return [1.0/counts[label] for label in labels]
    if scheme == "sqrt":
        return [min(10, math.sqrt(len(labels)/counts[label])) for label in labels]
    raise ValueError(f"Unknown sampling scheme: {scheme}")


def checkpoint_location(root):
    root = Path(root)
    if (root / "latest.json").exists():
        pointer = json.loads((root / "latest.json").read_text())
        directory = (root / pointer["directory"]).resolve()
        directory.relative_to(root.resolve())
        if not (directory / "checkpoint.json").exists():
            raise ValueError("Checkpoint pointer does not reference a complete snapshot")
        if json.loads((directory / "checkpoint.json").read_text())["step"] != pointer["step"]:
            raise ValueError("Checkpoint pointer and snapshot disagree")
        return directory / "adapter", directory / "optimizer.pt"
    if (root / "optimizer.pt").exists():
        return root / "last-adapter", root / "optimizer.pt"
    return None


def publish_checkpoint(root, step, save):
    """Publish a pointer only after every snapshot file is written successfully."""
    root = Path(root)
    directory = root / "checkpoints" / f"step-{step:06d}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite immutable checkpoint {directory}")
    temporary = Path(tempfile.mkdtemp(prefix=f".writing-{step}-", dir=directory.parent))
    try:
        save(temporary)
        if not (temporary / "adapter").is_dir() or not (temporary / "optimizer.pt").is_file():
            raise ValueError("Snapshot must contain both adapter and optimizer state")
        (temporary / "checkpoint.json").write_text(json.dumps(dict(step=step)))
        temporary.rename(directory)
        pointer = root / f"latest-{uuid.uuid4().hex}.tmp"
        pointer.write_text(json.dumps(dict(step=step, directory=str(directory.relative_to(root)))))
        pointer.replace(root / "latest.json")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return directory
