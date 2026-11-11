"""Restore archived component raw rows without replacing differing evidence."""

from __future__ import annotations

import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def restore(target, data):
    if target.exists() and target.read_bytes() != data:
        raise ValueError(f"refusing to replace differing existing evidence: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def main():
    source = ROOT.parent / "component-study/results"
    target = ROOT / "runs/components"
    for stage in ("pilot", "test"):
        restore(target/f"{stage}.jsonl", gzip.decompress((source/f"{stage}.jsonl.gz").read_bytes()))
    restore(target/"lock.json", (source/"lock.json").read_bytes())
    print("Restored archived rows and source lock; differing existing files were not replaced")


if __name__ == "__main__":
    main()
