"""Restore packaged raw rows and locks to ignored run directories for analysis."""

from __future__ import annotations

import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "research/results"


def write_identical_or_new(target: Path, contents: bytes) -> None:
    if target.exists() and target.read_bytes() != contents:
        raise ValueError(f"Refusing to replace different existing data: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(contents)


def main() -> None:
    for suite in ("isolation", "roads", "phase"):
        dest = ROOT / "runs" / f"research-{suite}"
        for stage in ("validation", "test"):
            write_identical_or_new(
                dest / f"{stage}.jsonl",
                gzip.decompress((RESULTS / "data" / f"{suite}-{stage}.jsonl.gz").read_bytes()),
            )
        write_identical_or_new(
            dest / "locked-selection.json",
            (RESULTS / f"{suite}-locked-selection.json").read_bytes(),
        )
    for suite, stem in (("controller", "controller-ablation"), ("clock", "clock-control")):
        dest = ROOT / "runs" / f"research-{suite}"
        write_identical_or_new(
            dest / "test.jsonl",
            gzip.decompress((RESULTS / "data" / f"{stem}.jsonl.gz").read_bytes()),
        )
        lock_name = "clock-lock.json" if suite == "clock" else "controller-ablation-lock.json"
        write_identical_or_new(dest / "lock.json", (RESULTS / lock_name).read_bytes())
    write_identical_or_new(
        ROOT / "runs/research-clock/analysis.json", (RESULTS / "clock-analysis.json").read_bytes()
    )
    for source in (RESULTS / "live-qwen").rglob("*"):
        if source.is_file():
            write_identical_or_new(
                ROOT / "runs/research-live/exploratory-single-seed"
                / source.relative_to(RESULTS / "live-qwen"),
                source.read_bytes(),
            )
    print("Packaged rows and locks restored; existing differing files were not replaced.")


if __name__ == "__main__":
    main()
