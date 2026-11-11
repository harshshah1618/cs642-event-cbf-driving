"""Validate final saved action chains and replay seven representative episodes.

Run from the package directory after producing the final study datasets.
Missing datasets or any disagreement fail the audit; inference is not rerun.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from event_cbf_driving.io_utils import atomic_write_json
from event_cbf_driving.runtime import load_episode, replay_open_loop


def main():
    patterns = {
        "runs/metadrive-study/study-final/episodes/*.json": 90,
        "runs/robustness-final/*/episodes/*.json": 42,
        "runs/metadrive-local/live-qwen-final/episodes/*.json": 18,
        "runs/metadrive-local/matched-rule/live-qwen-final-*/episodes/*.json": 18,
    }
    paths = []
    for pattern, count in patterns.items():
        found = sorted(Path().glob(pattern))
        if len(found) != count:
            raise ValueError(f"{pattern}: expected {count} episodes, found {len(found)}")
        paths.extend(found)
    ticks = 0
    for path in paths:
        result = load_episode(path).result
        if len(result.actions) != len(result.interactions):
            raise ValueError(f"{path}: incomplete interaction log")
        for index, (action, row) in enumerate(
            zip(result.actions, result.interactions, strict=True)
        ):
            if not (
                np.array_equal(action, row["applied_action"])
                and np.array_equal(action, row["plant"]["requested_acceleration"])
            ):
                raise ValueError(f"{path}: action-chain mismatch at tick {index}")
            ticks += 1
    selected = [
        Path(f"runs/metadrive-study/study-final/episodes/{name}-proposed-seed0.json")
        for name in ("dense_highway", "merge", "two_trap")
    ]
    selected.append(
        Path(
            "runs/robustness-final/target_lane_brakes/episodes/dense_highway-proposed-seed307.json"
        )
    )
    selected.extend(
        Path(f"runs/metadrive-local/live-qwen-final/episodes/{name}-proposed-seed11.json")
        for name in ("dense_highway", "merge", "two_trap")
    )
    replays = []
    for path in selected:
        report = replay_open_loop(path, tolerance=0.0001)
        if not report.ok:
            raise ValueError(f"{path}: {report.errors}")
        replays.append({"episode": str(path), **report.to_dict()})
    audit = {"episodes": len(paths), "interaction_actions": ticks, "replays": replays}
    atomic_write_json(Path("outputs/research-study/audit.json"), audit)
    print(f"Validated {len(paths)} episodes, {ticks} action chains, {len(replays)} replays")


if __name__ == "__main__":
    main()
