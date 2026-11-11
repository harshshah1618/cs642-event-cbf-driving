"""Record/replay fixed illustrative cells; compare against complete main rows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from event_cbf_driving.runtime import replay_open_loop, write_episode
from event_cbf_driving.stack.result import RolloutResult

import component_ablation as experiment

TARGET = experiment.ROOT.parent / "component-study/results/traces"


def selected():
    jobs = [
        experiment.Job("side_occupied", 5001, 0.125, "front_blind_rule", "ppo", *mask)
        for mask in [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (False, False, False),
        ]
    ]
    jobs += [
        experiment.Job("lead_brakes", 5001, 0.125, "copy_current", "ppo", *mask)
        for mask in [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ]
    ]
    jobs += [
        experiment.Job("side_occupied", 5001, 0.125, "front_blind_rule", "pd", *mask)
        for mask in [(True, True, True), (False, False, False)]
    ]
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("record", "verify"))
    args = p.parse_args()
    experiment.verify()
    TARGET.mkdir(parents=True, exist_ok=True)
    summary_path = TARGET / "recorded-summary.json"
    if args.stage == "verify":
        rows = experiment.read_rows(experiment.OUT / "test.jsonl", experiment.jobs("test"))
        by_id = {r["id"]: r for r in rows}
        recorded = json.loads(summary_path.read_text())
        for item in recorded:
            expected = by_id[item["row"]["id"]]
            for metric in (
                "route_pct",
                "crashed",
                "router_calls",
                "end_time",
                "admission_rejections",
                "cbf_override_fraction",
                "braking_override_fraction",
                "achieved_lateral_jerk_rms",
            ):
                if abs(item["row"][metric] - expected[metric]) > 1e-6:
                    raise ValueError(f"recorded/replayed cell differs from frozen row: {metric}")
        (TARGET / "main-comparison.json").write_text(
            json.dumps(
                {"matched_main_cells": len(recorded), "all_metrics_identical": True}, indent=2
            )
            + "\n"
        )
        print("Representative reruns match all frozen main outcomes")
        return
    if summary_path.exists():
        raise ValueError("representative recordings already exist; use verify")
    experiment.initialize()
    recordings = []
    for job in selected():
        row = experiment.execute(job, save_trace=True)
        raw = experiment.OUT / "traces" / f"{job.key()}.json"
        payload = json.loads(raw.read_text())
        if payload.get("min_barrier_h") is None:
            payload["min_barrier_h"] = float("inf")
        result = RolloutResult(**payload)
        if len(result.actions) != len(result.interactions):
            raise ValueError("trace incomplete")
        for action, tick in zip(result.actions, result.interactions, strict=True):
            if (
                action != tick["applied_action"]
                or action != tick["plant"]["requested_acceleration"]
            ):
                raise ValueError("action chain mismatch")
        path = TARGET / f"{job.key()}.json"
        write_episode(path, result, scenario=experiment.scene_spec(job))
        replay = replay_open_loop(path, tolerance=1e-4)
        if not replay.ok:
            raise ValueError(replay.errors)
        recordings.append(
            {"job": asdict(job), "path": path.name, "row": row, "replay": replay.to_dict()}
        )
        print(
            job.scene,
            job.controller,
            job.cbf,
            job.braking,
            job.admission,
            "crash",
            row["crashed"],
            "replay",
            replay.ok,
            flush=True,
        )
    summary_path.write_text(json.dumps(recordings, indent=2) + "\n")


if __name__ == "__main__":
    main()
