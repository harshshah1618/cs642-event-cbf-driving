"""Re-record and replay a fixed representative subset of the frozen native test."""

from __future__ import annotations

import json
from dataclasses import asdict

from event_cbf_driving.runtime import load_episode, replay_open_loop, write_episode

import research_isolation as base
from research_freeze import verify


def main():
    verify("isolation")
    target = base.ROOT.parent / "research/results/traces"
    target.mkdir(parents=True, exist_ok=True)
    selection = json.loads(
        (base.ROOT / "runs/research-isolation/locked-selection.json").read_text()
    )
    base.initialize()
    original_run = base.run_episode
    replay_reports = []
    heldout_path = base.ROOT / "runs/research-isolation/test.jsonl"
    heldout = {
        r["id"]: r for r in (json.loads(line) for line in heldout_path.read_text().splitlines())
    }

    def record(*args, **kwargs):
        kwargs["stack_config"].record_interactions = True
        result = original_run(*args, **kwargs)
        path = target / f"{result.scenario}-{CURRENT.method}-seed{result.seed}.json"
        write_episode(path, result, scenario=args[1])
        episode = load_episode(path)
        if len(episode.result.actions) != len(episode.result.interactions):
            raise ValueError("incomplete trace")
        for action, row in zip(episode.result.actions, episode.result.interactions, strict=True):
            if action != row["applied_action"] or action != row["plant"]["requested_acceleration"]:
                raise ValueError("action-chain mismatch")
        replay = replay_open_loop(path, tolerance=1e-4)
        if not replay.ok:
            raise ValueError(replay.errors)
        replay_reports.append({"job": asdict(CURRENT), "path": str(path.name), **replay.to_dict()})
        return result

    base.run_episode = record
    for scenario in ("two_trap", "merge", "dense_highway"):
        for method, value in (
            ("deadline", selection["deadline"]),
            ("periodic", selection["periodic"]),
            ("periodic_gated", 10.0),
        ):
            global CURRENT
            CURRENT = base.Job(scenario, 1001, "long", method, value)
            row = base.execute(CURRENT)
            if row["id"] not in heldout:
                raise ValueError("representative test cell has not completed")
            expected = heldout[row["id"]]
            for metric in ("route_pct", "router_calls", "crashed", "end_time"):
                if abs(float(row[metric]) - float(expected[metric])) > 1e-6:
                    raise ValueError(f"recording changed frozen result: {metric}")
            print(scenario, method, row["route_pct"], "replayed", flush=True)
    (target / "replay-audit.json").write_text(json.dumps(replay_reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
