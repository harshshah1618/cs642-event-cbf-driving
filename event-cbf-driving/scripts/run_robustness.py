"""Held-out physical events and variable-latency evaluation; retain failures."""

from __future__ import annotations

import argparse
from pathlib import Path

from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.eval.runner import run_evaluation
from event_cbf_driving.io_utils import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/robustness-v2"))
    output = parser.parse_args().output
    cases = [
        (
            "lead_brakes",
            "merge",
            {"traffic_events": [{"at": 2.0, "vehicle_index": 0, "target_speed": 0.0}]},
            False,
        ),
        (
            "moving_cut_in",
            "merge",
            {"traffic_events": [{"at": 2.0, "vehicle_index": 1, "target_y": 0.0}]},
            False,
        ),
        (
            "target_lane_brakes",
            "dense_highway",
            {"traffic_events": [{"at": 1.0, "vehicle_index": 4, "target_speed": 0.0}]},
            False,
        ),
        ("tight_cut_in", "two_trap", {"cut_in_gap": [8.0, 14.0], "cut_in_speed": [2.0, 6.0]}, True),
    ]
    cases.extend(
        (f"jitter_{name}", name, {}, False) for name in ("two_trap", "merge", "dense_highway")
    )
    records = []
    for name, scenario, params, adversarial in cases:
        cfg = load_config("configs/metadrive-study.yaml")
        cfg.name = name
        cfg.eval.scenarios = [
            ScenarioConfig(
                scenario, simulator="metadrive", randomize=True, duration=20.0, params=params
            )
        ]
        cfg.eval.seeds = [301, 307, 311]
        cfg.eval.modes = ["proposed", "proposed_no_margin"]
        cfg.router.latency_s = 0.8 if name.startswith("jitter") else 0.3
        cfg.router.latency_jitter_s = 0.3 if name.startswith("jitter") else 0.2
        cfg.shield.vlm_delay_s = cfg.router.latency_s + cfg.router.latency_jitter_s
        cfg.eval.fail_on_regression = False
        cfg.eval.figures = False
        result = run_evaluation(cfg, out_dir=output, run_id=name)
        for episode in result.episodes:
            records.append({"case": name, "adversarial": adversarial, **episode})
        print(
            name,
            [
                (r["mode"], round(r["route_pct"], 1), r["crashed"], r["freeze"])
                for r in result.episodes
            ],
            flush=True,
        )
    atomic_write_json(
        output / "summary.json",
        {
            "episodes": records,
            "note": "Tight cut-ins deliberately test outside the nominal stopping envelope. Failures are retained.",
        },
    )


if __name__ == "__main__":
    main()
