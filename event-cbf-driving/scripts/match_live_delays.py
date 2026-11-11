"""Compare the rule router with the live model's recorded delay sequences.

    python scripts/match_live_delays.py RUN_DIRECTORY

If routing decisions change the number of calls, excess rule calls reuse the
last observed model delay. This limitation is explicit in the saved manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.eval.runner import run_evaluation
from event_cbf_driving.io_utils import atomic_write_json, read_json
from event_cbf_driving.runtime import load_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    records = []
    for path in sorted((args.run / "episodes").glob("*.json")):
        episode = load_episode(path)
        calls = [call for call in episode.result.calls if not call.get("suppressed")]
        delays = [float(c["latency_s"]) for c in calls if c.get("latency_s") is not None]
        if not delays:
            continue
        cfg = load_config("configs/metadrive-study.yaml")
        source = read_json(args.run / "manifest.json")["config"]
        # Reuse physical, skill and monitor settings recorded for the actual run.
        for section in ("shield", "stack"):
            for key, value in source[section].items():
                setattr(getattr(cfg, section), key, value)
        cfg.stack.router_async = cfg.stack.realtime = False
        cfg.stack.simulate_deadline_wait = True
        cfg.router.backend = "scripted"
        cfg.router.latency_schedule_s = delays
        cfg.eval.scenarios = [ScenarioConfig(**episode.scenario.__dict__)]
        cfg.eval.seeds = [int(episode.result.seed)]
        cfg.eval.modes = [episode.result.mode]
        cfg.eval.figures = cfg.eval.fail_on_regression = False
        cfg.eval.tags = [f"matched-to:{args.run.name}/{path.name}", "excess-calls-reuse-last-delay"]
        result = run_evaluation(
            cfg, out_dir=args.run.parent / "matched-rule", run_id=f"{args.run.name}-{path.stem}"
        )
        records.extend(result.episodes)
        print(path.name, result.episodes[0]["route_pct"], flush=True)
    atomic_write_json(args.run / "matched-rule-summary.json", {"episodes": records})


if __name__ == "__main__":
    main()
