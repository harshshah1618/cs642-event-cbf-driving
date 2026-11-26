"""Summarise closed-loop pilot results (one JSON list of episode rows per scene/trigger).

Usage: .venv/bin/python tools/irregular_closedloop_report.py runs/irregular/closedloop/<tag> [...]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def rows_in(path):
    for f in sorted(Path(path).glob("*.json")):
        rows = json.loads(f.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{f}: expected a list of episode rows")
        yield from rows


def summarise(rows):
    """Keep unaudited final-step crash flags out of episode-collision totals."""
    verified = [r for r in rows if r.get("metrics_schema") == "irregular-episode-v2"]
    crashes = sum(any(r[key] for key in ("crash_object", "crash_human", "crash_vehicle"))
                  for r in verified)
    pass_speeds = [r["pass_speed"] for r in rows if r.get("pass_speed") is not None]
    return dict(
        episodes=len(rows), collision_verified_episodes=len(verified),
        collision_episodes=crashes if len(verified) == len(rows) else None,
        legacy_nonzero_cost_without_final_crash=sum(
            r.get("cost", 0) > 0 and not any(r.get(k, False) for k in
                ("crash_object", "crash_human", "crash_vehicle"))
            for r in rows if r.get("metrics_schema") != "irregular-episode-v2"),
        passed_stop=sum(r["passed_stop"] for r in rows),
        mean_cost=float(np.mean([r["cost"] for r in rows])),
        mean_calls=float(np.mean([len(r["calls"]) for r in rows])),
        mean_stopped_seconds=float(np.mean([r["seconds_stopped_near_object"] for r in rows])),
        mean_episode_seconds=float(np.mean([r["steps"] for r in rows]) / 10),
        mean_route_completion=float(np.mean([r["route_completion"] for r in rows])),
        mean_pass_speed=float(np.mean(pass_speeds)) if pass_speeds else None,
    )


def main(paths, output=None):
    rows = [r for p in paths for r in rows_in(p)]
    groups = {}
    for r in rows:
        groups.setdefault((r["model"].split("/")[-1], r["scene"], r["config"]["trigger"]), []).append(r)
    print("| model | scene | trigger | n | passed STOP | collision episodes | calls/episode | s stopped near object "
          "| episode s | route % | pass speed m/s | VLM answers |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    summaries = []
    for (model, scene, trig), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        if len({r["seed"] for r in rs}) != len(rs):
            raise ValueError(f"Duplicate map seeds in {model}/{scene}/{trig}; separate repeated runs")
        summary = summarise(rs)
        summaries.append(dict(model=model, scene=scene, trigger=trig, **summary))
        answers = Counter(c["action"] for r in rs for c in r["calls"])
        crashes = summary["collision_episodes"]
        crash_text = str(crashes) if crashes is not None else "unverified"
        speed = summary["mean_pass_speed"]
        speed_text = f"{speed:.1f}" if speed is not None else "-"
        print(f"| {model} | {scene} | {trig} | {len(rs)} | {sum(r['passed_stop'] for r in rs)} | {crash_text} | "
              f"{np.mean([len(r['calls']) for r in rs]):.1f} | "
              f"{np.mean([r['seconds_stopped_near_object'] for r in rs]):.1f} | "
              f"{np.mean([r['steps'] for r in rs]) / 10:.1f} | "
              f"{100 * np.mean([r.get('route_completion', np.nan) for r in rs]):.0f} | "
              f"{speed_text} | "
              + ", ".join(f"{a} {n}" for a, n in answers.most_common()) + " |")
    if any(s["collision_episodes"] is None for s in summaries):
        print("\nLegacy collision counts are unverified: final-step flags can miss earlier contact.")
    if output is not None:
        Path(output).write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", dest="output")
    args = parser.parse_args()
    main(args.paths, args.output)
