"""Summarise closed-loop pilot results (one JSON list of episode rows per scene/trigger).

Usage: .venv/bin/python tools/irregular_closedloop_report.py runs/irregular/closedloop/<tag> [...]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def rows_in(path):
    for f in sorted(Path(path).glob("*.json")):
        yield from json.loads(f.read_text())


def main(paths):
    rows = [r for p in paths for r in rows_in(p)]
    groups = {}
    for r in rows:
        groups.setdefault((r["model"].split("/")[-1], r["scene"], r["config"]["trigger"]), []).append(r)
    print("| model | scene | trigger | n | passed STOP | crashes | calls/episode | s stopped near object "
          "| episode s | route % | pass speed m/s | VLM answers |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for (model, scene, trig), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        answers = Counter(c["action"] for r in rs for c in r["calls"])
        crashes = sum(bool(r["crash_object"] or r["crash_human"] or r["crash_vehicle"]) for r in rs)
        print(f"| {model} | {scene} | {trig} | {len(rs)} | {sum(r['passed_stop'] for r in rs)} | {crashes} | "
              f"{np.mean([len(r['calls']) for r in rs]):.1f} | "
              f"{np.mean([r['seconds_stopped_near_object'] for r in rs]):.1f} | "
              f"{np.mean([r['steps'] for r in rs]) / 10:.1f} | "
              f"{100 * np.mean([r.get('route_completion', np.nan) for r in rs]):.0f} | "
              f"{np.nanmean([r.get('pass_speed') if r.get('pass_speed') is not None else np.nan for r in rs]):.1f} | "
              + ", ".join(f"{a} {n}" for a, n in answers.most_common()) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
