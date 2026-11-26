"""Irregular-scene table: every arm scored identically on the same maps.

Per arm and scene (n maps each):
  violation   = any collision OR passed the flagger while it showed STOP
  needless    = stopped > 3 s near an object that does not require stopping
                (every scene except flagger, pedestrian-lane and the wet road closure)
  deadlock    = flagger scene: never got past the flagger within the step budget
  route       = mean route completion within the fixed budget (efficiency)
  calls       = slow-model calls per episode
Rates carry Wilson 95% intervals; the VLM arm is compared to each baseline with an
exact paired McNemar test on violations over identical (scene, map) pairs.

  python tools/irregular_final_report.py --arm vlm=runs/final/irregular-vlm/ \
      --arm normal,careful,oracle,lidar-careful,lidar-stop=runs/final/irregular-arms.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import fmt_rate, mcnemar_exact  # noqa: E402

STOP_SCENES = ("flagger", "pedestrian-lane", "wet-closed")  # stopping can be correct here


def load(spec: str) -> dict[str, list[dict]]:
    names, _, path = spec.partition("=")
    arms = names.split(",")
    rows: list[dict] = []
    if Path(path).is_dir():
        for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True)):
            data = json.loads(Path(f).read_text())
            if isinstance(data, list):
                rows += data
    else:
        data = json.loads(Path(path).read_text())
        rows = data["rows"] if isinstance(data, dict) else data
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        arm = r.get("arm", arms[0]) if len(arms) > 1 else arms[0]
        if arm in arms:
            out[arm].append(r)
    return out


def violation(r: dict) -> bool:
    return bool(r["crash_vehicle"] or r["crash_object"] or r["crash_human"] or r["passed_stop"])


def needless(r: dict) -> bool:
    return r["scene"] not in STOP_SCENES and r["seconds_stopped_near_object"] > 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name[,name...]=path")
    ap.add_argument("--ref", default="vlm")
    args = ap.parse_args()
    arms: dict[str, list[dict]] = {}
    for spec in args.arm:
        arms.update(load(spec))

    scenes = sorted({r["scene"] for rows in arms.values() for r in rows})
    print("### Per scene (violations / needless stops / route %)\n")
    print("| Scene | " + " | ".join(arms) + " |")
    print("|---|" + "---:|" * len(arms))
    for scene in scenes:
        cells = []
        for rows in arms.values():
            sub = [r for r in rows if r["scene"] == scene]
            if not sub:
                cells.append("–")
                continue
            v = sum(map(violation, sub))
            nd = sum(map(needless, sub))
            route = 100 * sum(r["route_completion"] for r in sub) / len(sub)
            cells.append(f"{v}/{len(sub)} · {nd} · {route:.0f}%")
        print(f"| {scene} | " + " | ".join(cells) + " |")

    print("\n### Overall\n")
    print("| Arm | Episodes | Violations | STOP passed (flagger) | Needless stops | Flagger deadlocks "
          "| Route % | Calls/ep | Paired violations vs ref |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    ref = {(r["scene"], r["seed"]): violation(r) for r in arms.get(args.ref, [])}
    for name, rows in arms.items():
        n = len(rows)
        flag = [r for r in rows if r["scene"] == "flagger"]
        stop_pass = sum(r["passed_stop"] for r in flag)
        dead = sum(r["end_reason"] != "passed_object" for r in flag)
        paired = ""
        if ref and name != args.ref:
            keys = [k for k in ref if any((r["scene"], r["seed"]) == k for r in rows)]
            mine = {(r["scene"], r["seed"]): violation(r) for r in rows}
            if keys:
                # mcnemar(a=ref safe, b=this arm safe): ref-only safe vs arm-only safe
                a_only, b_only, p = mcnemar_exact([not ref[k] for k in keys], [not mine[k] for k in keys])
                paired = f"only-{args.ref}-safe {a_only} / only-{name}-safe {b_only}, p={p:.2g}"
        print(f"| {name} | {n} | {fmt_rate(sum(map(violation, rows)), n)} | "
              f"{stop_pass}/{len(flag)} | {fmt_rate(sum(map(needless, rows)), n)} | {dead}/{len(flag)} | "
              f"{100 * sum(r['route_completion'] for r in rows) / n:.1f} | "
              f"{sum(len(r['calls']) for r in rows) / n:.2f} | {paired} |")


if __name__ == "__main__":
    main()
