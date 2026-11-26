"""Development-split comparison of UnifiedDrivingSystem variants (sharded, CPU).

Only for development maps (100-149, plus the previously reported 500-549 and
600-649 splits, which are now development data). Final numbers come from
tools/final_eval.py on untouched maps.

Usage:
  PYTHONPATH=. .venv/bin/python tools/dev_eval.py --out runs/dev/guard.json \
      --variant base:crossing_guard=False --variant guard:crossing_guard=True
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_unified_suite import _eval_group  # noqa: E402

PPO = "runs/safemetadrive/skills-1m-v1/skills-seed{}/ppo.zip"


def parse_variant(text: str) -> tuple[str, dict]:
    name, _, rest = text.partition(":")
    params: dict = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        params[key] = json.loads(value.lower() if value in ("True", "False") else value) \
            if value.replace(".", "", 1).lstrip("-").isdigit() or value in ("True", "False", "true", "false") \
            else value
    return name, params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--splits", nargs="+", default=["100:50", "500:50", "600:50"])
    ap.add_argument("--trackers", nargs="+", default=["analytic", "ppo0"])
    ap.add_argument("--shard", type=int, default=25)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    tasks = []
    for text in args.variant:
        vname, params = parse_variant(text)
        for tracker in args.trackers:
            tp = dict(tracker="analytic") if tracker == "analytic" else \
                dict(tracker="ppo", skill_checkpoint=PPO.format(tracker[-1]))
            for split in args.splits:
                start, count = map(int, split.split(":"))
                for s in range(start, start + count, args.shard):
                    tasks.append(dict(split=f"{start}", name=f"{vname}|{tracker}", start_seed=s,
                                      num_scenarios=min(args.shard, start + count - s),
                                      parameters={**tp, **params}))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_eval_group, tasks))

    groups: dict[tuple[str, str], list] = {}
    for r in results:
        groups.setdefault((r["summary"]["name"], r["summary"]["split"]), []).extend(r["episodes"])
    rows = []
    for (name, split), eps in sorted(groups.items()):
        n = len(eps)
        row = dict(name=name, split=split, n=n,
                   success=sum(e["success"] for e in eps) / n,
                   zero_cost=sum(e["zero_cost_success"] for e in eps) / n,
                   any_cost=sum(e["cost"] > 0 for e in eps) / n,
                   cost=sum(e["cost"] for e in eps) / n,
                   steps=sum(e["steps"] for e in eps) / n)
        rows.append(row)
        print(f"{name:32s} {split:>5s} n={n:3d} succ={row['success']*100:5.1f}% "
              f"zc={row['zero_cost']*100:5.1f}% anycost={row['any_cost']*100:5.1f}% "
              f"cost={row['cost']:.3f} steps={row['steps']:.0f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows=rows, groups={f"{k[0]}@{k[1]}": v for k, v in groups.items()}),
                                   indent=1))


if __name__ == "__main__":
    main()
