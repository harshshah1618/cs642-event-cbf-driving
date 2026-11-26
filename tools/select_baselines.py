"""Pick learned-baseline checkpoints on development maps only (100-149).

For each algorithm, every saved epoch of every seed is evaluated (stochastic actor,
as in the original study); one common epoch per algorithm is chosen by mean
zero-cost success across seeds (ties: lower mean cost). Writes the JSON consumed
by tools/final_eval.py --baselines.

  PYTHONPATH=. .venv/bin/python tools/select_baselines.py --root runs/baselines-5m-defaults-v2 \
      --label-suffix "(5M, OmniSafe defaults)" --out runs/final/baselines-5m.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safemetadrive_fair_benchmark import _run_group  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--label-suffix", default="")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    ckpts: dict[tuple[str, int, int], Path] = {}
    for path in args.root.rglob("epoch-*.pt"):
        algo, seed = re.search(r"/(PPOLag|CPO)-seed(\d)/", str(path)).groups()
        epoch = int(path.stem.split("-")[1])
        if epoch > 0:
            ckpts[(algo, int(seed), epoch)] = path
    tasks = [dict(split_label="dev", name=f"{a}|{s}|{e}", policy="native_learned",
                  parameters=dict(checkpoint=str(p), deterministic=False, policy_seed=1000 * s),
                  start_seed=start, num_scenarios=25)
             for (a, s, e), p in sorted(ckpts.items()) for start in (100, 125)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_run_group, tasks))
    by: dict[str, list] = defaultdict(list)
    for r in results:
        by[r["summary"]["name"]].extend(r["episodes"])

    table: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for name, eps in by.items():
        algo, seed, epoch = name.split("|")
        zc = sum(e["zero_cost_success"] for e in eps) / len(eps)
        cost = sum(e["cost"] for e in eps) / len(eps)
        table[algo][int(epoch)].append((zc, cost))
    selection, report = {}, {}
    for algo, epochs in table.items():
        scored = {e: (sum(z for z, _ in v) / len(v), sum(c for _, c in v) / len(v))
                  for e, v in epochs.items() if len(v) == 3}
        best = max(scored, key=lambda e: (scored[e][0], -scored[e][1]))
        report[algo] = {e: dict(dev_zero_cost=round(z, 3), dev_cost=round(c, 3))
                        for e, (z, c) in sorted(scored.items())}
        report[algo]["selected_epoch"] = best
        label = {"PPOLag": "PPO-Lag", "CPO": "CPO"}[algo] + (f" {args.label_suffix}" if args.label_suffix else "")
        selection[label] = [str(ckpts[(algo, s, best)]) for s in range(3)]
        print(f"{algo}: selected epoch {best} (dev zero-cost {scored[best][0]:.3f}, cost {scored[best][1]:.3f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selection, indent=1))
    args.out.with_suffix(".selection.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
