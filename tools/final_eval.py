"""Frozen final SafeMetaDrive evaluation on untouched maps, every method on the same maps.

All rows go through one episode loop (safemetadrive_fair_benchmark._run_group), so
metrics are computed identically. Tables report Wilson 95% intervals for rates,
bootstrap intervals for mean cost, and exact paired McNemar tests against the
reference system on the same maps.

  PYTHONPATH=. .venv/bin/python tools/final_eval.py --first 2000 --count 200 \
      --out runs/final/safemetadrive-2000-2199.json
  PYTHONPATH=. .venv/bin/python tools/final_eval.py --report runs/final/safemetadrive-2000-2199.json
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

from safemetadrive_fair_benchmark import _run_group  # noqa: E402
from stats import bootstrap_mean, fmt_rate, mcnemar_exact  # noqa: E402

PPO = "runs/safemetadrive/skills-1m-v1/skills-seed{}/ppo.zip"
REFERENCE = "Ours (analytic tracker)"


def methods(baselines: dict[str, list[str]]) -> list[tuple[str, str, dict]]:
    ours = dict(tracker="analytic")
    rows = [
        (REFERENCE, "unified_system", ours),
        *[(f"Ours (PPO tracker seed{i})", "unified_system",
           dict(tracker="ppo", skill_checkpoint=PPO.format(i))) for i in range(3)],
        # One-switch ablations of the reference system
        ("Ablation: no crossing guard", "unified_system", {**ours, "crossing_guard": False}),
        ("Ablation: no safety filter", "unified_system", {**ours, "use_safety_filter": False}),
        ("Ablation: no replanning", "unified_system", {**ours, "router": "none", "option_reserve_m": 0.0}),
        ("Ablation: no blending", "unified_system", {**ours, "blend_steps": 0}),
        # Baselines
        ("IDM (privileged object state)", "idm", {}),
        ("MetaDrive PPO-Expert", "upstream_expert", {}),
    ]
    for delay_s in (1, 2, 4, 6, 8):
        for reserve in (6.5, 0.0):
            rows.append((f"Delay {delay_s}s, reserve {reserve:g} m", "unified_system",
                         {**ours, "decision_delay_steps": 10 * delay_s, "option_reserve_m": reserve}))
    for label, ckpts in baselines.items():
        for i, ckpt in enumerate(ckpts):
            rows.append((f"{label} seed{i}", "native_learned",
                         dict(checkpoint=ckpt, deterministic=False, policy_seed=1000 * i)))
    return rows


def run(args: argparse.Namespace) -> None:
    baselines = json.loads(Path(args.baselines).read_text()) if args.baselines else {}
    tasks = []
    for name, policy, params in methods(baselines):
        if args.only and not any(tag in name for tag in args.only):
            continue
        for s in range(args.first, args.first + args.count, args.shard):
            tasks.append(dict(split_label=f"{args.first}-{args.first + args.count - 1}", name=name,
                              policy=policy, parameters=params, start_seed=s,
                              num_scenarios=min(args.shard, args.first + args.count - s)))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_run_group, tasks))
    episodes: dict[str, list] = {}
    for r in results:
        episodes.setdefault(r["summary"]["name"], []).extend(r["episodes"])
    for eps in episodes.values():
        eps.sort(key=lambda e: e["seed"])
    out = Path(args.out)
    if out.exists() and args.merge:
        previous = json.loads(out.read_text())
        previous["episodes"].update(episodes)
        episodes = previous["episodes"]
        baselines = {**previous.get("baselines", {}), **baselines}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(first=args.first, count=args.count, baselines=baselines,
                                   episodes=episodes), indent=1))
    report(out)


def report(path: Path) -> None:
    data = json.loads(Path(path).read_text())
    episodes = data["episodes"]
    ref = episodes.get(REFERENCE)
    print("| Method | n | Finished | Zero-cost finished | Any-cost episodes | Mean cost [95% CI] "
          "| Mean steps | Paired vs ours (zero-cost) |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, eps in episodes.items():
        n = len(eps)
        succ = sum(e["success"] for e in eps)
        zc = sum(e["zero_cost_success"] for e in eps)
        anyc = sum(e["cost"] > 0 for e in eps)
        costs = [e["cost"] for e in eps]
        lo, hi = bootstrap_mean(costs)
        paired = ""
        if ref is not None and name != REFERENCE and [e["seed"] for e in eps] == [e["seed"] for e in ref]:
            ours_only, theirs_only, p = mcnemar_exact([e["zero_cost_success"] for e in ref],
                                                      [e["zero_cost_success"] for e in eps])
            paired = f"ours-only {ours_only} / theirs-only {theirs_only}, p={p:.3g}"
        print(f"| {name} | {n} | {fmt_rate(succ, n)} | {fmt_rate(zc, n)} | {fmt_rate(anyc, n)} | "
              f"{sum(costs) / n:.3f} [{lo:.3f}, {hi:.3f}] | {sum(e['steps'] for e in eps) / n:.0f} | {paired} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2000)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--shard", type=int, default=25)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--baselines", help="JSON {label: [checkpoint, ...]} of selected learned baselines")
    ap.add_argument("--only", nargs="*", help="run only methods whose name contains one of these")
    ap.add_argument("--merge", action="store_true", help="merge into an existing output file")
    ap.add_argument("--out")
    ap.add_argument("--report")
    args = ap.parse_args()
    if args.report:
        report(Path(args.report))
    else:
        if not args.out:
            ap.error("--out is required to run")
        run(args)


if __name__ == "__main__":
    main()
