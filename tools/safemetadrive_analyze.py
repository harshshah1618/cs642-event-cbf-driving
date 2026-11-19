"""Summarize explicitly selected experiment reports without mixing development/test maps."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(groups, split="test", bootstrap_samples=10000):
    from driving.safemetadrive.protocol import SPLITS

    start, count = SPLITS[split]
    expected = set(range(start, start+count))
    matrices = {}
    summaries = {}
    for label, replicas in groups.items():
        runs = []
        for paths in replicas:
            reports = [json.loads(Path(path).read_text()) for path in paths]
            if any(report["split"] != split for report in reports):
                raise ValueError(f"{label}: mixed evaluation partitions")
            episodes = [row for report in reports for row in report["episodes"]]
            if {row["seed"] for row in episodes} != expected or len(episodes) != count:
                raise ValueError(f"{label}: replica must cover each {split} map exactly once")
            rows = sorted(episodes, key=lambda row: row["seed"])
            runs.append([[row["success"], row["cost"], row["collision_free_success"], row["reward"]]
                         for row in rows])
        array = np.asarray(runs, dtype=float)
        matrices[label] = array.mean(axis=0)
        summaries[label] = dict(
            replicas=len(runs), unique_maps=count, episodes=len(runs)*count,
            success_rate=float(array[:, :, 0].mean()), mean_cost=float(array[:, :, 1].mean()),
            collision_free_success_rate=float(array[:, :, 2].mean()),
            mean_reward=float(array[:, :, 3].mean()),
            per_replica_success=array[:, :, 0].mean(1).tolist(),
            per_replica_cost=array[:, :, 1].mean(1).tolist(),
        )
        if np.all(array[:, :, 0] == 1):
            summaries[label]["zero_failures_one_sided_95pct_failure_upper_bound"] = 1-0.05**(1/count)
    rng = np.random.default_rng(0)
    indices = rng.integers(count, size=(bootstrap_samples, count))
    comparisons = []
    labels = list(groups)
    for i, candidate in enumerate(labels):
        for reference in labels[i+1:]:
            difference = matrices[candidate]-matrices[reference]
            sampled = difference[indices].mean(1)
            comparisons.append(dict(candidate=candidate, reference=reference,
                success_difference=float(difference[:, 0].mean()),
                success_difference_95pct_map_bootstrap=np.quantile(sampled[:, 0], [.025, .975]).tolist(),
                cost_difference=float(difference[:, 1].mean()),
                cost_difference_95pct_map_bootstrap=np.quantile(sampled[:, 1], [.025, .975]).tolist()))
    return dict(split=split, groups=summaries, paired_comparisons=comparisons,
                uncertainty_note="Paired bootstrap resamples maps after averaging the supplied trained replicas. "
                                 "Intervals are conditional on these trained models, not full retraining uncertainty. "
                                 "Zero observed cost does not bound future cost severity.")


def markdown(result):
    lines = [f"# SafeMetaDrive {result['split']} results", "",
             "| Method | Replicas | Maps | Success | Collision-free success | Mean cost | Return |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["groups"].items():
        lines.append(f"| {name} | {row['replicas']} | {row['unique_maps']} | "
                     f"{row['success_rate']:.1%} | {row['collision_free_success_rate']:.1%} | "
                     f"{row['mean_cost']:.3f} | {row['mean_reward']:.2f} |")
    lines += ["", result["uncertainty_note"], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups", required=True, help="JSON: label -> replicas -> shard report paths")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = summarize(json.loads(Path(args.groups).read_text()), args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    output.with_suffix(".md").write_text(markdown(result))
    print(markdown(result))
