"""Verify complete frozen grids, paired physical states, and create evidence figures."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from maneuver_matrix import design, summarize
from maneuver_probe import ROOT


def main():
    output = ROOT.parent / "maneuver-study"
    analysis = {}
    heldout = []
    for stage in ("development", "heldout"):
        directory = ROOT / "runs/maneuver-matrix" / stage
        lock_json = (directory / "lock.json").read_text()
        lock = json.loads(lock_json)
        lock_sha = hashlib.sha256(lock_json.encode()).hexdigest()
        rows = [json.loads(s) for s in (directory / "episodes.jsonl").read_text().splitlines()]
        assert len(rows) == len(design(stage))
        assert {r["id"] for r in rows} == {j.key() for j in design(stage)}
        assert len({r["id"] for r in rows}) == len(rows)
        assert all(r["lock_sha256"] == lock_sha for r in rows)
        for name, sha in lock["sources"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
        pairs = defaultdict(list)
        for r in rows:
            pairs[(r["seed"], r["controller"], r["delay"])].append(r)
            assert r["success"] == (not r["crashed"] and r["route_pct"] >= 99)
        for group in pairs.values():
            assert all(r["initial"] == group[0]["initial"] for r in group)
        initial_hashes = {
            hashlib.sha256(json.dumps(r["initial"], sort_keys=True).encode()).hexdigest()
            for r in rows
        }
        analysis[stage] = {
            "episodes": len(rows),
            "condition_combinations": len(pairs),
            "unique_initial_physical_states_including_actuator_geometry": len(initial_hashes),
            "lock_sha256": lock_sha,
            "summary": summarize(rows),
        }
        if stage == "heldout":
            heldout = rows
            subgroups = {}
            for ctrl in ("ppo", "pd"):
                for delay in (0.125, 6.0, 18.0, 30.0):
                    subgroups[f"{ctrl}:{delay:g}s"] = summarize(
                        [r for r in rows if r["controller"] == ctrl and r["delay"] == delay]
                    )
            analysis[stage]["subgroups"] = subgroups
            comparison = defaultdict(int)
            for group in pairs.values():
                by_mode = {r["mode"]: r for r in group}
                baseline, fixed, geometric = [
                    by_mode[m] for m in ("baseline", "fixed_latched", "geometric_latched")
                ]
                comparison["fixed_success_baseline_failure"] += (
                    fixed["success"] and not baseline["success"]
                )
                comparison["fixed_failure_baseline_success"] += (
                    baseline["success"] and not fixed["success"]
                )
                comparison["geometry_success_fixed_failure"] += (
                    geometric["success"] and not fixed["success"]
                )
                comparison["geometry_failure_fixed_success"] += (
                    fixed["success"] and not geometric["success"]
                )
            analysis[stage]["paired_comparisons"] = dict(comparison)
    mechanism = json.loads((ROOT / "runs/maneuver-mechanism/episodes.json").read_text())
    mechanism_summary = []
    traces = {}
    for row in mechanism:
        trace = json.loads(
            (ROOT / f"runs/maneuver-development/traces/{row['id']}.json").read_text()
        )
        traces[row["mode"]] = trace
        before = min(trace["interactions"], key=lambda r: abs(r["t"] - 17.98))
        at = min(trace["interactions"], key=lambda r: abs(r["t"] - 18.0))
        mechanism_summary.append(
            {
                "mode": row["mode"],
                "success": row["success"],
                "route_pct": row["route_pct"],
                "before_intent": before,
                "at_intent": at,
                "reserve_events": row["reserve_events"],
            }
        )
    analysis["mechanism"] = mechanism_summary
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False))
    colors = {
        "baseline": "#777777",
        "fixed_latched": "#007e7f",
        "geometric_latched": "#4c60b4",
        "static": "#b85d39",
    }
    labels = {
        "baseline": "Original",
        "fixed_latched": "8m, released after admission",
        "geometric_latched": "Geometry heuristic",
        "static": "8m, never released",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, ctrl in zip(axes[0], ("ppo", "pd"), strict=True):
        for mode in colors:
            values = []
            for delay in (0.125, 6.0, 18.0, 30.0):
                rs = [
                    r
                    for r in heldout
                    if r["controller"] == ctrl and r["delay"] == delay and r["mode"] == mode
                ]
                values.append(sum(r["success"] for r in rs))
            ax.plot(
                [0.125, 6, 18, 30],
                values,
                marker="o",
                label=labels[mode],
                color=colors[mode],
                linestyle="--" if mode == "geometric_latched" else "-",
                alpha=0.85,
            )
        ax.set(
            title=f"Held-out conditions: {ctrl.upper()} controller",
            xlabel="First correct intent delay (s)",
            ylabel="Finished conditions (out of 6)",
            ylim=(-0.2, 6.4),
            yticks=range(7),
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    for mode in ("baseline", "fixed_latched", "static"):
        trace = traces[mode]
        times = [r["t"] for r in trace["interactions"]]
        xs = [r["ego"][0] for r in trace["interactions"]]
        ys = [r["ego"][1] for r in trace["interactions"]]
        axes[1, 0].plot(times, xs, label=labels[mode], color=colors[mode])
        axes[1, 1].plot(xs, ys, color=colors[mode], label=labels[mode])
    axes[1, 0].axvline(18, color="black", linestyle=":", label="Correct intent arrives")
    axes[1, 0].set(
        title="Development mechanism case: same late intent",
        xlabel="Simulation time (s)",
        ylabel="Distance travelled (m)",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].add_patch(
        Rectangle((42.5, -1), 5, 2, color="#b72c2c", label="Parked lead (schematic footprint)")
    )
    axes[1, 1].axhline(4, color="black", alpha=0.2, linestyle=":")
    axes[1, 1].set(
        title="Keeping room allows the turn",
        xlabel="Forward position (m)",
        ylabel="Lateral position (m)",
        xlim=(0, 80),
        ylim=(-2, 6),
    )
    for ax in axes[1]:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Correct decisions still need room to execute\nDeterministic physical condition grid; no population safety guarantee",
        fontsize=13,
    )
    fig.savefig(output / "results.png", dpi=170)
    plt.close(fig)
    print(json.dumps({k: v["summary"] for k, v in analysis.items() if k != "mechanism"}, indent=2))


if __name__ == "__main__":
    main()
