"""Audit and summarize the frozen final comparison without tuning any policy."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from integrated_eval import OUT, ROOT, digest, jobs, verify_sources
from integrated_vlm import design as live_design

RESULTS = ROOT.parent / "cs642-driving-runs/final-analysis"
LABELS = {
    "unshielded_rl": "Unprotected RL",
    "static_cbf": "Static CBF",
    "timer_hard_unshielded": "1 Hz, unshielded",
    "uncertainty_wakeup": "Uncertainty wake-up",
    "barrier_replan_local": "Local barrier replan",
    "original_proposal": "Original proposal",
    "final_integrated": "Final integrated",
    "matched_timer_hard": "1 Hz, matched safety",
    "high_rate_poll": "10 Hz polling",
    "gated_poll": "10 Hz, gated",
    "immediate_benefit": "Immediate benefit",
    "no_reserve": "No extra reserve",
    "no_blend": "No blending",
    "speed_delay_margin": "Add delay margin",
    "critic_margin": "Add critic margin",
}


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summary(rows):
    if not rows:
        return {"n": 0}
    result = {
        "n": len(rows),
        "success": sum(r["success"] for r in rows),
        "crashes": sum(r["crashed"] for r in rows),
        "terminal_stalls": sum(r["terminal_stall"] for r in rows),
        "legacy_freezes": sum(r["freeze"] for r in rows),
        "route_complete": sum(r["end_reason"] == "route_complete" for r in rows),
        "pending_crashes": sum(not r["pending_survival"] for r in rows),
        "cut_ins_observed": sum(r["cut_in_observed"] for r in rows),
        "cut_ins_first_seen_pending": sum(r["cut_in_first_seen_while_pending"] for r in rows),
        "end_reasons": dict(Counter(r["end_reason"] for r in rows)),
    }
    for field in (
        "route_pct",
        "expensive_calls",
        "local_replans",
        "stopped_s",
        "end_time",
        "achieved_jerk_rms",
        "achieved_jerk_peak",
        "achieved_jerk_peak_after_0_2s",
        "override_fraction",
        "braking_infeasible_ticks",
        "calls_applied",
        "calls_rejected",
    ):
        result["mean_" + field] = float(np.mean([r[field] for r in rows]))
    return result


def group_summary(rows):
    return {
        m: summary([r for r in rows if r["method"] == m])
        for m in sorted({r["method"] for r in rows})
    }


def pair_key(row):
    return row["scene"], row["seed"], row["delay"]


def paired_contrast(rows, opponent, *, joint_success=False):
    a = {pair_key(r): r for r in rows if r["method"] == "final_integrated"}
    b = {pair_key(r): r for r in rows if r["method"] == opponent}
    assert a.keys() == b.keys()
    keys = [k for k in a if not joint_success or (a[k]["success"] and b[k]["success"])]
    seeds = sorted({k[1] for k in a})
    rng = np.random.default_rng(975130)
    indices = rng.integers(0, len(seeds), (10000, len(seeds)))
    result = {"pairs": len(keys), "joint_success_only": joint_success, "seed_clusters": len(seeds)}
    for metric in (
        "success",
        "crashed",
        "terminal_stall",
        "route_pct",
        "expensive_calls",
        "achieved_jerk_rms",
        "achieved_jerk_peak",
        "achieved_jerk_peak_after_0_2s",
    ):
        totals = np.array(
            [
                sum(float(a[k][metric]) - float(b[k][metric]) for k in keys if k[1] == seed)
                for seed in seeds
            ]
        )
        counts = np.array([sum(k[1] == seed for k in keys) for seed in seeds])
        if not counts.sum():
            result[metric] = None
            continue
        denominators = counts[indices].sum(axis=1)
        bootstrap = totals[indices].sum(axis=1)[denominators > 0] / denominators[denominators > 0]
        result[metric] = {
            "final_minus_opponent": float(totals.sum() / counts.sum()),
            "descriptive_seed_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
    return result


def audit_controlled(rows, lock):
    scale = lock["selection"]["sigma_scale"]
    expected = {j.key() for j in jobs("test", scale)}
    assert len(rows) == len(expected) and {r["id"] for r in rows} == expected
    sha = hashlib.sha256((OUT / "test/lock.json").read_bytes()).hexdigest()
    initial = defaultdict(list)
    for row in rows:
        assert row["source_lock_sha256"] == sha and not row["actual_model"]
        assert row["success"] == (not row["crashed"] and row["route_pct"] >= 99)
        initial[(row["scene"], row["seed"])].append(row["initial"])
        path = OUT / "test/traces" / f"{row['id']}.json.gz"
        assert digest(path) == row["trace_sha256"], path
    assert all(all(v == values[0] for v in values) for values in initial.values())
    verify_sources(lock["sources"])
    unique_initials = len({json.dumps(v[0], sort_keys=True) for v in initial.values()})
    return {
        "rows": len(rows),
        "unique_job_ids": len(expected),
        "paired_initial_states_exact": True,
        "scene_seed_groups": len(initial),
        "unique_initial_state_values": unique_initials,
        "all_trace_hashes_verified": True,
        "all_source_hashes_verified": True,
        "source_lock_sha256": sha,
    }


def audit_traces(rows, stage):
    total_ticks, final_left = 0, Counter()
    for i, row in enumerate(rows):
        path = (
            (OUT / "test/traces" / f"{row['id']}.json.gz")
            if stage == "test"
            else (OUT / "actual-vlm/episodes" / row["id"] / f"{row['id']}.json.gz")
        )
        assert digest(path) == row["trace_sha256"]
        trace = json.loads(gzip.decompress(path.read_bytes()))
        interactions = trace["interactions"]
        applied = np.asarray([r["applied_action"] for r in interactions])
        plant = np.asarray([r["plant"]["requested_acceleration"] for r in interactions])
        assert np.allclose(applied, plant, atol=1e-6, rtol=0), row["id"]
        assert np.allclose(applied, trace["actions"], atol=1e-6, rtol=0), row["id"]
        achieved = np.asarray([r["plant"]["achieved_acceleration"] for r in interactions])
        jerk = np.diff(achieved[:, 1]) / 0.02
        assert np.isclose(
            row["achieved_jerk_rms"], np.sqrt(np.mean(jerk**2)) if len(jerk) else 0
        ), row["id"]
        assert np.isfinite(applied).all() and np.isfinite(achieved).all()
        total_ticks += len(interactions)
        final_left[row["method"]] += trace["y"] >= 2
        if (i + 1) % 200 == 0:
            print(f"Audited {stage} physical traces {i + 1}/{len(rows)}", flush=True)
    return {
        "episodes": len(rows),
        "physical_ticks": total_ticks,
        "plant_actions_match": True,
        "achieved_jerk_rms_recomputed": True,
        "final_left_lane_counts": dict(final_left),
        "note": "Route success does not require returning to the right lane",
    }


def plots(data):
    names = [
        "unshielded_rl",
        "static_cbf",
        "timer_hard_unshielded",
        "uncertainty_wakeup",
        "barrier_replan_local",
        "original_proposal",
        "matched_timer_hard",
        "gated_poll",
        "immediate_benefit",
        "final_integrated",
    ]
    nominal = data["nominal"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8), constrained_layout=True)
    colors = ["#007b7b" if m == "final_integrated" else "#7b8793" for m in names]
    for ax, field, title in zip(
        axes,
        ("success", "crashes", "terminal_stalls"),
        ("Finished", "Collided", "Terminal stall"),
        strict=True,
    ):
        values = [100 * nominal[m][field] / nominal[m]["n"] for m in names]
        ax.barh(range(len(names)), values, color=colors)
        ax.set(
            yticks=range(len(names)),
            yticklabels=[LABELS[m] for m in names] if ax == axes[0] else [],
            title=title,
            xlabel="Percent of nominal conditions",
            xlim=(0, 105),
        )
        ax.invert_yaxis()
        for i, value in enumerate(values):
            ax.text(min(value + 1, 99), i, f"{value:g}", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Controlled state-rule decisions: 72 nominal episodes per method", fontsize=12)
    fig.savefig(RESULTS / "nominal-outcomes.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.5), constrained_layout=True)
    for method, color, style in (
        ("final_integrated", "#007b7b", "-"),
        ("high_rate_poll", "#9b5638", "-"),
        ("gated_poll", "#656db3", "--"),
        ("uncertainty_wakeup", "#777777", ":"),
        ("original_proposal", "#b49434", "-."),
    ):
        vals = [data["by_delay"][str(d)][method] for d in (0.0, 0.125, 1.0, 18.0)]
        x = range(4)
        axes[0].plot(
            x,
            [v["mean_expensive_calls"] for v in vals],
            marker="o",
            color=color,
            linestyle=style,
            label=LABELS[method],
        )
        axes[1].plot(
            x, [100 * v["success"] / v["n"] for v in vals], marker="o", color=color, linestyle=style
        )
    for ax in axes:
        ax.set(
            xticks=range(4),
            xticklabels=["0", ".125", "1", "18 (stress)"],
            xlabel="Simulated request delay (s)",
        )
        ax.grid(alpha=0.2)
    axes[0].set(ylabel="Simulated costly requests per episode", yscale="symlog", ylim=(0, None))
    axes[0].legend(fontsize=7.5)
    axes[1].set(ylabel="Finished (%)", ylim=(-3, 104))
    fig.savefig(RESULTS / "cost-and-delay.png", dpi=180)
    plt.close(fig)


def main():
    global OUT, RESULTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with-live", action="store_true")
    p.add_argument("--audit-traces", action="store_true")
    p.add_argument("--runs-dir", type=Path, default=OUT)
    p.add_argument("--results-dir", type=Path, default=ROOT.parent / "cs642-driving-runs/final-analysis")
    a = p.parse_args()
    OUT, RESULTS = a.runs_dir, a.results_dir
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = read_rows(OUT / "test/episodes.jsonl")
    lock = json.loads((OUT / "test/lock.json").read_text())
    audit = audit_controlled(rows, lock)
    nominal = [r for r in rows if r["delay"] <= 1]
    stress = [r for r in rows if r["delay"] > 1]
    data = {
        "audit": audit,
        "all": group_summary(rows),
        "nominal": group_summary(nominal),
        "stress": group_summary(stress),
        "by_delay": {
            str(d): group_summary([r for r in rows if r["delay"] == d])
            for d in (0.0, 0.125, 1.0, 18.0)
        },
        "by_scene": {
            s: group_summary([r for r in nominal if r["scene"] == s])
            for s in sorted({r["scene"] for r in rows})
        },
        "contrasts": {
            subset: {m: paired_contrast(rs, m) for m in LABELS if m != "final_integrated"}
            for subset, rs in (("nominal", nominal), ("stress", stress))
        },
        "joint_success_comfort": {
            subset: paired_contrast(rs, "no_blend", joint_success=True)
            for subset, rs in (("nominal", nominal), ("stress", stress))
        },
    }
    final, table = data["nominal"]["final_integrated"], data["nominal"]
    comfort = data["joint_success_comfort"]["nominal"]
    pass_bars = {
        "fewer_crashes_than_unprotected_arms": all(
            final["crashes"] < table[m]["crashes"]
            for m in ("unshielded_rl", "timer_hard_unshielded")
        ),
        "fewer_terminal_stalls_than_static_cbf": final["terminal_stalls"]
        < table["static_cbf"]["terminal_stalls"],
        "fewer_calls_than_high_rate_with_no_observed_success_loss": (
            final["mean_expensive_calls"] < table["high_rate_poll"]["mean_expensive_calls"]
            and final["success"] >= table["high_rate_poll"]["success"]
        ),
        "blend_mean_rms_and_mean_peak_not_worse_on_joint_success_subset": bool(
            comfort["pairs"]
            and all(
                comfort[m]["final_minus_opponent"] <= 1e-10
                for m in ("achieved_jerk_rms", "achieved_jerk_peak")
            )
        ),
        "no_observed_pending_collisions_with_inflation_on": (
            data["all"]["speed_delay_margin"]["pending_crashes"] == 0
            and data["all"]["original_proposal"]["pending_crashes"] == 0
        ),
        "pending_margin_necessity": "Not established by zero-event equality; inspect matched ablation counts",
        "published_system_superiority": "Not tested",
    }
    write_json(RESULTS / "controlled.json", data)
    write_json(RESULTS / "pass-bars.json", pass_bars)
    if a.audit_traces:
        write_json(RESULTS / "controlled-trace-audit.json", audit_traces(rows, "test"))
    plots(data)
    if a.with_live:
        complete = json.loads((OUT / "actual-vlm/COMPLETE.json").read_text())
        live = read_rows(OUT / "actual-vlm/episodes.jsonl")
        selection = json.loads((ROOT / "study-selection.json").read_text())
        assert len(live) == len(live_design(selection["sigma_scale"])) == 36
        assert {r["id"] for r in live} == {j.key() for j in live_design(selection["sigma_scale"])}
        initials = defaultdict(list)
        for r in live:
            initials[(r["scene"], r["seed"])].append(r["initial"])
        assert all(all(v == vs[0] for v in vs) for vs in initials.values()), (
            "actual subset initial states differ"
        )
        predictions = [p for r in live for p in r["prediction_latency_s"]]
        actual = {
            "completion": complete,
            "summary": group_summary(live),
            "model_enabled_episodes": sum(r["actual_model"] for r in live),
            "episodes_with_actual_predictions": sum(
                r["actual_model_predictions"] > 0 for r in live
            ),
            "actual_predictions": sum(r["actual_model_predictions"] for r in live),
            "fallbacks": sum(r["router_fallbacks"] for r in live),
            "initial_states_match_exactly": True,
            "prediction_latency_s": {
                "median": float(np.median(predictions)),
                "min": min(predictions),
                "max": max(predictions),
            },
            "by_world": {
                f"{s}:{k}": group_summary([r for r in live if r["scene"] == s and r["seed"] == k])
                for s, k in sorted(initials)
            },
            "note": "Six worlds, CPU inference; descriptive integration result, not population performance",
        }

        def changed_target(call):
            return bool(
                call.get("applied")
                and (call.get("from") == "OVERTAKE_LEFT") != (call.get("to") == "OVERTAKE_LEFT")
            )

        actual["model_episodes_with_admitted_target_change"] = sum(
            r["actual_model"] and any(changed_target(c) for c in r["calls"]) for r in live
        )
        for method, values in actual["summary"].items():
            group = [r for r in live if r["method"] == method]
            values["actual_predictions"] = sum(r["actual_model_predictions"] for r in group)
            values["admitted_target_changes"] = sum(
                changed_target(c) for r in group for c in r["calls"]
            )
            values["mean_wall_s"] = float(np.mean([r["wall_s"] for r in group]))
        write_json(RESULTS / "actual-vlm.json", actual)
        if a.audit_traces:
            write_json(RESULTS / "actual-trace-audit.json", audit_traces(live, "actual-vlm"))
    print(
        json.dumps(
            {
                "audit": audit,
                "nominal": data["nominal"],
                "stress": data["stress"],
                "pass_bars": pass_bars,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
