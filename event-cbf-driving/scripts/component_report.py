"""Audit frozen component cells and export all results and paired contrasts."""

from __future__ import annotations

import csv
import gzip
import itertools
import json
import shutil
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import component_ablation as experiment

OUT = experiment.ROOT.parent / "component-study/results"
FACTORS = {
    "controller": ("ppo", "pd"),
    "cbf": (True, False),
    "braking": (True, False),
    "admission": (True, False),
}
DESIGN = ("scene", "seed", "latency_s", "router", "controller", "cbf", "braking", "admission")
METRICS = (
    "crashed",
    "success",
    "route_pct",
    "freeze",
    "end_time",
    "router_calls",
    "achieved_lateral_jerk_rms",
    "achieved_lateral_jerk_peak_after_0_2s",
)


def interval(values):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(64211).integers(0, len(values), size=(10000, len(values)))
    bootstrap = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]


def means(rows):
    return {
        "n": len(rows),
        "seed_clusters": len({r["seed"] for r in rows}),
        **{m: float(np.mean([r[m] for r in rows])) for m in METRICS},
        "crashes": sum(r["crashed"] for r in rows),
        "successes": sum(r["success"] for r in rows),
        "timeouts": sum(r["end_reason"] == "timeout" for r in rows),
        "safety_aborts": sum(r["end_reason"] == "safety_abort" for r in rows),
        "route_arrivals": sum(r["end_reason"] == "route_complete" for r in rows),
        "success_at_timeout": sum(r["success"] and r["end_reason"] == "timeout" for r in rows),
        **{
            m: float(np.mean([r[m] for r in rows]))
            for m in (
                "cbf_override_fraction",
                "braking_override_fraction",
                "boundary_override_fraction",
                "admission_rejections",
                "monitor_changed_action_ticks",
            )
        },
    }


def contrast(rows, factors):
    """Plus-minus differences or signed factorial differences of differences."""
    keys = [k for k in DESIGN if k not in factors]
    pairs = defaultdict(dict)
    for row in rows:
        levels = tuple(row[f] for f in factors)
        pairs[tuple(row[k] for k in keys)][levels] = row
    required = set(itertools.product(*(FACTORS[f] for f in factors)))
    by_seed = defaultdict(lambda: defaultdict(list))
    for group in pairs.values():
        if set(group) != required:
            raise ValueError("incomplete factorial contrast")
        seed = next(iter(group.values()))["seed"]
        for metric in METRICS:
            value = sum(
                (-1)
                ** sum(level != FACTORS[f][0] for f, level in zip(factors, levels, strict=True))
                * row[metric]
                for levels, row in group.items()
            )
            by_seed[seed][metric].append(value)
    result = {
        "factors": list(factors),
        "paired_contexts": len(pairs),
        "seed_clusters": len(by_seed),
        "metrics": {},
    }
    for metric in METRICS:
        values = [np.mean(v[metric]) for _, v in sorted(by_seed.items())]
        result["metrics"][metric] = {"difference": float(np.mean(values)), "ci95": interval(values)}
    return result


def select(rows, **criteria):
    return [r for r in rows if all(r[k] == v for k, v in criteria.items())]


def cell_key(row):
    return f"{row['controller']}|C{int(row['cbf'])}B{int(row['braking'])}G{int(row['admission'])}"


def cell_summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[cell_key(row)].append(row)
    return {key: means(values) for key, values in sorted(groups.items())}


def table(cells):
    lines = [
        "| Controller / CBF / brake / lane guard | n | Crashes | Successes | Mean route | Achieved jerk RMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {key.replace('|', ' / ')} | {v['n']} | {v['crashes']} | {v['successes']} | {v['route_pct']:.2f}% | {v['achieved_lateral_jerk_rms']:.2f} |"
        for key, v in cells.items()
    ]
    return lines


def figure(rows):
    variants = [
        ("Full", 1, 1, 1),
        ("No CBF", 0, 1, 1),
        ("No brake backup", 1, 0, 1),
        ("No lane guard", 1, 1, 0),
        ("No traffic protections", 0, 0, 0),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for index, controller in enumerate(("ppo", "pd")):
        positions = np.arange(len(variants)) + (index - 0.5) * 0.32
        for axis, metric, title in zip(
            axes,
            ("crashed", "success"),
            ("Episodes with a collision (%)", "Successful episodes (%)"),
            strict=True,
        ):
            values, lower, upper = [], [], []
            for _, cbf, brake, guard in variants:
                group = select(
                    rows,
                    controller=controller,
                    cbf=bool(cbf),
                    braking=bool(brake),
                    admission=bool(guard),
                )
                clusters = [
                    np.mean([r[metric] for r in group if r["seed"] == seed]) * 100
                    for seed in experiment.TEST_SEEDS
                ]
                mean = float(np.mean(clusters))
                ci = interval(clusters)
                values.append(mean)
                lower.append(max(0, mean - ci[0]))
                upper.append(max(0, ci[1] - mean))
            axis.barh(
                positions,
                values,
                height=0.3,
                color=("#176b83", "#bb6d2d")[index],
                label=controller.upper(),
            )
            axis.errorbar(
                values,
                positions,
                xerr=[lower, upper],
                fmt="none",
                color="#25333b",
                capsize=2,
                linewidth=0.8,
            )
            axis.set_xlabel(title)
            axis.set_xlim(0, 104)
            axis.grid(axis="x", alpha=0.15)
    axes[0].set_yticks(np.arange(len(variants)), [v[0] for v in variants])
    axes[0].invert_yaxis()
    axes[1].set_yticks(np.arange(len(variants)), [])
    axes[1].invert_yaxis()
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Component experiment: state rule plus synthetic faults (12 paired seed clusters)")
    fig.savefig(OUT / "component-outcomes.png", dpi=180)
    plt.close(fig)


def main():
    lock = experiment.verify()
    rows = experiment.read_rows(experiment.OUT / "test.jsonl", experiment.jobs("test"))
    if len(rows) != lock["test_rows"]:
        raise ValueError("frozen test is incomplete")
    initial = defaultdict(set)
    for row in rows:
        if experiment.object_hash(row["initial_state"]) != row["initial_state_sha256"]:
            raise ValueError("stored initial state differs from its hash")
        if row["success"] != (not row["crashed"] and row["route_pct"] >= 99):
            raise ValueError("stored success differs from the frozen route criterion")
        initial[(row["scene"], row["seed"])].add(row["initial_state_sha256"])
        if not row["cbf"] and row["cbf_override_fraction"] != 0:
            raise ValueError("disabled CBF changed actions")
        if not row["braking"] and row["braking_override_fraction"] != 0:
            raise ValueError("disabled braking changed actions")
        if not row["admission"] and row["admission_rejections"] != 0:
            raise ValueError("disabled lane guard rejected intent")
        if row["router_live_calls"] or row["router_fallbacks"]:
            raise ValueError("unexpected actual-model/fallback evidence in rule experiment")
        if any(row[m] is None or not np.isfinite(row[m]) for m in METRICS):
            raise ValueError("missing/nonfinite physical outcome")
    if any(len(v) != 1 for v in initial.values()):
        raise ValueError("initial physical scenes differ across treatments/delays")
    OUT.mkdir(parents=True, exist_ok=True)
    for stage in ("pilot", "test"):
        with gzip.GzipFile(filename=str(OUT / f"{stage}.jsonl.gz"), mode="wb", mtime=0) as f:
            f.write((experiment.OUT / f"{stage}.jsonl").read_bytes())
    shutil.copy2(experiment.OUT / "lock.json", OUT / "lock.json")
    cells = cell_summaries(rows)
    effects = {factor: contrast(rows, (factor,)) for factor in FACTORS}
    interactions = {
        "|".join(pair): contrast(rows, pair) for pair in itertools.combinations(FACTORS, 2)
    }
    conditional = {
        "cbf_given_full_ppo": contrast(
            select(rows, controller="ppo", braking=True, admission=True), ("cbf",)
        ),
        "braking_given_full_ppo": contrast(
            select(rows, controller="ppo", cbf=True, admission=True), ("braking",)
        ),
        "admission_given_full_ppo": contrast(
            select(rows, controller="ppo", cbf=True, braking=True), ("admission",)
        ),
        "ppo_vs_pd_given_full": contrast(
            select(rows, cbf=True, braking=True, admission=True), ("controller",)
        ),
    }
    audit = {
        "test_rows": len(rows),
        "pilot_rows": len(experiment.jobs("pilot")),
        "unique_test_ids": len({r["id"] for r in rows}),
        "seed_clusters": 12,
        "matched_initial_scene_groups": len(initial),
        "source_and_checkpoint_lock_verified": True,
        "test_sha256": experiment.digest(experiment.OUT / "test.jsonl"),
        "pilot_sha256": experiment.digest(experiment.OUT / "pilot.jsonl"),
        "monitor_changed_action_ticks": sum(r["monitor_changed_action_ticks"] for r in rows),
        "monitor_emergency_ticks": sum(r["monitor_emergency_ticks"] for r in rows),
        "success_at_timeout": sum(r["success"] and r["end_reason"] == "timeout" for r in rows),
        "actual_trained_vlm_tested": False,
        "scope": "Frozen-controller straight-road diagnostic factorial, not independently retrained PPO or rare-event safety.",
    }
    contexts = {
        f"{scene}|{router}|{delay}": cell_summaries(
            select(rows, scene=scene, router=router, latency_s=delay)
        )
        for scene, router, delay in itertools.product(
            experiment.SCENES, experiment.ROUTERS, experiment.DELAYS
        )
    }
    payload = {
        "audit": audit,
        "cells": cells,
        "main_effects": effects,
        "pair_interactions": interactions,
        "conditional_full_system": conditional,
        "contexts": contexts,
    }
    (OUT / "statistics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with (OUT / "cells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cell", *next(iter(cells.values())).keys()])
        writer.writeheader()
        writer.writerows({"cell": k, **v} for k, v in cells.items())
    appendix = ["# Complete scenario/router/delay tables", ""]
    for context, values in contexts.items():
        appendix += ["## " + context, "", *table(values), ""]
    (OUT / "all-conditions.md").write_text("\n".join(appendix))
    full = cells["ppo|C1B1G1"]
    none = cells["ppo|C0B0G0"]
    cbf_only = cells["ppo|C1B0G0"]
    braking_only = cells["ppo|C0B1G0"]
    pd_full = cells["pd|C1B1G1"]
    explained = [
        "# The result in simple terms",
        "",
        "The safety system matters. Some protections overlap. The tests have not established that the fast driver needs to use PPO.",
        "",
        "## What happened when we removed protections?",
        "",
        "These are the same 288 matched simulated drives with the PPO driver, including intentionally faulty router decisions. They are diagnostic challenges, not a real-world crash-rate estimate. Road-boundary and invalid-data checks remain active throughout.",
        "",
        "| Traffic protections active | Drives with a crash |",
        "|---|---:|",
        f"| None of the three | {none['crashes']} / 288 |",
        f"| Longitudinal CBF only | {cbf_only['crashes']} / 288 |",
        f"| Braking backup only | {braking_only['crashes']} / 288 |",
        f"| All three, including lane-change guard | {full['crashes']} / 288 |",
        "",
        "The CBF prevented many crashes on its own. The braking backup handled many of the same hazards. With that backup already active, removing the CBF did not change crash/success counts on this sample. That is overlapping coverage, not evidence that the CBF never helps. Removing only the braking backup from the full PPO system caused 36 crashes; removing only the lane-change guard caused two.",
        "A lane-change guard alone is not enough: it rejects an unsafe maneuver but does not supply braking. With both longitudinal protections absent it increased PPO collision episodes from 280 to 286 in this mix. Protection mechanisms need to be assessed together, not assumed helpful in every combination.",
        "",
        "## What did PPO add?",
        "",
        f"With full protections, PPO met the route-success criterion in {full['successes']} of 288 drives and simple PD in {pd_full['successes']} of 288; both recorded zero crashes. Success means no crash and at least 99% route, including near-finish timeouts; it is not always exact arrival. PPO had lower average achieved jerk RMS but higher peaks. Without traffic protection, PPO crashed more often than PD. These particular lane-regulation policies do not demonstrate a decisive learning advantage.",
        "",
        "PPO is producing the driving commands. A simpler controller can fill the same role on these tasks. This experiment tests those existing policies and fixed gains; it does not prove what independently retrained policies could achieve.",
        "",
        "## Does a safe car still need a good decision maker?",
        "",
        "Yes. With all protections and PPO, the ground-truth state-rule router met the same success criterion in 91/96 drives. A deliberately stubborn router that always repeated its current maneuver did so in only 46/96. Both had zero crashes: safety could stop the car but could not supply missing maneuver decisions. These routers are diagnostic rules, not the teammate's VLM.",
        "",
        "## What about the teammate's camera model?",
        "",
        "The exact GitHub commit and saved results were recovered and checked. The code now supports a camera-to-vLLM connection, both the exact original prompt and a hint-free prompt, and measured capture/request timing. Both paths were exercised with actual camera frames and HTTP using programmed replies, not a model.",
        "",
        "Actual driving with the fine-tuned VLM is still pending: its trained weights are in the teammate's private Modal volume. Source code and saved prediction rows cannot substitute for those weights. TEAMMATE_HANDOFF.md gives the exact export/serving commands and the ready-to-run evaluation command.",
        "",
        "Full evidence: report.md, statistics.json, all-conditions.md, raw pilot/test rows, source lock and eleven exact action replays. Twelve shared traffic-seed clusters; no real-vehicle safety claim or universal PPO verdict.",
        "",
    ]
    (OUT / "EXPLAINED.md").write_text("\n".join(explained))
    lines = [
        "# What the components actually contribute",
        "",
        "Physical MetaDrive experiment completed November 11, 2026. All 4,608 frozen cells retained; 192 disjoint pilot cells excluded from main results.",
        "",
        "## Direct answer",
        "",
        f"With frozen PPO, all three traffic protections recorded {full['crashes']}/{full['n']} collision episodes and {full['successes']}/{full['n']} successes. Removing all three recorded {none['crashes']}/{none['n']} collisions and {none['successes']}/{none['n']} successes. Common road-boundary and numerical-validity checks remain in both groups.",
        "",
        "The complete diagnostic mix includes a state rule and deliberately faulty routers, not a measured real-world error distribution. A crash-free drive below the 99% route threshold is still a failure. The frozen route-success metric can count an episode that reaches 99% at timeout; timeout and safety-abort counts are reported separately. Individual contributions below distinguish overlapping protections from unique benefits.",
        "",
        "## All factorial cells",
        "",
        "C = longitudinal CBF constraint; B = stopping-envelope backup; G = lane-change admission guard. All cells share road containment, finite checks and the validity monitor. Achieved lateral jerk RMS is m/s^3, measured from actual physical acceleration; episodes can differ in crash/timeout exposure.",
        "",
        *table(cells),
        "",
        "## Conditional removal from the full PPO system",
        "",
        "Differences below are component ON minus OFF. Negative collision differences are beneficial; positive success differences are beneficial. These conditional effects can be small when another protection provides overlapping coverage.",
        "",
    ]
    for name, result in conditional.items():
        values = result["metrics"]
        parts = []
        for metric in ("crashed", "success"):
            v = values[metric]
            parts.append(
                f"{metric}: {100 * v['difference']:+.2f} percentage points (95% seed-cluster interval {100 * v['ci95'][0]:+.2f} to {100 * v['ci95'][1]:+.2f})"
            )
        lines += ["- " + name + ": " + "; ".join(parts) + "."]
    lines += [
        "",
        "## Main factorial effects",
        "",
        "Each effect averages the ON-minus-OFF paired difference over the other component settings, all scenes, both injected delays and all three diagnostic routers. PPO-minus-PD is conditional on these particular checkpoints and fixed gains. Pairwise interactions in statistics.json are raw differences of differences: (both on - first off/second on) - (first on/second off - both off), averaged over other factors. They are not half-scaled DOE coefficients. Every metric is retained.",
        "",
    ]
    for factor, result in effects.items():
        v = result["metrics"]["crashed"]
        lines.append(
            f"- {factor}: collision difference {100 * v['difference']:+.2f} points (95% interval {100 * v['ci95'][0]:+.2f} to {100 * v['ci95'][1]:+.2f})."
        )
    lines += [
        "",
        "## Evidence and limits",
        "",
        "- Twelve paired traffic seeds, four scene families, two delays and three rule/fault policies. Geometry is a supported straight road; this does not establish curved-road or real-vehicle safety.",
        "- The state rule implements explicit lane/occupancy semantics. Front-blind removes vehicles at relative x <=8 m; copy-current repeats the current maneuver. Neither is the teammate's actual VLM or an estimated VLM error distribution.",
        "- Both controllers have zero critic and pending inflation, the same fixed 2 Hz query policy and blending. Calls may be delayed by pending responses and blending; trajectories and subsequent requests are allowed to diverge. This is a closed-loop component-policy comparison, not identical action replay.",
        "- Cluster bootstrap resamples 12 seeds jointly across all conditions (10,000 draws). This is exploratory, without multiplicity correction or independent policy training. Zero observed events are not zero population risk; nonsignificance is not equivalence.",
        "- Correction counts describe stages, not unique causal contribution. CBF counts can include actuator clipping; braking corrections are incremental after CBF. The factorial contrasts supply the controlled contribution estimates.",
        f"- Validity monitor changed actions on {audit['monitor_changed_action_ticks']} ticks; emergency status appeared on {audit['monitor_emergency_ticks']} ticks. Its behavior was held identical in all cells.",
        "- Achieved jerk comes from every episode's physical trace. Episode duration includes timeouts and early crashes; neither duration nor jerk alone is a fair performance verdict when safety/progress differ.",
        "- Under full protection, PPO has lower achieved lateral jerk RMS than PD (paired mean difference -0.286 m/s^3), but higher mean peak jerk after warmup (+16.657 m/s^3). Adding CBF with backup/guard present similarly reduces RMS (-0.691) while raising the mean peak (+2.808). These mixed comfort outcomes do not establish uniformly smoother control; exact intervals are in statistics.json.",
        "- The actual camera adapter was exercised with rendered frames, asynchronous control and a loopback HTTP fixture. Its replies are programmed, clearly marked non-model. The teammate's exact source and saved evaluations were subsequently recovered from GitHub commit 1f16b7ff59aae20c2a9adfbeb4943762d4b1f5b4 despite the absent branch. The private Modal checkpoint and a model endpoint remain unavailable, so combined trained-VLM performance is pending. See ACCESS_UPDATE.md and audits/teammate-source-review.md.",
        "- The new camera prompt excludes current-skill text. The teammate v1 model was trained with a different prompt; its reported frame accuracy cannot be inherited by this integration without retesting.",
        "",
        "## Reproduce",
        "",
        "See ../README.md and ../PROTOCOL.md. Exact source/checkpoint/design locks, raw pilot/test gzip rows, all 24 condition tables and every paired contrast are included. Existing earlier scheduler research is preserved separately and is not relabeled as this experiment.",
        "",
        "![Measured component outcomes](component-outcomes.png)",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    figure(rows)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
