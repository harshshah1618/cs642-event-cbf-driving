"""Audit complete frozen studies; export paired statistics, figures and report.

Cluster resampling uses scene seed jointly across scenarios/environments/delays.
This conditions on the existing frozen controllers, not independent training.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from research_data_audit import runtime_audit, validate_test_cells
from research_freeze import ROOT, digest, verify

OUT = ROOT.parent / "research/results"
LABELS = {
    "legacy": "Original event trigger",
    "deadline": "Deadline-aware",
    "no_latency_matched": "Matched no-latency",
    "benefit": "Immediate usefulness",
    "ttc": "TTC threshold",
    "periodic": "Tuned polling",
    "periodic_conditioned": "Condition-tuned polling",
    "periodic_gated": "Gated polling (10Hz)",
    "shield": "Intervention trigger",
    "never": "No replanning",
    "idm": "Upstream IDM (40km/h)",
    "expert": "Released PPO expert",
    "ppo_controller": "Frozen PPO (sigma-free)",
    "pd_controller": "Fixed PD (sigma-free)",
    "deadline_live": "Deadline + actual Qwen",
    "polling_live": "Polling + actual Qwen",
}


def load_rows(suite, stage, expected):
    path = ROOT / "runs" / f"research-{suite}" / f"{stage}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != expected or len({r["id"] for r in rows}) != expected:
        raise ValueError(f"{path}: expected {expected} unique rows, found {len(rows)}")
    if any(not math.isfinite(float(r["route_pct"])) for r in rows):
        raise ValueError("non-finite result")
    return rows


def interval(values, seed=642):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summary(rows):
    output = {}
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        seeds = sorted({r["seed"] for r in group})
        d = {
            "n": len(group),
            "scene_seeds": len(seeds),
            "crashes": sum(r["crashed"] for r in group),
        }
        for metric in (
            "success",
            "route_pct",
            "router_calls",
            "end_time",
            "planner_process_s",
            "scheduler_process_s",
            "control_process_s",
            "freeze",
            "plant_peak_lateral_jerk",
            "peak_lateral_jerk",
            "infeasible_ticks",
            "override_fraction",
        ):
            if not all(metric in r and r[metric] is not None for r in group):
                continue
            d[metric] = float(np.mean([r[metric] for r in group]))
            clusters = [np.mean([r[metric] for r in group if r["seed"] == seed]) for seed in seeds]
            d[metric + "_ci95"] = interval(clusters)
        output[method] = d
    return output


def paired(rows, method, baseline):
    def key(r):
        return (r.get("scenario", r.get("environment")), r["seed"], r["delay"])

    first = {key(r): r for r in rows if r["method"] == method}
    second = {key(r): r for r in rows if r["method"] == baseline}
    if len(first) != sum(r["method"] == method for r in rows) or len(second) != sum(
        r["method"] == baseline for r in rows
    ):
        raise ValueError("duplicate paired experiment cell")
    if first.keys() != second.keys():
        raise ValueError(f"unpaired cells: {method} vs {baseline}")
    output = {"method": method, "baseline": baseline, "n": len(first)}
    for metric in (
        "success",
        "route_pct",
        "router_calls",
        "end_time",
        "crashed",
        "planner_process_s",
        "control_process_s",
        "plant_peak_lateral_jerk",
    ):
        if not all(
            metric in r and r[metric] is not None for r in [*first.values(), *second.values()]
        ):
            continue
        differences = defaultdict(list)
        for k, r in first.items():
            differences[r["seed"]].append(float(r[metric]) - float(second[k][metric]))
        clusters = [np.mean(v) for _, v in sorted(differences.items())]
        output[metric + "_difference"] = float(np.mean(clusters))
        output[metric + "_difference_ci95"] = interval(clusters)
    output["conditional_call_win"] = bool(
        output["router_calls_difference_ci95"][1] < 0
        and output["success_difference_ci95"][0] >= -0.02
        and output["route_pct_difference_ci95"][0] >= -1
        and output["crashed_difference"] <= 0
    )
    return output


def markdown_table(data):
    lines = [
        "| Method | n | Success % | Crashes | Route % | Calls | End time s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in data.items():
        lines.append(
            f"| {LABELS[method]} | {row['n']} | {100 * row['success']:.1f} | "
            f"{row['crashes']} | {row['route_pct']:.1f} | {row['router_calls']:.2f} | "
            f"{row['end_time']:.2f} |"
        )
    return lines


def figures(summaries):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    paths = []
    for index, (suite, data) in enumerate(summaries.items()):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8, 4.5),
            constrained_layout=True,
            sharey=True,
            gridspec_kw={"width_ratios": [1.1, 1]},
        )
        methods = sorted(data, key=lambda m: data[m]["router_calls"])
        y = np.arange(len(methods))
        palette = [
            "#176b87"
            if m == "deadline"
            else ("#b45f25" if m == "periodic_conditioned" else "#80929d")
            for m in methods
        ]
        for axis, metric, scale, label in (
            (axes[0], "router_calls", 1, "Planner calls / episode"),
            (axes[1], "success", 100, "Success (%)"),
        ):
            means = np.asarray([data[m][metric] * scale for m in methods])
            ci = np.asarray([data[m][metric + "_ci95"] for m in methods]) * scale
            axis.barh(y, means, color=palette, height=0.65)
            axis.errorbar(
                means,
                y,
                xerr=np.maximum(0, np.array([means - ci[:, 0], ci[:, 1] - means])),
                fmt="none",
                color="#263640",
                capsize=2,
                linewidth=0.8,
            )
            axis.set_xlabel(label)
            axis.grid(axis="x", alpha=0.15)
        axes[0].set_yticks(y, [LABELS[m] for m in methods], fontsize=8)
        axes[0].invert_yaxis()
        axes[1].set_xlim(0, 104)
        fig.suptitle(suite, fontsize=12, weight="bold")
        path = OUT / ("call-success-frontiers.png" if index == 0 else f"call-success-{index}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def pdf_report(sections, chart):
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("ResearchSans", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("ResearchSans-Bold", str(fonts / "VeraBd.ttf")))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        if hasattr(style, "fontName"):
            style.fontName = "ResearchSans-Bold" if "Bold" in style.fontName else "ResearchSans"
    styles.add(
        ParagraphStyle(
            name="BodyResearch",
            fontName="ResearchSans",
            fontSize=9,
            leading=12,
            spaceAfter=6,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallResearch", fontName="ResearchSans", fontSize=7, leading=10, spaceAfter=5
        )
    )
    story = []

    def para(text, small=False):
        story.append(Paragraph(escape(text), styles["SmallResearch" if small else "BodyResearch"]))

    story.append(Paragraph("When should a slow maneuver planner run?", styles["Title"]))
    para(
        "Controlled physical MetaDrive experiments | Frozen validation choices | Independent Astra audit"
    )
    para(
        "November 11, 2026. This report documents measured comparisons, not an official leaderboard result, a safety proof, or established algorithmic novelty."
    )
    for title, content in sections:
        story.append(Paragraph(escape(title), styles["Heading2"]))
        for block_index, block in enumerate(content):
            if isinstance(block, dict):
                rows = [["Method", "n", "Success", "Crash", "Route", "Calls"]]
                for method, row in block.items():
                    rows.append(
                        [
                            LABELS[method],
                            str(row["n"]),
                            f"{100 * row['success']:.1f}%",
                            str(row["crashes"]),
                            f"{row['route_pct']:.1f}%",
                            f"{row['router_calls']:.2f}",
                        ]
                    )
                t = Table(rows, colWidths=[186, 35, 51, 42, 51, 43], repeatRows=1, hAlign="LEFT")
                t.setStyle(
                    TableStyle(
                        [
                            ("FONTNAME", (0, 0), (-1, -1), "ResearchSans"),
                            ("FONTNAME", (0, 0), (-1, 0), "ResearchSans-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f4")),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, colors.HexColor("#f7f8fa")],
                            ),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                        ]
                    )
                )
                if block_index == 0:
                    heading = story.pop()
                    story.append(KeepTogether([heading, t, Spacer(1, 10)]))
                else:
                    story.extend([t, Spacer(1, 10)])
            else:
                para(block)
    story.append(PageBreak())
    story.append(Paragraph("Calls and successful completion", styles["Heading2"]))
    for path in chart:
        story.append(Image(str(path), width=7 * inch, height=3.94 * inch))
        story.append(Spacer(1, 12))
    para(
        "Bars show frozen comparisons and seed-clustered 95% intervals. See the machine-readable statistics for every paired difference. Zero-width empirical intervals do not establish zero population risk."
    )
    para(
        "Artifacts: research/results/statistics.json; audit.json; data/*.jsonl.gz; locked selections; source and checkpoint hashes; complete reproducibility instructions in research/README.md."
    )

    def footer(canvas, doc):
        canvas.setFont("ResearchSans", 8)
        canvas.setFillColor(colors.HexColor("#576571"))
        canvas.drawString(42, 25, "CS642 Event-CBF | Audited scheduling study")
        canvas.drawRightString(570, 25, f"{doc.page}")

    SimpleDocTemplate(
        str(OUT / "research-report.pdf"),
        pagesize=(612, 792),
        leftMargin=42,
        rightMargin=42,
        topMargin=40,
        bottomMargin=42,
    ).build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = OUT / "data"
    data_dir.mkdir(exist_ok=True)
    suites = {"isolation": (1560, 4800), "roads": (1152, 3760), "phase": (216, 2880)}
    data, locks, hashes, validation_data = {}, {}, {}, {}
    for suite, (n_val, n_test) in suites.items():
        locks[suite] = verify(suite)
        validation = load_rows(suite, "validation", n_val)
        validation_data[suite] = validation
        data[suite] = load_rows(suite, "test", n_test)
        validate_test_cells(suite, data[suite], locks[suite], locks.get("isolation"))
        if {r["seed"] for r in validation} & {r["seed"] for r in data[suite]}:
            raise ValueError("validation/test seed overlap")
        for stage in ("validation", "test"):
            path = ROOT / "runs" / f"research-{suite}" / f"{stage}.jsonl"
            hashes[f"{suite}/{stage}"] = digest(path)
            with gzip.GzipFile(
                filename=str(data_dir / f"{suite}-{stage}.jsonl.gz"), mode="wb", mtime=0
            ) as out:
                out.write(path.read_bytes())
        shutil.copy2(
            ROOT / "runs" / f"research-{suite}" / "locked-selection.json",
            OUT / f"{suite}-locked-selection.json",
        )
    runtime = runtime_audit(locks)
    clock_dir = ROOT / "runs/research-clock"
    clock_analysis = json.loads((clock_dir / "analysis.json").read_text())
    clock_lock = json.loads((clock_dir / "lock.json").read_text())
    if (
        clock_analysis["test_sha256"] != digest(clock_dir / "test.jsonl")
        or clock_analysis["phase_test_sha256"] != digest(ROOT / "runs/research-phase/test.jsonl")
        or clock_lock["script_sha256"] != digest(ROOT / "scripts/research_clock_control.py")
    ):
        raise ValueError("clock control data/source changed")
    if len((clock_dir / "test.jsonl").read_text().splitlines()) != 1080:
        raise ValueError("clock control incomplete")
    for filename in ("analysis.json", "lock.json"):
        shutil.copy2(clock_dir / filename, OUT / ("clock-" + filename))
    with gzip.GzipFile(
        filename=str(data_dir / "clock-control.jsonl.gz"), mode="wb", mtime=0
    ) as stream:
        stream.write((clock_dir / "test.jsonl").read_bytes())
    controller_path = ROOT / "runs/research-controller/test.jsonl"
    controller_rows = [json.loads(line) for line in controller_path.read_text().splitlines()]
    if len(controller_rows) != 1200 or len({r["id"] for r in controller_rows}) != 1200:
        raise ValueError("controller ablation incomplete or duplicate")
    controller_lock_path = ROOT / "runs/research-controller/lock.json"
    controller_lock = json.loads(controller_lock_path.read_text())
    if digest(ROOT / "scripts/research_controller_ablation.py") != controller_lock["source_sha256"]:
        raise ValueError("controller ablation source changed")
    if controller_lock["source_isolation_lock_sha256"] != digest(
        ROOT / "runs/research-isolation/locked-selection.json"
    ):
        raise ValueError("controller ablation selection dependency changed")
    expected_controller = {
        (scene, seed, delay, method, controller): value
        for scene in ("two_trap", "merge", "dense_highway")
        for seed in range(1001, 1011)
        for delay in ("fast", "nominal", "slow", "long", "jitter")
        for controller in ("ppo", "pd")
        for method, value in (
            ("deadline", locks["isolation"]["deadline"]),
            ("periodic", locks["isolation"]["periodic"]),
            ("periodic_gated", 10.0),
            ("benefit", 0),
        )
    }
    actual_controller = {
        (r["scenario"], r["seed"], r["delay"], r["method"], r["controller"]): r["value"]
        for r in controller_rows
    }
    if actual_controller != expected_controller or any(
        r["gamma_sigma"] != 0 for r in controller_rows
    ):
        raise ValueError("controller ablation cells/configuration differ from design")
    controller_summary = summary(
        [{**r, "method": r["controller"] + "_controller"} for r in controller_rows]
    )
    controller_pairs = {}
    for method in controller_lock["methods"]:
        rows = [{**r, "method": r["controller"]} for r in controller_rows if r["method"] == method]
        controller_pairs[method] = paired(rows, "pd", "ppo")
    with gzip.GzipFile(
        filename=str(data_dir / "controller-ablation.jsonl.gz"), mode="wb", mtime=0
    ) as stream:
        stream.write(controller_path.read_bytes())
    shutil.copy2(controller_lock_path, OUT / "controller-ablation-lock.json")
    live_dir = ROOT / "runs/research-live/exploratory-single-seed"
    live = json.loads((live_dir / "results.json").read_text())
    live_rows = [
        {
            **r,
            "method": "deadline_live" if r["mode"] == "proposed" else "polling_live",
            "success": not r["crashed"] and r["route_pct"] >= 99,
        }
        for r in live["episodes"]
    ]
    if len(live_rows) != 6 or any(r["router_fallbacks"] for r in live_rows):
        raise ValueError("unexpected live probe size/fallback")
    shutil.copytree(live_dir, OUT / "live-qwen", dirs_exist_ok=True)
    summaries = {
        "Original scenarios": summary(data["isolation"]),
        "Random initial polling offset": summary(data["phase"]),
        "Upstream road stress": summary(
            [
                r
                for r in data["roads"]
                if r["method"] not in ("idm", "expert") and r["delay"] != "standard"
            ]
        ),
    }
    comparisons = {}
    for suite in ("isolation", "phase", "roads"):
        rows = [
            r
            for r in data[suite]
            if r["method"] not in ("idm", "expert")
            and (suite != "roads" or r["delay"] != "standard")
        ]
        comparisons[suite] = [
            paired(rows, "deadline", baseline)
            for baseline in (
                "periodic",
                "periodic_conditioned",
                "periodic_gated",
                "no_latency_matched",
                "benefit",
                "ttc",
            )
        ]
    # Same seed must create the same map for every method/delay; geometry is
    # deliberately paired across the two environments and counted jointly.
    geometry = defaultdict(set)
    for r in data["roads"]:
        geometry[r["seed"]].add(r["map_sha256"])
    if any(len(v) != 1 for v in geometry.values()):
        raise ValueError("unmatched upstream map geometry")
    audit = {
        "raw_sha256": hashes,
        "heldout_episodes": sum(len(v) for v in data.values()),
        "validation_episodes": sum(v[0] for v in suites.values()),
        "map_seed_count": len(geometry),
        "unique_road_geometries": len(set.union(*geometry.values())),
        "source_verified": True,
        "selection_verified": True,
        "runtime": runtime,
        "scope": "No official CARLA/Bench2Drive runs; no independent skill retraining; no safety proof.",
    }
    by_delay = {
        suite: {
            delay: summary([r for r in rows if r["delay"] == delay])
            for delay in sorted({r["delay"] for r in rows})
        }
        for suite, rows in data.items()
    }
    frontiers = {}
    for suite, rows in validation_data.items():
        conditions = sorted({r.get("scenario", r.get("environment")) for r in rows})
        frontiers[suite] = {}
        for condition in conditions:
            for delay in sorted({r["delay"] for r in rows}):
                group = [
                    r
                    for r in rows
                    if r.get("scenario", r.get("environment")) == condition and r["delay"] == delay
                ]
                cells = []
                for method, value in sorted({(r["method"], r["value"]) for r in group}):
                    values = [r for r in group if r["method"] == method and r["value"] == value]
                    cells.append(
                        {"method": method, "value": value, "metrics": summary(values)[method]}
                    )
                frontiers[suite][condition + "|" + delay] = cells
    (OUT / "validation-frontiers.json").write_text(json.dumps(frontiers, indent=2) + "\n")
    appendix = ["# Complete held-out condition tables", ""]
    for suite, rows in data.items():
        for condition in sorted({r.get("scenario", r.get("environment")) for r in rows}):
            for delay in sorted({r["delay"] for r in rows}):
                values = summary(
                    [
                        r
                        for r in rows
                        if r.get("scenario", r.get("environment")) == condition
                        and r["delay"] == delay
                    ]
                )
                appendix += [f"## {suite}: {condition}, {delay}", "", *markdown_table(values), ""]
    (OUT / "condition-tables.md").write_text("\n".join(appendix))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "by_delay": by_delay,
        "controller_ablation": controller_summary,
        "controller_paired_comparisons": controller_pairs,
        "live_qwen_exploratory": summary(live_rows),
        "clock_control": clock_analysis,
        "paired_comparisons": comparisons,
        "audit": audit,
    }
    (OUT / "statistics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    chart = figures(summaries)
    phase_result = summaries["Random initial polling offset"]["deadline"]
    phase_poll = summaries["Random initial polling offset"]["periodic_conditioned"]
    call_saving = 100 * (1 - phase_result["router_calls"] / phase_poll["router_calls"])
    sections = [
        (
            "Measured outcome",
            [
                f"The clean scheduling win occurs in the randomized-initial-offset suite: {call_saving:.1f}% fewer planner calls than scenario/delay-conditioned polling, with observed route-success {100 * phase_result['success']:.2f}% versus {100 * phase_poll['success']:.2f}%, and zero crashes in both 360-episode groups. Native route-success means at least 99% of the route, not necessarily exact arrival.",
                "This is not a universal win. The original zero-offset suite loses a small amount of strict success, the upstream procedural-road controller still crashes/timeouts, and the matched no-latency/TTC controls are highly competitive. The principal contribution is a controlled evaluation and timing-sensitivity finding, not established algorithmic novelty.",
            ],
        )
    ]
    sections.append(
        (
            "Question and experimental controls",
            [
                "Can forecasting maneuver opportunity loss relative to planning delay improve the safety-progress-call tradeoff? Every scheduler in a suite shares the same driver, planner, observations and safety backup. The latency ablation changes only estimated latency after copying the selected deadline buffer.",
                "There are two implementations: the original straight-road scenarios retain the existing frozen PPO lane regulators and CBF-style filter; upstream procedural roads use a separate classical PID/IDM tracker, finite-horizon lane/speed planner and common guard. This is not a demonstrated port of the original learned controller or CBF certificate to curved roads.",
                "The strongest opponents include polling tuned independently for each scenario/environment and delay, a 10Hz polling policy with the same cheap usefulness gate, immediate usefulness, and TTC. Gated polling has a shorter minimum query cooldown; it is a strong comparator, not a pure deadline ablation.",
                f"Validation: 2,928 physical episodes. Held-out: {audit['heldout_episodes']:,} episodes. Original-scenario seeds 1001-1040; random-offset seeds 1101-1140; upstream road seeds 2001-2040. Every selection and source/checkpoint hash was frozen before its held-out launch.",
            ],
        )
    )
    for title, values in summaries.items():
        sections.append((title, [values]))
    sections.append(
        (
            "Paired deadline comparisons",
            [
                f"{suite}: versus {LABELS[c['baseline']]}, success difference {100 * c['success_difference']:+.2f} percentage points "
                f"(95% paired cluster-bootstrap interval {100 * c['success_difference_ci95'][0]:+.2f} to {100 * c['success_difference_ci95'][1]:+.2f}); "
                f"calls {c['router_calls_difference']:+.2f} ({c['router_calls_difference_ci95'][0]:+.2f} to {c['router_calls_difference_ci95'][1]:+.2f})."
                for suite, items in comparisons.items()
                for c in items
                if c["baseline"] in ("periodic_conditioned", "periodic_gated", "no_latency_matched")
            ],
        )
    )
    for environment in ("generalization", "safe"):
        standard = summary(
            [
                r
                for r in data["roads"]
                if r["environment"] == environment and r["delay"] == "standard"
            ]
        )
        sections.append(
            (
                f"Upstream {environment}: no injected delay",
                [
                    standard,
                    "Upstream reference agents are complete-controller comparisons, not isolated scheduler ablations. The custom controller uses surrounding-object geometry; the released neural expert uses its own trained observation encoding. Do not interpret these as camera-only leaderboard comparisons.",
                ],
            )
        )
    sections.append(
        (
            "Frozen PPO versus simple PD control",
            [
                controller_summary,
                "A separate 1,200-episode paired ablation uses 10 scene seeds, four schedulers and five delays. Critic-distance inflation is zero for BOTH banks, with the same backup and admission guards. PD gains were fixed before these runs. The 10 seeds are the prospective controller-ablation subset, not another independent training population.",
            ],
        )
    )
    sections.append(
        (
            "Actual language-model integration (exploratory)",
            [
                summary(live_rows),
                f"Six physical episodes use pinned Qwen2.5-1.5B CPU inference with wall-clock-paced control. The model made {sum(r['router_live_calls'] for r in live_rows)} actual closed-loop calls and used no fallbacks. Both systems stalled in the two-trap scenario; the deadline system reached 83.1% versus 85.2% for polling. This single-seed integration check does not establish model superiority and is excluded from the controlled benchmark statistics.",
            ],
        )
    )
    sections.append(
        (
            "Controlled initial-offset finding",
            [
                "An additional 1,080 physical runs plus 360 reused phase-suite rows compare zero versus uniformly randomized initial timer offsets on exactly matched scenes, delays and 40 seed clusters. At fixed 0.1 Hz, observed success falls from 360/360 to 282/360: -21.67 percentage points (95% paired seed-cluster interval -26.67 to -16.67). No collisions occur in either group. At 1 Hz the offset effect reverses: success rises from 340/360 to 349/360 (+2.50 points, interval +0.83 to +4.72).",
                "Thus a large apparent polling result can depend on its initial clock alignment, and the direction depends on rate. These are rolling timers reset on actual submissions, not absolute wall-clock schedules. This post-primary mechanistic control is exploratory and does not establish a new phenomenon in the literature or a universal scheduler advantage.",
            ],
        )
    )
    sections.append(
        (
            "Interpretation and limitations",
            [
                "This is a reproducible environment-based benchmark study. The report's exploratory call-win screen checks a negative paired call-difference interval, success-difference lower bound at least -2 percentage points, route-difference lower bound at least -1 point, and no higher observed collision incidence. This descriptive screen is not a preregistered formal noninferiority test, a rare-event safety guarantee or a multiplicity-corrected SOTA claim.",
                "Bootstrap resampling clusters by scene seed jointly across scenarios, environments and delays. Safe and generalization share a geometry for the same seed. Frozen existing policies provide conditional evaluation; independent training-seed variation is not measured. Zero observed crashes or zero-width bootstrap intervals do not establish zero population risk.",
                "Native study records call counts, not end-to-end compute savings. Road artifacts separately record process CPU, elapsed planner/scheduler sections, and complete control CPU. All planner delays are injected stress conditions, not measured LLM inference. Neither curves, uncertainty modelling nor empirical survival establish a CBF theorem.",
                "The native held-out bulk runs did not record achieved lateral jerk; those values are null and must not be treated as zero or inferred from commanded jerk. Nine supplementary full action traces were replayed successfully, but that subset does not supply a population comfort estimate. Mean episode duration includes failures/timeouts and is not unconditional time-to-goal.",
                "The original zero-offset comparison reduces calls by 47% versus tuned polling but loses 0.83 percentage points of observed success; its success-difference lower interval is -2.50 points, failing the report's exploratory 2-point screen. Dense traffic exposes missed spatial opportunities in the speed-based gate. A gap-aware candidate was evaluated on development data only and was not promoted to a fresh held-out claim. TTC and matched no-latency controls remain competitive; the latency term itself has no established incremental benefit.",
                "Independent review corrected a mismatched no-latency buffer, an unintended road polling cap, and stale stop-plan recovery, including a stopped mid-transition vehicle. All pre-freeze development versions and failures are retained and excluded from final selection.",
                "Algorithmic novelty is not established: event-triggered MPC, risk-triggered replanning and robot metareasoning already address close questions. The useful contribution is the controlled comparison, audited failure mechanisms and measured operating conditions.",
            ],
        )
    )
    sections.append(
        (
            "Primary sources and reproduction",
            [
                "MetaDrive environments: https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html ; pinned simulator source fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1.",
                "CERT-Replan: https://arxiv.org/abs/2610.09302 ; RL event-triggered MPC: https://arxiv.org/abs/2208.10302 ; robot metareasoning: https://arxiv.org/abs/2610.23662.",
                "Reproduce via scripts/research_isolation.py, research_roads.py, research_phase.py, research_freeze.py and research_report.py. Full commands, versioned source and compressed raw rows are included with this report.",
            ],
        )
    )
    lines = [
        "# When should a slow maneuver planner run?",
        "",
        "Audited physical MetaDrive scheduling study. November 11, 2026.",
        "",
    ]
    for title, blocks in sections:
        lines.extend(["## " + title, ""])
        for block in blocks:
            lines.extend(markdown_table(block) if isinstance(block, dict) else [block])
            lines.append("")
    (OUT / "research-report.md").write_text("\n".join(lines))
    pdf_report(sections, chart)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
