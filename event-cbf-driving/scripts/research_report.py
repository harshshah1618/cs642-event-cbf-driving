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
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
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
        ):
            if not all(metric in r for r in group):
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
    ):
        if not all(metric in r for r in first.values()):
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, (suite, data) in zip(axes, summaries.items(), strict=True):
        for method, row in data.items():
            if method in ("idm", "expert"):
                continue
            color = (
                "#176b87"
                if method == "deadline"
                else ("#b45f25" if method == "periodic_conditioned" else "#667581")
            )
            axis.scatter(
                row["router_calls"],
                row["success"] * 100,
                s=55,
                c=color,
                marker="D" if method == "deadline" else "o",
            )
            axis.annotate(
                LABELS[method],
                (row["router_calls"], row["success"] * 100),
                xytext=(3, 5 if method != "benefit" else -14),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set(title=suite, xlabel="Planner calls per episode", ylabel="Success (%)")
        axis.grid(alpha=0.15)
    path = OUT / "call-success-frontiers.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def pdf_report(sections, chart):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyResearch",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            spaceAfter=7,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallResearch", fontName="Helvetica", fontSize=7, leading=10, spaceAfter=5
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
        for block in content:
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
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
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
                story.extend([t, Spacer(1, 10)])
            else:
                para(block)
    story.append(PageBreak())
    story.append(Paragraph("Call / success frontiers", styles["Heading2"]))
    story.append(Image(str(chart), width=7 * inch, height=2.1 * inch))
    para(
        "Each point summarizes a frozen comparison. See the machine-readable statistics for seed-clustered confidence intervals and every paired difference. Overlapping labels do not imply equal performance."
    )
    para(
        "Artifacts: research/results/statistics.json; audit.json; data/*.jsonl.gz; locked selections; source and checkpoint hashes; complete reproducibility instructions in research/README.md."
    )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
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
        "paired_comparisons": comparisons,
        "audit": audit,
    }
    (OUT / "statistics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    chart = figures(summaries)
    sections = []
    sections.append(
        (
            "Question and experimental controls",
            [
                "Can forecasting maneuver opportunity loss relative to planning delay improve the safety-progress-call tradeoff? Every scheduler in a suite shares the same driver, planner, observations and safety backup. The latency ablation changes only estimated latency after copying the selected deadline buffer.",
                "The strongest opponents include polling tuned independently for each scenario/environment and delay, a10Hz polling policy with the same cheap usefulness gate, immediate usefulness, and TTC. Gated polling has a shorter minimum query cooldown; it is a strong comparator, not a pure deadline ablation.",
                f"Validation: 2,928 physical episodes. Held-out: {audit['heldout_episodes']:,} episodes. Original-scenario seeds1001-1040; random-offset seeds1101-1140; upstream road seeds2001-2040. Every selection and source/checkpoint hash was frozen before its held-out launch.",
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
            "Interpretation and limitations",
            [
                "This is a reproducible environment-based benchmark study. The report's exploratory call-win screen checks a negative paired call-difference interval, success-difference lower bound at least -2 percentage points, route-difference lower bound at least -1 point, and no higher observed collision incidence. This descriptive screen is not a preregistered formal noninferiority test, a rare-event safety guarantee or a multiplicity-corrected SOTA claim.",
                "Bootstrap resampling clusters by scene seed jointly across scenarios, environments and delays. Safe and generalization share a geometry for the same seed. Frozen existing policies provide conditional evaluation; independent training-seed variation is not measured. Zero observed crashes or zero-width bootstrap intervals do not establish zero population risk.",
                "Native study records call counts, not end-to-end compute savings. Road artifacts separately record process CPU, elapsed planner/scheduler sections, and complete control CPU. All planner delays are injected stress conditions, not measured LLM inference. Neither curves, uncertainty modelling nor empirical survival establish a CBF theorem.",
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
        "Audited physical MetaDrive scheduling study. November11,2026.",
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
