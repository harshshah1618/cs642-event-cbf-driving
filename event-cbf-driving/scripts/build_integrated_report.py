"""Build a seven-page measured report plus references using bundled ReportLab.

Run with the Codex bundled Python. Policy evaluation uses the project's Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "final-study"
RESULTS = STUDY / "results"

# Embed fonts so system substitution cannot change character metrics.
for face, filename in (
    ("ReportSerif", "LiberationSerif-Regular.ttf"),
    ("ReportSerif-Bold", "LiberationSerif-Bold.ttf"),
    ("ReportSerif-Italic", "LiberationSerif-Italic.ttf"),
    ("ReportSerif-BoldItalic", "LiberationSerif-BoldItalic.ttf"),
    ("ReportSans", "LiberationSans-Regular.ttf"),
    ("ReportSans-Bold", "LiberationSans-Bold.ttf"),
    ("ReportMono", "LiberationMono-Regular.ttf"),
):
    pdfmetrics.registerFont(TTFont(face, str(STUDY / "fonts" / filename)))
pdfmetrics.registerFontFamily(
    "ReportSerif",
    normal="ReportSerif",
    bold="ReportSerif-Bold",
    italic="ReportSerif-Italic",
    boldItalic="ReportSerif-BoldItalic",
)
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
ORDER = (
    "unshielded_rl",
    "static_cbf",
    "timer_hard_unshielded",
    "uncertainty_wakeup",
    "barrier_replan_local",
    "original_proposal",
    "final_integrated",
    "matched_timer_hard",
    "high_rate_poll",
    "gated_poll",
    "immediate_benefit",
    "no_reserve",
    "no_blend",
    "speed_delay_margin",
    "critic_margin",
)
W = 516
BODY = ParagraphStyle("body", fontName="ReportSerif", fontSize=10, leading=13, spaceAfter=8)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.6, leading=11, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
HEAD = ParagraphStyle(
    "heading",
    fontName="ReportSerif-Bold",
    fontSize=14,
    leading=17,
    spaceBefore=4,
    spaceAfter=10,
    keepWithNext=True,
)
SUB = ParagraphStyle(
    "subheading",
    fontName="ReportSerif-Bold",
    fontSize=11,
    leading=13,
    spaceBefore=5,
    spaceAfter=6,
    keepWithNext=True,
)
TITLE = ParagraphStyle(
    "title",
    fontName="ReportSerif-Bold",
    fontSize=20,
    leading=23,
    spaceAfter=10,
    alignment=TA_CENTER,
)
CAPTION = ParagraphStyle("caption", parent=SMALL, fontSize=8.2, leading=10.5)


def clean(text):
    for a, b in (
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("\u2011", "-"),
        ("\u2265", ">="),
        ("\u00d7", "x"),
        ("\u2192", "->"),
    ):
        text = text.replace(a, b)
    return text


def para(text, style=BODY):
    return Paragraph(clean(text), style)


def table(headers, rows, widths, *, highlight=None):
    data = [[para(escape(str(v)), CELL) for v in headers]]
    data += [[para(escape(str(v)), CELL) for v in row] for row in rows]
    item = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e7ea")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8cbd0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if highlight is not None:
        commands.append(
            ("BACKGROUND", (0, highlight + 1), (-1, highlight + 1), colors.HexColor("#eef3f3"))
        )
    item.setStyle(TableStyle(commands))
    return item


def architecture():
    drawing = Drawing(W, 142)
    boxes = [
        ("RGB camera", "on request"),
        ("Slow model", "A / B / C"),
        ("PPO skill", "50 Hz action"),
        ("Blend + filters", "CBF / backup"),
        ("Physical vehicle", "MetaDrive"),
    ]
    for i, (title, desc) in enumerate(boxes):
        x = i * 104
        drawing.add(
            Rect(x, 56, 94, 46, strokeColor=colors.black, fillColor=colors.white, strokeWidth=0.7)
        )
        drawing.add(
            String(x + 47, 83, title, textAnchor="middle", fontName="ReportSans-Bold", fontSize=8)
        )
        drawing.add(
            String(x + 47, 68, desc, textAnchor="middle", fontName="ReportSans", fontSize=8)
        )
        if i < 4:
            drawing.add(Line(x + 94, 79, x + 103, 79, strokeWidth=0.7))
            drawing.add(Polygon([x + 103, 79, x + 99, 82, x + 99, 76], fillColor=colors.black))
    drawing.add(
        String(
            258,
            122,
            "An event scheduler decides when the slow model is asked",
            textAnchor="middle",
            fontName="ReportSans",
            fontSize=9,
        )
    )
    drawing.add(Line(463, 56, 463, 32, strokeWidth=0.6))
    drawing.add(Line(463, 32, 160, 32, strokeWidth=0.6))
    drawing.add(
        String(
            258,
            15,
            "Fresh simulator state feeds scheduling and fast safety checks",
            textAnchor="middle",
            fontName="ReportSans",
            fontSize=9,
        )
    )
    return drawing


def picture(path, height=None):
    from PIL import Image as PILImage

    with PILImage.open(path) as p:
        width, native_height = p.size
    h = W * native_height / width if height is None else height
    return Image(str(path), width=W, height=h)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#b0b0b0"))
    canvas.line(48, 39, 564, 39)
    canvas.setFont("ReportSans", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(48, 27, "Course project final evaluation | 12 November 2026")
    canvas.drawRightString(564, 27, str(doc.page))
    canvas.restoreState()


def build(draft=False):
    data = json.loads((RESULTS / "controlled.json").read_text())
    nominal, stress = data["nominal"], data["stress"]
    final = nominal["final_integrated"]
    actual_path = RESULTS / "actual-vlm.json"
    actual = json.loads(actual_path.read_text()) if actual_path.exists() else None
    if not draft and actual is None:
        raise ValueError("Wait for complete actual-model subset before final report")
    out = ROOT / "tmp/pdfs/integrated-draft.pdf" if draft else STUDY / "final-report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    story = []
    markdown = []

    def heading(text, sub=False):
        story.append(para(text, SUB if sub else HEAD))
        markdown.append(("### " if sub else "## ") + text + "\n")

    def p(text, small=False):
        story.append(para(text, SMALL if small else BODY))
        markdown.append(text.replace("<b>", "**").replace("</b>", "**") + "\n")

    def t(headers, rows, widths, highlight=None):
        story.append(table(headers, rows, widths, highlight=highlight))
        story.append(Spacer(1, 7))
        markdown.extend(
            [
                "| " + " | ".join(map(str, headers)) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
        )
        markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
        markdown.append("")

    def page():
        story.append(PageBreak())

    def fmt(r, field, decimals=2):
        return f"{r[field]:.{decimals}f}"

    calls = final["mean_expensive_calls"]
    saving = 100 * (1 - calls / nominal["high_rate_poll"]["mean_expensive_calls"])
    story.append(para("Safe Fast Driving While a Slow Camera Model Thinks", TITLE))
    story.append(
        para(
            "An integrated evaluation of PPO skills, event routing and runtime protection",
            ParagraphStyle("subtitle", parent=BODY, alignment=TA_CENTER, fontSize=11),
        )
    )
    story.append(
        para(
            "Course project final report - 12 November 2026",
            ParagraphStyle("date", parent=SMALL, alignment=TA_CENTER),
        )
    )
    markdown.append("# Safe Fast Driving While a Slow Camera Model Thinks\n")
    heading("Abstract", sub=True)
    p(
        f"We evaluate a closed-loop driving stack in which a slow camera model chooses a discrete maneuver, "
        f"frozen PPO skills execute it, and fast filters modify unsafe commands. The final comparison contains "
        f"1,440 physical MetaDrive episodes across fifteen policies, three scenario families, eight seed clusters "
        f"and four request delays. In the 0-1 s nominal stratum, the frozen integrated system completes "
        f"{final['success']}/72 episodes with {final['crashes']} collisions, using {calls:.2f} simulated costly requests "
        f"per episode versus {nominal['high_rate_poll']['mean_expensive_calls']:.2f} for 10 Hz polling ({saving:.1f}% fewer). "
        f"All protected replanning controls also complete every nominal episode. Simpler event triggers match or "
        f"improve the final scheduler; a cheap local rule requires no model calls. The contribution is a reproducible "
        f"engineering comparison and an analysis of delay, stopping room and transitions, not a new safety theorem "
        f"or superiority over published driving systems."
    )
    heading("1  Problem and course scope")
    p(
        "The proposal asks whether a vehicle can keep controlling safely while a slower planner thinks, avoid "
        "stopping indefinitely behind an obstacle, reduce unnecessary planning calls, and change maneuvers smoothly. "
        "It explicitly proposes combining established components under application/evaluation categories, not inventing "
        "a completely new algorithm [1]. We retain that scope. A correct late instruction can still fail if braking "
        "has left too little room for the installed forward-driving controller to turn."
    )
    story.append(architecture())
    p(
        "Figure 1. The model chooses a maneuver; it does not supply continuous acceleration. Fresh simulator state "
        "is available to scheduling and safety even when the model has not answered. This is a hybrid state/vision "
        "simulator stack, not a vision-only driving system.",
        small=True,
    )
    p(
        "The final variant revises the original margin hypothesis: it retains 8 m of optional stopping room while "
        "intent is unresolved, then releases that optional room after a fresh target-lane check accepts a changed "
        "target. Ordinary collision, braking and boundary checks remain active. The original uncertainty and "
        "speed-times-delay margins are tested explicitly rather than assumed necessary."
    )
    page()

    heading("2  One implementation and fifteen comparison policies")
    p(
        "Every policy uses the same 50 Hz DrivingStack, physical acceleration adapter and frozen PPO bank [2]. "
        "The two distinct learned policies regulate lane position and speed from ego state (y, vy, vx); they do "
        "not perceive traffic. RETURN_RIGHT reuses the cruise-right checkpoint. A CBF-style action correction, "
        "stopping-envelope backup, boundary check and fresh lane-admission check supply runtime protection."
    )
    rows = [
        ["Unprotected RL", "Cruise skill, no router", "Shield disabled", "None"],
        ["Static CBF", "Cruise skill, no router", "CBF + boundary", "No maneuver change"],
        ["1 Hz, unshielded", "Timer; same planner", "Shield disabled", "Hard switch"],
        ["Uncertainty wake-up", "Critic threshold only", "Full package", "8 m + blend"],
        ["Local barrier replan", "Sustained override; cheap rule", "Full package", "8 m + blend"],
        ["Original proposal", "Critic OR override", "Full + original inflation", "No 8 m; blend"],
        ["Final integrated", "Opportunity deadline", "Full, no added inflation", "8 m + blend"],
        ["1 Hz, matched safety", "Timer; same planner", "Same as final", "8 m; hard switch"],
        ["10 Hz polling", "Timer; same planner", "Same as final", "8 m + blend"],
        ["10 Hz, gated", "Timer + shared useful gate", "Same as final", "8 m + blend"],
        ["Immediate benefit", "Ask as soon as useful", "Same as final", "8 m + blend"],
        ["No extra reserve", "Same deadline", "Same as final", "No 8 m; blend"],
        ["No blending", "Same deadline", "Same as final", "8 m; hard switch"],
        ["Add delay margin", "Same deadline", "Final + pending margin", "8 m + blend"],
        ["Add critic margin", "Same deadline", "Final + critic margin", "8 m + blend"],
    ]
    t(
        ["Policy", "Decision timing", "Protection", "Transition"],
        rows,
        [119, 155, 128, 114],
        highlight=6,
    )
    p(
        "The uncertainty and barrier controls are local implementations inspired by the proposal [3,4], not "
        "ports of FASIONAD or CERT-Replan. We do not reproduce their training, visual fusion, calibrated risk "
        "or finite-horizon mode search. The local rule has zero artificial inference delay; its replans are "
        "reported separately from costly requests. The matched polling controls prevent a weak unprotected "
        "timer from being the only comparator.",
        small=True,
    )
    page()

    heading("3  Frozen experimental protocol")
    t(
        ["Component", "Frozen choice"],
        [
            [
                "Physical simulator",
                "MetaDrive/Bullet; pinned source fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1",
            ],
            [
                "Scenario families",
                "Parked lead; parked lead plus later moving cut-in; dense two-lane traffic",
            ],
            [
                "Worlds and matching",
                "Eight traffic seeds, three scene profiles; identical initial state within each comparison",
            ],
            ["Nominal delay stratum", "0, 0.125 and 1 s; 72 episodes per method"],
            ["CPU stress stratum", "18 s; 24 episodes per method; reported separately"],
            ["Episode budget", "200 m route, 60 simulated seconds, 50 Hz physical control"],
            [
                "Development",
                "36 threshold-selection episodes; 90 contract-pilot episodes; excluded from test",
            ],
            [
                "Actual model subset",
                "Six additional worlds; three queried and three no-model policies",
            ],
        ],
        [127, 389],
    )
    p(
        "The 8 m reserve was selected in the earlier disjoint maneuver development study. For this final test, "
        "critic-threshold scales 0.2, 0.5 and 1 were compared on 36 development episodes. All achieved 12/12 "
        "success and zero collisions; scale 1 used eight requests versus fifteen and twenty-nine, so it was frozen "
        "for the uncertainty-only control. No parameter was retuned on the held-out results."
    )
    p(
        "The controlled matrix uses one privileged-state lane/occupancy rule, not a VLM. It extends the original "
        "parked-car semantics by treating moving leads below 15 m/s as blockers. All synchronous responses, "
        "including same-skill answers, occupy the specified delay. Polling starts at a seed-randomized fraction "
        "of its period. The local barrier rule alone receives zero artificial compute delay."
    )
    p(
        f"Source, config, checkpoint and complete job-design hashes were frozen before the test. All "
        f"{data['audit']['scene_seed_groups']} scene/seed groups match initial states exactly; there are "
        f"{data['audit']['unique_initial_state_values']} distinct initial-state values because the parked-lead "
        f"and cut-in profiles share their start and differ in a later event. Eight seeds, not 1,440 episodes, "
        f"are the resampling clusters. Descriptive 95% paired intervals use 10,000 joint seed resamples; "
        f"they are not population safety or equivalence certificates."
    )
    heading("Metrics and exposure", sub=True)
    p(
        "Success requires no collision and at least 99% route progress. Exact route-complete end reasons, "
        "timeouts, terminal stalls and the legacy freeze flag are retained separately. A terminal stall means "
        "remaining route with final speed below 0.5 m/s and no collision. We also record stopped duration, "
        "requests, cheap local replans, rejection/expiry, crashes during pending calls, interventions and "
        "infeasible ticks. Achieved lateral jerk comes from changes in physical lateral acceleration, not "
        "requested commands. Failed drives remain in primary summaries; a jointly successful subset is "
        "reported only as a separately labelled comfort comparison."
    )
    p(
        "The cut-in activates only after x = 92 m. A car that stalls earlier never encounters it. Exposure "
        "and first observation during a pending call are therefore measured, not assumed. Route success "
        "does not require returning to the right lane; request savings can partly reflect remaining left "
        "while still making progress.",
        small=True,
    )
    page()

    heading("4  Integrated comparison under nominal delays")
    p(
        "Each row contains the same 72 nominal scene/seed/delay conditions. Costly requests are simulated "
        "delayed planning requests in this matrix; no actual model inference is hidden in these counts. "
        "The local baseline makes cheap replans instead."
    )
    rows = [
        [
            LABELS[m],
            f"{nominal[m]['success']}/72",
            nominal[m]["crashes"],
            nominal[m]["terminal_stalls"],
            f"{nominal[m]['mean_expensive_calls']:.2f}",
            f"{nominal[m]['mean_achieved_jerk_rms']:.2f}",
        ]
        for m in ORDER
    ]
    t(
        ["Policy", "Success", "Crash", "Stall", "Requests", "Jerk RMS"],
        rows,
        [174, 66, 52, 52, 83, 89],
        highlight=6,
    )
    story.append(picture(RESULTS / "nominal-outcomes.png", 212))
    p(
        "Figure 2. Counts retain crashes and timeouts. Extra protections differ in the deliberately weak "
        "unshielded/static arms, so their gap measures the whole package rather than one CBF term. "
        "Jerk units are m/s^3; early crashes have shorter comfort exposure.",
        small=True,
    )
    p(
        f"The final variant records {final['success']}/72 success and {final['crashes']} collisions. "
        f"The original proposal and every protected replanning control also finish 72/72. Thus safety/progress "
        f"are improved over the weak baselines but do not distinguish the stronger protected systems. "
        f"The local rule finishes all conditions without model-like calls; the visual model is not shown "
        f"necessary by a privileged-state benchmark.",
        small=True,
    )
    page()

    heading("5  Request cost, delay and component ablations")
    story.append(picture(RESULTS / "cost-and-delay.png", 175))
    p(
        f"Figure 3. Nominal and 18 s stress are separate strata. The final variant averages {calls:.2f} "
        f"requests versus {nominal['high_rate_poll']['mean_expensive_calls']:.2f} for 10 Hz polling. "
        f"This is {saving:.1f}% fewer with equal observed nominal success. Gating matches the request count; "
        f"uncertainty wake-up uses fewer requests ({nominal['uncertainty_wakeup']['mean_expensive_calls']:.2f}).",
        small=True,
    )
    rows = [
        [
            LABELS[m],
            f"{nominal[m]['success']}/72",
            f"{stress[m]['success']}/24",
            f"{nominal[m]['mean_achieved_jerk_rms']:.2f}",
            f"{nominal[m]['mean_achieved_jerk_peak']:.2f}",
            f"{stress[m]['mean_end_time']:.2f}",
        ]
        for m in (
            "final_integrated",
            "no_reserve",
            "no_blend",
            "speed_delay_margin",
            "critic_margin",
            "original_proposal",
        )
    ]
    t(
        ["Ablation", "Nom. success", "Stress success", "Nom. RMS", "Nom. peak", "Stress end s"],
        rows,
        [154, 76, 82, 68, 68, 68],
        highlight=0,
    )
    p(
        "The added reserve changes no nominal success count. Under 18 s delay, removing it drops success "
        "from 24/24 to 8/24. This is a progress result: stalled cars often never reach the cut-in. The original "
        "inflation variant also completes 24/24 stress episodes, so a reserve is one successful repair rather "
        "than the only possible solution."
    )
    comfort = data["joint_success_comfort"]["nominal"]
    rms = comfort["achieved_jerk_rms"]["final_minus_opponent"]
    peak = comfort["achieved_jerk_peak"]["final_minus_opponent"]
    p(
        f"On {comfort['pairs']} jointly successful nominal pairs, blending lowers mean achieved RMS by "
        f"{-rms:.2f} m/s^3 and mean peak by {-peak:.2f} m/s^3 versus no blending. This is an aggregate result, "
        f"not uniform improvement on every case. Crash/timeout exposure and per-seed differences remain in "
        f"the full data. Pending-margin necessity is unestablished when both matched variants avoid collision; "
        f"the stress comfort/progress tradeoff remains meaningful."
    )
    p(
        f"At 18 s, adding the pending margin lowers mean achieved RMS from "
        f"{stress['final_integrated']['mean_achieved_jerk_rms']:.2f} to "
        f"{stress['speed_delay_margin']['mean_achieved_jerk_rms']:.2f} m/s^3, while increasing mean end time "
        f"from {stress['final_integrated']['mean_end_time']:.2f} to "
        f"{stress['speed_delay_margin']['mean_end_time']:.2f} s; both finish 24/24. "
        f"This is a comfort/progress tradeoff, not evidence that the margin is useless.",
        small=True,
    )
    ci_rows = []
    for opponent, metric, label in (
        ("high_rate_poll", "expensive_calls", "Requests vs 10 Hz"),
        ("no_blend", "achieved_jerk_rms", "RMS jerk vs no blend"),
        ("no_blend", "achieved_jerk_peak", "Peak jerk vs no blend"),
    ):
        contrast = data["contrasts"]["nominal"][opponent][metric]
        lo, hi = contrast["descriptive_seed_bootstrap_95pct"]
        ci_rows.append([label, f"{contrast['final_minus_opponent']:.2f}", f"[{lo:.2f}, {hi:.2f}]"])
    t(
        ["Final minus comparator", "Paired mean difference", "Descriptive 95% interval"],
        ci_rows,
        [204, 148, 164],
    )
    p(
        "The immediate-benefit control matches final completion and request count and has lower mean RMS "
        "and peak jerk. The deadline term therefore has no demonstrated incremental advantage. We keep the "
        "preselected final variant in the table instead of silently switching to the held-out winner.",
        small=True,
    )
    page()

    heading("6  Actual trained camera model integration")
    p(
        "The actual subset uses the exported teammate BF16 LoRA with Qwen2.5-VL-3B-Instruct [7]. "
        "All 504 adapter tensors are verified against the saved file. The base is pinned to revision "
        "66285546d2b821cf421d4f5eb2576359d3770cd3; the original training did not pin its base revision. "
        "The unchanged v1 prompt and one-token A/B/C constrained inference are used. There are no "
        "scripted fallback replies. Scheduling and safety still observe simulator state; only the router "
        "model receives image, lane text and current-skill hint."
    )
    if actual is None:
        p(
            "<b>DRAFT: Actual-model subset remains in progress. No provisional live totals are reported.</b>"
        )
        t(
            ["Policy", "Planned worlds", "Source"],
            [
                [
                    LABELS[m],
                    6,
                    "Actual camera model"
                    if m in ("matched_timer_hard", "uncertainty_wakeup", "final_integrated")
                    else "No model",
                ]
                for m in (
                    "unshielded_rl",
                    "static_cbf",
                    "barrier_replan_local",
                    "matched_timer_hard",
                    "uncertainty_wakeup",
                    "final_integrated",
                )
            ],
            [230, 100, 186],
        )
    else:
        raw = [
            json.loads(s)
            for s in (ROOT / "event-cbf-driving/runs/final-integration/actual-vlm/episodes.jsonl")
            .read_text()
            .splitlines()
        ]
        rows = []
        for m in (
            "unshielded_rl",
            "static_cbf",
            "barrier_replan_local",
            "matched_timer_hard",
            "uncertainty_wakeup",
            "final_integrated",
        ):
            r = actual["summary"][m]
            predictions = sum(x["actual_model_predictions"] for x in raw if x["method"] == m)
            rows.append(
                [
                    LABELS[m],
                    f"{r['success']}/6",
                    r["crashes"],
                    r["terminal_stalls"],
                    f"{r['mean_route_pct']:.1f}%",
                    predictions,
                ]
            )
        t(
            ["Policy", "Success", "Crash", "Stall", "Mean route", "Predictions"],
            rows,
            [174, 66, 52, 52, 83, 89],
            highlight=5,
        )
        p(
            f"The completed subset contains 36 physical episodes, including 18 VLM-enabled runs over six "
            f"scene/seed worlds. {actual['episodes_with_actual_predictions']} episodes actually request predictions; "
            f"the model produces {actual['actual_predictions']} answers with {actual['fallbacks']} fallbacks. "
            f"{actual['model_episodes_with_admitted_target_change']} model-enabled episodes admit a model-proposed "
            f"change of lane target. "
            f"Initial states match exactly. Queried policy order rotates evenly. Median recorded prediction "
            f"time is {actual['prediction_latency_s']['median']:.2f} s, range "
            f"{actual['prediction_latency_s']['min']:.2f}-{actual['prediction_latency_s']['max']:.2f} s."
        )
    p(
        "A fixed 30 s scheduling estimate and 45 s pending deadline are used; actual measured inference "
        "plus capture time determines when an answer becomes ready. This estimate is not an imposed "
        "delay or a proven upper bound. The model stays loaded locally with three CPU threads and a "
        "10 GiB/no-swap limit. The controlled pool overlaps the initial live cases, so timings include "
        "contention and are not isolated inference-throughput measurements."
    )
    p(
        "The 50 Hz loop is measured in simulation time. CPU inference and rendering can make wall pacing "
        "fall behind; this study does not establish a hard real-time 50 Hz control guarantee.",
        small=True,
    )
    p(
        "The original prompt assumes all other cars are parked and says 20 m/s even when actual speed "
        "or traffic differs. The model was trained mostly on static lane/skill combinations; transitions "
        "can fall outside that support. Slow CPU response can also hide rapid premature reversals seen "
        "with faster inference. These runs cannot establish GPU closed-loop behavior or representative "
        "model accuracy. Zero-request episodes, rejected answers and in-flight replies returned after "
        "termination remain in the records."
    )
    p(
        "An inherited telemetry path omits capture time when joining an unfinished answer after driving "
        "has ended. Such answers are never applied. Raw records are preserved; capture and model wall "
        "times are separately available, and no end-to-end latency claim relies on that cleanup value.",
        small=True,
    )
    p(
        "Interpret this subset as integration evidence on six worlds. It is deliberately separate from "
        "the 1,440 controlled state-rule episodes and is too small for a broad safety or VLM-superiority claim."
    )
    if actual is not None:
        p(
            "All twelve admitted model target changes are cruise-right to overtake-left. No model-proposed "
            "return-right target is applied before termination; the final variant's two C replies arrive after "
            "the episode ends. Timer routing completes five of six worlds; its missed route includes a response "
            "above the fixed 45 s deadline. The timing difference prevents clean scheduler-causality claims. "
            "The local rule and uncertainty wake-up both complete six of six.",
            small=True,
        )
    page()

    heading("7  Proposal outcomes and interpretation")
    t(
        ["Proposal objective", "Measured status"],
        [
            [
                "Lower crash than unprotected RL and timer-hard",
                "Met on nominal observed counts; whole-package comparison",
            ],
            [
                "Lower freeze/stall than static CBF",
                "Met for terminal stalls; legacy freeze also retained",
            ],
            [
                "Fewer requests with matched progress",
                f"Met versus high-rate polling ({saving:.1f}% fewer); simpler triggers tie or improve",
            ],
            [
                "Switches not dominated in comfort",
                "Lower mean RMS and mean peak than matched no-blend; not every individual case",
            ],
            [
                "Survive pending calls with inflation on",
                "Zero observed pending collisions in 96 margin-on episodes; also zero with margin off",
            ],
            [
                "Added pending inflation is necessary",
                "Not established; zero-collision equality cannot demonstrate necessity",
            ],
            [
                "New algorithm or official published-system win",
                "Not claimed by proposal and not established by this study",
            ],
        ],
        [218, 298],
    )
    p(
        "The evidence supports the course contribution: implementing known pieces as a reproducible physical "
        "stack and measuring safety, delay, progress, calls and achieved motion. The system includes real "
        "learned policies and an actual fine-tuned camera router. It does not show that PPO beats a simple "
        "driver, or that the slow model is necessary when a cheap rule already receives privileged state."
    )
    p(
        "Several engineering choices matter jointly. A safety filter can prevent immediate collision but "
        "cannot invent a missing maneuver; stopping too close can also make a later correct command "
        "unusable for the installed controller. Extra stopping room must be available to spend during "
        "execution. This is controller-relative failure and recovery, not proof that no alternative "
        "steering trajectory could escape. Nonzero infeasible ticks remain in successful traces."
    )
    heading("Related work and claim boundary", sub=True)
    p(
        "PPO supplies the existing learned lane regulators [2]. FASIONAD motivates selective use of a "
        "slower reasoning pathway [3]. CERT-Replan motivates reacting to sustained planning risk, but "
        "our local trigger does not implement its calibrated horizon-level machinery [4]. Adaptive "
        "network-delay margins motivate the original margin variant without transferring a theorem "
        "to this high-level intent delay [5]. SHARP already studies planning with anticipated shielding "
        "effects [6]. Our claim is an application/evaluation result, with no priority or official-benchmark claim."
    )
    heading("Limits that change the interpretation", sub=True)
    p(
        "The roads are straight, the traffic distributions are narrow, and only one frozen PPO bank is "
        "evaluated. Scheduling and filters have ground-truth simulator object state; sensing errors are "
        "not propagated through an end-to-end perception system. Eight seed clusters and six actual-model "
        "worlds do not establish general driving safety. The route objective does not enforce right-lane "
        "return. All protection mechanisms are approximate sampled controllers, not a validated invariant "
        "set under uncertain vehicle dynamics. The final deadline is not the strongest simple trigger in "
        "this benchmark; the negative comparison is part of the result."
    )
    p(
        "The next practical configuration choice would favor the simplest method that meets the desired "
        "safety, progress and comfort constraints. That recommendation is not used to retune or relabel "
        "the held-out results in this report.",
        small=True,
    )
    page()

    heading("References")
    references = [
        "[1] Project team. Safe Fast Control While a Slow Language Model Thinks. Original course proposal, 18 October 2026. Included as proposal.pdf.",
        "[2] J. Schulman et al. Proximal Policy Optimization Algorithms. 2017. https://arxiv.org/abs/1707.06347",
        "[3] K. Qian et al. FASIONAD. 2024. https://arxiv.org/abs/2411.18013",
        "[4] R. R. Suganda and B. Hu. Adaptive Risk-Certified Event-Triggered Replanning for Dynamic Navigation. 2026. https://arxiv.org/abs/2610.09302",
        "[5] R. Periotto et al. MPC-CBF with Adaptive Safety Margins for Safety-critical Teleoperation over Imperfect Network Connections. 2024. https://arxiv.org/abs/2403.18650",
        "[6] H. Hu, K. Nakamura and J. F. Fisac. SHARP. 2022. https://arxiv.org/abs/2110.00843",
        "[7] Qwen team. Qwen2.5-VL-3B-Instruct model card. https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
    ]
    for ref in references:
        p(escape(ref), small=True)
    heading("Artifact and reproduction notes")
    p(
        "The release includes the frozen controller source, simulator configs, PPO checkpoints, trained "
        "adapter, full compressed action/plant traces, raw rows, source/checkpoint locks, selection record, "
        "all-condition tables, paired statistics and independent audits. Base weights and Python environments "
        "are external. The base revision is pinned; the original user checkout and previous studies are "
        "preserved. No cloud jobs or repository pushes are needed for this evaluation."
    )
    p(
        "From event-cbf-driving with the recorded project environment and restored results:",
        small=True,
    )
    for cmd in (
        "PYTHONPATH=. python scripts/integrated_eval.py test --workers 4",
        "PYTHONPATH=.:scripts python scripts/integrated_analysis.py --with-live --audit-traces",
        "PYTHONPATH=. pytest -q",
    ):
        story.append(
            para(
                escape(cmd),
                ParagraphStyle(
                    "code", fontName="ReportMono", fontSize=7.6, leading=11, spaceAfter=5
                ),
            )
        )
        markdown.append("`" + cmd + "`\n")
    p(
        "Complete frozen runs resume without overwriting, and changed source/design is rejected. To make "
        "new policy or scenario changes, use a fresh versioned output location rather than relabeling "
        "the held-out suite. Actual-model reruns use integrated_vlm.py with the local pinned base path; "
        "the report can be reanalyzed from the saved data without loading that base.",
        small=True,
    )
    p(
        "Observed environment versions, final test/lint/type results and archive-restoration checks are "
        "recorded in verification.json. The source lock for this main comparison is "
        + data["audit"]["source_lock_sha256"]
        + ".",
        small=True,
    )
    p(
        "Report status: "
        + (
            "DRAFT - actual-model subset pending."
            if draft
            else "Complete controlled and actual-model evaluation."
        ),
        small=True,
    )
    doc = SimpleDocTemplate(
        str(out),
        pagesize=(612, 792),
        rightMargin=48,
        leftMargin=48,
        topMargin=43,
        bottomMargin=51,
        title="Safe Fast Driving While a Slow Camera Model Thinks",
        author="Project team",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    if not draft:
        (STUDY / "final-report.md").write_text("\n".join(markdown))
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true")
    build(parser.parse_args().draft)
