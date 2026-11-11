"""Build the final measured report with the bundled reportlab Python runtime."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from statistics import mean

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research-study"


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return read(path)["episodes"]


def average(records, key):
    values = [r[key] for r in records if r.get(key) is not None]
    return mean(values) if values else 0.0


def stats(records):
    return [
        f"{average(records, 'route_pct'):.1f}",
        str(sum(r["crashed"] for r in records)),
        str(sum(r["freeze"] for r in records)),
        f"{average(records, 'router_calls'):.1f}",
    ]


def main():
    import reportlab

    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("StudySans", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("StudySans-Bold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily(
        "StudySans",
        normal="StudySans",
        bold="StudySans-Bold",
        italic="StudySans",
        boldItalic="StudySans-Bold",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows = rows("runs/metadrive-study/study-final/results.json")
    style_rows = rows("runs/metadrive-study/style-baselines/results.json")
    robust = rows("runs/robustness-final/summary.json")
    live = rows("runs/metadrive-local/live-qwen-final/results.json")
    matched = rows("runs/metadrive-local/live-qwen-final/matched-rule-summary.json")
    calibration = read("checkpoints/metadrive-calibrated/calibration.json")
    audit = read("outputs/research-study/audit.json")
    ablation = [
        r
        for r in rows("runs/safety-ablation/summary.json")
        if r["stage"] != "calibration_braking_intent"
    ]
    ablation += rows("runs/safety-ablation-final/summary.json")
    checks = (ROOT / "runs/research-final-checks.log").read_text()
    tests = int(re.findall(r"(\d+) passed", checks)[-1])
    assert len(main_rows) == 90 and len(style_rows) == 45 and len(robust) == 42
    assert len(live) == 18 and len(matched) == 18 and len(ablation) == 45
    assert "Success: no issues found" in checks and "All checks passed!" in checks
    calls = []
    for path in (ROOT / "runs/metadrive-local/live-qwen-final/episodes").glob("*.json"):
        calls += [c for c in json.loads(path.read_text())["calls"] if not c.get("suppressed")]
    delays = sorted(c["latency_s"] for c in calls if c.get("latency_s") is not None)
    p95 = delays[min(len(delays) - 1, int(0.95 * (len(delays) - 1)))]
    telemetry = read("runs/metadrive-local/live-qwen-final/results.json")["router_telemetry"]
    main_table = [["Mode", "n", "Route %", "Crashes", "Stalls", "Calls"]]
    for mode in [
        "unshielded_skill",
        "static_cbf",
        "fasionad_wakeup",
        "cert_replan",
        "fixed_rate_shielded",
        "always_on_shielded",
        "proposed",
        "proposed_no_margin",
        "proposed_no_blend",
    ]:
        rr = [r for r in main_rows + style_rows if r["mode"] == mode]
        main_table.append([mode, str(len(rr)), *stats(rr)])
    robust_table = [["Case", "n", "Route %", "Crashes", "Stalls", "Min route %"]]
    for case in sorted({r["case"] for r in robust}):
        rr = [r for r in robust if r["case"] == case]
        robust_table.append(
            [case, str(len(rr)), *stats(rr)[:3], f"{min(r['route_pct'] for r in rr):.1f}"]
        )
    ablation_table = [
        ["Stage (same 3 seeds)", "Route %", "Infeasible ticks", "Two-trap infeasible"]
    ]
    for stage in [
        "legacy",
        "calibration_only",
        "braking_only",
        "calibration_braking",
        "calibration_braking_intent",
    ]:
        rr = [r for r in ablation if r["stage"] == stage]
        tt = [r for r in rr if r["scenario"] == "two_trap"]
        ablation_table.append(
            [
                stage,
                f"{average(rr, 'route_pct'):.1f}",
                f"{average(rr, 'infeasible_ticks'):.1f}",
                f"{average(tt, 'infeasible_ticks'):.1f}",
            ]
        )
    calibration_table = [
        ["Skill", "Fit / validation ticks", "Old exceedance %", "New exceedance %"]
    ]
    for name, r in calibration["skills"].items():
        calibration_table.append(
            [
                name,
                f"{r['samples']} / {r['validation_samples']}",
                f"{100 * r['original_validation_exceedance']:.2f}",
                f"{100 * r['validation_exceedance']:.2f}",
            ]
        )
    live_table = [["Mode", "LLM route %", "Rule route %", "LLM / rule stalls", "LLM / rule calls"]]
    for mode in ["fixed_rate_shielded", "proposed", "proposed_no_margin"]:
        a = [r for r in live if r["mode"] == mode]
        b = [r for r in matched if r["mode"] == mode]
        live_table.append(
            [
                mode,
                f"{average(a, 'route_pct'):.1f}",
                f"{average(b, 'route_pct'):.1f}",
                f"{sum(r['freeze'] for r in a)} / {sum(r['freeze'] for r in b)}",
                f"{average(a, 'router_calls'):.1f} / {average(b, 'router_calls'):.1f}",
            ]
        )
    protocol = [
        ["Experiment", "Episodes / rollouts", "Seeds / condition"],
        ["Main comparison", 90, "0-4; 3 scenarios; 6 modes"],
        ["Additional style baselines", 45, "Same 5 seeds; 3 modes"],
        ["Safety ablation", 45, "0,1,2; 5 stages"],
        ["Robustness", 42, "301,307,311; 7 cases; margin on/off"],
        ["Actual local LLM", 18, "11,23; 3 scenarios; 3 modes"],
        ["Delay-replayed rule control", 18, "Same scenarios/seeds/modes as LLM"],
        ["Nominal calibration", 28, "16 fit + 12 held-out rollouts"],
    ]
    proposed = [r for r in main_rows if r["mode"] == "proposed"]
    summary = (
        f"The proposed stack averages {average(proposed, 'route_pct'):.1f}% route completion in the final nominal suite, "
        f"with {sum(r['crashed'] for r in proposed)} crashes and {sum(r['freeze'] for r in proposed)} stalls in 15 episodes. "
        "All 42 expanded robustness episodes avoid crashes and stalls; some time out short of the full route."
    )
    conclusion = (
        "The system is a reproducible engineering and measurement project. The safety additions improve tested behavior, "
        "but infeasible/negative barrier states remain. The local LLM offers no established advantage over the rule router. "
        "The pending margin has no demonstrated universal collision-prevention benefit; at multi-second model delays it can reduce progress."
    )

    def mdtable(table):
        return "\n".join(
            [
                "| " + " | ".join(map(str, r)) + " |"
                for r in [table[0], ["---"] * len(table[0]), *table[1:]]
            ]
        )

    md = [
        "# Final research report: Event-triggered skill routing in MetaDrive",
        "",
        summary,
        "",
        conclusion,
        "",
        "## Protocol",
        "",
        mdtable(protocol),
        "",
        "## Architecture and safety investigation",
        "",
        "The router selects a frozen lane skill. The skill emits physical accelerations and critic disagreement. The shield filters the action using fresh state, the intended maneuver corridor, a stopping-distance backup, and road-edge constraints. Accepted new maneuvers re-arm intervention monitoring; identical replies do not. Replies are validated against fresh lane gaps and deadlines.",
        "",
        "The original clearance target can become negative when critic or pending-delay margins jump. Fresh perception does not make a speed-times-model-delay margin a safety theorem. We retain that target and its infeasibility flags and separately log the stopping envelope. Assumptions: ego and lead braking magnitude 6 m/s^2, 2 m extra clearance, 0.15 s response allowance, current footprints, and 50 Hz sampled dynamics. Tracking error, moving cut-ins and initial unsafe states can still violate this model.",
        "",
        mdtable(ablation_table),
        "",
        "## Calibration",
        "",
        "Weights remain frozen; only metadata is recalibrated. Fit seeds: 101,103,107,109. Held-out validation: 211,223,227. Each skill is exercised in lane holding and acquisition with varied initial speed and lateral velocity. Threshold = empirical 99th percentile of nominal critic disagreement. Exceedance is not a collision probability; temporally correlated samples do not create thousands of independent trials.",
        "",
        mdtable(calibration_table),
        "",
        "## Main and style comparisons",
        "",
        mdtable(main_table),
        "",
        "Route and calls are episode means. Crashes/stalls are episode counts. Stalls mean speed below 0.5 m/s for at least one second while progress remains, even if the vehicle later recovers. Named style baselines are local approximations, not reproductions of the original papers. Requested-action jerk and achieved-motion jerk are reported separately in the detailed run report.",
        "",
        "## Robustness",
        "",
        mdtable(robust_table),
        "",
        "Traffic events change speed/lane targets; Bullet computes subsequent motion. No vehicle is teleported after spawn. Tight cut-ins deliberately challenge the stopping envelope. An earlier target-lane-braking test exposed a collision, then an intent-only version avoided the collision but stalled; re-arming after an accepted new intent fixed the tested recovery failure. The failing runs are retained under runs/robustness and runs/robustness-v2.",
        "",
        "## Actual LLM and delay-replayed rule comparison",
        "",
        mdtable(live_table),
        "",
        f"Qwen/Qwen2.5-1.5B-Instruct, revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, BF16 on CPU. {telemetry['router_live_calls']} actual model calls, {telemetry['router_fallbacks']} fallbacks, {telemetry['total_tokens']} generated/input tokens counted by the tokenizer. Observed request delay: {min(delays):.2f}-{max(delays):.2f} s; empirical p95 {p95:.2f} s. The live test uses real background inference and wall-clock pacing; configured margin horizon is 4.3 s and pending deadline is 6 s. Requests may expire; expired replies cannot switch skills.",
        "",
        "The matched rule control replays each episode's observed delay sequence by call ordinal, including deadline waits and backend-busy time. If the rule makes more calls, the final observed delay is reused. Different decisions change later states and call counts; this is a delay-controlled diagnostic, not an exact paired causal trial or a speed benchmark. Only two scenario seeds were used for the model comparison. In four simple diagnostic states, the model agreed with the heuristic label in two; labels are not a ground-truth safety benchmark.",
        "",
        "## Verification and reproduction",
        "",
        f"{tests} tests pass; Ruff and mypy pass; checkpoint verification and the original native CI regression gate pass. {audit['episodes']} fully recorded episodes and {audit['interaction_actions']:,} action-chain checks passed schema/consistency validation. {len(audit['replays'])} selected episodes replayed within 0.0001 m tolerance on this machine. Added CI configuration has not been executed remotely.",
        "",
        "```bash\nmake check\nmake calibrate\nmake study\nmake robustness\nmake study-live\npython scripts/match_live_delays.py runs/metadrive-local/RUN_ID\n```",
        "",
        "The live profile requires the local model extra (`pip install -e '.[local,metadrive]'`) and downloads the pinned public model. The configured OpenCode service was also probed but rejected the request because its free quota was exhausted; this is saved in runs/live-router-probe.json. No remote API credentials were required for the local study.",
        "",
        "## Deliverables and limits",
        "",
        "- [Three-layer demo](outputs/research-study/three-layer-demo.mp4): top-down reconstruction from actual logs; yaw approximated from velocity, not camera footage.\n- [Audit](outputs/research-study/audit.json).\n- [Nominal results](runs/metadrive-study/study-final/report.md).\n- [Live results](runs/metadrive-local/live-qwen-final/report.md).\n- [Calibration artifact](checkpoints/metadrive-calibrated/calibration.json).\n- [Setup and scope](METADRIVE.md).",
        "",
        "Validated scope: controlled straight roads, privileged state, two distinct transferred frozen PPO skills (RETURN_RIGHT aliases CRUISE_RIGHT), and bounded scenario traffic. No camera/VLM perception, curved-road validation, naturalistic traffic certification, real-vehicle validation, hard real-time guarantee, or end-to-end CBF proof is claimed. Zero sampled crashes are not evidence that all barriers remained feasible. This completes the proposal's systems-and-measurement objective, including negative results; it does not establish a new algorithm.",
        "",
        "References: [MetaDrive documentation](https://metadrive-simulator.readthedocs.io/en/latest/), [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), and the local proposal and recorded run artifacts.",
        "",
    ]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(md))
    (OUT / "final-report.md").write_text("\n".join(md))

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyStudy",
            fontName="StudySans",
            fontSize=10,
            leading=14,
            spaceAfter=8,
            textColor=colors.HexColor("#263748"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallStudy", fontName="StudySans", fontSize=8, leading=11, spaceAfter=5
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleStudy",
            fontName="StudySans-Bold",
            fontSize=25,
            leading=29,
            spaceAfter=14,
            textColor=colors.HexColor("#163e58"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingStudy",
            fontName="StudySans-Bold",
            fontSize=15,
            leading=19,
            spaceAfter=10,
            textColor=colors.HexColor("#163e58"),
        )
    )
    story = []

    def para(text, small=False):
        story.append(Paragraph(text, styles["SmallStudy" if small else "BodyStudy"]))

    def heading(text):
        story.append(Paragraph(text, styles["HeadingStudy"]))

    def table(data, widths=None):
        wrapped = [[Paragraph(escape(str(v)), styles["SmallStudy"]) for v in r] for r in data]
        t = Table(
            wrapped,
            colWidths=widths or [480 / len(data[0])] * len(data[0]),
            repeatRows=1,
            hAlign="LEFT",
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eef3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d3dc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))

    def page():
        story.append(PageBreak())

    story.append(Paragraph("Safe fast control while a slow model decides", styles["TitleStudy"]))
    para("CS642 course project | Final experimental report | MetaDrive study", small=True)
    heading("Outcome")
    para(summary)
    para(conclusion)
    heading("Recorded experimental protocol")
    table(protocol, [150, 95, 235])
    para(
        "The contribution is an instrumented composition of known techniques and controlled comparisons. All reported quantities come from saved runs. Failures and negative results are retained."
    )
    page()
    heading("1. Architecture and safety investigation")
    table(
        [
            ["Router", "Frozen skills", "Safety layer"],
            [
                "Selects one maneuver; runs on events",
                "PPO requests ax/ay and provides critic disagreement",
                "Checks fresh geometry, intended corridor, stopping envelope and road edges",
            ],
        ]
    )
    para(
        "Feedback runs both ways: disagreement or shield intervention requests a new maneuver; accepted intent changes which corridor the shield checks. A new accepted intent re-arms intervention monitoring. Identical replies do not, and rate limits plus blending still gate calls."
    )
    para(
        "The original headway target subtracts critic and pending-call margins. Turning on a large margin can instantly move the current state outside that target. Fresh local state is available throughout; model delay alone is not a proof that a speed-times-delay clearance is required or achievable."
    )
    para(
        "We preserve the original target h and feasibility flags. A separate sampled stopping-envelope backup checks full footprints, relative stopping distance, 2 m clearance, and 0.15 s response allowance, assuming both vehicles can brake at 6 m/s squared. A one-step prediction bounds acceleration by bisection. Tracking error and unanticipated lane entry can still violate these assumptions."
    )
    heading("Paired safety ablation")
    table(ablation_table, [195, 65, 105, 115])
    para(
        "Infeasible ticks are episode means, not crash counts. The final stage includes intended-corridor checking and re-arming on accepted intent changes. Negative target values remain visible; the new physical envelope is not used to relabel them as safe."
    )
    page()
    heading("2. Frozen-policy calibration and nominal comparisons")
    para(
        "Policy weights stay unchanged. Fit seeds 101,103,107,109 and validation seeds 211,223,227 are disjoint. Each skill is exercised in lane holding and acquisition with varied initial speed and lateral velocity. The threshold is the empirical 99th nominal-exposure percentile; critic disagreement is not a collision probability."
    )
    table(calibration_table, [135, 120, 112, 113])
    para(
        "Old/new exceedances use each base threshold. Temporally correlated physics ticks are not independent trials. RETURN_RIGHT remains an explicit CRUISE_RIGHT alias.",
        small=True,
    )
    table(main_table, [170, 35, 65, 65, 65, 80])
    para(
        "Route and calls are episode means; crashes and stalls are counts. A stall means speed below 0.5 m/s for one second with progress left, even if the vehicle later recovers. Named style baselines are local approximations, not paper reproductions. Shielded polling shares the proposed stack's safeguards and blending; the routing schedule changes."
    )
    page()
    heading("3. Held-out traffic events and variable delay")
    para(
        "Seven cases use seeds 301,307,311 and margin on/off. Traffic events change vehicle speed or lane targets; Bullet computes motion without teleporting vehicles. Jitter cases use 0.8 s nominal router delay with uniform +/-0.3 s jitter. Other event cases use 0.3 +/-0.2 s."
    )
    table(robust_table, [155, 35, 75, 70, 70, 75])
    para(
        "All 42 final episodes avoid crashes and stalls. Tight cut-ins deliberately challenge the stopping envelope; a shortfall in route completion is retained rather than hidden by a lower pass threshold."
    )
    heading("Failure found and corrected")
    para(
        "A target-lane braking event exposed a collision after the router had accepted a maneuver. Checking the intended corridor removed that collision but could leave the vehicle stopped behind the new blocker. The old once-per-intervention latch never re-armed because the shield had not cleared. Re-arming for a newly accepted intent lets the router choose an escape before the vehicle stops. The tested six reruns recover; a physical regression test protects the previously failing seed."
    )
    para(
        "Historical failing runs remain available under runs/robustness and runs/robustness-v2. These are controlled engineering cases, not a claim of naturalistic traffic coverage."
    )
    page()
    heading("4. Real local model and delay-replayed rule control")
    para(
        "The live backend is Qwen/Qwen2.5-1.5B-Instruct, pinned to revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, BF16 on CPU. A single background worker performs actual inference while control continues. Expired or invalid answers cannot switch skills."
    )
    para(
        f"The final live suite contains {telemetry['router_live_calls']} actual model calls, {telemetry['router_fallbacks']} fallbacks and {telemetry['total_tokens']:,} tokenizer-counted input/output tokens. Observed request delay is {min(delays):.2f}-{max(delays):.2f} s (empirical p95 {p95:.2f} s). The margin horizon is 4.3 s and pending deadline is 6 s. These workstation measurements are not an optimized speed benchmark."
    )
    table(live_table, [145, 80, 80, 90, 85])
    para(
        "The rule control replays the model's delay sequence by call ordinal, including pending deadlines and backend-busy time. Excess rule calls reuse the final observed delay. Decisions change later states and call counts, so this is a delay-controlled diagnostic rather than an exact paired causal trial. Only two seeds are used for the live comparison."
    )
    para(
        "This model has no established advantage over the rule router. In four simple diagnostic states it agreed with the heuristic label in two; these labels are not ground-truth safety labels. The pending margin does not show a universal collision-prevention benefit and can cost progress at multi-second delays. One live no-margin episode stalled; that small sample does not establish a general stall-prevention effect."
    )
    para(
        "The configured OpenCode endpoint was also probed, but rejected the request because its free quota was exhausted. The completed comparison uses the local model and no remote API credentials.",
        small=True,
    )
    page()
    heading("5. Demonstration, verification and reproducibility")
    frame = sorted(
        OUT.glob("three-layer-demo-frame-*.png"), key=lambda p: int(p.stem.split("-")[-1])
    )
    if frame:
        story.append(Image(str(frame[len(frame) // 2]), width=480, height=300))
        story.append(Spacer(1, 8))
    para(
        "three-layer-demo.mp4 is a top-down reconstruction of recorded states and layer logs. Vehicle yaw is approximated from velocity; this is not camera footage.",
        small=True,
    )
    para(
        f"{tests} tests pass, plus Ruff, mypy, checkpoint verification and the native CI gate. {audit['episodes']} fully recorded episodes pass schema validation; {audit['interaction_actions']:,} final-action chains agree. {len(audit['replays'])} selected episodes replay within 0.0001 m tolerance on this machine. Remote CI has not been executed by this task."
    )
    para(
        "Reproduce: make check; make calibrate; make study; make robustness; make study-live. Install the local and metadrive extras first. Use scripts/match_live_delays.py with the live run directory for the delay-replayed rule comparison.",
        small=True,
    )
    para(
        "Scope: controlled straight roads and privileged state. No camera/VLM perception, curved-road validation, real-vehicle validation, hard real-time guarantee or end-to-end safety proof is claimed. Negative barriers and infeasible ticks remain in the detailed reports. The results support the proposal's systems-and-measurement objective, including negative findings.",
        small=True,
    )
    para(
        'Primary references: <link href="https://metadrive-simulator.readthedocs.io/en/latest/">MetaDrive documentation</link>; <link href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct">Qwen model card</link>; project proposal and recorded run manifests.',
        small=True,
    )

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#c9d3dc"))
        canvas.line(48, 39, 564, 39)
        canvas.setFont("StudySans", 8)
        canvas.setFillColor(colors.HexColor("#627384"))
        canvas.drawString(
            48, 26, "CS642 | Event-triggered driving | Recorded results, restricted research scope"
        )
        canvas.drawRightString(564, 26, str(doc.page))

    doc = SimpleDocTemplate(
        str(OUT / "final-report.pdf"),
        pagesize=(612, 792),
        rightMargin=66,
        leftMargin=66,
        topMargin=48,
        bottomMargin=52,
        title="Event-triggered driving: final experimental report",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT / "final-report.pdf")


if __name__ == "__main__":
    main()
