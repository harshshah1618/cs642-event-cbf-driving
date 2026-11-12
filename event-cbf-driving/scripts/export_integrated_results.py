"""Export the completed final comparison for human review and CSV analysis."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "final-study"
RUNS = ROOT / "event-cbf-driving/runs/final-integration"


def main():
    controlled = json.loads((STUDY / "results/controlled.json").read_text())
    actual = json.loads((STUDY / "results/actual-vlm.json").read_text())
    rows = [json.loads(s) for s in (RUNS / "test/episodes.jsonl").read_text().splitlines()]
    live = [json.loads(s) for s in (RUNS / "actual-vlm/episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 1440 and len(live) == 36
    fields = (
        "scene",
        "seed",
        "method",
        "success",
        "crashed",
        "terminal_stall",
        "freeze",
        "route_pct",
        "end_reason",
        "end_time",
        "expensive_calls",
        "local_replans",
        "calls_applied",
        "calls_rejected",
        "achieved_jerk_rms",
        "achieved_jerk_peak",
        "stopped_s",
        "pending_survival",
        "cut_in_observed",
        "cut_in_first_seen_while_pending",
        "braking_infeasible_ticks",
        "wall_s",
    )
    extra = (
        "stage",
        "controlled_delay_s",
        "live_scheduling_estimate_s",
        "actual_model_predictions",
        "id",
    )
    with (STUDY / "results/per-episode.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=extra + fields, lineterminator="\n")
        writer.writeheader()
        for stage, group in (("controlled", rows), ("actual_subset", live)):
            for r in group:
                value = {k: r[k] for k in fields}
                value.update(
                    stage=stage,
                    controlled_delay_s=r["delay"] if stage == "controlled" else "",
                    live_scheduling_estimate_s=r["delay"]
                    if stage == "actual_subset" and r["actual_model"]
                    else "",
                    actual_model_predictions=r.get("actual_model_predictions", 0),
                    id=r["id"],
                )
                writer.writerow(value)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scene"], r["delay"], r["method"])].append(r)
    data = []
    for (scene, delay, method), rs in sorted(groups.items()):
        data.append(
            dict(
                scene=scene,
                delay_s=delay,
                method=method,
                n=len(rs),
                successes=sum(r["success"] for r in rs),
                crashes=sum(r["crashed"] for r in rs),
                terminal_stalls=sum(r["terminal_stall"] for r in rs),
                mean_route_pct=sum(r["route_pct"] for r in rs) / len(rs),
                mean_requests=sum(r["expensive_calls"] for r in rs) / len(rs),
                mean_local_replans=sum(r["local_replans"] for r in rs) / len(rs),
                mean_achieved_jerk_rms=sum(r["achieved_jerk_rms"] for r in rs) / len(rs),
                cut_ins_observed=sum(r["cut_in_observed"] for r in rs),
            )
        )
    with (STUDY / "results/all-conditions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    text = [
        "# Every controlled scene and delay condition",
        "",
        "Each row contains eight matched traffic seeds.18s is stress, not pooled into nominal headlines.",
        "",
    ]
    for scene, delay in sorted({(r["scene"], r["delay_s"]) for r in data}):
        text += [
            f"## {scene} - {delay:g}s",
            "",
            "| Method | Success /8 | Crash | Terminal stall | Mean route % | Requests | Local replans | Jerk RMS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in [v for v in data if v["scene"] == scene and v["delay_s"] == delay]:
            text.append(
                f"| {r['method']} | {r['successes']} | {r['crashes']} | {r['terminal_stalls']} | "
                f"{r['mean_route_pct']:.2f} | {r['mean_requests']:.3f} | {r['mean_local_replans']:.3f} | {r['mean_achieved_jerk_rms']:.3f} |"
            )
        text.append("")
    (STUDY / "results/all-conditions.md").write_text("\n".join(text))
    nominal = controlled["nominal"]
    final = nominal["final_integrated"]
    saving = 100 * (
        1 - final["mean_expensive_calls"] / nominal["high_rate_poll"]["mean_expensive_calls"]
    )
    (STUDY / "EXPLAINED.md").write_text(f"""# What the final comparison found

We completed the promised comparison and report:1,440 controlled physical simulator drives, then36 integration drives with the actual trained model enabled in18of them. No reply was substituted for the trained model. The original source and previous studies remain preserved.

The integrated version finished72/72 nominal conditions with no crashes. Unprotected RL finished0/72; static CBF finished24/72; unshielded timer routing finished14/72. The full protection and replanning package clearly helps compared with those simpler systems.

It made{final["mean_expensive_calls"]:.2f} delayed requests per nominal drive versus{nominal["high_rate_poll"]["mean_expensive_calls"]:.2f} for10Hz polling: {saving:.1f}% fewer, with equal observed success. These controlled requests use a state rule, not actual VLM inference. Simple gated/benefit triggers match that count, and uncertainty wake-up uses fewer. The fancy deadline is not the strongest method here.

Blending reduced mean physical jerk RMS from5.07to3.97and mean peak from93.25to75.74m/s^3 compared with the matched no-blend version. Individual cases can get worse; this is an average result.

With18seconds of decision delay, extra stopping room raised completion from8/24to24/24. The original inflation version also finished24/24. Adding the pending margin reduced stress jerk RMS from7.28to2.31but made completion about4seconds slower. The margin-on arms had zero observed pending collisions; their necessity is unproved because protected margin-off arms were also safe.

In the actual camera-model subset, integrated, uncertainty-triggered and cheap local replanning each finished6/6worlds; matched timer routing finished5/6. All four had zero crashes. The timer miss included a response longer than45seconds, so different CPU timing prevents a clean scheduler comparison.

There were {actual["actual_predictions"]} real predictions across {actual["episodes_with_actual_predictions"]} queried episodes, zero fallbacks, and 12 applied model lane-target changes. All twelve were overtakes; no model return-right target was applied before termination. The model is genuinely connected, but this small test does not show it beats a cheap rule that already knows traffic positions.

This meets the original course scope: combine existing robotics and machine-learning pieces, compare them fairly, and report what helps. It does not establish a new algorithm, official benchmark superiority, or real-vehicle safety.
""")
    (STUDY / "README.md").write_text("""# Final integrated comparison

Start with final-report.pdf (seven body pages plus references) or EXPLAINED.md. This is the single proposal-aligned evaluation, not a merger of old favorable numbers.

Source branch:codex/final-integrated-evaluation, based on6090300. Original harsh-testing source is preserved. Source locks, full physical traces, raw response records and images belong to this evaluation. No official FASIONAD/CERT port or safety theorem is claimed.

## Files

- final-report.pdf and final-report.md: measured final report.
- PROTOCOL.md, selection.json and proposal.pdf: design, development choice and original scope.
- results/controlled.json: all15methods, nominal/stress separation, paired contrasts and eight-cluster descriptive intervals.
- results/actual-vlm.json:36integration episodes,18model-enabled,15actually queried and22real predictions.
- results/per-episode.csv and all-conditions.csv/md: reviewable tables.
- results/*trace-audit.json and audits/: action/plant integrity and independent reviews.
- verification.json and environment.json: checks, versions and preservation evidence.

## Reproduction

The release ZIP restores the event-cbf-driving/runs/final-integration paths. Those raw runs are intentionally outside Git; obtain the complete release, not only a source clone. Base weights (7.51GB) and Python environments are external; trained adapter and PPO checkpoints are included. Analysis needs no model loading. Embedded report fonts and their license are included.

From event-cbf-driving in the recorded project environment:

```bash
PYTHONPATH=. python scripts/integrated_eval.py test --workers 4
PYTHONPATH=.:scripts python scripts/integrated_analysis.py --with-live --audit-traces
PYTHONPATH=. pytest -q
```

Completed frozen runs are hash-checked and resumed without overwriting. Changing source/design requires a separately versioned directory. The first command does not repeat already complete drives. For fresh actual-model runs, use scripts/integrated_vlm.py --base PINNED_BASE_PATH --output NEW_DIRECTORY, with task-local PEFT0.17.1, P3D_DISPLAY=p3headlessgl and LP_NUM_THREADS=1. CPU model runs need roughly10GB and tens of minutes;45s is an admission deadline, not cancellation of a native Torch forward. The full user service also had an external90minute runtime cap. Actual timing varied with concurrent CPU work.

Build the PDF using a Python environment with ReportLab4.4.9 and Pillow:

```bash
python scripts/build_integrated_report.py
```

The complete release is delivered under the original checkout's research-delivery/final-integrated directory. No repository push, merge, external message, or cloud job was performed.
""")
    # Improve prose spacing while preserving numeric values and code identifiers.
    for name in ("EXPLAINED.md", "README.md"):
        p = STUDY / name
        s = p.read_text()
        for a, b in [
            ("report:1", "report: 1"),
            ("then36", "then 36"),
            ("in18of", "in 18 of"),
            ("finished72", "finished 72"),
            ("finished0", "finished 0"),
            ("finished24", "finished 24"),
            ("finished14", "finished 14"),
            ("made1", "made 1"),
            ("versus75", "versus 75"),
            ("for10Hz", "for 10 Hz"),
            ("from5.07to3.97and", "from 5.07 to 3.97 and"),
            ("from93.25to75.74m", "from 93.25 to 75.74 m"),
            ("With18seconds", "With 18 seconds"),
            ("from8/24to24/24", "from 8/24 to 24/24"),
            ("RMS from7.28to2.31but", "RMS from 7.28 to 2.31 but"),
            ("about4seconds", "about 4 seconds"),
            ("finished6/6worlds", "finished 6/6 worlds"),
            ("finished5/6", "finished 5/6"),
            ("than45seconds", "than 45 seconds"),
            ("There were22", "There were 22"),
            ("across15queried", "across 15 queried"),
            ("and12applied", "and 12 applied"),
            ("branch:codex", "branch: codex"),
            ("on6090300", "on 6090300"),
            ("all15methods", "all 15 methods"),
            ("36integration", "36 integration"),
            ("18model-enabled", "18 model-enabled"),
            ("15actually", "15 actually"),
            ("22real", "22 real"),
            ("PEFT0.17.1", "PEFT 0.17.1"),
            ("roughly10GB", "roughly 10 GB"),
            ("ReportLab4.4.9", "ReportLab 4.4.9"),
            ("external90minute", "external 90-minute"),
        ]:
            s = s.replace(a, b)
        p.write_text(s)
    print("Exported 1476 episode rows and180 controlled condition rows; report guides written")


if __name__ == "__main__":
    main()
