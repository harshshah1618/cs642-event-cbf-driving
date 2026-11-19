"""Read-only assessment of existing study logs. No training, inference, or cloud use."""

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/safemetadrive/heldout-study-v1"
OUT = ROOT / "reports/safemetadrive-assessment"
OUT.mkdir(parents=True, exist_ok=True)
groups = json.loads((STUDY / "groups-local.json").read_text())
records = {}
for label, replicas in groups.items():
    rows = []
    for replica, paths in enumerate(replicas):
        for path in paths:
            report = json.loads(Path(path).read_text())
            for episode in report["episodes"]:
                calls = [
                    call for call in report.get("calls", []) if call["seed"] == episode["seed"]
                ]
                row = dict(replica=replica, **episode)
                if calls:
                    stopped_tail = []
                    for call in reversed(calls):
                        if call["speed_mps"] >= 0.1:
                            break
                        stopped_tail.append(call)
                    longest = run = 0
                    for call in calls:
                        run = run + 1 if call.get("response_status") == "rejected" else 0
                        longest = max(longest, run)
                    row.update(
                        request_labels=dict(Counter(c["letter"] for c in calls)),
                        last_requests=[
                            {
                                k: c.get(k)
                                for k in [
                                    "captured_t",
                                    "letter",
                                    "speed_mps",
                                    "current_lane",
                                    "target_lane",
                                    "response_status",
                                ]
                            }
                            for c in calls[-6:]
                        ],
                        low_speed_request_fraction=mean(c["speed_mps"] < 0.1 for c in calls),
                        terminal_low_speed_samples=len(stopped_tail),
                        terminal_low_speed_sample_span=(
                            calls[-1]["captured_t"] - stopped_tail[-1]["captured_t"]
                        )
                        if stopped_tail
                        else 0,
                        longest_consecutive_rejections=longest,
                        mean_top_choice_probability=mean(max(c["probabilities"]) for c in calls),
                    )
                rows.append(row)
    records[label] = rows
primary = records["Camera-delay0"]
failed = [r for r in primary if not r["success"]]
costly = [r for r in primary if r["cost"] > 0]
bad = [r for r in primary if not r["success"] or r["cost"] > 0]


def pair(a, b):
    left = {(r["replica"], r["seed"]): r for r in records[a]}
    right = {(r["replica"], r["seed"]): r for r in records[b]}
    assert left.keys() == right.keys()
    return dict(
        episodes=len(left),
        a=a,
        b=b,
        a_only_success=sum(left[k]["success"] and not right[k]["success"] for k in left),
        b_only_success=sum(right[k]["success"] and not left[k]["success"] for k in left),
        both_fail=sum(not left[k]["success"] and not right[k]["success"] for k in left),
        a_lower_cost=sum(left[k]["cost"] < right[k]["cost"] for k in left),
        b_lower_cost=sum(right[k]["cost"] < left[k]["cost"] for k in left),
        equal_cost=sum(right[k]["cost"] == left[k]["cost"] for k in left),
    )


def summary(rows):
    return dict(
        episodes=len(rows),
        successes=sum(r["success"] for r in rows),
        zero_cost_successes=sum(r["collision_free_success"] for r in rows),
        positive_cost_episodes=sum(r["cost"] > 0 for r in rows),
        total_cost=sum(r["cost"] for r in rows),
        timeouts=sum(r["truncated"] for r in rows),
        failed_violations=dict(
            Counter(v for r in rows if not r["success"] for v in r["violations"])
        ),
        mean_steps=mean(r["steps"] for r in rows),
        mean_interventions=mean(
            r["interventions"] for r in rows if r.get("interventions") is not None
        )
        if any(r.get("interventions") is not None for r in rows)
        else None,
        per_replica=[
            dict(
                replica=i,
                successes=sum(r["success"] for r in rows if r["replica"] == i),
                cost=sum(r["cost"] for r in rows if r["replica"] == i),
            )
            for i in sorted({r["replica"] for r in rows})
        ],
    )


result = dict(
    scope="Retrospective audit of frozen test outputs; diagnostic, not a new model-selection set.",
    summaries={label: summary(rows) for label, rows in records.items()},
    camera_failures=failed,
    camera_positive_cost=costly,
    bad_episode_count=len(bad),
    unique_failed_maps=sorted({r["seed"] for r in failed}),
    largest_cost_share=max(r["cost"] for r in primary) / sum(r["cost"] for r in primary),
    paired=[
        pair("Camera-delay0", label)
        for label in ["Fixed-normal", "Rule-PPO", "Camera-delay4", "Camera-unprotected"]
    ],
    sampling_note="Low-speed spans describe observations at query times, not continuous vehicle telemetry. "
    "Model choice probabilities are not calibrated correctness probabilities.",
)
(OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
with (OUT / "episode-audit.csv").open("w") as f:
    fields = [
        "replica",
        "seed",
        "success",
        "cost",
        "steps",
        "route_completion",
        "interventions",
        "terminal_low_speed_samples",
        "terminal_low_speed_sample_span",
        "longest_consecutive_rejections",
    ]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(bad)
print(
    json.dumps(
        {
            k: v
            for k, v in result.items()
            if k not in ["summaries", "camera_failures", "camera_positive_cost"]
        },
        indent=2,
    )
)
print("PRIMARY", json.dumps(result["summaries"]["Camera-delay0"]))
print("UNPROTECTED", json.dumps(result["summaries"]["Camera-unprotected"]))
print("FAILURES")
for r in failed:
    print(
        r["replica"],
        r["seed"],
        "stopped tail",
        r["terminal_low_speed_sample_span"],
        "seconds at",
        r["terminal_low_speed_samples"],
        "queries; max rejection run",
        r["longest_consecutive_rejections"],
    )
