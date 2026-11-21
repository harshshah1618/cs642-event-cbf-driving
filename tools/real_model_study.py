"""Real-camera-model study. The router is the trained VLM on every call.

No scripted router drives. A geometric reading of the trained prompt is scored
offline from positions and is never sent back as a skill.

Two state lines, same system text:
  pinned   — models/camera/prompts.py, always "Ego speed: 20 m/s"
  measured — models/camera/prompts_measured_speed.py, measured ego speed

Stack is the selected final integrated stack (deadline trigger, shield, blend).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from driving import DT
from driving.assets import asset_directory
from driving.config import ScenarioConfig
from driving.controller import DrivingStack
from driving.env_registry import make_env
from driving.errors import RouterError
from driving.evaluation import load_skill_bank
from driving.maneuver_reserve import ManeuverReserveShield

from integrated_eval import Job, configs, scene_spec, trace_metrics
from trained_vlm_cpu import TrainedVLMCPU, load_prompts
from trained_vlm_drive import ActualCPURouter

OUT = ROOT / "runs/real-model-study"
BASE = Path(
    os.environ.get("QWEN_BASE_MODEL_PATH")
    or json.loads((ROOT / "models/camera/base-model.json").read_text())["snapshot_path"]
)
SIGMA = json.loads((ROOT / "configs/study-selection.json").read_text())["sigma_scale"]
BUDGET = 6
DELAY_S = 30.0

# Same seeds for both prompt conditions. Layouts are fixed before any episode.
CELLS = {
    "stall": [9101, 9102, 9103, 9104],
    "cut_in": [9201, 9202, 9203, 9204],
    "pass_return": [9301, 9302, 9303, 9304],
    "dense": [9401, 9402],
}
PROMPTS = ("pinned", "measured")
PROMPT_PATHS = {
    "pinned": ROOT / "models/camera/prompts.py",
    "measured": ROOT / "models/camera/prompts_measured_speed.py",
}


def prompt_rule_letter(ego, others):
    """Offline reading of the trained prompt. Not a controller.

    Windows match the prompt's own words as operationalized in PromptRule:
    a right-lane blocker is a vehicle ahead within 100 m slower than 15 m/s;
    the left lane is occupied by a vehicle within 20 m.
    """
    right_ahead = any(
        0 < other.x - ego.x < 100 and abs(other.y) < 1.85 and other.vx < 15 for other in others
    )
    left_occupied = any(abs(other.x - ego.x) < 20 and abs(other.y - 4) < 1.85 for other in others)
    if ego.y >= 2:
        return "B" if right_ahead else "C"
    return "B" if right_ahead and not left_occupied else "A"


class StudyRouter(ActualCPURouter):
    def __init__(self, policy, output, budget):
        super().__init__(policy, output)
        self.budget = budget

    def submit(self, req):
        if len(self.predictions) >= self.budget:
            raise RouterError(f"study prediction budget exhausted ({self.budget})")
        response = super().submit(req)
        ego = req.snapshot.ego
        rule = prompt_rule_letter(ego, req.snapshot.others)
        self.predictions[-1]["prompt_rule_letter"] = rule
        self.predictions[-1]["agrees_with_prompt_rule"] = self.predictions[-1]["letter"] == rule
        self.predictions[-1]["ego"] = {"x": ego.x, "y": ego.y, "vx": ego.vx}
        self.predictions[-1]["others"] = [
            {"x": other.x, "y": other.y, "vx": other.vx} for other in req.snapshot.others
        ]
        return response


def design():
    rows = []
    for cell, seeds in CELLS.items():
        for seed in seeds:
            for prompt in PROMPTS:
                rows.append({"cell": cell, "seed": seed, "prompt": prompt})
    return rows


def protocol():
    return {
        "kind": "real_trained_camera_model_only",
        "question": (
            "With the trained camera model as the only router, how often does the "
            "final stack finish, crash, or stall, and does reporting measured speed "
            "instead of a constant 20 m/s change the letter or the outcome?"
        ),
        "router": "Qwen2.5-VL-3B-Instruct + teammate LoRA, CPU, no fallback",
        "stack": "final_integrated, deadline trigger, live measured latency, blend, skill-switch guard, 8 m reserve",
        "scripted_router": False,
        "system_prompt": "unchanged from the training prompt",
        "prompt_factor": {
            "pinned": "state line always says Ego speed: 20 m/s",
            "measured": "identical system text; state line uses the measured ego speed",
        },
        "cells": {
            "stall": "study stall_ahead generator, one slowed/stopped lead, open left",
            "cut_in": "study cut_in generator, lead plus a vehicle that appears during the wait",
            "pass_return": "parked car in the right lane, then a parked car in the left lane; finishing requires a return",
            "dense": "study dense_highway generator",
        },
        "design": design(),
        "seeds_fixed_before_running": True,
        "prediction_budget": BUDGET,
        "trigger_latency_bound_s": DELAY_S,
        "max_pending_s": 90.0,
        "sigma_scale": SIGMA,
        "success": "not crashed and route_pct >= 99",
        "hypotheses": [
            "Stall, either prompt: no crashes, and a majority finish.",
            "Cut-in at the measured answer delay: report crashes separately from stalls. Do not assume zero.",
            "Pass-return: a finish is a completed return; a stall is a failure to leave the blocked left lane.",
            "On calls made while slower than 5 m/s, the measured speed line can change the letter. Completion is not assumed to rise.",
        ],
    }


def make_spec(cell, seed):
    if cell in ("stall", "cut_in", "dense"):
        scene = {"stall": "stall_ahead", "cut_in": "cut_in", "dense": "dense"}[cell]
        job = Job(scene, seed, DELAY_S, "final_integrated", SIGMA)
        return job, scene_spec(job, camera=True), None
    import numpy as np

    rng = np.random.default_rng(seed + 4242)
    right_x = float(rng.uniform(75, 95))
    left_x = float(rng.uniform(230, 270))
    job = Job("stall_ahead", seed, DELAY_S, "final_integrated", SIGMA)
    spec = ScenarioConfig(
        "two_trap",
        simulator="metadrive",
        randomize=False,
        duration=75,
        route_length=360,
        params={
            "rgb_camera": True,
            "cut_in_trigger_x": 10000.0,
            "stalled_x": 5000.0,
            "spawn_x": 0.0,
            "ego_vx": 20.0,
            "initial_traffic": [
                {"x": right_x, "y": 0.0, "vx": 0.0},
                {"x": left_x, "y": 4.0, "vx": 0.0},
            ],
        },
    )
    return job, spec, {"right_x": right_x, "left_x": left_x}


def slim_call(call):
    keys = ("t", "from", "to", "applied", "rejected", "error", "reason", "ready_at")
    return {key: call.get(key) for key in keys}


def slim_prediction(row):
    keep = (
        "letter",
        "probs",
        "latency_s",
        "model_forward_s",
        "request_t",
        "current_skill",
        "ego_lane",
        "prompt_rule_letter",
        "agrees_with_prompt_rule",
        "ego",
        "others",
        "state_text",
        "prompt_mode",
    )
    return {key: row.get(key) for key in keep}


def run_case(case, policy, bank):
    job, spec, layout = make_spec(case["cell"], case["seed"])
    cfg, reserve = configs(job, live=True)
    cfg.stack.max_pending_s = 90.0
    cfg.validate()
    folder = OUT / "episodes" / f"{case['cell']}-{case['seed']}-{case['prompt']}"
    folder.mkdir(parents=True, exist_ok=True)
    policy.prompts = load_prompts(PROMPT_PATHS[case["prompt"]])
    router = StudyRouter(policy, folder, BUDGET)
    shield = ManeuverReserveShield(cfg.shield, mode=reserve, reserve_m=8.0)
    stack = DrivingStack(bank, router, shield=shield, config=cfg.stack)
    env = make_env(spec.name, spec)
    started = time.perf_counter()
    try:
        result = stack.run(env, seed=case["seed"])
    finally:
        env.close()
    wall = time.perf_counter() - started
    metrics = trace_metrics(result)
    counts = Counter(result.skills)
    row = {
        **case,
        "route_pct": result.route_pct,
        "crashed": result.crashed,
        "crash_time": result.crash_time,
        "freeze": result.freeze,
        "success": (not result.crashed) and result.route_pct >= 99,
        "terminal_stall": metrics["terminal_stall"],
        "end_time": result.end_time,
        "end_reason": result.end_reason,
        "x": result.x,
        "y": result.y,
        "vx": result.vx,
        "avg_speed": result.avg_speed,
        "wall_s": wall,
        "real_predictions": len(router.predictions),
        "calls_applied": result.calls_applied,
        "calls_rejected": result.calls_rejected,
        "fallbacks": result.router_fallbacks,
        "stopped_s": metrics["stopped_s"],
        "override_fraction": result.override_fraction,
        "braking_infeasible_ticks": result.braking_infeasible_ticks,
        "min_braking_h": result.min_braking_h,
        "skill_seconds": {name: ticks * DT for name, ticks in counts.items()},
        "ended_in_right_lane": result.y < 2.0,
        "calls": [slim_call(call) for call in result.calls],
        "predictions": [slim_prediction(item) for item in router.predictions],
        "layout": layout,
    }
    return row


def wilson(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(rows):
    report = {}
    for cell in CELLS:
        report[cell] = {}
        for prompt in PROMPTS:
            group = [row for row in rows if row["cell"] == cell and row["prompt"] == prompt and "error" not in row]
            n = len(group)
            finished = sum(bool(row["success"]) for row in group)
            crashes = sum(bool(row["crashed"]) for row in group)
            stalls = sum(bool(row["terminal_stall"]) for row in group)
            preds = [p for row in group for p in row.get("predictions", [])]
            slow = [p for p in preds if p.get("ego") and p["ego"]["vx"] < 5]
            report[cell][prompt] = {
                "n": n,
                "finished": finished,
                "crashes": crashes,
                "terminal_stalls": stalls,
                "finish_wilson95": wilson(finished, n),
                "mean_route_pct": (sum(row["route_pct"] for row in group) / n) if n else None,
                "mean_real_predictions": (sum(row["real_predictions"] for row in group) / n) if n else None,
                "letters": [p["letter"] for p in preds],
                "agreement": (
                    sum(bool(p["agrees_with_prompt_rule"]) for p in preds) / len(preds) if preds else None
                ),
                "slow_calls": len(slow),
                "slow_letter_counts": dict(Counter(p["letter"] for p in slow)),
            }
    paired = []
    for cell, seeds in CELLS.items():
        for seed in seeds:
            a = next((row for row in rows if row["cell"] == cell and row["seed"] == seed and row["prompt"] == "pinned" and "error" not in row), None)
            b = next((row for row in rows if row["cell"] == cell and row["seed"] == seed and row["prompt"] == "measured" and "error" not in row), None)
            if a is None or b is None:
                continue
            paired.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "pinned_success": a["success"],
                    "measured_success": b["success"],
                    "pinned_letters": [p["letter"] for p in a["predictions"]],
                    "measured_letters": [p["letter"] for p in b["predictions"]],
                    "letters_changed": [p["letter"] for p in a["predictions"]] != [p["letter"] for p in b["predictions"]],
                    "outcome_changed": a["success"] != b["success"],
                }
            )
    changed = [item for item in paired if item["letters_changed"]]
    return {"by_cell": report, "paired": paired, "paired_letter_changes": len(changed), "paired_n": len(paired)}


def write_report(rows):
    summary = summarize(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Real camera model, final stack",
        "",
        "The router on every call is the trained camera model. No scripted lane rule drove the car.",
        "The system text is the training prompt. The only difference between the two arms is the speed number in the state line: always 20 m/s, or the measured speed.",
        "",
        "A finish is no crash and at least 99% of the route. Intervals are 95% Wilson intervals.",
        "",
        "| Scene | State line | Finished | Crashed | Stalled | Mean route | Calls | Letter agreement |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        for prompt in PROMPTS:
            item = summary["by_cell"][cell][prompt]
            if not item["n"]:
                continue
            interval = item["finish_wilson95"]
            interval_txt = f"{item['finished']}/{item['n']} ({interval[0]:.0%}–{interval[1]:.0%})"
            agree = "—" if item["agreement"] is None else f"{item['agreement']:.0%}"
            lines.append(
                f"| {cell} | {prompt} | {interval_txt} | {item['crashes']} | {item['terminal_stalls']} | "
                f"{item['mean_route_pct']:.0f}% | {item['mean_real_predictions']:.1f} | {agree} |"
            )
    lines += [
        "",
        f"Paired seeds: {summary['paired_n']}. The letter sequence changed on {summary['paired_letter_changes']} of them.",
        "",
        "Letter agreement is an offline score against a geometric reading of the trained prompt (right-lane blocker within 100 m and slower than 15 m/s; left lane occupied within 20 m). That score did not choose a skill.",
        "",
    ]
    slow_bits = []
    for cell in CELLS:
        for prompt in PROMPTS:
            item = summary["by_cell"][cell][prompt]
            if item["slow_calls"]:
                slow_bits.append(f"{cell}/{prompt}: {item['slow_letter_counts']} on {item['slow_calls']} calls below 5 m/s")
    if slow_bits:
        lines.append("Calls made while slower than 5 m/s: " + "; ".join(slow_bits) + ".")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "REPORT.md").write_text(text + "\n")
    report_path = ROOT / "reports/real-model-study.md"
    report_path.write_text(text + "\n")
    print(text, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(protocol(), indent=2, sort_keys=True)
    lock = OUT / "protocol.json"
    if lock.exists() and lock.read_text() != encoded:
        raise SystemExit("protocol changed; use a fresh output directory")
    lock.write_text(encoded)
    done_path = OUT / "episodes.jsonl"
    done = []
    if done_path.exists():
        done = [json.loads(line) for line in done_path.read_text().splitlines() if line.strip()]
    seen = {f"{row['cell']}-{row['seed']}-{row['prompt']}" for row in done}
    remaining = [case for case in design() if f"{case['cell']}-{case['seed']}-{case['prompt']}" not in seen]
    if not remaining:
        write_report(done)
        return
    print(f"stage: loading model, {len(remaining)} episodes left", flush=True)
    policy = TrainedVLMCPU(
        BASE,
        asset_directory(ROOT / "models/camera"),
        PROMPT_PATHS["pinned"],
        threads=4,
    )
    bank, _ = load_skill_bank(ROOT / "models/physical")
    with done_path.open("a") as handle:
        for index, case in enumerate(remaining, start=1):
            label = f"{case['cell']}-{case['seed']}-{case['prompt']}"
            print(f"stage: episode {index}/{len(remaining)} {label}", flush=True)
            try:
                row = run_case(case, policy, bank)
            except Exception as exc:
                row = {**case, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                print(row["traceback"], flush=True)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if "error" in row:
                print(f"FAIL {label}: {row['error']}", flush=True)
            else:
                print(
                    f"DONE {label} route={row['route_pct']:.1f} crash={row['crashed']} "
                    f"success={row['success']} calls={row['real_predictions']} "
                    f"letters={[p['letter'] for p in row['predictions']]} wall={row['wall_s']:.0f}s",
                    flush=True,
                )
            done.append(row)
    write_report(done)


if __name__ == "__main__":
    main()
