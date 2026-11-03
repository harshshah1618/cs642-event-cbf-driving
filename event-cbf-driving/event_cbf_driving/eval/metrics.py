"""Measured metrics helpers — no fabricated numbers."""

from __future__ import annotations

from typing import Dict, List

from event_cbf_driving.stack.controller import RolloutResult


def result_to_row(r: RolloutResult) -> Dict:
    return {
        "mode": r.mode,
        "crash_time": None if r.crash_time is None else round(r.crash_time, 2),
        "crashed": r.crashed,
        "route_pct": round(r.route_pct, 2),
        "avg_speed": round(r.avg_speed, 3),
        "router_calls": r.router_calls,
        "peak_lateral_jerk": round(r.peak_lateral_jerk, 2),
        "freeze": r.freeze,
        "peak_sigma": round(r.peak_sigma, 3),
        "x": round(r.x, 2),
        "y": round(r.y, 3),
        "vx": round(r.vx, 3),
        "end_time": round(r.end_time, 2),
    }


def ablation_table(results: List[RolloutResult]) -> str:
    headers = ["Mode", "Crash (s)", "Route %", "Avg speed", "Freeze", "Router calls", "Peak jerk"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    labels = {
        "unshielded_skill": "1 Unshielded skill",
        "static_cbf": "2 Static CBF",
        "fixed_rate_vlm": "3 Fixed-rate VLM",
        "proposed": "4 Proposed",
        "fasionad_wakeup": "5 FASIONAD-style",
        "cert_replan": "6 CERT-Replan-style",
    }
    for r in results:
        crash = "none" if not r.crashed else f"{r.crash_time:.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    labels.get(r.mode, r.mode),
                    crash,
                    f"{r.route_pct:.1f}",
                    f"{r.avg_speed:.2f}",
                    "yes" if r.freeze else "no",
                    str(r.router_calls),
                    f"{r.peak_lateral_jerk:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
