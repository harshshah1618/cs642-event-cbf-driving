"""Figure writers for media/full-stack/."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from event_cbf_driving.env.skill_lane import LANE_Y
from event_cbf_driving.stack.controller import RolloutResult


def plot_training_curves(histories: Dict[str, List[float]], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, hist in histories.items():
        if not hist:
            continue
        xs = np.arange(1, len(hist) + 1)
        ax.plot(xs, hist, label=name, linewidth=1.6)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Mean episode return")
    ax.set_title("Skill training curves (empty-road lane regulators)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_gauntlet_rollouts(results: List[RolloutResult], out_path: Path, stalled_x: float = 75.0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    titles = {
        "unshielded_skill": "Unshielded skill",
        "static_cbf": "Static CBF (no router)",
        "fixed_rate_vlm": "Fixed-rate VLM + hard switch",
        "proposed": "Proposed (event + inflated + blend)",
    }
    for ax, r in zip(axes, results):
        ax.plot(r.xs, r.ys, color="#1d4e89", linewidth=1.8)
        ax.axhline(LANE_Y[0], color="#888", ls="--", lw=0.8)
        ax.axhline(LANE_Y[1], color="#888", ls="--", lw=0.8)
        ax.scatter([stalled_x], [LANE_Y[0]], c="k", s=60, zorder=5, label="stalled")
        if r.crashed:
            ax.scatter([r.x], [r.y], c="red", marker="x", s=80, zorder=6)
        ax.set_title(titles.get(r.mode, r.mode), fontsize=10)
        ax.set_ylim(-1.5, 5.5)
        ax.set_xlim(0, max(200, max(r.xs) if r.xs else 200))
        ax.grid(True, alpha=0.25)
    axes[2].set_xlabel("x (m)")
    axes[3].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[2].set_ylabel("y (m)")
    fig.suptitle("Two-Trap Gauntlet closed-loop rollouts", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_ablation_bars(results: List[RolloutResult], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    route, calls, jerks = [], [], []
    short = {
        "unshielded_skill": "Unshielded",
        "static_cbf": "Static CBF",
        "fixed_rate_vlm": "Fixed VLM",
        "proposed": "Proposed",
    }
    for r in results:
        labels.append(short.get(r.mode, r.mode))
        route.append(r.route_pct)
        calls.append(r.router_calls)
        jerks.append(r.peak_lateral_jerk)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    colors = ["#b23a48", "#6c757d", "#e09f3e", "#2a9d8f"]
    for ax, vals, title, ylab in zip(
        axes,
        [route, calls, jerks],
        ["Route completion %", "Router calls", "Peak lateral jerk"],
        ["%", "count", "m/s³"],
    ):
        ax.bar(labels, vals, color=colors[: len(labels)])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Gauntlet ablation (measured)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
