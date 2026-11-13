"""M3 baseline comparison figures → media/full-stack/m3-*.png."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controller import RolloutResult
from env_lane import LANE_Y

SHORT = {
    "unshielded_skill": "Unshielded",
    "static_cbf": "Static CBF",
    "fixed_rate_vlm": "Fixed VLM",
    "proposed": "Proposed",
    "fasionad_wakeup": "FASIONAD-style",
    "cert_replan": "CERT-Replan-style",
}

COLORS = {
    "unshielded_skill": "#b23a48",
    "static_cbf": "#6c757d",
    "fixed_rate_vlm": "#e09f3e",
    "proposed": "#2a9d8f",
    "fasionad_wakeup": "#4c6ef5",
    "cert_replan": "#9b5de5",
}


def plot_m3_scenario_bars(
    results: List[RolloutResult],
    out_path: Path,
    title: str,
):
    """Route %, router calls, peak lateral jerk for one scenario (all modes)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels, route, calls, jerks, colors = [], [], [], [], []
    for r in results:
        labels.append(SHORT.get(r.mode, r.mode))
        route.append(r.route_pct)
        calls.append(r.router_calls)
        jerks.append(r.peak_lateral_jerk)
        colors.append(COLORS.get(r.mode, "#888888"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, vals, ttl, ylab in zip(
        axes,
        [route, calls, jerks],
        ["Route completion %", "Router calls", "Peak lateral jerk"],
        ["%", "count", "m/s³"], strict=False,
    ):
        ax.bar(labels, vals, color=colors)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_m3_gauntlet_rollouts(
    results: List[RolloutResult],
    out_path: Path,
    stalled_x: float = 75.0,
):
    """Trajectory panels for M3 modes (and optional reference modes if present)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.4 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    titles = {
        "unshielded_skill": "Unshielded skill",
        "static_cbf": "Static CBF",
        "fixed_rate_vlm": "Fixed-rate VLM",
        "proposed": "Proposed",
        "fasionad_wakeup": "FASIONAD-style (σ→nudge)",
        "cert_replan": "CERT-Replan-style (barrier replan)",
    }
    for ax, r in zip(axes, results, strict=False):
        ax.plot(r.xs, r.ys, color=COLORS.get(r.mode, "#1d4e89"), linewidth=1.8)
        ax.axhline(LANE_Y[0], color="#888", ls="--", lw=0.8)
        ax.axhline(LANE_Y[1], color="#888", ls="--", lw=0.8)
        ax.scatter([stalled_x], [LANE_Y[0]], c="k", s=50, zorder=5)
        if r.crashed:
            ax.scatter([r.x], [r.y], c="red", marker="x", s=80, zorder=6)
        ax.set_title(titles.get(r.mode, r.mode), fontsize=10)
        ax.set_ylim(-1.5, 5.5)
        ax.set_xlim(0, max(200, max(r.xs) if r.xs else 200))
        ax.grid(True, alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    axes[min(n - 1, cols)].set_xlabel("x (m)") if n > cols else None
    for i, ax in enumerate(axes[:n]):
        if i % cols == 0:
            ax.set_ylabel("y (m)")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("x (m)")
    fig.suptitle("M3 baselines — Two-Trap rollouts", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_m3_cross_scenario(
    by_scenario: Dict[str, List[RolloutResult]],
    out_path: Path,
    mode_keys: List[str],
):
    """Grouped bars: route % for selected modes across scenarios."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = list(by_scenario.keys())
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(mode_keys), 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for i, mk in enumerate(mode_keys):
        vals = []
        for scen in scenarios:
            match = [r for r in by_scenario[scen] if r.mode == mk]
            vals.append(match[0].route_pct if match else 0.0)
        ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=SHORT.get(mk, mk),
            color=COLORS.get(mk, None),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Route completion %")
    ax.set_title("M3-style baselines vs proposed (route %)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
