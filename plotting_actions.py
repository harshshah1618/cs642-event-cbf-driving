"""Plots of recorded decisions, skill requests, safety corrections and motion."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from result import RolloutResult


def plot_interactions(result: RolloutResult, path: Path) -> None:
    rows = result.interactions
    if not rows:
        raise ValueError("an interaction plot requires recorded per-tick interactions")
    t = np.asarray([r["t"] for r in rows])
    requested = np.asarray([r["requested_action"] for r in rows])
    applied = np.asarray([r["applied_action"] for r in rows])
    achieved = np.asarray(
        [r.get("plant", {}).get("achieved_acceleration", [np.nan, np.nan]) for r in rows]
    )
    pending = np.asarray([r["router_pending"] for r in rows])
    names = ["CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT"]
    colors = ["#2263a1", "#b76816", "#41834a"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    path_ax, router_ax, skill_ax, shield_ax, ax_ax, ay_ax = axes.ravel()
    for y in (0, 4):
        path_ax.axhline(y, color="#b6bec8", linestyle="--", linewidth=0.8)
    for i, name in enumerate(names):
        selected = np.array([row["skill"] == name for row in rows])
        path_ax.scatter(
            np.asarray(result.xs)[selected],
            np.asarray(result.ys)[selected],
            s=5,
            color=colors[i],
            label=name,
        )
    # Obstacle identities remain in spawn order throughout the episode.
    count = max(len(row.get("obstacles", [])) for row in rows)
    for index in range(count):
        points = [row["obstacles"][index] for row in rows if len(row.get("obstacles", [])) > index]
        path_ax.plot(
            [p[0] for p in points], [p[1] for p in points], color="#999999", linewidth=1, alpha=0.65
        )
    path_ax.set(
        title="Physical trajectory and traffic",
        xlabel="Distance along road (m)",
        ylabel="Lateral position (m)",
        ylim=(-2, 6),
    )
    path_ax.legend(fontsize=7, loc="upper left")
    router_ax.step(t, [names.index(row["skill"]) for row in rows], where="post", color="#2263a1")
    router_ax.set(
        yticks=range(3), yticklabels=names, title="Tier 1: selected maneuver", ylim=(-0.5, 2.5)
    )
    for call in result.calls:
        if call.get("suppressed"):
            continue
        router_ax.axvline(
            call["t"],
            color="#b33b3b" if call.get("rejected") else "#555555",
            alpha=0.5,
            linewidth=0.8,
        )
    router_ax.fill_between(
        t,
        -0.5,
        2.5,
        where=pending,
        color="#edc765",
        alpha=0.25,
        step="post",
        label="Router pending",
    )
    router_ax.legend(fontsize=8, loc="upper left")
    skill_ax.plot(t, [r["sigma"] for r in rows], color="#2263a1")
    skill_ax.set(title="Tier 2: critic disagreement", ylabel="Sigma (heuristic)")
    h = [r["barrier_h"] if r["barrier_h"] is not None else np.nan for r in rows]
    shield_ax.plot(t, h, label="Barrier h", color="#2263a1")
    shield_ax.plot(t, [r["inflation"] for r in rows], label="Extra margin", color="#b76816")
    if any(r.get("braking_h") is not None for r in rows):
        shield_ax.plot(t, [r.get("braking_h", np.nan) if r.get("braking_h") is not None else np.nan for r in rows],
                       label="Stopping envelope", color="#41834a", linestyle="--")
    shield_ax.axhline(0, color="#b33b3b", linewidth=0.8)
    shield_ax.set(title="Tier 3: barrier and margin", ylabel="Distance (m)")
    shield_ax.legend(fontsize=8)
    for index, axis in enumerate((ax_ax, ay_ax)):
        axis.plot(t, requested[:, index], label="Skill request", color="#b76816", alpha=0.8)
        axis.plot(t, applied[:, index], label="After safety gate", color="#2263a1")
        axis.plot(t, achieved[:, index], label="Achieved motion", color="#41834a", alpha=0.8)
        axis.set(
            title="Forward acceleration" if index == 0 else "Lateral acceleration",
            ylabel="Acceleration (m/s²)",
        )
        axis.legend(fontsize=8)
    for axis in (router_ax, skill_ax, shield_ax, ax_ax, ay_ax):
        axis.set_xlabel("Simulation time (s)")
    for axis in axes.ravel():
        axis.grid(alpha=0.18)
    fig.suptitle(
        f"{result.scenario} · {result.mode} · seed {result.seed} | "
        f"route {result.route_pct:.1f}% | crash {result.crashed}"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    finally:
        plt.close(fig)
