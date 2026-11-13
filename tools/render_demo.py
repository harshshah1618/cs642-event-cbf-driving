"""Make an annotated video from actual recorded MetaDrive states and layer logs.

This is a top-down reconstruction, not a camera recording. No trajectories or
decisions are invented. Vehicle yaw is approximated from recorded velocity.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from driving import DT
from driving.runtime import load_episode
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D


def car(axis, state, color, ego=False):
    x, y, vx, vy = state
    yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > 1 else 0.0
    rect = Rectangle(
        (-2.2575, -0.926),
        4.515,
        1.852,
        linewidth=1.1,
        edgecolor="white" if ego else "#374151",
        facecolor=color,
        zorder=4,
    )
    rect.set_transform(Affine2D().rotate(yaw).translate(x, y) + axis.transData)
    axis.add_patch(rect)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()
    if not 1 <= args.fps <= 60:
        raise ValueError("fps must be between 1 and 60")
    episode = load_episode(args.episode)
    result = episode.result
    rows = result.interactions
    if not rows:
        raise ValueError("enable stack.record_interactions to make a demonstration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 800
    fig = plt.figure(figsize=(12.8, 8), dpi=100, facecolor="#f4f6f8")
    road = fig.add_axes((0.055, 0.52, 0.89, 0.34))
    cards = [fig.add_axes((0.045 + i * 0.318, 0.16, 0.294, 0.28)) for i in range(3)]
    timeline = fig.add_axes((0.08, 0.055, 0.84, 0.055))
    name_to_value = {"CRUISE_RIGHT": 0, "OVERTAKE_LEFT": 1, "RETURN_RIGHT": 2}
    end = rows[-1]["t"] + DT
    count = int(math.ceil(end * args.fps))
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    colors = {"CRUISE_RIGHT": "#176c9b", "OVERTAKE_LEFT": "#248a61", "RETURN_RIGHT": "#a15a23"}
    try:
        for frame in range(count):
            row = rows[min(int(frame / args.fps / DT), len(rows) - 1)]
            road.clear()
            for axis in cards:
                axis.clear()
                axis.set_axis_off()
                axis.add_patch(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=axis.transAxes,
                        facecolor="white",
                        edgecolor="#d8dfe5",
                    )
                )
            road.set_facecolor("#333d49")
            x, y, vx, vy = row["ego"]
            road.axhspan(-2, 6, color="#333d49")
            road.axhline(2, color="#e0e5e9", linestyle=(0, (10, 8)), linewidth=1.5)
            for edge in (-2, 6):
                road.axhline(edge, color="white", linewidth=2)
            for obs in row.get("obstacles", []):
                car(road, obs, "#9aa7b3")
            car(road, row["ego"], colors[row["skill"]], ego=True)
            road.set(
                xlim=(x - 25, x + 65),
                ylim=(-3, 7),
                xlabel="Distance along road (m)",
                ylabel="Lateral position (m)",
            )
            road.tick_params(colors="#344054")
            for spine in road.spines.values():
                spine.set_visible(False)
            road.text(x - 23, 5.2, "LEFT", color="#d5dce3", fontsize=9)
            road.text(x - 23, 1.1, "RIGHT", color="#d5dce3", fontsize=9)
            titles = [
                "TIER 1  |  MANEUVER ROUTER",
                "TIER 2  |  FROZEN PPO SKILL",
                "TIER 3  |  SAFETY FILTER",
            ]
            for axis, title in zip(cards, titles, strict=True):
                axis.text(
                    0.05,
                    0.91,
                    title,
                    fontsize=11,
                    weight="bold",
                    color="#273b50",
                    transform=axis.transAxes,
                )
            calls = [
                c for c in result.calls if c.get("t", 0) <= row["t"] and not c.get("suppressed")
            ]
            last = calls[-1] if calls else {}
            pending = row["router_pending"]
            cards[0].text(
                0.05,
                0.71,
                row["skill"],
                fontsize=16,
                weight="bold",
                color=colors[row["skill"]],
                transform=cards[0].transAxes,
            )
            lines = [
                "DECISION PENDING" if pending else "Active maneuver executing",
                "Control continues with fresh state" if pending else "Event-triggered decisions",
                f"Calls so far: {len(calls)}",
                f"Last trigger: {last.get('reason', 'none')}",
            ]
            for i, line in enumerate(lines):
                cards[0].text(
                    0.05,
                    0.52 - i * 0.12,
                    line,
                    fontsize=10,
                    color="#855b13" if pending else "#344054",
                    transform=cards[0].transAxes,
                )
            ax, ay = row["requested_action"]
            cards[1].text(
                0.05,
                0.70,
                f"Requested ax: {ax:+.2f} m/s²",
                fontsize=13,
                transform=cards[1].transAxes,
            )
            cards[1].text(
                0.05,
                0.54,
                f"Requested ay: {ay:+.2f} m/s²",
                fontsize=13,
                transform=cards[1].transAxes,
            )
            cards[1].text(
                0.05,
                0.35,
                f"Critic disagreement: {row['sigma']:.2f}",
                fontsize=11,
                transform=cards[1].transAxes,
            )
            cards[1].text(
                0.05,
                0.19,
                f"Skill blend weight: {row['blend_alpha']:.2f}",
                fontsize=11,
                transform=cards[1].transAxes,
            )
            final = row["applied_action"]
            cards[2].text(
                0.05,
                0.72,
                f"Filtered ax / ay: {final[0]:+.2f} / {final[1]:+.2f}",
                fontsize=12,
                transform=cards[2].transAxes,
            )
            desired = row.get("barrier_h")
            physical = row.get("braking_h")
            cards[2].text(
                0.05,
                0.55,
                f"Clearance target h: {desired:.2f} m"
                if desired is not None
                else "No relevant front obstacle",
                fontsize=11,
                transform=cards[2].transAxes,
            )
            cards[2].text(
                0.05,
                0.40,
                f"Stopping envelope h: {physical:.2f} m"
                if physical is not None
                else "Stopping envelope: clear",
                fontsize=11,
                transform=cards[2].transAxes,
            )
            cards[2].text(
                0.05,
                0.25,
                f"Shield override: {'YES' if row['shield_override'] else 'no'}",
                fontsize=11,
                weight="bold",
                color="#b15c1a" if row["shield_override"] else "#248a61",
                transform=cards[2].transAxes,
            )
            cards[2].text(
                0.05, 0.10, f"Monitor: {row['monitor']}", fontsize=10, transform=cards[2].transAxes
            )
            timeline.clear()
            times = np.array([r["t"] for r in rows])
            values = np.array([name_to_value[r["skill"]] for r in rows])
            timeline.step(times, values, where="post", color="#377a94", linewidth=2)
            timeline.axvline(row["t"], color="#c35832", linewidth=2)
            timeline.set(xlim=(0, end), ylim=(-0.3, 2.3), yticks=[], xlabel="Simulation time (s)")
            timeline.spines[["left", "right", "top"]].set_visible(False)
            fig.suptitle(
                f"Three-layer driving stack  |  {result.scenario}  |  t={row['t']:.2f}s  |  speed={vx:.1f} m/s",
                fontsize=17,
                weight="bold",
                y=0.965,
                color="#23364d",
            )
            if frame == 0:
                fig.text(
                    0.5,
                    0.895,
                    "Top-down reconstruction from recorded MetaDrive physics and layer logs",
                    ha="center",
                    fontsize=11,
                    color="#5a6979",
                )
                fig.text(
                    0.5,
                    0.451,
                    "ROUTER SELECTS  →  SKILL PROPOSES  →  SHIELD FILTERS  →  PHYSICAL VEHICLE",
                    ha="center",
                    fontsize=11,
                    color="#42566a",
                )
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes()
            if process.stdin is None:
                raise RuntimeError("ffmpeg input unavailable")
            process.stdin.write(rgb)
            if frame in (0, count // 2, count - 1):
                fig.savefig(args.output.with_name(f"{args.output.stem}-frame-{frame}.png"), dpi=100)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
