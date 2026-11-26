"""Controlled delay sweep (0 s to 18 s) to find the exact tipping point where
extra turning room (8 m maneuver reserve) starts to matter.

Directly addresses milestone feedback:
"You need the controlled delay experiments (slowing the answers down on purpose
to find the delay at which the extra room starts to matter)."

Reuses the exact physical MetaDrive simulator, frozen PPO skill bank, 3 scenarios
(stall_ahead, cut_in, dense), and 8 seeds (9001-9008) from tools/integrated_eval.py.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-delay-sweep")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tools.integrated_eval import Job, SCENES, execute, initialize


SEEDS = tuple(range(9001, 9009))  # Exact 8 milestone seeds -> 24 runs per (delay, method)
DELAYS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0)
METHODS = ("final_integrated", "no_reserve")


def _run_job(job: Job) -> dict:
    return execute(job)


def plot_sweep(summary_rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delays = sorted({r["delay"] for r in summary_rows})
    by_m = {m: {r["delay"]: r for r in summary_rows if r["method"] == m} for m in METHODS}

    y_on = [by_m["final_integrated"][d]["completion_rate"] * 100.0 for d in delays]
    y_off = [by_m["no_reserve"][d]["completion_rate"] * 100.0 for d in delays]
    stall_on = [by_m["final_integrated"][d]["terminal_stall_rate"] * 100.0 for d in delays]
    stall_off = [by_m["no_reserve"][d]["terminal_stall_rate"] * 100.0 for d in delays]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=160)

    # Panel 1: Completion rate vs controlled delay
    ax1.plot(delays, y_on, marker="o", linewidth=2.4, color="#1b7837", label="With Extra Room (8 m reserve)")
    ax1.plot(delays, y_off, marker="s", linewidth=2.4, linestyle="--", color="#c51b7d", label="Without Extra Room (0 m)")
    ax1.fill_between(delays, y_off, y_on, where=[a > b for a, b in zip(y_on, y_off)],
                     alpha=0.14, color="#1b7837", interpolate=True, label="Benefit of extra room")
    ax1.set_title("Route Completion vs. Controlled Planner Delay", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Controlled Slow-AI Delay (seconds)")
    ax1.set_ylabel("Completed Runs (%)  [n = 24 per point]")
    ax1.set_ylim(-5, 108)
    ax1.set_xlim(-0.5, 18.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)

    # Annotate first tipping point where reserve ON beats reserve OFF
    for d, on, off in zip(delays, y_on, y_off):
        if on > off + 1e-6:
            ax1.axvline(d, color="#555555", linestyle=":", linewidth=1.3)
            ax1.annotate(
                f"Tipping point: {d:g} s\n({off:.0f}% vs {on:.0f}%)",
                xy=(d, off),
                xytext=(min(d + 1.5, 11.5), max(off - 22, 12)),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.9),
            )
            break

    # Panel 2: Terminal freeze / stall rate vs controlled delay
    ax2.plot(delays, stall_on, marker="o", linewidth=2.4, color="#1b7837", label="With Extra Room (8 m reserve)")
    ax2.plot(delays, stall_off, marker="s", linewidth=2.4, linestyle="--", color="#c51b7d", label="Without Extra Room (0 m)")
    ax2.set_title("Trapped / Frozen Runs vs. Controlled Delay", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Controlled Slow-AI Delay (seconds)")
    ax2.set_ylabel("Stalled / Trapped Behind Obstacle (%)")
    ax2.set_ylim(-5, 105)
    ax2.set_xlim(-0.5, 18.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    all_jobs = [
        Job(scene, seed, delay, method, 0.2)
        for delay, method, scene, seed in itertools.product(DELAYS, METHODS, SCENES, SEEDS)
    ]
    print(f"Launching {len(all_jobs)} controlled-delay physical MetaDrive episodes across 8 workers...", flush=True)
    t0 = time.monotonic()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=8, initializer=initialize) as pool:
        futs = [pool.submit(_run_job, j) for j in all_jobs]
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 96 == 0 or done == len(all_jobs):
                print(f"  progress: {done}/{len(all_jobs)} ({time.monotonic() - t0:.1f}s)", flush=True)

    summary_rows: list[dict] = []
    print("\n=== CONTROLLED DELAY SWEEP (24 runs per point: 3 scenes x 8 milestone seeds) ===")
    print(f"{'Delay (s)':>9s} | {'With Room (8m)':>16s} | {'No Room (0m)':>16s} | {'Crashes (On/Off)':>16s} | {'Stalls (On/Off)':>15s}")
    print("-" * 82)
    for d in DELAYS:
        per_m = {}
        for m in METHODS:
            sub = [r for r in rows if r["delay"] == d and r["method"] == m]
            rec = dict(
                delay=d,
                method=m,
                n=len(sub),
                finished=sum(int(r["success"]) for r in sub),
                completion_rate=float(np.mean([r["success"] for r in sub])),
                crashes=sum(int(r["crashed"]) for r in sub),
                terminal_stalls=sum(int(r["terminal_stall"]) for r in sub),
                terminal_stall_rate=float(np.mean([r["terminal_stall"] for r in sub])),
                mean_duration_s=float(np.mean([r["tick_duration_s"] for r in sub])),
                mean_calls=float(np.mean([r["expensive_calls"] for r in sub])),
            )
            summary_rows.append(rec)
            per_m[m] = rec
        on, off = per_m["final_integrated"], per_m["no_reserve"]
        print(
            f"{d:9.1f} | {on['finished']:2d}/24 ({on['completion_rate']*100:5.1f}%) | "
            f"{off['finished']:2d}/24 ({off['completion_rate']*100:5.1f}%) | "
            f"{on['crashes']:2d} / {off['crashes']:2d}          | "
            f"{on['terminal_stalls']:2d} / {off['terminal_stalls']:2d}"
        )

    out_json = Path("runs/benchmarks/controlled-delay-sweep-2026-11-25.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dict(summary=summary_rows, episodes=rows), indent=2))

    out_png = Path("reports/delay-tipping-point.png")
    plot_sweep(summary_rows, out_png)
    print(f"\nSaved JSON: {out_json}")
    print(f"Saved Plot: {out_png}")
    print(f"Total wall time: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
