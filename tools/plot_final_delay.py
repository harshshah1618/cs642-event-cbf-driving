"""Delay-vs-reserve figure and paired tests from the final SafeMetaDrive run (random untouched maps).

  PYTHONPATH=. .venv/bin/python tools/plot_final_delay.py runs/final/safemetadrive-2000-2199.json \
      reports/final/delay-reserve.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stats import mcnemar_exact, wilson  # noqa: E402

REFERENCE = "Ours (analytic tracker)"


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    episodes = json.loads(src.read_text())["episodes"]
    series: dict[float, dict[float, list]] = {6.5: {}, 0.0: {}}
    series[6.5][0.0] = series[0.0][0.0] = episodes[REFERENCE]
    for name, eps in episodes.items():
        m = re.match(r"Delay (\d+)s, reserve ([\d.]+) m", name)
        if m:
            series[float(m.group(2))][float(m.group(1))] = eps
    print("| Delay | Finished, reserve 6.5 m | Finished, no reserve | Paired (reserve-only / no-reserve-only), p |")
    print("|---:|---:|---:|---|")
    for delay in sorted(series[6.5]):
        a, b = series[6.5][delay], series[0.0][delay]
        sa, sb = [e["success"] for e in a], [e["success"] for e in b]
        x, y, p = mcnemar_exact(sa, sb)
        print(f"| {delay:g} s | {100 * sum(sa) / len(sa):.1f}% | {100 * sum(sb) / len(sb):.1f}% | "
              f"{x} / {y}, p={p:.2g} |")

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for reserve, colour, label in ((6.5, "#2563eb", "With turning-room reserve (6.5 m)"),
                                   (0.0, "#dc2626", "Without reserve")):
        delays = sorted(series[reserve])
        rates, lo, hi = [], [], []
        for d in delays:
            eps = series[reserve][d]
            k, n = sum(e["success"] for e in eps), len(eps)
            low, high = wilson(k, n)
            rates.append(100 * k / n)
            lo.append(100 * low)
            hi.append(100 * high)
        ax.plot(delays, rates, marker="o", color=colour, label=label, linewidth=2)
        ax.fill_between(delays, lo, hi, color=colour, alpha=0.15, linewidth=0)
    n = len(episodes[REFERENCE])
    ax.set_xlabel("Slow-planner decision delay (s)")
    ax.set_ylabel("Routes finished (%)")
    ax.set_ylim(50, 100)
    ax.set_title(f"SafeMetaDrive, {n} untouched maps; shaded: Wilson 95% CI", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
