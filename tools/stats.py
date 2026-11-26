"""Small, dependency-free statistics for the paper tables.

- Wilson score interval for a binomial rate.
- Percentile bootstrap interval for a mean (fixed RNG seed, reproducible).
- Exact two-sided McNemar (sign) test for paired binary outcomes on the same maps.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean(values: Sequence[float], reps: int = 10000, seed: int = 0,
                   alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return (means[int(alpha / 2 * reps)], means[int((1 - alpha / 2) * reps) - 1])


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Paired outcomes on the same maps. Returns (a-only wins, b-only wins, two-sided p)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = a_only + b_only
    if n == 0:
        return (0, 0, 1.0)
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return (a_only, b_only, min(1.0, 2 * tail))


def fmt_rate(successes: int, n: int) -> str:
    lo, hi = wilson(successes, n)
    return f"{100 * successes / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
