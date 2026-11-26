"""Statistics helpers used by the paper tables, and the highway-env crossing test."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from stats import bootstrap_mean, mcnemar_exact, wilson  # noqa: E402


def test_wilson_known_values():
    lo, hi = wilson(97, 100)
    assert lo == pytest.approx(0.9154, abs=1e-3) and hi == pytest.approx(0.9897, abs=1e-3)
    assert wilson(0, 10)[0] == pytest.approx(0.0) and wilson(10, 10)[1] == pytest.approx(1.0)


def test_mcnemar_exact_matches_binomial():
    a = [True] * 9 + [False] * 1 + [True] * 50
    b = [False] * 9 + [True] * 1 + [True] * 50
    a_only, b_only, p = mcnemar_exact(a, b)
    assert (a_only, b_only) == (9, 1)
    assert p == pytest.approx(0.02148, abs=1e-4)  # 2 * P(X <= 1), X ~ Bin(10, 0.5)
    assert mcnemar_exact([True, False], [True, False])[2] == 1.0


def test_bootstrap_brackets_mean():
    values = [0.0] * 90 + [1.0] * 10
    lo, hi = bootstrap_mean(values)
    assert lo < 0.1 < hi


def test_highway_crossing_conflict():
    pytest.importorskip("highway_env")
    from benchmark_highway_env import HighwayEnvStack

    obs = np.zeros((3, 5))
    obs[0] = [1, 0, 0, 10, 0]
    obs[1] = [1, 15.0, -12.0, -10.0, 8.0]   # converging from the side
    obs[2] = [1, 20.0, 4.0, 0.0, 0.0]       # parallel, adjacent lane
    assert HighwayEnvStack.crossing_conflict(obs)
    obs[1] = [1, 15.0, -12.0, 0.0, 0.0]
    assert not HighwayEnvStack.crossing_conflict(obs)
