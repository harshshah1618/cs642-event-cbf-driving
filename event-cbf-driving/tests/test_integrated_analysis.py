"""Paired inference must retain missing failures and cluster weighting."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("integrated_analysis")


def row(method, seed, scene, value, success=True):
    return dict(
        method=method,
        seed=seed,
        scene=scene,
        delay=0.125,
        success=success,
        crashed=False,
        terminal_stall=not success,
        route_pct=100 * success,
        expensive_calls=value,
        achieved_jerk_rms=value,
        achieved_jerk_peak=value,
        achieved_jerk_peak_after_0_2s=value,
    )


def test_pairing_rejects_a_missing_failed_comparator(analysis):
    rows = [
        row("final_integrated", 1, "s", 1),
        row("gated_poll", 1, "s", 2),
        row("final_integrated", 2, "s", 1, success=False),
    ]
    with pytest.raises(AssertionError):
        analysis.paired_contrast(rows, "gated_poll")


def test_paired_counts_keep_failures_and_report_signed_difference(analysis):
    rows = [
        row(m, k, "s", v, success=ok)
        for k, m, v, ok in (
            (1, "final_integrated", 1, True),
            (1, "gated_poll", 2, True),
            (2, "final_integrated", 3, False),
            (2, "gated_poll", 1, True),
        )
    ]
    result = analysis.paired_contrast(rows, "gated_poll")
    assert result["pairs"] == 2 and result["seed_clusters"] == 2
    assert result["success"]["final_minus_opponent"] == -0.5
    assert result["expensive_calls"]["final_minus_opponent"] == 0.5


def test_joint_success_comfort_explicitly_changes_denominator(analysis):
    rows = [
        row(m, k, "s", v, success=ok)
        for k, m, v, ok in (
            (1, "final_integrated", 1, True),
            (1, "no_blend", 4, True),
            (2, "final_integrated", 50, False),
            (2, "no_blend", 1, True),
        )
    ]
    result = analysis.paired_contrast(rows, "no_blend", joint_success=True)
    assert result["pairs"] == 1 and result["achieved_jerk_rms"]["final_minus_opponent"] == -3
