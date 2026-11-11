"""Hand-computable contrasts independent of any running experiment outcomes."""

import importlib
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
report = importlib.import_module("component_report")


def factorial_rows():
    rows = []
    for seed, controller, cbf, braking, admission in itertools.product(
        [1, 2], ["pd", "ppo"], [False, True], [False, True], [False, True]
    ):
        # Deliberately large seed intercept cancels in paired contrasts.
        outcome = (100 * seed + 2 * (controller == "ppo") + 3 * cbf + 5 * braking
                   + 7 * admission + 11 * cbf * braking)
        rows.append({
            "scene": "test", "seed": seed, "latency_s": .125, "router": "state_rule",
            "controller": controller, "cbf": cbf, "braking": braking, "admission": admission,
            **{metric: outcome if metric == "route_pct" else 0 for metric in report.METRICS},
        })
    return rows


@pytest.mark.parametrize("factor,expected", [
    ("controller", 2), ("cbf", 8.5), ("braking", 10.5), ("admission", 7),
])
def test_main_effects_average_paired_plus_minus_over_other_factors(factor, expected):
    result = report.contrast(factorial_rows(), (factor,))
    assert result["paired_contexts"] == 16
    assert result["seed_clusters"] == 2
    assert result["metrics"]["route_pct"]["difference"] == pytest.approx(expected)
    assert result["metrics"]["route_pct"]["ci95"] == pytest.approx([expected, expected])


def test_interaction_is_unscaled_difference_of_differences_and_order_invariant():
    rows = factorial_rows()
    for factors in [("cbf", "braking"), ("braking", "cbf")]:
        result = report.contrast(rows, factors)
        assert result["paired_contexts"] == 8
        assert result["metrics"]["route_pct"]["difference"] == pytest.approx(11)
    independent = report.contrast(rows, ("controller", "admission"))
    assert independent["metrics"]["route_pct"]["difference"] == 0


def test_conditional_leave_one_out_uses_full_other_components():
    rows = report.select(factorial_rows(), controller="ppo", braking=True, admission=True)
    result = report.contrast(rows, ("cbf",))
    assert result["paired_contexts"] == 2
    # Conditional effect3+11 differs from main effect3+11/2.
    assert result["metrics"]["route_pct"]["difference"] == 14
    controller_rows = report.select(factorial_rows(), cbf=True, braking=True, admission=True)
    assert report.contrast(controller_rows, ("controller",))["metrics"]["route_pct"]["difference"] == 2


def test_seed_bootstrap_keeps_shared_contexts_together():
    rows = []
    for seed, scene, cbf in itertools.product([1, 2], ["a", "b", "c"], [False, True]):
        effect = 2 if seed == 1 else 6
        rows.append({
            "scene": scene, "seed": seed, "latency_s": .125, "router": "state_rule",
            "controller": "ppo", "cbf": cbf, "braking": True, "admission": True,
            **{metric: effect * cbf if metric == "route_pct" else 0 for metric in report.METRICS},
        })
    result = report.contrast(rows, ("cbf",))
    assert result["seed_clusters"] == 2 and result["paired_contexts"] == 6
    assert result["metrics"]["route_pct"] == {"difference": 4., "ci95": [2., 6.]}


def test_missing_treatment_level_cannot_enter_contrast():
    with pytest.raises(ValueError, match="incomplete factorial contrast"):
        report.contrast(factorial_rows()[:-1], ("cbf",))


def test_cell_keys_distinguish_all_sixteen_factorial_settings():
    keys = {report.cell_key(row) for row in factorial_rows()}
    assert len(keys) == 16
    assert "ppo|C1B1G1" in keys and "pd|C0B0G0" in keys
