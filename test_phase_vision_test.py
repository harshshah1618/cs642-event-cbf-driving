"""Diagnostic design/aggregation tests; no model or simulator execution."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "phase_diag", Path(__file__).resolve().parent / "phase_vision_test.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def rows():
    return [
        {
            "case": name,
            "skill": skill,
            "prediction": {"image_hash": name, "letter": expected, "latency_s": 1.0},
        }
        for name, *_, expected in mod.CASES
        for skill in mod.SKILLS
    ]


def test_design_and_causal_flip_summary():
    data = rows()
    assert len(data) == 24 and mod.summarize(data)["design_label_matches"] == 24
    data[0]["prediction"]["letter"] = "A"
    result = mod.summarize(data)
    assert result["images_with_text_flip"] == 1 and result["design_label_matches"] == 23


def test_reject_missing_hint_or_changed_pixels():
    with pytest.raises(ValueError, match="three distinct"):
        mod.summarize(rows()[:-1])
    data = rows()
    data[0]["prediction"]["image_hash"] = "changed"
    with pytest.raises(ValueError, match="identical"):
        mod.summarize(data)
