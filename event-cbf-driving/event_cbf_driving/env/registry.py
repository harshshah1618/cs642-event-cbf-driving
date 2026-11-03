"""Scenario registry — all eval scenarios go through the same stack."""

from __future__ import annotations

from typing import Any

from event_cbf_driving.env.dense_highway import DenseHighwayEnv
from event_cbf_driving.env.merge import MergeScenarioEnv
from event_cbf_driving.env.skill_lane import SkillLaneEnv
from event_cbf_driving.env.two_trap import TwoTrapGauntletEnv

SCENARIO_IDS = ("two_trap", "merge", "dense_highway")


def make_env(name: str, **kwargs: Any):
    name = name.lower().replace("-", "_")
    if name in ("two_trap", "two_trap_gauntlet", "gauntlet"):
        return TwoTrapGauntletEnv(**kwargs)
    if name == "merge":
        return MergeScenarioEnv(**kwargs)
    if name in ("dense", "dense_highway", "highway"):
        return DenseHighwayEnv(**kwargs)
    if name.startswith("skill"):
        return SkillLaneEnv(**kwargs)
    raise KeyError(f"Unknown scenario: {name}. Choose from {SCENARIO_IDS}")
