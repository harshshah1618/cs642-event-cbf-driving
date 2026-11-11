"""Scenario environments implementing the shared :class:`DrivingEnv` contract."""

from .base import (
    DrivingEnv,
    copy_snapshot,
    detached_snapshot,
    validate_action,
    validate_snapshot,
    validate_vehicle,
)
from .carla_stub import CarlaAdapterStub, MockCarlaAdapter
from .dense_highway import DenseHighwayEnv, make_highway_env_continuous
from .highway_adapter import (
    HighwayEnvAdapter,
    accel_to_highway_action,
    configure_highway_continuous,
    make_highway_v0,
    make_merge_v0,
)
from .merge import MergeScenarioEnv
from .registry import (
    SCENARIO_IDS,
    list_scenarios,
    make_env,
    scenario_defaults,
)
from .skill_lane import LANE_Y, TARGET_SPEED, SkillLaneEnv
from .two_trap import TwoTrapGauntletEnv
from .types import (
    ACTION_HIGH,
    ACTION_LOW,
    AX_HIGH,
    AX_LOW,
    AY_HIGH,
    AY_LOW,
    OBS_DIM,
    VehicleState,
    WorldSnapshot,
    collide,
    integrate_point_mass,
)

__all__ = [
    "DrivingEnv",
    "SkillLaneEnv",
    "TwoTrapGauntletEnv",
    "MergeScenarioEnv",
    "DenseHighwayEnv",
    "HighwayEnvAdapter",
    "CarlaAdapterStub",
    "MockCarlaAdapter",
    "make_env",
    "scenario_defaults",
    "list_scenarios",
    "SCENARIO_IDS",
    "LANE_Y",
    "TARGET_SPEED",
    "VehicleState",
    "WorldSnapshot",
    "integrate_point_mass",
    "collide",
    "validate_action",
    "validate_snapshot",
    "validate_vehicle",
    "detached_snapshot",
    "copy_snapshot",
    "accel_to_highway_action",
    "configure_highway_continuous",
    "make_highway_env_continuous",
    "make_highway_v0",
    "make_merge_v0",
    "AX_LOW",
    "AX_HIGH",
    "AY_LOW",
    "AY_HIGH",
    "ACTION_LOW",
    "ACTION_HIGH",
    "OBS_DIM",
]
