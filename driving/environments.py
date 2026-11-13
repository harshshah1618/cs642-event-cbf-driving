"""Scenario environments implementing the shared :class:`DrivingEnv` contract."""

from driving.env_base import (
    DrivingEnv,
    copy_snapshot,
    detached_snapshot,
    validate_action,
    validate_snapshot,
    validate_vehicle,
)
from driving.env_carla_stub import CarlaAdapterStub, MockCarlaAdapter
from driving.env_dense import DenseHighwayEnv, make_highway_env_continuous
from driving.env_highway import (
    HighwayEnvAdapter,
    accel_to_highway_action,
    configure_highway_continuous,
    make_highway_v0,
    make_merge_v0,
)
from driving.env_lane import LANE_Y, TARGET_SPEED, SkillLaneEnv
from driving.env_merge import MergeScenarioEnv
from driving.env_registry import SCENARIO_IDS, list_scenarios, make_env, scenario_defaults
from driving.env_two_trap import TwoTrapGauntletEnv
from driving.env_types import (
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
