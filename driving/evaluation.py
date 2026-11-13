"""Run one episode. The full comparison and analysis live in tools/."""

import copy
from typing import Mapping

from driving.config import EVAL_MODES, ScenarioConfig, ShieldConfig, StackConfig
from driving.controller import DrivingStack
from driving.env_registry import make_env
from driving.errors import ConfigError
from driving.model_loader import load_skill_bank as _load_bank
from driving.policy import SkillPolicy
from driving.safety import InflatedCBFShield


def load_skill_bank(path):
    bank = _load_bank(path)
    return bank, bank.metadata


def run_episode(
    mode: str,
    scenario: ScenarioConfig,
    seed: int,
    skills: Mapping[str, SkillPolicy],
    router,
    *,
    stack_config=None,
    shield_config=None,
):
    if mode not in EVAL_MODES:
        raise ConfigError(f"Unknown mode: {mode}")
    stack_config = copy.deepcopy(stack_config or StackConfig())
    stack_config.mode = mode
    shield_config = copy.deepcopy(shield_config or ShieldConfig())
    env = make_env(scenario.name, copy.deepcopy(scenario))
    try:
        return DrivingStack(
            dict(skills),
            router,
            config=stack_config,
            shield=InflatedCBFShield(config=shield_config),
        ).run(env, seed=seed)
    finally:
        env.close()
