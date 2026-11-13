"""Compatibility imports; use train_ppo.py, policy.py and checkpoints.py directly."""

import torch

from .checkpoints import (
    _load_torch_checkpoint,
    build_checkpoint_provenance,
    find_duplicate_fingerprints,
    inspect_checkpoint,
    load_skill,
    save_skill,
    state_dict_fingerprint,
    validate_skill_meta,
)
from .diagnostics import calibrate_threshold, evaluate_skill
from .lane_env import EmptyRoadVec
from .legacy_ppo import PPOTrainer, compute_gae
from .model import (
    DEVICE,
    TARGET_SPEED,
    ActorCritic,
    encode,
    env_action_np,
    pi_parameters,
    v_parameters,
)
from .policy import SkillMeta, SkillPolicy

__all__ = [
    "torch",
    "_load_torch_checkpoint",
    "build_checkpoint_provenance",
    "find_duplicate_fingerprints",
    "inspect_checkpoint",
    "load_skill",
    "save_skill",
    "state_dict_fingerprint",
    "validate_skill_meta",
    "calibrate_threshold",
    "evaluate_skill",
    "EmptyRoadVec",
    "PPOTrainer",
    "compute_gae",
    "DEVICE",
    "TARGET_SPEED",
    "ActorCritic",
    "encode",
    "env_action_np",
    "pi_parameters",
    "v_parameters",
    "SkillMeta",
    "SkillPolicy",
]
