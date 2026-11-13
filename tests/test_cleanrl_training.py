"""A new trainer must produce usable drivers, not just write model files."""

import argparse
import json

import numpy as np
import pytest
import torch
from driving.checkpoints import load_skill, save_skill
from driving.config import StackConfig
from driving.controller import DrivingStack
from driving.env_two_trap import TwoTrapGauntletEnv
from driving.errors import ArtifactError, SafetyError
from driving.network import CleanRLAgent, encode, env_action_np
from driving.planner import ScriptedRouter
from driving.policy import SkillMeta, SkillPolicy
from driving.train import train


def test_training_changes_weights_and_checkpoint_preserves_actions(tmp_path):
    args = argparse.Namespace(
        updates=2,
        num_envs=2,
        num_steps=16,
        num_minibatches=2,
        update_epochs=2,
        skill="CRUISE_RIGHT",
        seed=7,
        output=tmp_path / "CRUISE_RIGHT.pt",
    )
    torch.manual_seed(args.seed)
    original = CleanRLAgent()
    trained = train(args)
    assert any(
        not torch.equal(v, original.state_dict()[k]) for k, v in trained.state_dict().items()
    )
    loaded = load_skill(args.output, strict=True)
    assert not loaded.supports_uncertainty
    obs = (2.0, -0.3, 12.0)
    with torch.no_grad():
        expected = trained.actor_mean(torch.from_numpy(encode(*obs, 0.0)).view(1, 3)).numpy()[0]
    ax, ay = env_action_np(np.clip(expected, -1, 1))
    np.testing.assert_array_equal(loaded.act(*obs), [ax, ay, 0.0])
    rows = [
        json.loads(line)
        for line in args.output.with_suffix(".metrics.jsonl").read_text().splitlines()
    ]
    assert [row["steps"] for row in rows] == [32, 64]
    before = args.output.read_bytes()
    with pytest.raises(FileExistsError):
        train(args)
    assert args.output.read_bytes() == before


def test_single_critic_driver_rejects_uncertainty_routing_but_can_drive():
    torch.manual_seed(0)
    policies = {
        name: SkillPolicy(name, CleanRLAgent(), SkillMeta(name, lane, y, 1, 0, 0, 1))
        for name, lane, y in (("CRUISE_RIGHT", 0, 0.0), ("OVERTAKE_LEFT", 1, 4.0))
    }
    stack = DrivingStack(
        policies, ScriptedRouter(), config=StackConfig(trigger_policy="uncertainty")
    )
    with pytest.raises(SafetyError, match="dual-critic"):
        stack.run(None)
    env = TwoTrapGauntletEnv(duration=0.1)
    try:
        stack = DrivingStack(
            policies,
            ScriptedRouter(),
            config=StackConfig(trigger_policy="deadline"),
        )
        result = stack.run(env, seed=0)
        assert result.end_time > 0
    finally:
        env.close()


def test_unknown_checkpoint_architecture_is_rejected(tmp_path):
    path = tmp_path / "CRUISE_RIGHT.pt"
    save_skill(path, CleanRLAgent(), SkillMeta("CRUISE_RIGHT", 0, 0, 1, 0, 0, 1))
    payload = torch.load(path, weights_only=True)
    payload["architecture"] = "unknown"
    torch.save(payload, path)
    with pytest.raises(ArtifactError, match="unsupported policy architecture"):
        load_skill(path)
