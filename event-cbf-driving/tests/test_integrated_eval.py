"""Semantic trigger isolation, pending latency and effective baseline masks."""

import importlib.util
import sys
from pathlib import Path

import pytest
from event_cbf_driving.config import StackConfig, StackMode
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.stack.controller import DrivingStack

from tests.test_controller import FakeEnv, FakeRouter, FakeSkill

spec = importlib.util.spec_from_file_location(
    "integrated_eval", Path(__file__).resolve().parents[1] / "scripts/integrated_eval.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


@pytest.mark.parametrize(
    "trigger,sigma,expected",
    [
        ("uncertainty", 0.0, 0),
        ("uncertainty", 4.0, 1),
        ("barrier", 4.0, 0),
        ("legacy", 4.0, 1),
    ],
)
def test_uncertainty_isolated_from_barrier(trigger, sigma, expected):
    skills = {n: FakeSkill(n, sigma=sigma) for n in ("CRUISE_RIGHT", "OVERTAKE_LEFT")}
    cfg = StackConfig(
        trigger_policy=trigger,
        sigma_threshold_scale=1.0,
        pending_margin=False,
        pending_margin_on_all_calls=True,
    )
    stack = DrivingStack(
        skills, FakeRouter(latency_s=0.2), shield=InflatedCBFShield(inflate=False), config=cfg
    )
    result = stack.run(FakeEnv(duration=0.5))
    assert result.router_calls == expected
    if expected:
        assert result.pending_ticks >= 9  # Same-skill answers really wait.


def test_matched_polling_respects_explicit_hard_switch():
    job = mod.Job("stall_ahead", 8201, 0.125, "matched_timer_hard")
    cfg, _ = mod.configs(job)
    shield = InflatedCBFShield(config=cfg.shield)
    DrivingStack._configure_mode(shield, StackMode.FIXED_VLM, cfg.stack, variant=cfg.stack.mode)
    assert shield.enabled and cfg.stack.hard_switch and not cfg.stack.use_blend


@pytest.mark.parametrize("method", mod.METHODS)
def test_every_method_effective_masks_and_latency_contract(method):
    cfg, reserve = mod.configs(mod.Job("cut_in", 8201, 18.0, method))
    shield = InflatedCBFShield(config=cfg.shield)
    base = DrivingStack._resolve_mode(cfg.stack.mode)
    DrivingStack._configure_mode(shield, base, cfg.stack, variant=cfg.stack.mode)
    assert cfg.stack.pending_margin_on_all_calls
    assert cfg.stack.max_pending_s > 18
    assert shield.road_boundary_guard
    if method in ("unshielded_rl", "timer_hard_unshielded"):
        assert not shield.enabled and not shield.braking_guard
    elif method == "static_cbf":
        assert shield.enabled and not shield.braking_guard and reserve == "baseline"
    else:
        assert shield.enabled and shield.braking_guard and cfg.stack.skill_switch_guard


def test_matched_scheduling_controls_share_protection_and_blending():
    protected = ("final_integrated", "high_rate_poll", "gated_poll", "immediate_benefit")
    for method in protected:
        cfg, reserve = mod.configs(mod.Job("dense", 8201, 1.0, method))
        assert reserve == "fixed_latched" and cfg.stack.use_blend
        assert not cfg.stack.hard_switch and not cfg.stack.pending_margin
        assert cfg.shield.gamma_sigma == 0 and cfg.stack.skill_switch_guard


@pytest.mark.parametrize("scene", mod.SCENES)
def test_scene_contract_accepts_only_supported_params(scene):
    job = mod.Job(scene, 8201, .125, "final_integrated")
    spec = mod.scene_spec(job)
    env = mod.make_env(spec.name, spec)
    env.close()
