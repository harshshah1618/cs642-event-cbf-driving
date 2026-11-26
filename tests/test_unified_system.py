"""Tests for the canonical 3-layer UnifiedDrivingSystem."""

import inspect
from pathlib import Path
import numpy as np

from driving.safemetadrive.unified import UnifiedConfig, UnifiedDrivingSystem
from tests.test_lidar_driver import _make_obs


def test_unified_zero_peeking_contract():
    sig = inspect.signature(UnifiedDrivingSystem.__init__)
    assert "vehicle" not in sig.parameters
    assert "env" not in sig.parameters
    sys_drv = UnifiedDrivingSystem(_make_obs())
    assert not hasattr(sys_drv, "vehicle")
    assert not hasattr(sys_drv, "env")
    act = sys_drv.act()
    assert len(act) == 2 and all(np.isfinite(x) and -1.0 <= x <= 1.0 for x in act)


def test_unified_semantic_stop_and_careful_leases():
    obs = _make_obs(speed_kmh=30.0)
    drv = UnifiedDrivingSystem(obs)
    drv.set_semantic_lease(stop=True)
    _, throttle = drv.act()
    assert throttle <= -0.9

    drv.set_semantic_lease(stop=False, careful=True)
    _, throttle_careful = drv.act()
    assert throttle_careful < 0.0


def test_unified_loads_real_ppo_checkpoint_pure_pytorch():
    ckpt = Path("runs/safemetadrive/skills-1m-v1/skills-seed0/ppo.zip")
    if not ckpt.exists():
        return
    obs = _make_obs(speed_kmh=25.0)
    drv = UnifiedDrivingSystem(obs, UnifiedConfig(tracker="ppo", skill_checkpoint=str(ckpt)))
    act = drv.act()
    assert len(act) == 2 and all(np.isfinite(x) and -1.0 <= x <= 1.0 for x in act)


def test_unified_always_poll_counts_every_step():
    obs = _make_obs(speed_kmh=25.0)
    drv = UnifiedDrivingSystem(obs, router="always_poll")
    for _ in range(5):
        drv.act()
    assert drv.router_calls == 5
