"""Cross-layer contracts, fresh-state switch rejection and real inference delay."""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest
from event_cbf_driving.config import StackConfig
from event_cbf_driving.env.types import VehicleState
from event_cbf_driving.errors import ConfigError
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.stack.controller import DrivingStack

from tests.test_controller import FakeEnv, FakeRouter, FakeSkill


def skills():
    bank = {
        name: FakeSkill(name, ay=0.0, sigma=2.0)
        for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")
    }
    for name, policy in bank.items():
        policy.meta.y_target = 4.0 if name == "OVERTAKE_LEFT" else 0.0
    return bank


def test_async_requires_clock_pacing():
    with pytest.raises(ConfigError, match="requires realtime"):
        StackConfig(router_async=True).validate()


class SlowRouter(FakeRouter):
    def submit(self, request):
        self.started = time.perf_counter()
        time.sleep(0.09)
        response = super().submit(request)
        response.wall_latency_s = time.perf_counter() - self.started
        self.finished = time.perf_counter()
        return response


def test_control_continues_during_actual_inference_and_logs_all_layers():
    class ObservedEnv(FakeEnv):
        def step(self, action):
            if hasattr(router, "started") and not hasattr(router, "finished"):
                stepped_while_thinking.append(self.t)
            return super().step(action)

    router = SlowRouter(
        skill="OVERTAKE_LEFT",
        latency_s=0.02,
        live=True,
        model="test-model",
        usage={"total_tokens": 7},
    )
    stepped_while_thinking = []
    cfg = StackConfig(
        router_async=True,
        realtime=True,
        record_interactions=True,
        pending_margin_on_all_calls=True,
        max_pending_s=0.4,
    )
    result = DrivingStack(skills(), router, config=cfg).run(ObservedEnv(duration=0.32))
    assert len(stepped_while_thinking) >= 2
    assert result.calls_applied == 1
    assert result.router_tokens == 7
    assert result.router_live_calls == 1
    assert result.calls[0]["latency_s"] >= 0.09
    assert result.calls[0]["to"] == "OVERTAKE_LEFT"
    assert "OVERTAKE_LEFT" in result.skills
    assert len(result.interactions) == len(result.actions)
    pending = [row for row in result.interactions if row["router_pending"]]
    assert len(pending) >= 3
    assert all(row["inflation"] >= 3 for row in pending)
    assert any(0 < row["blend_alpha"] < 1 for row in result.interactions)


def test_late_background_answer_cannot_switch_skill_or_queue_another_request():
    router = SlowRouter(skill="OVERTAKE_LEFT", latency_s=0.01)
    cfg = StackConfig(
        router_async=True,
        realtime=True,
        max_pending_s=0.04,
        min_call_ticks=100,
        record_interactions=True,
    )
    result = DrivingStack(skills(), router, config=cfg).run(FakeEnv(duration=0.16))
    assert result.calls_rejected == 1
    assert result.calls_applied == 0
    assert set(result.skills) == {"CRUISE_RIGHT"}
    assert len(router.requests) == 1


def test_response_after_episode_end_is_accounted_for_but_never_applied():
    router = SlowRouter(skill="OVERTAKE_LEFT", latency_s=0.01, live=True,
                        model="test-model", usage={"total_tokens": 13})
    cfg = StackConfig(router_async=True, realtime=True, max_pending_s=0.4)
    result = DrivingStack(skills(), router, config=cfg).run(FakeEnv(duration=0.04))
    assert result.calls_applied == 0
    assert result.router_live_calls == 1 and result.router_tokens == 13
    assert result.calls[0]["aborted"] and not result.calls[0]["applied"]
    assert result.end_time == pytest.approx(0.04)


def test_switch_rechecks_current_traffic_and_control_still_advances():
    class ChangedTraffic(FakeEnv):
        def snapshot(self):
            snap = super().snapshot()
            # Target lane was clear when requested, occupied at answer delivery.
            if self.t >= 0.04:
                snap.others = [VehicleState(snap.ego.x + 2, 4.0, 10.0)]
            return snap

    cfg = StackConfig(skill_switch_guard=True, record_interactions=True, min_call_ticks=100)
    router = FakeRouter(skill="OVERTAKE_LEFT", latency_s=0.08)
    result = DrivingStack(skills(), router, config=cfg).run(ChangedTraffic(duration=0.3))
    assert result.calls_rejected == 1
    assert result.calls_applied == 0
    assert set(result.skills) == {"CRUISE_RIGHT"}
    assert result.end_time == pytest.approx(0.3)
    assert len(result.actions) == 15


@pytest.mark.parametrize("mode", ["fixed_rate_shielded", "always_on_shielded"])
def test_matched_polling_keeps_uncertainty_margin_shield_and_blending(mode):
    class ObstacleEnv(FakeEnv):
        def snapshot(self):
            snap = super().snapshot()
            snap.others = [VehicleState(snap.ego.x + 12.0, 0.0, 0.0)]
            return snap

    cfg = StackConfig(mode=mode, record_interactions=True, pending_margin_on_all_calls=True)
    env = ObstacleEnv(duration=0.5)
    router = FakeRouter(skill="OVERTAKE_LEFT", latency_s=0.08)
    stack = DrivingStack(skills(), router, InflatedCBFShield(), cfg)
    result = stack.run(env)
    assert stack.last_shield.enabled and stack.last_shield.inflate
    assert result.override_fraction > 0
    assert any(0 < row["blend_alpha"] < 1 for row in result.interactions)
    assert any(row["inflation"] > 5 for row in result.interactions)
    assert np.min(np.asarray(result.actions)[:, 0]) < 0


def test_guard_includes_fast_approaching_rear_vehicle():
    stack = DrivingStack(skills(), FakeRouter())
    snap = FakeEnv().snapshot()
    snap.others = [VehicleState(snap.ego.x - 20, 4.0, 25)]
    assert not stack._switch_clear("OVERTAKE_LEFT", snap)
    safe = copy.deepcopy(snap)
    safe.others[0].vx = 5
    assert stack._switch_clear("OVERTAKE_LEFT", safe)


def test_safety_abort_does_not_record_an_unapplied_action():
    from event_cbf_driving.config import ScenarioConfig
    from event_cbf_driving.runtime import Episode, replay_open_loop, validate_episode

    from tests.test_controller import ScriptedShield

    cfg = StackConfig(abort_on_emergency=True, record_interactions=True)
    result = DrivingStack(skills(), FakeRouter(), ScriptedShield(invalid=True), cfg).run(FakeEnv())
    assert result.end_reason == "safety_abort"
    assert len(result.actions) * 0.02 == pytest.approx(result.end_time)
    episode = Episode(result, ScenarioConfig("two_trap"))
    assert replay_open_loop(episode, env=FakeEnv()).ok
    assert not validate_episode(episode.to_payload())
    corrupt = episode.to_payload()
    corrupt["interactions"][0]["requested_action"][0] = float("nan")
    assert any("finite two-component" in p for p in validate_episode(corrupt))


def test_road_edge_filter_brakes_lateral_motion_and_fails_closed_on_missing_geometry():
    from event_cbf_driving.config import ShieldConfig

    shield = InflatedCBFShield(config=ShieldConfig(road_boundary_guard=True))
    snap = FakeEnv().snapshot()
    snap.info.update(road_bounds=[-2.0, 6.0], heading=0.0)
    snap.ego.y = -0.5
    snap.ego.vy = -1.5
    result = shield.filter(np.array([0.0, 0.0]), snap)
    assert result.action[1] > 2
    assert result.lateral_overridden and result.overridden
    assert result.lateral_h > 0
    del snap.info["road_bounds"]
    invalid = shield.filter(np.array([0.0, 0.0]), snap)
    assert invalid.emergency and invalid.action[0] < 0


def test_scenario_simulator_typo_rejected():
    from event_cbf_driving.config import ScenarioConfig

    with pytest.raises(ConfigError, match="simulator"):
        ScenarioConfig(simulator="metdrive").validate()
