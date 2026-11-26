"""Contracts for delayed advice, stop leases and whole-episode collision telemetry."""

from types import SimpleNamespace

import pytest
from driving.irregular.escalation import Escalation, EscalationConfig, run_episode
from tools.irregular_closedloop_report import summarise


class FakeDriver:
    def __init__(self, vehicle, **kwargs):
        self.vehicle = vehicle
        self.hold = False
        self.careful = False
        self.planner = "rule"
        self.interventions = 0
        self.closed = False

    def synchronize_lane(self):
        pass

    def act(self):
        return [0., 0.]

    def close(self):
        self.closed = True


class FakeEnv:
    def __init__(self, rows):
        self.rows = rows
        self.agent = SimpleNamespace(speed=1.)

    def reset(self, seed):
        self.index = 0

    def step(self, action):
        info = dict(cost=0., crash_vehicle=False, crash_object=False, crash_human=False,
                    route_completion=.1)
        info.update(self.rows[self.index])
        self.index += 1
        return None, 0., self.index == len(self.rows), False, info


@pytest.mark.parametrize("kind", ["crash_vehicle", "crash_object", "crash_human"])
def test_collision_before_final_step_survives_in_episode_summary(monkeypatch, kind):
    monkeypatch.setattr("driving.irregular.escalation.EscalatingDriver", FakeDriver)
    env = FakeEnv([{kind: True, "cost": 1.}, {}])
    row = run_episode(env, 1000, [], EscalationConfig(), max_steps=2)
    assert row[kind] is True
    assert row["crash_steps"][kind] == 1
    assert row["collision_events"] == [dict(step=1, seconds=.1, kinds=[kind])]
    assert row["cost"] == 1.
    assert row["end_reason"] == "terminated"
    assert row["metrics_schema"] == "irregular-episode-v2"
    assert summarise([row])["collision_episodes"] == 1


def test_contact_steps_do_not_become_multiple_collision_episodes(monkeypatch):
    monkeypatch.setattr("driving.irregular.escalation.EscalatingDriver", FakeDriver)
    row = run_episode(FakeEnv([dict(crash_object=True)] * 2 + [{}]), 1000, [],
                      EscalationConfig(), max_steps=3)
    assert row["crash_steps"]["crash_object"] == 2
    assert summarise([row])["collision_episodes"] == 1


def test_zero_native_cost_does_not_hide_human_collision(monkeypatch):
    monkeypatch.setattr("driving.irregular.escalation.EscalatingDriver", FakeDriver)
    row = run_episode(FakeEnv([dict(crash_human=True, cost=0.)]), 1000, [],
                      EscalationConfig(), max_steps=1)
    assert row["cost"] == 0.
    assert summarise([row])["collision_episodes"] == 1


def test_legacy_flags_cannot_establish_zero_collision_episodes(monkeypatch):
    monkeypatch.setattr("driving.irregular.escalation.EscalatingDriver", FakeDriver)
    row = run_episode(FakeEnv([{}]), 1000, [], EscalationConfig(), max_steps=1)
    del row["metrics_schema"]
    row["cost"] = 1.
    result = summarise([row])
    assert result["collision_episodes"] is None
    assert result["collision_verified_episodes"] == 0
    assert result["legacy_nonzero_cost_without_final_crash"] == 1


def test_answer_does_not_apply_before_its_simulated_arrival():
    driver = FakeDriver(SimpleNamespace(speed=2.))
    cfg = EscalationConfig(trigger="poll", cooldown=10)
    esc = Escalation(cfg, vlm=lambda *args: ("stop", .5))
    def capture():
        return None, {}
    esc.step(0., driver, None, capture, True)
    assert not driver.hold
    esc.step(.4, driver, None, capture, False)
    assert not driver.hold
    esc.step(.5, driver, None, capture, False)
    assert driver.hold


def test_stop_is_held_during_recheck_and_requires_two_release_answers():
    driver = FakeDriver(SimpleNamespace(speed=0.))
    answers = iter([("stop", .1), ("continue", .1), ("continue", .1)])
    cfg = EscalationConfig(trigger="poll", hold_recheck=1., cooldown=10)
    esc = Escalation(cfg, vlm=lambda *args: next(answers))
    def capture():
        return None, {}
    esc.step(0., driver, None, capture, True)
    esc.step(.1, driver, None, capture, False)
    assert driver.hold
    esc.step(1.2, driver, None, capture, False)
    assert driver.hold and esc.pending is not None
    esc.step(1.4, driver, None, capture, False)
    assert driver.hold and esc.calls[1]["applied"] == "confirming"
    esc.step(1.6, driver, None, capture, False)
    assert not driver.hold
