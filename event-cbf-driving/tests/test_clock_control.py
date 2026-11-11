"""Protect paired phase assignment and distinct resumable job identities."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
clock = importlib.import_module("research_clock_control")


def test_job_design_and_identity():
    jobs = clock.jobs()
    assert len(jobs) == len({j.key() for j in jobs}) == 1080
    for seed in range(1101, 1141):
        assert len([j for j in jobs if j.seed == seed]) == 27
    zero = clock.ClockJob("merge", 1101, "long", "zero", 0.1)
    uniform = clock.ClockJob("merge", 1101, "long", "uniform", 0.1)
    assert zero.key() != uniform.key()


def test_phase_assignment_is_paired_and_resets(monkeypatch):
    observed = []

    def fake_episode(*args, **kwargs):
        observed.append(kwargs["stack_config"].polling_phase_s)

    monkeypatch.setattr(clock, "ORIGINAL_RUN", fake_episode)

    def fake_execute(job):
        cfg = SimpleNamespace(pending_margin=False, pending_margin_on_all_calls=True,
                              fixed_rate_hz=job.value)
        clock.phase_run("fixed_rate_shielded", stack_config=cfg)
        return {"id": job.key()}

    monkeypatch.setattr(clock.base, "execute", fake_execute)
    for mode, rate in (("uniform", .1), ("uniform", 1), ("zero", .1)):
        clock.execute(clock.ClockJob("merge", 1101, "long", mode, rate))
    fraction = float(np.random.default_rng(1101 + 72001).uniform())
    assert observed == [fraction / .1, fraction, 0.0]
