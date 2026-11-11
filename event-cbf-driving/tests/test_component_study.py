"""Protect experiment pairing and distinguish synthetic faults from rule labels."""

import importlib
import sys
from pathlib import Path

import pytest
from event_cbf_driving.env.types import VehicleState, WorldSnapshot
from event_cbf_driving.router.base import RouterRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
study = importlib.import_module("component_ablation")


def test_exact_factorial_design_and_disjoint_seeds():
    design = study.jobs("test")
    assert len(design) == len({j.key() for j in design}) == 4608
    assert set(study.TEST_SEEDS).isdisjoint(study.PILOT_SEEDS)
    for seed in study.TEST_SEEDS:
        assert len([j for j in design if j.seed == seed]) == 384


def test_scene_and_timer_are_paired_across_all_treatments():
    group = [j for j in study.jobs("test") if j.seed == 5001 and j.scene == "side_occupied"]
    assert len({study.object_hash(study.asdict(study.scene_spec(j))) for j in group}) == 1
    assert len({study.configs(j).stack.polling_phase_s for j in group}) == 1
    assert all(study.configs(j).shield.gamma_sigma == 0 for j in group)


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("state_rule", "CRUISE_RIGHT"),
        ("front_blind_rule", "OVERTAKE_LEFT"),
        ("copy_current", "CRUISE_RIGHT"),
    ],
)
def test_blindside_is_a_deliberate_information_fault(policy, expected):
    snap = WorldSnapshot(
        0,
        VehicleState(0, 0, 20),
        [
            VehicleState(40, 0, 9),
            VehicleState(0, 4, 20),
        ],
    )
    req = RouterRequest(0, "CRUISE_RIGHT", snap, 0, "fixed_rate")
    assert study.RuleRouter(policy, 0.125, 1).decide(req) == expected


def test_state_rule_returns_when_right_lane_clears():
    snap = WorldSnapshot(0, VehicleState(0, 4, 20), [])
    req = RouterRequest(0, "OVERTAKE_LEFT", snap, 0, "fixed_rate")
    assert study.RuleRouter("state_rule", 0.125, 1).decide(req) == "RETURN_RIGHT"


def test_resume_rejects_mismatched_cell_metadata(tmp_path):
    job = study.jobs("pilot")[0]
    row = {"id": job.key(), **study.asdict(job), "cbf": not job.cbf}
    target = tmp_path / "rows.jsonl"
    target.write_text(study.json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="metadata"):
        study.read_rows(target, [job])
