import importlib
import sys
from pathlib import Path

from event_cbf_driving.env.types import VehicleState, WorldSnapshot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spatial = importlib.import_module("research_spatial_gate")


def test_farther_slower_adjacent_lead_can_offer_more_progress():
    snap = WorldSnapshot(0, VehicleState(0, 0, 20),
                         [VehicleState(35, 0, 9), VehicleState(90, 4, 7)], 200)
    assert not spatial.ORIGINAL(snap, "CRUISE_RIGHT").useful
    assert spatial.spatial_opportunity(snap, "CRUISE_RIGHT").useful


def test_blocked_adjacent_lane_is_not_useful():
    snap = WorldSnapshot(0, VehicleState(0, 0, 20),
                         [VehicleState(35, 0, 9), VehicleState(8, 4, 20)], 200)
    assert not spatial.spatial_opportunity(snap, "CRUISE_RIGHT").useful


def test_empty_lanes_and_router_noop_do_not_request():
    snap = WorldSnapshot(0, VehicleState(0, 0, 20), [], 200)
    assert not spatial.spatial_opportunity(snap, "CRUISE_RIGHT").useful
    snap.others = [VehicleState(25, 0, 16)]
    assert not spatial.spatial_opportunity(snap, "CRUISE_RIGHT").useful


def test_unsettled_maneuver_does_not_reverse():
    snap = WorldSnapshot(0, VehicleState(0, 2, 20), [VehicleState(35, 0, 9)], 200)
    assert not spatial.spatial_opportunity(snap, "CRUISE_RIGHT").useful


def test_process_local_injection_is_restored(monkeypatch):
    observed = []

    def fake_execute(job):
        observed.append((job.method, spatial.controller.lane_opportunity))
        return {"success": True}

    monkeypatch.setattr(spatial.base, "execute", fake_execute)
    new = spatial.base.Job("merge", 101, "long", "spatial_gate", 0)
    row = spatial.execute(new)
    assert row["id"] == new.key() and row["method"] == "spatial_gate"
    assert spatial.controller.lane_opportunity is spatial.ORIGINAL
    spatial.execute(spatial.base.Job("merge", 101, "long", "ttc", 0))
    assert observed == [("benefit", spatial.spatial_opportunity), ("ttc", spatial.ORIGINAL)]
