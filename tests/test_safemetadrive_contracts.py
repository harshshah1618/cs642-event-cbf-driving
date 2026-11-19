"""Regression checks for geometry, delayed intent and baseline inference."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from driving.safemetadrive.camera_planner import PendingIntent, reflect_example
from driving.safemetadrive.policies import LaneCautionDriver, NativeLearnedDriver
from driving.safemetadrive.protocol import SPLITS
from driving.safemetadrive.training_state import (
    checkpoint_location,
    publish_checkpoint,
    sampling_weights,
)


def test_training_development_and_test_maps_do_not_overlap():
    partitions = [set(range(start, start+count)) for start, count in SPLITS.values()]
    assert all(not left & right for i, left in enumerate(partitions) for right in partitions[i+1:])


def test_reflected_camera_example_exchanges_lane_intent_and_preserves_caution():
    row = dict(label="E", current_lane=0, target_lane=1, lane_count=3, speed_mps=5)
    reflected = reflect_example(row)
    assert reflected == dict(label="F", current_lane=2, target_lane=1, lane_count=3, speed_mps=5)
    assert reflect_example(reflected) == row


def test_balancing_gives_rare_maneuvers_equal_training_probability():
    labels = ["A"]*1000+["D"]*500+["B"]*2+["C"]*3+["E"]+["F"]
    weights = sampling_weights(labels, "balanced")
    mass = {label: sum(w for y,w in zip(labels,weights,strict=True) if y == label) for label in set(labels)}
    assert all(value == pytest.approx(1.0) for value in mass.values())


def test_interrupted_checkpoint_does_not_replace_last_complete_state(tmp_path):
    def save(directory):
        (directory/"adapter").mkdir()
        (directory/"optimizer.pt").write_bytes(b"complete")
    first = publish_checkpoint(tmp_path, 50, save)
    def interrupted(directory):
        (directory/"adapter").mkdir()
        raise RuntimeError("simulated interruption")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        publish_checkpoint(tmp_path, 100, interrupted)
    assert checkpoint_location(tmp_path) == (first/"adapter", first/"optimizer.pt")
    assert not list((tmp_path/"checkpoints").glob(".writing-*"))


def test_incomplete_checkpoint_cannot_be_published(tmp_path):
    with pytest.raises(ValueError, match="both adapter and optimizer"):
        publish_checkpoint(tmp_path, 50, lambda directory: (directory/"adapter").mkdir())
    assert checkpoint_location(tmp_path) is None


def test_complete_checkpoint_survives_interruption_before_pointer_publication(tmp_path, monkeypatch):
    from pathlib import Path

    def save(directory):
        (directory/"adapter").mkdir()
        (directory/"optimizer.pt").write_bytes(b"complete")
    publish_checkpoint(tmp_path, 50, save)
    def interrupted_replace(self, target):
        raise RuntimeError("interrupted before pointer update")
    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(RuntimeError, match="before pointer update"):
        publish_checkpoint(tmp_path, 100, save)
    adapter, optimizer = checkpoint_location(tmp_path)
    assert adapter.parent.name == "step-000100"
    assert optimizer.is_file()
    assert not list(tmp_path.glob("latest-*.tmp"))


def test_delayed_relative_intent_cannot_retarget_after_another_lane_change():
    first, second = object(), object()
    driver = object.__new__(LaneCautionDriver)
    driver.vehicle = SimpleNamespace(navigation=SimpleNamespace(current_ref_lanes=[first, second]))
    driver.lane, driver.careful = first, False
    response = PendingIntent("D", captured_at=1, delay=2, lane=first)
    assert response.apply(2, driver) == "pending"
    driver.lane = second
    assert response.apply(3, driver) == "rejected"
    assert not driver.careful


def test_expired_caution_is_not_applied_even_on_same_lane():
    lane = object()
    response = PendingIntent("D", 0, 5, lane, ttl=4.5)
    assert response.apply(5, SimpleNamespace()) == "expired"
    with pytest.raises(ValueError):
        PendingIntent("A", 0, math.nan, lane)


def test_footprint_prediction_uses_metadrive_right_positive_lateral_axis():
    driver = object.__new__(LaneCautionDriver)
    driver.vehicle = SimpleNamespace(position=np.array([0., -3.5]), speed=2,
                                    heading_theta=0, LENGTH=4.5, WIDTH=1.8)
    driver.lane = SimpleNamespace(local_coordinates=lambda _: (0, 3.5))
    driver.path_point = lambda distance: (np.array([distance, 0.]), 0.)
    # A sign error predicts the ego on the opposite side of its target lane,
    # causing a false emergency stop and permanent stall beside this obstacle.
    cone = SimpleNamespace(position=np.array([2., 3.5]), velocity=np.zeros(2),
                           heading_theta=0, LENGTH=0.3, WIDTH=0.3)
    assert math.isinf(driver.conflict_time([cone]))


def test_prediction_detects_crossing_traffic_outside_current_lane():
    driver = object.__new__(LaneCautionDriver)
    driver.vehicle = SimpleNamespace(position=np.array([0., 0.]), speed=3,
                                    heading_theta=0, LENGTH=4.5, WIDTH=1.8)
    driver.lane = SimpleNamespace(local_coordinates=lambda _: (0, 0))
    driver.path_point = lambda distance: (np.array([distance, 0.]), 0.)
    car = SimpleNamespace(position=np.array([5., 5.]), velocity=np.array([0., -2.]),
                          heading_theta=-math.pi/2, LENGTH=4.5, WIDTH=1.8)
    assert driver.conflict_time([car]) < 3


def test_native_baseline_loader_preserves_deterministic_actor_output(tmp_path):
    actor = torch.nn.Sequential(torch.nn.Linear(7, 16), torch.nn.Tanh(),
                                torch.nn.Linear(16, 8), torch.nn.Tanh(),
                                torch.nn.Linear(8, 2), torch.nn.Identity())
    pi = {f"mean.{name}": value for name, value in actor.state_dict().items()}
    pi["log_std"] = torch.zeros(2)
    checkpoint = tmp_path / "actor.pt"
    torch.save({"pi": pi}, checkpoint)
    observation = np.linspace(0, 1, 7, dtype=np.float32)
    driver = NativeLearnedDriver(checkpoint, observation)
    with torch.inference_mode():
        expected = actor(torch.from_numpy(observation)).numpy()
    np.testing.assert_allclose(driver.act(), expected, rtol=1e-6, atol=1e-7)
    left = NativeLearnedDriver(checkpoint, observation, deterministic=False, policy_seed=123)
    right = NativeLearnedDriver(checkpoint, observation, deterministic=False, policy_seed=123)
    first = left.act()
    np.testing.assert_array_equal(first, right.act())
    assert not np.allclose(first, expected)
    assert not np.allclose(left.act(), first)
