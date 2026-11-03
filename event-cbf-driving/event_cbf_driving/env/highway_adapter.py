"""Optional highway-env (Leurent) adapters.

Primary scenarios (`two_trap`, `merge`, `dense_highway`) share a 50 Hz
gymnasium API with continuous [ax, ay] actions so the stack, shield, and
skills stay simulator-agnostic. Use these helpers when you want the upstream
MultiAgentRoad dynamics directly.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def configure_highway_continuous(env, *, lanes: int = 2, vehicles: int = 15, duration: int = 30):
    """Configure an existing highway-env env for ContinuousAction @ 50 Hz."""
    env.unwrapped.configure(
        {
            "observation": {"type": "Kinematics", "vehicles_count": 5},
            "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
            "lanes_count": lanes,
            "vehicles_count": vehicles,
            "duration": duration,
            "policy_frequency": 50,
            "simulation_frequency": 50,
        }
    )
    return env


def make_highway_v0(**kwargs: Any):
    import gymnasium as gym

    env = gym.make("highway-v0", render_mode=None)
    return configure_highway_continuous(env, **kwargs)


def make_merge_v0(**kwargs: Any):
    import gymnasium as gym

    env = gym.make("merge-v0", render_mode=None)
    return configure_highway_continuous(env, vehicles=kwargs.pop("vehicles", 8), **kwargs)


def accel_to_highway_action(ax: float, ay: float) -> np.ndarray:
    """Map our [ax, ay] box roughly into highway-env ContinuousAction [-1, 1]^2."""
    # highway ContinuousAction: [steering, acceleration] normalized.
    steer = float(np.clip(ay / 5.0, -1.0, 1.0))
    acc = float(np.clip(ax / 6.0, -1.0, 1.0))
    return np.array([steer, acc], dtype=np.float32)
