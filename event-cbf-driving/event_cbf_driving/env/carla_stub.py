"""Optional CARLA adapter stub.

This project CI/evals use the highway-compatible kinematics envs. CARLA is
documented here so CoDrivingLLM / CoLMDriver integration can plug in later
without changing the Tier-1/2/3 stack interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CarlaAdapterStub:
    """Placeholder for a CARLA client that exposes snapshot() / step(ax, ay)."""

    def __init__(self, host: str = "localhost", port: int = 2000, **kwargs: Any):
        self.host = host
        self.port = port
        self.kwargs = kwargs
        self._connected = False

    def connect(self) -> None:
        raise NotImplementedError(
            "CARLA adapter is a stub. Install CARLA and implement transform "
            "from CARLA actor state -> WorldSnapshot and ax/ay -> vehicle control. "
            "See docs/event-cbf-driving/PROJECT.md."
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.connect()

    def step(self, action):
        self.connect()

    def snapshot(self):
        self.connect()
