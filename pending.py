"""Explicit in-flight router-call state.

The simulator advances in discrete ticks while a router call owns a wall-clock
latency. A ``PendingCall`` records both clocks and its deadline so the
controller can:

* keep the delay margin active for the whole call, even when the answer is
  "stay in the current skill";
* reject answers that arrive after their deadline;
* report how long the stack was locked into the previous skill.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Optional

from router_base import RouterRequest, RouterResponse


@dataclass
class PendingCall:
    request: RouterRequest
    response: RouterResponse
    submitted_t: float
    deadline_t: float
    applied: bool = False
    expired: bool = False
    record: dict = field(default_factory=dict)

    @property
    def request_id(self) -> str:
        return self.request.request_id or self.response.request_id

    @property
    def ready_at(self) -> float:
        return float(self.response.ready_at)

    def is_ready(self, t: float) -> bool:
        return t + 1e-12 >= self.ready_at

    def is_expired(self, t: float) -> bool:
        return t > self.deadline_t + 1e-12

    def age_s(self, t: float) -> float:
        return max(0.0, min(float(t), self.ready_at) - self.submitted_t)

    def deliver(self, current_skill: str) -> Optional[str]:
        """Return the new skill if the answer changes it, else ``None``."""

        self.applied = True
        if self.response.skill == current_skill:
            return None
        return self.response.skill


@dataclass
class FuturePendingCall:
    """A real in-flight request; control never waits on ``Future.result()``.

    The controller polls done() first and validates a completed response before
    setting it here. Unfinished/late work cannot switch the active skill.
    """

    request: RouterRequest
    future: Future
    submitted_t: float
    deadline_t: float
    record: dict = field(default_factory=dict)
    response: Optional[RouterResponse] = None
    applied: bool = False
    expired: bool = False

    @property
    def request_id(self):
        return self.request.request_id

    @property
    def ready_at(self):
        return self.response.ready_at if self.response is not None else self.deadline_t

    def is_ready(self, t):
        return self.response is not None and t + 1e-12 >= self.ready_at

    def is_expired(self, t):
        return t > self.deadline_t + 1e-12

    def age_s(self, t):
        return max(0.0, float(t) - self.submitted_t)

    def deliver(self, current_skill):
        if self.response is None:
            raise RuntimeError("an unfinished router call cannot be delivered")
        self.applied = True
        return None if self.response.skill == current_skill else self.response.skill
