"""Pluggable Tier-1 skill router interface.

Called ONLY on interrupt (critic σ above threshold OR sustained CBF residual).
Backends may be OpenAI-compatible, local transformers/ollama, or scripted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from event_cbf_driving.env.types import WorldSnapshot

SKILL_NAMES = ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")


@dataclass
class RouterRequest:
    t: float
    current_skill: str
    snapshot: WorldSnapshot
    sigma: float
    reason: str  # "sigma" | "shield" | "sigma+shield" | "fixed_rate"


@dataclass
class RouterResponse:
    skill: str
    ready_at: float  # simulation time when answer becomes available
    backend: str
    raw_text: str = ""
    latency_s: float = 0.30


class RouterBackend(ABC):
    name: str = "base"
    latency_s: float = 0.30

    @abstractmethod
    def submit(self, req: RouterRequest) -> RouterResponse:
        """Start a skill query. May return immediately with ready_at = t + latency."""

    def describe(self) -> str:
        return self.name
