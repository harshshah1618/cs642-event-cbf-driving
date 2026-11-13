"""Small runtime interface used by the driving stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch

from legacy_network import ActorCritic, encode, env_action_np
from network import CleanRLAgent


@dataclass
class SkillMeta:
    name: str
    target_lane: int
    y_target: float
    threshold: float
    sigma_mean: float
    sigma_std: float
    spread: float


class SkillPolicy:
    def __init__(
        self,
        name: str,
        net: ActorCritic | CleanRLAgent,
        meta: SkillMeta,
        checkpoint_info: Optional[Mapping[str, Any]] = None,
    ):
        self.name = name
        self.net = net
        self.meta = meta
        self.checkpoint_info: Dict[str, Any] = dict(checkpoint_info or {})
        self.source_path = self.checkpoint_info.get("path")
        self.net.eval()

    @property
    def supports_uncertainty(self) -> bool:
        return bool(getattr(self.net, "supports_uncertainty", True))

    def act(self, y: float, vy: float, vx: float, deterministic: bool = True):
        o = torch.from_numpy(encode(y, vy, vx, self.meta.y_target)).view(1, 3)
        with torch.no_grad():
            a, _, v1, v2 = self.net.act(o, deterministic=deterministic)
        ax, ay = env_action_np(a.numpy()[0])
        raw = abs(float(v1) - float(v2))
        sigma = raw / max(self.meta.spread, 1e-6)
        return float(ax), float(ay), float(sigma)

    def interrupt(self, sigma: float) -> bool:
        return sigma > self.meta.threshold
