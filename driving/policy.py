"""Small runtime interface used by the driving stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from driving import DT
from driving.network import ActorCritic, CleanRLAgent, encode, env_action_np


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


@torch.no_grad()
def evaluate_skill(net: ActorCritic | CleanRLAgent, y_target: float) -> Dict:
    def rollout(y0, vy0, vx0, steps=500):
        y, vy, vx, x = float(y0), float(vy0), float(vx0), 0.0
        ys, vxs = [], []
        t16 = None
        other = 0.0 if y_target > 2.0 else 4.0
        for i in range(steps):
            o = torch.from_numpy(encode(y, vy, vx, y_target)).view(1, 3)
            a, _, _, _ = net.act(o, deterministic=True)
            ax, ay = env_action_np(a.numpy()[0])
            x = x + vx * DT + 0.5 * float(ax) * DT * DT
            y = y + vy * DT + 0.5 * float(ay) * DT * DT
            vx = max(0.0, vx + float(ax) * DT)
            vy = vy + float(ay) * DT
            ys.append(y)
            vxs.append(vx)
            if t16 is None and abs(y - other) >= 1.6:
                t16 = (i + 1) * DT
        ys_array = np.asarray(ys)
        return {
            "mean_abs_y": float(np.mean(np.abs(ys_array - y_target))),
            "mean_abs_v": float(np.mean(np.abs(np.asarray(vxs) - 20.0))),
            "final_y": float(ys_array[-1]),
            "final_vx": float(vxs[-1]),
            "t16": t16,
        }

    other = 0.0 if y_target > 2.0 else 4.0
    hold = rollout(y_target, 0.0, 20.0)
    acq = rollout(other, 0.0, 20.0)
    # Probe lateral authority from the other lane.
    o = torch.from_numpy(encode(other, 0.0, 20.0, y_target)).view(1, 3)
    a, _, _, _ = net.act(o, deterministic=True)
    _, ay_other = env_action_np(a.numpy()[0])
    learned = (
        hold["mean_abs_y"] < 0.30
        and hold["mean_abs_v"] < 1.0
        and abs(hold["final_y"] - y_target) < 0.35
        and 18.5 <= hold["final_vx"] <= 21.5
        and abs(acq["final_y"] - y_target) < 0.5
        and (acq["t16"] is not None and 1.0 <= acq["t16"] <= 2.4)
        and abs(float(ay_other)) >= 1.0
    )
    return {"hold": hold, "acquire": acq, "learned": learned, "ay_at_other": float(ay_other)}
