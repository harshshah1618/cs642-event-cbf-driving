"""Fixed PD lane controller used as an explicit nonlearned comparator."""

from __future__ import annotations

import numpy as np

from event_cbf_driving.config import SKILL_NAMES
from event_cbf_driving.skills.ppo import SkillMeta


class PDLanePolicy:
    def __init__(self, name: str):
        if name not in SKILL_NAMES:
            raise ValueError(f"unknown skill {name!r}")
        self.name = name
        y = 4.0 if name == "OVERTAKE_LEFT" else 0.0
        self.meta = SkillMeta(name, int(y > 0), y, 1e6, 0.0, 0.0, 1.0)

    def act(self, y, vy, vx, deterministic=True):
        return (
            float(np.clip(1.2 * (20.0 - vx), -7.5, 3.5)),
            float(np.clip(1.6 * (self.meta.y_target - y) - 2.5 * vy, -5.0, 5.0)),
            0.0,
        )
