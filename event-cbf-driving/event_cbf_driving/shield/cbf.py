"""Latency- and uncertainty-inflated CBF shield at control rate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from event_cbf_driving import DT
from event_cbf_driving.env.types import AX_HIGH, AX_LOW, VehicleState, WorldSnapshot


@dataclass
class ShieldResult:
    action: np.ndarray
    residual: float  # how much ax was reduced (override magnitude)
    overridden: bool
    inflation: float
    h: float


class InflatedCBFShield:
    """Longitudinal CBF on obstacles ahead in the lane corridor.

    h = (x_obs - x - standoff) - headway*(vx - v_obs) - inflation
    inflation = gamma * sigma + eta * I(VLM pending) * v * delta_t_vlm

    Class-K: require ḣ + α h >= 0 with α = class_k, solved for ax upper bound.
    """

    def __init__(
        self,
        standoff: float = 6.0,
        headway_s: float = 1.1,
        class_k: float = 2.5,
        lane_half_width: float = 1.85,
        gamma_sigma: float = 2.8,  # proposed inflate path only (static uses inflate=False)
        eta_latency: float = 1.0,
        vlm_delay_s: float = 0.30,
        ax_min: float = -7.5,
        ax_max: float = AX_HIGH,
        enabled: bool = True,
        inflate: bool = True,
    ):
        self.standoff = standoff
        self.headway_s = headway_s
        self.class_k = class_k
        self.lane_half_width = lane_half_width
        self.gamma_sigma = gamma_sigma
        self.eta_latency = eta_latency
        self.vlm_delay_s = vlm_delay_s
        self.ax_min = ax_min
        self.ax_max = ax_max
        self.enabled = enabled
        self.inflate = inflate

    def inflation(self, sigma: float, vlm_pending: bool, vx: float) -> float:
        if not self.inflate:
            return 0.0
        # Match toy: 4σ + 0.30*vx while VLM pending. Cap σ-term so a brief
        # OOD critic spike cannot demand an unreachable standoff.
        lat = self.eta_latency * (1.0 if vlm_pending else 0.0) * max(vx, 0.0) * self.vlm_delay_s
        sig = min(self.gamma_sigma * max(sigma, 0.0), 10.0)
        return sig + lat

    def filter(
        self,
        action: np.ndarray,
        snap: WorldSnapshot,
        sigma: float = 0.0,
        vlm_pending: bool = False,
    ) -> ShieldResult:
        ax_rl, ay = float(action[0]), float(action[1])
        if not self.enabled:
            return ShieldResult(
                action=np.array([ax_rl, ay], dtype=np.float32),
                residual=0.0,
                overridden=False,
                inflation=0.0,
                h=1e9,
            )
        ego = snap.ego
        infl = self.inflation(sigma, vlm_pending, ego.vx)
        ax_cap = self.ax_max
        h_min = 1e9
        for obs in snap.obstacles_ahead(self.lane_half_width):
            dx = obs.x - ego.x - self.standoff
            h = dx - self.headway_s * (ego.vx - obs.vx) - infl
            h_min = min(h_min, h)
            # ḣ ≈ (v_obs - vx) - headway*(ax - 0) ; require ḣ + α h >= 0
            # (v_obs - vx) - headway * ax + α h >= 0
            # -headway * ax >= - (v_obs - vx) - α h
            # ax <= ((v_obs - vx) + α h) / headway
            ax_safe = ((obs.vx - ego.vx) + self.class_k * h) / self.headway_s
            ax_cap = min(ax_cap, ax_safe)
        ax = float(np.clip(min(ax_rl, ax_cap), self.ax_min, self.ax_max))
        residual = max(0.0, ax_rl - ax)
        return ShieldResult(
            action=np.array([ax, ay], dtype=np.float32),
            residual=residual,
            overridden=ax < ax_rl - 1e-3,
            inflation=infl,
            h=float(h_min),
        )
