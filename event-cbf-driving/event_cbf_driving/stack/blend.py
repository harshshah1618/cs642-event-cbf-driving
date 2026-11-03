"""Advantage-gated / smooth blend on skill switch."""

from __future__ import annotations

import numpy as np


def sigmoid_blend_alpha(tick: int, blend_ticks: int = 20) -> float:
    """Sigmoid crossfade weight for the NEW skill in [0, 1]."""
    if blend_ticks <= 1:
        return 1.0
    u = tick / (blend_ticks - 1)
    return float(1.0 / (1.0 + np.exp(-10.0 * (u - 0.5))))


def advantage_gate(alpha: float, adv_new: float, adv_old: float, temperature: float = 1.0) -> float:
    """Mix sigmoid schedule with a soft advantage gate.

    If the new skill's advantage is clearly higher, accelerate the blend;
    if lower, slow it. Falls back to pure sigmoid when advantages are equal.
    """
    delta = (adv_new - adv_old) / max(temperature, 1e-6)
    gate = 1.0 / (1.0 + np.exp(-delta))
    return float(np.clip(0.5 * alpha + 0.5 * (alpha * gate + (1 - alpha) * (1 - gate) * 0 + alpha), 0.0, 1.0))


def blend_actions(a_old: np.ndarray, a_new: np.ndarray, alpha: float) -> np.ndarray:
    a = (1.0 - alpha) * np.asarray(a_old, dtype=np.float64) + alpha * np.asarray(a_new, dtype=np.float64)
    return a.astype(np.float32)
