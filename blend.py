"""Smooth skill crossfade and an optional advantage-aware gate.

The default controller uses :func:`sigmoid_blend_alpha` (a fixed, documented
crossfade). :func:`advantage_gate` is provided for callers that have critic
advantages for both skills; it nudges the crossfade earlier when the new skill
is clearly better and later when it is worse. It is deliberately bounded and
falls back to the base schedule when the advantages are equal.
"""

from __future__ import annotations

import numpy as np


def sigmoid_blend_alpha(tick: int, blend_ticks: int = 20) -> float:
    """Sigmoid crossfade weight for the NEW skill in [0, 1]."""

    if blend_ticks <= 1:
        return 1.0
    u = tick / (blend_ticks - 1)
    return float(1.0 / (1.0 + np.exp(-10.0 * (u - 0.5))))


def advantage_gate(
    alpha: float,
    adv_new: float,
    adv_old: float,
    temperature: float = 1.0,
) -> float:
    """Bend the crossfade schedule using the two skills' advantages.

    ``adv_new > adv_old`` moves the blend slightly earlier; the reverse moves
    it later. The correction is bounded by 25% of ``alpha*(1-alpha)``, so it
    can never invert the schedule or jump straight to the new skill.
    """

    alpha = float(np.clip(alpha, 0.0, 1.0))
    temp = max(float(temperature), 1e-6)
    delta = (float(adv_new) - float(adv_old)) / temp
    delta = float(np.clip(delta, -60.0, 60.0))
    gate = 1.0 / (1.0 + np.exp(-delta))
    correction = 0.5 * (gate - 0.5) * alpha * (1.0 - alpha)
    return float(np.clip(alpha + correction, 0.0, 1.0))


def blend_actions(a_old: np.ndarray, a_new: np.ndarray, alpha: float) -> np.ndarray:
    a = (1.0 - alpha) * np.asarray(a_old, dtype=np.float64) + alpha * np.asarray(
        a_new, dtype=np.float64
    )
    return a.astype(np.float32)
