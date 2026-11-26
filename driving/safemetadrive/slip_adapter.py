"""Upgrade #4: Online Slip-History Friction Estimator for Unknown Variable-Friction Roads.

Watches an 8-step sliding window of strictly observation-only signals:
    [last_throttle, actual_decel, speed_mps, yaw_rate]
Whenever braking is commanded (throttle < -0.25) while moving (v > 2.5 m/s),
compares achieved deceleration against the dry-asphalt nominal braking rate
(~4.5 m/s^2 per unit brake) to infer road friction scale mu_hat in [0.22, 1.0]
within ~0.3 seconds—feeding mu_hat directly into UnifiedDrivingSystem's
Option-Viability Barrier and curve speed limits.
"""

from __future__ import annotations

from collections import deque
import numpy as np

from driving.safemetadrive.lidar_driver import decode_observation


class SlipHistoryFrictionEstimator:
    """Pure observation-only online friction estimator from an 8-step slip history."""

    def __init__(self, *, window: int = 8, dry_brake_mps2: float = 5.5, min_mu: float = 0.22):
        self.window = int(window)
        self.dry_brake_mps2 = float(dry_brake_mps2)
        self.min_mu = float(min_mu)
        self.prev_speed: float | None = None
        self.prev_throttle: float = 0.0
        self.ratios: deque[float] = deque(maxlen=window)
        self.mu_hat: float = 1.0

    def update(self, obs_259: np.ndarray, last_throttle: float) -> float:
        d = decode_observation(obs_259)
        if self.prev_speed is not None and last_throttle < -0.25 and self.prev_speed > 2.5:
            actual_decel = max(0.0, (self.prev_speed - d.speed_mps) / 0.1)
            expected_dry = self.dry_brake_mps2 * min(1.0, abs(last_throttle))
            if expected_dry > 0.5:
                ratio = float(np.clip(actual_decel / expected_dry, self.min_mu, 1.0))
                self.ratios.append(ratio)
                # Conservative 35th-percentile estimate over recent braking window
                est = float(np.percentile(list(self.ratios), 35))
                # Fast drop on slippery surface, slow recovery
                if est < self.mu_hat:
                    self.mu_hat = 0.45 * self.mu_hat + 0.55 * est
                else:
                    self.mu_hat = 0.90 * self.mu_hat + 0.10 * est
        self.prev_speed = d.speed_mps
        self.prev_throttle = float(last_throttle)
        return float(np.clip(self.mu_hat, self.min_mu, 1.0))
