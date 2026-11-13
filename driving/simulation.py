"""Vectorized empty-road practice: reset, movement, and scoring."""

from __future__ import annotations

import numpy as np

from driving import DT
from driving.env_types import AX_HIGH, AX_LOW, AY_HIGH, AY_LOW
from driving.network import encode


class EmptyRoadVec:
    def __init__(self, n: int, y_target: float, rng: np.random.Generator):
        self.n = n
        self.yt = float(y_target)
        self.rng = rng
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        self.steps = np.zeros(n, np.int32)
        self.ep_ret = np.zeros(n)
        self.ep_steps = int(8.0 / DT)
        self.reset_mask(np.ones(n, dtype=bool))

    def obs(self):
        return encode(self.y, self.vy, self.vx, self.yt)

    def reset_mask(self, mask):
        idx = np.flatnonzero(mask)
        k = idx.size
        if k == 0:
            return
        draw = self.rng.random(k)
        y = np.empty(k)
        vx = np.empty(k)
        near = draw < 0.55
        slow = (draw >= 0.55) & (draw < 0.80)
        wide = ~near & ~slow
        y[near] = self.yt + self.rng.uniform(-1.5, 1.5, int(near.sum()))
        vx[near] = self.rng.uniform(12.0, 22.0, int(near.sum()))
        y[slow] = self.yt + self.rng.uniform(-1.0, 1.0, int(slow.sum()))
        vx[slow] = self.rng.uniform(0.0, 8.0, int(slow.sum()))
        y[wide] = self.yt + self.rng.uniform(-4.5, 4.5, int(wide.sum()))
        vx[wide] = self.rng.uniform(12.0, 22.0, int(wide.sum()))
        self.y[idx] = y
        self.vy[idx] = self.rng.uniform(-1.5, 1.5, k)
        self.vx[idx] = vx
        self.x[idx] = 0.0
        self.steps[idx] = 0
        self.ep_ret[idx] = 0.0

    def step(self, ax, ay):
        ax = np.clip(np.asarray(ax, np.float64), AX_LOW, AX_HIGH)
        ay = np.clip(np.asarray(ay, np.float64), AY_LOW, AY_HIGH)
        self.x = self.x + self.vx * DT + 0.5 * ax * DT * DT
        self.y = self.y + self.vy * DT + 0.5 * ay * DT * DT
        self.vx = np.maximum(0.0, self.vx + ax * DT)
        self.vy = self.vy + ay * DT
        self.steps += 1
        err = self.y - self.yt
        rew = (
            1.0
            - err**2
            - 0.05 * (self.vx - 20.0) ** 2
            - 0.02 * ax**2
            - 0.35 * ay**2
            - 0.05 * self.vy**2
        )
        fail = np.abs(err) > 7.0
        timeout = self.steps >= self.ep_steps
        done = fail | timeout
        rew = np.where(fail, rew - 5.0, rew)
        # Bootstrap features from the pre-reset terminal state (matches validated toy).
        boot = encode(self.y, self.vy, self.vx, self.yt)
        self.ep_ret += rew
        finished = self.ep_ret[done].copy()
        self.reset_mask(done)
        return (
            self.obs(),
            rew.astype(np.float32),
            done.astype(np.float32),
            fail.astype(np.float32),
            boot,
            finished,
        )
