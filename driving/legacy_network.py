"""Network and input/action conversion for the recorded dual-critic policies."""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from torch.distributions import Normal

from driving.env_types import AX_HIGH, AX_LOW, AY_HIGH, AY_LOW

DEVICE = torch.device("cpu")
TARGET_SPEED = 20.0


def encode(y: ArrayLike, vy: ArrayLike, vx: ArrayLike, y_target: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    vy = np.asarray(vy, dtype=np.float32)
    vx = np.asarray(vx, dtype=np.float32)
    return np.stack(
        [
            (y - np.float32(y_target)) / np.float32(2.0),
            vy / np.float32(2.0),
            (vx - np.float32(20.0)) / np.float32(5.0),
        ],
        axis=-1,
    ).astype(np.float32)


def env_action_np(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map tanh actions with 0 at the origin, then clip to the accel box."""
    ax = np.clip(6.0 * a[..., 0], AX_LOW, AX_HIGH)
    ay = np.clip(5.0 * a[..., 1], AY_LOW, AY_HIGH)
    return ax, ay


def _mlp(in_dim, hidden, out_dim, out_gain):
    net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.414)
            nn.init.zeros_(m.bias)
    output_layer = cast(nn.Linear, net[-1])
    nn.init.orthogonal_(output_layer.weight, gain=out_gain)
    nn.init.zeros_(output_layer.bias)
    return net


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, 2)
        self.k_ax = nn.Parameter(torch.zeros(1))
        self.k_e = nn.Parameter(torch.zeros(1))
        self.k_v = nn.Parameter(torch.zeros(1))
        self.log_std = nn.Parameter(torch.full((2,), -1.6))
        self.v1 = _mlp(3, hidden, 1, 1.0)
        self.v2 = _mlp(3, hidden, 1, 1.0)
        for m in (self.fc1, self.fc2):
            nn.init.orthogonal_(m.weight, gain=1.414)
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)

    def _std(self):
        return self.log_std.clamp(-2.8, -1.2).exp()

    def _mu(self, obs):
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        base = self.mu(h)
        ax = base[:, 0] + self.k_ax * obs[:, 2]
        ay = base[:, 1] + self.k_e * obs[:, 0] + self.k_v * obs[:, 1]
        return torch.stack([ax, ay], dim=-1)

    def _dist(self, obs):
        return Normal(self._mu(obs), self._std())

    def values(self, obs):
        return self.v1(obs).squeeze(-1), self.v2(obs).squeeze(-1)

    def act(self, obs, deterministic=False):
        dist = self._dist(obs)
        u = dist.mean if deterministic else dist.sample()
        a = torch.tanh(u)
        logp = dist.log_prob(u) - torch.log(1.0 - a.pow(2) + 1e-6)
        v1, v2 = self.values(obs)
        return a, logp.sum(-1), v1, v2

    def log_prob(self, obs, a):
        dist = self._dist(obs)
        a = a.clamp(-0.999999, 0.999999)
        u = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        logp = dist.log_prob(u) - torch.log(1.0 - a.pow(2) + 1e-6)
        return logp.sum(-1), dist.entropy().sum(-1)


def pi_parameters(net: ActorCritic):
    return (
        list(net.fc1.parameters())
        + list(net.fc2.parameters())
        + list(net.mu.parameters())
        + [net.k_ax, net.k_e, net.k_v, net.log_std]
    )


def v_parameters(net: ActorCritic):
    return list(net.v1.parameters()) + list(net.v2.parameters())
