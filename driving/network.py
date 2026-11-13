"""Single-critic agent adapted from CleanRL's continuous-action PPO.

Upstream: vwxyzjn/cleanrl@35896b1fefa9898b904f7e09bcbe6e168e15d2a9
See driving/CLEANRL_LICENSE.txt. Observation/action sizes are fixed for lane driving.
"""

from typing import Tuple, cast

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn
from torch.distributions import Normal

from driving.env_types import AX_HIGH, AX_LOW, AY_HIGH, AY_LOW


def layer_init(layer, std=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class CleanRLAgent(nn.Module):
    """A driver and one score estimator; no critic-disagreement signal."""

    supports_uncertainty = False

    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(3, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(3, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        distribution = Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(-1),
            distribution.entropy().sum(-1),
            self.get_value(obs),
        )

    def act(self, obs, deterministic=False):
        """Existing driver interface. Repeated value is not a second critic."""
        action = self.actor_mean(obs) if deterministic else None
        action, logprob, _, value = self.get_action_and_value(obs, action)
        return action, logprob, value, value


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
