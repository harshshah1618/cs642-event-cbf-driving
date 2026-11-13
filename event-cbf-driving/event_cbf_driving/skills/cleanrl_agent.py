"""Single-critic agent adapted from CleanRL's continuous-action PPO.

Upstream: vwxyzjn/cleanrl@35896b1fefa9898b904f7e09bcbe6e168e15d2a9
See licenses/CleanRL-MIT.txt. Observation/action sizes are fixed for lane driving.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


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
