"""OmniSafe 0.5 adapter retaining the native SafeMetaDrive observation/action API."""

import random

import gymnasium as gym
import numpy as np
import torch
from omnisafe.envs.core import CMDP, env_register

from driving.safemetadrive.protocol import environment_config


def make_native_env(seed):
    from metadrive import SafeMetaDriveEnv

    class SeededMaps(gym.Wrapper):
        def __init__(self):
            super().__init__(SafeMetaDriveEnv(environment_config("train")))
            self.rng = np.random.default_rng(seed)

        def reset(self, *, seed=None, options=None):
            # Algorithm seeds randomize map order, not the reserved map split.
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            map_seed = int(self.rng.integers(0, 100))
            return self.env.reset(seed=map_seed)

    return SeededMaps()


@env_register
class BenchmarkMetaDrive(CMDP):
    _support_envs = ["CS642SafeMetaDrive-v0"]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    def __init__(self, env_id, num_envs=1, device="cpu", **kwargs):
        from functools import partial

        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._env = gym.vector.AsyncVectorEnv(
            [partial(make_native_env, i) for i in range(num_envs)], context="spawn")
        self._action_space = self._env.single_action_space
        self._observation_space = self._env.single_observation_space
        self._metadata = self._env.metadata
        self._seed = 0

    def step(self, action):
        a = action.detach().cpu().numpy().reshape(self._num_envs, -1)
        obs, reward, terminated, truncated, info = self._env.step(a)
        cost = np.asarray(info.get("cost", np.zeros(self._num_envs))).copy()
        # Gym vector auto-reset moves terminal info into final_info.
        for i, done in enumerate(terminated | truncated):
            if done:
                cost[i] = info["final_info"][i]["cost"]
        if "final_observation" in info:
            info["final_observation"] = torch.as_tensor(np.stack([
                x if x is not None else np.zeros(obs.shape[-1])
                for x in info["final_observation"]]), dtype=torch.float32, device=self._device)
        result = [torch.as_tensor(x, dtype=torch.float32, device=self._device)
                  for x in (obs, reward, cost, terminated, truncated)]
        # OmniSafe's single-env Unsqueeze wrapper supplies the batch axis.
        if self._num_envs == 1:
            result = [x[0] for x in result]
            if "final_observation" in info:
                info["final_observation"] = info["final_observation"][0]
        return (*result, info)

    def reset(self, seed=None, options=None):
        seed = self._seed if seed is None else seed
        obs, _ = self._env.reset(seed=[seed + i for i in range(self._num_envs)])
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        return (obs[0] if self._num_envs == 1 else obs), {}

    def set_seed(self, seed):
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def sample_action(self):
        return torch.as_tensor(self._action_space.sample(), device=self._device)

    def render(self):
        return None

    def close(self):
        self._env.close()
