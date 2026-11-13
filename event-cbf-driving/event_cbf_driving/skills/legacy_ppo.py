"""Recorded dual-critic trainer. Kept for reproducing existing experiments."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from .diagnostics import evaluate_skill
from .lane_env import EmptyRoadVec
from .model import DEVICE, ActorCritic, env_action_np, pi_parameters, v_parameters


def compute_gae(rew, val, v_boot, done, gamma=0.99, lam=0.95):
    t_steps, n = rew.shape
    adv = np.zeros((t_steps, n), np.float32)
    last = np.zeros(n, np.float32)
    for t in range(t_steps - 1, -1, -1):
        delta = rew[t] + gamma * v_boot[t] - val[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + val


class PPOTrainer:
    def __init__(
        self,
        name: str,
        y_target: float,
        target_lane: int,
        n_envs: int = 16,
        seed: int = 0,
        pi_lr: float = 3e-4,
        v_lr: float = 1e-3,
    ):
        self.name = name
        self.y_target = float(y_target)
        self.target_lane = int(target_lane)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.env = EmptyRoadVec(n_envs, y_target, np.random.default_rng(seed + 17))
        self.net = ActorCritic().to(DEVICE)
        self.opt_pi = torch.optim.Adam(pi_parameters(self.net), lr=pi_lr)
        self.opt_v = torch.optim.Adam(v_parameters(self.net), lr=v_lr)
        self.history: List[float] = []
        self.rollout_steps = 400
        self.ppo_epochs = 8
        self.minibatch = 512
        self.clip = 0.2
        self.ent_coef = 0.0

    def collect(self):
        n = self.env.n
        T = self.rollout_steps
        obs_buf = np.zeros((T, n, 3), np.float32)
        act_buf = np.zeros((T, n, 2), np.float32)
        logp_buf = np.zeros((T, n), np.float32)
        rew_buf = np.zeros((T, n), np.float32)
        done_buf = np.zeros((T, n), np.float32)
        val_buf = np.zeros((T, n), np.float32)
        boot_buf = np.zeros((T, n), np.float32)
        returns: List[float] = []
        obs = self.env.obs()
        self.net.eval()
        for t in range(T):
            obs_buf[t] = obs
            o = torch.from_numpy(obs)
            with torch.no_grad():
                a, logp, v1, v2 = self.net.act(o, deterministic=False)
                v = 0.5 * (v1 + v2)
            act_buf[t] = a.numpy()
            logp_buf[t] = logp.numpy()
            val_buf[t] = v.numpy()
            ax, ay = env_action_np(act_buf[t])
            obs, rew, done, fail, boot, finished = self.env.step(ax, ay)
            rew_buf[t] = rew
            done_buf[t] = done
            with torch.no_grad():
                bv1, bv2 = self.net.values(torch.from_numpy(boot))
                bv = (0.5 * (bv1 + bv2)).numpy()
            boot_buf[t] = np.where(fail > 0.5, 0.0, bv).astype(np.float32)
            if finished.size:
                returns.extend(float(r) for r in finished)
        adv, ret = compute_gae(rew_buf, val_buf, boot_buf, done_buf)
        mean_ret = float(np.mean(returns)) if returns else float("nan")
        self.history.append(mean_ret)
        return {
            "obs": obs_buf.reshape(-1, 3),
            "act": act_buf.reshape(-1, 2),
            "logp": logp_buf.reshape(-1),
            "adv": adv.reshape(-1),
            "ret": ret.reshape(-1),
            "mean_return": mean_ret,
        }

    def update(self, batch):
        obs = torch.from_numpy(batch["obs"])
        act = torch.from_numpy(batch["act"])
        old_logp = torch.from_numpy(batch["logp"])
        adv = torch.from_numpy(batch["adv"])
        ret = torch.from_numpy(batch["ret"])
        adv = (adv - adv.mean()) / (adv.std().clamp(min=1e-6))
        n = obs.shape[0]
        self.net.train()
        pi_params = pi_parameters(self.net)
        v_params = v_parameters(self.net)
        for _ in range(self.ppo_epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, self.minibatch):
                j = perm[s : s + self.minibatch]
                if j.size < 64:
                    continue
                jj = torch.from_numpy(j.copy())
                logp, entropy = self.net.log_prob(obs[jj], act[jj])
                ratio = torch.exp(logp - old_logp[jj])
                unclipped = ratio * adv[jj]
                clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[jj]
                loss_pi = -torch.min(unclipped, clipped).mean() - self.ent_coef * entropy.mean()
                self.opt_pi.zero_grad(set_to_none=True)
                loss_pi.backward()
                nn.utils.clip_grad_norm_(pi_params, 0.5)
                self.opt_pi.step()

                v1, v2 = self.net.values(obs[jj])
                m1 = torch.bernoulli(torch.full((jj.shape[0],), 0.8))
                m2 = torch.bernoulli(torch.full((jj.shape[0],), 0.8))
                loss_v1 = ((v1 - ret[jj]).pow(2) * m1).sum() / m1.sum().clamp(min=1.0)
                loss_v2 = ((v2 - ret[jj]).pow(2) * m2).sum() / m2.sum().clamp(min=1.0)
                loss_v = 0.5 * 0.5 * (loss_v1 + loss_v2)
                self.opt_v.zero_grad(set_to_none=True)
                loss_v.backward()
                nn.utils.clip_grad_norm_(v_params, 0.5)
                self.opt_v.step()

    def train(self, n_updates: int = 140, min_updates: int = 40) -> ActorCritic:
        for i in range(n_updates):
            batch = self.collect()
            self.update(batch)
            if (i + 1) % 10 == 0 or (i + 1) == n_updates:
                diag = evaluate_skill(self.net, self.y_target)
                ok = bool(diag.get("learned"))
                print(
                    f"  [{self.name}] update {i + 1}/{n_updates}  return={batch['mean_return']:.1f}  "
                    f"hold|y|={diag['hold']['mean_abs_y']:.3f}  acq|y|={diag['acquire']['mean_abs_y']:.3f}  "
                    f"{'OK' if ok else '...'}"
                )
                if ok and (i + 1) >= min_updates:
                    break
        return self.net
