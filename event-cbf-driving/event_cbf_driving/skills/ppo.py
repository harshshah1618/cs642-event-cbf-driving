"""Dual-critic PPO skills with epistemic uncertainty for interrupt gating.

Architecture mirrors the validated toy trainer: relative (y−y*) encoding,
tanh-Gaussian with 0→0 accel map, linear lateral skip, bootstrap critics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from event_cbf_driving.env.types import ACTION_HIGH, ACTION_LOW, AX_HIGH, AX_LOW, AY_HIGH, AY_LOW

DEVICE = torch.device("cpu")
TARGET_SPEED = 20.0


def encode(y: np.ndarray, vy: np.ndarray, vx: np.ndarray, y_target: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    vy = np.asarray(vy, dtype=np.float32)
    vx = np.asarray(vx, dtype=np.float32)
    return np.stack(
        [(y - np.float32(y_target)) / np.float32(2.0), vy / np.float32(2.0), (vx - np.float32(20.0)) / np.float32(5.0)],
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
    nn.init.orthogonal_(net[-1].weight, gain=out_gain)
    nn.init.zeros_(net[-1].bias)
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


@dataclass
class SkillMeta:
    name: str
    target_lane: int
    y_target: float
    threshold: float
    sigma_mean: float
    sigma_std: float
    spread: float


class SkillPolicy:
    def __init__(self, name: str, net: ActorCritic, meta: SkillMeta):
        self.name = name
        self.net = net
        self.meta = meta
        self.net.eval()

    def act(self, y: float, vy: float, vx: float, deterministic: bool = True):
        o = torch.from_numpy(encode(y, vy, vx, self.meta.y_target)).view(1, 3)
        with torch.no_grad():
            a, _, v1, v2 = self.net.act(o, deterministic=deterministic)
        ax, ay = env_action_np(a.numpy()[0])
        raw = abs(float(v1) - float(v2))
        sigma = raw / max(self.meta.spread, 1e-6)
        return float(ax), float(ay), float(sigma)

    def interrupt(self, sigma: float) -> bool:
        return sigma > self.meta.threshold


def save_skill(path: Path, net: ActorCritic, meta: SkillMeta, history: Optional[List[float]] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": net.state_dict(), "meta": meta.__dict__, "history": history or []}
    torch.save(payload, path)
    meta_path = path.with_suffix(".json")
    import json

    with open(meta_path, "w") as f:
        json.dump({**meta.__dict__, "history_len": len(history or [])}, f, indent=2)


def load_skill(path: Path, device: torch.device = DEVICE) -> SkillPolicy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = ActorCritic().to(device)
    net.load_state_dict(ckpt["state_dict"])
    meta = SkillMeta(**ckpt["meta"])
    return SkillPolicy(meta.name, net, meta)


# ---------------------------------------------------------------------------
# Vectorized empty-road trainer
# ---------------------------------------------------------------------------

from event_cbf_driving import DT
from event_cbf_driving.env.types import integrate_point_mass, VehicleState


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
        rew = 1.0 - err**2 - 0.05 * (self.vx - 20.0) ** 2 - 0.02 * ax**2 - 0.35 * ay**2 - 0.05 * self.vy**2
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
        returns = []
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
                    f"  [{self.name}] update {i+1}/{n_updates}  return={batch['mean_return']:.1f}  "
                    f"hold|y|={diag['hold']['mean_abs_y']:.3f}  acq|y|={diag['acquire']['mean_abs_y']:.3f}  "
                    f"{'OK' if ok else '...'}"
                )
                if ok and (i + 1) >= min_updates:
                    break
        return self.net


@torch.no_grad()
def calibrate_threshold(net: ActorCritic, y_target: float, seed: int = 0, n_steps: int = 4000, k_std: float = 3.0):
    rng = np.random.default_rng(seed)
    env = EmptyRoadVec(8, y_target, rng)
    spreads = []
    obs = env.obs()
    net.eval()
    for _ in range(n_steps // 8):
        o = torch.from_numpy(obs)
        a, _, v1, v2 = net.act(o, deterministic=False)
        spreads.append(torch.abs(v1 - v2).numpy())
        ax, ay = env_action_np(a.numpy())
        obs, _, _, _, _, _ = env.step(ax, ay)
    spreads = np.concatenate(spreads)
    spread = float(max(np.std(spreads), 1e-3))
    sigmas = spreads / spread
    mean = float(sigmas.mean())
    std = float(sigmas.std() + 1e-8)
    threshold = mean + k_std * std
    return threshold, mean, std, spread


@torch.no_grad()
def evaluate_skill(net: ActorCritic, y_target: float) -> Dict:
    def rollout(y0, vy0, vx0, steps=500):
        y, vy, vx, x = float(y0), float(vy0), float(vx0), 0.0
        ys, vxs = [], []
        t16 = None
        other = 0.0 if y_target > 2.0 else 4.0
        for i in range(steps):
            o = torch.from_numpy(encode(y, vy, vx, y_target)).view(1, 3)
            a, _, _, _ = net.act(o, deterministic=True)
            ax, ay = env_action_np(a.numpy()[0])
            x = x + vx * DT + 0.5 * float(ax) * DT * DT
            y = y + vy * DT + 0.5 * float(ay) * DT * DT
            vx = max(0.0, vx + float(ax) * DT)
            vy = vy + float(ay) * DT
            ys.append(y)
            vxs.append(vx)
            if t16 is None and abs(y - other) >= 1.6:
                t16 = (i + 1) * DT
        ys = np.asarray(ys)
        return {
            "mean_abs_y": float(np.mean(np.abs(ys - y_target))),
            "mean_abs_v": float(np.mean(np.abs(np.asarray(vxs) - 20.0))),
            "final_y": float(ys[-1]),
            "final_vx": float(vxs[-1]),
            "t16": t16,
        }

    other = 0.0 if y_target > 2.0 else 4.0
    hold = rollout(y_target, 0.0, 20.0)
    acq = rollout(other, 0.0, 20.0)
    # Probe lateral authority from the other lane.
    o = torch.from_numpy(encode(other, 0.0, 20.0, y_target)).view(1, 3)
    a, _, _, _ = net.act(o, deterministic=True)
    _, ay_other = env_action_np(a.numpy()[0])
    learned = (
        hold["mean_abs_y"] < 0.30
        and hold["mean_abs_v"] < 1.0
        and abs(hold["final_y"] - y_target) < 0.35
        and 18.5 <= hold["final_vx"] <= 21.5
        and abs(acq["final_y"] - y_target) < 0.5
        and (acq["t16"] is not None and 1.0 <= acq["t16"] <= 2.4)
        and abs(float(ay_other)) >= 1.0
    )
    return {"hold": hold, "acquire": acq, "learned": learned, "ay_at_other": float(ay_other)}
