#!/usr/bin/env python3
"""Train two empty-road PPO skills and roll them through the two-trap gauntlet.

  python train_and_run.py

Trains CRUISE_RIGHT (y=0, v=20) and OVERTAKE_LEFT (y=4, v=20) with PPO.
Each policy sees only ego y, vy, vx. Obstacles are never in the training
distribution. Checkpoints are reloaded before the closed-loop rollouts.

The router is scripted (300 ms delay). It does not call a language model.
Epistemic uncertainty is the normalized gap between two value heads.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Paths and fixed experiment constants
# ---------------------------------------------------------------------------

STORE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / "checkpoints"
MEDIA = STORE / "media"
CURVE_PATH = MEDIA / "training-curves.png"
ROLLOUT_PATH = MEDIA / "gauntlet-rollout.png"
GIF_PATH = MEDIA / "gauntlet-proposed.gif"
RESULTS_PATH = SCRIPT_DIR / "results.json"

DT = 0.02
AX_LOW, AX_HIGH = -6.0, 3.5
AY_LOW, AY_HIGH = -5.0, 5.0
AX_MID = 0.5 * (AX_LOW + AX_HIGH)
AX_HALF = 0.5 * (AX_HIGH - AX_LOW)
AY_MID = 0.0
AY_HALF = 0.5 * (AY_HIGH - AY_LOW)

# Reward on the empty road. Speed weight is the one given in the spec.
# ax is penalized lightly so the speed loop stays stiff. ay is penalized
# more so the learned lane change is a smooth acquisition, not a bang-bang
# snap. That is still an action penalty inside the same reward.
ALIVE = 1.0
SPEED_W = 0.05
AX_W = 0.02
AY_W = 0.35
VY_W = 0.05

NEAR_FRACTION = 0.65
NEAR_SPAN = 1.5
WIDE_SPAN = 4.5
EP_STEPS = 400  # 8 s
N_ENVS = 16
ROLLOUT_STEPS = 400
PPO_EPOCHS = 8
MINIBATCH = 512
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.0
VF_COEF = 0.5
PI_LR = 3e-4
V_LR = 1e-3
MAX_UPDATES = 140
MIN_UPDATES = 40
BOOTSTRAP_P = 0.80
SIGMA_K = 3.0  # threshold = mean + k * std on the training distribution

# CBF. The writeup used a 0.8 s headway. Under ax >= -7.5 that barrier is
# not control-invariant from ~20 m/s: the static shield still enters the
# |dx|<4.5 box. 1.1 s is the smallest tenth that holds a stop on the 6 m
# standoff from 18–22 m/s. Alpha, standoff, inflation, and the clip stay.
HEADWAY = 1.1
STANDOFF = 6.0
CBF_ALPHA = 2.5
AX_SAFE_MIN, AX_SAFE_MAX = -7.5, 3.5
LANE_CORRIDOR = 1.85
SIGMA_INFLATION = 4.0
PENDING_INFLATION = 0.30

COLLIDE_DX = 4.5
COLLIDE_DY = 1.6
ROAD_M = 200.0
GAUNTLET_STEPS = 600  # 12 s
ROUTER_DELAY = 15  # 300 ms
BLEND_TICKS = 20
ROUTER_PERIOD = 50  # 1 Hz
SLOW_OBS = 15.0  # scripted router: slower than the 20 m/s cruise

CRUISE = "CRUISE_RIGHT"
OVERTAKE = "OVERTAKE_LEFT"
LANE_Y = {CRUISE: 0.0, OVERTAKE: 4.0}
SKILL_YT = {CRUISE: 0.0, OVERTAKE: 4.0}

MODE_ORDER = (
    "unshielded_cruise",
    "static_cbf",
    "fixed_1hz",
    "proposed",
)
MODE_TITLE = {
    "unshielded_cruise": "1  Unshielded cruise",
    "static_cbf": "2  Cruise + static CBF",
    "fixed_1hz": "3  Fixed 1 Hz router",
    "proposed": "4  Proposed interrupt + CBF",
}


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(y, vy, vx, y_target) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    vy = np.asarray(vy, dtype=np.float32)
    vx = np.asarray(vx, dtype=np.float32)
    obs = np.stack(
        [(y - np.float32(y_target)) / np.float32(2.0), vy / np.float32(2.0), (vx - np.float32(20.0)) / np.float32(5.0)],
        axis=-1,
    )
    return np.ascontiguousarray(obs.astype(np.float32))


def env_action_np(a: np.ndarray):
    """Map tanh actions with 0 at the origin, then apply the spec clip.

    An affine map of tanh onto [-6, 3.5] puts ax=0 off-center. Gaussian noise
    then makes E[ax] more negative than the deterministic mean, and PPO shifts
    that mean positive so the rollout average is right. Deterministic playback
    would accelerate forever. A symmetric map keeps the mean at ax=0.
    """
    ax = np.clip(6.0 * a[..., 0], AX_LOW, AX_HIGH)
    ay = np.clip(5.0 * a[..., 1], AY_LOW, AY_HIGH)
    return ax, ay


def kinematics(x, y, vx, vy, ax, ay, dt=DT):
    x2 = x + vx * dt + 0.5 * ax * dt * dt
    y2 = y + vy * dt + 0.5 * ay * dt * dt
    vx2 = np.maximum(0.0, vx + ax * dt) if np.ndim(vx) else max(0.0, vx + ax * dt)
    vy2 = vy + ay * dt
    return x2, y2, vx2, vy2


def collide(x, y, ox, oy) -> bool:
    return abs(ox - x) < COLLIDE_DX and abs(oy - y) < COLLIDE_DY


# ---------------------------------------------------------------------------
# Shield and scripted router
# ---------------------------------------------------------------------------

def cbf_shield(ax_rl, x, y, vx, obstacles, inflation):
    """Most restrictive ahead-in-lane CBF brake, then the shield clip."""
    ax = float(ax_rl)
    for ox, oy, ov in obstacles:
        dx = ox - x
        if dx <= 0.0 or abs(oy - y) >= LANE_CORRIDOR:
            continue
        h = (dx - STANDOFF) - HEADWAY * (vx - ov) - inflation
        ax_cbf = ((ov - vx) + CBF_ALPHA * h) / HEADWAY
        ax = min(ax, ax_cbf)
    ax_out = float(np.clip(ax, AX_SAFE_MIN, AX_SAFE_MAX))
    overriding = ax_out < float(ax_rl) - 1e-3
    return ax_out, overriding


def router_decide(skill, x, obstacles):
    """Scripted stand-in for a slow scene router. No language model."""
    lane = LANE_Y[skill]
    threat = False
    for ox, oy, ov in obstacles:
        dx = ox - x
        if dx <= 0.0 or dx > ROAD_M:
            continue
        if abs(oy - lane) < LANE_CORRIDOR and ov < SLOW_OBS:
            threat = True
            break
    if skill == CRUISE and threat:
        return OVERTAKE
    if skill == OVERTAKE and threat:
        return CRUISE
    return skill


def blend_alpha(i: int) -> float:
    u = i / (BLEND_TICKS - 1)
    return float(1.0 / (1.0 + np.exp(-10.0 * (u - 0.5))))


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

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
    """Squashed Gaussian actor with a linear skip, plus two value heads.

    The skip lets the feedback keep working a full lane away, where a plain
    tanh stack saturates and the maneuver dies.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, 2)
        # Lateral skip has no vx term. A vx→ay weight blows up at vx=0
        # because the speed feature is (vx-20)/5 = -4, and that bias walks
        # the stopped car out of the shield corridor.
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


# ---------------------------------------------------------------------------
# Empty-road environment
# ---------------------------------------------------------------------------

class EmptyRoad:
    def __init__(self, n, y_target, rng):
        self.n = n
        self.yt = float(y_target)
        self.rng = rng
        self.x = np.zeros(n, np.float64)
        self.y = np.zeros(n, np.float64)
        self.vx = np.zeros(n, np.float64)
        self.vy = np.zeros(n, np.float64)
        self.steps = np.zeros(n, np.int32)
        self.ep_ret = np.zeros(n, np.float64)
        self.reset_mask(np.ones(n, dtype=bool))

    def obs(self):
        return encode(self.y, self.vy, self.vx, self.yt)

    def reset_mask(self, mask):
        idx = np.flatnonzero(mask)
        k = idx.size
        if k == 0:
            return
        # 55% cruise-speed near the lane, 25% the same lane at the low speeds
        # the shield produces, 20% a full lane away at cruise speed.
        draw = self.rng.random(k)
        y = np.empty(k, np.float64)
        vx = np.empty(k, np.float64)
        near = draw < 0.55
        slow = (draw >= 0.55) & (draw < 0.80)
        wide = ~near & ~slow
        y[near] = self.yt + self.rng.uniform(-NEAR_SPAN, NEAR_SPAN, int(near.sum()))
        vx[near] = self.rng.uniform(12.0, 22.0, int(near.sum()))
        y[slow] = self.yt + self.rng.uniform(-1.0, 1.0, int(slow.sum()))
        vx[slow] = self.rng.uniform(0.0, 8.0, int(slow.sum()))
        y[wide] = self.yt + self.rng.uniform(-WIDE_SPAN, WIDE_SPAN, int(wide.sum()))
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
        self.x, self.y, self.vx, self.vy = kinematics(self.x, self.y, self.vx, self.vy, ax, ay)
        self.steps += 1
        err = self.y - self.yt
        rew = (
            ALIVE
            - err**2
            - SPEED_W * (self.vx - 20.0) ** 2
            - AX_W * ax**2
            - AY_W * ay**2
            - VY_W * self.vy**2
        )
        fail = np.abs(err) > 7.0
        timeout = self.steps >= EP_STEPS
        done = fail | timeout
        rew = np.where(fail, rew - 5.0, rew)
        boot = encode(self.y, self.vy, self.vx, self.yt)
        self.ep_ret += rew
        finished_idx = np.flatnonzero(done)
        finished_ret = self.ep_ret[finished_idx].copy()
        self.reset_mask(done)
        return (
            self.obs(),
            rew.astype(np.float32),
            done.astype(np.float32),
            fail.astype(np.float32),
            boot,
            finished_ret,
        )


def compute_gae(rew, val, v_boot, done):
    t_steps, n = rew.shape
    adv = np.zeros((t_steps, n), np.float32)
    last = np.zeros(n, np.float32)
    for t in range(t_steps - 1, -1, -1):
        delta = rew[t] + GAMMA * v_boot[t] - val[t]
        last = delta + GAMMA * GAE_LAMBDA * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + val


def collect_rollout(net, env):
    n = env.n
    obs_buf = np.zeros((ROLLOUT_STEPS, n, 3), np.float32)
    act_buf = np.zeros((ROLLOUT_STEPS, n, 2), np.float32)
    logp_buf = np.zeros((ROLLOUT_STEPS, n), np.float32)
    rew_buf = np.zeros((ROLLOUT_STEPS, n), np.float32)
    done_buf = np.zeros((ROLLOUT_STEPS, n), np.float32)
    val_buf = np.zeros((ROLLOUT_STEPS, n), np.float32)
    boot_buf = np.zeros((ROLLOUT_STEPS, n), np.float32)
    returns = []
    obs = env.obs()
    net.eval()
    for t in range(ROLLOUT_STEPS):
        obs_buf[t] = obs
        o = torch.from_numpy(obs)
        with torch.no_grad():
            a, logp, v1, v2 = net.act(o, deterministic=False)
        v = 0.5 * (v1 + v2)
        act_buf[t] = a.numpy()
        logp_buf[t] = logp.numpy()
        val_buf[t] = v.numpy()
        ax, ay = env_action_np(act_buf[t])
        obs, rew, done, fail, boot, finished = env.step(ax, ay)
        rew_buf[t] = rew
        done_buf[t] = done
        with torch.no_grad():
            b = torch.from_numpy(boot)
            bv1, bv2 = net.values(b)
            bv = (0.5 * (bv1 + bv2)).numpy()
        boot_buf[t] = np.where(fail > 0.5, 0.0, bv).astype(np.float32)
        if finished.size:
            returns.extend(float(r) for r in finished)
    adv, ret = compute_gae(rew_buf, val_buf, boot_buf, done_buf)
    return {
        "obs": obs_buf.reshape(-1, 3),
        "act": act_buf.reshape(-1, 2),
        "logp": logp_buf.reshape(-1),
        "adv": adv.reshape(-1),
        "ret": ret.reshape(-1),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "n_episodes": len(returns),
    }


def ppo_update(net, opt_pi, opt_v, batch, rng):
    obs = torch.from_numpy(batch["obs"])
    act = torch.from_numpy(batch["act"])
    old_logp = torch.from_numpy(batch["logp"])
    adv = torch.from_numpy(batch["adv"])
    ret = torch.from_numpy(batch["ret"])
    adv = (adv - adv.mean()) / (adv.std().clamp(min=1e-6))
    n = obs.shape[0]
    net.train()
    pi_params = pi_parameters(net)
    v_params = v_parameters(net)
    for _ in range(PPO_EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, MINIBATCH):
            j = perm[s : s + MINIBATCH]
            if j.size < 64:
                continue
            jj = torch.from_numpy(j.copy())
            logp, entropy = net.log_prob(obs[jj], act[jj])
            ratio = torch.exp(logp - old_logp[jj])
            unclipped = ratio * adv[jj]
            clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv[jj]
            loss_pi = -torch.min(unclipped, clipped).mean() - ENT_COEF * entropy.mean()
            opt_pi.zero_grad(set_to_none=True)
            loss_pi.backward()
            nn.utils.clip_grad_norm_(pi_params, 0.5)
            opt_pi.step()

            v1, v2 = net.values(obs[jj])
            m1 = torch.bernoulli(torch.full((jj.shape[0],), BOOTSTRAP_P))
            m2 = torch.bernoulli(torch.full((jj.shape[0],), BOOTSTRAP_P))
            loss_v1 = ((v1 - ret[jj]).pow(2) * m1).sum() / m1.sum().clamp(min=1.0)
            loss_v2 = ((v2 - ret[jj]).pow(2) * m2).sum() / m2.sum().clamp(min=1.0)
            loss_v = VF_COEF * 0.5 * (loss_v1 + loss_v2)
            opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            nn.utils.clip_grad_norm_(v_params, 0.5)
            opt_v.step()


@torch.no_grad()
def policy_action(net, y, vy, vx, y_target):
    net.eval()
    o = torch.from_numpy(encode(y, vy, vx, y_target)).view(1, 3)
    a, _, v1, v2 = net.act(o, deterministic=True)
    ax, ay = env_action_np(a.numpy()[0])
    raw = abs(float(v1) - float(v2))
    return float(ax), float(ay), raw


def rollout_policy(net, y_target, y0, vy0, vx0, steps):
    y, vy, vx = float(y0), float(vy0), float(vx0)
    x = 0.0
    ys, vxs, ays = [], [], []
    t16 = None
    for i in range(steps):
        ax, ay, _ = policy_action(net, y, vy, vx, y_target)
        ax = float(np.clip(ax, AX_LOW, AX_HIGH))
        ay = float(np.clip(ay, AY_LOW, AY_HIGH))
        x, y, vx, vy = kinematics(x, y, vx, vy, ax, ay)
        ys.append(y)
        vxs.append(vx)
        ays.append(ay)
        if t16 is None and abs(y - y0) >= 1.6:
            t16 = (i + 1) * DT
    ys = np.asarray(ys)
    vxs = np.asarray(vxs)
    return {
        "mean_abs_y": float(np.mean(np.abs(ys - y_target))),
        "mean_abs_v": float(np.mean(np.abs(vxs - 20.0))),
        "max_abs_y": float(np.max(np.abs(ys - y_target))),
        "final_y": float(ys[-1]),
        "final_vx": float(vxs[-1]),
        "t16": None if t16 is None else float(t16),
        "final_x": float(x),
    }


def empty_road_ok(diag) -> bool:
    hold = diag["hold"]
    acquire = diag["acquire"]
    slow = diag["slow"]
    t16 = acquire["t16"]
    return (
        hold["mean_abs_y"] < 0.30
        and hold["mean_abs_v"] < 1.0
        and hold["final_y_err"] < 0.30
        and 18.5 <= hold["final_vx"] <= 21.5
        and acquire["reached"]
        and acquire["final_y_err"] < 0.45
        and t16 is not None
        and 1.05 <= t16 <= 2.4
        and slow["max_abs_y"] < 0.8
        and abs(diag["ax_at_setpoint"]) < 0.8
        and diag["ax_at_26"] < -0.4
    )


def evaluate_skill(net, y_target):
    other = 0.0 if y_target > 2.0 else 4.0
    hold = rollout_policy(net, y_target, y_target, 0.0, 20.0, 500)
    hold["final_y_err"] = abs(hold["final_y"] - y_target)
    slow = rollout_policy(net, y_target, y_target + 0.2, 0.0, 3.0, 400)
    acq = rollout_policy(net, y_target, other, 0.0, 20.0, 500)
    acq["final_y_err"] = abs(acq["final_y"] - y_target)
    acq["reached"] = acq["final_y_err"] < 0.50 and acq["t16"] is not None
    _, ay1, _ = policy_action(net, y_target + 1.0, 0.0, 20.0, y_target)
    ax_set, _, _ = policy_action(net, y_target, 0.0, 20.0, y_target)
    ax_fast, _, _ = policy_action(net, y_target, 0.0, 26.0, y_target)
    _, ay_stop, _ = policy_action(net, y_target, 0.0, 0.0, y_target)
    _, ay_stop_off, _ = policy_action(net, y_target + 0.5, 0.0, 0.0, y_target)
    _, ay4, _ = policy_action(net, other, 0.0, 20.0, y_target)
    return {
        "hold": hold,
        "slow": {
            "max_abs_y": slow["max_abs_y"],
            "final_y": slow["final_y"],
            "final_vx": slow["final_vx"],
        },
        "acquire": {
            "t16": acq["t16"],
            "final_y": acq["final_y"],
            "final_vx": acq["final_vx"],
            "final_y_err": acq["final_y_err"],
            "mean_abs_y": acq["mean_abs_y"],
            "reached": bool(acq["reached"]),
        },
        "ay_at_1m": ay1,
        "ay_at_other_lane": ay4,
        "ax_at_setpoint": ax_set,
        "ax_at_26": ax_fast,
        "ay_at_stop": ay_stop,
        "ay_at_stop_offset": ay_stop_off,
    }


def calibrate_sigma(net, y_target, rng):
    """Spread of |V1-V2| on states the stochastic policy actually visits."""
    env = EmptyRoad(8, y_target, rng)
    raws = []
    obs = env.obs()
    net.eval()
    for _ in range(EP_STEPS):
        o = torch.from_numpy(obs)
        with torch.no_grad():
            a, _, v1, v2 = net.act(o, deterministic=False)
        raws.append(torch.abs(v1 - v2).numpy())
        ax, ay = env_action_np(a.numpy())
        obs, _, _, _, _, _ = env.step(ax, ay)
    raw = np.concatenate(raws).astype(np.float64)
    spread = float(np.std(raw))
    # If the heads agree, std is ~0 and dividing by it blows σ up. Use the
    # mean gap as the scale in that case so in-distribution σ stays near 1.
    spread = max(spread, float(np.mean(raw)), 1e-3)
    sigma = raw / spread
    mean = float(sigma.mean())
    std = float(sigma.std())
    return {
        "spread": spread,
        "sigma_mean": mean,
        "sigma_std": std,
        "threshold": mean + SIGMA_K * std,
        "sigma_p95": float(np.quantile(sigma, 0.95)),
        "sigma_max": float(sigma.max()),
        "raw_mean": float(raw.mean()),
    }


def train_skill(name, y_target, seed, max_updates):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = EmptyRoad(N_ENVS, y_target, np.random.default_rng(seed + 17))
    net = ActorCritic()
    opt_pi = torch.optim.Adam(pi_parameters(net), lr=PI_LR)
    opt_v = torch.optim.Adam(v_parameters(net), lr=V_LR)
    curve = []
    t0 = time.time()
    diag = None
    used = 0
    for update in range(1, max_updates + 1):
        batch = collect_rollout(net, env)
        if not np.isfinite(batch["mean_return"]):
            raise RuntimeError(f"{name} produced a non-finite return at update {update}")
        ppo_update(net, opt_pi, opt_v, batch, rng)
        curve.append(batch["mean_return"])
        used = update
        if update % 10 == 0 or update == max_updates:
            diag = evaluate_skill(net, y_target)
            ok = empty_road_ok(diag)
            print(
                f"  {name} update {update:3d}  return {batch['mean_return']:8.1f}  "
                f"hold |y| {diag['hold']['mean_abs_y']:.3f}  |v| {diag['hold']['mean_abs_v']:.3f}  "
                f"final y {diag['hold']['final_y']:.3f} vx {diag['hold']['final_vx']:.2f}  "
                f"acquire t16 {diag['acquire']['t16']} final_y {diag['acquire']['final_y']:.3f}  "
                f"ay@1m {diag['ay_at_1m']:.2f} ay@other {diag['ay_at_other_lane']:.2f}  "
                f"slow|y| {diag['slow']['max_abs_y']:.2f}  "
                f"ax@20 {diag['ax_at_setpoint']:.2f} ax@26 {diag['ax_at_26']:.2f}  "
                f"{'OK' if ok else '...'}",
                flush=True,
            )
            if ok and update >= MIN_UPDATES:
                break
    if diag is None:
        diag = evaluate_skill(net, y_target)
    stats = calibrate_sigma(net, y_target, np.random.default_rng(seed + 99))
    stats["curve"] = curve
    stats["updates"] = used
    stats["seconds"] = round(time.time() - t0, 1)
    stats["y_target"] = y_target
    stats["diag"] = diag
    stats["learned"] = bool(empty_road_ok(diag))
    print(
        f"  {name} sigma mean {stats['sigma_mean']:.3f} std {stats['sigma_std']:.3f} "
        f"thr {stats['threshold']:.3f} spread {stats['spread']:.4f} "
        f"learned {stats['learned']}",
        flush=True,
    )
    return net, stats


def save_skill(name, net, stats):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), CKPT_DIR / f"{name}.pt")
    payload = json.loads(json.dumps(stats))
    (CKPT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))


def load_skill(name, y_target):
    net = ActorCritic()
    try:
        state = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    stats = json.loads((CKPT_DIR / f"{name}.json").read_text())
    if abs(stats["y_target"] - y_target) > 1e-6:
        raise RuntimeError(f"checkpoint {name} has y_target {stats['y_target']}")
    return net, stats


# ---------------------------------------------------------------------------
# Gauntlet
# ---------------------------------------------------------------------------

def run_gauntlet(mode, cruise, overtake):
    nets = {CRUISE: cruise[0], OVERTAKE: overtake[0]}
    stats = {CRUISE: cruise[1], OVERTAKE: overtake[1]}
    x = y = vy = 0.0
    vx = 20.0
    committed = CRUISE
    delay = 0
    delay_to = None
    blend = None  # [index, src, dst]
    calls = 0
    call_log = []
    sigma_latch = False
    override_off = 100
    since_call = 1000
    spawned = False
    cut = None
    prev_ay = 0.0
    peak_jerk = 0.0
    peak_sigma = 0.0
    hist = {k: [] for k in ("t", "x", "y", "vx", "ay", "skill", "sigma", "shield", "cutx")}
    crashed = False
    end_step = GAUNTLET_STEPS

    def obstacles():
        obs = [(75.0, 0.0, 0.0)]
        if cut is not None:
            obs.append((cut[0], cut[1], cut[2]))
        return obs

    def log(t, skill, sigma, shield):
        hist["t"].append(t)
        hist["x"].append(x)
        hist["y"].append(y)
        hist["vx"].append(vx)
        hist["ay"].append(prev_ay)
        hist["skill"].append(skill)
        hist["sigma"].append(sigma)
        hist["shield"].append(shield)
        hist["cutx"].append(np.nan if cut is None else cut[0])

    log(0.0, committed, 0.0, False)

    for step in range(GAUNTLET_STEPS):
        t = step * DT
        obs_list = obstacles()
        if (not spawned) and x >= 92.0:
            spawned = True
            cut = [x + 23.0, 4.0, 9.5]
            obs_list = obstacles()

        pending = delay > 0
        if delay > 0:
            delay -= 1
            if delay == 0:
                if mode == "fixed_1hz":
                    committed = delay_to
                else:
                    blend = [0, committed, delay_to]
                    committed = delay_to

        def act_of(skill):
            ax_s, ay_s, raw = policy_action(nets[skill], y, vy, vx, SKILL_YT[skill])
            sigma = raw / stats[skill]["spread"]
            return ax_s, ay_s, sigma

        if mode in ("unshielded_cruise", "static_cbf"):
            ax, ay, sigma = act_of(CRUISE)
            skill_label = CRUISE
            sigma_skill = CRUISE
        elif blend is not None:
            src, dst = blend[1], blend[2]
            ax0, ay0, s0 = act_of(src)
            ax1, ay1, s1 = act_of(dst)
            alpha = blend_alpha(blend[0])
            ax = (1.0 - alpha) * ax0 + alpha * ax1
            ay = (1.0 - alpha) * ay0 + alpha * ay1
            if alpha >= 0.5:
                sigma, sigma_skill = s1, dst
            else:
                sigma, sigma_skill = s0, src
            skill_label = "BLEND"
            blend[0] += 1
            if blend[0] >= BLEND_TICKS:
                blend = None
        else:
            ax, ay, sigma = act_of(committed)
            skill_label = committed
            sigma_skill = committed

        ax_rl = float(ax)
        ay = float(np.clip(ay, AY_LOW, AY_HIGH))
        use_shield = mode in ("static_cbf", "proposed")
        overriding = False
        if use_shield:
            infl = 0.0
            if mode == "proposed":
                infl = SIGMA_INFLATION * max(sigma, 0.0) + (PENDING_INFLATION * vx if pending else 0.0)
            ax, overriding = cbf_shield(ax_rl, x, y, vx, obs_list, infl)
        else:
            ax = float(np.clip(ax_rl, AX_LOW, AX_HIGH))

        # Router. Fixed 1 Hz always queries. Proposed queries on a rising interrupt.
        if mode == "fixed_1hz" and step % ROUTER_PERIOD == 0:
            calls += 1
            dec = router_decide(committed, x, obs_list)
            call_log.append({"t": round(t, 3), "from": committed, "to": dec, "reason": "1hz"})
            if dec != committed and delay == 0 and blend is None:
                delay = ROUTER_DELAY
                delay_to = dec
        elif mode == "proposed":
            thr = stats[sigma_skill]["threshold"]
            sigma_high = sigma > thr
            # Ignore one-tick shield chatter so a single brake event is one call.
            shield_rise = overriding and override_off >= 8
            # A sigma-only query that does not change skill must not block the
            # next shield rising edge. Otherwise one early σ spike latches the
            # router and the cut-in never gets a call.
            can_call = delay == 0 and blend is None and since_call >= 10
            if can_call and (shield_rise or (sigma_high and not sigma_latch)):
                calls += 1
                since_call = 0
                reason = "shield+sigma" if (overriding and sigma_high) else ("shield" if overriding else "sigma")
                dec = router_decide(committed, x, obs_list)
                call_log.append(
                    {"t": round(t, 3), "from": committed, "to": dec, "reason": reason, "sigma": round(float(sigma), 3)}
                )
                if dec != committed:
                    delay = ROUTER_DELAY
                    delay_to = dec
                elif sigma_high and not shield_rise:
                    sigma_latch = True
            if not sigma_high:
                sigma_latch = False
            override_off = 0 if overriding else override_off + 1
            since_call += 1

        peak_sigma = max(peak_sigma, sigma)
        jerk = abs(ay - prev_ay) / DT
        peak_jerk = max(peak_jerk, jerk)
        prev_ay = ay

        x, y, vx, vy = kinematics(x, y, vx, vy, ax, ay)
        if cut is not None:
            cut[0] = cut[0] + cut[2] * DT
        log((step + 1) * DT, skill_label, sigma, overriding)

        hit = False
        for ox, oy, ov in obstacles():
            if collide(x, y, ox, oy):
                hit = True
                break
        if hit or x >= ROAD_M:
            crashed = hit
            end_step = step + 1
            break

    end_t = end_step * DT
    route = 100.0 * min(x, ROAD_M) / ROAD_M
    avg_speed = float(x / end_t) if end_t > 0 else 0.0
    for k, v in list(hist.items()):
        if k == "skill":
            continue
        hist[k] = np.asarray(v, dtype=np.float64)
    return {
        "mode": mode,
        "crash_time": round(end_t, 4) if crashed else None,
        "crashed": crashed,
        "route_pct": round(route, 2),
        "avg_speed": round(avg_speed, 3),
        "router_calls": calls,
        "peak_lateral_jerk": round(peak_jerk, 2),
        "peak_sigma": round(peak_sigma, 3),
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "vx": round(float(vx), 3),
        "end_time": round(end_t, 4),
        "calls": call_log,
        "hist": hist,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_curves(cruise_curve, overtake_curve, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=140)
    u1 = np.arange(1, len(cruise_curve) + 1)
    u2 = np.arange(1, len(overtake_curve) + 1)
    ax.plot(u1, cruise_curve, color="#1f4e79", lw=1.8, label="CRUISE_RIGHT")
    ax.plot(u2, overtake_curve, color="#c46b2c", lw=1.8, label="OVERTAKE_LEFT")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Mean episode return")
    ax.set_title("Empty-road training")
    ax.grid(True, color="0.85", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _cars(ax, stalled=True, cut=None):
    if stalled:
        ax.add_patch(
            FancyBboxPatch(
                (75.0 - 2.25, -0.8),
                4.5,
                1.6,
                boxstyle="round,pad=0.02,rounding_size=0.3",
                facecolor="#4a4a4a",
                edgecolor="black",
                lw=0.4,
                zorder=3,
            )
        )
    if cut is not None and np.isfinite(cut):
        ax.add_patch(
            Rectangle((cut - 2.25, 4.0 - 0.8), 4.5, 1.6, facecolor="#b03a2e", edgecolor="black", lw=0.4, zorder=3)
        )


def plot_rollouts(results, path):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), dpi=140, sharex=True, sharey=True)
    for ax, mode in zip(axes.ravel(), MODE_ORDER):
        r = results[mode]
        h = r["hist"]
        ax.axhspan(-1.6, 5.6, color="#f6f3ec", zorder=0)
        ax.axhline(0.0, color="0.55", ls="--", lw=0.7, zorder=1)
        ax.axhline(4.0, color="0.55", ls="--", lw=0.7, zorder=1)
        ax.plot(h["x"], h["y"], color="#1f4e79", lw=1.6, zorder=2, label="ego")
        _cars(ax, True, None)
        cut = h["cutx"]
        finite = np.isfinite(cut)
        if finite.any():
            ax.plot(cut[finite], np.full(finite.sum(), 4.0), color="#b03a2e", lw=1.0, alpha=0.8, zorder=2)
            _cars(ax, False, float(cut[finite][-1]))
        if r["crashed"]:
            ax.scatter([r["x"]], [r["y"]], marker="x", s=46, color="#9b1c1c", zorder=4)
        crash = "no crash" if r["crash_time"] is None else f"crash {r['crash_time']:.2f} s"
        ax.set_title(
            f"{MODE_TITLE[mode]}\n{crash} · route {r['route_pct']:.1f}% · {r['avg_speed']:.1f} m/s",
            fontsize=10,
        )
        ax.set_xlim(0, 210)
        ax.set_ylim(-2.2, 6.6)
        ax.grid(True, color="0.88", lw=0.5)
    axes[1, 0].set_xlabel("x (m)")
    axes[1, 1].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")
    fig.suptitle("Two-trap gauntlet with the trained policies", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_gif(result, path):
    h = result["hist"]
    stride = 5
    frames = []
    idxs = list(range(0, len(h["t"]), stride))
    if idxs[-1] != len(h["t"]) - 1:
        idxs.append(len(h["t"]) - 1)
    for k in idxs:
        fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=90)
        ax.axhspan(-1.6, 5.6, color="#f6f3ec", zorder=0)
        ax.axhline(0.0, color="0.55", ls="--", lw=0.7)
        ax.axhline(4.0, color="0.55", ls="--", lw=0.7)
        ax.plot(h["x"][: k + 1], h["y"][: k + 1], color="#1f4e79", lw=1.5)
        _cars(ax, True, None)
        if np.isfinite(h["cutx"][k]):
            _cars(ax, False, float(h["cutx"][k]))
        ex, ey = float(h["x"][k]), float(h["y"][k])
        ax.add_patch(Rectangle((ex - 2.25, ey - 0.8), 4.5, 1.6, facecolor="#2e86ab", edgecolor="black", lw=0.4, zorder=4))
        shield = "shield on" if h["shield"][k] else "shield off"
        ax.set_title(
            f"Proposed  t={h['t'][k]:.1f}s  {h['skill'][k]}  σ={h['sigma'][k]:.2f}  {shield}  vx={h['vx'][k]:.1f}",
            fontsize=9,
        )
        ax.set_xlim(0, 210)
        ax.set_ylim(-2.2, 6.6)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.tight_layout()
        fig.canvas.draw()
        w, ht = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(ht, w, 4)
        frames.append(Image.fromarray(buf[:, :, :3].copy()))
        plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _fmt_crash(v):
    return "none" if v is None else f"{v:.2f}"


def write_readme(cruise_stats, overtake_stats, results):
    rows = [results[m] for m in MODE_ORDER]
    learned_c = cruise_stats["learned"]
    learned_o = overtake_stats["learned"]
    hc, ho = cruise_stats["diag"]["hold"], overtake_stats["diag"]["hold"]
    ac, ao = cruise_stats["diag"]["acquire"], overtake_stats["diag"]["acquire"]

    def row(r):
        return (
            f"| {MODE_TITLE[r['mode']]} | {_fmt_crash(r['crash_time'])} | {r['route_pct']:.1f} | "
            f"{r['avg_speed']:.2f} | {r['router_calls']} | {r['peak_lateral_jerk']:.1f} |"
        )

    m1, m2, m3, m4 = rows
    story_ok = (
        m1["crashed"]
        and m1["x"] < 85
        and (not m2["crashed"])
        and m2["vx"] < 0.5
        and m2["x"] < 72
        and abs(m2["y"]) < 1.2
        and m3["crashed"]
        and (not m4["crashed"])
        and abs(m4["y"]) < 0.8
        and m4["x"] > 110
    )
    if story_ok:
        reading = (
            "The four rows separate the way the stack is built to separate. "
            f"Unshielded cruise hits the stalled car at x={m1['x']:.1f} m. "
            f"The static shield stops short of it (final x={m2['x']:.1f} m, vx={m2['vx']:.2f} m/s) and stays there, "
            "because nothing changes lanes. "
            f"The 1 Hz hard switch books {m3['router_calls']} router calls and still collides "
            f"at {m3['crash_time']:.2f} s (x={m3['x']:.1f} m, y={m3['y']:.2f} m), with peak lateral jerk "
            f"{m3['peak_lateral_jerk']:.1f} m/s³. "
            f"The proposed stack collides with nothing, covers {m4['route_pct']:.1f}% of the road, "
            f"calls the router {m4['router_calls']} times, and peaks at {m4['peak_lateral_jerk']:.1f} m/s³ of lateral jerk."
        )
    else:
        reading = (
            "These measurements are what the closed loop did on this run. "
            "They do not all match the intended mechanism split "
            "(unshielded crash, static shield stops, 1 Hz router crashes, proposed clears both traps)."
        )

    text = f"""# Two-trap gauntlet

Closed-loop proof of concept. Two continuous-action PPO policies are trained on an empty road, saved, loaded back, and driven through the gauntlet. A kinematic tracker is not used in the rollout.

## What is trained

Both skills are separate actor-critics. The observation is ego `y`, `vy`, `vx` only. Nothing in training is an obstacle.

- `CRUISE_RIGHT` regulates y = 0 m and vx = 20 m/s.
- `OVERTAKE_LEFT` regulates y = 4 m and vx = 20 m/s.

The action is longitudinal and lateral acceleration. The policy box is ax in [-6, 3.5] m/s² and ay in [-5, 5] m/s². The step is dt = 0.02 s:

```
x  += vx dt + 0.5 ax dt²
y  += vy dt + 0.5 ay dt²
vx  = max(0, vx + ax dt)
vy += ay dt
```

Episodes last 8 s. Initial vy is uniform on [-1.5, 1.5] m/s. Resets are a mixture: 55% within 1.5 m of the target lane at vx in [12, 22] m/s, 25% within 1 m of the lane at vx in [0, 8] m/s (the speeds the shield creates), and 20% up to 4.5 m from the lane at vx in [12, 22] so the skill can acquire the lane when the router engages it. The reward is an alive bonus minus (y − y_target)² minus 0.05 (vx − 20)² minus an action penalty (0.02 ax² + 0.35 ay²) and a small lateral-rate penalty (0.05 vy²).

The actor is a small tanh MLP plus a linear skip, with a squashed Gaussian whose zero maps to zero acceleration (the box [-6, 3.5] × [-5, 5] is applied after that). Two value heads sit beside it and are trained with bootstrap masks (keep probability 0.8), so they are not the same fit. Epistemic uncertainty is

```
σ = |V1 − V2| / spread
```

where `spread` is the standard deviation of |V1 − V2| on states the stochastic policy visits during a calibration rollout after training. The interrupt threshold is mean(σ) + 3 std(σ) on that same training distribution. It is computed once per skill and frozen. It is not adjusted to the gauntlet crash.

This run: cruise threshold {cruise_stats['threshold']:.3f} (mean {cruise_stats['sigma_mean']:.3f}, std {cruise_stats['sigma_std']:.3f}); overtake threshold {overtake_stats['threshold']:.3f} (mean {overtake_stats['sigma_mean']:.3f}, std {overtake_stats['sigma_std']:.3f}).

## What is scripted

The router does not call a language model. It is a lane rule with a 300 ms delay (15 ticks at 50 Hz): a vehicle ahead in the active lane traveling under 15 m/s requests the other skill. `CRUISE_RIGHT` switches toward `OVERTAKE_LEFT` for the stalled car. `OVERTAKE_LEFT` switches toward `CRUISE_RIGHT` for the cut-in.

- Fixed baseline: the rule is queried at 1 Hz and the new skill is applied with a hard switch after the delay. No shield.
- Proposed: the rule runs only when an interrupt fires, then the two trained policies are mixed with a 20-tick sigmoid. An interrupt is σ above that skill's training threshold, or the shield overriding ax.

The shield is a CBF on obstacles ahead in the lane corridor |y_obs − y| < 1.85 m:

```
h = (x_obs − x − 6) − 1.1 (vx − v_obs) − inflation
ax = clip(min(ax_rl, ((v_obs − vx) + 2.5 h) / 1.1), −7.5, 3.5)
inflation = 4 σ + (0.30 vx if the router delay is still pending else 0)
```

The headway in that expression is 1.1 s, not 0.8 s. With a 0.8 s headway and ax limited to −7.5 m/s², the barrier is not control-invariant from 20 m/s: a static shield still enters the |dx| < 4.5 m collision box (impact near 4.1 s at about 6 m/s). 1.1 s is the smallest tenth of a second that holds a stop on the 6 m standoff from 18–22 m/s. The standoff, the class-K gain 2.5, the inflation formula, and the shield clip are unchanged. Static CBF mode sets inflation to 0 and does not run the router. Proposed mode uses the inflation formula.

The road is 200 m and 12 s. Ego starts at x = 0, y = 0, vx = 20. A stalled car sits at x = 75, y = 0. Once ego reaches x = 92, a cut-in appears 23 m ahead in the left lane (y = 4) at 9.5 m/s. A collision is |dx| < 4.5 m and |dy| < 1.6 m.

## How to rerun

From this directory, with numpy, matplotlib, and torch:

```bash
python train_and_run.py
```

The script trains both skills on CPU, writes `checkpoints/`, reloads those weights, runs the four closed-loop modes, and overwrites the figures and this README with the numbers from that run. A few minutes is typical.

## Figures

`media/training-curves.png` is the mean undiscounted episode return against PPO update, one curve per skill. The points are the returns collected on that update.

`media/gauntlet-rollout.png` is the four closed-loop paths. Dashed lines are the lane centers y = 0 and y = 4. The dark car is the stalled obstacle. The red car and its trail are the cut-in. A red cross is a collision.

`media/gauntlet-proposed.gif` is the proposed rollout only.

## Empty-road check (10 s, deterministic)

| Skill | Mean \\|y − target\\| (m) | Mean \\|vx − 20\\| (m/s) | Final y (m) | Final vx (m/s) | Time to move 1.6 m from the other lane (s) | Learned |
| --- | --- | --- | --- | --- | --- | --- |
| CRUISE_RIGHT | {hc['mean_abs_y']:.3f} | {hc['mean_abs_v']:.3f} | {hc['final_y']:.3f} | {hc['final_vx']:.2f} | {ac['t16']:.2f} | {"yes" if learned_c else "no"} |
| OVERTAKE_LEFT | {ho['mean_abs_y']:.3f} | {ho['mean_abs_v']:.3f} | {ho['final_y']:.3f} | {ho['final_vx']:.2f} | {ao['t16']:.2f} | {"yes" if learned_o else "no"} |

Cruise is started on y = 0 and, for the lane-change time, from y = 4. Overtake is started on y = 4 and from y = 0. Both targets are vx = 20.

## Measured gauntlet

Route % is 100 × min(x, 200) / 200 at the end of the rollout. Average speed is distance traveled divided by time until collision, the 200 m mark, or 12 s. Lateral jerk is |Δay| / dt. Router calls count every query, including queries that do not change skill.

| Mode | Crash time (s) | Route % | Avg speed (m/s) | Router calls | Peak lateral jerk (m/s³) |
| --- | --- | --- | --- | --- | --- |
{row(m1)}
{row(m2)}
{row(m3)}
{row(m4)}

{reading}

Proposed peak σ was {m4['peak_sigma']:.2f}. Static CBF ended at x = {m2['x']:.2f} m, y = {m2['y']:.2f} m, vx = {m2['vx']:.2f} m/s.
"""
    (SCRIPT_DIR / "README.md").write_text(text)
    return story_ok


def self_check():
    # One kinematics step matches the discrete update.
    x, y, vx, vy = kinematics(0.0, 0.0, 20.0, 0.0, 1.0, 2.0)
    assert abs(x - (20 * DT + 0.5 * 1 * DT * DT)) < 1e-9
    assert abs(y - (0.5 * 2 * DT * DT)) < 1e-9
    assert abs(vx - 20.02) < 1e-9
    assert abs(vy - 0.04) < 1e-9

    # Squashed-Gaussian log prob is consistent, and ax=0 is the initial mean.
    seed_everything(0)
    net = ActorCritic()
    obs = torch.zeros(32, 3)
    with torch.no_grad():
        a, logp, _, _ = net.act(obs, deterministic=False)
        logp2, _ = net.log_prob(obs, a)
    if not torch.allclose(logp, logp2, atol=1e-4):
        raise RuntimeError("tanh-gaussian log-prob mismatch")
    with torch.no_grad():
        a0, _, _, _ = net.act(torch.zeros(1, 3), deterministic=True)
    ax0, ay0 = env_action_np(a0.numpy()[0])
    if abs(ax0) > 1e-4 or abs(ay0) > 1e-4:
        raise RuntimeError(f"initial mean action should be ~0, got {ax0}, {ay0}")

    # Static shield, ax_rl = 0, must stop outside the collision box and hold.
    x = y = vy = 0.0
    vx = 20.0
    for _ in range(600):
        ax, _ = cbf_shield(0.0, x, y, vx, [(75.0, 0.0, 0.0)], 0.0)
        x, y, vx, vy = kinematics(x, y, vx, vy, ax, 0.0)
        if collide(x, y, 75.0, 0.0):
            raise RuntimeError(f"static shield self-check collided at x={x:.2f}")
    if vx > 0.3 or not (60.0 < x < 70.5):
        raise RuntimeError(f"static shield self-check did not hold, x={x:.2f} vx={vx:.2f}")

    # Unshielded coast into the stall collides.
    x, vx = 0.0, 20.0
    hit = False
    for _ in range(400):
        x, _, vx, _ = kinematics(x, 0.0, vx, 0.0, 0.0, 0.0)
        if collide(x, 0.0, 75.0, 0.0):
            hit = True
            break
    if not hit:
        raise RuntimeError("unshielded coast should hit the stalled car")
    print("self-check ok", flush=True)


def dump_public(results):
    public = []
    for mode in MODE_ORDER:
        r = dict(results[mode])
        r.pop("hist", None)
        public.append(r)
    return public


def main():
    import argparse

    p = argparse.ArgumentParser(description="Train PPO skills and run the two-trap gauntlet")
    p.add_argument("--smoke", action="store_true", help="one update and a short rollout, for debugging")
    p.add_argument("--updates", type=int, default=MAX_UPDATES)
    args = p.parse_args()
    torch.set_num_threads(4)
    self_check()
    updates = 1 if args.smoke else args.updates

    print("training CRUISE_RIGHT", flush=True)
    cruise_net, cruise_stats = train_skill(CRUISE, 0.0, seed=7, max_updates=updates if not args.smoke else 1)
    print("training OVERTAKE_LEFT", flush=True)
    overtake_net, overtake_stats = train_skill(OVERTAKE, 4.0, seed=11, max_updates=updates if not args.smoke else 1)
    save_skill("cruise", cruise_net, cruise_stats)
    save_skill("overtake", overtake_net, overtake_stats)
    # Closed loop must use the files on disk, not the in-memory nets.
    cruise = load_skill("cruise", 0.0)
    overtake = load_skill("overtake", 4.0)

    if not args.smoke:
        plot_curves(cruise_stats["curve"], overtake_stats["curve"], CURVE_PATH)

    results = {}
    steps_note = "full"
    for mode in MODE_ORDER:
        results[mode] = run_gauntlet(mode, cruise, overtake)
        r = results[mode]
        print(
            f"{mode:20s}  crash {r['crash_time']}  route {r['route_pct']:.1f}  "
            f"avg {r['avg_speed']:.2f}  calls {r['router_calls']}  "
            f"jerk {r['peak_lateral_jerk']:.1f}  final x {r['x']:.1f} y {r['y']:.2f} vx {r['vx']:.2f}",
            flush=True,
        )
        for c in r["calls"]:
            print(f"    call {c}", flush=True)

    if args.smoke:
        print("smoke finished", steps_note, flush=True)
        return

    plot_rollouts(results, ROLLOUT_PATH)
    print("writing gif", flush=True)
    write_gif(results["proposed"], GIF_PATH)
    story_ok = write_readme(cruise_stats, overtake_stats, results)
    payload = {
        "cruise": {k: cruise_stats[k] for k in ("threshold", "sigma_mean", "sigma_std", "spread", "learned", "updates", "diag")},
        "overtake": {k: overtake_stats[k] for k in ("threshold", "sigma_mean", "sigma_std", "spread", "learned", "updates", "diag")},
        "rollouts": dump_public(results),
        "story_ok": story_ok,
        "headway_s": HEADWAY,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print("story_ok", story_ok, flush=True)
    print("wrote", CURVE_PATH, ROLLOUT_PATH, GIF_PATH, RESULTS_PATH, flush=True)


if __name__ == "__main__":
    main()
