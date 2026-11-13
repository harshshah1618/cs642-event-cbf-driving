"""Lane-driving checks and legacy critic-disagreement calibration."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from event_cbf_driving import DT

from .cleanrl_agent import CleanRLAgent
from .lane_env import EmptyRoadVec
from .model import ActorCritic, encode, env_action_np


@torch.no_grad()
def calibrate_threshold(
    net: ActorCritic, y_target: float, seed: int = 0, n_steps: int = 4000, k_std: float = 3.0
):
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
    spread_array = np.concatenate(spreads)
    spread = float(max(np.std(spread_array), 1e-3))
    sigmas = spread_array / spread
    mean = float(sigmas.mean())
    std = float(sigmas.std() + 1e-8)
    threshold = mean + k_std * std
    return threshold, mean, std, spread


@torch.no_grad()
def evaluate_skill(net: ActorCritic | CleanRLAgent, y_target: float) -> Dict:
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
        ys_array = np.asarray(ys)
        return {
            "mean_abs_y": float(np.mean(np.abs(ys_array - y_target))),
            "mean_abs_v": float(np.mean(np.abs(np.asarray(vxs) - 20.0))),
            "final_y": float(ys_array[-1]),
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
