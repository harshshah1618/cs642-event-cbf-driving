"""Train a shielded, sensor-feature Layer 2 PPO tracker inside UnifiedDrivingSystem.

Warm-starts with behavior cloning from the analytic Layer 2 teacher on real SafeMetaDrive
training maps (0-99, accident_prob=0.8, traffic_density=0.05) with Layer 1 safety filter
and crossing guard active in the loop, then fine-tunes with PPO for high zero-cost completion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np

from driving.safemetadrive.lidar_driver import decode_observation
from driving.safemetadrive.protocol import environment_config, manifest
from driving.safemetadrive.unified import UnifiedConfig, UnifiedDrivingSystem


class ShieldedSensorSkillEnv(gym.Env):
    """Exact closed-loop SafeMetaDrive training env for Layer 2 PPO.

    Observation: 12-dim strictly observation-only features from UnifiedDrivingSystem._build_ppo_features_v2.
    Action: [steer, throttle] in [-1, 1]^2, shielded by Layer 1 (throttle cap + road-boundary guard + blending).
    """

    def __init__(self, seed: int = 0):
        from metadrive import SafeMetaDriveEnv

        self.native = SafeMetaDriveEnv(environment_config("train"))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (12,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.system: UnifiedDrivingSystem | None = None
        self._last_obs: np.ndarray | None = None
        self._last_feats: np.ndarray = np.zeros(12, dtype=np.float32)
        self._last_teacher: np.ndarray = np.zeros(2, dtype=np.float32)
        self._last_safe_throttle: float = 0.0
        self._last_d = None

    def _prepare_step(self, raw_obs: np.ndarray) -> np.ndarray:
        assert self.system is not None
        self.system.on_observation(raw_obs)
        d = decode_observation(raw_obs)
        c = self.system.config
        analytic_steer, safe_throttle = self.system._core.act()
        cur_target = self.system._core.target_lane_idx
        if (
            self.system._prev_target_lane is not None
            and cur_target != self.system._prev_target_lane
            and c.blend_steps > 0
        ):
            self.system._blend_rem = c.blend_steps
            self.system._blend_from_steer = self.system._prev_steer
        self.system._prev_target_lane = cur_target

        centers = self.system._core._lane_centers_from_left(d)
        target_offset_m = centers[cur_target] - d.dist_left
        desired_est = max(
            0.0,
            d.speed_mps + (safe_throttle / 0.45 if safe_throttle >= 0 else safe_throttle / 0.75),
        )
        feats = self.system._build_ppo_features_v2(d, target_offset_m, desired_est)
        self._last_obs = raw_obs
        self._last_feats = feats
        self._last_teacher = np.array([float(analytic_steer), float(safe_throttle)], dtype=np.float32)
        self._last_safe_throttle = float(safe_throttle)
        self._last_d = d
        return feats

    def teacher_action(self) -> np.ndarray:
        return self._last_teacher.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        map_seed = int(self.rng.integers(0, 100))
        raw_obs, _ = self.native.reset(seed=map_seed)
        self.system = UnifiedDrivingSystem(
            raw_obs,
            UnifiedConfig(tracker="analytic", use_safety_filter=True, crossing_guard=True),
        )
        feats = self._prepare_step(raw_obs)
        return feats, {}

    def step(self, action):
        assert self.system is not None and self._last_d is not None
        c = self.system.config
        d = self._last_d

        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = min(float(np.clip(action[1], -1.0, 1.0)), self._last_safe_throttle)

        # Exact Layer 1 road-boundary steering guard
        border_margin = 1.65
        if d.dist_left < border_margin:
            steer = min(steer, -(border_margin - d.dist_left) * 1.35)
        if d.dist_right < border_margin:
            steer = max(steer, (border_margin - d.dist_right) * 1.35)
        steer = float(np.clip(steer, -1.0, 1.0))

        # Exact Layer 2 smooth transition blending
        if self.system._blend_rem > 0 and c.blend_steps > 0:
            w = 1.0 - (self.system._blend_rem / float(c.blend_steps + 1))
            steer = float(np.clip((1.0 - w) * self.system._blend_from_steer + w * steer, -1.0, 1.0))
            self.system._blend_rem -= 1

        steer_delta = abs(steer - self.system._prev_steer)
        self.system._prev_steer = steer

        raw_obs, native_reward, terminated, truncated, info = self.native.step([steer, throttle])
        cost = float(info.get("cost", 0.0))

        # Reward: maximize progress & zero-cost completion, gently center in target lane, penalize cost/out-of-road
        lane_lat_m = abs(float(self._last_feats[3]) * 5.0)
        reward = float(native_reward) - 0.10 * min(lane_lat_m, 4.0) - 0.15 * steer_delta
        if cost > 0.0:
            reward -= 15.0 * cost
        if info.get("out_of_road", False):
            reward -= 35.0
        if info.get("arrive_dest", False):
            reward += 30.0

        if terminated or truncated:
            obs = self._last_feats.copy()
        else:
            obs = self._prepare_step(raw_obs)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.native.close()


def main(seed: int, output: str, steps: int = 500_000, demos: int = 30_000, workers: int = 8) -> None:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    scratch = demos <= 0
    record = manifest(
        "train",
        "shielded_sensor_ppo_scratch" if scratch else "shielded_sensor_skill_v2",
        dict(seed=seed, steps=steps, demos=demos, workers=workers, scratch=scratch),
    )
    record.update(status="running", feature_dim=12, algorithm_library="stable-baselines3==2.3.2")
    (out / "manifest.json").write_text(json.dumps(record, indent=2))

    env = SubprocVecEnv([partial(ShieldedSensorSkillEnv, seed * 100 + i) for i in range(workers)], start_method="spawn")
    if scratch:
        # Standard CleanRL continuous-control PPO defaults from random weights (zero BC demos)
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=lambda prog: 3e-4 * prog,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-0.5),
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=lambda prog: 5e-5 * prog + 1e-5,
            n_steps=512,
            batch_size=256,
            n_epochs=6,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            target_kl=0.015,
            ent_coef=0.0,
            verbose=1,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-2.5),
        )

    if not scratch:
        # Phase 1: Behavior cloning from Analytic teacher inside the exact shielded loop
        obs = env.reset()
        xs, ys = [], []
        for _ in range((demos + workers - 1) // workers):
            actions = np.stack(env.env_method("teacher_action"))
            xs.append(obs.copy())
            ys.append(actions.copy())
            obs, _, _, _ = env.step(actions)
        x = torch.as_tensor(np.concatenate(xs), dtype=torch.float32)
        y = torch.as_tensor(np.concatenate(ys), dtype=torch.float32)
        opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
        for epoch in range(25):
            total_loss = 0.0
            n_batches = 0
            for idx in torch.randperm(len(x)).split(256):
                pred = model.policy.get_distribution(x[idx]).distribution.mean
                loss = (pred - y[idx]).square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss)
                n_batches += 1
            print(json.dumps(dict(bc_epoch=epoch, mse=round(total_loss / max(1, n_batches), 6))), flush=True)
        model.save(out / "cloned")
        record["demonstration_steps"] = int(len(x))
    else:
        record["demonstration_steps"] = 0

    record["ppo_steps_requested"] = int(steps)
    (out / "manifest.json").write_text(json.dumps(record, indent=2))

    # Phase 2: Shielded PPO training
    ckpt_every = 100_000 if scratch else 25_000
    ckpt_freq = max(1, ckpt_every // workers)
    model.learn(
        total_timesteps=steps,
        callback=CheckpointCallback(save_freq=ckpt_freq, save_path=str(out / "checkpoints")),
    )
    model.save(out / "ppo")
    env.close()
    record.update(status="completed", ppo_steps_actual=int(model.num_timesteps))
    (out / "manifest.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--demos", type=int, default=30_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args.seed, args.output, args.steps, args.demos, args.workers)
