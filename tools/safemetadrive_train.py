"""Train reference OmniSafe PPO-Lag/CPO without changing their update rules."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def train(algorithm, seed, steps, output, workers=4, critic_lr=5e-5, defaults=False, save_freq=25):
    import driving.safemetadrive.omnisafe_env  # noqa: F401 -- registers the native adapter
    import omnisafe
    from driving.safemetadrive.protocol import manifest

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        seed=seed,
        train_cfgs=dict(device="cpu", torch_threads=2, vector_env_nums=workers,
                        parallel=1, total_steps=steps),
        algo_cfgs=dict(steps_per_epoch=4000, update_iters=10, batch_size=256,
                       obs_normalize=False, reward_normalize=False, cost_normalize=False,
                       gamma=0.99, cost_gamma=0.99, lam=0.95, lam_c=0.95,
                       standardized_rew_adv=True, standardized_cost_adv=True),
        model_cfgs=dict(actor=dict(hidden_sizes=[256, 256], activation="tanh",
                                   lr=None if algorithm == "CPO" else 5e-5),
                        critic=dict(hidden_sizes=[256, 256], activation="tanh", lr=critic_lr)),
        logger_cfgs=dict(log_dir=str(output), use_wandb=False, use_tensorboard=False,
                         save_model_freq=save_freq),
    )
    if defaults:
        # OmniSafe 0.5 official algorithm/model configs; only the cost limit is set below.
        cfg.pop("model_cfgs")
        cfg["algo_cfgs"] = {}
    if algorithm == "PPOLag":
        cfg["lagrange_cfgs"] = dict(cost_limit=1.0)
    elif algorithm == "CPO":
        cfg["algo_cfgs"]["cost_limit"] = 1.0
    else:
        raise ValueError(algorithm)
    record = manifest("train", algorithm, cfg)
    record.update(algorithm_library="omnisafe==0.5.0", training_seed=seed,
                  requested_steps=steps, status="running")
    (output / "manifest.json").write_text(json.dumps(record, indent=2))
    agent = omnisafe.Agent(algorithm, "CS642SafeMetaDrive-v0", custom_cfgs=cfg)
    try:
        result = agent.learn()
    finally:
        agent.agent._env.close()
    record.update(status="completed", final_training_metrics=[float(x) for x in result])
    (output / "manifest.json").write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=["PPOLag", "CPO"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--critic-lr", type=float, default=5e-5)
    p.add_argument("--output", required=True)
    p.add_argument("--defaults", action="store_true", help="OmniSafe default model/learning rates")
    p.add_argument("--save-freq", type=int, default=25, help="epochs between checkpoints")
    args = p.parse_args()
    train(args.algorithm, args.seed, args.steps, args.output, args.workers, args.critic_lr,
          args.defaults, args.save_freq)
