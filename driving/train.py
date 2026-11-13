"""Train a lane driver with single-critic PPO, adapted from CleanRL.

Source: https://github.com/vwxyzjn/cleanrl/blob/
35896b1fefa9898b904f7e09bcbe6e168e15d2a9/cleanrl/ppo_continuous_action.py
MIT license: driving/CLEANRL_LICENSE.txt. Task-specific differences are documented
in the project README. No online services or new dependencies are required.

Run: python -m driving.train --skill CRUISE_RIGHT
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from driving.checkpoints import save_skill
from driving.network import CleanRLAgent, env_action_np
from driving.policy import SkillMeta
from driving.simulation import EmptyRoadVec

UPSTREAM_REVISION = "35896b1fefa9898b904f7e09bcbe6e168e15d2a9"
TARGETS = {"CRUISE_RIGHT": 0.0, "OVERTAKE_LEFT": 4.0}


def train(args):
    batch_size = args.num_envs * args.num_steps
    if (
        min(args.updates, args.num_envs, args.num_steps, args.update_epochs, args.num_minibatches)
        < 1
    ):
        raise ValueError("Training counts must be positive.")
    if batch_size % args.num_minibatches or batch_size // args.num_minibatches < 2:
        raise ValueError("Batch size must divide into minibatches with at least two samples.")
    output = (
        args.output or Path(__file__).resolve().parents[1] / f"runs/cleanrl-ppo/{args.skill}.pt"
    )
    metrics_path = output.with_suffix(".metrics.jsonl")
    if any(p.exists() for p in (output, output.with_suffix(".json"), metrics_path)):
        raise FileExistsError(f"Choose a fresh --output; files already exist for {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    target = TARGETS[args.skill]
    env = EmptyRoadVec(args.num_envs, target, np.random.default_rng(args.seed + 17))
    agent = CleanRLAgent()
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    shape = (args.num_steps, args.num_envs)
    observations = np.zeros((*shape, 3), np.float32)
    actions = np.zeros((*shape, 2), np.float32)
    logprobs, rewards, values, next_values, dones = [np.zeros(shape, np.float32) for _ in range(5)]
    obs = env.obs()
    history = []

    with metrics_path.open("x") as metrics:
        for update in range(args.updates):
            optimizer.param_groups[0]["lr"] = 3e-4 * (1 - update / args.updates)
            episode_returns = []
            # 1. Drive and remember observations, actions, and scores.
            for step in range(args.num_steps):
                observations[step] = obs
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(torch.from_numpy(obs))
                actions[step], logprobs[step], values[step] = (
                    action.numpy(),
                    logprob.numpy(),
                    value.numpy(),
                )
                ax, ay = env_action_np(np.clip(actions[step], -1, 1))
                obs, reward, done, failed, terminal_obs, finished = env.step(ax, ay)
                rewards[step], dones[step] = reward, done
                with torch.no_grad():
                    bootstrap = agent.get_value(torch.from_numpy(terminal_obs)).numpy()
                # A time limit may bootstrap; an actual failure must not.
                next_values[step] = np.where(failed, 0, bootstrap)
                episode_returns.extend(finished.tolist())

            # 2. Compare what happened with the critic's earlier prediction.
            advantages = np.zeros(shape, np.float32)
            last_advantage = np.zeros(args.num_envs, np.float32)
            for step in reversed(range(args.num_steps)):
                delta = rewards[step] + 0.99 * next_values[step] - values[step]
                last_advantage = delta + 0.99 * 0.95 * (1 - dones[step]) * last_advantage
                advantages[step] = last_advantage
            returns = advantages + values
            batch_obs = torch.from_numpy(observations.reshape(-1, 3))
            batch_actions = torch.from_numpy(actions.reshape(-1, 2))
            batch_logprobs = torch.from_numpy(logprobs.reshape(-1))
            batch_advantages = torch.from_numpy(advantages.reshape(-1))
            batch_returns = torch.from_numpy(returns.reshape(-1))
            batch_values = torch.from_numpy(values.reshape(-1))

            # 3. PPO: improve the driver while limiting each change.
            for _ in range(args.update_epochs):
                indices = rng.permutation(batch_size)
                minibatch_size = batch_size // args.num_minibatches
                for start in range(0, batch_size, minibatch_size):
                    mb = indices[start : start + minibatch_size]
                    _, new_logprob, _, new_value = agent.get_action_and_value(
                        batch_obs[mb], batch_actions[mb]
                    )
                    ratio = (new_logprob - batch_logprobs[mb]).exp()
                    advantage = batch_advantages[mb]
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    policy_loss = torch.maximum(
                        -advantage * ratio, -advantage * ratio.clamp(0.8, 1.2)
                    ).mean()
                    clipped_value = batch_values[mb] + (new_value - batch_values[mb]).clamp(
                        -0.2, 0.2
                    )
                    value_loss = (
                        0.5
                        * torch.maximum(
                            (new_value - batch_returns[mb]).square(),
                            (clipped_value - batch_returns[mb]).square(),
                        ).mean()
                    )
                    loss = policy_loss + 0.5 * value_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("PPO produced a non-finite loss.")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

            mean_return = float(np.mean(episode_returns)) if episode_returns else None
            if mean_return is not None:
                history.append(mean_return)
            row = {
                "update": update + 1,
                "steps": (update + 1) * batch_size,
                "mean_episode_return": mean_return,
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
            }
            metrics.write(json.dumps(row, allow_nan=False) + "\n")
            metrics.flush()
            print(json.dumps(row), flush=True)

    # Single-critic policies have no disagreement signal; runtime enforces this.
    meta = SkillMeta(args.skill, int(target > 0), target, 1.0, 0.0, 0.0, 1.0)
    save_skill(
        output,
        agent,
        meta,
        history=history,
        provenance={
            "agent": "cleanrl_ppo",
            "seed": args.seed,
            "updates": args.updates,
            "upstream_revision": UPSTREAM_REVISION,
            "training_args": vars(args) | {"output": str(output)},
        },
    )
    print(f"Saved {output}")
    return agent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=TARGETS, default="CRUISE_RIGHT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=400)
    parser.add_argument("--num-minibatches", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        train(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
