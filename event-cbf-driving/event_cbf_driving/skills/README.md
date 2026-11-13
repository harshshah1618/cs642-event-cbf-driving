# Lane drivers: start here

For new training, read **[train_ppo.py](train_ppo.py)**. It has three steps:
drive and collect scores, work out which actions helped, then make a limited
PPO update. The network is in [cleanrl_agent.py](cleanrl_agent.py).

```bash
# Run from event-cbf-driving/ in the project Python environment.
python -m event_cbf_driving.skills.train_ppo --skill CRUISE_RIGHT --seed 0
python -m event_cbf_driving.skills.train_ppo --skill OVERTAKE_LEFT --seed 1
```

Files go under `runs/cleanrl-ppo/`, separate from the existing checkpoints.
Each skill produces a `.pt` model, a `.json` sidecar and `.metrics.jsonl`
training scores. Existing output files are never overwritten; use `--output`
with a fresh path for another run. These are practice-road policies, not new
benchmark results. They need evaluation before replacing the measured drivers.

For a quick wiring check, not a useful trained driver:

```bash
python -m event_cbf_driving.skills.train_ppo --updates 2 \
  --num-envs 2 --num-steps 16 --num-minibatches 2 --update-epochs 2 \
  --output runs/cleanrl-smoke/CRUISE_RIGHT.pt
```

## What each file does

| File | Job |
|---|---|
| `train_ppo.py` | New, single-critic PPO training command. |
| `cleanrl_agent.py` | The new network: a driver and one score estimator. |
| `lane_env.py` | Practice-road reset, movement and reward. |
| `policy.py` | Runtime `act(y, vy, vx)` interface and skill metadata. |
| `checkpoints.py` | Save/load models, validate metadata and detect damaged files. |
| `model.py`, `legacy_ppo.py` | The recorded dual-critic network and trainer. |
| `diagnostics.py` | Lane-holding checks and legacy disagreement calibration. |
| `loader.py` | Load both skills and the documented return-right alias. |
| `ppo.py` | Compatibility imports for existing callers; no training implementation. |

## CleanRL source and task-specific changes

The agent and PPO update are adapted from CleanRL's
[`ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/35896b1fefa9898b904f7e09bcbe6e168e15d2a9/cleanrl/ppo_continuous_action.py),
pinned at `35896b1fefa9898b904f7e09bcbe6e168e15d2a9`. Its MIT notice is in
[`licenses/CleanRL-MIT.txt`](../../licenses/CleanRL-MIT.txt).

This is an adaptation, not the unmodified upstream benchmark script. It uses
our existing vectorized empty-road task and fixed observation scaling instead
of Gym wrappers and running observation/reward normalization. Gaussian actions
are clipped and mapped to our physical acceleration limits. Time-limit endings
bootstrap from the last observation; actual failures do not. The PPO update
uses clipped policy/value losses, minibatch advantage normalization, learning
rate annealing and gradient clipping. Online tracking, video, model uploads and
the extra CLI/logging dependencies are omitted.

The new agent has **one critic**. `sigma` is zero because it has no disagreement
estimate. The runtime rejects uncertainty-based routing with this model; choose
a traffic-based trigger (`deadline`, `benefit` or `barrier`). Its compatibility
`act` method returns the same value twice to fit older evaluation helpers; this
does not create a second critic. Saved models explicitly record their network
architecture so the loader does not guess.

The existing `event-cbf train` command still runs the recorded dual-critic
training and calibration path. Existing checkpoints and historical import paths
remain supported. No trained weights or recorded results were changed by this
cleanup. Reproduce frozen studies from commit `8c85e1c` (or the original release
archive); source locks intentionally reject changed source. New experiments
must use fresh output directories.
