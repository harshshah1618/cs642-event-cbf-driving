# Driving project

Everything is directly in this folder: code, tests, configuration and reports.

## Read the code in this order

1. **[simulation.py](simulation.py)** — how position and speed change, and how practice is scored.
2. **[simple_driver.py](simple_driver.py)** — a handwritten driving rule.
3. **[policy.py](policy.py)** and **[network.py](network.py)** — the trained driver's interface and network.
4. **[train.py](train.py)** — the small CleanRL-based PPO training loop.
5. **[safety.py](safety.py)** — checking and adjusting a proposed movement.
6. **[planner.py](planner.py)** / **[camera.py](camera.py)** — selecting a maneuver.
7. **[controller.py](controller.py)** — connecting the parts in the driving loop.

`main.py` is the command-line entry point. Files starting with `test_` are tests;
files starting with `env_` or `router_` support the simulators and model backends.
`legacy_` and `train_legacy` files preserve the measured older driver.

## Run

```bash
.venv/bin/python main.py doctor
make check
```

For a fresh clone, create an environment with `uv venv` and install with
`uv pip install -e ".[dev]"`.

Train a new single-critic driver with `.venv/bin/python train.py --skill CRUISE_RIGHT` or
`.venv/bin/python train.py --skill OVERTAKE_LEFT --seed 1`. This training code is adapted
from [CleanRL](https://github.com/vwxyzjn/cleanrl/blob/35896b1fefa9898b904f7e09bcbe6e168e15d2a9/cleanrl/ppo_continuous_action.py);
see `CLEANRL_LICENSE.txt`. It uses our practice task and fixed input scaling,
with no online tracking. It does not replace the measured drivers automatically.

Single-critic models have no critic-disagreement signal. Their routing must use
a traffic-based trigger such as `deadline`, `benefit` or `barrier`.
`make help` lists the other commands. Physical simulation needs the `metadrive`
extra and `python -m metadrive.pull_asset`; PDF building needs the `report` extra.

## Reports and saved assets

- [Milestone report](milestone-report.pdf) and [editable text](milestone-report.md).
- [Full final report](final-report.pdf), [editable text](final-report.md) and [plain-language findings](FINDINGS.md).
- `study-evidence.zip`: intact final-study tables, audits, protocol and report fonts.
- `checkpoints.zip`: recorded native-simulation driver bank.
- `physical-checkpoints.zip`: recorded physical-simulation driver bank.
- `camera-adapter.zip`: trained camera adapter, pinned prompt and model metadata.

Assets unpack into the system temporary cache when needed. New runs, training
outputs and analysis default to `../cs642-driving-runs/`, outside this folder.
Model weights and report contents were preserved during the layout change.

The recorded results do not establish that PPO or a language model is essential
for the tested tasks. Simulation results do not prove real-vehicle safety.
See [HISTORY.md](HISTORY.md) for older source and recovery instructions.
