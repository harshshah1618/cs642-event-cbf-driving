# Driving package

The maintained code is under `event_cbf_driving/`. The router selects a maneuver,
the driver requests acceleration, and the shield and final monitor check each
command before the simulator applies it.

## Install and check

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
make check
```

`make check` runs lint, type checks, tests, checkpoint validation and a small
offline evaluation. `make help` lists the maintained targets. The main entry
point is `event-cbf`; use `event-cbf --help` and each subcommand's `--help`.

## Train a driver

```bash
python -m event_cbf_driving.skills.train_ppo --skill CRUISE_RIGHT --seed 0
python -m event_cbf_driving.skills.train_ppo --skill OVERTAKE_LEFT --seed 1
```

This CleanRL-based command writes fresh models to `runs/cleanrl-ppo/`. It refuses
to overwrite output files. It does not replace the measured checkpoints.
See [the lane-driver guide](event_cbf_driving/skills/README.md) for the short
training loop, file layout and differences from upstream CleanRL.

`event-cbf train` retains the recorded dual-critic training/calibration path.
Single-critic models cannot provide critic disagreement; the runtime requires a
traffic-based routing trigger when those models are used.

## Evaluate

```bash
event-cbf doctor
event-cbf verify --profile default
event-cbf eval --profile ci --router scripted --out-dir runs/ci --run-id manual-smoke
```

Profiles live in `configs/`. The small native simulation is used for quick
checks; [MetaDrive](METADRIVE.md) supplies the physical simulator.
The [final study](../final-study/README.md) documents the completed comparison.
Use [scripts/README.md](scripts/README.md) for its current runners and analysis.

## Code map

| Module | Job |
|---|---|
| `skills/` | Networks, training, saved policies and checkpoint loading. |
| `router/` | Scripted, text and image-based maneuver choices. |
| `stack/controller.py` | Driving loop, pending answers, switching and request timing. |
| `stack/scheduling.py` | Traffic-based rules for when to ask. |
| `shield/` | Action filtering and final safety monitor. |
| `env/` | Native scenarios and physical simulator adapter. |
| `eval/` | Shared experiment execution, metrics and reports. |
| `cli.py` | Installed commands. |

`RETURN_RIGHT` uses the right-lane policy unless a separate checkpoint exists.
The driver and shield receive simulator state; this is not camera-only driving.
Control ticks in simulation do not establish hard real-time operation.

Further detail: [architecture](ARCHITECTURE.md), [safety](SAFETY.md),
[operations](OPERATIONS.md), [checkpoints](checkpoints/README.md), and
[history/recovery](../docs/history.md).
