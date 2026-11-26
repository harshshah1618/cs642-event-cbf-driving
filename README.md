# Driving project

Ved's shared-workspace records are now available: [audited historical Qwen
results and corrected comparison](reports/irregular/ved-saved-audit-2026-11-25/README.md).

Latest result audit: [800 local irregular-scene control episodes](reports/irregular/local-controls-2026-11-25/README.md),
with saved raw rows and corrected whole-episode collision counts. Ved's
[historical irregular results](reports/irregular/final-test.md) retain their
original aggregates with the collision-accounting limitation marked.

> **SafeMetaDrive handoff:** Start with [TEAMMATE_HANDOFF.md](TEAMMATE_HANDOFF.md).
> Training records, datasets, resumable checkpoints and large weights are exported
> as checksummed private-release bundles; restore them with
> `python tools/restore_safemetadrive.py`. No Modal account is needed to inspect
> the results or run the local checks. The older straight-road stack below is a
> separate study.

The project has one root with a few folders grouped by purpose.

| Folder | What belongs here |
|---|---|
| **`driving/`** | Driving, planning, safety and learning code. |
| `tests/` | Automated checks. |
| `tools/` | Experiment runners, analysis, camera tools and report builders. |
| `configs/` | YAML profiles and the recorded study selection. |
| `models/` | Saved drivers: `native/`, `physical/` and `camera/`. |
| `reports/` | Milestone, full report, proposal and supporting research evidence. |
| `runs/` | Generated training/evaluation output; ignored by Git. |

There is no second project folder inside the project. Model files are visible
under `models/`; supporting report tables, audits and fonts are kept together in
`reports/study-evidence.zip`.

## Where to begin reading

1. **[driving/simulation.py](driving/simulation.py)** — how the car moves and how practice is scored.
2. **[driving/simple_driver.py](driving/simple_driver.py)** — a handwritten driving rule.
3. **[driving/policy.py](driving/policy.py)** and **[driving/network.py](driving/network.py)** — the learned driver.
4. **[driving/train.py](driving/train.py)** — the small CleanRL-based PPO training loop.
5. **[driving/safety.py](driving/safety.py)** — checking and adjusting movement commands.
6. **[driving/planner.py](driving/planner.py)** / **[driving/camera.py](driving/camera.py)** — choosing a maneuver.
7. **[driving/controller.py](driving/controller.py)** — connecting the pieces.

The remaining `env_` files cover three native test scenarios and the MetaDrive
physics adapter. `router_base.py` defines the request/response contract;
`planner.py` supplies the scripted reference and `camera.py` the actual image
router. The old saved drivers need the dual-critic architecture in `network.py`;
new training uses the single-critic agent in the same file.

## What was removed

The old text-only model backends, CARLA placeholder, highway wrapper, duplicate
practice environment, legacy trainer, compatibility imports and generic plotting/
evaluation framework have been retired. Their feature-specific tests and profiles
were retired with them. The maintained comparison still has its 15 methods in
`tools/integrated_eval.py`; safety, model loading, camera contracts, replay and
physical integration remain tested. Earlier experiments are recoverable from Git.

The pruning reduced `driving/` from 60 Python files / 16,433 lines to 37 files /
8,549 lines. Validation: 240 tests (85% coverage), model checks and replay, nine
physical smoke episodes, and before/after physical runs of all 15 methods on one
cut-in case (seed 8201, 1-second delay). Those 15 runs matched actions, positions,
maneuvers and safety settings exactly; this is a cleanup check, not a new
benchmark result.

`python -m driving` now exposes just `train`, `eval`, `verify` and `doctor`.
Evaluation writes episode recordings plus `results.json`; the full study and its
statistics use the tools below.

## Run

```bash
.venv/bin/python -m driving doctor
make check
make train SKILL=CRUISE_RIGHT SEED=0
```

For a fresh clone, use `uv venv` and `uv pip install -e ".[dev]"`.
`event-cbf` and `python -m driving` provide the same CLI. `make help` lists the
maintained commands; [tools/README.md](tools/README.md) explains the experiment tools.
Physical simulation needs the `metadrive` extra and `python -m metadrive.pull_asset`.
PDF building needs the `report` extra or a suitable `PDF_PYTHON`.

New single-critic training writes to `runs/cleanrl-ppo/` and refuses to overwrite
existing output. It does not replace the recorded models. Single-critic models
need a traffic-based routing trigger (`deadline`, `benefit` or `barrier`), since
they have no critic-disagreement estimate. The trainer is adapted from
[CleanRL](https://github.com/vwxyzjn/cleanrl/blob/35896b1fefa9898b904f7e09bcbe6e168e15d2a9/cleanrl/ppo_continuous_action.py);
its notice is in [driving/CLEANRL_LICENSE.txt](driving/CLEANRL_LICENSE.txt).

## Reports

- [Two-page milestone](reports/milestone-report.pdf) ([editable text](reports/milestone-report.md)).
- [Full final report](reports/final-report.pdf) ([editable text](reports/final-report.md)).
- [Plain-language findings](reports/FINDINGS.md).

The results do not establish that PPO or a language model is essential for the
tested tasks, or prove real-vehicle safety. Saved weights and report contents
were preserved during the cleanup. See [HISTORY.md](HISTORY.md) for source
snapshots and recovery instructions.
