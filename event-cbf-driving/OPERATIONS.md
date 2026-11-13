# Operations

Run commands from `event-cbf-driving/`. Install with
`python -m pip install -e ".[dev]"`; use `make check` before committing changes.

## Maintained commands

| Command | Purpose |
|---|---|
| `event-cbf doctor` | Check dependencies, config and checkpoint availability. |
| `event-cbf verify --profile default` | Check saved models and learned diagnostics. |
| `event-cbf eval --profile ci --router scripted` | Small offline evaluation. |
| `python -m event_cbf_driving.skills.train_ppo --help` | New single-critic training. |
| `event-cbf train --help` | Recorded dual-critic training and verification. |
| `make metadrive-check` | Physical integration tests and a small simulator run. |
| `make study` | Final comparison design with the current source version. |
| `make analyze` | Analyze those runs into a separate generated result folder. |
| `make report` | Rebuild the recorded final PDF from its saved tables. |
| `make milestone` | Rebuild the current two-page milestone PDF. |

See [the script index](scripts/README.md) for direct commands and optional tools.
PDF generation needs the `report` extra, or set `PDF_PYTHON` to an environment
with ReportLab. Camera inference additionally needs the pinned base weights,
Transformers and the recorded compatible PEFT environment.

## Outputs and reproduction

Generated runs belong under `runs/`; they are ignored by Git. The final recorded
report and summary tables are tracked under `../final-study/`. New analysis
defaults to `runs/final-analysis/` so it does not overwrite those saved summaries.
Set `RUN_DIR` and `RESULTS_DIR` in Make when using different run versions.

Experiments hash their source and settings. An existing source lock must match
before work resumes. After source changes, choose a fresh run directory; do not
edit old locks to bypass the check. Full saved-trace audits require the original
raw release, which is kept outside Git. See [history](../docs/history.md).

Use `event-cbf replay --help` for recorded action replay. Traces record the
requested, filtered and applied commands; physics runs also record achieved
acceleration. Check these when debugging movement or smoothness.

`make clean` removes tool caches and build files. It leaves runs and model
weights intact. Retired local experiment output is held in the external archive.
