# Event-triggered CBF driving stack

A runnable three-tier highway stack:

**skill router (Tier 1) → frozen PPO skills (Tier 2) → latency/uncertainty-inflated CBF shield (Tier 3).**

The router is interrupt-only. It is queried when the active skill's critic
uncertainty crosses a calibrated threshold, or when the CBF shield has sustained
an override. Continuous PPO skills own the normal control loop and never see the
router decision. The shield grows its standoff with critic uncertainty and, only
while a router call is in flight, by `v · Δt`. Skill changes are blended.

This repository is the complete research package: scenarios, training,
checkpoints, nine evaluation modes, ablations, a regression gate, provenance,
episode replay, CI, and generated results. It is **research software**, not a
safety-certified driving system. See [SAFETY.md](SAFETY.md) for the threat model
and limits.

## Completed physical study

Read [FINAL_REPORT.md](FINAL_REPORT.md), the [six-page PDF](outputs/research-study/final-report.pdf),
and the [annotated three-layer demo](outputs/research-study/three-layer-demo.mp4).
The `metadrive-study` profile adds held-out uncertainty calibration with frozen
weights, a separate stopping-envelope backup, intended-lane checking and
replanning after a newly accepted intent. The final suite includes 90 nominal
comparisons, 42 traffic/delay stress episodes, a paired safety ablation and an
18-episode actual local-model pilot with delay-replayed rule controls.

The proposed nominal mean is 99.8% route completion with zero crashes/stalls;
all 42 stress episodes avoid crashes/stalls. The local model makes 62 calls
with zero fallbacks, but shows no established advantage over the rule router.
Infeasible barrier states remain. The pending margin can reduce progress at
multi-second delays. See [METADRIVE.md](METADRIVE.md) for reproduction and scope.
The native tables below remain a separate historical measurement.

## Measured results

Canonical run: `runs/production/production-v1`, production profile, scripted
router, 3 scenarios × 5 seeds × 9 modes = 135 episodes. Scenarios are
deterministic (`randomize: false`), so all five seeds produce the same episode;
the reported `ci95` is therefore 0.0, not evidence of statistical power.

| Scenario | Proposed route | Crash | Freeze | Router calls | Peak lateral jerk |
| --- | ---: | ---: | ---: | ---: | ---: |
| `two_trap` | **71.5%** | 0% | 0% | 5.0 | 14.5 m/s³ |
| `merge` | **100.0%** | 0% | 0% | 6.0 | 13.2 m/s³ |
| `dense_highway` | **97.3%** | 0% | 0% | 2.0 | 14.4 m/s³ |
| **All 15 episodes** | **89.6%** | **0%** | **0%** | **4.3** | **14.1 m/s³** |

The regression gate passed at the production floor (`min_route_pct=70.0`).

Key comparisons from the same run (route / crash / freeze):

| Scenario | Proposed | Unshielded skill | Static CBF | Fixed-rate VLM | Always-on VLM |
| --- | --- | --- | --- | --- | --- |
| `two_trap` | 71.5 / 0 / 0 | 35.4 / 100 / 0 | 34.5 / 0 / 100 | 62.9 / 100 / 0 | 62.8 / 100 / 0 |
| `merge` | 100 / 0 / 0 | 66.9 / 100 / 0 | 100 / 0 / 0 | 41.1 / 100 / 0 | 41.1 / 100 / 0 |
| `dense_highway` | 97.3 / 0 / 0 | 39.5 / 100 / 0 | 97.7 / 0 / 0 | 38.1 / 100 / 0 | 38.2 / 100 / 0 |

`proposed_no_blend` reaches higher route in `two_trap` (76.5% vs 71.5%) but
incurs roughly 9× the peak lateral jerk across all episodes (125.6 vs 14.1),
which is the reason blending is part of the proposed stack. Full tables, caveats, and
per-scenario metrics are in [RESULTS.md](RESULTS.md); machine-readable
aggregates are in [results.json](results.json).

**Research compatibility:** `configs/default.yaml` preserves the original
lane-corridor configuration. A proposed-only default-profile run
(`runs/default/default-proposed-v1`, 5 deterministic seeds) measures
`two_trap` 80.6%, `merge` 100%, and `dense_highway` 97.4% route, with 0 crashes,
0 freezes, and 5/6/2 router calls respectively. Reproduce it with:

```bash
event-cbf eval --profile default --modes proposed \
  --out-dir runs/default --run-id default-proposed-v1 --no-figures
```

**Honest caveat:** the production shield (`footprint` mode, `lateral_margin:
0.35`) is more conservative than the research configuration and lowers
two-trap route from 80.6% to 71.5%. The proposed stack also reports many
infeasible ticks and a negative minimum barrier value in `two_trap` while still
completing the route without collision or freeze; the monitor classifies this
as `DEGRADED`, not `EMERGENCY`. These are real limitations, documented in
[SAFETY.md](SAFETY.md), not hidden by the headline table.

## Install

Python ≥ 3.10. CPU PyTorch is sufficient.

```bash
cd event-cbf-driving
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[highway]"   # upstream highway-env adapter
pip install -e ".[local]"     # transformers/accelerate for the local LLM router
pip install -e ".[metadrive]" # pinned MetaDrive/Bullet physical backend
pip install -e ".[report]"    # PDF report generation (demo also needs ffmpeg)
```

## Quick start

```bash
# 1. Verify the environment, dependencies, router, and checkpoints.
event-cbf doctor

# 2. Validate checkpoint fingerprints and learned diagnostics.
event-cbf verify --profile default

# 3. Fast offline smoke evaluation (CI profile, one scenario, four modes).
event-cbf eval --profile ci --router scripted \
  --out-dir runs/ci --run-id ci-smoke

# 4. Canonical production evaluation + top-level RESULTS.md/results.json.
event-cbf eval --profile production --router scripted \
  --out-dir runs/production --run-id production-v1 --update-docs
```

Or use `make`: `make check`, `make smoke`, `make eval`, `make doctor`.

## Profiles

| Profile | Purpose |
| --- | --- |
| `configs/default.yaml` | Research defaults. Preserves the published behavior, including the lane-corridor shield and `shield_hold_ticks: 1`. Use this to reproduce the original numbers. |
| `configs/production.yaml` | Hardened runtime profile: footprint shield with lateral margin, longer sustained-override guard, pending margin on all calls, five seeds, no-crash/no-freeze gate, route floor 70%. |
| `configs/ci.yaml` | Fast, offline CI smoke: one scenario, four modes, one seed, no figures, route floor 60%. |
| `configs/metadrive.yaml` | Randomized physical MetaDrive evaluation, shared frozen skills, fresh-gap switch admission, interaction traces, matched shielded polling comparisons. |
| `configs/metadrive-live.yaml` | Explicit live inference in a background worker while the physical control loop continues with wall-clock pacing. |
| `configs/metadrive-study.yaml` | Completed physical study: frozen-weight calibration, stopping backup, intended corridor and intent-change replanning. |
| `configs/metadrive-local.yaml` | Actual pinned Qwen2.5-1.5B CPU inference, no fallback, six-second deadline; extends the study profile. |
| `configs/live-openai.yaml` | Explicit live profile for an OpenAI-compatible endpoint. One seed, core comparison modes, no silent fallback. Requires `OPENAI_API_KEY` or `EVENT_CBF_API_KEY`. |
| `configs/live-ollama.yaml` | Explicit live profile for a local Ollama daemon (`qwen2.5:0.5b` by default). No Python LLM client required. |
| `configs/live-local.yaml` | Explicit live profile for a local Hugging Face transformers model (`Qwen/Qwen2.5-0.5B-Instruct` by default). Requires the `local` extra and downloads the model on first use. |

Profiles may be extended with any `--override KEY=VALUE` (repeatable):

```bash
event-cbf eval --profile production \
  -o eval.seeds=[0,1] -o shield.lateral_margin=0.2 \
  --scenarios two_trap --modes proposed
```

Unknown keys, unknown profiles, and malformed values are rejected. The complete
effective config is embedded in the run manifest.

## CLI

| Command | Purpose |
| --- | --- |
| `event-cbf eval` | Multi-mode, multi-scenario, multi-seed evaluation; writes `manifest.json`, `results.json`, `report.md`, figures, and optional episode traces. |
| `event-cbf router-check` | Send one synthetic interrupt and report live/fallback status, model, latency, token usage, and health. |
| `event-cbf train` | Reproduce the recorded dual-critic lane-skill training and uncertainty calibration. |
| `event-cbf verify` | Validate checkpoint fingerprints, duplicates, and learned diagnostics without training. |
| `event-cbf replay` | Strictly validate an episode artifact and replay its recorded action log. |
| `event-cbf doctor` | Check Python, dependencies, config, router construction, and checkpoint health. |

Stable exit codes:

| Code | Meaning |
| ---: | --- |
| 0 | Success |
| 1 | Unexpected/internal failure |
| 2 | Config or validation error |
| 3 | Checkpoint error |
| 4 | Router error |
| 5 | Safety invariant failure |
| 6 | Evaluation regression (crash, freeze, or route below floor) |
| 7 | Artifact/replay error |
| 130 | Interrupted |

Run `event-cbf <command> --help` for the full flag set.

## Reproducing and auditing a run

```bash
# Record full per-tick traces and actions for every episode.
event-cbf eval --profile ci --record-episodes --out-dir runs/ci --run-id audit-1

# Validate one artifact without re-running the environment.
event-cbf replay runs/ci/audit-1/episodes/two_trap-proposed-seed0.json --validate-only

# Reapply its exact action log and compare the final state.
event-cbf replay runs/ci/audit-1/episodes/two_trap-proposed-seed0.json
```

Every run artifact carries package version, git commit/branch/dirty flag,
Python/platform/dependency versions, config SHA-256, router description and
health, checkpoint fingerprints, and aliases. JSON is strict: `NaN` and
`Infinity` are rejected on write and read. Episode replay is deliberately open
loop; re-running the router would test the router, not the recording.

## Checkpoints and the `RETURN_RIGHT` alias

Two distinct legacy checkpoints are shipped:

- `CRUISE_RIGHT` — right-lane cruise;
- `OVERTAKE_LEFT` — left-lane overtake.

`RETURN_RIGHT` has the same lane target as `CRUISE_RIGHT`. No distinct
`RETURN_RIGHT.pt` exists; the loader treats it as an explicit documented alias
of `CRUISE_RIGHT` and renames the returned policy. The duplicate historical
checkpoint is preserved under `checkpoints/legacy/`. `checkpoints/validation.json`
is the authoritative verification artifact; `train_summary.json` is historical
and stale. Legacy checkpoints load with a compatibility warning and are
fingerprint-checked. Details: [checkpoints/README.md](checkpoints/README.md).

## Repository layout

```text
event_cbf_driving/
  config.py       typed config, profiles, dotted overrides
  runtime.py      episode schema, atomic recording, open-loop replay
  cli.py          eval / router-check / train / verify / replay / doctor
  env/            two_trap, merge, dense_highway, highway adapter, CARLA stub
  skills/         focused training, network, driver and checkpoint modules; see skills/README.md
  router/         scripted, OpenAI-compatible, local transformers/Ollama, dry-run
  shield/         inflated CBF and SafetyMonitor gate
  stack/          proposed controller, baselines, blending, result schema
  eval/           runner, metrics, reporting, figures
  training.py     training/verification implementation used by the CLI
scripts/          thin standalone wrappers
configs/          default / production / ci / live-*
checkpoints/      skill weights, validation artifact, legacy duplicates
tests/            unit, integration, CLI, runtime, and evaluation tests
```

## Router backends

| Backend | Behavior |
| --- | --- |
| `scripted` | Deterministic lane-threat rule with configured latency. Used for all canonical results. |
| `dryrun` | Forced deterministic OpenAI-compatible dry run. Always offline; exercises the real prompt/HTTP JSON/parse path through an injected transport. |
| `openai` | Live OpenAI-compatible chat completions. Requires explicit opt-in (`router.allow_live: true` or `--allow-live`); otherwise construction fails with a config error. |
| `local` | Live local model: Ollama or Hugging Face transformers, selected by `router.local_backend`. Also requires explicit opt-in. |
| `auto` | Conservative: live OpenAI-compatible calls only when live is explicitly allowed **and** a key is present; otherwise dry-run. |

Live calls are never implicit. `configs/default.yaml`, `configs/production.yaml`,
and `configs/ci.yaml` all set `allow_live: false`; every canonical number in
this repository is produced by the offline `scripted` backend.

## Live LLM routing

Live support is first-class but deliberately separate from the published
results. The factory, evaluation runner, and CLI share one gate: `openai` and
`local` are rejected unless `router.allow_live` is true (profiles) or
`--allow-live` is passed (CLI). This prevents a stray key in the environment
from turning an offline evaluation into a paid/live run.

### OpenAI-compatible endpoint

```bash
export OPENAI_API_KEY=...                 # or EVENT_CBF_API_KEY
# Optional: export OPENAI_BASE_URL=https://your-endpoint/v1
# Optional: export EVENT_CBF_OPENAI_MODEL=gpt-4o-mini

event-cbf router-check --profile live-openai --require-live
event-cbf eval --profile live-openai --run-id live-openai-v1
```

`router-check` sends one synthetic interrupt and exits 4 unless a real model
answers when `--require-live` is set. It reports the skill, model, latency,
fallback/error, and provider token usage. `eval` performs the same one-request
preflight before spending money on a multi-episode run; use
`--no-live-preflight` only for diagnostics. Add `--require-live` to `eval` to
fail if any call fell back.

### Ollama (local daemon)

```bash
ollama serve
ollama pull qwen2.5:0.5b
event-cbf router-check --profile live-ollama --require-live
event-cbf eval --profile live-ollama --run-id live-ollama-v1
```

### Local transformers

```bash
pip install -e ".[local]"
event-cbf router-check --profile live-local --require-live
event-cbf eval --profile live-local --run-id live-local-v1
```

The first run downloads `Qwen/Qwen2.5-0.5B-Instruct` (or
`EVENT_CBF_LOCAL_MODEL`). `trust_remote_code` is off by default. Set
`EVENT_CBF_LOCAL_DEVICE=cpu|cuda|mps|auto` as needed.

### Live telemetry and fallbacks

Every live response carries `live`, `model`, `finish_reason`, and normalized
`usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`). Call records and
episode aggregates expose `router_live_calls`, `router_tokens`, and
`router_models`; the run manifest and `results.json` include a
`router_telemetry` block plus router health token totals. `scripted`, `dryrun`,
and fallback answers report `live: false` and zero tokens.

The live profiles set `allow_fallback: false`: an unavailable key/daemon/model
aborts instead of silently producing scripted numbers. If you deliberately
enable fallback, every fallback is counted in `router_fallbacks` and reported
in the run report. Live-model quality is not claimed by
[RESULTS.md](RESULTS.md); run the live profile yourself and inspect its
artifacts.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — control loop, trigger state machine, shield, evaluation pipeline, provenance.
- [METADRIVE.md](METADRIVE.md) — physical backend, three-layer interaction, install/evaluation/replay, live inference, and limits.
- [METADRIVE_RESULTS.md](METADRIVE_RESULTS.md) — measured physical comparison, held-out delay tests, and validation evidence.
- [SAFETY.md](SAFETY.md) — fail-closed behavior, monitor policy, hazards, and limits.
- [OPERATIONS.md](OPERATIONS.md) — installation, configuration, artifacts, CI, troubleshooting, release checklist.
- [RESULTS.md](RESULTS.md) — generated measured results and caveats.
- [checkpoints/README.md](checkpoints/README.md) — checkpoint provenance and alias policy.
