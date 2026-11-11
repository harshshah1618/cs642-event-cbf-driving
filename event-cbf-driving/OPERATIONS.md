# Operations

How to install, validate, run, inspect, and release this package.

## Requirements

- Python ≥ 3.10 (CI uses 3.11; the reference development environment is 3.12)
- CPU PyTorch is sufficient; CUDA is optional
- Several GB for the Python environment; MetaDrive assets, saved episodes and
  optional local-model weights need additional space (the tested 1.5B model
  alone is about 3 GB)
- Linux/macOS shell examples; Windows paths are not covered

The native canonical evaluation is fully offline: the `scripted` router needs
no API key and scenarios are deterministic. Physical study scenarios randomize
traffic by seed; actual local-model evaluation is a separate explicit profile.
See [METADRIVE.md](METADRIVE.md) and [FINAL_REPORT.md](FINAL_REPORT.md).

## Install

### Editable development install

```bash
cd event-cbf-driving
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

If PyPI's default torch wheel is large or unsuitable, install CPU torch first:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[highway]"   # highway-env adapter
pip install -e ".[local]"     # transformers + accelerate for the local router
pip install -e ".[metadrive]" # pinned physical backend (Python 3.11/3.12)
pip install -e ".[report]"    # measured PDF; system ffmpeg needed for demo
pip install -e ".[all]"       # both, plus dev tools
```

### Verify the installation

```bash
event-cbf doctor
event-cbf verify --profile default
```

`doctor` checks imports, config construction, checkpoint health, and router
construction. `verify` checks checkpoint fingerprints, duplicates, legacy
schema, and the minimum number of learned distinct skills. Both exit non-zero
on failure.

Expected legacy warning: `CRUISE_RIGHT.pt` and `OVERTAKE_LEFT.pt` are schema-v1
files without v2 provenance. The loader accepts them in compatibility mode and
fingerprint-checks them. `RETURN_RIGHT` is reported as an alias of
`CRUISE_RIGHT`; see [checkpoints/README.md](checkpoints/README.md).

## Configuration

### Precedence

1. `--config FILE` or `--profile {default,production,ci,live-openai,live-ollama,live-local}`
   selects the base file (mutually exclusive).
2. `--override KEY=VALUE` (repeatable) applies dotted changes.
3. Command-specific flags (`--seeds`, `--scenarios`, `--modes`, `--router`,
   `--ckpt-dir`, `--out-dir`, ...) apply last.

Without either flag, `configs/default.yaml` is used when present. Unknown
keys, unknown profiles, invalid enum values, and non-finite numbers are
rejected with exit code 2.

```bash
# Inspect the effect of an override via the produced manifest.
event-cbf eval --profile ci -o stack.min_call_ticks=20 \
  --out-dir runs/ci --run-id override-demo --no-figures

jq '.config.stack.min_call_ticks' runs/ci/override-demo/manifest.json
```

### Profiles

| Profile | Use |
| --- | --- |
| `default` | Reproduce research behavior: lane-corridor shield, `shield_hold_ticks: 1`, route floor 60%, no-crash/no-freeze gate. |
| `production` | Hardened runtime: footprint shield with 0.35 m lateral margin, 0.6 s lateral lookahead, `shield_hold_ticks: 10`, pending margin on all calls, emergency abort, five seeds, route floor 70%. |
| `ci` | One scenario, four modes, one seed, no figures, route floor 60%, `fail_on_fallback: true`. |
| `live-openai` | Explicit live OpenAI-compatible profile: one seed, core modes, no silent fallback, `allow_live: true`. |
| `live-ollama` | Explicit live local Ollama profile (`qwen2.5:0.5b`), one seed, core modes, no silent fallback. |
| `live-local` | Explicit live Hugging Face transformers profile (`Qwen/Qwen2.5-0.5B-Instruct`), one seed, core modes, no silent fallback. |

All canonical published results use `default`/`production` with the offline
`scripted` router. The `live-*` profiles are for exploratory live runs and never
update the tracked `results.json`/`RESULTS.md` unless you explicitly pass
`--update-docs`; do not do that for live claims without documenting the model,
endpoint, date, and fallback count.

See [SAFETY.md](SAFETY.md) for the behavioral trade-offs and measured caveats.

## Evaluation

### Canonical production run

```bash
event-cbf eval --profile production --router scripted \
  --out-dir runs/production --run-id production-v1 --update-docs
```

`--update-docs` rewrites the repository-level `results.json` and `RESULTS.md`
from this run. Without it, only the run directory is populated. Use a new
`--run-id` for each experiment; a random ID is generated when omitted.

### Fast offline run

```bash
event-cbf eval --profile ci --router scripted \
  --out-dir runs/ci --run-id ci-smoke
```

### Live LLM runs

Live calls are opt-in. The factory rejects `backend: openai` or `backend: local`
unless `router.allow_live` is true (or `--allow-live` is passed). `auto` only
chooses OpenAI-compatible live calls when live is allowed and a key exists.

```bash
# OpenAI-compatible
export OPENAI_API_KEY=...            # EVENT_CBF_API_KEY also works
event-cbf router-check --profile live-openai --require-live
event-cbf eval --profile live-openai --run-id live-openai-v1

# Ollama
ollama serve
ollama pull qwen2.5:0.5b
event-cbf router-check --profile live-ollama --require-live
event-cbf eval --profile live-ollama --run-id live-ollama-v1

# Local transformers
pip install -e ".[local]"
event-cbf router-check --profile live-local --require-live
event-cbf eval --profile live-local --run-id live-local-v1
```

`router-check` sends exactly one synthetic interrupt and exits 4 unless
`--require-live` is satisfied by a real model answer. It reports the selected
skill, backend/model, wall latency, fallback/error reason, token usage, and
health. Use it before an evaluation to avoid running a long job against a
missing key, stopped daemon, or unloaded model.

`eval` runs the same one-request preflight whenever live is requested and fails
fast with exit 4 if no live answer is available. `--no-live-preflight` skips the
probe for diagnostics only. `eval --require-live` additionally fails if the run
made no live calls or if any call used a fallback.

The live profiles set `allow_fallback: false`, so a failed model call is an
error rather than a silent scripted answer. If you override that deliberately,
fallbacks are still explicit: call records have `fallback_used`/`error`, and
aggregates/reports/manifests expose `router_fallbacks`, `router_live_calls`,
`router_tokens`, `router_models`, and the `router_telemetry` block. Offline
backends and fallbacks always report `live: false` and zero tokens.

### Selected scenarios, modes, seeds

```bash
event-cbf eval --profile production \
  --scenarios two_trap --modes proposed proposed_no_blend \
  --seeds 0 1 --no-figures --run-id focused-1
```

### Recording and replaying episodes

```bash
event-cbf eval --profile ci --record-episodes --out-dir runs/ci --run-id audit-1
ls runs/ci/audit-1/episodes/

event-cbf replay runs/ci/audit-1/episodes/two_trap-proposed-seed0.json \
  --validate-only
event-cbf replay runs/ci/audit-1/episodes/two_trap-proposed-seed0.json \
  --json
```

Replay resets the recorded scenario with the recorded seed and reapplies the
recorded `[ax, ay]` log. It is open loop: the router is not re-run. Divergence
beyond `--tolerance` (default `1e-6` m) is a failure (exit 7).

### Regression gate

The gate runs only when `eval.fail_on_regression` is true. For every scenario
with a `proposed` mode it checks:

- route mean ≥ `eval.min_route_pct` (or `--min-route-pct`);
- crash rate is zero unless `--allow-crash`;
- freeze rate is zero unless `--allow-freeze`;
- proposed router calls do not exceed the configured budget
  (`MAX_PROPOSED_ROUTER_CALLS`).

A failure is raised **after** artifacts are written, so the failing run remains
inspectable. Exit code is 6. Do not use `--no-fail-on-regression` in release
validation; use it only for diagnostics.

## Artifacts

```text
<out_dir>/<run_id>/   (default out_dir: runs/)
  manifest.json     package/git/python/dependency/config/router/checkpoint provenance
  results.json      schema version, aggregates, compact episode records, warnings
  report.md         human-readable tables and caveats
  figures/          generated plots when figures are enabled
  episodes/         full traces + action logs when --record-episodes is used
  logs/             reserved for run-scoped logs
```

- `results.json` is strict JSON: non-finite values fail the write.
- `manifest.json` is validated before the run is considered successful.
- All writes are atomic (same-directory temp file + rename).
- `runs/` is generated output and is git-ignored.
- The top-level `results.json` and `RESULTS.md` are only updated with
  `--update-docs`; they are release artifacts, not per-run scratch files.

### Reading a run

```bash
jq '.warnings' runs/production/production-v1/results.json
jq '.aggregates.two_trap.proposed.route_pct' runs/production/production-v1/results.json
jq '.config.eval.seeds, .git.dirty' runs/production/production-v1/manifest.json
jq '.extra.router_telemetry' runs/production/production-v1/manifest.json
sed -n '1,80p' runs/production/production-v1/report.md
```

`router_telemetry` is always present. Canonical offline runs report zero live
calls and zero tokens; a live run records the model name(s), live-call count,
fallback count, and prompt/completion/total token totals.

Aggregates contain `mean`, `std` (sample, `ddof=1`), `min`, `max`, `n`, and a
95% normal `ci95` (`1.96 · std / sqrt(n)`). With deterministic scenarios and
identical seeds, `std` and `ci95` are zero by construction.

## Training and verification

```bash
# Validate only: never mutates weights, writes checkpoints/validation.json.
event-cbf verify --profile default

# Train the default skills (two distinct policies; RETURN_RIGHT is an alias).
event-cbf train --profile default --updates 160 --skills CRUISE_RIGHT OVERTAKE_LEFT

# Override output locations.
event-cbf train --ckpt-dir /tmp/opencode/ckpts --fig-dir /tmp/opencode/figures
```

The CLI is a thin wrapper around the packaged implementation in
`event_cbf_driving/training.py`. Checkpoint writes are atomic; new checkpoints
carry schema-v2 provenance and a state-dict fingerprint. Legacy v1 files remain
readable in compatibility mode.

## Logging

- Human-readable logs go to stderr with a `LEVEL logger: message | fields`
  format.
- Set `EVENT_CBF_JSON_LOGS=1` or pass `--json-logs` for one JSON object per
  line, suitable for an aggregator.
- `--log-level DEBUG|INFO|WARNING|ERROR` overrides the config level.
- Run IDs are included in log records when a run is active.

Redirect logs when you need a durable copy:

```bash
event-cbf eval --profile production --run-id production-v1 \
  > /tmp/opencode/production.stdout 2> /tmp/opencode/production.log
```

## CI and Make targets

| Target | Action |
| --- | --- |
| `make install` | Editable install with dev extras. |
| `make test` | Full pytest suite. |
| `make lint` | Ruff correctness/import checks. |
| `make typecheck` | Mypy over the package. |
| `make verify` | Checkpoint validation. |
| `make smoke` | CI-profile offline evaluation. |
| `make eval` | Production-profile evaluation with docs update. |
| `make doctor` | Dependency/config/checkpoint diagnostics. |
| `make check` | `lint + test + verify + smoke`. |
| `make clean` | Remove generated runs and caches. |

The GitHub Actions workflow runs on every push/PR: CPU torch install, editable
install, ruff, pytest with coverage (`--cov-fail-under=60`), `verify --profile
ci`, `doctor --json`, and the offline CI evaluation. Artifacts are uploaded even
after a failure.

## Environment variables

| Variable | Used by | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI-compatible router | Primary API key (configurable via `router.api_key_env`). |
| `EVENT_CBF_API_KEY` | OpenAI-compatible router | Fallback key. |
| `OPENAI_BASE_URL` / `EVENT_CBF_BASE_URL` | OpenAI-compatible router | Endpoint override. |
| `EVENT_CBF_OPENAI_MODEL` | OpenAI-compatible router | Model override. |
| `EVENT_CBF_LOCAL_MODEL` | Local router | Hugging Face model id. |
| `EVENT_CBF_LOCAL_DEVICE` | Local router | `cpu`, `cuda`, `mps`, or `auto`. |
| `EVENT_CBF_OLLAMA_URL` | Local router | Ollama base URL. |
| `EVENT_CBF_OLLAMA_MODEL` | Local router | Ollama tag. |
| `EVENT_CBF_JSON_LOGS` | Logging | `1/true/yes/on` enables JSON logs. |

Secrets are read from the environment and are not written to the manifest.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `doctor` import failure | Activate the intended virtualenv; reinstall `-e ".[dev]"`. |
| `verify` fingerprint mismatch | Treat the checkpoint as corrupt/tampered. Do not silently retrain; inspect `checkpoints/validation.json` and restore a known-good file. |
| `RETURN_RIGHT` warning | Expected. The alias is documented; place a distinct `RETURN_RIGHT.pt` only if you actually train one. |
| Route below the gate | Inspect `report.md` and `results.json`; do not lower the floor to hide a regression. Use `--no-fail-on-regression` only for diagnosis. |
| Crash/freeze in proposed mode | Run `make check`, then inspect the affected episode with `--record-episodes` and compare `shield_hold_ticks`, `abort_on_emergency`, and shield mode. |
| Live backend rejected with `allow_live` | Expected. Set `router.allow_live: true` in the profile or pass `--allow-live`; use `dryrun`/`scripted` for offline runs. |
| Live router never used | Run `event-cbf router-check --profile <live-profile> --require-live`; inspect the reported fallback/error. Canonical results do not need a live backend. |
| Live eval aborts preflight | Fix the key/daemon/model first. The live profiles disable fallback by design; use `--no-live-preflight` only for diagnostics, not release runs. |
| Local model download fails | Pre-download the model or use Ollama; the router reports load/fallback reasons rather than silently replacing the model. |
| Token usage is zero on a live run | Some providers omit `usage`; the run is still live but telemetry is unavailable. Check `router_live_calls` and the provider response. |
| Matplotlib errors on a headless box | `MPLBACKEND=Agg` or `--no-figures`. |
| Evaluation too slow | Reduce `--seeds`, `--scenarios`, `--modes`; CI profile is the fast path. |
| Replay diverges | Confirm the artifact was recorded by this version, the scenario seed matches, and the environment is deterministic. Negative results are not ignored. |
| `runs/` appears in git status | It should be ignored by the package `.gitignore`; do not commit generated artifacts. |

## Release checklist

1. `make lint` and `make typecheck` pass.
2. `make test` passes.
3. `event-cbf doctor --json` reports `"ok": true`.
4. `event-cbf verify --profile default` passes and updates
   `checkpoints/validation.json` intentionally.
5. `make smoke` passes.
6. Run the canonical production evaluation with a fresh run ID and
   `--update-docs`; confirm exit code 0.
7. Inspect `report.md`: crashes, freezes, warnings, invalid results, fallback
   usage, and the regression gate.
8. Confirm `results.json` contains the `production` block and legacy keys, and
   `RESULTS.md` matches the run ID and numbers.
9. If shipping live support, run
   `event-cbf router-check --profile <live-profile> --require-live` with the
   intended endpoint/model, then a separate live evaluation. Inspect
   `router_telemetry`; do not mix live results into the canonical offline
   numbers.
10. Review `git diff` for accidental secrets, large artifacts, or generated
    caches.
11. Tag the release only after the docs and measured numbers agree.
