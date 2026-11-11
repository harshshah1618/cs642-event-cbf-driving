# Delayed maneuver scheduling: reproducible research study

This work is isolated on `codex/latency-aware-research` in
`/home/harsh/Projects/cs642-event-cbf-research`. The original dirty
`harsh-testing` checkout at `/home/harsh/Projects/cs642-event-cbf-driving` was
preserved. Commit `50bcbde` captures its source before this study.

The research question is whether observed maneuver opportunities and planner
latency improve the safety/progress/call tradeoff over reactive triggering and
well-tuned polling. The method is an empirical scheduling hypothesis. It is not
a new CBF theorem, a verified novel algorithm, or a CARLA leaderboard submission.

## What was implemented

- `stack/scheduling.py`: explicit two-lane opportunity and deadline triggers.
- `StackConfig.trigger_policy`: legacy, deadline, no-latency, TTC, usefulness.
- `configs/metadrive-deadline.yaml`: runnable integration with the original stack.
- `stack/road_planning.py`: road-relative lane/speed candidate planning, observed
  footprint and cut-in checks, PID/IDM tracking, and a short world-coordinate
  forecast brake backup for upstream MetaDrive roads. Its models are approximate.
- Matched delayed-decision experiments, global and condition-specific polling
  selection, a cheap gated-polling opponent, and initial timer-offset robustness.
- Exact-cell, source, checkpoint, runtime-metadata and replay audits.

The native driver remains the two existing frozen PPO lane regulators. The road
suite uses one common classical driver/planner across scheduling methods. Upstream
IDM and the released PPO expert are separate whole-controller references. There
is no newly trained perception model or independent policy-training-seed study.

## Artifacts

`results/research-report.md` and `.pdf` contain the completed measured assessment.
The randomized-initial-offset comparison cuts planner calls by 63.0% versus
condition-tuned polling, with observed success 100% versus 99.17%. This scoped
win does not extend to every suite or establish benefit from the latency term.
`results/statistics.json` contains seed-clustered paired
intervals, including every matched latency ablation. `condition-tables.md` reports
every scenario/environment and delay separately. `validation-frontiers.json`
contains every validation rate/buffer candidate, not only selected methods.

`results/data/*.jsonl.gz` preserves complete raw validation/test rows, with hashes
in `audit.json`. Road rows include full resolved upstream configurations and map
geometry hashes. `*-locked-selection.json` includes every condition-specific
polling rate, source/checkpoint hashes, runtime metadata and freeze timestamp.
Representative complete action traces and replay checks live under `results/traces`.

Development failures and pre-freeze road versions remain under the worktree's
`event-cbf-driving/runs/`; they are excluded from the final locked comparisons.
The original historical research artifacts remain in the original checkout.

## Reproduce

Use Python 3.12.13 and the exact versions in `environment-freeze.txt` for closest
local reproduction. Create your own environment and install the package with
`.[dev,metadrive,report]`; `.[local]` is needed only for real-model diagnostics.
The worktree reuses the pre-existing local environment by an ignored symlink.
Always set `PYTHONPATH=.` from the package directory to select this checkout.
For analysis of the packaged results without rerunning physics, first run
`.venv/bin/python scripts/restore_research_data.py`, then
`PYTHONPATH=. .venv/bin/python scripts/research_report.py`. The restore command
refuses to replace differing existing files. Read `ASSESSMENT.md` for a plain
language explanation and candid interpretation.

```bash
cd /home/harsh/Projects/cs642-event-cbf-research/event-cbf-driving
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check event_cbf_driving tests scripts
.venv/bin/python -m mypy event_cbf_driving

# Fresh study: validation must finish before freezing or starting tests.
.venv/bin/python scripts/research_isolation.py validation --workers 4
.venv/bin/python scripts/research_freeze.py isolation
.venv/bin/python scripts/research_isolation.py test --workers 4

.venv/bin/python scripts/research_roads.py validation --workers 4
.venv/bin/python scripts/research_freeze.py roads
.venv/bin/python scripts/research_roads.py test --workers 4

.venv/bin/python scripts/research_phase.py validation --workers 4
.venv/bin/python scripts/research_freeze.py phase
.venv/bin/python scripts/research_phase.py test --workers 4

# Fixed-controller ablation and exploratory clock mechanism control.
.venv/bin/python scripts/research_controller_ablation.py
.venv/bin/python scripts/research_clock_control.py run --workers 2
.venv/bin/python scripts/research_clock_control.py analyze

.venv/bin/python scripts/research_trace.py
.venv/bin/python scripts/research_report.py
```

An existing lock cannot be overwritten. Held-out runners verify source and
checkpoint hashes and the validation data before starting. They resume completed
jobs only within that frozen experiment. To start a materially changed study,
preserve the old directory and start a separately versioned run; do not append
rows produced by a different implementation. CLI `--output` paths must have their
own frozen selection; the default freeze script targets the standard directories.

For a quick independent episode:

```bash
.venv/bin/python -m event_cbf_driving.cli eval --profile metadrive-deadline \
  --modes proposed fixed_rate_shielded --seeds 42 --no-figures
```

The full studies inject simulated planner latency while keeping fresh feedback.
They do not claim those waits are measured inference times. A real pinned Qwen
1.5B CPU integration probe and an exploratory single-seed closed-loop run are
stored separately under `runs/research-live*`; these are not pooled into the
controlled benchmark or treated as evidence of language-model superiority.

The 1,200-episode controller ablation observes the same success for the frozen
PPO and fixed PD banks under a common sigma-free filter. The clock control adds
1,080 runs and reuses 360 phase rows: at 0.1 Hz a randomized initial offset drops
success by 21.67 percentage points; at 1 Hz the effect reverses. These are
exploratory mechanism findings, not evidence of a novel algorithm. The spatial
gate candidate in `audits/spatial-gate.md` has development data only and no new
held-out result.

## Interpretation rules

- Success means no collision and at least 99% route in the native suite; upstream
  success uses its actual arrival flag and no observed collision.
- Keep standard/no-added-delay upstream results separate from delay stress.
- Pair by scene seed jointly across delays and both upstream environments, whose
  geometry is shared for a seed. Report the actual distinct geometry count.
- The matched no-latency control copies the chosen deadline buffer and removes
  only latency. Old unmatched development controls are not the final ablation.
- Gated 10Hz polling has a shorter cooldown than event policies; it is an
  intentionally strong comparator, not an isolated deadline ablation.
- The random timer experiment varies the initial rolling-timer offset. It does
  not simulate an absolute fixed wall-clock schedule or preemptive inference.
- Call savings are not automatically CPU savings. The native rule router is
  cheap. Road rows distinguish planner/scheduler CPU, elapsed section time, and
  total control CPU. Physics simulation time is outside those control timings.
- Frozen controllers do not measure independent training variability. Zero
  sampled collisions and zero-width bootstrap intervals do not prove safety.
- No result here establishes a win on SafeBench, Bench2Drive, or against faithful
  CERT-Replan/FASIONAD implementations. Those systems are discussed as prior work.

## Independent review and primary prior work

Astra reviewed experiment correctness and research positioning. Its audit files
are preserved under `audits/`; the implementation corrected material findings
before the final road/phase tests. The report audit added exact-cell checks,
runtime metadata/inventory verification, full validation frontiers, and separate
no-delay/stress reporting.

- [MetaDrive environments](https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html)
- [CERT-Replan](https://arxiv.org/abs/2610.09302)
- [RL event-triggered MPC](https://arxiv.org/abs/2208.10302)
- [Learning Robot Metareasoning](https://arxiv.org/abs/2610.23662)

These papers are close enough that a broad claim such as "first latency-aware
event-triggered planner" is unsupported. Benchmark improvements and mechanistic
findings must be stated at the scope supported by the recorded experiments.
