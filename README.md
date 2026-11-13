# CS642 — Event-CBF Driving

A three-tier highway driving stack: an interrupt-only skill router, frozen PPO
lane skills, and a latency/uncertainty-inflated CBF safety shield.

The maintained system lives in [`event-cbf-driving/`](event-cbf-driving/).
Earlier sketches (`wip-minimal/`, `two-trap-gauntlet/`) are preserved for
history but are not the production entry point.

MetaDrive/Bullet is available as an optional physical backend, with randomized
traffic scenarios, recorded interaction traces for all three layers, matched
shielded polling baselines, and a nonblocking live-router path. See
[`METADRIVE.md`](event-cbf-driving/METADRIVE.md) for setup and scope.

## Latest results and reports

Start with the [two-page milestone report](docs/milestone-report.pdf)
([editable text](docs/milestone-report.md)) for a plain-language account of what
works, what remains unproved, and the proposed next steps.

The [final integrated study](final-study/README.md) compares 15 versions across
1,440 physical simulator drives, followed by 36 camera-model integration drives.
The integrated system finished all 72 short-delay conditions without crashes,
making 1.08 delayed requests per drive versus 75.82 for frequent polling
(98.6% fewer). These controlled requests use a traffic rule with a simulated
wait. Simpler rules also completed the routes; our particular trigger has not
proved best, and PPO or the language model has not proved essential here.

The actual camera-model check produced 22 real replies. Our system and a cheap
local rule each finished six of six traffic setups; timer routing finished five.
Different model response times prevent a clean claim that our trigger is better.
See the [full report](final-study/final-report.pdf),
[plain-language findings](final-study/EXPLAINED.md), and
[comparison tables](final-study/results/all-conditions.md).

Earlier studies are preserved under [research/](research/README.md),
[component-study/](component-study/README.md),
[vlm-study/](vlm-study/README.md), and [maneuver-study/](maneuver-study/README.md).
Their measurements describe separate experiments. The earlier physical study is
in [event-cbf-driving/FINAL_REPORT.md](event-cbf-driving/FINAL_REPORT.md).

## Quick start

For a readable explanation of the PPO code and the new CleanRL-based training
command, start with [the lane-driver guide](event-cbf-driving/event_cbf_driving/skills/README.md).

```bash
cd event-cbf-driving
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"

event-cbf doctor
event-cbf verify --profile default
event-cbf eval --profile ci --router scripted \
  --out-dir runs/ci --run-id ci-smoke
```

Canonical production evaluation:

```bash
event-cbf eval --profile production --router scripted \
  --out-dir runs/production --run-id production-v1 --update-docs
```

## Historical native-simulator results

`production-v1`: 3 scenarios × 5 deterministic seeds, 9 modes, scripted
router. Proposed stack across the three scenarios:

| Scenario | Route | Crash | Freeze | Calls |
| --- | ---: | ---: | ---: | ---: |
| `two_trap` | 71.5% | 0% | 0% | 5 |
| `merge` | 100% | 0% | 0% | 6 |
| `dense_highway` | 97.3% | 0% | 0% | 2 |

The research-compatible `default` profile reaches 80.6% on `two_trap`; the
production shield is more conservative. Scenarios are deterministic, so the
reported confidence intervals are zero and do not imply statistical power.
This is research software, not a safety-certified driving system.

## Documentation

- [`event-cbf-driving/README.md`](event-cbf-driving/README.md) — install, CLI, profiles, results.
- [`event-cbf-driving/ARCHITECTURE.md`](event-cbf-driving/ARCHITECTURE.md) — control loop and evaluation pipeline.
- [`event-cbf-driving/SAFETY.md`](event-cbf-driving/SAFETY.md) — fail-closed behavior, monitor policy, limitations.
- [`event-cbf-driving/OPERATIONS.md`](event-cbf-driving/OPERATIONS.md) — operations, artifacts, CI, troubleshooting.
- [`event-cbf-driving/RESULTS.md`](event-cbf-driving/RESULTS.md) — generated results.
- [`final-study/README.md`](final-study/README.md) — final comparison, evidence and reproduction instructions.
- [`docs/milestone-report.pdf`](docs/milestone-report.pdf) — current two-page milestone report.
- [`docs/`](docs/) — proposal background, method novelty, and source feasibility notes.

Rebuild the milestone PDF from the repository root with
`python docs/build_milestone_report.py` after installing ReportLab. The font
files are included under `final-study/fonts/`. Previous milestone reports are
kept in `docs/archive/`.

Raw final-study driving traces and the complete release ZIP remain in the local
delivery folder and are excluded from Git. A source clone includes the source,
reports, summaries, trained adapter and PPO checkpoints; the full raw release is
needed to re-audit the saved final traces. Base model weights and Python
environments are external. See the final study's reproduction instructions.

## Repository map

| Path | What it is |
| --- | --- |
| `event-cbf-driving/` | Main package: scenarios, training, checkpoints, router, shield, controller, evaluation, CLI, tests, CI. |
| `final-study/` | Latest integrated comparison, report, tables, audits and report fonts. |
| `research/`, `component-study/`, `vlm-study/`, `maneuver-study/` | Earlier experiment protocols, results and integration evidence. |
| `wip-minimal/` | Standalone early Two-Trap Gauntlet demo; no shared imports. |
| `two-trap-gauntlet/` | Earlier scripted gauntlet/training sketch. |
| `proposal/` | Course proposal (LaTeX + PDF). |
| `docs/` | Planning, novelty, and feasibility notes. |
| `media/` | Historical figures; production figures live under `event-cbf-driving/runs/` (generated). |

## Legacy demo

```bash
cd wip-minimal
pip install -r requirements.txt
python run_minimal.py
```

The legacy demo is intentionally independent and is not covered by the main
package's test, lint, or regression gates.
