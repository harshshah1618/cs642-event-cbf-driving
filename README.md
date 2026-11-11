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

The completed physical study is in the [final report](event-cbf-driving/FINAL_REPORT.md)
([PDF](event-cbf-driving/outputs/research-study/final-report.pdf),
[three-layer demo](event-cbf-driving/outputs/research-study/three-layer-demo.mp4)).
The proposed stack averages 99.8% route completion in 15 nominal episodes;
all 42 expanded stress episodes avoid crashes and stalls. The actual local
LLM pilot makes 62 model calls with zero fallbacks, but establishes no advantage
over rule routing. Negative barrier states remain; these are research results.
Use `metadrive-study` for this study; the native results below are historical,
separate experiments.

## Quick start

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

## Measured results

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
- [`docs/`](docs/) — proposal background, method novelty, and source feasibility notes.

## Repository map

| Path | What it is |
| --- | --- |
| `event-cbf-driving/` | Main package: scenarios, training, checkpoints, router, shield, controller, evaluation, CLI, tests, CI. |
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
