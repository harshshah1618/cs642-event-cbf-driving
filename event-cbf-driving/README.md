# Event-triggered CBF driving stack

Runnable 3-tier highway stack: **skill router (Tier 1) → PPO skills (Tier 2) → latency/uncertainty-inflated CBF shield (Tier 3)**.

This is the full research package. The earlier kinematic cartoon under [`../two-trap-gauntlet/`](../two-trap-gauntlet/) is a proof-only artifact; all new work lives here.

## What is novel (honest)

**Wiring, not a new method.** Close published neighbors already cover event-triggered slow planners, critic-tightened barriers, delay margins, and skill mixing. What this repo ships is a single closed-loop stack that combines:

1. Interrupt-only VLM/skill routing (σ above a **training-calibrated** threshold, or sustained CBF override),
2. Separate continuous PPO skills that never see the router decision,
3. One CBF whose standoff grows with critic σ and, **only while a router call is in flight**, by \(v\cdot\Delta t\),
4. Smooth skill blend on switch,

with measured ablations on a shared scenario API (Two-Trap Gauntlet, merge, dense highway). See [`../method-novelty.md`](../method-novelty.md).

## Install

```bash
cd event-cbf-driving
pip install -r requirements.txt
pip install -e .
# Optional LLM backends (M2):
pip install 'transformers>=4.40' 'accelerate>=0.30' 'openai>=1.0'
```

Requires Python ≥ 3.10, CPU torch is enough. `highway-env` is a dependency for the optional upstream factory; primary scenarios use a shared 50 Hz gymnasium API compatible with highway-env continuous control.

## Train

```bash
cd event-cbf-driving
PYTHONPATH=. python scripts/train_skills.py --updates 160
```

Trains `CRUISE_RIGHT`, `OVERTAKE_LEFT`, and `RETURN_RIGHT` with dual-critic PPO on an empty two-lane road, calibrates interrupt thresholds on the training distribution, and writes:

- `checkpoints/*.pt`
- `media/full-stack/training-curves.png`

## Eval

### M1 (scripted router)

```bash
PYTHONPATH=. python scripts/eval_stack.py --router scripted --fig-prefix m1
```

### M2 (LLM router — commands that worked on this machine)

No API key required. `--router openai` uses an OpenAI-compatible dry-run that still runs **prompt → mock HTTP chat/completions → parse_skill → skill** through the same interrupt contract (σ / CBF residual → 300 ms pending → blend):

```bash
cd event-cbf-driving
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python scripts/eval_stack.py --router openai --fig-prefix m2
```

Measured M2 gauntlet proposed: **route 80.9%, 5 router calls, 0 crash, no freeze** (vs M1 scripted 80.6% / 9 calls). See [`RESULTS.md`](RESULTS.md) section **M2 router**.

Figures: `../media/full-stack/m2-gauntlet-*.png` (optional; not shipped in this zip)

#### Live OpenAI-compatible API

```bash
export OPENAI_API_KEY=...          # or EVENT_CBF_API_KEY
# optional:
export OPENAI_BASE_URL=https://api.openai.com/v1   # or EVENT_CBF_BASE_URL
export EVENT_CBF_OPENAI_MODEL=gpt-4o-mini
PYTHONPATH=. python scripts/eval_stack.py --router openai --fig-prefix m2
```

#### Real local transformers (CPU)

Downloads `Qwen/Qwen2.5-0.5B-Instruct` by default (override with `EVENT_CBF_LOCAL_MODEL`). End-to-end interrupt path works; **0.5B free-form policy quality is weak on the gauntlet**, so closed-loop M2 numbers above use the openai dry-run. Smoke:

```bash
export EVENT_CBF_LOCAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct   # default
PYTHONPATH=. python - <<'PY'
from event_cbf_driving.router.factory import make_router
from event_cbf_driving.env.types import VehicleState, WorldSnapshot
from event_cbf_driving.router.base import RouterRequest
r = make_router("local")
print(r.describe())
snap = WorldSnapshot(t=1, ego=VehicleState(60,0,20), others=[VehicleState(75,0,0,speed_limit=0.01)], route_length=200)
print(r.submit(RouterRequest(1,"CRUISE_RIGHT",snap,4,"sigma")))
PY
```

Ollama alternative: set `EVENT_CBF_OLLAMA_URL` / `EVENT_CBF_OLLAMA_MODEL` and ensure the daemon is up; `LocalLLMRouter(backend="ollama")` probes `/api/tags`.

Router backends (`--router`):

| Value | Behavior |
| --- | --- |
| `scripted` | Lane-threat rule, 300 ms latency (M1) |
| `openai` | Live Chat Completions if API key set; else **dry-run mock HTTP** |
| `local` | Real HF transformers (or Ollama); else dry-run |
| `dryrun` | Forced openai dry-run |
| `auto` | API key → openai; else dry-run |

## Layout

```
event_cbf_driving/
  env/        SkillLaneEnv, TwoTrapGauntletEnv, MergeScenarioEnv, DenseHighwayEnv, CarlaAdapterStub
  skills/     Dual-critic PPO, calibrate_threshold, checkpoints I/O
  shield/     InflatedCBFShield (γσ + η·I(pending)·v·Δt)
  router/     Scripted / OpenAI-compat / local LLM / dry-run (same interrupt API)
  stack/      DrivingStack + blend
  eval/       Metrics + figures
scripts/      train_skills.py, eval_stack.py, run_all.py
tests/        Dry-run LLM prompt→parse unit tests
checkpoints/  Saved skill weights
```

## CARLA

[`event_cbf_driving/env/carla_stub.py`](event_cbf_driving/env/carla_stub.py) documents the adapter surface (`snapshot()` / `step(ax,ay)`). Not required for CI; highway-compatible envs are the primary path.

## One-shot

```bash
PYTHONPATH=. python scripts/run_all.py
```
