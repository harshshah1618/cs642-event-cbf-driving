# Project plan: 3-tier event-triggered VLM + RL + inflated-CBF stack

## Goal

Build and measure a closed-loop driving stack that wires three known pieces—event-triggered VLM skill routing, fast RL skills, and a CBF whose headway grows with critic uncertainty and (while a model call is in flight) speed × call delay—then report how that wiring behaves against the closest published systems (FASIONAD / FASIONAD++ for uncertainty wake-up; CERT-Replan for barrier-risk replanning).

This is an engineering and evaluation project. It does **not** claim a new method. Novelty of each hook is already thin; the deliverable is a reproducible stack and head-to-head numbers, not a theory paper.

The two-trap gauntlet under `docs/two-trap-gauntlet/` is a scripted proof of concept only. This plan is for the full stack that replaces that script with a real sim path, a real LLM/VLM router, and multi-scenario baselines.

## Stack (what we build)

| Tier | Role | Rate |
| --- | --- | --- |
| Slow router | VLM/LLM picks a skill (e.g. cruise, overtake, return) | Event-triggered only |
| Fast skills | Separate RL policies; do not see the router decision as an input | Control rate (~50 Hz) |
| Safety filter | CBF on corridor obstacles; inflates with critic σ and, during pending router delay, `k · vx · delay` | Same as skills |

**Triggers (proposed wiring):** call the router when critic epistemic uncertainty crosses a frozen training threshold, or when the CBF has been overriding the skill’s longitudinal acceleration. Skill changes blend over a short window (demo used a 20-tick sigmoid; intended writeup: advantage-gated smooth blending).

**Borrow / path order**

1. **First path — highway-env (lightest closed loop).** Start from [CoDrivingLLM](https://github.com/FanGShiYuu/CoDrivingLLM) (highway-env + API LLM). Pull technique ideas from [DiLu](https://github.com/PJLab-ADG/DiLu) (memory/reflection on highway-env). Use a research-usable 7B-class instruct model or a current cheap API; do not hard-depend on GPT-4o-mini past its announced sunset.
2. **Later path — CARLA (optional fidelity).** Port the same three-tier contract onto [CoLMDriver](https://github.com/cxliu0314/CoLMDriver) (Apache-2.0, CARLA 0.9.10.1) once highway-env eval is solid. Optional donors: LeapAD (slow/fast split), LMDrive (LangAuto benchmark).

Do not build on AgentsCoDriver, Talk2Drive, GPT-Driver, DriveGPT4, DriveVLM (page-only), or EMMA for the closed-loop core—see `docs/source-feasibility.md`.

## Milestones

### M1 — Closed-loop highway-env stack
Replace the gauntlet’s empty-road PPO + scripted lane rule with highway-env vehicles, trained multi-skill RL (cruise / overtake / return as needed), inflated CBF at control rate, and a **stub** router (rule or fixed delay) that still uses the interrupt path (σ threshold or ax override → pending delay → blend). Deliver: runnable env, checkpoints, one scripted multi-obstacle scenario, logged metrics below.

### M2 — Real LLM/VLM router
Swap the stub for a real model that outputs a discrete skill. Keep the same interrupt contract and delay model. Prefer a self-hosted Apache/MIT 7B instruct (or vision sibling only if the observation is image-based); keep an API fallback. Deliver: same scenarios as M1 with measured router latency, call count, and survival under inflated CBF during in-flight calls.

### M3 — Multi-scenario eval vs baselines
Run a small scenario suite (stall ahead, cut-in, dense merge / freeze-risk, optional unequipped human traffic). Report the proposed stack against FASIONAD/CERT-Replan-**style** baselines we implement in the same env (not necessarily their original code):
- **Always-on / high-rate LLM** (no event trigger)
- **Timer-polled LLM** (e.g. 1 Hz hard switch, gauntlet Mode 3)
- **Static CBF only** (no skill change — freeze trap)
- **Uncertainty wake-up into same planner** (FASIONAD-style; no skill bank)
- **Barrier-risk replan without VLM** (CERT-Replan-style plan class switch)

Deliver: tables + plots; claim only wiring performance, not method novelty.

### M4 — Optional CARLA
If M3 numbers justify it: same tier contract on CoLMDriver/CARLA. Same metrics. No new science claims. Skip if highway-env already answers the measurement question.

## Success metrics

Record per mode / scenario (align with gauntlet logging where possible):

| Metric | Why |
| --- | --- |
| Crash rate / time-to-crash | Safety |
| Freeze / stall (vx ≈ 0 with progress left) | Barrier-only failure mode |
| Route completion % | Progress under shield |
| Latency survival | No crash while a router call is pending |
| Peak / mean jerk (esp. lateral) | Blend quality vs hard switch |
| Router call count | Cost of event trigger vs polling |

Secondary: average speed, override fraction (CBF replacing ax), peak critic σ at interrupt.

**Pass bar (M3):** proposed stack lower crash than unshielded and timer-polled hard switch; lower freeze than static CBF; fewer router calls than 1 Hz polling at matched or better route %; jerk not dominated by hard skill switches; no collision attributable to in-flight router delay when inflation is on.

## Non-goals

- Claiming a new algorithm or publishing “first to combine X+Y+Z.”
- Matching or beating OEM stacks; field robots; V2X radio hardware.
- Training foundation VLMs or multi-day CARLA from-scratch pipelines (use released weights / APIs).
- Expanding the toy gauntlet into the paper’s main result—it stays a unit demo.
- Formal CBF invariance proofs under learned dynamics (engineering filter + empirical survival only).
- AgentsCoDriver / Talk2Drive / archived open-loop LLM planners as the primary platform.
- Perfect human-likeness, negotiation protocols, or full CoDrivingLLM multi-agent intent sharing in M1–M2.

## References in this store

- Feasibility and licenses: `docs/source-feasibility.md`
- Closest papers and honesty on novelty: `docs/method-novelty.md`
- Toy closed-loop numbers: `docs/two-trap-gauntlet/README.md`
