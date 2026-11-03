# Project goals / non-goals / next steps

## Goals

- Ship a **runnable** 3-tier stack (router → PPO skills → inflated CBF) with install, train, eval, and figures.
- Isolate the mechanism on the **Two-Trap Gauntlet** (stalled blocker + cut-in during VLM delay) using the **same** stack code as multi-vehicle scenarios.
- Measure an honest ablation: unshielded / static CBF / fixed-rate VLM+hard switch / proposed event+inflate+blend.
- Keep novelty claims modest: composition of published ideas, not a new theorem.

## Non-goals

- Not a new CBF certificate or a new RL algorithm.
- Not a trained vision-language model (scripted router is first-class when no API key).
- Not full CARLA closed-loop (stub only).
- Not beating every baseline on every secondary scenario metric (e.g. dense-highway route can favor a slow static shield); the gauntlet is the mechanism-isolation test.

## Architecture map

| Tier | Role | Rate | This repo |
| --- | --- | --- | --- |
| 1 | Skill choice | On interrupt (~Hz) | `router/` — OpenAI-compat, local LLM, or scripted |
| 2 | Continuous control | 50 Hz | `skills/` — separate PPO policies per skill |
| 3 | Safety filter | 50 Hz | `shield/` — inflated CBF |

## Milestone status

- **M1** — package, PPO skills, scripted router on interrupt API, CBF, eval (done).
- **M2** — real LLM router contract (`openai` / `local` / dry-run mock HTTP), retuned interrupts, re-measured (done; see RESULTS.md **M2 router**).

## Next steps toward CoDrivingLLM / CoLMDriver

1. **CoDrivingLLM** — replace dry-run / scripted teacher with CoDrivingLLM’s high-level speech/act interface; keep the interrupt gate and 300 ms (or measured) latency model.
2. **Larger local model** — Qwen2.5-0.5B loads on CPU but is weak as a closed-loop policy; try 1.5B+ or a fine-tuned skill classifier.
3. **CoLMDriver** — swap Tier-2 PPO skills for CoLMDriver’s language-conditioned low-level policies; keep the inflated CBF and blend on skill/language switch.
4. **CARLA** — implement `CarlaAdapterStub` transforms; reuse `DrivingStack.run` unchanged.
5. **Real VLM latency** — log wall-clock router RTT and set `vlm_delay_s` from percentiles instead of a fixed 0.30 s.
6. **Advantage-gated blend** — `stack/blend.py` already exposes `advantage_gate`; wire critic advantages from both skills into the crossfade (demo currently uses the sigmoid schedule).

## Success metrics (gauntlet)

Proposed should dominate the three baselines jointly on: crash, freeze, route completion, peak lateral jerk, and router-call efficiency (calls only when needed). See [`RESULTS.md`](RESULTS.md) for the latest measured numbers.
