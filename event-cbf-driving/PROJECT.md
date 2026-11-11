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

The five follow-up deliverables are complete: actual local-model comparison,
investigation and reduction of safety failures, frozen-weight uncertainty
calibration, broader physical traffic/delay stress, and a measured report/demo.
See [FINAL_REPORT.md](FINAL_REPORT.md) for current evidence and negative findings.

- **M1** — package, PPO skills, scripted router on interrupt API, CBF, eval (done).
- **M2** — live router implementations and bounded background inference are
  available. Slow/expired response contracts are tested. Canonical physical
  nominal evaluations use scripted decisions. An additional 18-episode pinned
  Qwen2.5-1.5B pilot makes 62 actual calls with no fallback; it establishes no
  advantage over a delay-replayed rule control.
- **MetaDrive** — optional physical backend, randomized scenario parameters,
  shared frozen-policy transfer, road-edge constraints, fresh-gap maneuver
  admission, matched shielded polling comparisons, replay and interaction
  plots, calibration, stopping backup and intended-corridor recovery. See
  `METADRIVE.md` for commands and `FINAL_REPORT.md` for current evidence.

## Next steps toward CoDrivingLLM / CoLMDriver

1. **CoDrivingLLM** — replace dry-run / scripted teacher with CoDrivingLLM’s high-level speech/act interface; keep the interrupt gate and 300 ms (or measured) latency model.
2. **Stronger decision model** — the completed 1.5B CPU pilot establishes no benefit; a larger model or fine-tuned classifier needs a new controlled study.
3. **CoLMDriver** — swap Tier-2 PPO skills for CoLMDriver’s language-conditioned low-level policies; keep the inflated CBF and blend on skill/language switch.
4. **CARLA** — implement `CarlaAdapterStub` transforms; reuse `DrivingStack.run` unchanged.
5. **Camera/VLM latency** — actual text-model timing is now measured; adding perception requires a separate camera/VLM experiment.
6. **Advantage-gated blend** — `stack/blend.py` already exposes `advantage_gate`; wire critic advantages from both skills into the crossfade (demo currently uses the sigmoid schedule).

## Success metrics (gauntlet)

Compare crash, freeze, route completion, peak lateral jerk and router-call
efficiency without assuming universal dominance. See [FINAL_REPORT.md](FINAL_REPORT.md)
for the physical study and [RESULTS.md](RESULTS.md) for native results.
