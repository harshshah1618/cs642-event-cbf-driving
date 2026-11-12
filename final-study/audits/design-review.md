# Final integrated comparison: independent design review

Reviewed 2026-11-12, before the final runner/matrix was available. Scope: proposal PDF, current `config.py`, `stack/controller.py`, `stack/scheduling.py`, `stack/maneuver_reserve.py`, and safety-filter implementation in the final worktree based on 6090300. This is a design audit, not certification of an as-yet-unseen runner or results. No implementation changed and no simulations run by this review.

## Recommended locked comparison

Use exactly 12 previously unused physical-scene seeds × 3 named scene families × delays {0, 0.125, 0.5, 1, 18} seconds × 14 methods = **2,520 episodes**. Store the exact seed list and generated initial states before evaluation; a seed is not evidence of variation unless physical states actually differ. Pair every method on the same scene instance, scheduled external disturbances, duration, vehicle model, PPO checkpoints and termination criterion. Freeze all configuration and source/checkpoint hashes, including controlled-router label semantics, before launch. Never replace a difficult seed after inspecting outcomes.

The proposed list currently describes 13 distinct methods. A defensible fourteenth is the matched immediate-benefit scheduler: the same opportunity information, admission guard, reserve, margin and blend as final, but requesting as soon as useful. This is a strong, inexpensive control for whether deadline prediction adds value.

| Method | Intended comparison |
|---|---|
| Unshielded RL, no routing | Historical unsafe skill baseline |
| Static safety filter, no routing | Historical fixed-plan freezing baseline |
| 1 Hz router, hard switch, no shield | Historical timer baseline; bundled difference |
| Uncertainty-only wake, same controlled router | Critic-trigger inspiration, not FASIONAD reproduction |
| Barrier-override local replan, no VLM | Local replanning inspiration, not CERT-Replan reproduction |
| Original proposal: sigma/override wake + inflation + blend | Original integrated design |
| Final: deadline + fixed latched 8 m reserve + blend + ordinary protection | Main engineering candidate |
| Matched 1 Hz | Scheduling-only control |
| Matched 10 Hz | Scheduling-only control |
| Matched gated 10 Hz | Scheduling and usefulness control |
| Matched immediate benefit | Strong cheap scheduling control |
| Final without reserve | Incremental reserve effect |
| Final without blend | Incremental blend effect |
| Final without delay inflation | Incremental pending-margin effect |

Specify whether “without margin” removes only the pending-delay term or also critic inflation. Prefer removing only the pending term, retaining the fixed 8 m maneuver reserve; reserve and pending inflation are different mechanisms. If 14 methods are already otherwise allocated, document the actual list rather than silently adding a fifteenth.

## Implementation checks required before freeze

1. **Effective masks, not requested flags.** `_configure_mode` disables the entire shield for ordinary fixed/always-on/unshielded modes; this also disables braking and road-boundary guards. Matched shielded polling re-enables inflation and now respects the caller's blend choice. Assert/log resolved shield, CBF, brake, boundary, admission, monitor, pending inflation, critic inflation, blend and reserve settings after mode configuration. A historical unshielded arm may intentionally lack these protections, but it is a bundled baseline, not a causal estimate of one component.
2. **Same-skill latency.** Set `pending_margin_on_all_calls=True` consistently for delayed routers. Its dataclass default is False: synchronous same-skill replies otherwise bypass pending wait. Keep deadline expiry, cancellation, error handling and maximum pending time identical; 18 s answers must not be inadvertently rejected by a shorter timeout.
3. **Matching scheduler arms.** Final and matched 1/10 Hz/gated/benefit arms must have identical PPO, current observations, margin constants, reserve arming/release, fresh admission, monitor and blend. Report nominal timer rate versus achieved request rate: pending calls and blending suppress all requests, and event modes additionally obey `min_call_ticks`. These are policies, not an identical absolute request-time grid.
4. **Reserve state machine.** The wrapper adds clearance to the braking backup, not the base CBF formula. It arms only on a filter tick with `vlm_pending=True`, a slow lead, and no active transition. Submission occurs after filtering; an answer ready at the next tick may therefore never arm it. Same-skill replies, rejection and timeout do not release a latched reserve; only a changed admitted target does. This can create permanent extra clearance and must be visible in traces. A fixed 8 m choice from prior development is legitimate but is not a geometric viability guarantee or a novel adaptive algorithm.
5. **Actual information.** The deadline/usefulness heuristic sees object positions and velocities. Critic uncertainty sees only ego `(y, vy, vx)`, not surrounding traffic. Controlled routing sees privileged simulator state; matching a prompt's label semantics does not make it equivalent to visual perception. State this clearly, including any oracle visibility of off-camera neighbors. No perfect-router result should be described as actual trained-VLM performance.
6. **Gate limitations.** Current usefulness compares minimum absolute lead speeds and requires lane settlement; an adjacent slower vehicle far away can suppress a useful maneuver. This is a known limitation, not a reason to revise the gate after final test failures. The 1.5 s admission predictor is not a full swept-path safety certificate.
7. **Local replanning control.** Record its real measured local compute time and chosen intentional delay. A local planner with no model calls should not be charged artificial VLM inference latency merely to equalize call counts. Compare progress/safety and computational scope separately; “zero VLM calls” is not “zero planning work.”

## Analysis and claim boundaries

Use strict destination completion as the primary success metric, alongside crash, remaining-route freeze, route fraction and end-reason counts. If a legacy >=99% threshold is retained, label it threshold completion and show destination attainment separately. Timeouts and collision-free stops are not successes. Show call submissions, applied answers, rejected/expired answers, latency and pending duration separately. A crash during pending is an association; “attributable to delay” requires paired counterfactual evidence and remains trajectory-dependent.

Report delay strata first. The 18 s condition is CPU-scale stress motivated by observed local inference, not representative of the recovered GPU timing; do not let equal weighting with this extreme condition imply an everyday deployment advantage. A prespecified equal-weight overall mean is permissible as a benchmark summary, with its weighting stated. Include every method and every condition, not only winning comparisons.

For stochastic physical instances, resample the 12 shared seed clusters jointly across all scene/delay/method cells and compute paired differences. Do not treat 2,520 episodes as independent observations. With only 12 clusters and rare crashes, intervals have limited resolution; zero observed crashes is not a universal safety claim. Distinguish exploratory multiple comparisons from a preregistered primary contrast. If seeds do not change physical states, report deterministic condition coverage rather than manufactured seed uncertainty.

Comfort requires achieved physical longitudinal/lateral jerk, both RMS/mean and peak, with crash/timeout exposure reported. Lower jerk from stopping early, crashing or never maneuvering is not improved driving. Show completion-conditioned comfort only as secondary, because conditioning on a method-dependent outcome can bias comparisons. The proposal's hard-switch comfort goal may fail even with successful integration; report mixed evidence honestly.

For the actual trained-model subset, freeze selection before examining new outcomes: include an easy pass, dense/occluded case and delayed/stall case, with the same paired baselines where affordable. Preserve raw camera inputs, prompt text, model/checkpoint identity, outputs/errors and timing decomposition (capture, preprocessing, inference, transport, queueing). Separate physical-camera closed-loop evidence from earlier point-mass/ghost-camera logs. Report this small subset descriptively, not as general VLM reliability or as confirmation of every controlled-router contrast.

The proposal explicitly permits an application-dependent integration of established techniques and disclaims a new barrier theorem. A defensible conclusion is an evaluated engineering combination with stated operating conditions. Historical baseline wins alone cannot attribute gains to deadline scheduling, reserve, blending or CBF; the matched controls and ablations provide those narrower comparisons. FASIONAD-style uncertainty and CERT-style override triggers are inspirations, not reproduced published systems. The previous component study already found stopping backup/admission could account for safety without incremental CBF benefit; the final report must preserve that negative finding rather than relabel all protection as a CBF contribution.
