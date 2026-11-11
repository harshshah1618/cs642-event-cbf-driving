# Component contribution study: independent design review

2026-11-11. Read-only source inspection in the new component worktree; no package
changes or simulations were performed. Existing frozen research results must
remain unchanged. This document proposes prospective experiments, not results.

## What must be separated

The implemented protection stack has more than one "CBF": the longitudinal
headway filter, stopping-envelope backup, road-boundary lateral guard, maneuver
admission check, uncertainty/pending inflation, intent-corridor obstacle selection,
and malformed-data monitor. Reporting full protection versus `unshielded` as the
contribution of the longitudinal CBF would be incorrect.

Source findings:

1. `DrivingStack._configure_mode` rewrites `shield.enabled` and `shield.inflate`.
   `fixed_rate_shielded` also forces smooth blending and disables hard switches.
   The plain polling modes disable the whole shield and can hard-switch. Named
   modes therefore do not form an orthogonal component factorial. Effective
   settings after mode configuration must be recorded/asserted, not just YAML.
2. `InflatedCBFShield.filter` gates the braking and road-boundary filters on the
   same `enabled` flag as longitudinal CBF. Turning `enabled=False` removes all
   three. Independent CBF-off/brake-on requires a deliberate component-specific
   filter path or wrapper; a parameter change alone cannot implement it.
3. Legacy queries depend on shield override episodes and critic sigma. Removing
   protection changes queries, accepted maneuvers and pending inflation, so an
   event-triggered ablation estimates a total coupled-stack effect, not a direct
   filter effect. Use a common exogenous query clock in the primary mechanical
   factorial; reserve adaptive triggering for a separate coupled-system study.
4. PPO observes only `[y, vy, vx]`; it is a lane/speed regulator, not a traffic-aware
   policy. Its critic spread depends on those observations and calibration, not
   explicit obstacle state. Matched polling supplies sigma to inflation, whereas
   static-CBF supplies zero. PD generally supplies zero too. Disable both sigma
   and pending inflation in the initial component comparison, or factor them
   separately. Never attribute a PPO-versus-PD difference to learning alone when
   it also changes inflation or query frequency.
5. Monitor behavior is narrower than a hidden collision backup: default
   `latch_on_violation=False` means finite but infeasible CBF results do not latch
   braking. Malformed/emergency outputs cause emergency actions/latching; config
   can abort before another plant step. Keep numerical validation/monitor fixed,
   record its interventions and termination reasons, and count safety abort as
   failure. Do not disable finite checks merely to create a no-protection cell.
6. Intent corridor spans physical and target lanes and influences both CBF and
   stopping backup. The target switches immediately even while actions blend.
   Keep intent geometry, footprint selection and blending fixed across cells.
7. Admission guard checks current/projected1.5s absolute front AND rear gaps,
   including sign crossings. Longitudinal filters only consider obstacles ahead
   in their lateral corridor. They are not a substitute for adjacent/rear gap
   admission; braking can even worsen a fast rear-approach hazard. The proposed
   "shield covers unseen car beside us" story needs physical evidence and may fail.
8. Physical adapter clips steering/throttle and tracks requested acceleration;
   realized deceleration can differ from requested values. Keep that adapter and
   bounds identical. A negative mathematical barrier or requested brake is not
   proof that a physical collision was prevented.

## Recommended small prospective study

Use two experiments with distinct estimands rather than one large confounded grid.
All comparisons are paired initial scenes and deterministic environment seeds;
trajectories and reactive traffic may diverge after intervention, as expected.

### A. Which protection mechanism contributes?

An8-cell full2x2x2 factorial: longitudinal CBF on/off, stopping-envelope backup
on/off, maneuver-admission check on/off. Fixed PPO checkpoints, deterministic
scripted router, identical observation access, smooth blending, road-boundary
guard ON, intent corridor ON, numerical monitor ON; both inflation terms OFF.
Call this "longitudinal protection factorial with common road-boundary guard",
not "all safety off" for the all-zero cell.

Use a fixed request schedule independent of filter outputs and critic spread.
For example requests at simulation times0,3,6,9,12,15,18s with nominal0.125s and
stress1.0s response delay; nonoverlapping requests avoid busy-queue changes in
nominal call times. Any deferral due to blending must be explicitly logged and
either made common by a fixed decision contract or treated as an endogenous
mediator. Fixed queries do not force identical answers: each answer correctly
uses that cell's current observed state. This estimates closed-loop component
effects under a common request policy, not identical action-sequence effects.

Suggested size:12 fresh held-out seeds x4 scenario families x2 delays x8cells
=768 episodes. Three existing physical families plus one prospectively defined
adjacent/fast-rear occupancy family are reasonable if the adapter supports the
fourth correctly. Otherwise use the three existing families (576 episodes) and
make the admission counterexample a separately labeled mechanism test. Use a
small disjoint development set to verify each intervention actually activates;
do not select held-out seeds by whether an ablation crashes.

Include predeclared injected wrong-maneuver cases in the admission mechanism
test: e.g. request occupied adjacent lane while cruising, with front and rear
occupancy variants. This is synthetic router fault injection. It is neither a
VLM rollout nor a measured VLM error distribution. Keep injection schedule and
answers fixed across protection cells; report these stress cases separately from
correct-router driving and include scenarios where every cell succeeds/fails.

### B. What do maneuver routing and fast control contribute together?

An8-cell2x2x2 factorial: fixed-lane skill versus scripted maneuver routing;
frozen PPO versus fixed-gain PD regulator; protection bundle off versus on.
Define the bundle explicitly as the three factors in A, with road-boundary guard
and numerical validation still common. Keep query timing, blending, inflation
and desired-speed/acceleration limits matched. Fixed-lane has no meaningful
planning output and should be reported as no routing, not an equal-compute router.

Use the same12 fresh seeds and three standard physical families, initially one
nominal delay:288 episodes. Reusing A's scenes increases precision and is valid
if these contrasts are declared before outcomes. Reuse exactly identical cells
rather than resimulating or counting duplicate episodes as independent evidence.
This identifies routing/control/bundle interactions conditional on these
particular checkpoints and PD gains. It does not establish superiority of PPO
as a learning algorithm without independent training seeds and comparable tuning.

Only after these are frozen should a small paired deployment comparison restore
the original adaptive event scheduler and inflation. Label its result as coupled
stack behavior; it cannot substitute for the isolated factorial effects.

## Required implementation checks before simulation

- For every cell assert the final component mask after `_configure_mode`.
- In CBF-off/brake-on, confirm longitudinal action stays raw until stopping
  backup; in CBF-on/brake-off, ensure no backup reduction; road-boundary behavior
  remains identical in each. Avoid calling the base filter twice and accidentally
  applying a supposedly disabled mechanism.
- Check monitor has no unexpected intervention on valid raw outputs and retains
  identical malformed-data behavior across cells.
- Record proposed PPO/PD action, post-CBF action, post-braking action, post-boundary
  action, monitor output, physical command/achieved acceleration, admission result,
  request/response times and actual skill/blend state. Component effect counts
  cannot be reconstructed from the final merged `overridden` flag alone.
- Treat `RETURN_RIGHT` checkpoint aliasing as fixed shared implementation; log
  loaded weights and effective model metadata. Never silently load fallback skills.
- Freeze script/config/checkpoint/runtime hashes, exact cell inventory and seed
  split before test. Store raw failures and all cells, including nonfinite or
  aborted episodes with a defined failure accounting rule.

## Analysis and stopping

Primary outcomes: collision-episode incidence, strict success (no collision and
at least99% route), route completion, and timeout/stall. Secondary outcomes:
calls, interventions separated by mechanism, admission rejections, travel time,
and physically achieved jerk/acceleration. Keep full-horizon exposure and early
crash termination visible; crashes can misleadingly reduce total time/calls.

Compute within-seed cell contrasts, average over declared families/delays, then
resample seeds jointly for paired cluster-bootstrap intervals. Estimate each
factor's main effect averaged over the other factors and selected two-factor
interactions; also show all cell means. Do not infer synergy merely because the
full stack beats one removed-component cell: report interaction contrasts with
uncertainty. With12 seeds this is a small mechanism study, not rare-event safety
certification. Zero observed events do not imply zero risk; nonsignificance does
not establish equal safety. Report all effects regardless of sign.

## Teammate VLM evidence boundary

The supplied report describes unpushed code/weights, single-frame classification
accuracy and GPU inference latency, not executed integrated driving. No teammate
source, model or runtime is available here, so no real VLM result can be produced
from it. A0.125s injected latency may be labeled "illustrative latency motivated
by teammate report", never measured local inference.

There is also a target-policy mismatch: teammate labels keepLEFT while a right
lane obstacle remains and returnRIGHT when it clears; current ScriptedRouter
returnsRIGHT only when the active LEFT lane has a slow lead. Teammate labels
explicitly require an adjacent-free condition to overtake; current router does
not, relying instead on admission/protection. Thus scripted behavior cannot be
called a faithful surrogate for that VLM policy without an explicit label-rule
implementation and separate identification. The report's57/60 blind-spot errors
motivate a synthetic hazard family, but are not calibrated closed-loop fault rates.

## Frozen-run analysis review

The eventual frozen design uses a full controller/CBF/backup/admission factorial
under three explicit rule/fault routers, with a shared2Hz rolling request policy.
This differs from the illustrative study sizes above and is defined by the
actual frozen component protocol. No package/protocol changes were made during
this analysis review, and no incomplete main-run outcome aggregate was inspected.

Independent hand-computable tests in `tests/test_component_report.py` verify:

- Main factor contrasts are plus-minus differences, averaged over the other
  settings. Plus means PPO for controller and enabled for each protection.
- Two-factor interactions are **unscaled differences of differences**:
  `Y++ - Y+- - Y-+ + Y--`. They are not divided by2 as in some DOE effect
  conventions, or by4 as in a +/-1 regression coefficient. An interaction of+11
  in a synthetic additive-plus-product response is recovered exactly and is
  invariant to factor order. For collision outcomes, positive interaction means
  the ON effect becomes more positive/less protective when the other factor is
  ON; for success, interpret signs in the opposite performance direction. Do not
  automatically call an interaction synergy without specifying outcome direction.
- Conditional full-system contrasts hold all other protections enabled and PPO
  fixed; the PPO-minus-PD contrast holds the full protection bundle fixed.
- Resampling acts on shared seed-level paired contrasts, preserving dependence
  across scenarios, faults and delays. A two-seed hand example with effects2/6
  returns mean4 and interval[2,6] despite repeated contexts per seed.
- Missing factor levels are rejected and all16 controller/protection keys remain
  distinct. Nine focused tests pass; Ruff is clean.

The report's exact job audit combines unique expected IDs, validated job metadata
and the frozen total row count, which establishes complete main cells. Source
verification recomputes the full file/checkpoint inventory, including newly added
files, and compares the job-design and pilot hashes. Runtime versions are recorded
but that source verifier alone does not attest installed dependency contents or
remote authenticity; preserve this scope distinction in packaging.

Root agreed to add report-only checks recomputing initial-state hashes from their
payloads, checking the recorded frozen success definition, and displaying ending
reasons. Frozen success is no crash and route>=99%; a timeout above that threshold
can technically satisfy it. Do not claim every timeout fails, and do not silently
change the frozen endpoint after results. A supplementary completed-route/terminal
success count may be clearly labeled separately.

Fault/router contexts form an equally weighted diagnostic mixture, not an
estimated deployment error distribution. Per-context tables must accompany the
pooled contrasts. Physical jerk and duration have unequal exposure after crashes,
stalls and timeouts: neither lower jerk nor shorter time alone establishes a
better controller. Conditional-on-success comfort summaries, if provided, are
descriptive selected subsets and cannot replace the all-episode safety/progress
endpoints. No critical contrast-sign or clustering defect was found.

## Dated availability correction, 2026-11-11

The earlier statement that teammate source was unavailable is superseded. Root
fetched exact commit `1f16b7ff59aae20c2a9adfbeb4943762d4b1f5b4` despite the branch
not appearing in advertised remote refs. Actual source and committed evaluation
logs were independently inspected read-only; see `teammate-source-review.md`.
Fine-tuned ID results recompute to300/300 and50/50; OOD252/420 (60%), including
beside3/60. Weights, dataset images and authenticated live endpoint are still
not available. The unchanged factorial remains rule/fault evidence, not an
integrated trained-VLM rollout. Branch absence alone was insufficient evidence
for the earlier source-availability inference.
