# Final independent component-results review

2026-11-11. Completed results audited read-only; no new experiment, frozen-source
edit or protocol change. Existing11 representative trace repetitions were
checked against the original main rows, not counted as additional evidence.

## Verification performed

- Frozen source/checkpoint inventory, protocol, exact job design and pilot hashes
  pass `component_ablation.verify()`.
- Exactly4608 unique expected test jobs, with matching metadata and no extra or
  duplicate cells. Compressed test artifact decompresses byte-for-byte to raw
  JSONL. Main raw SHA256:
  `fbd1cc70509b76912cc72c608e244d2bcbe7ab9be48b108481365d71320254cd`.
- Each stored initial-state payload recomputes to its hash; report verifies the
 48 scene/seed groups share initial states across all treatments and delays.
- Every success flag equals frozen `not crashed and route_pct >=99`.
  There are100 threshold successes ending at timeout and zero safety aborts.
  These are valid under the frozen criterion, but not100 additional destination
  completions. Do not replace or ambiguously rename this endpoint.
- Independently recomputed all16 cell means for collision, success, route and
  both displayed achieved-jerk metrics; all agree with statistics.json.
- Independently recomputed all four main-effect contrasts and six two-factor
  difference-of-difference contrasts for collision/success/route. Also recomputed
  all main-effect metric confidence intervals from per-seed plus/minus means
  and the stated10000-draw bootstrap; all agree.
- All11 exported trace records match their frozen main row on every saved field
  except separately measured repeat wall time. Recomputed achieved jerk and
  stage/monitor/admission summaries from actual trace arrays reproduce the rows.
  Applied actions equal monitor-input shield actions throughout those traces.
  The full study has zero recorded monitor changed-action or emergency ticks.

No arithmetic, paired-contrast-sign, bootstrap-clustering or trace-reproduction
defect was found. Markdown pipe escaping is a presentation fix being handled by
root and does not alter the results.

## Verified headline counts

Every cell below covers288 diagnostic episodes (12 seeds x4 scenes x2 delays
x3 routers), not288 independent traffic seeds. Common road-boundary protection
and numerical-validity monitor remain active throughout.

| Controller and traffic protections | Crashes | Threshold successes |
|---|---:|---:|
| PPO, CBF+braking+admission |0|228|
| PPO, none of these three |280|8|
| PPO, CBF only |42|211|
| PPO, braking only |2|226|
| PPO, braking+admission, no CBF |0|228|
| PD, CBF+braking+admission |0|226|
| PD, none of these three |222|66|

The strongest supported interpretation is that this diagnostic stack needs
traffic-aware protection; the stopping-envelope backup supplies most of the
observed collision reduction. Longitudinal CBF is strongly protective when
other traffic protections are absent, but adds **no observed collision/success
benefit on top of braking+admission** in this sample. It is incorrect to credit
the full stack's safety result uniquely to the CBF or claim all three protections
are individually necessary.

With full PPO protection, removing braking produces36/288 crashes versus0;
removing admission produces2/288 versus0. The latter paired collision interval
reaches zero, so describe the two prevented collisions and interval rather than
claiming a precise universal admission benefit. Admission alone increases crash
counts in the no-CBF/no-brake cells (PPO280 to286; PD222 to226), illustrating why
component interactions and the entire factorial matter.

PPO is not globally superior to PD: without traffic protections it crashes more,
and with full protection its228 versus226 threshold successes are only a small
conditional sample difference. Neither controller has independent training/gain
replications. No algorithm-level RL superiority follows.

## Routing and completion

Within full PPO protection, state-rule and synthetic front-blind each achieve
91/96 threshold successes with zero crashes; copy-current achieves46/96 with
zero crashes. State-rule successes include87 destination completions plus4
threshold timeouts; copy-current includes44 completions plus2 threshold timeouts.
Protection alone can avoid collisions while progress still fails. This supports
useful maneuver selection relative to a deliberately stubborn router, not a
claim that a trained VLM provides those decisions or that visual faults are safe
at a deployment-calibrated frequency. Per-router/scenario/delay tables remain
necessary beside the equally weighted diagnostic aggregate.

## Comfort and exposure

Full PPO achieved lateral jerk RMS is3.223m/s³ versus PD3.508, but the mean
per-episode post0.2s peak is57.93 versus41.27m/s³. Adding CBF to PPO braking+
admission reduces RMS3.914 to3.223 while increasing that peak55.12 to57.93.
Thus comfort is mixed; no blanket smoothness improvement is justified.

Crashes terminate exposure early; stalls have little movement. The particularly
low RMS of unprotected PPO is not a desirable safety/comfort result. Do not rank
drivers by jerk or duration without their collision/progress outcomes. Full PPO
mean end time15.861s versus PD15.559s is also not an unconditional speed advantage.

## Claim boundary

This is a frozen-checkpoint, straight-road, physical MetaDrive diagnostic
factorial with12 seed clusters, synthetic faults, common containment, and
injected delays. It demonstrates measured conditional component contributions
and overlaps, not rare-event safety, independent-learning superiority, curved-road
transfer or theoretical CBF guarantees. Zero observed crashes is not zero risk.

Actual teammate VLM source and committed frame-classification results were
recovered and separately audited. Private weights/endpoint remain unavailable;
these4608 episodes contain no true trained-VLM rollout. Camera/HTTP fixtures and
synthetic rule faults must stay distinctly labeled. The results are ready to
package with these qualifications and the source/trace/raw artifacts.
