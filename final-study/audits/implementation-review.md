# Independent implementation review

2026-11-12. Read-only review of `scripts/integrated_eval.py`, `integrated_vlm.py`, the proposal PDF, controller/mode resolution, scheduling, physical telemetry, result accounting, and the actual CPU router. No source edits or simulations. This is an implementation audit before final aggregate results, not a data audit or approval of outcome claims.

## Overall verdict

The runner supports a defensible controlled comparison plus a small actual-model integration subset. The method masks appear consistent with their implementation labels, full physical traces are retained, and development selection is separate from planned test seeds. No obvious arm accidentally retains the full shield when labeled unshielded. The most important risks are interpretation: the final mechanism differs materially from the proposal; stronger local rules have privileged state and no model delay; several comparisons change more than one component; and actual-model timing/cost cannot be inferred from the controlled rule matrix.

## Effective masks and fair comparisons

- `unshielded_rl`: no router calls, no admission; `_configure_mode` disables the entire shield, including boundary/braking/CBF. It is a fixed cruise lane regulator, not a traffic-trained end-to-end RL driving policy.
- `static_cbf`: no router calls, no admission, no braking backup, no inflation; CBF and boundary guard remain. Report those protections explicitly rather than imply every safety layer is shared.
- `timer_hard_unshielded`:1Hz timer, hard changes, no admission and whole shield disabled. This is an intentionally weak system baseline, not an isolated test of timer scheduling.
- `matched_timer_hard`:1Hz timer, common CBF/braking/boundary/admission and8m reserve, hard changes. Controller mode resolution respects the requested hard/blend mask. It differs from final in **both scheduling and blending**. Use `no_blend` versus final to isolate blend effects.
- `uncertainty_wakeup`: common safety/reserve/blend with critic-threshold-only wake-up; shield wake-up is excluded. The selected threshold scale1 is frozen from development; all three scales had12/12 successes and scale1 used the fewest calls. This is a same-stack uncertainty-inspired baseline, not an implementation of full FASIONAD.
- `barrier_replan_local`: shield-override wake-up, same common safety/reserve/blend, zero-delay state rule. This is a legitimate cheap local system comparator, not an equal-latency scheduling ablation or the original calibrated CERT-Replan algorithm.
- `original_proposal`: legacy critic/shield triggers, gamma_sigma2.8, pending speed-delay inflation, no reserve. It differs from final in several components; compare as historical proposal configuration, not a single-factor experiment.
- Final uses deadline trigger,8m released reserve, blend, gamma_sigma0 and pending margin off. `no_reserve`, `no_blend`, `speed_delay_margin`, `critic_margin`, and `immediate_benefit` each change their named mechanism relative to final. These are the clearest causal component contrasts.
- High-rate and gated polling share10Hz clock and common protections/blending; usefulness gate is the difference. Timer phase is randomized by physical seed, consistently across methods as a fraction of their period. Busy and blending periods suppress new calls in every method, so10Hz is a maximum requested cadence, not realized call frequency.

## Proposal alignment and claim restrictions

The proposal centers on critic uncertainty plus speed-times-delay barrier inflation. The deployed final method instead disables both and adds geometric-state deadline scheduling plus fixed maneuver reserve. That is an empirical revision of the proposal, not confirmation of its original mechanisms. Keep original_proposal and the separate margin arms visible, including negative outcomes. The final report can satisfy the course application/combination goal while stating that the initially proposed inflation mechanism was superseded.

The controlled matrix uses exact current simulator object positions/speeds in `PromptRule`; moving leads below15m/s are treated as blockers even though the original camera prompt says all cars are parked. The rule returns a decision from request-time state and delivers it after the prescribed delay. It is not a recorded or actual VLM decision. Its call count measures simulated expensive-planner invocations, not real tokens or CPU compute.

Even the actual-camera final system uses privileged simulator state for deadline opportunity tests, reserve arming, safety filtering and admission. Only the router input is restricted to image/lane/current skill. Thus it is not an end-to-end image-only driving evaluation, and a cheap local rule has access to much of the information already used by the trigger. A strong local baseline matching or beating final must be reported; no evidence of VLM necessity follows from successful integration.

Ordinary delays0/0.125/1s must be analyzed separately from18s stress. The actual subset uses30s estimated latency and45s admission deadline, with actual CPU inference timing. These are different timing regimes; do not pool them or transfer CPU findings to a GPU deployment.

## Metrics reviewed

- Success is no recorded collision and at least99% route progress. Out-of-road termination is separately retained in `end_reason`; a noncollision failure must not be counted as successful survival/task completion. Data audit should enumerate all termination reasons.
- `achieved_jerk_*` uses finite differences of measured world-y acceleration from successive physical steps, rather than commanded lateral acceleration. This is the right physical quantity for the comparison, but describe it as world lateral jerk, not necessarily body-frame passenger comfort. Legacy `peak_lateral_jerk`/`mean_lateral_jerk` measure commands; do not mix the two. New metrics provide RMS and peak, not mean absolute achieved jerk.
- Peak after0.2s excludes initial transient samples; retain full peak too. Crashed/early-terminated episodes have shorter exposure and can look deceptively comfortable. Compare successful matched episodes or report exposure/termination alongside jerk; do not silently drop failed arms.
- `stopped_s` measures total ticks with vx<0.5m/s. `terminal_stall` is a terminal low-speed incomplete/noncrashed condition, not proof of sustained deadlock. Legacy freeze records a low-speed window plus incomplete route. Use separate labels.
- Pending survival is vacuously true when no call is pending; report pending ticks/exposure and restrict meaningful comparisons to arms with pending exposure. Crash while pending does not establish crash attributable to latency.
- `cut_in_observed` detects increased obstacle count in logged pre-step state; `cut_in_first_seen_while_pending` describes the first logged observation, not exact spawn-time alignment. A cut-in is triggered by crossing x92, so policies that stall earlier never encounter it. Report the observed/pending counts before making a cut-in robustness claim; the planned scene name alone is insufficient.
- Override fractions are per executed tick. They are not probabilities of unsafe state, and zero collisions do not establish formal CBF invariance. Reserve activation can produce negative augmented stopping margins; report infeasible ticks and emergency behavior.
- `wall_s` includes environment startup and joining unfinished model work. It is not model-only inference latency or clean controller runtime. Actual prediction logs preserve individual inference timings.

## Actual-model subset caveats

The36-episode design has18 actual-model episodes (three queried policies across six scene/seed worlds) and18 no-model episodes. Model loading is shared; queried order is cyclically balanced, and local/query blocks alternate. This reduces obvious fixed-order bias but is not randomized timing control. Compare matched initial snapshots explicitly and retain measured latency distributions per arm. Local arms run without camera/realtime and have zero model calls; their wall times are not directly comparable to live runs.

ActualCPURouter verifies the exported trained adapter through the unchanged CPU loader and stores image bytes, hint text, logits/probabilities and prediction timings. The prompt includes current_skill and literal20m/s, and assumes parked traffic. Dense/cut-in scenes can violate those assumptions. Earlier fixed-image diagnostic found hint sensitivity, much of it outside the inspected training generator's lane/skill support. Label these deployment limitations, not proof of leakage.

The CPU router has an inherited eight-prediction-per-episode cap. The final data audit should verify that no run hits it; if hit, failures and attempted versus completed predictions must remain visible. The45s pending deadline limits admissible application, not cancellation of a running Torch forward: cleanup joins outstanding work. Do not call it a45s bound on total computation.

Small telemetry issue: the controller's final join of a response that completes after control ends reconstructs `ready_at` using model/wrapper latency without adding `request.camera_capture_wall_s`, whereas normal async delivery includes capture. This affects aborted/in-flight-call timing telemetry only, not decisions already applied. Use separate capture timing plus prediction timing for total request latency on those calls, and flag this difference in data analysis. No frozen-source edit is requested for this audit.

## Final data audit checklist

Verify exact1440-job controlled design and36-job live design (18 actual), unique IDs, lock hashes, checkpoint/model identity, matched initial worlds, effective safety masks, successful trace decompression/hash, and independent recomputation of success/jerk/calls from traces. Report ordinary versus stress results; per-scene cut-in exposure; actual calls completed/applied/rejected/aborted;8-call cap/deadline events; and local-rule performance. Pair comparisons by scene/seed/delay, cluster repeated delays/scenes appropriately, and avoid presenting deterministic condition repetitions as independent world samples. Conclude with whichever method wins, including a simpler or non-VLM comparator.
