# Independent controlled-results review

2026-11-12. Scope: completed controlled1440-episode matrix only. Actual-model subset is excluded because it was still running. No source/model changes or simulations. Computed checks are in `results-review-checks.json`; this review independently checked row/design integrity and contrasts, while the root's separate full-trace audit owns exhaustive trace-level recomputation.

## Integrity checks passed

There are1440 rows and1440 unique IDs. The ID set exactly equals the frozen design, and every ID independently recomputes from its scene,seed,delay,method,sigma_scale. Every row carries the actual lock SHA256, and every locked source/config/checkpoint hash still matches. Each of15 methods has72 ordinary-delay cases (0/0.125/1s) and24 stress cases (18s). Success independently equals no recorded crash and route≥99% in every row.

All methods/delays within each of24 scene/seed groups have byte-equivalent serialized initial states. There are16 distinct initial states: stall-ahead and cut-in share initial layouts before the later event. Thus1440 episodes are not1440 independent worlds, and72 nominal cases per method are repeated delays over24 named worlds. Pair by scene/seed/delay and account for common seeds and repeated states when quantifying uncertainty. The design lock hash is `b533186dead4df77f40b916a6341e00bfa2ecca4d578b7026d1998e3d3f0aad5`.

## Main findings the data support

For ordinary delays, final completes72/72 with zero crashes and terminal stalls, versus unshielded0/72 with72 crashes, literal timer-hard-unshielded14/72 with58 crashes, and static CBF24/72 with12 crashes and36 terminal stalls. These establish improvement over those limited systems, not superiority over every protected planner.

Every protected replanning comparator also completes72/72. Final averages1.0833 simulated expensive calls versus75.8194 for ungated10Hz polling (about98.57% fewer), with no observed success loss. However gated polling and immediate-benefit triggering also average1.0833, and tuned uncertainty triggering averages0.6667. Therefore the large saving versus ungated polling is not unique to the deadline scheduler.

The local barrier-triggered state-rule planner also completes72/72 ordinary and24/24 stress cases with zero expensive calls. It has higher ordinary mean achieved jerk than final, but establishes that an expensive language/vision model is not necessary to solve this privileged-state task suite. The controlled matrix contains rule decisions, not actual VLM perception. Retain that distinction throughout.

## Deadline scheduler does not win

The simpler immediate-benefit scheduler weakly dominates final on every paired ordinary-delay case for the examined success/call/comfort/time measures:

- Success is identical72/72; expensive calls are exactly equal in all72 pairs.
- Achieved jerk RMS and peak are lower for immediate-benefit in24 pairs and equal in48; final wins none.
- Completion time is lower for immediate-benefit in20 pairs and equal in52; final wins none.
- Mean achieved RMS is3.5431 versus final3.9679; mean completion time is0.1261s shorter.

This is stronger evidence against a deadline advantage than a pooled average alone. Report it prominently. The current benchmark does not justify deadline scheduling as its novel or best component. It remains a tested design choice, and a future task would need a separately frozen condition set where waiting saves resources without losing useful opportunities.

Tuned uncertainty costs fewer calls but takes4.5239s longer on average and has0.9897 higher RMS jerk than final, so it demonstrates a real cost/progress/comfort tradeoff rather than uniformly dominating final. Gated polling has equal call count and lower average RMS, again weakening any uniquely deadline-based efficiency claim.

## Ablations and proposal pass bars

Blending lowers ordinary mean achieved RMS from5.0666 to3.9679 and mean peak from93.2455 to75.7385 relative to no_blend, with identical success and calls. It also adds0.2364s mean completion time. The benefit is not pointwise: final has lower/equal/higher RMS in33/18/21 pairs and lower/equal/higher peak in31/18/23. The pass-bar field `blend_rms_and_peak_not_worse_on_jointly_successful_nominal_cases` must be described as a statement about **means**, not all joint-success cases. At18s stress, blend does not improve mean RMS (7.2811 versus7.2269).

Reserve necessity is supported only in the extreme-delay regime tested here: no_reserve completes8/24 stress cases with16 terminal stalls; final completes24/24. At ordinary delays both complete72/72. This is consistent with the prior physical stall study and does not establish that8m reserve is required at ordinary inference latency.

Neither critic margin nor speed-delay margin improves observed ordinary safety/success: their respective arms and final all complete72/72 without crashes. Original proposal also succeeds72/72, with3.0556 calls and7.4087 mean achieved RMS. The integrated revision is more efficient/smoother than that historical configuration, but does not validate the original margin necessity claim.

Do not discard favorable margin effects: at18s, speed-delay margin has mean RMS2.3074 versus final7.2811 and mean peak55.9875 versus142.1029, with identical24/24 success; completion is3.9917s slower. This is a substantial comfort/progress tradeoff, not safety necessity or universal inferiority of inflation.

## Exposure and causal limits

Final records24 observed cut-ins in the ordinary regime,18 first observed during pending calls; stress records8 and6 respectively. No_reserve observes zero stress cut-ins because its affected runs do not reach the trigger. Its failure cannot be framed as a cut-in-safety deficit, and zero crash counts cannot establish safety against an event that was never encountered.

Zero pending crashes in final is a measured outcome, not proof of no crash attributable to delay. Margin necessity remains unproved under zero-event equality. Likewise very low unshielded jerk reflects straight-line driving until collision, not superior comfort. Keep failures/exposure visible when discussing jerk.

Matched timer-hard changes both scheduling and blending, while literal timer-hard-unshielded also removes safety. Use the matched single-component ablations for mechanism claims. Fresh admission versus release-on-any-response is not isolated by the older all-correct reserve experiment; the new broad controlled matrix does not by itself supply that missing causal contrast.

## Recommended final conclusion

“The integrated physical stack solved all tested ordinary-delay cases and reduced calls dramatically relative to ungated polling, while blending improved mean physical jerk. A retained maneuver reserve prevented tested18s-delay stalls. The simpler immediate-benefit trigger matched success/call counts and outperformed deadline timing on ordinary comfort and completion time; tuned uncertainty used still fewer calls. A local rule planner also solved the suite without model calls. These findings support a reproducible integration and failure-mode evaluation, not a new best scheduler or a demonstrated need for VLM routing.”

Keep the actual-model subset as separate evidence with its measured timing and limited six-world support once complete. Do not use the controlled success counts as actual-model accuracy or driving performance.
