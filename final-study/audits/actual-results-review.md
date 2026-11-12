# Independent actual-model results review

2026-11-12. Completed36-row actual-subset audit; no source edits, model loading or simulations. Independently checked exact frozen design membership and recomputed all36 job IDs, row lock hashes, all package/model/prompt/config hashes, six matched scene/seed initial-state groups, all36 compressed trace hashes/decompression, and per-episode prediction-file counts/letters. Every recorded prediction reports504 verified adapter tensors and no fallback. The runner's completion record agrees:36 episodes,18 model-enabled episodes,22 actual predictions.

## Enabled, queried and applied are different

| Method | Episodes | Success | Crashes | Terminal stalls | Model-enabled | Actually queried episodes | Actual predictions | Applied model decisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Final integrated |6|6|0|0|6|5|7|4|
| Tuned uncertainty |6|6|0|0|6|4|4|4|
| Matched timer-hard |6|5|0|0|6|6|11|4|
| Local barrier replan |6|6|0|0|0|0|0|0|
| Static CBF |6|2|0|4|0|0|0|0|
| Unshielded RL |6|0|6|0|0|0|0|0|

Only15 of18 model-enabled episodes actually queried the model. The `actual_model` row flag denotes a model-enabled arm, not proof that a prediction affected that episode. Final's dense9102 and both uncertainty dense episodes made no model call. Final's dense9101 and the two timer dense episodes queried but their answers completed after control ended, so their success does not demonstrate successful use of a model decision.

All12 applied model decisions were CRUISE_RIGHT→OVERTAKE_LEFT (four per queried method). No model-driven return maneuver was applied. The two C predictions from final's cut-in episodes completed after route completion and were retained as aborted/unapplied calls. Report these accurately; a successful full route does not establish successful end-to-end execution of all three semantic skills.

There were22 actual forwards:17 B,2 C,3 A; zero fallback. Final used7 forwards, uncertainty4, timer11. Completed predictions include work finishing after route completion, which is correctly retained as real compute. Suppressed trigger entries also appear in `calls`; do not treat every list entry as an inference. Cross-check `actual_model_predictions` and prediction files for model cost, and `applied` plus changed `from`/`to` for behavioral influence. The inherited eight-prediction cap was not hit (maximum3 in an episode).

## One timer failure is confounded by actual timing

Timer-hard on stall_ahead9102 reached57.3% and timed out without crashing or ending at low speed. Its first B response took47.066s including capture, exceeding the45s pending deadline and was explicitly rejected. A later B was applied, and another completed after control ended. This is not a terminal stall and must not be labeled a crash or freeze.

Observed model-only latency ranges were approximately21.17–42.64s for final,22.96–39.16s for uncertainty, and19.23–47.01s for timer. The30s estimated latency was not a hard bound, and several responses exceeded it. The45s deadline correctly prevented one late response from being applied; it did not cancel its CPU computation. CPU load, evolving images and actual timing differ across arms despite balanced ordering and matched initial worlds. Therefore6/6 versus5/6 is supporting integration evidence, not a clean causal estimate of scheduler superiority or GPU performance. The controlled-delay matrix supplies the cleaner timing comparison.

## Supported conclusion

The trained adapter was genuinely invoked, its decisions were applied in12 episodes, and final/uncertainty each completed all six tested worlds without recorded collision. However a cheap local barrier-rule policy also completed all six with no model calls. This subset does not establish model necessity, model superiority, representative traffic accuracy, or reliable return-skill behavior. It demonstrates a working actual-camera/adapter/physical-controller integration under severe local CPU latency, with one observed deadline rejection retained rather than hidden.

The finite six-world design and deliberate state access by triggers/safety remain the scope. Current-skill text and the original parked-traffic/literal-speed prompt remain potential deployment mismatches. Keep this subset separate from the1440 controlled rule episodes.

## Proposal fifth pass bar clarification

The original observational requirement was zero collisions during in-flight delay when inflation is on. The controlled margin-on arm met that observed bar (96/96 collision-free), as did the no-pending-margin final arm (96/96). Thus the **observed bar passes**, while **causal necessity of inflation is not established**. These are compatible statements and should occupy separate report rows. No claim of a formal safety guarantee follows from either.
