# Contribution verdict after the locked held-out matrix

2026-11-11. Inspected development data before held-out completion; opened held-out data only after the root reported all192 episodes complete. Recomputed row counts and source hashes: development252/252, held-out192/192, no source-lock mismatches. This audit made no package changes.

## Strongest defensible result

The experiment establishes a **scoped physical control failure and a simple effective repair**: in a deterministic straight-road parked-obstacle task, a delayed correct lane-change command can arrive after collision-avoiding braking has left the installed forward-driving controller stalled. Preserving8m of additional room while waiting and releasing that extra room on an admitted target change restored route completion throughout the tested held-out grid. Keeping the same extra room permanently did not.

This is stronger than a single failed VLM demo and weaker than a new general safety/planning theory. Existing shielding-aware planning, predictive safety certification, backup CBFs and CBF deadlock literature preclude broad novelty claims; see [prior-work.md](prior-work.md). A useful contribution is a reproducible cross-layer failure analysis, a controlled physical benchmark, and an admission-released reserve baseline. Whether that constitutes a publishable research contribution depends on venue and broader validation. It is not evidence that a new geometric reserve algorithm is needed.

## What the controlled data say

| Stage/policy | Conditions | Successes | Crashes | Mean end time, s |
|---|---:|---:|---:|---:|
| Development baseline |36|12|0|42.970|
| Development fixed2m, released |36|20|0|35.363|
| Development fixed4m, released |36|28|0|27.533|
| Development fixed8m, released |36|36|0|19.547|
| Development geometry, released |36|36|0|19.527|
| Held-out baseline |48|12|0|47.180|
| Held-out static8m |48|12|0|47.256|
| Held-out fixed8m, released |48|48|0|23.035|
| Held-out geometry, released |48|48|0|23.071|

Development also tested12m and20m fixed reserves, both36/36. The selected8m was the smallest swept full-success constant, locked before held-out outcomes. Failed episodes hit the60s horizon, so mean end time across successful and failed episodes is a timeout-inclusive descriptive quantity, not a clean successful-trip speed comparison.

Held-out conditions cross PPO/PD controllers, steering limits18/30/40degrees, speed/gap pairs18m/s–55m and24m/s–80m, and decision delays0.125/6/18/30s. Every policy succeeds in the12 short-delay conditions. Baseline and static8m fail all36 longer-delay conditions; fixed and geometric released reserves succeed in all36. Thus the held-out improvement is36/48 conditions, or75 percentage points, on this grid. There are48 distinct deterministic combinations, not48 independent samples from a traffic population. Do not attach a population success guarantee or conventional IID confidence interval.

## Geometry does not win

Against fixed8m, geometry changes held-out mean end time by **+0.0367s** (slower), with identical48/48 success and zero crashes. It reduces mean peak extra clearance from8.000m to7.697m and its time integral from108.12 to104.02m·s. That small bookkeeping reduction is not a demonstrated comfort, throughput, or task benefit. Development had the opposite resource direction: geometry averaged8.443m and84.653m·s versus8m and80.213m·s, with only0.0194s faster mean completion. There is no consistent useful advantage.

The implemented geometric estimate is a two-arc primitive based on wheelbase, steering limit and fixed4m lateral displacement. It is not a dynamic reachability calculation: it does not account for current heading, speed, footprint clearance or steering-rate limits. Its successful test performance is empirical. The simpler fixed8m release policy is the appropriate headline method.

## What is and is not isolated

The correct-intent matrix holds decision content and delay fixed between policy arms, so its improvement is not attributable to better classification or faster model inference. Static8m versus released8m shows that retaining additional room forever is insufficient in these cases. It supports the importance of consuming the reserve when executing the maneuver.

It does not isolate **fresh guard admission** from simpler release-on-any-B or release-on-a-known timer: all commands are the same valid B. Stale intents, unsafe target lanes, rejected changes, timeouts and no-response cases were not included. Do not claim the matrix proves the value of the admission semantics themselves. The method arms only when a request is pending and a lead is detected; it cannot recover space already lost before its first request.

No crash was recorded, but the enlarged stopping envelope frequently reports infeasibility (held-out fixed mean494.625 ticks; geometry493.938). Abruptly adding reserve can place the current state outside the enlarged envelope. Consequently the data do not establish recursive feasibility, forward invariance of the reserve, or a formal safety theorem. The ordinary protections remain present; scoped empirical collision absence is the supported statement.

“Unrecoverable” should mean unrecovered by the tested controller/shield over the episode horizon. No exhaustive bounded-control reachability proof or alternative steering rescue was performed here. The evidence does not establish that no physically feasible escape trajectory exists.

## Actual trained-model evidence: supporting, not controlled causal comparison

The seed6301 pair used identical physical initial states and the actual verified BF16 LoRA. Baseline produced3 B predictions, stopped at33.2668% route completion and froze; released8m produced2 B predictions, completed100% at34.84s and did not freeze. Both had zero crashes and zero fallbacks. However CPU inference timings differed: baseline24.54/20.81/20.61s versus reserve18.15/19.23s. This is a successful integration demonstration, not an isolated estimate of reserve effect or a model-performance benchmark. The controlled-delay matrix carries the causal comparison.

The separate24-forward image diagnostic establishes current-skill hint sensitivity on fixed pictures, with important training-support caveats. It is a distinct result and must not be used as the causal explanation for the all-B physical stall. The real pilot's decisions were correct for its tested blocked-lane task.

## Recommended claim

“On a prespecified physical lane-change condition grid, delayed correct high-level decisions caused a collision-free progress failure under two low-level controllers. A development-selected stopping reserve, released when a changed maneuver target was admitted, completed all48 held-out conditions versus12/48 for both the original filter and a permanently enlarged clearance. A geometric reserve did not improve on the fixed reserve. An actual trained-model CPU pair reproduced the integration failure and recovery, with uncontrolled timing differences.”

Do not claim state-of-the-art performance, a new CBF deadlock phenomenon, general VLM safety, a certified maneuverability kernel, or novelty of simply preserving future options. The clean next experiment—if further work is justified—is a separately frozen stale/rejected-intent study comparing identical reserves released on any response, timer, and fresh admission, plus a saved-state alternative-steering rescue. Those discriminate the remaining interface hypothesis without inventing a more elaborate model.
