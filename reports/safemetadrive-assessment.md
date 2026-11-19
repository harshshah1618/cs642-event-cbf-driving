# SafeMetaDrive assessment from saved runs

**The system works, but its main observed completion problem is getting stuck, and the current evidence does not establish a benefit from a large camera model or PPO refinement.** The most useful next work is to diagnose and repair coordination between planning, maneuver execution and protection on training/development maps. Another broad training round is not justified yet.

This assessment used existing local reports and CPU replays only. No Modal jobs, camera-model inference or training were launched. The benchmark implementation and trained weights were not changed. All original frozen source hashes still match. The test results below are retrospective diagnostics, not new validation data for selecting an improved model.

The primary camera/PPO/protection system made 150 drives across 50 shared maps and three paired camera/controller replicas:

| Outcome | Drives |
|---|---:|
| Reached destination, zero native penalty | 134 |
| Reached destination, some native penalty | 4 |
| Timed out, zero native penalty | 11 |
| Failed after vehicle collision and leaving the road | 1 |
| **Total** | **150** |

Thus 138 drives succeeded, 134 were clean successes, and only five accumulated positive native cost. The 0.333 average is 50 summed penalty units divided by 150 drives. It does not mean 33% crashed or exactly one collision every three drives. A single contact can produce penalties over multiple commands. One drive contributed 27 of those units, or 54% of all cost.

| Camera/controller replica | Successes / 50 | Summed cost |
|---|---:|---:|
| 0 | 46 | 0 |
| 1 | 46 | 41 |
| 2 | 46 | 9 |

Equal completion hides considerable variation in safety cost. The camera and PPO random seeds were paired, so variation cannot be assigned independently to one model without further comparisons. Positive-cost drives were replica1/maps513,529,533 and replica2/maps513,533. Map513 also produced cost in the rule-plus-PPO comparison, so the camera is not necessary for every observed collision problem. The 27-cost drive on map529 requested no executed lane changes; collisions cannot all be explained as unsafe lane-change admission.

All eleven timeout drives were below 0.1 m/s at every request during their final 40.5–87 seconds of sampled history. These are query-time observations, not continuous telemetry from the original run. A subset was then replayed to inspect continuous control behavior.

| Replica | Map | Original outcome | Final span sampled below 0.1 m/s | Longest consecutive rejected requests |
|---|---:|---|---:|---:|
| 0 | 514 | Timeout | 64.5 s | 1 |
| 0 | 524 | Timeout | 87.0 s | 60 |
| 0 | 541 | Timeout | 70.5 s | 46 |
| 0 | 543 | Timeout | 40.5 s | 6 |
| 1 | 502 | Timeout | 48.0 s | 5 |
| 1 | 522 | Timeout | 66.0 s | 37 |
| 1 | 524 | Timeout | 85.5 s | 3 |
| 1 | 533 | Collision/road departure | — | 1 |
| 2 | 522 | Timeout | 69.0 s | 3 |
| 2 | 530 | Timeout | 64.5 s | 1 |
| 2 | 543 | Timeout | 58.5 s | 2 |
| 2 | 548 | Timeout | 85.5 s | 8 |

I replayed the saved camera letters and measured response times for all 16 problematic drives plus one successful control drive. Camera inference was not rerun. The deterministic 6–64–64–2 tracking actor was reconstructed from saved weights and run on CPU. Fourteen of seventeen replays matched original success, cost, command count, lane-change count and intervention count. Seven passed stricter checks as well: identical request counts/lane metadata/response statuses, maximum capture-speed difference below 1e-4 m/s and return difference below 1e-3. Those seven comprise six timeouts and the successful control.

Only these strict matches support the detailed original-run diagnoses below. This is stronger than guessing from outcome summaries, but it is not a comparison of every original physical state: original per-step positions and camera frames were not saved in the final reports. The local Torch version differs from cloud training/evaluation. Three collision replays changed cost or other endpoint counts, so the exact original collision sequences remain unresolved. See [replay verification](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-assessment/replay-verification.json).

Two reproduced cases explain why blaming camera classification alone is insufficient:

- **Map524, replica0:** protection held the target speed at zero while the car remained about 1.12 m from its committed lane center. The system rejected 60 requests because another lane change is admitted only within 0.65 m of that center. The camera kept proposing changes, but the car had no explicit process for resolving the unfinished maneuver. It timed out without hitting anything.
- **Map530, replica2:** the car remained about 0.93 m from the committed lane center, again with a zero target speed. In its last eight requests the camera and the current-state rule teacher both chose stay/careful. Agreement on that instruction did not restore movement. A different earlier maneuver history may have avoided the state, but changing the final camera label alone is not demonstrated to solve it.

![CPU replays of two stalled drives](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-assessment/stall-diagnostics.png)

Source inspection supports an interface limitation. [Lane admission](/home/harsh/Projects/cs642-event-cbf-driving/driving/safemetadrive/policies.py:43) returns only true/false. The [camera prompt](/home/harsh/Projects/cs642-event-cbf-driving/driving/safemetadrive/camera_planner.py:29) contains pictures, speed and lane indices, but no rejection reason or elapsed time without progress. The [fast controller](/home/harsh/Projects/cs642-event-cbf-driving/driving/safemetadrive/policies.py:203) can stop for a forecast conflict while a lane transition remains unfinished; it has no explicit blocked-maneuver recovery state. In the two plotted replays, a stationary object remained about 6.3 m ahead and the forecast repeatedly returned a 2.8 s conflict. The forecast assumes at least 2 m/s ego motion; its under-3-second branch caps the target at max(0,current speed minus 1 m/s), which becomes zero at standstill. This explains the persistent hold under unchanged geometry; it does not show that moving forward would have been safe. Caution and lane choice are also admitted together: rejecting a lane change rejects its caution update. These are candidate design weaknesses, not proof that simply relaxing checks would be safe or improve results.

The written-rule planner with the same PPO replica and protection completed all 150 drives, with total cost 3. That means the PPO controllers are capable of completing these routes under a different sequence of targets. It does not prove PPO is optimal, that every camera instruction was wrong, or that the filter is infallible. The copying-only controllers with the rule planner also completed all drives with zero cost; an advantage from PPO refinement remains unestablished.

The camera nevertheless helped relative to the fixed-lane controller. On paired map/replica drives, it completed 70 drives that fixed-normal control failed, and lost one that fixed-normal completed. That establishes a benefit of this active lane/caution planning branch over the fixed choice. It does not distinguish language-model reasoning from a small learned image classifier or separate lane selection from caution.

Protection had a substantial system-level effect:

| Condition | Successes | Clean successes | Drives with positive cost | Summed native cost |
|---|---:|---:|---:|---:|
| Protection enabled | 138 | 134 | 5 | 50 |
| Protection disabled | 143 | 85 | 65 | 899 |

The enabled configuration recorded 94.4% less cost and 49 more clean successes, at five fewer destination completions overall. This supports keeping protection while studying progress recovery. The switch changes lane-gap admission, target-speed limits/forecasting and the final longitudinal cap together; it does not isolate one final safety-filter operation. Steering is not projected through a certified lateral safety constraint.

The delay results remain a limitation, not a demonstrated research victory. The 4-second condition completed149/150 drives with total cost3, but it also reduced query frequency and changed input-history spacing. Only 31.6% of its requests were applied versus 90.9% at zero added delay. Fewer applied changes or different traffic timing are hypotheses, not established explanations. Selecting 4 seconds now would be test-set tuning. The previously reported paired-map intervals are conditional on the three supplied trained systems, and the 50 shared maps are not 150 independent maps.

My recommendation is to pause new PPO/LoRA training. First, check a proposed lane-change trajectory for feasibility before committing, add diagnostic feedback, and design an explicit maneuver lifecycle: accepted, executing, blocked, completed or safely abandoned. Progress recovery must assess a feasible path and current traffic; it must not blindly override stopping or lane-gap checks. Develop and evaluate that behavior on training/development scenarios, then use fresh held-out maps for any improved-system claim. Separately, the camera needs a same-input small-model comparison before claiming a particular benefit from a large VLM. The original collision causes require better event telemetry or a matching-runtime replay before prescribing a collision-specific fix.

The defensible current project is a hierarchical driving system with demonstrated protection benefits and identifiable coordination failures. Universal safety, official SOTA, a PPO-refinement advantage, a large-VLM advantage and monotonic harm from delay are not established by these runs.

Detailed evidence: [numerical assessment](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-assessment/analysis.json), [problem-drive table](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-assessment/episode-audit.csv), and [original frozen-study report](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results.md). Recompute existing-log analysis locally with `.venv/bin/python tools/safemetadrive_assess.py`. Reproduce a diagnostic with `CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/safemetadrive_replay.py --cases 0:524 2:530`. Neither command uses Modal or runs the camera model.
