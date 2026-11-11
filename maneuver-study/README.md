# A correct answer can arrive after the car has lost room to turn

The new tests found a concrete failure and a working repair. The original car brakes safely behind a parked vehicle, but stops so close that its installed forward-driving controller cannot complete a late lane-change command. Keep 8 m of extra stopping room while waiting, then allow that extra room to be used after a fresh target-lane check accepts the turn. The ordinary collision/braking/boundary protections stay active.

**Held-out result: 48/48 completed conditions with the repair, versus 12/48 with the original. No observed collisions in either arm.** Keeping the extra 8 m permanently also completed only 12/48: room must be available to spend during the turn. These are deterministic physical condition comparisons, not 48 independently sampled traffic scenes or a population safety guarantee.

The actual trained camera model also completed the previously failing physical drive with this repair. This is supporting integration evidence; CPU inference timings differed between arms.

## What was actually run

All new work is isolated in `codex/maneuver-reserve`, based on `e891b3d`. Original `harsh-testing` source remains intact. This study adds:

| Experiment | Executions | Meaning |
|---|---:|---|
| Development grid |252 physical episodes |36 conditions×7 reserve policies; choose constant reserve |
| Held-out grid |192 physical episodes |48new condition combinations×4 policies; no retuning |
| Actual trained-model pair |2 physical episodes,5 real predictions |Same initial world; measured CPU timing differs |
| Identical-image text intervention |24 real predictions on 8 images |Only current-skill text changes within each image |
| Mechanism traces |3 physical reruns |Original, released 8 m, permanent 8 m; same 18 s correct command |
| Earlier development pilot |12 physical episodes |Exploratory and excluded from grid totals |

The 252+192 controlled episodes use a programmed correct command, not a VLM. This deliberately isolates delayed execution from visual mistakes. PPO and a simple PD controller are both tested. The underlying driving is MetaDrive/Bullet physics, not a laterally translating point-mass vehicle.

## Controlled results

Development crosses steering limits 15/25/40 degrees, speed/gap 14 m/s/45 m or 26 m/s/90 m, first-command delay 0.125/12/18 s, and PPO/PD. A constant 2 m reserve completed 20/36;4 m completed 28/36;8 m,12 m,20 m and the geometric heuristic each completed 36/36. Select 8 m as the smallest swept fully successful constant; this selection rule was chosen after seeing development, explicitly documented before held-out execution.

Held-out combinations cross steering 18/30/40 degrees, speed/gap 18 m/s/55 m or 24 m/s/80 m, delay 0.125/6/18/30 s, and both controllers. All 48 conditions were frozen before their first outcomes. Every policy receives the same actual initial state and actuator geometry within its matched condition.

| Policy | Completed /48 | Crashes | Freezes | Mean route | Mean end time, including timeouts |
|---|---:|---:|---:|---:|---:|
| Original protections |12|0|36|53.42%|47.18 s|
| Selected 8 m, released on admitted target change |48|0|0|100%|23.03 s|
| Geometry heuristic, same release |48|0|0|100%|23.07 s|
| Permanent extra 8 m |12|0|36|49.69%|47.26 s|

Each controller has 24 conditions: original/permanent reserve 6/24, released/geometric 24/24. At 0.125 s all policies finish. At 6/18/30 s the original and permanent-reserve arms finish 0/6 for each controller; released/geometric finish 6/6. The selected repair turns 36 matched failures into successes and turns no matched successes into failures.

There are only 6 unique initial physical states per grid, including steering geometry. Delays and controllers reuse them; do not manufacture confidence intervals by treating all rows as independent scene samples. Success means no collision and ≥99% route within 60 simulated seconds. These successful held-out runs report 100% route. Every timeout is retained in time/progress summaries.

The synchronous compatibility backend delays the first changed-target command; its subsequent same-skill B replies bypass pending delay. Thus this is specifically a late-first-intent experiment, not a repeated asynchronous perception workload. The separate actual-model pair uses real asynchronous inference.

## Why the repair works in the trace

In the recorded development case the lead is atx45 m, ego starts atx0 at 14 m/s, steering limit 15 degrees, and the identical correct command arrives at 18 s.

Just before that command, the original car is atx38.12 m and essentially stopped. The reserve versions are atx30.16 m, also stopped. After target admission the released-reserve version can accelerate and steer around the lead, completing the route. Permanent reserve keeps asking for hard braking and remains stopped. Original protection remains active after optional reserve release.

This demonstrates a failure of the installed controller/filter combination. It does not prove that no other steering trajectory could escape the original stopped state. The release comparison isolates allowing optional room to be used; because all controlled commands are valid B, it does not independently establish that a fresh guard is better than releasing on any answer.

## Actual trained-model rerun

The same exported teammate BF16 LoRA is used, with all 504 adapter tensors verified against the saved file. The cached base is Qwen/Qwen2.5-VL-3B-Instruct revision `66285546d2b821cf421d4f5eb2576359d3770cd3`; the original training did not pin its base revision. Exact v1 prompt and constrained one-token A/B/C inference are used. No substituted/fallback answers.

Both physical runs use the prior failure seed 6301 and exactly equal saved initial states, including the original randomized two-trap scenario. The repaired car reaches the second moving lead and follows it to the route end.

| Actual CPU camera run | Route | Crash | Frozen | Simulated end | Actual predictions / applied |
|---|---:|---:|---:|---:|---:|
| Original |33.27%|No|Yes|60 s timeout|3/2|
| Released8 m |100%|No|No|34.84 s route complete|2/1|

All five actual predictions were B. Baseline prediction durations were 24.54/20.81/20.61 s, repair 18.15/19.23 s; capture overhead and in-flight cleanup are retained in traces. One prediction in each run finished after the route/time limit and did not drive the car. Episode wall times including startup/cleanup were 126.46 s and 71.56 s. The different timing prevents a clean causal claim from this one pair; the controlled grid covers matched delays through 30 s. Concurrent CPU work also affected these local timings. This is not evidence about GPU throughput or general driving accuracy.

The teammate's separately saved A100 closed-loop results already showed a different problem: initial correct B followed by premature C while still physically right. Their saved static 300/300 and 50/50 classification scores therefore do not establish closed-loop competence. See the pinned GPU files and audit; do not combine those point-mass/ghost-render experiments with the physical grid as one benchmark.

## Image/text finding and its limits

Across eight selected physical-camera images, changing only current-skill text changes letters in 8/8 and lane targets in 6/8. A and C share a right-lane target and currently alias the same calibrated low-level policy, so an A/C flip is not demonstrated distinct driving behavior. All RETURN_RIGHT hints produceC. All 24 forwards are actual trained-model evaluations with identical PNG bytes within each group.

The pinned training generator assigns CRUISE_RIGHT only when physically RIGHT, OVERTAKE_LEFT only when physically LEFT, and never RETURN_RIGHT. Many interventions therefore cross combinations absent from that generator. Its lane-change OOD generator explicitly creates such cross-combinations. The finding demonstrates dependence on hint text and a coverage gap worth testing; it does not prove data leakage, identify the training cause, or measure representative accuracy. In the 8 supported lane/skill pairings, predictions match all 8 design labels. Detailed logits, probabilities and caveats are in[vision/RESULTS.md](vision/RESULTS.md).

A separate attempt to prove exact camera aliasing failed: 0/6 bit-identical image pairs. Near-identity and differing ground-truth admission do not prove identical-input impossibility. Failed rendering controls remain preserved;[audit](audits/aliasing.md).

## Novelty verdict

**A measurable project improvement is established; algorithmic novelty is not.** The sophisticated geometry version fails to beat a tuned constant with identical release logic. Generic CBF deadlock, keeping future maneuvers feasible, and planning around safety interventions are established topics;[SHARP](https://arxiv.org/abs/2110.00843) is especially close. The full[prior-work audit](audits/prior-work.md) and[independent contribution review](audits/contribution-verdict.md) explain the overlap.

The defensible research story is the evaluation gap: static visual accuracy, timely intent, and executable physical action are different requirements. This project now has reproducible examples of a fast model issuing premature reversal and a slow model giving the correct instruction after useful turning room is lost. The present repair addresses the latter. A publishable new method would still need a stronger contract for executing/rejecting asynchronous intentions, adversarial/wrong/absent answers, moving and blocked lanes, and credible predictive/reachability baselines. Do not relabel this 8 m heuristic as a certified viability method.

Braking-infeasible ticks remain nonzero in successful traces (held-out selected-reserve mean 494.6 ticks). Keeping existing filters active and observing no collision is not proof of recursive feasibility, invariance, or formal safety. Enlarging a clearance can itself abruptly make an augmented stopping envelope infeasible; the ordinary fallback still runs.

## Evidence and reproduction

The release contains source, configs, checkpoints, model adapter, locks, all new raw results and figures. The Git branch also carries `raw-runs.tar.gz`; extract it from the repository root with `tar -xzf maneuver-study/raw-runs.tar.gz` to restore the original run paths if needed. The 7.51 GB base weights and Python environments are external. Use a compatible existing environment or install the package's MetaDrive/dev dependencies plus Transformers 4.57.6, Accelerate 1.15.0, PEFT 0.17.1 and Safetensors 0.8.0. Exact observed versions are in `environment.json`; original CPU helper uses native BF16/SDPA with no merge. CPU model runs need about 9 GB working memory and take minutes.

From the extracted`event-cbf-driving`directory, with the package environment active:

```bash
PYTHONPATH=. python scripts/maneuver_matrix.py development --workers 4
PYTHONPATH=. python scripts/maneuver_matrix.py heldout --workers 4
PYTHONPATH=.:scripts python scripts/maneuver_analyze.py
PYTHONPATH=. pytest -q
```

Existing complete grids are hash-checked and resumed without overwriting. Use a fresh versioned output/source checkout if changing the design; frozen runs reject changed sources. The mechanism rerun is saved, so analysis works directly on the delivered evidence. Actual-model commands/protocols are in `PROTOCOL.md`, script CLI help, and`vision/PROTOCOL.md`. Use `P3D_DISPLAY=p3headlessgl` and `LP_NUM_THREADS=1` for local software camera rendering.

The image experiment executed an earlier development version of the reserve module, before an unused speed-scaled branch's divide-by-zero bug was fixed. Its exact 57 hashed source files are preserved in `vision/executed-runtime-sources.zip`and match its pre-inference lock. The reserve policy is not used by that image-only experiment. The final matrix and actual-drive locks match the final reserve source.

Verification: 342 tests passed; Ruff passed; mypy passed for 53 package files; complete job IDs/source/checkpoint hashes and all matched initial states checked; actual-model pair initial states equal;57image-experiment archived-source hashes verified. The 119 original package snapshot files, excluding a research-only ignore helper, remain byte-identical. No cloud jobs, repository pushes, merges, or base-weight modifications.
