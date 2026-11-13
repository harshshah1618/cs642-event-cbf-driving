# Safe Fast Driving While a Slow Camera Model Thinks

### Abstract

We evaluate a closed-loop driving stack in which a slow camera model chooses a discrete maneuver, frozen PPO skills execute it, and fast filters modify unsafe commands. The final comparison contains 1,440 physical MetaDrive episodes across fifteen policies, three scenario families, eight seed clusters and four request delays. In the 0-1 s nominal stratum, the frozen integrated system completes 72/72 episodes with 0 collisions, using 1.08 simulated costly requests per episode versus 75.82 for 10 Hz polling (98.6% fewer). All protected replanning controls also complete every nominal episode. Simpler event triggers match or improve the final scheduler; a cheap local rule requires no model calls. The contribution is a reproducible engineering comparison and an analysis of delay, stopping room and transitions, not a new safety theorem or superiority over published driving systems.

## 1  Problem and course scope

The proposal asks whether a vehicle can keep controlling safely while a slower planner thinks, avoid stopping indefinitely behind an obstacle, reduce unnecessary planning calls, and change maneuvers smoothly. It explicitly proposes combining established components under application/evaluation categories, not inventing a completely new algorithm [1]. We retain that scope. A correct late instruction can still fail if braking has left too little room for the installed forward-driving controller to turn.

Figure 1. The model chooses a maneuver; it does not supply continuous acceleration. Fresh simulator state is available to scheduling and safety even when the model has not answered. This is a hybrid state/vision simulator stack, not a vision-only driving system.

The final variant revises the original margin hypothesis: it retains 8 m of optional stopping room while intent is unresolved, then releases that optional room after a fresh target-lane check accepts a changed target. Ordinary collision, braking and boundary checks remain active. The original uncertainty and speed-times-delay margins are tested explicitly rather than assumed necessary.

## 2  One implementation and fifteen comparison policies

Every policy uses the same 50 Hz DrivingStack, physical acceleration adapter and frozen PPO bank [2]. The two distinct learned policies regulate lane position and speed from ego state (y, vy, vx); they do not perceive traffic. RETURN_RIGHT reuses the cruise-right checkpoint. A CBF-style action correction, stopping-envelope backup, boundary check and fresh lane-admission check supply runtime protection.

| Policy | Decision timing | Protection | Transition |
| --- | --- | --- | --- |
| Unprotected RL | Cruise skill, no router | Shield disabled | None |
| Static CBF | Cruise skill, no router | CBF + boundary | No maneuver change |
| 1 Hz, unshielded | Timer; same planner | Shield disabled | Hard switch |
| Uncertainty wake-up | Critic threshold only | Full package | 8 m + blend |
| Local barrier replan | Sustained override; cheap rule | Full package | 8 m + blend |
| Original proposal | Critic OR override | Full + original inflation | No 8 m; blend |
| Final integrated | Opportunity deadline | Full, no added inflation | 8 m + blend |
| 1 Hz, matched safety | Timer; same planner | Same as final | 8 m; hard switch |
| 10 Hz polling | Timer; same planner | Same as final | 8 m + blend |
| 10 Hz, gated | Timer + shared useful gate | Same as final | 8 m + blend |
| Immediate benefit | Ask as soon as useful | Same as final | 8 m + blend |
| No extra reserve | Same deadline | Same as final | No 8 m; blend |
| No blending | Same deadline | Same as final | 8 m; hard switch |
| Add delay margin | Same deadline | Final + pending margin | 8 m + blend |
| Add critic margin | Same deadline | Final + critic margin | 8 m + blend |

The uncertainty and barrier controls are local implementations inspired by the proposal [3,4], not ports of FASIONAD or CERT-Replan. We do not reproduce their training, visual fusion, calibrated risk or finite-horizon mode search. The local rule has zero artificial inference delay; its replans are reported separately from costly requests. The matched polling controls prevent a weak unprotected timer from being the only comparator.

## 3  Frozen experimental protocol

| Component | Frozen choice |
| --- | --- |
| Physical simulator | MetaDrive/Bullet; pinned source fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1 |
| Scenario families | Parked lead; parked lead plus later moving cut-in; dense two-lane traffic |
| Worlds and matching | Eight traffic seeds, three scene profiles; identical initial state within each comparison |
| Nominal delay stratum | 0, 0.125 and 1 s; 72 episodes per method |
| CPU stress stratum | 18 s; 24 episodes per method; reported separately |
| Episode budget | 200 m route, 60 simulated seconds, 50 Hz physical control |
| Development | 36 threshold-selection episodes; 90 contract-pilot episodes; excluded from test |
| Actual model subset | Six additional worlds; three queried and three no-model policies |

The 8 m reserve was selected in the earlier disjoint maneuver development study. For this final test, critic-threshold scales 0.2, 0.5 and 1 were compared on 36 development episodes. All achieved 12/12 success and zero collisions; scale 1 used eight requests versus fifteen and twenty-nine, so it was frozen for the uncertainty-only control. No parameter was retuned on the held-out results.

The controlled matrix uses one privileged-state lane/occupancy rule, not a VLM. It extends the original parked-car semantics by treating moving leads below 15 m/s as blockers. All synchronous responses, including same-skill answers, occupy the specified delay. Polling starts at a seed-randomized fraction of its period. The local barrier rule alone receives zero artificial compute delay.

Source, config, checkpoint and complete job-design hashes were frozen before the test. All 24 scene/seed groups match initial states exactly; there are 16 distinct initial-state values because the parked-lead and cut-in profiles share their start and differ in a later event. Eight seeds, not 1,440 episodes, are the resampling clusters. Descriptive 95% paired intervals use 10,000 joint seed resamples; they are not population safety or equivalence certificates.

### Metrics and exposure

Success requires no collision and at least 99% route progress. Exact route-complete end reasons, timeouts, terminal stalls and the legacy freeze flag are retained separately. A terminal stall means remaining route with final speed below 0.5 m/s and no collision. We also record stopped duration, requests, cheap local replans, rejection/expiry, crashes during pending calls, interventions and infeasible ticks. Achieved lateral jerk comes from changes in physical lateral acceleration, not requested commands. Failed drives remain in primary summaries; a jointly successful subset is reported only as a separately labelled comfort comparison.

The cut-in activates only after x = 92 m. A car that stalls earlier never encounters it. Exposure and first observation during a pending call are therefore measured, not assumed. Route success does not require returning to the right lane; request savings can partly reflect remaining left while still making progress.

## 4  Integrated comparison under nominal delays

Each row contains the same 72 nominal scene/seed/delay conditions. Costly requests are simulated delayed planning requests in this matrix; no actual model inference is hidden in these counts. The local baseline makes cheap replans instead.

| Policy | Success | Crash | Stall | Requests | Jerk RMS |
| --- | --- | --- | --- | --- | --- |
| Unprotected RL | 0/72 | 72 | 0 | 0.00 | 0.02 |
| Static CBF | 24/72 | 12 | 36 | 0.00 | 3.10 |
| 1 Hz, unshielded | 14/72 | 58 | 0 | 4.65 | 6.73 |
| Uncertainty wake-up | 72/72 | 0 | 0 | 0.67 | 4.96 |
| Local barrier replan | 72/72 | 0 | 0 | 0.00 | 6.02 |
| Original proposal | 72/72 | 0 | 0 | 3.06 | 7.41 |
| Final integrated | 72/72 | 0 | 0 | 1.08 | 3.97 |
| 1 Hz, matched safety | 72/72 | 0 | 0 | 13.82 | 8.39 |
| 10 Hz polling | 72/72 | 0 | 0 | 75.82 | 3.94 |
| 10 Hz, gated | 72/72 | 0 | 0 | 1.08 | 3.68 |
| Immediate benefit | 72/72 | 0 | 0 | 1.08 | 3.54 |
| No extra reserve | 72/72 | 0 | 0 | 1.08 | 3.97 |
| No blending | 72/72 | 0 | 0 | 1.08 | 5.07 |
| Add delay margin | 72/72 | 0 | 0 | 1.08 | 3.96 |
| Add critic margin | 72/72 | 0 | 0 | 1.08 | 4.71 |

Figure 2. Counts retain crashes and timeouts. Extra protections differ in the deliberately weak unshielded/static arms, so their gap measures the whole package rather than one CBF term. Jerk units are m/s^3; early crashes have shorter comfort exposure.

The final variant records 72/72 success and 0 collisions. The original proposal and every protected replanning control also finish 72/72. Thus safety/progress are improved over the weak baselines but do not distinguish the stronger protected systems. The local rule finishes all conditions without model-like calls; the visual model is not shown necessary by a privileged-state benchmark.

## 5  Request cost, delay and component ablations

Figure 3. Nominal and 18 s stress are separate strata. The final variant averages 1.08 requests versus 75.82 for 10 Hz polling. This is 98.6% fewer with equal observed nominal success. Gating matches the request count; uncertainty wake-up uses fewer requests (0.67).

| Ablation | Nom. success | Stress success | Nom. RMS | Nom. peak | Stress end s |
| --- | --- | --- | --- | --- | --- |
| Final integrated | 72/72 | 24/24 | 3.97 | 75.74 | 26.46 |
| No extra reserve | 72/72 | 8/24 | 3.97 | 76.28 | 45.16 |
| No blending | 72/72 | 24/24 | 5.07 | 93.25 | 26.50 |
| Add delay margin | 72/72 | 24/24 | 3.96 | 75.39 | 30.45 |
| Add critic margin | 72/72 | 24/24 | 4.71 | 92.23 | 26.87 |
| Original proposal | 72/72 | 24/24 | 7.41 | 146.35 | 34.67 |

The added reserve changes no nominal success count. Under 18 s delay, removing it drops success from 24/24 to 8/24. This is a progress result: stalled cars often never reach the cut-in. The original inflation variant also completes 24/24 stress episodes, so a reserve is one successful repair rather than the only possible solution.

On 72 jointly successful nominal pairs, blending lowers mean achieved RMS by 1.10 m/s^3 and mean peak by 17.51 m/s^3 versus no blending. This is an aggregate result, not uniform improvement on every case. Crash/timeout exposure and per-seed differences remain in the full data. Pending-margin necessity is unestablished when both matched variants avoid collision; the stress comfort/progress tradeoff remains meaningful.

At 18 s, adding the pending margin lowers mean achieved RMS from 7.28 to 2.31 m/s^3, while increasing mean end time from 26.46 to 30.45 s; both finish 24/24. This is a comfort/progress tradeoff, not evidence that the margin is useless.

| Final minus comparator | Paired mean difference | Descriptive 95% interval |
| --- | --- | --- |
| Requests vs 10 Hz | -74.74 | [-76.12, -73.32] |
| RMS jerk vs no blend | -1.10 | [-1.35, -0.87] |
| Peak jerk vs no blend | -17.51 | [-24.39, -11.01] |

The immediate-benefit control matches final completion and request count and has lower mean RMS and peak jerk. The deadline term therefore has no demonstrated incremental advantage. We keep the preselected final variant in the table instead of silently switching to the held-out winner.

## 6  Actual trained camera model integration

The actual subset uses the exported teammate BF16 LoRA with Qwen2.5-VL-3B-Instruct [7]. All 504 adapter tensors are verified against the saved file. The base is pinned to revision 66285546d2b821cf421d4f5eb2576359d3770cd3; the original training did not pin its base revision. The unchanged v1 prompt and one-token A/B/C constrained inference are used. There are no scripted fallback replies. Scheduling and safety still observe simulator state; only the router model receives image, lane text and current-skill hint.

| Policy | Success | Crash | Stall | Mean route | Predictions |
| --- | --- | --- | --- | --- | --- |
| Unprotected RL | 0/6 | 6 | 0 | 37.7% | 0 |
| Static CBF | 2/6 | 0 | 4 | 55.7% | 0 |
| Local barrier replan | 6/6 | 0 | 0 | 100.0% | 0 |
| 1 Hz, matched safety | 5/6 | 0 | 0 | 92.9% | 11 |
| Uncertainty wake-up | 6/6 | 0 | 0 | 100.0% | 4 |
| Final integrated | 6/6 | 0 | 0 | 100.0% | 7 |

The completed subset contains 36 physical episodes, including 18 VLM-enabled runs over six scene/seed worlds. 15 episodes actually request predictions; the model produces 22 answers with 0 fallbacks. 12 model-enabled episodes admit a model-proposed change of lane target. Initial states match exactly. Queried policy order rotates evenly. Median recorded prediction time is 25.66 s, range 19.23-47.01 s.

A fixed 30 s scheduling estimate and 45 s pending deadline are used; actual measured inference plus capture time determines when an answer becomes ready. This estimate is not an imposed delay or a proven upper bound. The model stays loaded locally with three CPU threads and a 10 GiB/no-swap limit. The controlled pool overlaps the initial live cases, so timings include contention and are not isolated inference-throughput measurements.

The 50 Hz loop is measured in simulation time. CPU inference and rendering can make wall pacing fall behind; this study does not establish a hard real-time 50 Hz control guarantee.

The original prompt assumes all other cars are parked and says 20 m/s even when actual speed or traffic differs. The model was trained mostly on static lane/skill combinations; transitions can fall outside that support. Slow CPU response can also hide rapid premature reversals seen with faster inference. These runs cannot establish GPU closed-loop behavior or representative model accuracy. Zero-request episodes, rejected answers and in-flight replies returned after termination remain in the records.

An inherited telemetry path omits capture time when joining an unfinished answer after driving has ended. Such answers are never applied. Raw records are preserved; capture and model wall times are separately available, and no end-to-end latency claim relies on that cleanup value.

Interpret this subset as integration evidence on six worlds. It is deliberately separate from the 1,440 controlled state-rule episodes and is too small for a broad safety or VLM-superiority claim.

All twelve admitted model target changes are cruise-right to overtake-left. No model-proposed return-right target is applied before termination; the final variant's two C replies arrive after the episode ends. Timer routing completes five of six worlds; its missed route includes a response above the fixed 45 s deadline. The timing difference prevents clean scheduler-causality claims. The local rule and uncertainty wake-up both complete six of six.

## 7  Proposal outcomes and interpretation

| Proposal objective | Measured status |
| --- | --- |
| Lower crash than unprotected RL and timer-hard | Met on nominal observed counts; whole-package comparison |
| Lower freeze/stall than static CBF | Met for terminal stalls; legacy freeze also retained |
| Fewer requests with matched progress | Met versus high-rate polling (98.6% fewer); simpler triggers tie or improve |
| Switches not dominated in comfort | Lower mean RMS and mean peak than matched no-blend; not every individual case |
| Survive pending calls with inflation on | Zero observed pending collisions in 96 margin-on episodes; also zero with margin off |
| Added pending inflation is necessary | Not established; zero-collision equality cannot demonstrate necessity |
| New algorithm or official published-system win | Not claimed by proposal and not established by this study |

The evidence supports the course contribution: implementing known pieces as a reproducible physical stack and measuring safety, delay, progress, calls and achieved motion. The system includes real learned policies and an actual fine-tuned camera router. It does not show that PPO beats a simple driver, or that the slow model is necessary when a cheap rule already receives privileged state.

Several engineering choices matter jointly. A safety filter can prevent immediate collision but cannot invent a missing maneuver; stopping too close can also make a later correct command unusable for the installed controller. Extra stopping room must be available to spend during execution. This is controller-relative failure and recovery, not proof that no alternative steering trajectory could escape. Nonzero infeasible ticks remain in successful traces.

### Related work and claim boundary

PPO supplies the existing learned lane regulators [2]. FASIONAD motivates selective use of a slower reasoning pathway [3]. CERT-Replan motivates reacting to sustained planning risk, but our local trigger does not implement its calibrated horizon-level machinery [4]. Adaptive network-delay margins motivate the original margin variant without transferring a theorem to this high-level intent delay [5]. SHARP already studies planning with anticipated shielding effects [6]. Our claim is an application/evaluation result, with no priority or official-benchmark claim.

### Limits that change the interpretation

The roads are straight, the traffic distributions are narrow, and only one frozen PPO bank is evaluated. Scheduling and filters have ground-truth simulator object state; sensing errors are not propagated through an end-to-end perception system. Eight seed clusters and six actual-model worlds do not establish general driving safety. The route objective does not enforce right-lane return. All protection mechanisms are approximate sampled controllers, not a validated invariant set under uncertain vehicle dynamics. The final deadline is not the strongest simple trigger in this benchmark; the negative comparison is part of the result.

The next practical configuration choice would favor the simplest method that meets the desired safety, progress and comfort constraints. That recommendation is not used to retune or relabel the held-out results in this report.

## References

[1] Project team. Safe Fast Control While a Slow Language Model Thinks. Original course proposal, 18 October 2026. Included as proposal.pdf.

[2] J. Schulman et al. Proximal Policy Optimization Algorithms. 2017. https://arxiv.org/abs/1707.06347

[3] K. Qian et al. FASIONAD. 2024. https://arxiv.org/abs/2411.18013

[4] R. R. Suganda and B. Hu. Adaptive Risk-Certified Event-Triggered Replanning for Dynamic Navigation. 2026. https://arxiv.org/abs/2610.09302

[5] R. Periotto et al. MPC-CBF with Adaptive Safety Margins for Safety-critical Teleoperation over Imperfect Network Connections. 2024. https://arxiv.org/abs/2403.18650

[6] H. Hu, K. Nakamura and J. F. Fisac. SHARP. 2022. https://arxiv.org/abs/2110.00843

[7] Qwen team. Qwen2.5-VL-3B-Instruct model card. https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

## Artifact and reproduction notes

The release includes the frozen controller source, simulator configs, PPO checkpoints, trained adapter, full compressed action/plant traces, raw rows, source/checkpoint locks, selection record, all-condition tables, paired statistics and independent audits. Base weights and Python environments are external. The base revision is pinned; the original user checkout and previous studies are preserved. No cloud jobs or repository pushes are needed for this evaluation.

From event-cbf-driving with the recorded project environment and restored results:

`PYTHONPATH=. python scripts/integrated_eval.py test --workers 4`

`PYTHONPATH=.:scripts python scripts/integrated_analysis.py --with-live --audit-traces`

`PYTHONPATH=. pytest -q`

Complete frozen runs resume without overwriting, and changed source/design is rejected. To make new policy or scenario changes, use a fresh versioned output location rather than relabeling the held-out suite. Actual-model reruns use integrated_vlm.py with the local pinned base path; the report can be reanalyzed from the saved data without loading that base.

Observed environment versions, final test/lint/type results and archive-restoration checks are recorded in verification.json. The source lock for this main comparison is b533186dead4df77f40b916a6341e00bfa2ecca4d578b7026d1998e3d3f0aad5.

Report status: Complete controlled and actual-model evaluation.
