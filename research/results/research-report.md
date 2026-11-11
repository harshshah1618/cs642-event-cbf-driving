# When should a slow maneuver planner run?

Audited physical MetaDrive scheduling study. November 11, 2026.

## Measured outcome

The clean scheduling win occurs in the randomized-initial-offset suite: 63.0% fewer planner calls than scenario/delay-conditioned polling, with observed route-success 100.00% versus 99.17%, and zero crashes in both 360-episode groups. Native route-success means at least 99% of the route, not necessarily exact arrival.

This is not a universal win. The original zero-offset suite loses a small amount of strict success, the upstream procedural-road controller still crashes/timeouts, and the matched no-latency/TTC controls are highly competitive. The principal contribution is a controlled evaluation and timing-sensitivity finding, not established algorithmic novelty.

## Question and experimental controls

Can forecasting maneuver opportunity loss relative to planning delay improve the safety-progress-call tradeoff? Every scheduler in a suite shares the same driver, planner, observations and safety backup. The latency ablation changes only estimated latency after copying the selected deadline buffer.

There are two implementations: the original straight-road scenarios retain the existing frozen PPO lane regulators and CBF-style filter; upstream procedural roads use a separate classical PID/IDM tracker, finite-horizon lane/speed planner and common guard. This is not a demonstrated port of the original learned controller or CBF certificate to curved roads.

The strongest opponents include polling tuned independently for each scenario/environment and delay, a 10Hz polling policy with the same cheap usefulness gate, immediate usefulness, and TTC. Gated polling has a shorter minimum query cooldown; it is a strong comparator, not a pure deadline ablation.

Validation: 2,928 physical episodes. Held-out: 11,440 episodes. Original-scenario seeds 1001-1040; random-offset seeds 1101-1140; upstream road seeds 2001-2040. Every selection and source/checkpoint hash was frozen before its held-out launch.

## Original scenarios

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 600 | 98.3 | 0 | 100.0 | 1.08 | 13.89 |
| Deadline-aware | 600 | 98.3 | 0 | 100.0 | 1.06 | 14.16 |
| Original event trigger | 600 | 99.5 | 0 | 100.0 | 6.77 | 14.75 |
| Matched no-latency | 600 | 98.3 | 0 | 100.0 | 1.01 | 14.50 |
| Tuned polling | 600 | 99.2 | 0 | 100.0 | 2.00 | 14.62 |
| Condition-tuned polling | 600 | 99.2 | 0 | 100.0 | 2.00 | 14.62 |
| Gated polling (10Hz) | 600 | 98.3 | 0 | 100.0 | 1.08 | 13.90 |
| TTC threshold | 600 | 98.3 | 0 | 100.0 | 1.01 | 14.50 |

## Random initial polling offset

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 360 | 100.0 | 0 | 100.0 | 1.05 | 14.24 |
| Deadline-aware | 360 | 100.0 | 0 | 100.0 | 1.02 | 14.48 |
| Original event trigger | 360 | 96.9 | 0 | 99.9 | 6.85 | 15.03 |
| Matched no-latency | 360 | 100.0 | 0 | 100.0 | 0.96 | 14.96 |
| Tuned polling | 360 | 96.9 | 0 | 99.9 | 11.51 | 14.79 |
| Condition-tuned polling | 360 | 99.2 | 0 | 100.0 | 2.77 | 15.77 |
| Gated polling (10Hz) | 360 | 100.0 | 0 | 100.0 | 1.05 | 14.28 |
| TTC threshold | 360 | 100.0 | 0 | 100.0 | 0.95 | 14.96 |

## Upstream road stress

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 320 | 69.1 | 34 | 85.2 | 2.70 | 42.19 |
| Deadline-aware | 320 | 70.6 | 28 | 84.9 | 1.52 | 41.49 |
| No replanning | 320 | 53.8 | 24 | 75.9 | 0.00 | 49.81 |
| Matched no-latency | 320 | 72.5 | 28 | 85.3 | 1.20 | 41.63 |
| Tuned polling | 320 | 70.6 | 25 | 85.5 | 21.85 | 42.75 |
| Condition-tuned polling | 320 | 69.7 | 27 | 84.9 | 55.67 | 43.19 |
| Gated polling (10Hz) | 320 | 69.1 | 34 | 85.1 | 2.85 | 42.11 |
| Intervention trigger | 320 | 72.2 | 26 | 84.9 | 10.06 | 43.97 |
| TTC threshold | 320 | 72.5 | 28 | 85.3 | 1.20 | 41.63 |

## Paired deadline comparisons

isolation: versus Condition-tuned polling, success difference -0.83 percentage points (95% paired cluster-bootstrap interval -2.50 to +0.00); calls -0.94 (-0.99 to -0.89).

isolation: versus Gated polling (10Hz), success difference +0.00 percentage points (95% paired cluster-bootstrap interval +0.00 to +0.00); calls -0.02 (-0.04 to -0.01).

isolation: versus Matched no-latency, success difference +0.00 percentage points (95% paired cluster-bootstrap interval +0.00 to +0.00); calls +0.05 (+0.02 to +0.08).

phase: versus Condition-tuned polling, success difference +0.83 percentage points (95% paired cluster-bootstrap interval +0.00 to +2.50); calls -1.74 (-1.81 to -1.68).

phase: versus Gated polling (10Hz), success difference +0.00 percentage points (95% paired cluster-bootstrap interval +0.00 to +0.00); calls -0.03 (-0.06 to -0.01).

phase: versus Matched no-latency, success difference +0.00 percentage points (95% paired cluster-bootstrap interval +0.00 to +0.00); calls +0.07 (+0.03 to +0.11).

roads: versus Condition-tuned polling, success difference +0.94 percentage points (95% paired cluster-bootstrap interval -4.69 to +6.88); calls -54.15 (-60.15 to -47.93).

roads: versus Gated polling (10Hz), success difference +1.56 percentage points (95% paired cluster-bootstrap interval -3.75 to +6.56); calls -1.33 (-1.76 to -0.90).

roads: versus Matched no-latency, success difference -1.88 percentage points (95% paired cluster-bootstrap interval -4.06 to -0.31); calls +0.32 (+0.16 to +0.51).

## Upstream generalization: no injected delay

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 70.0 | 5 | 82.3 | 2.70 | 39.03 |
| Deadline-aware | 40 | 70.0 | 5 | 82.6 | 1.15 | 39.25 |
| Released PPO expert | 40 | 75.0 | 5 | 88.1 | 0.00 | 42.58 |
| Upstream IDM (40km/h) | 40 | 40.0 | 13 | 72.9 | 0.00 | 29.05 |
| No replanning | 40 | 70.0 | 4 | 83.3 | 0.00 | 39.67 |
| Matched no-latency | 40 | 70.0 | 5 | 82.6 | 1.15 | 39.25 |
| Tuned polling | 40 | 72.5 | 3 | 84.9 | 20.65 | 40.37 |
| Condition-tuned polling | 40 | 70.0 | 4 | 83.4 | 39.65 | 39.20 |
| Gated polling (10Hz) | 40 | 72.5 | 4 | 84.4 | 5.03 | 39.97 |
| Intervention trigger | 40 | 72.5 | 4 | 83.0 | 8.57 | 40.66 |
| TTC threshold | 40 | 70.0 | 5 | 82.6 | 1.15 | 39.25 |

Upstream reference agents are complete-controller comparisons, not isolated scheduler ablations. The custom controller uses surrounding-object geometry; the released neural expert uses its own trained observation encoding. Do not interpret these as camera-only leaderboard comparisons.

## Upstream safe: no injected delay

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 70.0 | 2 | 87.5 | 2.90 | 42.34 |
| Deadline-aware | 40 | 70.0 | 2 | 86.9 | 1.12 | 41.92 |
| Released PPO expert | 40 | 47.5 | 15 | 77.3 | 0.00 | 62.33 |
| Upstream IDM (40km/h) | 40 | 50.0 | 6 | 81.2 | 0.00 | 32.16 |
| No replanning | 40 | 37.5 | 2 | 68.5 | 0.00 | 59.95 |
| Matched no-latency | 40 | 70.0 | 2 | 86.9 | 1.12 | 41.92 |
| Tuned polling | 40 | 72.5 | 2 | 86.2 | 22.38 | 43.92 |
| Condition-tuned polling | 40 | 72.5 | 2 | 86.2 | 46.10 | 45.70 |
| Gated polling (10Hz) | 40 | 70.0 | 2 | 88.3 | 5.62 | 42.75 |
| Intervention trigger | 40 | 72.5 | 3 | 86.1 | 14.25 | 46.66 |
| TTC threshold | 40 | 70.0 | 2 | 86.9 | 1.12 | 41.92 |

Upstream reference agents are complete-controller comparisons, not isolated scheduler ablations. The custom controller uses surrounding-object geometry; the released neural expert uses its own trained observation encoding. Do not interpret these as camera-only leaderboard comparisons.

## Frozen PPO versus simple PD control

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Fixed PD (sigma-free) | 600 | 100.0 | 0 | 100.0 | 1.32 | 13.89 |
| Frozen PPO (sigma-free) | 600 | 100.0 | 0 | 100.0 | 1.32 | 14.13 |

A separate 1,200-episode paired ablation uses 10 scene seeds, four schedulers and five delays. Critic-distance inflation is zero for BOTH banks, with the same backup and admission guards. PD gains were fixed before these runs. The 10 seeds are the prospective controller-ablation subset, not another independent training population.

## Actual language-model integration (exploratory)

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Deadline + actual Qwen | 3 | 66.7 | 0 | 94.4 | 1.00 | 17.41 |
| Polling + actual Qwen | 3 | 66.7 | 0 | 95.1 | 2.00 | 17.42 |

Six physical episodes use pinned Qwen2.5-1.5B CPU inference with wall-clock-paced control. The model made 9 actual closed-loop calls and used no fallbacks. Both systems stalled in the two-trap scenario; the deadline system reached 83.1% versus 85.2% for polling. This single-seed integration check does not establish model superiority and is excluded from the controlled benchmark statistics.

## Controlled initial-offset finding

An additional 1,080 physical runs plus 360 reused phase-suite rows compare zero versus uniformly randomized initial timer offsets on exactly matched scenes, delays and 40 seed clusters. At fixed 0.1 Hz, observed success falls from 360/360 to 282/360: -21.67 percentage points (95% paired seed-cluster interval -26.67 to -16.67). No collisions occur in either group. At 1 Hz the offset effect reverses: success rises from 340/360 to 349/360 (+2.50 points, interval +0.83 to +4.72).

Thus a large apparent polling result can depend on its initial clock alignment, and the direction depends on rate. These are rolling timers reset on actual submissions, not absolute wall-clock schedules. This post-primary mechanistic control is exploratory and does not establish a new phenomenon in the literature or a universal scheduler advantage.

## Interpretation and limitations

This is a reproducible environment-based benchmark study. The report's exploratory call-win screen checks a negative paired call-difference interval, success-difference lower bound at least -2 percentage points, route-difference lower bound at least -1 point, and no higher observed collision incidence. This descriptive screen is not a preregistered formal noninferiority test, a rare-event safety guarantee or a multiplicity-corrected SOTA claim.

Bootstrap resampling clusters by scene seed jointly across scenarios, environments and delays. Safe and generalization share a geometry for the same seed. Frozen existing policies provide conditional evaluation; independent training-seed variation is not measured. Zero observed crashes or zero-width bootstrap intervals do not establish zero population risk.

Native study records call counts, not end-to-end compute savings. Road artifacts separately record process CPU, elapsed planner/scheduler sections, and complete control CPU. All planner delays are injected stress conditions, not measured LLM inference. Neither curves, uncertainty modelling nor empirical survival establish a CBF theorem.

The native held-out bulk runs did not record achieved lateral jerk; those values are null and must not be treated as zero or inferred from commanded jerk. Nine supplementary full action traces were replayed successfully, but that subset does not supply a population comfort estimate. Mean episode duration includes failures/timeouts and is not unconditional time-to-goal.

The original zero-offset comparison reduces calls by 47% versus tuned polling but loses 0.83 percentage points of observed success; its success-difference lower interval is -2.50 points, failing the report's exploratory 2-point screen. Dense traffic exposes missed spatial opportunities in the speed-based gate. A gap-aware candidate was evaluated on development data only and was not promoted to a fresh held-out claim. TTC and matched no-latency controls remain competitive; the latency term itself has no established incremental benefit.

Independent review corrected a mismatched no-latency buffer, an unintended road polling cap, and stale stop-plan recovery, including a stopped mid-transition vehicle. All pre-freeze development versions and failures are retained and excluded from final selection.

Algorithmic novelty is not established: event-triggered MPC, risk-triggered replanning and robot metareasoning already address close questions. The useful contribution is the controlled comparison, audited failure mechanisms and measured operating conditions.

## Primary sources and reproduction

MetaDrive environments: https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html ; pinned simulator source fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1.

CERT-Replan: https://arxiv.org/abs/2610.09302 ; RL event-triggered MPC: https://arxiv.org/abs/2208.10302 ; robot metareasoning: https://arxiv.org/abs/2610.23662.

Reproduce via scripts/research_isolation.py, research_roads.py, research_phase.py, research_freeze.py and research_report.py. Full commands, versioned source and compressed raw rows are included with this report.
