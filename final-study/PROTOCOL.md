# Final proposal-aligned integrated comparison

Frozen before held-out runs on November 12, 2026. Parent source:6090300. Original proposal is included unchanged as proposal.pdf. Existing user checkout and previous studies remain separate.

## Question and decision rules

Can one integrated controller reduce crashes, terminal stalls and costly planning requests without sacrificing route progress or worsening achieved lateral jerk? The revised final system combines the existing frozen PPO skill bank, event scheduling based on current opportunity and an estimated decision delay, the existing CBF/braking/admission protection package, an8m optional stopping reserve released on an admitted target change, and24-tick blending. Critic/delay inflation remains an explicitly tested alternative, not an assumed benefit.

The proposal's original implementation is retained as its own arm: critic-or-sustained-override interruption, gamma_sigma2.8, speed-times-delay inflation during pending calls, blending, and no optional8m reserve. It receives the same additional braking/admission safeguards as protected comparisons. This is an evolved project comparison, not a claim that the initial margin hypothesis was established.

Predeclared observed pass checks, always with denominators and paired scene/seed/delay outcomes:

1. Final has fewer collision episodes than unprotected fixed-skill RL and unshielded1Hz hard-switch routing.
2. Final has fewer terminal stalls with remaining route than static CBF without maneuver changes. Legacy ever-stopped/freeze and total stopped time are reported separately.
3. Final has fewer costly requests than10Hz polling with at least as many successes (no crash and≥99% route). If success drops, call reduction is reported as a tradeoff. No population noninferiority claim follows from equality of observed counts.
4. Blending is compared with the otherwise identical no-blend arm. Report achieved lateral jerk RMS and peak; both must improve or tie to claim unqualified smoothing. Show all episodes and the jointly successful subset with sample counts/exposure. No exclusion of failed episodes from primary tables.
5. Pending speed-times-delay inflation is compared with the otherwise identical final arm without that term. Record crashes during pending calls and progress; zero events in both arms does not establish the term's necessity or a safety theorem. The attribution of a collision to inference delay is not proved merely by its timestamp.

All negative and mixed outcomes remain in the final report. No tuning after held-out outcomes.

## Policies

All arms use the same DrivingStack implementation, PPO checkpoints and50Hz physical acceleration adapter. RETURN_RIGHT aliases the cruise-right checkpoint. Numerical validity monitoring is common. The unshielded arms disable the entire shield, including its boundary correction; static CBF includes boundary correction but no braking backup. The stronger matched scheduling controls share the complete final protection and reserve package.

| ID | Scheduling and decisions | Active traffic protection | Reserve and transition |
|---|---|---|---|
| unshielded_rl | Fixed cruise skill, no router | None | None |
| static_cbf | Fixed cruise skill, no router | Longitudinal CBF plus boundary | Static clearance, no maneuver |
| timer_hard_unshielded |1Hz, same controlled planner | None | Hard switch |
| uncertainty_wakeup | Critic threshold only, same discrete planner | Full common protected package |8m admitted-release reserve, blend |
| barrier_replan_local | Sustained shield override, cheap local rule, zero artificial model delay | Full package |8m admitted-release reserve, blend |
| original_proposal | Critic OR sustained override, original uncertainty/pending inflation | Full package | No8m reserve, blend |
| final_integrated | Opportunity deadline based on measured-state geometry and estimated call delay | Full package, no optional critic/delay inflation |8m admitted-release reserve, blend |
| matched_timer_hard |1Hz, same planner | Same as final | Same8m reserve, hard switch |
| high_rate_poll |10Hz, same planner | Same as final | Same8m reserve, blend |
| gated_poll |10Hz, shared usefulness gate, same planner | Same as final | Same8m reserve, blend |
| immediate_benefit | Same opportunity gate, query immediately when useful | Same as final | Same8m reserve, blend |
| no_reserve | Same deadline | Same as final | No optional reserve, blend |
| no_blend | Same deadline | Same as final | Same8m reserve, hard switch |
| speed_delay_margin | Same deadline | Final plus original pending speed-times-delay term | Same8m reserve, blend |
| critic_margin | Same deadline | Final plus gamma_sigma2.8 critic margin | Same8m reserve, blend |

Uncertainty-only wake-up and barrier-triggered local replanning are implementations inspired by the proposal's FASIONAD/CERT comparisons. They are not reproductions: no FASIONAD visual feedback fusion, CERT conformal calibration, horizon CVaR or MPC planning-mode search. Local replans are counted separately from expensive model-like requests, and receive zero artificial inference delay. Do not call a cheap local rule a VLM, or treat every request count as the same computational cost.

## Development and frozen selection

The8m reserve and deadline settings are inherited from earlier development; no new reserve search. A separate36-episode development suite crosses three scene families, two disjoint seeds8101/8102, delays.125/18s and critic-threshold scales.2/.5/1. The development-only ranking applied before held-out execution is fewer crashes, more successes, greater mean route, then fewer calls. All scales achieved12/12 success and zero crashes; scale1 made8calls, compared with15and29. Select scale1 for the uncertainty-only baseline. Other final parameters are fixed, not jointly optimized. Selection is in selection.json.

An initial incomplete development run rejected unsupported dense-scene parameters. Its partial rows, error log and executed sources are retained under tune-v0-rejected and are excluded. A later instrumentation-only addition records the effective runtime safety mask; executed tuning sources are archived before that addition. The90-episode pilot checks all fifteen methods on three scenes, seed8201 and delays.125/18s. It is excluded from held-out results and used to detect contract failures, not tune winners.

## Controlled held-out matrix

15methods×3scene families×8seed clusters9001–9008×4delays(0,.125,1,18s)=1440episodes,96per method. Report the0–1s strata separately:72episodes per method. The18s CPU stress stratum is24episodes per method and cannot inflate the nominal headline. No latent model is evaluated in this matrix.

Each scene/seed world is identical across methods and delays; initial ego/traffic states are recorded and matched. Scene/seed groups vary physical geometry and traffic, not only rendering colors. Three scene families: parked lead with no second event; parked lead then a moving left-lane cut-in afterx92m; randomized dense two-lane traffic. Ego speed/gap in the first two are sampled18–24m/s and58–85m; dense uses its own existing fixed20m/s ego start and randomized traffic. Route200m, horizon60s for all arms. Record whether a cut-in actually occurred and whether it was first observed while a call was pending; no assertion that every trajectory reaches the second trap.

The same privileged-state lane/occupancy rule supplies controlled decisions, using physical lane rather than prior-skill hint. It extends the teammate's all-parked prompt semantics by treating moving leads below15m/s as blockers. These are not actual trained-model outputs or a model-accuracy estimate. Every synchronous answer, including same-skill answers, occupies its specified simulated delay; pending_margin_on_all_calls=True. All polling uses a seed-fixed uniform fraction of its period; the actual initial polling offsets are recorded. The barrier local rule alone has zero artificial compute delay.

Source/config/checkpoint hashes and every job are locked before first execution. Full compressed physical traces and their hashes accompany each row. Metrics use actual achieved acceleration, not requested acceleration. Frozen run directories reject changed code/design. Completed rows are never silently discarded or overwritten.

## Actual camera-model subset

The exported teammate BF16 adapter, same cached base revision, v1 prompt and strict A/B/C inference are used without fine-tuning or fallback replies. Verify all504trained tensors. No scripted substitution. Model loading is bounded locally to10GiB/no swap and3CPU threads. The7.51GBbase cache is external; no paid/cloud jobs.

Queried arms:matched1Hz hard switch, tuned uncertainty-only wake-up, final integrated. No-model arms:fixed RL, static CBF, barrier-triggered local rule. Cross three scenes and two new seeds9101/9102:36physical episodes,18configured for actual model inference (zero requests are possible and reported). Rotate queried policy order so each appears first/second/third twice. A single model stays loaded; all predictions, raw frames, latencies, guard rejections and fallbacks are retained. All queried arms use real asynchronous calls and paced simulation; actual measured call plus capture time determines response readiness. A conservative30s scheduling estimate and45s pending deadline are fixed before execution. The estimate is not an imposed artificial response delay.

This is a six-world integration subset, not enough for a general VLM performance or safety claim. CPU timing is endogenous and differs between arms; sources, clocks, request order and full wall/simulation times are recorded. The original prompt assumes parked traffic and literal20m/s even when these scenes differ; report this limitation. The full fifteen-way controlled comparison and six-way actual subset must be presented separately.

## Analysis and reporting

Pair by scene/seed/delay. Aggregate over complete matched cells. Bootstrap eight seed clusters jointly across scene/delay conditions for descriptive paired intervals; do not treat1440episodes as independent scene samples or interpret nonsignificance as equivalence. Report raw counts and all per-scene/per-delay tables. Repeat no tuning or selection on held-out outcomes.

Report success, actual route-complete end reason, collision, crash time, terminal stall, legacy freeze, route fraction, stopped duration, costly requests, cheap local replans, rejected/expired requests, pending survival, achieved lateral jerk, intervention fractions and infeasible ticks. Comfort comparisons retain crash/timeout exposure and jointly successful denominators. Existing shields and heuristic reserves do not establish controlled invariance or guaranteed feasibility. Comparison to published methods' official benchmarks or SOTA is outside this protocol.

Runtime note: the controlled four-worker simulation pool ran concurrently during the initial actual-model cases. CPU inference and rendering times therefore include contention; policy order is balanced and every measured duration is retained. These are not isolated model-throughput measurements.
