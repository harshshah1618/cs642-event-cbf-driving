# Final research report: Event-triggered skill routing in MetaDrive

The proposed stack averages 99.8% route completion in the final nominal suite, with 0 crashes and 0 stalls in 15 episodes. All 42 expanded robustness episodes avoid crashes and stalls; some time out short of the full route.

The system is a reproducible engineering and measurement project. The safety additions improve tested behavior, but infeasible/negative barrier states remain. The local LLM offers no established advantage over the rule router. The pending margin has no demonstrated universal collision-prevention benefit; at multi-second model delays it can reduce progress.

## Protocol

| Experiment | Episodes / rollouts | Seeds / condition |
| --- | --- | --- |
| Main comparison | 90 | 0-4; 3 scenarios; 6 modes |
| Additional style baselines | 45 | Same 5 seeds; 3 modes |
| Safety ablation | 45 | 0,1,2; 5 stages |
| Robustness | 42 | 301,307,311; 7 cases; margin on/off |
| Actual local LLM | 18 | 11,23; 3 scenarios; 3 modes |
| Delay-replayed rule control | 18 | Same scenarios/seeds/modes as LLM |
| Nominal calibration | 28 | 16 fit + 12 held-out rollouts |

## Architecture and safety investigation

The router selects a frozen lane skill. The skill emits physical accelerations and critic disagreement. The shield filters the action using fresh state, the intended maneuver corridor, a stopping-distance backup, and road-edge constraints. Accepted new maneuvers re-arm intervention monitoring; identical replies do not. Replies are validated against fresh lane gaps and deadlines.

The original clearance target can become negative when critic or pending-delay margins jump. Fresh perception does not make a speed-times-model-delay margin a safety theorem. We retain that target and its infeasibility flags and separately log the stopping envelope. Assumptions: ego and lead braking magnitude 6 m/s^2, 2 m extra clearance, 0.15 s response allowance, current footprints, and 50 Hz sampled dynamics. Tracking error, moving cut-ins and initial unsafe states can still violate this model.

| Stage (same 3 seeds) | Route % | Infeasible ticks | Two-trap infeasible |
| --- | --- | --- | --- |
| legacy | 100.0 | 97.7 | 170.3 |
| calibration_only | 100.0 | 85.4 | 161.7 |
| braking_only | 99.8 | 43.2 | 89.0 |
| calibration_braking | 99.8 | 29.8 | 72.0 |
| calibration_braking_intent | 99.7 | 36.0 | 74.3 |

## Calibration

Weights remain frozen; only metadata is recalibrated. Fit seeds: 101,103,107,109. Held-out validation: 211,223,227. Each skill is exercised in lane holding and acquisition with varied initial speed and lateral velocity. Threshold = empirical 99th percentile of nominal critic disagreement. Exceedance is not a collision probability; temporally correlated samples do not create thousands of independent trials.

| Skill | Fit / validation ticks | Old exceedance % | New exceedance % |
| --- | --- | --- | --- |
| CRUISE_RIGHT | 2400 / 1800 | 0.00 | 0.56 |
| OVERTAKE_LEFT | 2400 / 1800 | 8.22 | 0.00 |

## Main and style comparisons

| Mode | n | Route % | Crashes | Stalls | Calls |
| --- | --- | --- | --- | --- | --- |
| unshielded_skill | 15 | 37.9 | 15 | 0 | 0.0 |
| static_cbf | 15 | 79.0 | 0 | 5 | 0.0 |
| fasionad_wakeup | 15 | 79.9 | 5 | 0 | 0.3 |
| cert_replan | 15 | 79.0 | 0 | 5 | 0.0 |
| fixed_rate_shielded | 15 | 99.8 | 0 | 0 | 14.5 |
| always_on_shielded | 15 | 100.0 | 0 | 0 | 36.5 |
| proposed | 15 | 99.8 | 0 | 0 | 9.2 |
| proposed_no_margin | 15 | 100.0 | 0 | 0 | 9.7 |
| proposed_no_blend | 15 | 99.8 | 0 | 0 | 12.7 |

Route and calls are episode means. Crashes/stalls are episode counts. Stalls mean speed below 0.5 m/s for at least one second while progress remains, even if the vehicle later recovers. Named style baselines are local approximations, not reproductions of the original papers. Requested-action jerk and achieved-motion jerk are reported separately in the detailed run report.

## Robustness

| Case | n | Route % | Crashes | Stalls | Min route % |
| --- | --- | --- | --- | --- | --- |
| jitter_dense_highway | 6 | 97.1 | 0 | 0 | 92.0 |
| jitter_merge | 6 | 100.0 | 0 | 0 | 100.0 |
| jitter_two_trap | 6 | 100.0 | 0 | 0 | 100.0 |
| lead_brakes | 6 | 100.0 | 0 | 0 | 100.0 |
| moving_cut_in | 6 | 100.0 | 0 | 0 | 100.0 |
| target_lane_brakes | 6 | 100.0 | 0 | 0 | 100.0 |
| tight_cut_in | 6 | 97.4 | 0 | 0 | 91.8 |

Traffic events change speed/lane targets; Bullet computes subsequent motion. No vehicle is teleported after spawn. Tight cut-ins deliberately challenge the stopping envelope. An earlier target-lane-braking test exposed a collision, then an intent-only version avoided the collision but stalled; re-arming after an accepted new intent fixed the tested recovery failure. The failing runs are retained under runs/robustness and runs/robustness-v2.

## Actual LLM and delay-replayed rule comparison

| Mode | LLM route % | Rule route % | LLM / rule stalls | LLM / rule calls |
| --- | --- | --- | --- | --- |
| fixed_rate_shielded | 91.6 | 93.9 | 0 / 0 | 4.8 / 4.8 |
| proposed | 92.5 | 91.6 | 0 / 0 | 2.8 / 2.8 |
| proposed_no_margin | 97.6 | 98.6 | 1 / 0 | 2.7 / 3.0 |

Qwen/Qwen2.5-1.5B-Instruct, revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, BF16 on CPU. 62 actual model calls, 0 fallbacks, 29942 generated/input tokens counted by the tokenizer. Observed request delay: 2.85-7.00 s; empirical p95 6.01 s. The live test uses real background inference and wall-clock pacing; configured margin horizon is 4.3 s and pending deadline is 6 s. Requests may expire; expired replies cannot switch skills.

The matched rule control replays each episode's observed delay sequence by call ordinal, including deadline waits and backend-busy time. If the rule makes more calls, the final observed delay is reused. Different decisions change later states and call counts; this is a delay-controlled diagnostic, not an exact paired causal trial or a speed benchmark. Only two scenario seeds were used for the model comparison. In four simple diagnostic states, the model agreed with the heuristic label in two; labels are not a ground-truth safety benchmark.

## Verification and reproduction

216 tests pass; Ruff and mypy pass; checkpoint verification and the original native CI regression gate pass. 168 fully recorded episodes and 130,594 action-chain checks passed schema/consistency validation. 7 selected episodes replayed within 0.0001 m tolerance on this machine. Added CI configuration has not been executed remotely.

```bash
make check
make calibrate
make study
make robustness
make study-live
python scripts/match_live_delays.py runs/metadrive-local/RUN_ID
```

The live profile requires the local model extra (`pip install -e '.[local,metadrive]'`) and downloads the pinned public model. The configured OpenCode service was also probed but rejected the request because its free quota was exhausted; this is saved in runs/live-router-probe.json. No remote API credentials were required for the local study.

## Deliverables and limits

- [Three-layer demo](outputs/research-study/three-layer-demo.mp4): top-down reconstruction from actual logs; yaw approximated from velocity, not camera footage.
- [Audit](outputs/research-study/audit.json).
- [Nominal results](runs/metadrive-study/study-final/report.md).
- [Live results](runs/metadrive-local/live-qwen-final/report.md).
- [Calibration artifact](checkpoints/metadrive-calibrated/calibration.json).
- [Setup and scope](METADRIVE.md).

Validated scope: controlled straight roads, privileged state, two distinct transferred frozen PPO skills (RETURN_RIGHT aliases CRUISE_RIGHT), and bounded scenario traffic. No camera/VLM perception, curved-road validation, naturalistic traffic certification, real-vehicle validation, hard real-time guarantee, or end-to-end CBF proof is claimed. Zero sampled crashes are not evidence that all barriers remained feasible. This completes the proposal's systems-and-measurement objective, including negative results; it does not establish a new algorithm.

References: [MetaDrive documentation](https://metadrive-simulator.readthedocs.io/en/latest/), [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), and the local proposal and recorded run artifacts.
