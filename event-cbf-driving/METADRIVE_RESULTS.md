# MetaDrive release validation

Historical release-v3 evidence (203 tests). The expanded, current study is
reported in [FINAL_REPORT.md](FINAL_REPORT.md), including calibrated metadata,
additional safety guards, physical traffic events and actual local inference.
The measurements below are retained for comparison with that later study.

The maintained stack now runs against physical MetaDrive/Bullet dynamics. The router selects frozen PPO skills, the safety layer filters their actions and can request replanning, and an acceleration tracker drives the simulated vehicle. Fresh target-lane checks and lateral road-edge constraints are enabled in the physical profile.

## Validation performed

- 203 tests passed, including actual physics, braking/lateral tracking, collision detection, all three layers interacting, background inference, expired answers, post-episode token accounting, and replay.
- Ruff passed; mypy passed on 48 source files. The checkpoint validator and native CI regression gate passed.
- Main comparison: 90 episodes = 3 scenarios × 5 randomized seeds × 6 modes. Controlled router delay: 0.30 s.
- Held-out delay stress: 18 episodes = 3 scenarios × 3 seeds (11, 23, 47) × margin on/off. Controlled router delay: 0.80 s.
- All 108 episode schemas passed validation. All 81,844 recorded final actions matched both the layer trace and the acceleration request delivered to the physical adapter.
- Six selected open-loop replays (one per scenario from each suite) reproduced the recorded final state with zero error on this machine.
- MetaDrive upstream commit: `fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`; Python 3.12.13; CPU execution. The manifests record actual dependency versions and the upstream source commit.
- A dedicated MetaDrive CI job was added. These are local validation results; remote CI has not been run by this task.

## Main comparison

All proposed episodes completed the route without a crash, freeze, off-road termination or safety abort. The polling modes below use the same skills, shield, blending and maneuver-admission guard. Only the routing schedule differs.

| Scenario | Proposed route | Crash / freeze | Proposed calls | 1 Hz calls | High-rate calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense_highway` | 100.0% | 0% / 0% | 2.8 | 17.0 | 45.8 |
| `merge` | 100.0% | 0% / 0% | 3.2 | 12.0 | 35.8 |
| `two_trap` | 100.0% | 0% / 0% | 4.2 | 13.2 | 40.4 |

Across all scenarios, the proposed stack averages 3.4 router calls per episode versus 14.1 for timer polling and 40.7 for high-rate polling, at the same measured route completion.

The static CBF completes the moving-traffic scenarios but freezes in every stopped-car scenario (37.7% mean route). Skill changes provide progress in that case.

Blending reduces the mean of episode peak **requested** lateral jerk from 85.5 to 9.3 m/s³. Achieved-motion jerk is reported separately in the full report; it is not interchangeable with requested-action jerk. The proposed stack does not beat shielded polling on every motion/safety metric.

## Held-out 800 ms delay

Both margin variants completed all 18 episodes without crashes or freezes. This test does not establish a collision-prevention benefit from the pending margin. Its effect on calls differs by scenario:

| Scenario | Calls with margin | Calls without margin |
| --- | ---: | ---: |
| `dense_highway` | 4.0 | 16.3 |
| `merge` | 3.7 | 3.3 |
| `two_trap` | 5.3 | 5.0 |

## Limits and evidence locations

All comparisons use the scripted router. Background real-inference behavior is tested with a deliberately delayed backend; these runs do not establish live LLM decision quality. The policies are transferred frozen checkpoints, and `RETURN_RIGHT` remains an explicit alias of `CRUISE_RIGHT`.

This release supports controlled straight-road scenarios with privileged state. The merge case is lane entry from an adjacent lane. Curved roads and camera-based perception are outside the validated scope.

Zero crashes do not mean the barrier stayed feasible. The proposed stack still records negative barrier values and infeasible ticks, especially in two-trap (about 170 infeasible ticks per episode and minimum longitudinal barrier around -14.9 m on average). The monitor reports degraded operation; this is an empirical research package without a formal safety guarantee.

- [Setup, architecture, commands and scope](METADRIVE.md)
- [Full comparison report](runs/metadrive/metadrive-release-v3/report.md)
- [Held-out delay report](runs/metadrive/metadrive-delay-stress-v3/report.md)
- [Three-layer interaction plot](runs/metadrive/metadrive-release-v3/figures/two_trap-interactions.png)
- [Full comparison audit](runs/metadrive/metadrive-release-v3/audit.json)
- [Held-out audit](runs/metadrive/metadrive-delay-stress-v3/audit.json)
- [Final test output](runs/release-tests-final.log)
- [Lint, typecheck, checkpoint and native CI checks](runs/release-checks.log)

Run artifacts are generated and git-ignored. Reproduce them with `make metadrive-eval`; the held-out command is:

```bash
.venv/bin/python -m event_cbf_driving.cli eval --profile metadrive \
  --modes proposed proposed_no_margin --seeds 11 23 47 \
  --override router.latency_s=0.8 --override shield.vlm_delay_s=0.8 \
  --no-figures --run-id metadrive-delay-stress
```
