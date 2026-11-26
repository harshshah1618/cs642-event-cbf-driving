# Paper Results: Frozen Final Evaluation (2026-11-26)

Every number here comes from a single frozen code version evaluated **once** on map splits that
were never used during development. Rates carry Wilson 95% intervals; mean costs carry bootstrap
95% intervals; "paired" columns are exact two-sided McNemar tests over identical maps.

## Evaluation protocol

| Split | Maps | Role |
|---|---|---|
| SafeMetaDrive 0–99 | 100 | PPO tracker training |
| SafeMetaDrive 100–149 | 50 | Development |
| SafeMetaDrive 500–549, 600–649 | 100 | **Development** (earlier test sets; inspected during the 2026-11-26 diagnosis) |
| Irregular 1000–1049 | 50 × 8 scenes | **Development** (earlier test set; inspected during diagnosis) |
| **SafeMetaDrive 2000–2199** | **200** | **Final test** |
| **Irregular 3000–3099** | **100 × 8 scenes** | **Final test** |
| highway-env seeds 100–149 | 50 × 3 envs | Development |
| **highway-env seeds 1000–1199** | **200 × 3 envs** | **Final test** |

What changed after the previous report, all on development maps only:

1. **Crossing-traffic guard (Layer 1).** LiDAR clusters are tracked frame to frame (ego motion
   compensated with speed and the observed yaw rate, velocity by least squares over 8 frames); a moving
   object whose constant-velocity path enters the ego's swept corridor within 3 s makes the car yield.
   It uses only the 259-dim observation. It fixed the base-driver crash behind 5 of the previous VLM
   arm's 6 crashes (map 1040, a roundabout entry).
2. **LiDAR backup trigger and second look (Layer 3).** A static LiDAR object 10–40 m ahead that
   PatchCore did not flag triggers one VLM call, plus one more when it is within 20 m.
3. **Verified hold release (Layer 3).** During a STOP hold the VLM is asked a scenario-neutral binary
   question (keep waiting / proceed slowly). Release needs P(proceed) ≥ 0.47 (threshold chosen on
   development maps) on two consecutive answers, and releases into careful driving.
4. **VLM lane state from the observation.** The prompt's lane count/position now comes from the
   259-dim observation instead of the simulator's lane graph.
5. **Analytic tracker is the reference Layer 2.** The PPO tracker is reported as a variant.

## Table 1. SafeMetaDrive, 200 untouched maps (2000–2199)

All methods see the same 259-float LiDAR + ego-state observation except IDM (privileged simulator
object state). "Any-cost episodes" is the fraction of episodes with any safety cost; it is the fairest
cross-method safety metric because MetaDrive charges cost on every step of a continuing contact.

| Method | n | Finished | Zero-cost finished | Any-cost episodes | Mean cost [95% CI] | Mean steps | Paired vs ours (zero-cost) |
|---|---:|---:|---:|---:|---:|---:|---|
| Ours (analytic tracker) | 200 | 97.5% [94.3, 98.9] | 97.0% [93.6, 98.6] | 3.0% [1.4, 6.4] | 0.055 [0.010, 0.125] | 560 |  |
| Ours (PPO tracker seed0) | 200 | 94.0% [89.8, 96.5] | 93.5% [89.2, 96.2] | 5.5% [3.1, 9.6] | 0.140 [0.040, 0.275] | 569 | ours-only 9 / theirs-only 2, p=0.0654 |
| Ours (PPO tracker seed1) | 200 | 91.5% [86.8, 94.6] | 91.0% [86.2, 94.2] | 5.5% [3.1, 9.6] | 0.095 [0.035, 0.180] | 580 | ours-only 13 / theirs-only 1, p=0.00183 |
| Ours (PPO tracker seed2) | 200 | 92.0% [87.4, 95.0] | 91.0% [86.2, 94.2] | 5.5% [3.1, 9.6] | 0.090 [0.030, 0.170] | 576 | ours-only 13 / theirs-only 1, p=0.00183 |
| Ablation: no crossing guard | 200 | 95.5% [91.7, 97.6] | 91.5% [86.8, 94.6] | 8.5% [5.4, 13.2] | 1.910 [0.250, 4.920] | 525 | ours-only 14 / theirs-only 3, p=0.0127 |
| Ablation: no safety filter | 200 | 96.0% [92.3, 98.0] | 83.0% [77.2, 87.6] | 17.0% [12.4, 22.8] | 1.730 [1.030, 2.550] | 501 | ours-only 30 / theirs-only 2, p=2.46e-07 |
| Ablation: no replanning | 200 | 54.0% [47.1, 60.8] | 53.5% [46.6, 60.3] | 3.5% [1.7, 7.0] | 0.050 [0.015, 0.095] | 739 | ours-only 88 / theirs-only 1, p=2.91e-25 |
| Ablation: no blending | 200 | 97.5% [94.3, 98.9] | 97.0% [93.6, 98.6] | 3.0% [1.4, 6.4] | 0.050 [0.010, 0.110] | 558 | ours-only 1 / theirs-only 1, p=1 |
| IDM (privileged object state) | 200 | 85.0% [79.4, 89.3] | 79.5% [73.4, 84.5] | 20.5% [15.5, 26.6] | 0.990 [0.365, 1.860] | 429 | ours-only 37 / theirs-only 2, p=2.84e-09 |
| MetaDrive PPO-Expert | 200 | 55.0% [48.1, 61.7] | 46.5% [39.7, 53.4] | 45.0% [38.3, 51.9] | 187.370 [141.145, 235.465] | 609 | ours-only 102 / theirs-only 1, p=2.05e-29 |
| PPO-Lag (1M, study config) seed0 | 200 | 49.0% [42.2, 55.9] | 38.5% [32.0, 45.4] | 61.5% [54.6, 68.0] | 2.140 [1.605, 2.760] | 424 | ours-only 119 / theirs-only 2, p=5.55e-33 |
| PPO-Lag (1M, study config) seed1 | 200 | 60.5% [53.6, 67.0] | 50.5% [43.6, 57.4] | 49.5% [42.6, 56.4] | 1.655 [1.235, 2.115] | 477 | ours-only 97 / theirs-only 4, p=3.36e-24 |
| PPO-Lag (1M, study config) seed2 | 200 | 48.5% [41.7, 55.4] | 35.0% [28.7, 41.8] | 65.0% [58.2, 71.3] | 2.220 [1.725, 2.770] | 387 | ours-only 126 / theirs-only 2, p=4.85e-35 |
| CPO (1M, study config) seed0 | 200 | 5.5% [3.1, 9.6] | 5.0% [2.7, 9.0] | 22.5% [17.3, 28.8] | 2.340 [0.300, 6.130] | 875 | ours-only 185 / theirs-only 1, p=3.81e-54 |
| CPO (1M, study config) seed1 | 200 | 0.5% [0.1, 2.8] | 0.5% [0.1, 2.8] | 15.5% [11.1, 21.2] | 14.515 [5.075, 26.120] | 956 | ours-only 193 / theirs-only 0, p=1.59e-58 |
| CPO (1M, study config) seed2 | 200 | 2.0% [0.8, 5.0] | 2.0% [0.8, 5.0] | 91.0% [86.2, 94.2] | 1.525 [0.920, 2.535] | 352 | ours-only 190 / theirs-only 0, p=1.27e-57 |

> **Pending:** PPO-Lag and CPO retrained for 5M steps with OmniSafe default hyperparameters (Modal run `baselines-5m-defaults-v2`) replace/extend the 1M rows above once selected on development maps. See [TEAMMATE_HANDOFF.md](../TEAMMATE_HANDOFF.md).

**Reading.** The reference system finishes 97.5% of routes with zero safety cost on 97.0%, against
79.5% for IDM even though IDM reads simulator object state (paired p = 3e-9). Each Layer 1 component
is necessary: removing the safety filter raises any-cost episodes from 3.0% to 17.0% (p = 2e-7);
removing the crossing guard raises them to 8.5% (p = 0.013); removing replanning drops finishes to
54.0% (p = 3e-25). Two honest negatives: steering blending has no measurable effect (1 map each way),
and the learned PPO tracker is significantly worse than the analytic tracker (p = 0.002 for 2 of 3
seeds), so the paper's reference Layer 2 is analytic.

## Figure 1 and Table 2. Slow-planner delay vs turning-room reserve (same 200 maps)

![Delay vs reserve](final/delay-reserve.png)

| Delay | Finished, reserve 6.5 m | Finished, no reserve | Paired (reserve-only / no-reserve-only), p |
|---:|---:|---:|---|
| 0 s | 97.5% | 97.5% | 0 / 0, p=1 |
| 1 s | 98.0% | 97.0% | 3 / 1, p=0.62 |
| 2 s | 95.5% | 93.5% | 9 / 5, p=0.42 |
| 4 s | 96.0% | 89.0% | 19 / 5, p=0.0066 |
| 6 s | 96.5% | 67.5% | 60 / 2, p=8.5e-16 |
| 8 s | 95.0% | 67.0% | 57 / 1, p=4.1e-16 |

**Reading.** Up to 2 s the reserve makes no significant difference. From 4 s on it does, and at 6–8 s
it is the difference between finishing 95–97% and 67% of routes. This replaces the earlier
hand-built "tipping point" scenes with random untouched maps.

## Table 3. Irregular visual scenes, 100 untouched maps × 8 scenes (3000–3099)

> **Pending: final run not yet executed.** Development pilots on maps 1000–1049 (4 scenes × 50 maps):
>
> | | Previous VLM | Pilot v3 | Pilot v3b (frozen candidate) |
> |---|---:|---:|---:|
> | Passed flagger while showing STOP | 25/50 | 11/50 | 12/49 |
> | Stuck at the flagger until the step limit | 0/50 | 39/50 | 8/49 |
> | Billboard / shoulder-pedestrian decoys | – | no needless stops | no needless stops, no crashes |
>
> No-VLM arms on the same development maps all pass the STOP 50/50 (normal, always-careful,
> LiDAR→careful) or freeze at every object (LiDAR→stop, route 14.8% vs 31.0%). A ground-truth oracle
> gets 0/50 with normal efficiency. See [TEAMMATE_HANDOFF.md](../TEAMMATE_HANDOFF.md) for the commands
> that produce this table on maps 3000–3099.

## Table 4. highway-env (separate simulator), 200 fresh seeds × 3 environments

Same observation (Kinematics, 10 vehicles) for every method; 10 Hz control; 1.5 s slow-planner delay.
The relative observation is rotated into the ego heading frame (it is given in world axes, which made
the roundabout lateral test meaningless in the previous report). The crossing guard is the same
constant-velocity corridor test as in SafeMetaDrive, applied to the given relative velocities.

| Method | highway-v0 | merge-v1 | roundabout-v1 | Overall crash-free | Calls/ep | Paired vs ours |
|---|---:|---:|---:|---:|---:|---|
| Ours (event + filter + guard + reserve) | 80.5% [74.5, 85.4] | 100.0% [98.1, 100.0] | 67.5% [60.7, 73.6] | **82.7% [79.4, 85.5]** | 3.7 | – |
| Event + filter + guard, no reserve | 79.0% [72.8, 84.1] | 100.0% [98.1, 100.0] | 68.0% [61.2, 74.1] | 82.3% [79.1, 85.2] | 3.8 | 19 / 17, p=0.87 |
| Ours without crossing guard | 79.5% [73.4, 84.5] | 100.0% [98.1, 100.0] | 50.5% [43.6, 57.4] | 76.7% [73.1, 79.9] | 2.9 | 53 / 17, p=2e-5 |
| Always-ask 10 Hz **with** the same filter | 79.5% [73.4, 84.5] | 100.0% [98.1, 100.0] | 63.0% [56.1, 69.4] | 80.8% [77.5, 83.8] | 210.7 | 43 / 32, p=0.25 |
| Static safety filter (no replanning) | 74.0% [67.5, 79.6] | 83.0% [77.2, 87.6] | 71.0% [64.4, 76.8] | 76.0% [72.4, 79.2] | 0.0 | 82 / 42, p=4e-4 |
| Unshielded fast driver | 10.0% [6.6, 14.9] | 99.0% [96.4, 99.7] | 58.0% [51.1, 64.6] | 55.7% [51.7, 59.6] | 0.0 | 193 / 31, p=9e-30 |
| Always-ask 10 Hz, no filter | 4.5% [2.4, 8.3] | 39.5% [33.0, 46.4] | 57.5% [50.6, 64.1] | 33.8% [30.2, 37.7] | 129.4 | 321 / 28, p=3e-64 |

**Reading.** The architecture transfers: the same filter + guard lifts crash-free driving from 55.7%
(unshielded) and 76.0% (static filter) to 82.7%. The event trigger's benefit is **cost, not safety**:
a fair always-ask baseline behind the same filter is statistically as safe (p = 0.25) but makes 57×
more slow-planner calls. The turning-room reserve has no measurable effect at 1.5 s delay here,
consistent with Figure 1 (no effect below 4 s). The crossing guard matters on the roundabout
(50.5% → 67.5%, p = 2e-5).

## Table 5. Real-world replays (Waymo + nuScenes, 13 scenes): transfer check only

| Method | Crash-free (13 scenes) |
|---|---:|
| LiDAR safety filter + tracker | 76.9% [49.7, 91.8] |
| Unshielded tracker | 61.5% [35.5, 82.3] |
| Constant throttle | 30.8% [12.7, 57.6] |

With 13 scenes the intervals overlap; this is a qualitative transfer check, not a claim.
(The replay driver predates the crossing guard.)

## Limitations

- **Simulated perception.** The VLM sees MetaDrive renders; props and weather are synthetic.
- **Release threshold.** The verified-release threshold was chosen on development maps; its
  operating point (P(keep) ≈ 0.5–0.55) is narrow, so it may need recalibration for another VLM.
- **Remaining off-road exits.** About 1–2% of SafeMetaDrive maps still end off-road at tight
  roundabout entries from the outer lane (e.g. development map 501).
- **Baseline budgets.** Learned baselines are trained for a fixed budget; see Table 1 notes.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python tools/final_eval.py --first 2000 --count 200 --baselines runs/final/baselines.json --out runs/final/safemetadrive-2000-2199.json
PYTHONPATH=. .venv/bin/python tools/plot_final_delay.py runs/final/safemetadrive-2000-2199.json reports/final/delay-reserve.png
.venv/bin/python tools/benchmark_highway_env.py --seeds 1000 1200 --out runs/final/highway-env-1000-1199.json
PYTHONPATH=. .venv/bin/python tools/irregular_controls.py --seeds 3000 3100 --arms normal careful oracle lidar-careful lidar-stop --out runs/final/irregular-arms-3000-3099.json
.venv/bin/modal run --detach tools/irregular_modal.py --stage final-v3
.venv/bin/python tools/irregular_final_report.py --arm vlm=runs/final/irregular-vlm --arm normal,careful,oracle,lidar-careful,lidar-stop=runs/final/irregular-arms-3000-3099.json
```
