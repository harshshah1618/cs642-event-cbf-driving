# SafeMetaDrive development record

These are development results, not final held-out benchmark results. The final
50 maps (seeds 500–549) have not been evaluated at this point.

## Protocol and observation privileges

The pinned environment, map partitions, metrics and comparisons are in
[safemetadrive-protocol.md](safemetadrive-protocol.md). Every benchmark episode
uses upstream reward, cost and termination without modification. Training the
tracking skill uses a different tracking reward and traffic-free training maps;
this is disclosed separately from benchmark evaluation.

The fast controller uses navigation geometry and simulator object records from
the upstream 50 m lidar broad phase. Those records include exact positions and
velocities and do not constitute a demonstrated camera-only perception system.
The camera planner receives RGB, ego speed and lane/navigation state, without
object records. A sampled footprint forecast and stopping cap provide heuristic
protection; there is no formal CBF safety certificate for this general-road code.

## Initial 50-map comparison

All variants below used development seeds 100–149. These measurements preceded
the corrected road-continuity and crossing-traffic forecasts.

| Driver | Destination success | Collision-free success | Mean native cost |
|---|---:|---:|---:|
| Upstream IDM | 84% | 76% | 1.14 |
| Initial lane/caution rule, 25 km/h | 98% | 88% | 3.72 |
| Initial lane/caution rule, 30 km/h | 94% | 84% | 1.02 |
| Initial lane/caution rule, 35 km/h | 90% | 78% | 1.22 |
| Initial rule, 30 km/h, larger following room | 92% | 84% | 0.98 |
| Initial rule, 30 km/h, lower lateral acceleration | 90% | 78% | 1.34 |

Failure traces exposed unintended lane changes at road seams and collisions
with traffic crossing a roundabout entrance. Lane continuity, route preview,
and constant-velocity footprint forecasting were added. A positive-lateral-axis
error in the forecast was caught by a permanent-stall case and corrected.

The corrected 30 km/h rule completed **50/50 development maps with zero cost**.
This is a selected development result, not evidence of universal safety.

## Learned tracker check

A shared tracker is conditioned on the target lane and speed. It first learns
from 24,000 tracking demonstrations and then trains with PPO. Each seed requests
976,000 PPO transitions; SB3 rollout granularity produces 976,896 transitions,
for 1,000,896 total demonstration-plus-PPO transitions per seed.

Using the same rule planner and protection on the 50 development maps:

| Tracker | Seed 0 success / cost | Seed 1 success / cost | Seed 2 success / cost |
|---|---:|---:|---:|
| Cloned initialization | 100% / 0 | 100% / 0 | 100% / 0 |
| After 200,000 PPO transitions | 98% / 0.02 | 100% / 0 | 100% / 0 |

All three final PPO checkpoints subsequently completed all 50 development maps
with zero cost. The final PPO checkpoints are selected for the learned stack;
cloning remains an ablation. PPO improvement over cloning is not established
by these results.

## Camera data and training

Collection produced 9,621 examples across the 100 training maps. The teacher
reached all 100 destinations with total native cost 11 during collection. All
episodes, including contact costs, remain in the data manifests.

Labels: A (continue/normal) 5,837; B (left/normal) 25; C (right/normal) 30;
D (continue/careful) 3,687; E (left/careful) 25; F (right/careful) 17.
The rare lane-change labels make per-class recall and closed-loop evaluation
essential. Overall classification accuracy alone would be misleading.

The camera training split uses map seeds 0–79; image validation uses 80–99.
The planner sees two forward/rear mosaics, normally separated by 1.5 seconds,
to provide evidence of traffic motion. It predicts one of six constrained
tokens using Qwen2.5-VL-3B with language LoRA and a frozen vision encoder.
Training examples retain simulator-generated teacher labels; labels can depend
on information absent from the images, so imitation targets are imperfect.

The original image-validation through optimizer step200 reached 89–92% overall
accuracy but only 30–31% balanced accuracy, with zero recall for all four
lane-change classes. A closed-loop pilot on maps100–104 completed 3/5, 2/5,
and 2/5 drives for the three training seeds, respectively, with zero cost.
This did not establish a useful camera planner. A continuation with equal
sampling mass per class is running; its full driving evaluation is pending.

## Reference actor evaluation

The native checkpoint loader was checked against OmniSafe's own
GaussianLearningActor at every visited state in a six-map audit. Deterministic
actions matched exactly. The learned Gaussian policies have substantial action
variance, and their mean actions perform much worse than stochastic actions.
Both modes are therefore retained and compared on development maps before a
common inference mode is selected per algorithm for final testing.

On all50 development maps, completed PPO-Lag seed0 scored 4% success / cost1.54
with mean actions and 42% / cost1.20 with stochastic actions. Seed2 scored
2% / cost2.10 and 62% / cost2.48, respectively. These are two completed models,
not the final three-seed comparison. Each episode uses a recorded policy RNG
seed independent of the environment RNG. The remaining baseline models are
still training or awaiting evaluation.

## Cloud accounting and interruptions

Modal interrupted baseline workers during training. After the laptop disconnected,
the original locally coordinated mapped jobs also stopped; detaching that
launcher did not provide the durability originally assumed. PPO-Lag seeds0/2
completed; the other four declared baseline models are being retrained from
scratch on non-preemptible CPU workers. Their CPU/memory rates are three times
standard rates, with smaller reservations. Partial baseline checkpoints did
not contain optimizer state and are not treated as faithful resumptions.
Completed baselines are selected by declared run/seed, not by test results.
Wasted retry compute remains part of the compute accounting.

The replacement launcher uses a deployed remote coordinator that owns all
mapped work and saves completion/failure status. A disconnect probe verified
completion after the launching process exited. It can also wait for another
remote run and launch its evaluation without a local parent.

The first camera check exposed slow per-image cloud-volume reads. Dataset shards
were packed into archives and copied to each trainer's local disk. Camera
training saves adapter, optimizer and random-generator states periodically.
New checkpoints are published as immutable directories before updating an
atomic pointer, preserving the previous complete checkpoint on interruption.
Resumed weighted sampling uses fresh IID draws; bit-exact batch replay is not
claimed. Environment state is not relevant to this supervised checkpoint.

At the first cost check, Modal reported $2.12 for these experiments and $0 billed
after credits for the account. This is a delayed billing snapshot, not the final
cost. On Nov19, account-wide month-to-date gross usage reached $12.25, with $0
billed after credits. This account total is not project-only spend. The project
billing report showed $7.29 through its latest completed hourly bucket; current
partial-hour usage is excluded. The provisional estimate for the current
training/evaluation round was $20–$35. Final charges remain to be checked.
