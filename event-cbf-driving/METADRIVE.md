# MetaDrive integration

This backend runs the three-layer stack against MetaDrive/Bullet vehicle
dynamics. The native scenarios remain fast regression fixtures. The router,
frozen PPO bank, barrier filter, monitor, metrics, and replay path are shared.

See [FINAL_REPORT.md](FINAL_REPORT.md) for the completed study and
[METADRIVE_RESULTS.md](METADRIVE_RESULTS.md) for the earlier release-v3 evidence.

## Install and run

From `event-cbf-driving/`, using Python 3.11 or 3.12:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,metadrive]"
.venv/bin/python -m metadrive.pull_asset
.venv/bin/python -m event_cbf_driving.cli doctor --profile metadrive
make metadrive-check
make metadrive-eval
```

For the completed study rather than the original transfer profile:

```bash
make calibrate       # optional: regenerate the supplied calibrated metadata
make study           # 90 nominal comparisons, scripted decisions
make robustness      # 42 physical-event / variable-delay cases
python scripts/run_safety_ablation.py --help
```

The `metadrive` profile preserves the original physical transfer experiment.
`metadrive-study` enables the additions below; native defaults are unchanged.

## Calibration and safety additions

The supplied `checkpoints/metadrive-calibrated/` bank preserves the original
policy weights exactly. Sixteen nominal fit and twelve held-out validation
rollouts exercise lane holding/acquisition with varied initial speed and
lateral velocity. Fit seeds 101,103,107,109 and validation seeds 211,223,227
are disjoint. Metadata uses the empirical 99th percentile of nominal critic
disagreement. This calibrates nominal exposure, not collision probability;
correlated simulation ticks are not independent validation trials.

The study also enables a separate sampled stopping envelope, assuming 6 m/s²
braking for both vehicles, 2 m clearance and 0.15 s response allowance. It
checks full footprints and caps acceleration using a one-step prediction.
Original barrier values and infeasibility remain visible alongside
`braking_h` / `braking_feasible`; a physical envelope is never used to relabel
an inflated-barrier violation as safe. Tracking error, a sudden lane entry or
invalid braking assumptions can still defeat this approximation.

The intended corridor spans the current and requested lane. A newly accepted
different skill re-arms intervention monitoring so a blocker appearing in
that corridor can request another maneuver. Same-skill replies do not re-arm
it. This fixed a reproduced target-lane braking collision/stall, protected by
a physical regression test. Traffic-event stress cases change physical
vehicle controller targets, rather than teleporting vehicles.

The optional extra pins upstream commit
`fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1` (MetaDrive 0.4.3).
Physics runs headlessly; a GPU is not required for this state-based path.
Upstream assets require a network download on installation/first use.

For one scenario, preserving its simulator and randomized parameters:

```bash
.venv/bin/python -m event_cbf_driving.cli eval --profile metadrive \
  --scenarios two_trap --modes proposed --seeds 0 --run-id one-physical-run
```

For an onscreen run, supply a small YAML config with `params: {render: true}`
on the selected MetaDrive scenario. This requires a working graphics display.
Headless evaluation is the supported automation path.

## Interaction between the three layers

1. Each physics tick produces a fresh `WorldSnapshot` of the ego and all
   scenario traffic. The skill sees `[y, vy, vx]`; the router and shield see
   surrounding vehicles as well.
2. The selected PPO skill proposes physical `[ax, ay]`. Its dual-critic
   disagreement supplies `sigma`.
3. The shield uses traffic geometry, `sigma`, and pending-call status to
   restrict longitudinal acceleration. The physical profile also applies
   lateral road-edge constraints using vehicle footprint and heading. Its intervention can request a new
   maneuver through the router. Hysteresis and a rate limit prevent chatter.
4. While a decision is pending, the current skill and shield keep running on
   current state. A configurable pending margin is active through the wait.
5. A reply is checked for request identity, known skill, timing and deadline.
   The MetaDrive profile additionally checks **fresh target-lane gaps**,
   including rear traffic, before allowing a maneuver change. Rejected
   switches are logged and can be retried; they never bypass the shield.
6. Accepted changes crossfade old/new policy actions. The shield and final
   safety monitor check the resulting action before it reaches the plant.
7. `AccelerationTracker` converts acceleration intent to normalized steering
   and throttle/brake commands. Motion and collisions are computed by Bullet;
   neither the ego nor traffic is teleported during an episode. Initial
   positions/velocities and the explicit cut-in spawn are scenario events.

The adapter uses one 0.02 s physics step per control tick. Vehicle dimensions,
mass, wheelbase, and engine/brake parameters are read from the physical vehicle.
The mapping is approximate: requested and achieved accelerations are logged
separately, along with actuator saturation and tracking error.

## Recorded evidence

Each run contains `manifest.json`, `results.json`, `report.md`, figures, and
replayable episodes. The MetaDrive profile enables per-tick `interactions`:

- ego and obstacle state;
- active skill, sigma, blend weight, requested acceleration;
- shield intervention, barrier value, inflation and monitor status;
- lateral road-edge clearance, intervention and shield reason;
- router trigger, pending status and request id;
- final acceleration request, steering/throttle command, achieved
  acceleration, and actuator saturation.

Router call records separately document decisions, response times, switches,
rejections and live-model/token telemetry. `peak_lateral_jerk` is the jerk of
the **acceleration request**. `plant_peak_lateral_jerk` measures achieved motion;
`acceleration_tracking_rmse` and `actuator_saturation_fraction` quantify tracking.
These quantities must not be substituted for each other in the report.

Replay an episode with the CLI, for example:

```bash
.venv/bin/python -m event_cbf_driving.cli replay \
  runs/metadrive/RUN_ID/episodes/two_trap-proposed-seed0.json --tolerance 0.0001
```

Use the actual episode filename printed/listed in your run directory. Exact
floating-point reproducibility across different platforms is not promised.

## Live inference without blocking control

`metadrive.yaml` is entirely offline and uses the scripted router. To make
explicit live calls with credentials configured in the environment:

```bash
.venv/bin/python -m event_cbf_driving.cli router-check \
  --profile metadrive-live --require-live
.venv/bin/python -m event_cbf_driving.cli eval \
  --profile metadrive-live --require-live
```

The live profile uses a single background worker and wall-clock pacing. The
controller polls completion without waiting, measures actual inference/retry
time, and rejects expired answers. It never queues unbounded work or lets an
old worker race a later episode's backend reset. At shutdown it joins the
backend's bounded request before cleanup. Python scheduling is not a hard
real-time guarantee. Tests exercise slow, late and rejected responses without
credentials; scripted evaluations are not evidence of live model quality.

The completed local-model pilot needs no API credentials:

```bash
python -m pip install -e ".[local,metadrive]"
make study-live
python scripts/match_live_delays.py runs/metadrive-local/LIVE_RUN_ID
```

`metadrive-local` pins Qwen/Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16 CPU inference, no fallback,
a 6 s pending deadline and 4.3 s margin horizon. It downloads about 3 GB of
weights on first use. The final 18-episode pilot records 62 actual calls,
zero fallbacks, 29,942 tokens and 2.85–7.00 s observed latency. It establishes
no model advantage over rule routing; one no-margin model episode stalls.
The margin can reduce progress. A delay-replayed rule control preserves
observed call delays, deadlines and backend-busy time, but decisions change
later states and call counts; excess calls reuse the final delay. This is a
diagnostic comparison, not an exact paired causal trial.

## Report and demonstration

Install `.[report]` and system `ffmpeg` to regenerate the PDF and video.
The report builder reads the final saved runs named in its source, including
the safety ablation, matched-delay controls and `outputs/research-study/audit.json`.
It does not fabricate missing experiments or run inference automatically.

```bash
python scripts/render_demo.py \
  runs/metadrive-study/study-final/episodes/two_trap-proposed-seed0.json \
  --output outputs/research-study/three-layer-demo.mp4
python scripts/build_research_report.py
```

`make study-audit` regenerates the 168-episode schema/action-chain audit and
seven physical replays from the final saved run paths. `make study-report`
rebuilds the report; `PDF_PYTHON=/path/to/python` selects a ReportLab runtime.

The demo reconstructs recorded MetaDrive state and all three layer logs in a
top-down view; vehicle yaw is approximated from velocity. It is not camera
footage. Generated runs and outputs are ignored by Git and remain available
locally. Preserve both folders when archiving the measured study.

## Comparisons and limits

`fixed_rate_shielded` and `always_on_shielded` retain the same uncertainty
inflation, pending margin, road-edge guard, frozen skills, blending and switch guard as
`proposed`; their routing schedule changes. The older `fixed_rate_vlm` and
`always_on_vlm` modes remain historical, unshielded/hard-switch comparisons.
The native default profile retains its original nine modes.

The initial physical scope is straight roads with 4 m lanes. The merge case
adds a third adjacent lane for the merging vehicle; this is a controlled
lane-entry scenario, not a geometric highway on-ramp. Initial traffic and
seeded parameters come from the native generators. Traffic follows physical
lane/speed controllers with a simple following response. This is not a claim
of naturalistic traffic behavior or a reconstruction of an external dataset.

Upstream's thin road-mesh ray can miss at segment seams. This restricted
backend uses explicit straight-lane geometry for off-road termination and logs
the original `upstream_on_lane` signal. Physical collision flags remain active.
Off-road exits and safety aborts fail the proposed-stack regression gate.

The existing PPO checkpoints are **transferred frozen policies**, not policies
newly trained in MetaDrive. Original-profile thresholds remain transfer
heuristics; study thresholds are nominal-exposure estimates, not calibrated
collision probabilities. Obstacle filtering is
longitudinal, the optional road-edge filter bounds lateral acceleration, and
the switch guard is a constant-velocity gap check. The boundary model treats
the current rotated footprint as fixed over the next tick and does not prove
safety under tracking error. Camera perception, curved roads,
real-vehicle operation, and formal safety guarantees are outside this release.
