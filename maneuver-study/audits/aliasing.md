# Front-camera counterfactual aliasing diagnostic

2026-11-11. Actual MetaDriveAdapter camera capture, one renderer process,
`LP_NUM_THREADS=1`, no model inference/cloud. No package or frozen experiment
source was modified. Implemented `scripts/blindspot_alias_test.py`; artifacts
are under `event-cbf-driving/runs/maneuver-alias/`.

## Result: exact aliasing was not demonstrated

All six prescribed pairs have matching ego state/text and differing physical
traffic/admission, but **zero of six pairs have bit-identical decoded RGB or PNG**.
Differences are small:20-37 pixels out of230400, maximum channel difference2-10
on an8-bit scale. Repeated captures within an unchanged physical world also differ
by35-76 pixels. Thus small cross-world discrepancies cannot be cleanly attributed
to visibility of the hidden car rather than rendering variability.

| Right lead x | Added left car x | Different pixels | Max channel change | Repeat free/occupied pixels |
|---:|---:|---:|---:|---:|
|25|−2|34|2|76 /69|
|25|0|20|5|64 /56|
|40|−2|37|10|53 /42|
|40|0|32|10|48 /51|
|65|−2|23|2|37 /41|
|65|0|30|3|35 /36|

All twelve640x360 RGB PNGs, hashes, exact snapshots, pixel/channel differences,
same-world repeat counts and guard results are retained in `results.json` and
the corresponding PNG files. The original executed source snapshot is
`source-original.py`, SHA256
`b262608afe3bf41f1d34855ed780e6a9e2cb7e367c853dbc2920f5968dd1c35d`.
Its results' scope string describes the tested hypothesis; the actual exact-pair
counts are zero. No masking, quantization, cropping or perceptual similarity
threshold was used to manufacture equivalence.

## Matched construction and what the labels mean

Each world is a fresh instance of the same physical adapter/configuration,
reset with seed7301, ego x0/y0/vx20, stationary right lead atx25/40/65. Occupied
world adds a left-lane car atx−2 or0, y4, vx20. Both worlds use identical text:
RIGHT lane, current CRUISE_RIGHT, ego20m/s. Capture is twice atsimt0 with no
control/physics advance; snapshots before/after capture are asserted identical.
Paired ego states are exactly equal, and physical-world hashes differ.

The real frozen `DrivingStack._switch_clear` is evaluated with OVERTAKE_LEFT's
target y4 (no PPO inference/loading needed). It admits the free-lane world and
rejects the occupied-lane world in every pair. This is a verified difference in
the stack's maneuver-admission predicate, **not a proof that B is uniquely optimal
or physically safe in the free world**. The query does not execute a trajectory.

A policy can conservatively avoid overtaking in both worlds; distinguishing
safe-progress opportunity from unnecessary conservatism needs an explicit task
objective. Conversely, the high-level A label is not by itself a guaranteed safe
stop: at20m/s and25m lead distance, braking feasibility requires particular
physical assumptions. This diagnostic establishes neither an inevitable crash
nor universal safety of A. It cannot establish that every trained model would
ignore the tiny image differences; no model was run.

## Separate fixed-clock control, retained as unsuccessful

At root's request, one additional lead40/hidden0 pair used
`ClockObject.M_slave`, frame time123, frame count100 anddt1/60 set immediately
before each capture. Outputs are isolated in `fixed-visual-clock/`.
The free-world repeat still changed54 pixels; occupied repeat changed0. Cross-world
images changed148730 pixels, maximum215, indicating substantial clock/setup
visual effects. This condition did not establish the needed within-world
determinism and is not evidence for or against hidden-car perceptibility.
No further search for an equality-producing rendering setting was performed.

## Conditional information argument

If a future controlled benchmark obtains exactly the same image/text input o in
two worlds with different desired admission labels, any deterministic memoryless
classifier must output the same answer in both. A randomized classifier likewise
has the same conditional action distribution. On a balanced two-label evaluation,
one can derive a maximum50% correctness for that isolated pair. This does **not**
mean safety is impossible: abstaining/stopping, extra sensing, memory or active
perception may resolve or conservatively handle the ambiguity.

That conditional argument is elementary partial observability; its exact-input
premise was not satisfied by these measurements. Do not use the near-identical
pairs as an impossibility proof or a verified model-independent error lower bound.

## Primary literature and novelty boundary

- **Chrisman, AAAI1992, Reinforcement Learning with Perceptual Aliasing: The
  Perceptual Distinctions Approach.** Explicitly discusses physically distinct
  situations indistinguishable from immediate perception but requiring different
  responses, including a forward-looking robot affected by an object behind it.
  The broad blind-spot/one-frame impossibility idea is therefore established.
  [AAAI paper](https://cdn.aaai.org/AAAI/1992/AAAI92-029.pdf)
- **Carr, Jansen, Junges and Topcu, Safe Reinforcement Learning via Shielding under
  Partial Observability.** Develops shielding for partially observed environments,
  making clear that partial-observation safety and performance are existing
  research topics rather than a new consequence of this VLM example.
  [Author paper](https://arxiv.org/abs/2204.00755)
- **Zhang and Fisac, RSS2021, Safe Occlusion-Aware Autonomous Driving via
  Game-Theoretic Active Perception.** Treats initially hidden traffic participants
  and future observations, balancing safety with conservatism in driving.
  [RSS proceedings](https://roboticsconference.org/2021/program/papers/066/)
- **Orzechowski, Meyer and Lauer, Tackling Occlusions & Limited Sensor Range with
  Set-based Safety Verification.** Provides prior work on validating trajectories
  under unobserved traffic and restricted sensing.
  [Author paper](https://arxiv.org/abs/1807.01262)

These sources were consulted directly. They rule out claiming the general
observability limitation as novel. A carefully controlled, hash-verifiable
physical-camera counterfactual benchmark could still be a useful diagnostic
artifact, but this small search does not establish novelty of that benchmark
format. Current contribution is an honest failed exact-alias test with retained
near-identical pairs, ground-truth admission contrast and rendering-repeat controls.

## Reproduction

From the package directory, run:

```bash
LP_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/blindspot_alias_test.py
LP_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/blindspot_alias_test.py --fixed-visual-clock
```

The second command is the explicitly separate unsuccessful rendering control.
The current script passes Ruff. Exact RGB/PNG equality is reported without a
tolerance; results should not be relabeled if a later renderer condition differs.
