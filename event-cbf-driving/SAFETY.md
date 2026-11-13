# Safety behavior and limits

This is simulation research software. Zero crashes in a finite experiment do
not prove real-vehicle safety or guarantee that every barrier constraint holds.
See [the measured final report](../final-study/final-report.pdf).

## Runtime checks

- The shield filters the skill's requested acceleration using fresh traffic state.
- A braking backup limits forward acceleration using an approximate stopping model.
- Road-boundary checks constrain lateral motion.
- A final monitor can replace the action or abort when configured to do so.
- Invalid model answers, unsupported skills and expired decisions are rejected.
- Current target-lane gaps are checked before accepting a maneuver change.
- Pending model inference does not stop the control loop from observing traffic.

The particular safeguards enabled depend on the experiment's configuration.
Unprotected baselines deliberately disable some checks; do not interpret them as
recommended driving configurations.

## Assumptions

The driving and safety layers receive simulator positions and velocities.
The learned skills were trained for a limited lane-driving task on empty roads.
Straight-road maneuver checks use current and short-horizon projected gaps.
The braking model assumes response time and achievable deceleration; actual
physics tracking can differ. An infeasible or already violated constraint cannot
be repaired retroactively by a filter.

The tested extra stopping reserve can help preserve room for a later turn, but
its safety necessity was not established: protected reserve-off variants also
avoided crashes. It can trade progress against smoothness. Smoother averages do
not mean every individual drive improved.

Critic disagreement is not a calibrated traffic-risk estimate. Single-critic PPO
models have no disagreement signal; the runtime rejects routing modes that require
it. A model reply proposes a maneuver and does not bypass the action checks.

## Checkpoints, inference and evidence

Checkpoint loading verifies structure, metadata and recorded weight fingerprints.
Restricted deserialization is preferred; older trusted checkpoints may require the
explicitly logged compatibility fallback. Fingerprints detect inconsistent files;
they do not establish who trained a model or certify its driving behavior.

Real camera-model inference needs bounded requests and controlled resources.
A Python deadline rejects an answer but cannot forcibly cancel a native Torch
forward. CPU inference and simulated control frequency do not establish real-time
operation on a vehicle. See [operations](OPERATIONS.md).

Newly trained models must be evaluated before replacing measured drivers. Preserve
source locks, compare methods under matching conditions, and report progress,
crashes, requests and smoothness separately. The completed comparison did not
establish that PPO or a language model is essential for its tested scenarios.
