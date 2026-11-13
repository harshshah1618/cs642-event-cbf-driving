# Architecture

The main loop is `event_cbf_driving/stack/controller.py`. Its parts have separate
jobs; the slow maneuver chooser never directly controls the car's actuators.

1. Read a fresh `WorldSnapshot` from the simulator.
2. Deliver any completed decision, subject to expiry and current lane-gap checks.
3. Ask the active skill for forward/lateral acceleration. During a skill change,
   gradually mix the old and new commands.
4. Filter that request through the shield.
5. Decide whether a new maneuver choice is useful and can be requested.
6. Apply the final monitor's action, advance the simulator and record the step.

## Driver

`skills/policy.py` provides the small runtime interface. `skills/train_ppo.py`
trains a new single-critic driver; `skills/legacy_ppo.py` preserves the recorded
dual-critic trainer. Model loading and validation live in `skills/checkpoints.py`.
See [the driver guide](event_cbf_driving/skills/README.md).

The measured skills use lane error and velocity, not nearby traffic. The right
and left skills target different lane centers; return-right aliases the right
skill when no separate checkpoint exists. Critic disagreement is an internal
signal for certain variants, not a collision probability. Single-critic models
have no such estimate and cannot use those routing variants.

## Maneuver chooser and request timing

`router/` implements interchangeable backends. The image router accepts a camera
observation and restricted maneuver choices. Controlled comparisons use a traffic
rule with an imposed wait; those answers are not reported as model inference.

`stack/scheduling.py` estimates whether changing lanes could help and when the
opportunity may close. Other variants ask on a timer, on critic disagreement, or
when the safety layer keeps intervening. These are experimental alternatives.

`stack/pending.py` holds in-flight decisions. Actual inference can run in one
background worker. An unfinished or expired decision cannot change the active
skill. `stack/blend.py` gradually changes commands after an accepted switch.

## Protection and simulator

`shield/cbf.py` limits acceleration and implements the braking backup.
`shield/monitor.py` supplies the final action gate. The maneuver-reserve variant
in `stack/maneuver_reserve.py` keeps room for a delayed lane change and releases
it once a new target is accepted. The combined protection is not a vehicle-safety
proof; see [safety](SAFETY.md).

`env/` supplies native scenarios and the [MetaDrive adapter](METADRIVE.md).
The latter converts requested acceleration into physical actuator commands.
`eval/runner.py` uses the same controller and metrics for compared methods.
`stack/result.py` accumulates calls, progress, safety interventions and traces.

The current measured evidence is in [the final study](../final-study/README.md).
Historical implementations are recorded in [history](../docs/history.md).
