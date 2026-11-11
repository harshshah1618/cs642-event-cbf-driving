# Safety

This document states what the stack does to fail safely, what it explicitly
does **not** guarantee, and how the measured results should be read.

## Scope

The optional MetaDrive backend adds physical vehicle dynamics for restricted
straight-road experiments. It uses the same longitudinal safety filter with
an approximate acceleration tracker, lateral road-edge constraints, and an
optional fresh-gap maneuver admission check. This does not establish safety under tracking error or
lateral interaction. The original kinematic evaluation and its limitations
below remain separate. See [METADRIVE.md](METADRIVE.md).

The completed `metadrive-study` profile adds a sampled stopping-envelope
backup, intended-lane relevance and intervention re-arming after a changed
intent. Its 6 m/s² braking assumption, 2 m clearance and 0.15 s response
allowance are engineering assumptions, not a verified invariant. Original
inflated barriers and separate stopping-envelope feasibility are recorded;
both can be violated. The proposed nominal suite averages 99.8% route with
zero crashes/stalls in 15 episodes; all 42 final stress cases avoid
crashes/stalls, with some incomplete routes. A small actual-model pilot
establishes no semantic advantage. See [FINAL_REPORT.md](FINAL_REPORT.md).

This is research software for simulated highway scenarios. It is not
safety-certified, has not been validated on a real vehicle, and must not be
used to control one. The native environment is kinematic; MetaDrive adds
physical dynamics and approximate actuation. Neither path has camera
perception, a validated sensor-noise model, or an end-to-end formal proof. The
CBF argument is a longitudinal barrier evaluated at 50 Hz, not an end-to-end
safety case.

The engineering around that research core is deliberately conservative:
malformed data is rejected, artifacts are validated, safety decisions are
observable, and failures are classified instead of silently ignored.

## Safety layers

| Layer | Protection |
| --- | --- |
| Router contract | Answers must have a known skill, valid request id, finite timing, and an allowed backend. Invalid or expired answers are rejected and the active skill is retained. |
| Pending-call margin | While a router call is in flight, the shield target standoff grows by `eta_latency · v · vlm_delay_s`. An instantaneous increase can make the current state infeasible; this is not a guarantee of latency safety. |
| CBF shield | Validates ego/obstacle state, computes the longitudinal barrier for relevant obstacles, clips acceleration, and reports `feasible`, `safe`, `reason`, and `emergency` explicitly. |
| `SafetyMonitor` | Last gate before `env.step`; classifies every shield result, tracks streaks, and can replace the action with emergency braking. |
| Controller | Never acts on a raw router answer; records rejected/suppressed calls, applies a sustained-override guard, and optionally aborts the episode on a latched emergency. |
| Artifact layer | Strict JSON (no `NaN`/`Infinity`), atomic writes, manifest validation, checkpoint fingerprints, config hashing, and deterministic open-loop replay. |

## Fail-closed behavior

With `shield.fail_closed: true` (both shipped profiles):

- a missing ego, non-finite position/velocity, non-finite width in footprint
  mode, non-finite critic uncertainty, non-finite obstacle state, or an
  internal error produces an **invalid** result;
- an invalid result carries emergency braking (`emergency_decel`, lateral
  command `0.0`), `feasible=False`, `safe=False`, `emergency=True`, and a
  reason beginning with `invalid`;
- `SafetyMonitor` counts consecutive invalid results and latches `EMERGENCY`
  after `max_invalid_ticks` (default 5);
- once latched, `EMERGENCY` is sticky until an explicit `reset()`, and
  `safe_action()` returns emergency braking regardless of the shield action;
- a malformed shield action (wrong shape or non-finite values) is also
  replaced by emergency braking.

`fail_closed: false` exists only for compatibility experiments; it raises a
`SafetyError` on invalid input instead of returning a braking command. It is
not used by either shipped profile.

## `DEGRADED` is not `EMERGENCY`

Valid-but-infeasible results are different from malformed data. Infeasibility
means the requested plan cannot satisfy the barrier with the available
deceleration; the shield already applies `ax_min`. This is a normal transient
for an aggressive request or a close cut-in.

The default monitor therefore does **not** latch an irreversible stop for a
valid infeasibility. It reports `DEGRADED` and continues applying the shield's
braking action. The rationale is explicit: turning a recoverable hazard into a
permanent stopped vehicle in traffic can itself be unsafe, and it destroys the
ability to observe whether the controller recovers.

For a stricter policy, construct `SafetyMonitor(latch_on_violation=True)`.
Then `violation_threshold` consecutive valid-but-unsafe ticks latch
`EMERGENCY` as well. This is an opt-in trade-off, not the default.

`stack.abort_on_emergency` controls whether a latched monitor terminates the
episode immediately (`end_reason="safety_abort"`). The production profile sets
it to `true`; research defaults leave it `false`.

## Event-trigger safety

The shield interrupt is a falling edge with hysteresis, not a level:

1. the shield must be clear for `shield_clear_ticks` consecutive ticks before
   the trigger re-arms;
2. once armed, the override must persist for `shield_hold_ticks` ticks;
3. at most one call is made per override episode, unless the study profile accepts a different intent and explicitly re-arms monitoring;
4. a suppressed edge is latched and retried when the call slot opens.

This prevents a feedback loop in which the pending-call margin pushes the
shield into override, the override requests another call, and the new call
extends the margin again. It also prevents dense traffic from generating
router calls every tick. The production profile uses a longer hold
(`shield_hold_ticks: 10`) than the research default (`1`).

## Router and fallback safety

- Live router use is opt-in (`--allow-live` for `auto`); canonical evaluations
  use the deterministic `scripted` backend and need no credentials.
- Fallbacks are recorded with `fallback_used=True` and a reason. Using
  `router.fail_on_fallback: true` rejects fallback output instead of accepting
  it; the CI profile enables this.
- A circuit breaker opens after repeated consecutive backend failures, and
  retries are bounded with backoff.
- `trust_remote_code` is off by default for local transformers models.

## Checkpoint integrity

- Every checkpoint has a state-dict SHA-256 fingerprint. A mismatch is rejected
  even in compatibility mode.
- New checkpoints use schema v2 with provenance; legacy v1 files load with a
  warning unless `strict=True`.
- Duplicate skill weights are detected during validation. `RETURN_RIGHT` is a
  documented alias of `CRUISE_RIGHT` when no distinct file exists.
- Checkpoints are loaded with `torch.load(weights_only=True)` first. A legacy
  file that needs full deserialization triggers an explicit warning. This is
  defense in depth, **not** a sandbox: only load checkpoint files from sources
  you trust.

## Measured evidence (and what it does not prove)

Canonical `production-v1` run: 15 proposed episodes across `two_trap`, `merge`,
and `dense_highway`, five deterministic seeds each.

- 0 crashes, 0 freezes, 89.6% mean route, 4.3 mean router calls.
- `two_trap`: 71.5% route, 0 crash, 0 freeze; `merge`: 100%; `dense_highway`:
  97.3%.
- Regression gate: route floor 70%, no-crash and no-freeze checks, maximum
  proposed router-call budget; all passed.

These results do **not** prove statistical robustness:

- scenarios use `randomize: false`;
- all five seeds produce identical episodes (`ci95 = 0.0`);
- the router is scripted, not a live VLM;
- the world is kinematic and no real sensing is involved.

### Known limitations, stated plainly

- **The CBF is longitudinal.** Lateral relevance is a corridor or footprint
  heuristic. `lookahead_s` assumes constant lateral velocity, clamps `vy`, and
  ignores lateral acceleration, tracking lag, and road curvature.
- **Barrier values can be negative during transients.** In `production-v1`
  `two_trap`, the proposed stack records roughly 157 infeasible ticks and a
  minimum barrier value of about `-15.1` m, while still completing 71.5% of the
  route with no collision or freeze. The monitor reports `DEGRADED`; this is
  evidence of aggressive transients, not evidence that the barrier held.
- **The production shield costs route.** `two_trap` falls from the research
  80.6% to 71.5% because the footprint margin treats more obstacles as
  relevant. The 70% gate leaves only ~1.5 percentage points of margin.
- **No formal guarantee.** Clipping at `ax_min` can saturate. If the physically
  required deceleration exceeds the actuator limit, the barrier cannot be
  enforced; the shield reports infeasibility rather than pretending otherwise.
- **No real-time guarantee.** Python at 50 Hz in simulation is not a real-time
  controller. There is no WCET analysis, lock-free control path, or watchdog.
- **No CARLA validation.** The CARLA adapter is a documented stub/skeleton,
  not a validated simulator integration.
- **No live-router safety claim.** Live LLM outputs can be malformed or
  delayed. The contract validation protects the control path, but no claim is
  made about live semantic decision quality.

## Incident and change policy

Before changing any safety-relevant behavior:

1. add or update a regression test for the failure mode;
2. run `make check` (lint, full tests, checkpoint verification, offline CI
   evaluation);
3. run the production evaluation and confirm the regression gate;
4. inspect `report.md` for new crash, freeze, `DEGRADED`, invalid-result, or
   router-rejection patterns;
5. document any change in measured numbers and any new limitation in this file
   and [RESULTS.md](RESULTS.md).

Do not tune a threshold solely to make the gate pass. The gate is a floor, not
a safety certificate.
