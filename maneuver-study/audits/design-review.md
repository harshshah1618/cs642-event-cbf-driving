# Maneuver reserve investigation: independent design review

2026-11-11. Read-only review of teammate GPU logs and proposed new physical
mechanism study. No simulation, core edit or prior-art novelty determination was
performed. Earlier worktrees/results remain frozen. This is a new investigation.

## Newly available GPU evidence

Reviewed `origin/ved-testing` commit
`c435af3cbcd890d7e4bd564bc6eae3acf1711c99`, including the closed-loop log introduced
by0b43bb4. The saved closed-loop metadata identifies **NVIDIA A100-SXM4-80GB**,
not L4. It is the previously audited point-mass control/MetaDrive ghost-render
bridge, not the physical controlled-ego camera experiment.

| Five-seed arm | Mean saved route | Crashes | Freezes | Calls |
|---|---:|---:|---:|---:|
| Instant state rule |85.72%|0/5|0/5|26|
| Ordinal delayed state rule |85.56%|0/5|0/5|26|
| Actual VLM |48.42%|0/5|2/5|20|

All episodes end at12s; **all three arms have0/5 successes at the99% route
criterion**. Those counts should not be relabeled a successful-driving benchmark.
The rounded saved rows support the means above; original report rounds to85.7,
85.6 and48.4. VLM matches its reference on13/20 visited-state calls.

Every seed first receives a correct B, then a C while the ego is still physically
RIGHT and current skill is OVERTAKE_LEFT; the reference still asks B. Those five
early maneuver reversals account for five errors. Seeds0/2 additionally receive
C instead of B on their next request and freeze. This is evidence of erroneous
early maneuver reversal in a closed-loop classifier/controller interaction, not
the same failure as the physical CPU pilot's correct but18s-late B. Pose/text
confounds remain; the log does not identify a purely visual versus text-induced
cause, and thirteen correct labels do not establish a deployment error rate.

Median inference is97.9ms, p9099.8ms, but excluded rendering has median320.6ms
and maximum6034.3ms. Median recorded render+inference sum is420.15ms. The delayed
rule consumes a globally pooled/cycled latency stream,26 calls against20 VLM
calls, without per-seed pairing; no-op answers bypass pending in that older
controller. Thus "the performance gap isolates perception cost" is not a valid
causal decomposition. It is a descriptive total-policy comparison under the
chosen inference-only delay model.

The new BF16-merge verification artifact independently recomputes to50/50 r2test
and300/300 ID test, on A100, medians72.95/73.9ms. This supplies the missing
re-evaluation of the exported BF16 adapter on those ID sets; it does not establish
OOD, physical closed-loop performance or bitwise equivalence to the earlier merge.

## Prospective physical question

Can preserving space before stopping allow an already accepted, fresh-clear
lane-change intent to be physically executed after a long planning delay?
Separate this from model competence using known-correct exogenous intent.
The CPU trace motivates the question: backup stops the vehicle before response;
then a slight negative stopping envelope and a near-zero-speed steering demand
leave it stalled. The experiment must distinguish finite numerical recovery,
spatial maneuver room, and an unsuitable low-level lane-change primitive.

### Essential controls

- Issue the same known-correct request at an exogenous simulation time and
  deliver at a predeclared delay. Do not let shield interventions change the
  request clock. Every answer still receives identical fresh admission checks.
- Pair initial physical states, initial lateral offsets/headings, ego speed,
  obstacle footprints and delays. Keep PPO weights, actuator adapter, road bounds,
  numerical monitor and physical time step fixed; archive effective configs.
- Include zero/short delays, delays before/around stopping and delays well after
  stopping. Use the same episode horizon and retain timeouts. Do not select test
  seeds based on whether the old controller deadlocked.
- Use CPU seed6301 and past failures only for development. Freeze fresh test
  seeds/design/constants before testing. Pair bootstrap by shared scene seed,
  including all speed/delay/offset conditions in that cluster.
- Report actual intent availability, admission, first motion, corridor-clearance
  time, completed lane change, collision and route success separately. A guard
  that simply refuses the maneuver may remain collision-free without solving it.

## Baselines needed to identify a contribution

1. Unchanged full physical controller/protection.
2. Numerical tolerance/recovery fix only, with predeclared tolerance derived from
   integration/measurement resolution, not a tolerance selected to pass a scene.
3. Fixed larger stopping clearance/headway, tuned on development only, with its
   safety/progress/call tradeoff shown rather than a single weak point.
4. Static reserved room **with the same acceptance/release rule** as the candidate.
5. Dynamic pending-dependent reserve with that identical release rule.
6. If a2D recovery primitive is needed, cross that primitive with static versus
   dynamic reserve or keep it common; do not give only the candidate a competent
   low-level escape controller.

The fourth control is critical. Permanently increasing braking clearance can
prevent the car from ever using the saved space. Beating that comparator may
demonstrate only margin release, not dynamic maneuverability forecasting. Factor
reserve placement and release separately, or present a transparent sequential
mechanism study if a full factorial is too large. A static reserve+release baseline
may explain the entire improvement; that is a useful finding rather than a reason
to weaken it.

## Geometry: what a viable reserve would mean

Physical steering cannot create the point-mass lateral acceleration at zero
speed. A reserve must enable a realizable forward path until the ego's swept
footprint clears the source-lane lead, while maintaining target/rear safety.
Fresh target-lane admission alone does not verify that transition path.

A useful derivation starting point is a kinematic bicycle with curvature bounded
by `kappa_max=tan(delta_max)/wheelbase`. For a feasible symmetric two-arc path of
radius R>=1/kappa_max, each arc angle theta, lateral displacement d and longitudinal
advance x satisfy:

`d=2R(1-cos(theta)); x=2R sin(theta); x=sqrt(4Rd-d²)`.

This is a geometric candidate path relation for0<=d<=2R, **not a universal minimal
reserve, viability theorem or guarantee for the actual PPO steering history**.
Its path length is2R*theta, not its longitudinal advance. Larger heading-dependent
footprint extents, finite steering slew, acceleration/braking bounds and tracking
error may demand more room. Use source-obstacle clearance displacement rather
than blindly substituting a full lane width; then verify the complete path until
that clearance is reached. A one-arc partial escape has different heading and
footprint consequences and cannot be treated as the completed two-arc maneuver.

Operationally: choose a specified realizable speed/steering primitive, propagate
the physical model from the measured stopped/near-stopped state, and check its
swept ego rectangle against every predicted obstacle rectangle and road boundary.
Include obstacle motion and rear approaches throughout latency plus execution,
not merely at response delivery. Any conservative tracking margin must be tied
to observed adapter errors or clearly identified as an empirical margin.

Reserve release should remove **optional stored maneuver room**, not disable the
baseline physical collision constraint or hide an infeasible barrier. Require a
fresh successful path/admission check at release. Log base versus optional barrier
values so a collision-free gain cannot be attributed to silently turning off a
guard. A discontinuous margin reduction is a controller mode transition requiring
explicit accounting, not proof of forward invariance across changing safe sets.

At almost zero speed, a small negative stopping-envelope h can make the discrete
recovery inequality impossible even at maximum braking. Test whether tolerance
alone restores tiny motion and whether that is enough for footprint clearance;
do not equate removing a−0.01m numerical deficit with proving adequate turn room.
Conversely, if a simple tolerance or fixed reserve+release solves all fresh cases,
the more elaborate candidate has not demonstrated added value.

## Guard, actuation and camera confounds

- Keep all stage masks observable: base CBF, stopping backup, road-boundary,
  intent corridor and fresh admission. Record requested/post-stage/applied
  accelerations, steering/throttle saturation, achieved acceleration and heading.
- Record the physical ego lane separately from selected skill and blended action;
  "OVERTAKE_LEFT selected" is not completed lane acquisition.
- Rear guard versus stopping guard interactions matter: braking for a forward
  obstacle may worsen a rear approach. An empty adjacent lane development case
  is insufficient for a broadly safe reserve claim.
- Known-correct intent makes camera/model inference unnecessary in this mechanism
  study. Do not conflate ghost-render point-mass GPU logs with the physical plant.
  Any subsequent true-model arm must use actual ego-camera frames and complete
  latency accounting, or explicitly declare another timing model.
- The GPU failure pattern suggests a separate commitment/intent-consistency issue:
  contradictory reversal during lane acquisition. A hold-until-settled heuristic
  is a strong simple comparator to any learned or viability-aware commitment
  mechanism. Do not silently add that improvement only to the reserve candidate.

## Reporting and novelty boundary

Primary endpoints should require collision-free progress/completed maneuver,
not merely fewer infeasible ticks, lower jerk or zero crashes while stopped.
Show failure regimes and conditional successes for all baselines. Distinguish
physical reachability predictions, empirical tracking success and formal claims.
The available evidence motivates a practical mechanism investigation, but no
novelty claim is established. A dedicated prior-art search and comparison to
existing maneuver planning, viability/backup control and switched safety-filter
methods must precede any claim of a new contribution.

## Prototype review: stopping reserve and fair fixed comparator

Read `stack/maneuver_reserve.py` and `scripts/maneuver_probe.py`; no code edited.
The wrapper adds an optional amount to stopping-envelope clearance and removes
that amount when target_y changes. In the current probe, fresh skill-switch
admission is enabled, so the target change follows an accepted maneuver. The
ordinary CBF, base stopping clearance and road guard still execute. This invariant
depends on the caller preserving admission; the wrapper has no independent
admission token and should not be advertised as safe when used with arbitrary
target changes or a disabled admission check.

Reserve arms as soon as a pending query and slow lead are observed. This is
early protective action during planning, not additional delay-aware reasoning
inside the reserve formula. Adding clearance at that instant can make an
otherwise feasible stopping constraint infeasible, so log base/extra barrier
values and early braking. The current geometry formula selects a full4m two-arc
longitudinal extent from wheelbase/maximum steering; it omits actual initial
lateral offset, heading, steering slew and the PPO's executed trajectory. It is
a geometric heuristic, not a viability certificate. With one vehicle geometry,
it reduces to one fixed reserve value.

Latched modes retain the reserve after a same-skill response or rejected target
change; they release only after an accepted changed target. Thus a wrong/stubborn
VLM can retain extra stopping room indefinitely. That may be deliberately
conservative, but its timeout/progress behavior must be declared and tested
separately from the always-OVERTAKE known-correct probe. A successful correct-
intent experiment does not establish recovery for incorrect, contradictory or
perpetually rejected VLM answers. Pending-only differs precisely in releasing
when a request completes, including a no-op; its safety/performance difference
is not solely a reserve-magnitude comparison.

The primary comparator is **tuned fixed_latched** with exactly the same arming,
fresh admission and release behavior. Constant-reserve development values must
span the geometric candidate's range, including a larger robust constant; do
not cap the baseline below what weak steering needs. A geometry-conditioned
fixed choice, selected from development only, is a useful additional reference
if claiming adaptation rather than simply avoiding manual tuning. Permanent
static clearance remains a mechanism diagnostic, not the sole strong opponent.

Root's declared development grid—steering15/25/40degrees, speed14/gap45 and
speed26/gap90, delays0.125/12/18s, PPO/PD, fixed2/4/8/12/20m plus geometry—can
map the mechanism without changing past frozen studies. Held-out combinations
must be genuinely distinct and frozen after choosing a constant baseline.
Report deterministic condition coverage, not independent-seed confidence:
changing only a seed while all physical params remain fixed does not create
independent scenes. Audit actual initial-state hashes and steering parameters.

Fair primary metrics are collision-free route/maneuver completion, stall/timeouts
and time to physically clear/acquire the lane. Also report admitted intent time,
first positive motion after reply, minimum base braking envelope, requested versus
achieved acceleration, steering saturation, extra-clearance peak/integral and
release events. Less reserved space alone is insufficient if progress/safety
deteriorates. Numerical-tolerance-only remains a required explanation check.
Any later2D recovery controller must be common to geometric and fixed-latched
arms, or explicitly factored as a separate contribution. No novelty claim follows
from the prototype or this diagnostic grid.
