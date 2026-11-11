# Latency-aware maneuver scheduling: prospective protocol

This protocol was written before validation or held-out experiments. This is an
empirical research investigation, not an assertion of novelty or a safety proof.

## Hypothesis

An observed-traffic opportunity trigger that includes planner response time can
improve progress per planner call over reactive intervention triggers and tuned
periodic scheduling. It may fail against a simple immediate-benefit heuristic;
that heuristic is a required opponent. Latency-aware triggering must also be
compared to the identical trigger with latency removed.

## Isolation study

Use the existing physical MetaDrive scenario adapter, frozen PPO bank, common
filter/backup, common admission check, and identical deterministic router. Disable
pending-distance inflation for every scheduler to isolate scheduling. Compare
legacy shield/sigma trigger; periodic rates 0.1, 0.25, 0.5, 1, 2, 10 Hz; TTC;
immediate-benefit; deadline; and deadline without latency. Delays: 0.05, 0.3, 1,
2 seconds, plus 0.8 +/- 0.5 seconds jitter. Simulation latency is injected, not a
hardware benchmark. The router seed is scenario-seed dependent and identical
across methods. Counterfactual trajectories can lead to different call times.

Validation seeds 101-108, test seeds 1001-1040, across three scenario families.
Use validation only to select polling rates and deadline buffer (0, 0.5, 1 s).
Selection: maximize success (no crash and >=99% route), then mean route, then
minimize calls. Report the entire frontier as well as this selected point. Freeze
selections to JSON before test launch; never retune on test. Test opportunity
thresholds/conditions are fixed in source before launch. Retain every failure.
Use a PD lane regulator as an additional controller ablation on a smaller paired
subset; do not call it a traffic-aware driver.

## External environment study

Use upstream SafeMetaDriveEnv and MetaDriveEnv with seeded procedural roads,
including curves and traffic objects. This is a reproducible environment-based
benchmark, not a claim of an official leaderboard result or reproduction of
historical numbers. Implement a shared traffic-aware PID/IDM fast controller and
finite-horizon lane planner. Compare identical schedulers with matched observation
access. Include upstream IDM and released PPO ExpertPolicy as external controller
references. Every method receives supported simulator state; do not compare scores
to camera-only agents. Publish exact configurations, map fingerprints, observations,
and upstream version. Verify train/validation/test seed sets and actual map hashes.

Standard/no-added-delay results and latency-stress results must be separate.
Planner timing and scheduler timing are measured on CPU. Injected delay is reported
separately. No meaningful theorem follows from a short model-predictive guard.

## Statistics and stopping criteria

Primary endpoints: success, collision episode incidence, route completion, calls,
planning CPU time, scheduler CPU time. Secondary: filter intervention, achieved
jerk, stalls, deadline/admission rejections. Bootstrap paired scene seeds (cluster
by scene across delays), report 95% intervals. Frozen existing checkpoints provide
conditional evidence, not variation across independent training runs. Any newly
trained comparison needs multiple independent training seeds.

The intended win is lower compute at matched safety/progress, or higher success
at matched compute. A reduction in calls alone is insufficient. If immediate
benefit or strong periodic scheduling wins, report that result and identify which
part of the hypothesis failed. Do not weaken opponents or change held-out scenes.

CARLA/SafeBench and Bench2Drive require a separately verified runtime and competent
base agent. They are subsequent transfer targets; do not claim execution or wins
on them unless their actual evaluators have run.

## Stronger polling reference (before held-out launch)

In addition to the globally selected polling rate, evaluate a polling rate selected
separately for each scenario/environment and delay condition using validation only.
This gives polling the same knowledge of delay regimes as the predictive scheduler.
Native held-out test size is therefore 4,200 episodes; upstream road held-out size
is 3,360 including the two native reference agents on standard/no-delay runs.
The road development run before the final physical-footprint implementation is
retained as `pre-freeze-validation.jsonl`; it is excluded from selection.

## Polling clock phase robustness (before held-out launch)

A periodic clock starting at time zero can align with scripted obstacle events.
Add a separate, disjoint phase study. Phase fraction is drawn uniformly once per
scene seed and shared across polling rates; phase seconds = fraction/rate. Tune
rates on seeds 111-114 (three scenarios, nominal/long/jitter delay, 216 episodes).
Freeze those rates before testing seeds 1101-1140 (2,520 episodes). Keep the native
zero-phase-selected deadline buffer; do not retune it on phase results. Both global
and scenario/delay-conditioned polling choices use phase-validation only. Report
this separately from the original zero-phase test. It tests sensitivity to timing,
not an established new algorithm. Neither study can establish rare-event safety.

## Independent audit corrections and final controls (before any held-out launch)

Astra independently identified an unmatched latency ablation, capped road polling,
missing speed-plan-obsolescence recovery, and phase/timing interpretation limits.
The final test uses `no_latency_matched`: identical selected deadline buffer,
zero latency in every trigger branch. Road event schedulers share speed-plan
obsolescence recovery, including a stopped vehicle mid-transition. Nominal 10Hz
road polling is no longer capped at2Hz. Pre-freeze road variants remain excluded.
Add a fixed10Hz periodic+usefulness-gate opponent in all held-out suites. Its
shorter query cooldown makes it a strong comparator, not a pure deadline ablation.
Deadline vs matched-no-latency remains the isolated latency ablation. Freeze both
global and full condition-specific polling mappings and source/checkpoint hashes.
Road timing distinguishes process CPU and elapsed section time and includes total
observation/controller CPU; injected wait is not actual CPU computation.

Final held-out counts: native4800, upstream roads3760 (including160 external-agent
references), initial-offset robustness2880. Validation counts1560/1152/216. PD
controller ablation1200 if executed. Phase is a random initial timer offset; the
existing rolling interval resets on submission, not an absolute wall-clock grid.
Road safe/generalization share geometry for a seed, so pair/bootstrap seed jointly
across both environments and delays. A measured call-count win does not establish
end-to-end compute savings; frozen checkpoints do not measure training variance.
