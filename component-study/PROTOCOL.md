# Component contribution experiment, specified before main runs

Question: which parts of the existing straight-road stack prevent traffic
collisions, preserve route progress, or improve control? This is a conditional
evaluation of existing frozen checkpoints, not independent PPO retraining.

## Factors

Full 2 x 2 x 2 x 2 factorial: PPO or the previously fixed PD lane controller;
longitudinal CBF constraint on/off; stopping-envelope backup on/off; fresh
maneuver-admission guard on/off. The road-boundary correction and numerical
validity monitor remain active in every cell. Thus the all-off traffic-safety
cell still retains road containment and invalid-action protection.

Both controller banks have zero critic-distance inflation and zero pending
margin. Maneuvers use the same 24-tick blend. Query policy is fixed 2 Hz with a
deterministic initial offset shared for each seed. This removes shield/sigma
events as query triggers; actual submissions can still be postponed by pending
calls and blending, and resulting state trajectories differ across treatments.
It is a common scheduling policy, not a forced identical absolute query trace.

Three explicit diagnostic routers use simulator state: a teammate-style lane
rule; that rule with all vehicles at relative x <= 8 m removed; and a router
that always repeats the current maneuver. The latter two are synthetic faults,
NOT outputs from the teammate's trained camera model. The state rule returns
right when the right-lane slow lead clears, unlike the old scripted router.
It checks right-lane slow vehicles within 80 m and left occupancy within +/-20 m.
Its label geometry is an explicit implementation choice, not a recovered copy
of the unpublished training generator.

## Scenes and data separation

Actual MetaDrive/Bullet dynamics in all runs. Four scenes: existing randomized
two_trap and dense_highway; a slow lead with a car initially beside the ego in
the target lane; and a lead vehicle that begins braking after about one second.
Custom positions/speeds/events are deterministic from the seed and stored in
each raw row. The same initial-state hash must match across all treatments.

Smoke/pilot: seed 4901, injected 0.125 s latency, all 192 cells. Final: fresh
seeds 5001-5012, injected latencies 0.125 and 1.0 s, all 4,608 cells. The fast
delay is inspired by the teammate's reported GPU timing but is simulated; it
does not measure their network, model, rendering or inference.

Pilot is for implementation checks, not gain selection. Controller gains come
unchanged from the prior study. Freeze source/config/checkpoint/protocol hashes,
full job design, pilot data hash and package versions before final launch.
Never append rows from changed frozen code. Exact job identities are unique;
resume only missing cells under the same lock. No test-seed retuning.

## Outcomes and analysis

Primary: observed collisions and native route success (no collision and >=99%
route). Also report route percentage, stalls, episode duration including
timeouts, achieved lateral jerk RMS/peak, stage correction fractions, admission
rejections, and validity-monitor changes. Every episode records actual motion
before reducing its trace; achieved jerk is never substituted by commanded jerk.

Report all factorial cells and per-scene/router/delay results, including adverse
effects and interactions. Factor differences are paired within seed, scene,
router and delay, averaging over other factorial settings. Resample the 12 scene
seeds jointly across all scenes and conditions for percentile 95% intervals.
Also report conditional full-system leave-one-out comparisons; stage correction
counts alone do not establish causal contribution. No multiplicity-adjusted
significance, rare-event safety, hardware compute saving, or universal necessity
claim is planned. Comfort comparisons that include different collision/timeout
outcomes must be interpreted together with those outcomes.

## Actual VLM arm and evidence boundary

The repository remote currently exposes only main and harsh-testing; the pasted
teammate report explicitly calls ved-testing unpushed. No trained vision weights,
vLLM URL or Modal credentials are available locally. Implement strict image+
ego-state VLM wiring and test its protocol/camera path. Do not label a mock
server, base text model, rule, or synthetic fault as their fine-tuned VLM.
An actual model arm requires an explicit accessible served-model identity;
measure capture-to-answer delay, late/wrong decisions and final driving outcomes
when it becomes available. Keep those results separate from this frozen factorial.
