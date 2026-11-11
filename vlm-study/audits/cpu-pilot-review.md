# Actual trained CPU physical pilot: trace review

2026-11-11. Read-only inspection of `vlm-study/drive/{summary.json,episode.json,
predictions.jsonl}` and the first two captured frames. No experiment or code
change performed. This is one exploratory physical MetaDrive episode, seed6301,
not a paired performance study or a GPU inference result.

## Observed outcome and model provenance

The episode ends by timeout at60 simulated seconds:33.2652% route completion,
no recorded crash, `freeze=true`, strict success false. Wall time including
render startup is136.504s. Four actual trained-model predictions are recorded,
all letterB/OVERTAKE_LEFT, with zero fallback and zero rejected calls. Three
responses apply; the fourth is marked aborted/unapplied at episode termination.
It would become ready near70.03s, beyond the60s horizon, so it is not a fourth
executed decision or a deadline rejection.

Stored model audit reports all504 adapter tensors equal to the loaded active
BF16 LoRA, nonzero adapter elements, the committed adapter hash and a pinned
local base snapshot. Prediction/call IDs and image hashes align. This supports
actual trained-model use in this pilot, unlike the earlier programmed fixtures.
This review did not reload/reverify the model tensors independently.

Prediction preprocessing+forward durations are18.476,19.517,18.486 and12.869s.
First request is captured atsimt0; first response's ready time is18.5288s including
recorded capture overhead, and the skill actually changes at18.54s. Later requests
occur at19.00,38.60 and57.16s. Same-skill replies remain pending; they do not cause
an immediate new maneuver. These CPU timings cannot substitute for the teammate's
warmed L4 timing or establish what a GPU-backed driving rollout would do.

## Timeline reconstructed from physical actions

Initial ego x=5.179m, y=0, vx=21.895m/s. A stationary car is atx=73.401m in the
right lane. Initial frame visibly shows that car ahead and the adjacent lane
clear. The first B answer is consistent with the scene's overtake intent.

| Sim time | Active skill/state | Requested action and protection |
|---|---|---|
|0s|Cruise right; vx21.895|First B inference begins; driver remains on original skill.|
|1s|vx20.437|PPO requests ax+0.313; shield applies−4.061, braking backup active.|
|3s|vx8.973|PPO requests ax+3.5; CBF applies−6.906.|
|6.04s|x66.495; vx0.096|Vehicle has almost stopped, long before first response.|
|18.54s|Switch to OVERTAKE_LEFT; y0.167; vx0.0033|Delayed B accepted, not rejected by admission.|
|19s|Still physically in right lane|PPO requests approximately ax2.394, ay2.552; steering command saturates+1.0.|
|20.02-59.98s|Stalled near x66.530,y0.174|Every tick reports braking_infeasible; ax forced−7.5 despite positive PPO request.|

After the first skill switch, the remaining2073 control ticks all have both
CBF and braking-stage overrides. Of these2002 are braking_infeasible,45 have
braking-envelope violation and26 report ok. The first post-switch infeasible
tick is19.92s; all ticks from20.02s onward are infeasible. The vehicle advances
only0.01137m and moves laterally0.00697m between switch and final recorded tick;
maximum post-switch vx is0.01446m/s. There is no achieved lane acquisition.

The settled failure state requests roughly `[ax2.386, ay2.570]` but receives
`[-7.5, ay2.570]`, mapped to normalized steering+1.0 and braking≈−0.461.
The stopping-envelope barrier is about−0.010m while the original longitudinal
CBF barrier remains about+0.87m. The backup is therefore the limiting final
longitudinal stage in this state. The slight negative envelope plus its recovery
constraint demands infeasible motion even at almost zero speed.

Validity monitor status is degraded rather than latched emergency; applied
actions equal shield actions on every tick. There are no post-switch lateral
boundary overrides. Thus neither a hidden monitor latch nor lateral road-boundary
correction explains this stall. The cut-in trigger atx92m is never reached;
this pilot never tests the scenario's second hazard.

## What mechanism the evidence supports

The observed chain is **slow CPU response, protective stop before the reply,
accepted overtake intent, then persistent low-speed maneuver/longitudinal-backup
conflict**. It is strongly consistent with a physical bicycle-like vehicle being
unable to translate a lateral acceleration demand into a lane change without
forward motion, while the occupied/intent corridor still includes the stalled
lead and the stopping backup prevents forward progress. This is a grounded
mechanistic interpretation of trace and source, not an intervention-proved causal
decomposition: no faster-model, modified-controller or protection ablation was
run on this exact pilot.

There is **no observed wrong-letter sequence** here: all four responses request
the left overtake, and both inspected frames show a right-lane blocker. The first
reply arrives over12s after vx first falls below0.1m/s. Later B predictions may
be consistent visual decisions or partly reflect the known current-skill text
shortcut; this single trajectory cannot distinguish those explanations. High
reported A/B/C probabilities are conditional on the restricted token set, not
calibrated scene-level confidence or evidence of generalization.

Do not describe this as "the model failed to recognize the obstacle", "a GPU VLM
cannot drive", or "the CBF guarantees safety". The actual trained CPU integration
ran, selected a sensible initial maneuver, avoided collision in this episode,
and failed progress under severe inference delay plus a controller/protection
recovery limitation. An earlier response might avoid the stopped configuration,
but that counterfactual remains untested. Zero collision in one stalled episode
does not establish useful or generally safe driving.

The appropriate deliverable is this successful integration proof with a failed
driving outcome and an explicit trace-supported recovery hypothesis. Future
timing/recovery tests require a separately specified and recorded comparison; none
were performed in this review.
