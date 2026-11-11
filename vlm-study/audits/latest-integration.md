# Latest teammate integration: independent source audit

2026-11-11. Read-only review of commit
`082deae3f941a686400bc85a8bd9ddaca8ee892d` using git objects and the exact teammate
checkout. No cloud launch, inference run or simulation batch performed. Original
and frozen component worktrees were not edited. This audit records source
behavior, not a tested integrated model result.

## Model availability is now resolved

The repository contains an actual adapter, not just a pointer:
`modal/models/qwen25vl3b-lora-r16/adapter/adapter_model.safetensors` is59,942,200
bytes, with504 BF16 tensors and no visual-tower tensor keys. Safetensors header
payload extent matches file size. SHA256:
`e0b999e98601d58bf8a180d1296d4d075fd40058dec5f55e05fb71ae8ea803ab`.
Adapter config specifies rank16, alpha32 and Qwen/Qwen2.5-VL-3B-Instruct base.

This resolves the prior private-adapter access blocker. It does not itself
verify inference correctness or availability of the base model/runtime. Base
revision is null/unpinned. `export_adapter` converts trainable adapter weights to
BF16 before committing; original training used FP32 trainable adapters before
merging. Rebuilding a merge from the exported adapter may not be bit-identical to
the earlier Modal merged model. Prior300/300 frame results must not be asserted
for the rebuilt artifact without a compatibility re-evaluation and weight hashes.

## Critical interpretation and integration issues

1. **Ghost rendering is not physical driving.** `loop_eval.episode` instantiates
   `TwoTrapGauntletEnv`, a point-mass simulation. `render_bridge.Bridge` creates a
   separate stationary MetaDrive scene, fixes ego x=60, maps4m lanes to3.5m and
   respawns visual cars from point-mass relative positions. MetaDrive executes two
   render steps but does not supply the dynamics/collisions of the controlled ego.
   This can test rendered-input point-mass control, not the existing physical
   MetaDrive stack. For physical claims use the actual controlled ego's camera.
2. **Inference-only simulated availability.** VLMRouter sets `ready_at=t+infer_s`.
   Render time is logged but deliberately excluded; the renderer is a remote T4
   call from the L4 container, so that excluded interval includes rendering RPC
   transport. VLLMPolicy constructs base64 URI/messages before its inference timer.
   Cold setup/warmups are also excluded. Rendering need not be counted as physical
   camera exposure in every modeling choice, but acquisition/preparation/network
   time cannot be called zero without qualification. Record component and full
   end-to-end timing separately; choose response availability prospectively.
3. **Blocking simulator, injected delay.** Calls execute synchronously, freezing
   simulation progress during rendering/inference, then add infer_s to simulation
   response time. This is a valid offline delay model if labeled as such; it is
   not verified real-time asynchronous driving. The physical adapter's asynchronous
   contract should capture on the simulation thread, pass detached bytes, and
   include actually observed response availability with timeouts.
4. **Latency replay is not paired causal decomposition.** `loop_eval.run` pools
   all VLM inference times across seeds, then a single LabelRuleRouter cycles that
   list across every rule-delayed episode. No per-scene latency schedule is used,
   no episode-reset replay index is provided, and different routing decisions
   induce different trajectories/triggers/call counts. Thus "instant vs delayed
   isolates delay; delayed vs VLM isolates perception" is too strong. It estimates
   total closed-loop policy differences under an empirical ordinal delay stream.
   Prefer a predeclared common delay schedule per seed/call index, preserving
   actual timing as a separately reported live-model arm; still acknowledge
   endogenous call times and unequal states.
5. **Same-skill response shortcut.** The teammate's old controller sets pending
   only when `resp.skill != skill`, so no-op answers bypass simulated delay and
   pending inflation/occupancy. This makes latency effects depend on the answer
   before that answer should be available. The current physical stack's all-call
   pending option must remain enabled for both model and reference routers.
6. **Invalid answers silently become valid no-ops.** VLMRouter uses
   `LETTER_TO_SKILL.get(letter, current_skill)`. It logs incorrect agreement but
   provides no parse-error/fallback field, so malformed output can look like a
   genuine decision. Renderer/inference exceptions are not converted into a
   bounded, logged router failure; no local timeout/retry/health policy exists.
   Constrained decoding reduces this exposure but does not remove transport,
   malformed output or nonfinite-latency cases. Explicitly reject/log failures.
7. **Old response contract.** The new teammate router uses the minimal older
   RouterRequest/Response API. Its responses lack request identity, submitted time,
   model ID, live/fallback flags, usage and health accounting. The current base's
   lazy health initialization supports older subclasses, so lack of super-init
   alone is not necessarily a crash; however telemetry would remain empty/default
   and actual inference could be classified as non-live. Adapt the full contract,
   do not drop this module unmodified into the current reporting pipeline.

## Observation/reference-policy differences

`label_rule` is only an approximation to the original frame-generation label
function. It ignores all dx<=0 cars and all cars beyond75m. The original labels
count any generated left-lane car, including the beside category's negative-dx
cars; that category labels blocked even when the new rule would overtake. Thus
agreement with this new rule is not identical to original training correctness,
nor a complete safety ground truth. State its reference semantics explicitly.

Renderer includes objects50m behind to150m ahead, while rule considers only75m
ahead. A visible right-lane car beyond75m may induce B and be called an error even
though the reference simply truncates its horizon. Bridge additionally clips
objects to a valid map x-range. It projects headings from velocities, compresses
lateral geometry, uses deterministic alternative vehicle shapes and makes every
rendered vehicle stationary. These are observation-domain changes, not faithful
copies of actual physical camera pixels/footprints.

The model prompt remains the trained v1 text, including current skill and fixed
ego speed20m/s, and says all other cars are parked. Point-mass two-trap includes
a moving cut-in and ego slows. Actual physical integration also changes lane
width, camera state and moving traffic. Preserve the trained prompt for a clearly
identified v1-compatibility baseline, but do not claim those inaccurate text
fields describe the real scene. A changed prompt is a distinct unvalidated model
configuration; the original current-skill shortcut concern still applies.

## Stack and experiment compatibility

Teammate loop uses its older default inflated CBF/PPO stack and root checkpoint
directory, not the frozen calibrated physical study configuration with stopping
backup, road containment and fresh maneuver admission. Its horizon/scenario
randomization and action/jerk meanings differ. Combining the new model with the
current physical stack is a new experiment requiring a new source/design lock;
do not merge teammate environment/controller files over frozen results.

`load_skills` falls back to CRUISE_RIGHT for *any* missing skill, including missing
OVERTAKE_LEFT, without failing. For the new study require the intended PPO bank
and validated metadata; only the declared RETURN_RIGHT alias may be substituted.
Router-return code does not reset LabelRuleRouter's `_i` per episode. Mutable
`VLMRouter.latency_s` also changes between requests/episodes; if current code uses
that attribute to configure a delay bound, set an explicit fixed bound rather
than inheriting the last measurement accidentally.

`loop_eval` summarizes crash/freeze/mean route but lacks a prespecified strict
success endpoint, complete-cell validation, source/data locks and paired
uncertainty estimates. It defaults to five seeds, and no closed-loop result JSON
is committed at the reviewed revision. Code availability is not execution
evidence. The exported adapter and previous static-frame logs are real useful
artifacts, but no new combined performance number follows yet.

## Recommended acceptance checks before a new physical batch

- Pin base-model revision plus exported adapter, merged weights, tokenizer,
  processor, prompt, decoding and dependency versions; verify real model identity.
- Exercise actual controlled-ego camera frames and geometry, with retained images
  showing current/adjacent hazards and precise image timestamps. Distinguish
  physical camera rendering from the optional point-mass ghost bridge.
- Validate delayed same-skill responses occupy the worker, stale/invalid replies
  cannot apply, and failures are counted without silent substitution. Record
  renderer/preparation/transport/inference/full response time separately.
- Confirm live/model/request/health telemetry, pending deadlines and camera-thread
  ownership through the current RouterBackend contract. Keep all common protection
  flags/checkpoints fixed and archive effective configs.
- Declare the reference rule's horizon/occupancy semantics and common per-seed
  delay schedule. Report decision agreement as conditional on visited states,
  alongside collision, strict success, progress, call budget and full timing.
- Freeze fresh scenes and reporting before examining outcomes. Keep the previous
 4608-component factorial and static-frame evidence separate from new real-model
  closed-loop results. No prior private-model blocker remains; runtime execution
  and compatibility verification are the next steps.
