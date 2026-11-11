# Actual teammate VLM source and committed-results review

2026-11-11. Read-only checkout inspected at
`/home/harsh/Projects/cs642-event-cbf-teammate`, commit
`1f16b7ff59aae20c2a9adfbeb4943762d4b1f5b4`. Git working tree was clean. No cloud
job, training, inference, rendering or teammate-file modification was performed.

**Correction to earlier availability assessment:** advertised branch absence did
not establish source absence. Root fetched the exact commit successfully. Actual
source and committed evaluation/training logs are now available and audited.
The trained adapter/merged weights, dataset images and authenticated inference
endpoint remain unavailable in this checkout. These are verified committed
results, not independently rerun inference results.

## Recomputed results

Every prediction's stored correctness flag was checked against `pred==want`;
counts, confusion matrices and rounded overall accuracy summaries were
independently recomputed and match. Within each base/fine-tuned pair, filenames,
labels, kind and obstacle metadata agree exactly. Filenames are unique within
each evaluation set. No image bytes/hashes are committed, so identical filenames
cannot independently prove identical rendered inputs or train/test deduplication.

| Evaluation set | Base correct | Fine-tuned correct | Fine-tuned accuracy |
|---|---:|---:|---:|
| v1 held-out generator frames |150/300|300/300|100%|
| Round-2 distribution redrawn |20/50|50/50|100%|
| OOD curve |30/60|42/60|70.00%|
| OOD lane change |26/60|22/60|36.67%|
| OOD oncoming |38/60|51/60|85.00%|
| OOD far |0/60|39/60|65.00%|
| OOD close |0/60|40/60|66.67%|
| OOD beside |60/60|3/60|5.00%|
| OOD multiple cars |27/60|55/60|91.67%|
| All seven OOD categories |181/420|252/420|60.00%|

Base OOD aggregate is43.10%. The base outputs A on all ID frames and all OOD
frames except three lane-change predictions; beside's100% base accuracy is a
single-class A artifact. Fine-tuned beside predictions are57 B and3 A, confirming
57/60 incorrect overtake classifications in this constructed frame category.

The committed training summary records NVIDIA L40S,18.1min,450 optimizer steps,
LoRA rank16, batch4 x accumulation4, seed0, learning rate0.0002,2 nominal epochs.
**Final validation accuracy is239/240=99.5833%, not100%** (one left_only miss).
The100% claim applies to the distinct300-frame test and50-frame r2test. Validation
history moves from50% atstep0 to66.67% at50,94.17% at100 and99.58% at450.
Only one training seed is represented.

Fine-tuned L4 timing recomputed from saved per-frame times: v1 median125.3ms,
p90128.4ms; r2 median125.65ms (rounded125.7 in artifact), p90129.3ms. OOD category
medians range124.2-127.3ms. Round-2 CSV separately reports the best tested
zero-shot configuration as7B front-camera direct76%, median191ms. The CSV is
summary evidence, not a fresh reproduction.

## Split and label audit

`modal_app.data_jobs` uses6 kinds with600 training frames/kind,40 validation/kind,
50 test/kind:3600/240/300. Per-shard seeds are1000+100*kind+shard for training,
5000+100*kind+shard for validation,9000+100*kind+shard for test. Those planned
seed sets are disjoint. r2test uses seed1 and variety0; OOD uses7000..7006.
The training loader reads train/val only, and evaluator reads explicit test paths.
No direct test-label training use is visible in this source.

That verifies a split construction, not full dataset provenance: labels/images
are in a Modal volume, not this checkout. Existing-data caching checks only for
`labels.json`, without verifying generator/config/image hashes, expected counts
or freshness. Thus changed generator settings could reuse stale data under the
same dataset name. The training summary also does not save dataset file hashes.

`gen_frames.rule` is a deterministic scene classifier:

- RIGHT: B if any right-ahead car and no left car; otherwise A.
- LEFT: B if any right-ahead car; otherwise C.

"Left car" means any generated left-lane car, without a distance-dependent gap
or velocity calculation. Right-ahead is generated dx>0, without a universal80m
cutoff. Geometry labels are computed from intended scene specifications, before
the two physics/render steps, not independently inferred from final pixels.
Generation assertions verify kind-label consistency, not frame visibility or
post-step geometry. Source shows no direct answer label in inference prompt:
`messages` uses image, ego lane and current skill, while `want` is scoring-only.
Training answer token indexing has a two-sample sanity check and causal next-token
logits; no obvious teacher-forcing answer leakage was found.

## What generalization was and was not measured

ID train/val/test use the same straight-road generator, largely the same fixed
map, parked other cars, tightly bounded offsets/headings and six scene templates.
The generator physically sets ego speed to0, while text always says20m/s. This
is deliberately static scene-image classification; no action is applied from the
model's prediction and no driving outcome is measured. A held-out random image
from that generator is not evidence of closed-loop planning safety or diverse
road generalization.

OOD uses an alternate generator for curves, inter-lane pose, oncoming headings,
distance extremes and multiple vehicles. Curves use maps seeded1000..1049,
possibly several frames per map. Oncoming vehicles are posed in opposite lanes
but remain static; this is visual oncoming context, not moving interaction.
Treat the seven categories separately; their equal-weight60% aggregate is a
diagnostic mixture, not deployment accuracy. Frame-level IID uncertainty would
ignore common templates/maps and there is no independent training-seed estimate.

Current-skill shortcut evidence is strong but not a clean causal ablation:
training always pairs RIGHT with CRUISE_RIGHT and LEFT with OVERTAKE_LEFT. OOD
lanechange changes both physical pose and skill text. Predictions contain only
B/C and are poor, but **"always copies current skill" overstates the raw data**:
source-implied agreement with skill letters is48/60=80%, not60/60. For example,
all9 clear lanechange frames predict C although their implied current skill is
OVERTAKE_LEFT. Proper causal isolation requires image-fixed text interventions,
text-only/image-only ablations or a controlled retraining comparison.

The exact fine-tuned overtake counts corroborate specific weaknesses: curves4/15,
oncoming1/10; keep-left far7/28 and close10/27. Proposed perceptual explanations
are plausible hypotheses, not proved mechanisms from aggregate accuracy alone.
Single forward-camera blind spots motivate extra sensing/admission protection;
these rows do not prove the CBF can safely correct every such maneuver.

## Timing scope

`eval_router.py` loads and perturbs each image, PNG/base64-encodes it and builds
messages **before** starting the timer. Timed section is warmed local `llm.chat`,
batch1, one constrained A/B/C token, prefix caching, NVIDIA L4. Three warmups
precede each set. Randomly changing one pixel is intended to avoid identical-image
vision caching. Loading time is recorded separately, not included in medians.

This timing excludes scene capture/rendering, file decode, explicit PNG/base64
preparation, request/network transport and long-running queue effects. Internal
vLLM work inside `llm.chat` remains included. It supports low warmed classifier
latency in that setup, not "latency solved" for the full asynchronous driving
pipeline. Model and processor use unpinned Hugging Face names, so exact model
revision hashes should be captured before any reproduction.

## Compatibility with current component experiment

The current factorial is unchanged and remains synthetic rule/fault evidence.
Its state rule is **teammate-inspired, not identical**: it imposes80m right-lead
lookahead, slow<15m/s and left occupancy within20m. Teammate labels use any
generated right-ahead and any left car. Core lanes are4m apart versus generator's
3.5m. Actual driving has moving ego/traffic, unlike parked-frame training. The
new camera prompt that omits current skill is not the trained v1 prompt.

A compatibility runner may import exact committed prompt/processor settings and
compare rule decisions or recorded labels, but it cannot synthesize genuine
model outputs without the checkpoint/endpoint. Replaying a static confusion
matrix or injecting57/60 blind-spot mistakes would still be synthetic and would
not establish the joint trained-VLM/PPO/protection result. Actual combined claims
require the real weights/model ID, matching input contract, timestamps, physical
closed-loop rollouts, and matched controls on new scenes.

## Artifact fingerprints

Results directory: `vlm-router/results/qwen25vl3b-lora-r16/` in the pinned teammate
checkout. SHA256 of reviewed evaluation JSONs:

- eval_base.json:755ef7802100ce86904e41ed9ad21ac81d790a7f1231b947bea54b3d43e0c9ad
- eval_finetuned.json:ad9a1b776ce622a9536f4fd8ffa97736c6e3fc32fb182338b44b44bb64f4ab9d
- eval_ood_base.json:ed016e0dfec32ad3e2571589664c37cd2b99f614a900198c64204f581e58a347
- eval_ood_finetuned.json:c8b7946c679909d15a0d419efe60adaa907366576d36cfcf38752bad3eed815e
