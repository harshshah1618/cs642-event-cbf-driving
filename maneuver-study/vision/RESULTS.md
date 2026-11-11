# Identical images change predictions when the current-skill hint changes

2026-11-11. Completed all 24 actual trained-model forwards over eight physical MetaDrive camera poses. Changing only current-skill text changed the chosen letter in **8/8 selected images**. Six images changed between a left-targeting B and a right-targeting A/C; the other two changed A versus C, which share a right-lane target. This demonstrates hint sensitivity in this diagnostic, not a novel shortcut-learning mechanism or a population accuracy result.

| Captured pose | Design label | Hint CRUISE_RIGHT | Hint OVERTAKE_LEFT | Hint RETURN_RIGHT |
|---|---|---|---|---|
| overtake | B | B | C | C |
| transitionRight | B | B | B | C |
| transitionLeft | B | B | B | C |
| settledLeft | B | B | B | C |
| returnLeft | C | B | C | C |
| clearRight | A | A | C | C |
| blockedRight | A | A | C | C |
| earlyturnRight | B | B | B | C |

The RETURN_RIGHT hint yielded C for every selected picture, including five images with a right-lane lead still ahead. The OVERTAKE_LEFT hint yielded C for the initial centered overtake view but B for the two transition views, early-turn view, and settled-left view. Thus these data do **not** support a blanket claim that transition images are worse: the pattern depends jointly on pose and hint. The CRUISE_RIGHT hint gave B even in the returnLeft view with clear road ahead. None of these contrasts identifies the training cause of the dependence.

There were 13/24 exact matches to prespecified scene-design letters: 7/8,5/8,1/8 for the respective hints. These are descriptive counts for deliberately selected pictures and counterfactual hints, not representative accuracy estimates. A posthoc lane-target grouping gives17/24 matches, since A and C both target RIGHT. In the current calibrated skill bank RETURN_RIGHT also aliases CRUISE_RIGHT weights when its checkpoint is absent. Therefore A/C differences must not be reported as demonstrated distinct vehicle behavior. No vehicle trajectory was executed here.

All eight rendered frames were visually inspected. Captured x/y/heading, lane text, obstacle states and frame hashes are stored in the lock; ego longitudinal speed was20m/s in all eight snapshots, matching the literal speed text. Within each group the exact PNG bytes and ego-lane text were unchanged. Across groups, pose/scene and sometimes lane text change together; this is not an isolated heading experiment. Conditional A/B/C probabilities are recorded but are not calibrated confidence.

The native BF16 model loaded once and verified all504 trained adapter tensors. No fallbacks, downloads, cloud resources, or core edits occurred. Median inference latency was12.45s (range12.37–17.24s) on three CPU threads. The bounded service exited successfully after6min4s including eight captures, model loading and cleanup; systemd reported8.1GiB peak memory. These timings are local CPU observations, not estimates of GPU performance.

Source, prompt and frame locks were verified after the final prediction. The design lock SHA256 is `c46e1ad648597db9bf87f111f4b98a01c0ba5a33588f416b4da5e910b87035f8`; script SHA256 is `35e85e4d1446560d004954c3387e554d4d36f5ccd7c80da992ea877e78d8658b`. Two lightweight aggregation/design tests passed; Ruff passed. They are separate from the24 real model calls.

Evidence is under `event-cbf-driving/runs/maneuver-vision/`: `lock.json`, eight `frames/*.png`, all `predictions.jsonl` plus deterministic gzip copy, `analysis.json`, explicitly posthoc `supplementary-analysis.json`, and `service.log`. The protocol was written before prediction completion. The inference script froze design/source/frame hashes before its first model prediction.

A useful next test would cross held-out rendered scenes with alternative hint wording or omission, holding image and other text fixed, and then test whether a cheap fresh-state admission guard prevents harmful target reversals. That would require a new protocol; these results do not establish that deleting the hint improves a trained model or that any physical guard is sufficient.

## Training-support qualification (source audit after diagnostic)

Inspection of teammate commit `c435af3cbcd890d7e4bd564bc6eae3acf1711c99` found that `vlm-router/gen_frames.py:153` assigns CRUISE_RIGHT whenever ego lane is RIGHT and OVERTAKE_LEFT whenever it is LEFT. That generator never emits RETURN_RIGHT as current-skill input. `gen_ood.py:94–103` deliberately changes the lanechange category to OVERTAKE_LEFT for right-lane mid-overtakes and RETURN_RIGHT for left-lane mid-returns; its other categories default to the original lane/skill mapping at lines127–129.

Consequently many counterfactual image/hint pairings here lie outside the support of the inspected training generator, and all RETURN_RIGHT inputs are unsupported by that generator. This establishes a source-level support limitation, not proof of the exact contents of every training artifact or proof of label leakage. Lane text and current-skill text are redundant in that generator; changing one can create a combination the trained model never saw. The observed flips remain causal effects of text substitution on fixed images, but cannot be interpreted as ordinary in-distribution error rates. In the generator-supported lane/hint pairing, these eight poses give8/8 design-label matches (RIGHT with CRUISE_RIGHT, LEFT with OVERTAKE_LEFT); some rendered poses themselves remain outside the centered training geometry. This is a posthoc descriptive grouping, not a new test split.
