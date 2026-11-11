# Fixed-image current-skill hint diagnostic

Specified 2026-11-11 before predictions. This is an exploratory eight-picture test, not a driving evaluation or an accuracy estimate for a population.

Eight MetaDrive physical scene poses are rendered using the existing front RGB camera. Vehicles are placed at the requested positions and ego heading is set before capture. No driving steps are taken. Every image is held byte-identical while current-skill text takes CRUISE_RIGHT, OVERTAKE_LEFT, and RETURN_RIGHT. The pinned teammate-v1 prompt, including literal ego speed 20m/s, is unchanged. Ego lane text is taken from the camera observation and held fixed within an image.

| Image | Ego y | Heading rad | Parked traffic (x,y) | Design-expected letter |
|---|---:|---:|---|---|
| overtake | 0 | 0 | (40,0) | B |
| transitionRight | 1 | .12 | (40,0) | B |
| transitionLeft | 3 | .12 | (40,0) | B |
| settledLeft | 4 | 0 | (40,0) | B |
| returnLeft | 4 | 0 | (-15,0) | C |
| clearRight | 0 | 0 | none | A |
| blockedRight | 0 | 0 | (40,0),(45,4) | A |
| earlyturnRight | .4 | .08 | (40,0) | B |

These expected letters follow the scene design and teammate prompt, not independent image annotation. Behind-camera traffic in returnLeft is intentionally absent from the forward view; the visible road should be clear. Position/heading changes affect camera content, so differences across poses are associations, not a separately isolated heading effect.

The model is the locally cached Qwen/Qwen2.5-VL-3B-Instruct revision 66285546d2b821cf421d4f5eb2576359d3770cd3 with the exact exported BF16 trained adapter. Original training did not pin its base revision. The unchanged trained_vlm_cpu.py verifies all 504 adapter tensors, active adapter, and SHA256, then performs one actual constrained A/B/C forward per case. No fallback, fine-tuning, network calls, or downloads. Native BF16 CPU with three Torch threads; a systemd user service bounds memory to10GiB, swap to0, CPU to300%, and runtime to900s. LP_NUM_THREADS=1 and P3D_DISPLAY=headless.

All eight captures occur before model loading. lock.json records the design, sources, prompt, adapter, base config, and frame hashes before the first prediction. Each prediction is flushed/fsynced to predictions.jsonl. Final checks rehash all sources and frames. Store logits, conditional A/B/C probabilities, timing and input metadata; conditional probabilities are not calibrated global confidence.

Primary diagnostic: count images whose selected letter changes across skill hints. Within an image this is a controlled text intervention. Secondary: exact pattern of differences from the prespecified design labels, especially transition poses. No statistical generalization, novelty, model superiority, or closed-loop safety claim follows from these eight images. A lack of letter flips does not establish hint independence: conditional probabilities/logits can still shift, and other images could flip.
