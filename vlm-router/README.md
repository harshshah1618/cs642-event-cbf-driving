# VLM router: LoRA fine-tuning on MetaDrive frames (Modal)

The router is the slow tier of the stack. It takes a front-camera frame plus a short state text and returns one letter:

- `A` → `CRUISE_RIGHT`
- `B` → `OVERTAKE_LEFT`
- `C` → `RETURN_RIGHT`

## Why fine-tune

The Colab rounds 1–2 used zero-shot Qwen2.5-VL. Calls were fast enough (one-letter output, prefix caching, about 300 image tokens):

| Model | GPU | Latency per call |
|---|---|---|
| 3B | L4 | ~120 ms |
| 7B | L4 | ~190 ms |

Accuracy was the problem. The 3B always answered `A`. The best result was 7B on the camera view at 76%, and that came mostly from the lane named in the state text. No model ever picked overtake for a car ahead in our lane.

MetaDrive gives exact labels for free, so we fine-tune the 3B on rendered frames and keep the same prompt and the same one-token output.

## Files

| File | Runs on | What it does |
|---|---|---|
| `gen_frames.py` | Modal T4 (EGL) | Renders labelled camera frames. The label comes from `rule()` applied to the ground truth. |
| `prompts.py` | — | The router prompt, shared by training and evaluation (same text as round 2 `direct`). |
| `train_lora.py` | Modal L40S | LoRA on the language model only; the vision encoder is frozen. Loss is on the answer token only. Logs to W&B. |
| `eval_router.py` | Modal L4 | vLLM, batch size 1, constrained letter. Reports accuracy per scene kind and median/p90 latency. |
| `render_bridge.py` | Modal T4 (EGL) | Draws a point-mass world state as one MetaDrive front frame (the closed loop's camera). |
| `vlm_policy.py` | Modal L4 | The fine-tuned model behind one call: (frame, ego lane, current skill) → (letter, seconds). |
| `loop_eval.py` | Modal L4 | First closed loop: two-trap scenario, rule vs delayed rule vs VLM. |
| `../modal/modal_app.py` | your laptop | Orchestrates everything on Modal: data → train → eval → closed loop. See `../modal/README.md`. |

## Scene kinds

The label is the router answer.

| Kind | Scene | Label |
|---|---|---|
| clear | ego RIGHT, nothing ahead | A |
| left_only | ego RIGHT, car ahead in LEFT lane only (distractor) | A |
| overtake | ego RIGHT, car ahead in RIGHT lane | B |
| blocked | ego RIGHT, car ahead in RIGHT lane and in LEFT lane | A |
| return | ego LEFT, passed car behind us | C |
| keep_left | ego LEFT, car still ahead in RIGHT lane | B |

The training data varies:

- car distances (5–75 m)
- vehicle type (S/M/L/XL)
- small lateral and heading offsets
- ego position

## Datasets

All datasets live on the Modal volume `vlm-router`.

- `v1/train`: 600 frames per kind, 3,600 in total.
- `v1/val`: 40 frames per kind.
- `v1/test`: 50 frames per kind. It uses different seeds from `train` and `val`.
- `r2test/test`: the exact round-2 camera test set (50 frames), for a direct before/after comparison.

## Run

```bash
pip install modal && modal setup            # once; needs a Modal secret named "wandb" with WANDB_API_KEY
modal run modal/modal_app.py --smoke   # tiny end-to-end check
modal run --detach modal/modal_app.py  # full: render (~30 T4 containers), train (L40S), eval (2x L4)
modal run modal/modal_app.py --stage eval --run-name qwen25vl3b-lora-r16   # one stage only
```

- Results are written to `vlm-router/results/<run>/`.
- Training curves are in the W&B project `cs642-vlm-router`.
- Checkpoints are saved to `/vol/ckpt/<run>/adapter` (LoRA) and `/vol/ckpt/<run>/merged` (bf16, which vLLM can load).

## GPU choice

- **Training: L40S** (48 GB, Ada). It gives the best bf16 throughput per dollar on Modal for this job, with room for batch 4 without gradient checkpointing. An H100 is faster but about 2× the price, and a 3B LoRA at this size doesn't keep it busy.
- **Evaluation: L4**, the same GPU as the Colab rounds, so latencies stay comparable.
