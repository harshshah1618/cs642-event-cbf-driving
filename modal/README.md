# Modal: everything that runs on a GPU

All GPU work for the VLM router runs on [Modal](https://modal.com): rendering frames, LoRA training, evaluation, and the closed loop. Nothing heavy runs on a laptop. The code lives in `../vlm-router/` and `../event-cbf-driving/`. This folder holds the Modal app and the trained model.

| Path | What it is |
|---|---|
| `modal_app.py` | The Modal app (`cs642-vlm-router`). Every stage is a flag: `--stage data/train/eval/all/ood/loop/export/merge`. |
| `models/qwen25vl3b-lora-r16/adapter/` | The trained LoRA adapter (Qwen2.5-VL-3B, r=16, language model only). bf16, 57 MiB. |
| `models/qwen25vl3b-lora-r16/train_summary.json` | Its training config and final validation numbers. |

The merged model (~7 GB) is too big for git. Rebuild it from the adapter with `--stage merge` (below).

The bf16 adapter was checked on 2026-11-11. After a merge it scores 1.00 on both in-distribution test sets (300 frames plus the 50 round-2 frames), the same as the original fp32 adapter. See `vlm-router/results/qwen25vl3b-lora-r16-bf16/`.

## Set up on your own Modal account

```bash
pip install modal && modal setup                # once
export CS642_NO_WANDB=1                         # skip this if your workspace has a Modal secret named "wandb"
modal run modal/modal_app.py --stage merge --adapter qwen25vl3b-lora-r16 --run-name qwen25vl3b-lora-r16
```

The merge stage:

- downloads the base model `Qwen/Qwen2.5-VL-3B-Instruct` (cached in the volume `vlm-router-hf`)
- applies the adapter from this folder
- writes `/vol/ckpt/<run-name>/{adapter,merged}` to the volume `vlm-router`

It runs on a CPU container. The volumes are created on first use.

## Run

```bash
# first closed loop: two-trap (stalled car + cut-in), seeds 0..4, rule vs delayed rule vs VLM
modal run modal/modal_app.py --stage loop --seeds 5
# accuracy and latency on the in-distribution test sets (renders them first if they're missing: --stage data)
modal run modal/modal_app.py --stage eval --skip-base
# harder held-out test (curves, lane changes, oncoming, ...)
modal run modal/modal_app.py --stage ood --skip-base
```

Add `--detach` (`modal run --detach modal/modal_app.py ...`) to anything longer than a few minutes. Without it, Modal stops the app as soon as your laptop's connection drops, for example when the laptop sleeps or the Wi-Fi changes. With `--detach` the run keeps going, and the results are on the volume even if the local copy step never runs.

Results land in `vlm-router/results/<run>/` locally and in `/vol/results/<run>/` on the volume. The closed loop also saves every frame the VLM saw to `/vol/results/<run>/closed_loop_<tag>/frames/`. Fetch them with `modal volume get vlm-router results/<run>/closed_loop_<tag>/frames .`.

## How the closed loop works

The point-mass sims have no camera. On each router call, `VLMRouter` (in `event-cbf-driving/event_cbf_driving/router/vlm.py`):

1. Converts the snapshot into a MetaDrive scene: the ego lane offset and heading, plus each car's distance and lane.
2. Asks the `Renderer` service (T4, `vlm-router/render_bridge.py`) for one front frame. The cars are moved without a reset; a render takes about 0.05 s.
3. Sends the frame, ego lane and current skill to the fine-tuned model (vLLM on an L4, the same setup as the evals).
4. Returns `ready_at = t + measured model latency`. Render time is logged but not added, because a real car gets its camera frame for free.

The skills, shield and controller are unchanged. `LabelRuleRouter` in the same file applies the training-label rule to ground truth. Use it with `latency_s=0` for the instant baseline, or with the VLM's measured latencies for the delayed baseline.

To use the VLM router from other code, pass it two callables:

```python
from event_cbf_driving.router.vlm import VLMRouter
router = VLMRouter(renderer=lambda scene: png_bytes, policy=lambda png, lane, skill: (letter, seconds))
```

Inside Modal these are `Renderer().render.remote` and `vlm_policy.VLLMPolicy(model_dir)`. See `closed_loop()` in `modal_app.py`.

## GPUs and cost

| Stage | GPU | Why |
|---|---|---|
| render (`data`, `ood`, `Renderer`) | T4 | MetaDrive needs a GPU for EGL rendering. A CPU container took more than 45 s per step. |
| `train` | L40S | Best bf16 throughput per dollar for a 3B LoRA. 18 minutes for 3,600 frames over 2 epochs. |
| `eval`, `loop` | L4 | Same GPU as the Colab rounds, so latencies stay comparable (~125 ms per call). |
| `merge`, `export` | CPU | No GPU needed. |
