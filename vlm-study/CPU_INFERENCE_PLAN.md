# Native CPU inference plan for the exported teammate adapter

2026-11-11. Source inspected at teammate commit `082deae3f941a686400bc85a8bd9ddaca8ee892d`. The earlier private-weights blocker is superseded: `modal/models/qwen25vl3b-lora-r16/adapter/adapter_model.safetensors` is now tracked. Its header contains 504 tensors (252 LoRA A/B pairs); target modules are language-model q/k/v/o and gate/up/down projections, rank 16, alpha 32, no bias, no vision LoRA. The adapter config names Qwen/Qwen2.5-VL-3B-Instruct but does not pin its revision. Record the actual downloaded base revision; do not claim recovery of an unknown original revision.

## First choice

Use native Transformers bf16 CPU with task-local PEFT 0.17.1, preserving the adapter unmerged. Base loading should explicitly request CPU, bf16, SDPA and low-memory initialization. Loading the base only once avoids a float32 duplicate or merge output. Install PEFT without dependencies into a task-local directory so the shared Torch/Transformers environment cannot be replaced. Root owns any installation and large downloads.

The currently inspected environment contains Transformers 4.57.6, Torch 2.15.0+cpu, Accelerate 1.15.0, safetensors 0.8.0 and HF Hub 0.36.2. The teammate used Transformers 4.57.1; record that difference rather than silently asserting identical numerics. Approximately 7.5 GB of bf16 parameters plus runtime workspace is a plausible but unverified fit under the current memory pressure. Do not promote this estimate to a measured peak-memory result.

Use `PeftModel.from_pretrained(..., is_trainable=False, low_cpu_mem_usage=True, autocast_adapter_dtype=False)`, assert an active adapter, and verify all saved adapter keys, shapes and values against `get_peft_model_state_dict`. That dictionary exposes adapter tensors only; do not materialize a full duplicate base state_dict. Require all 504 tensors and check finite/nonzero learned values. A warning followed by base-only inference is unacceptable.

For the one-letter decision, use one inference-mode forward with `use_cache=False, logits_to_keep=1`, then select the largest final-position logit among independently verified single-token A/B/C IDs. The installed Qwen2.5VL implementation supports logits_to_keep. This implements constrained one-token greedy selection without building a generation cache or all-position vocabulary logits. It is not evidence of numerical identity to vLLM.

## Preprocessing verified without torchvision

A real lightweight test succeeded using installed `Qwen2VLImageProcessor(min_pixels=784,max_pixels=1254400)` on a 640×360 PIL RGB image. It returned float32 pixel_values shape `[1196,1176]`, int64 image_grid_thw `[[1,26,46]]`, and merge_size=2, implying 299 language image tokens. No torchvision or qwen_vl_utils was needed.

The full AutoProcessor path may instantiate a video processor even for image-only use, pulling in missing torchvision. Avoid adding an unverified binary torchvision build to this Torch environment. A standalone image-only path can combine that official slow image processor with AutoTokenizer and the exact chat template. Follow installed Qwen2.5VLProcessor's image-pad expansion: each image placeholder expands to `prod(image_grid_thw)/merge_size**2` tokens. Confirm placeholder counts, token IDs, attention-mask length and image-grid metadata. Do not cast token IDs to bf16; cast only floating image inputs when needed. The model itself also casts image pixels to the vision dtype.

## Bounded execution and interpretation

Run a single actual frame first in a supervised process with a wall timeout and memory monitoring. A Python thread cannot reliably cancel a slow Torch forward. Measure preprocessing, model inference, peak memory and LoRA verification separately. AVX2 does not imply hardware bf16 acceleration, so latency may be large. Only after observing a completed trained-model prediction should a tiny closed-loop probe run. Preserve measured latency; if the answer arrives after the maneuver deadline, report rejection. A diagnostic that pauses or rescales simulation to accept the answer is a distinct experiment, not real-time performance.

## Fallback

llama.cpp officially supports Qwen2.5VL3B multimodal models and CPU projector execution. Its LoRA converter can operate from adapter weights plus base config, but the exact current adapter-name mapping and quantized-base compatibility have not been tested. A Q4 language model plus its correct vision projector would reduce memory substantially, but introduces quantization, conversion and tokenizer/template differences. Prefer native bf16 first; if it fails, verify the entire adapter conversion, runtime attachment and trained-versus-disabled outputs before claiming an actual trained-model result.

## Primary references

- [HF low-memory model loading](https://huggingface.co/docs/transformers/v4.47.1/en/big_models): meta initialization and device maps avoid duplicate randomly initialized full weights.
- [PEFT 0.17 model loading](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model): adapter low-memory loading and dtype controls.
- [PyTorch CPU bf16](https://pytorch.org/blog/empowering-pytorch-on-intel-xeon-scalable-processors-with-bfloat16/): optimized compute relies on AVX512_BF16 or AMX; speed on this AVX2 machine remains to be measured.
- [llama.cpp multimodal support](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) and [LoRA conversion source](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py): alternative route, not yet validated here.

No large model was loaded, downloaded, or run by this audit; only source/header inspection and the small image-preprocessing smoke test were performed.
