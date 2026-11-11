# Teammate VLM checkpoint and serving handoff

As of 2026-11-11, the exact source commit `1f16b7ff59aae20c2a9adfbeb4943762d4b1f5b4` and raw frame-evaluation results are available under `component-study/teammate-source/vlm-router`. The fine-tuned weights remain referenced only in the owner's Modal volume. No public checkpoint or hosted inference endpoint was found in that source. We have not launched cloud jobs, downloaded weights, or measured real-VLM closed-loop performance.

## Export from the existing owner's Modal workspace

Run these on a machine already authenticated to the workspace that owns the volume, choosing its existing Modal environment if it is not the active default. These commands download existing data; they do not create a GPU deployment.

```bash
modal volume ls vlm-router /ckpt/qwen25vl3b-lora-r16
modal volume get vlm-router /ckpt/qwen25vl3b-lora-r16/merged ./qwen25vl3b-lora-r16-merged
modal volume get vlm-router /ckpt/qwen25vl3b-lora-r16/adapter ./qwen25vl3b-lora-r16-adapter
modal volume get vlm-router /data/v1/test ./vlm-v1-test
modal volume get vlm-router /data/ood1 ./vlm-ood1
```

The merged export is sufficient for serving and includes processor/tokenizer files saved by the training script. The adapter is an optional smaller archival copy; using it alone additionally requires the exact base-model revision, which the training command did not pin. The held-out PNGs are needed for faithful frame replay; the JSON results in Git do not contain image pixels. Preserve SHA-256 hashes of every exported model, processor, tokenizer and dataset file, and confirm the resulting directory layout before serving.

Modal documents recursive download of a folder with `modal volume get VOLUME_NAME REMOTE_PATH LOCAL_DESTINATION`; `--env` selects an existing environment when necessary. No `--recursive` flag is needed for this get operation. [Official Modal CLI reference](https://modal.com/docs/cli/latest/volume).

## Serve on an existing suitable GPU host

This is a launch recipe, not a record of an executed deployment. Use an isolated environment with the teammate's vLLM 0.33.0 and its supported CUDA/PyTorch dependencies. In the directory containing the merged export:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ./qwen25vl3b-lora-r16-merged \
  --served-model-name qwen25vl3b-lora-r16-v1 \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --seed 0 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"min_pixels":784,"max_pixels":1254400}' \
  --trust-request-mm-kwargs
```

The explicit served model name is `qwen25vl3b-lora-r16-v1`; base API URL is `http://127.0.0.1:8000/v1`. Bind locally and use an existing approved connection to the host if the simulator runs elsewhere. No API key is embedded here.

The compatibility client sends the same min/max pixel bounds in `mm_processor_kwargs`. Current vLLM requires `--trust-request-mm-kwargs` to accept those per-request values; without it, requests fail explicitly. Processor limits and served-model naming are documented in the [official vLLM serving reference](https://docs.vllm.ai/en/stable/cli/serve/), and the request field is documented in the [v0.33.0 OpenAI-compatible protocol](https://docs.vllm.ai/en/v0.33.0/serving/online_serving/openai_compatible_server/).

## Choose the prompt profile explicitly

`event-cbf-driving/scripts/teammate_vlm_backend.py` exposes:

```python
TeammateV1VLMRouter(config, *, prompts_path=None, **vlm_kwargs)
```

It loads the copied exact `prompts.py` only if its SHA-256 is `0e6ea6d72a9492dc97c12ae71818aaf24d8cec3f656c8633148c8f0b913ebc5e`. It preserves the v1 SYSTEM text and state_text, including **current skill**, literal **20 m/s**, and the assumption that other vehicles are parked. It sends no additional surrounding-object state. A guarded context is reset after every success or failure; concurrent calls on one instance are refused.

The clean core VLMRouter remains unchanged and omits current-skill text. The root closed-loop runner exposes explicit clean versus teammate-v1 prompt selection. Choose teammate-v1 only as a diagnostic compatibility baseline. Actual dynamic driving is out of the v1 prompt's stated domain; changing to the clean prompt is also a distribution shift for those weights. Neither profile inherits the teammate's reported frame accuracy without new evaluation.

The compatibility client retains the current HTTP `structured_outputs.choice` plus one-token limit and strict decoding. The teammate offline benchmark instead restricts tokenizer-derived allowed_token_ids. Thus messages are exact, but decoding/workload is not bit-identical. The original benchmark also changes one random pixel, warms up, and times in-process inference; this client preserves image bytes and includes HTTP/client overhead. Report capture, model-call and end-to-end timing separately.

Before broad experiments: verify checkpoint hashes, replay the exported v1 held-out images using exact v1 messages, then run each prompt profile on separately identified closed-loop scenes. Preserve failures and front-camera blind-spot cases. All tests delivered with this adapter are mock transport contract tests, not a real-model evaluation.
