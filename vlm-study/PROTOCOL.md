# Actual trained-router smoke protocol

The new teammate commit 082deae3f941a686400bc85a8bd9ddaca8ee892d includes
adapter_model.safetensors (59,942,200 bytes), resolving the missing-model-file
block. Preserve all earlier component/scheduler experiments in their worktrees.
This is a separately identified exploratory inference/integration smoke.

Base model: Qwen/Qwen2.5-VL-3B-Instruct, pinned for this reproduction to
66285546d2b821cf421d4f5eb2576359d3770cd3. Original training did not record a
base-model revision. The newly exported adapter is BF16; no assumption that
reconstructed weights reproduce the earlier merged model's exact predictions.

First stage: verify the complete adapter tensor inventory and values after
loading, use the exact v1 prompt and verified single-token A/B/C choices, then
execute one native BF16 CPU forward on an actual captured simulator frame.
Record token counts, image and artifact hashes, elapsed time, probabilities,
memory usage and hardware. No base-only fallback, guessed output, fake server
or synthetic model result can be counted as trained inference.

This host has no NVIDIA GPU and is AVX2-only. Isolate the model in a transient
user service with MemoryMax=10G, MemorySwapMax=0, CPUQuota=400%, and at most
15 minutes runtime. Do not stop other applications or change global settings.
If the native probe cannot run within those limits, report the failure rather
than freeze the desktop or claim the trained model is unavailable.

Only if measured CPU speed permits useful closed-loop feedback should a small
physical-camera pilot follow, with actual wall-clock-paced asynchronous control,
all protections enabled, explicit CPU deadlines and no silent fallback. Keep it
separate from the original GPU claims and prior 4,608 diagnostic episodes.
Do not replace the tested physical camera/latency contracts with the teammate's
point-mass ghost renderer or inference-only timing without labeling that change.
No cloud GPU deployment or paid training is planned in this local smoke.
