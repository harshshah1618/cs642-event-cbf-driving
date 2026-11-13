# Trained camera router assets

This folder retains the teammate's trained LoRA adapter, training summary,
base-model metadata and exact pinned prompt needed by the current camera tools.
It replaces the historical `vlm-study/model/` and copied teammate-source paths.
No model or prompt bytes were changed during the move.

- `adapter/`: adapter configuration and weights.
- `prompts.py`: pinned teammate-v1 prompt.
- `train_summary.json`: original training summary.
- `base-model.json`, `base-shards.json`: original base-weight provenance.

Base weights remain external. Use `scripts/integrated_vlm.py --help` for the
actual-model comparison and `scripts/trained_vlm_cpu.py --help` for direct
inference. The adapter and prompt hashes are verified by the existing backend.
Older integration evidence is recoverable from commit `0dbffe7`; see
[history](../../../docs/history.md).
