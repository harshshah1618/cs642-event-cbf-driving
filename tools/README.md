# Experiment and reporting tools

Run from the project root with the environment installed. Use `make help` or
`python -m driving --help` for everyday commands.

| Tools | Purpose |
|---|---|
| `integrated_eval.py` | Controlled final comparison with source/design locks. |
| `integrated_vlm.py` | Smaller comparison using actual camera-model inference. |
| `integrated_analysis.py` | Compare completed runs and audit physical traces. |
| `build_integrated_report.py`, `build_milestone_report.py` | Rebuild the saved reports. |
| `calibrate_metadrive.py` | Recorded driver calibration for physical tracking. |
| `trained_vlm_cpu.py`, `trained_vlm_drive.py` | Actual CPU model inference and exploratory driving helper. |
| `record_cameras.py`, `render_demo.py`, `plot_interactions.py` | Capture and inspect driving behavior. |
| `phase_vision_test.py`, `teammate_vlm_backend.py` | Tested camera/prompt diagnostics. |

```bash
make study RUN_DIR=runs/final-v2
make analyze RUN_DIR=runs/final-v2 RESULTS_DIR=runs/analysis-v2
```

Use each script's `--help` for its options. The actual-model runner needs an
explicit base-weight path and the recorded compatible model environment; base
weights are external. The adapter and pinned prompt are in `models/camera/`.

Experiment locks check source and settings. Choose a fresh directory after a
code change; do not edit old locks to bypass the check. The full raw release is
needed to re-audit original traces. See [history](../HISTORY.md).

The report builders read preserved evidence under `reports/` and write the PDFs
there. They do not rerun driving experiments.
