# Maintained scripts

Run from `event-cbf-driving/` with the project environment active. Use the
installed `event-cbf` CLI for ordinary training, evaluation, verification and
replay. New single-critic training is
`python -m event_cbf_driving.skills.train_ppo`; no wrapper script is needed.

## Final comparison

| Script | Purpose |
|---|---|
| `integrated_eval.py` | Run the controlled comparison with source/design locks. |
| `integrated_vlm.py` | Run the smaller actual camera-model comparison. |
| `integrated_analysis.py` | Analyze runs and audit physical traces. |
| `export_integrated_results.py` | Export the recorded comparison's review tables. |
| `build_integrated_report.py` | Rebuild the recorded final PDF. |

```bash
PYTHONPATH=. python scripts/integrated_eval.py test --workers 4   --output runs/final-integration-v2
PYTHONPATH=.:scripts python scripts/integrated_analysis.py   --runs-dir runs/final-integration-v2 --results-dir runs/final-analysis-v2   --audit-traces
```

Add `--with-live` to analysis when that run directory also contains a completed
actual-model subset. Run `integrated_vlm.py --base PINNED_BASE_PATH --output
runs/final-integration-v2/actual-vlm` for that subset. It needs the recorded
compatible model environment, external base weights and substantial CPU time;
see [the final study](../../final-study/README.md).

Existing source locks must match. Use a new directory after a code change;
never edit an old lock to resume with different code. New analysis defaults to a
generated folder rather than overwriting the recorded `final-study/results/`.
The final report builder and table exporter operate on the recorded final-study
layout; reproduce frozen data with its original source snapshot and raw release.

## Model and inspection tools

| Script | Purpose |
|---|---|
| `calibrate_metadrive.py` | Recorded dual-critic policy calibration for physical tracking. |
| `trained_vlm_cpu.py` | Load the trained adapter and run actual CPU image inference. |
| `trained_vlm_drive.py` | CPU router helper and one exploratory physical drive. |
| `record_cameras.py` | Capture camera observations. |
| `render_demo.py` | Render a recorded episode as video. |
| `plot_interactions.py` | Plot a recorded action chain. |
| `phase_vision_test.py` | Tested diagnostic for image/current-skill hint sensitivity. |
| `teammate_vlm_backend.py` | Tested opt-in teammate-v1 prompt/HTTP contract. |

The last two retain tested diagnostic contracts; they are not alternative main
benchmark suites. The pinned adapter and prompt are now in `checkpoints/vlm/`.
Old study runners, redundant CLI wrappers and obsolete prototypes were retired;
see [history and recovery](../../docs/history.md).
