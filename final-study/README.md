# Final integrated comparison

Start with final-report.pdf (seven body pages plus references) or EXPLAINED.md. This is the single proposal-aligned evaluation, not a merger of old favorable numbers.

Source branch: codex/final-integrated-evaluation, based on 6090300. Original harsh-testing source is preserved. Source locks, full physical traces, raw response records and images belong to this evaluation. No official FASIONAD/CERT port or safety theorem is claimed.

## Files

- final-report.pdf and final-report.md: measured final report.
- PROTOCOL.md, selection.json and proposal.pdf: design, development choice and original scope.
- results/controlled.json: all 15 methods, nominal/stress separation, paired contrasts and eight-cluster descriptive intervals.
- results/actual-vlm.json:36 integration episodes,18 model-enabled,15 actually queried and22 real predictions.
- results/per-episode.csv and all-conditions.csv/md: reviewable tables.
- results/*trace-audit.json and audits/: action/plant integrity and independent reviews.
- verification.json and environment.json: checks, versions and preservation evidence.

## Reproduction

The release ZIP restores the event-cbf-driving/runs/final-integration paths. Those raw runs are intentionally outside Git; obtain the complete release, not only a source clone. Base weights (7.51GB) and Python environments are external; trained adapter and PPO checkpoints are included. Analysis needs no model loading. Embedded report fonts and their license are included.

From event-cbf-driving in the recorded project environment:

```bash
PYTHONPATH=. python scripts/integrated_eval.py test --workers 4
PYTHONPATH=.:scripts python scripts/integrated_analysis.py --with-live --audit-traces
PYTHONPATH=. pytest -q
```

Completed frozen runs are hash-checked and resumed without overwriting. Changing source/design requires a separately versioned directory. The first command does not repeat already complete drives. For fresh actual-model runs, use scripts/integrated_vlm.py --base PINNED_BASE_PATH --output NEW_DIRECTORY, with task-local PEFT 0.17.1, P3D_DISPLAY=p3headlessgl and LP_NUM_THREADS=1. CPU model runs need roughly 10 GB and tens of minutes;45s is an admission deadline, not cancellation of a native Torch forward. The full user service also had an external 90-minute runtime cap. Actual timing varied with concurrent CPU work.

Build the PDF using a Python environment with ReportLab 4.4.9 and Pillow:

```bash
python scripts/build_integrated_report.py
```

The complete release is delivered under the original checkout's research-delivery/final-integrated directory. No repository push, merge, external message, or cloud job was performed.
