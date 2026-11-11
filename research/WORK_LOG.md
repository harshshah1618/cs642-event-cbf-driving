# Active implementation and experiment log

Original source preserved: /home/harsh/Projects/cs642-event-cbf-driving (harsh-testing).
Worktree: /home/harsh/Projects/cs642-event-cbf-research, branch codex/latency-aware-research.
Baseline source snapshot commit 50bcbde. Current scheduler commit 4cd3c51 plus ongoing
road forecast backup / phase robustness changes. Existing environment is symlinked,
not committed. Always run package commands with PYTHONPATH=. to use this worktree.
No subagents (not explicitly authorized); process pools run physical simulations.

Implemented: stack/scheduling.py, StackConfig trigger options, controller integration,
road_planning.py generic road-relative PID/IDM + finite-horizon lane/speed planner,
footprint-aware cut-in observation, fresh lane admission, short 2D forecast brake
backup (approximate, no safety proof). Added original-scenario and upstream road
studies, validation-only global and per-condition polling selection, phase test.

Running sessions as of implementation log creation:
- 58599: isolation validation, 1560 jobs, near completion; writes selection.json.
- 94685: final road validation, 1152 jobs, 4 workers.
Previous road pilot/validation variants retained under runs; pre-freeze versions
must be excluded from final summaries. No held-out test has been launched yet.

Next: finish validation, freeze selections + source hashes, launch isolation test
4200 jobs (4 workers) and road test3360 jobs (4 workers); phase validation216 jobs
then phase test2520 jobs (prefer sequential with isolation to limit resource use).
PD controller ablation1050 jobs if valuable. All results in package/runs/research-*.
Initial tests220 passed; added road unit tests now pass9/9; run full suite again.
Ruff/mypy passed before final backup edits; rerun. Fixed the pre-existing three
lint issues in new scripts in this worktree only.

Pilot: native deadline buffer1 matched 100% success with ~1 call versus ~2 for best
0.1Hz polling; benefit/TTC also competitive. Smaller deadline buffers missed some
merge opportunities. High-rate polling sometimes worse than sparse polling. This
could be timing aliasing; random-clock-phase stress is prospective and necessary.
Road backup pilot48 episodes: benefit8/8, periodic8/8 success, zero crashes for both;
deadline7/8, never6/8; upstream Expert6/8 with2 crashes. These are pilot observations,
not held-out results or SOTA claims. Algorithm must now remain frozen for final road
validation/test unless a material correctness bug requires a clearly versioned rerun.

Need final artifacts: raw audited rows+hashes, paired cluster-bootstrap intervals,
all baseline frontiers, source/provenance snapshot, Markdown and PDF report, code
commits, reproducibility commands. Use PDF skill (already read) and load workspace
runtime (already loaded); PDF authoring marker must run before first PDF authoring.
Skill path /home/harsh/.codex/plugins/cache/openai-primary-runtime/pdf/26.1104.22356/skills/pdf.
PDF runtime Python /home/harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.
User explicitly steered: keep going until beating something or finding novelty.
Do not claim novelty from a benchmark win; do not stop on merely fewer calls if
safety/progress deteriorates. No CARLA/SafeBench/Bench2Drive actual runs yet.
