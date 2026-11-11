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

## Current execution state (after independent audits, all primary tests frozen)

Current source commits:2a99d95 fixed audited algorithms;01ba8e4 report/audit code.
All three locked-selection.json files exist and verify intact. Per-condition
mappings frozen and exact-cell audit implemented by Astra in research_data_audit.py
with10 passing focused tests. Full pre-report suite226 passed (expected236 with
new audit tests). Ruff/mypy pass. Do not edit any frozen package/*.py, configs,
research_isolation.py, research_roads.py, research_phase.py, research_freeze.py.
Report scripts, README, supplementary trace script can change without altering locks.

RUNNING:
-37928: isolation test4800, workers4, >2200completed at last snapshot.
-77888: upstream road test3760, workers4, >1000completed.
-48175: phase test2880, workers4, resumed safely from490completed (old19405 killed;
 no algorithm changes). >700completed. Old frozen source still intact.
-53275: controller ablation1200, workers2, separate scripts/research_controller_ablation.py
 source/config lock.json. Same sigma-free filter for PPO and PD; four methods,
10seeds1001-1010, five delays, both controllers. Prospective subset, not new training.
-13236 actual Qwen closed-loop is COMPLETE: six episodes,9real model calls,0fallbacks,
4169tokens. Both modes stall in two_trap; deadline83.1% vs polling85.2%, other scenarios100%.
Stored runs/research-live/exploratory-single-seed. Probe took7606.6ms; file was renamed
from misleading.json to.txt (CLI produced text), at runs/research-live-probe.

Validation final: native120deadline(buffer1) episodes all100%route/no crashes,
mean1.017calls vs selected0.1Hz polling2.0calls; legacy113/120success7.825calls.
Road selectionglobal0.5Hz/deadlinebuffer1. Phase initial-offset selectionglobal1Hz;
native zero-offset0.1Hz. These are validation observations, not heldout wins.
User asked status; reported these labelled validation and ~3270heldoutcompleted.
User asked keep going until beating something or novelty, and explicitly authorized
Astra; used two Astra agents for correctness/novelty/report audits. They are idle now.

Critical fixes already applied and tests pass: no_latency_matched shares chosenbuffer;
road zero-latency route branch truly0; nominal10Hz polling capremoved; common road
speed-reference recovery including stopped/unsettled vehicle; footprint/crossing
backup; curvature cap. Added10Hz periodic_gated strong baseline; its shorter cooldown
is disclosed, not a pure deadline ablation. No heldout data was retuned.
Old road variants are preserved pre-freeze-v1..v4 and excluded.

NEXT:
1. Wait all four physical suites to completion, keep meaningful commentary<=60s.
2. Run scripts/research_trace.py ONLY after native rows for dense_highway1001long exist
 (native roughly3200completed). Generates9full traces + action/replay audits, compares
 frozen row outcomes. Need no source edits.
3. Report generator scripts/research_report.py now imports exact-cell/runtime audits,
 separates standard/stress, exports every per-delay/condition summary + entire
 validationfrontierJSON and compressed rawdata. Added controller ablation and actual
 live check sections. Needs final236test check, optional sourcequalifications.
4. Run PYTHONPATH=. .venv/bin/python scripts/research_report.py after complete datasets.
 It requires counts4800/3760/2880 pluscontroller1200 andlive6. Willcreate final PDF/MD,
 stats/audit/data under ROOT.parent/research/results. PDF marker already successfully
 ran once; do NOT rerun it. PDFskill alreadyread and user informed. Render final PDF
 withpdftoppm, inspect ALLpages (contactsheet orindividual), fix layout ifneeded.
 Plot generator now produces readable per-suite bar/errorbar pairs instead of
 overlapping scatter labels. GeneratedPDF currently wordingneedsfinalinterpretation
 against actual data (do notclaimnovelty). All results conditional/frozencontroller;
 no CARLA/SafeBench/Bench2Drive officialrun or independentpolicytraining.
5. Add optional meaningful conclusion after reading complete results. Exact paired
 seedbootstrap (40clusters jointlyenvironments/scenarios/delays), exploratory2pp/1point
 win screen explicitly NOT formal preregisterednoninferiority. Bootstrap0width notsafetyproof.
6. Commit all source/audits/reports/compresseddata. Build verifiable source+results
 archive and hash, preserve originalcheckoutsource. Final telluser concretewins and
 strongestbaselinefailures/noveltyhonesty; linkreport andbranch/worktree.
7. Memory used: MEMORY.md52-61, rollout01a1aa4b-ac68-7001-93f0-75f7b24dd0c3,
 final needs memorycitationblock last. Do not update actualmemories (notrequested).

## Completed delivery - 2026-11-11 10:48 UTC

All experimental jobs listed above are COMPLETE. The preceding RUNNING/NEXT
entries are historical execution notes, not current pending work.

- Primary validation 2,928 and held-out 11,440 rows, with exact cells and frozen
  source/config/checkpoint selections verified. Separate controller ablation
  1,200 rows, clock control 1,080 new rows plus 360 reused, actual Qwen six episodes
  and nine calls with zero fallbacks. Spatial candidate: 120 development rows only.
- Achieved scoped win: random-initial-offset deadline 1.025 calls versus 2.7694
  condition-tuned polling (63.0% fewer); success 360/360 versus 357/360, zero crashes
  both. Native success is no crash and at least 99% route. Not universal dominance:
  the original suite fails the exploratory success-tolerance screen; TTC and
  matched no-latency remain competitive; road results include crashes/timeouts.
- Matched clock control: 0.1 Hz success 100% to 78.33% under random initial offset,
  difference -21.67 pp, seed-cluster CI [-26.67,-16.67]. At 1 Hz effect reverses.
  This is an exploratory implementation finding, not established novelty.
- Full pytest 243 passed in 25.36 seconds; Ruff clean; mypy clean in 50 source
  files. Nine full action traces replay with exactly zero final-state error.
  Original checkout's 102 Python/config/checkpoint files still match snapshot.
- Final seven-page PDF rendered and all pages visually inspected. Embedded fonts
  avoid the host's broken Helvetica substitution. Tables retain their headings.
- Raw gzip row counts/unique IDs and hashes verified; read-only restore utility
  successfully restores identical data and refuses differing existing files.
- Results, complete condition tables, all validation frontiers, source locks,
  runtime inventory, live evidence and development-only spatial data are preserved
  under research/results. Plain-language assessment: research/ASSESSMENT.md.

No additional experiments are required for the authorized benchmark-win goal.
No remote push, official leaderboard submission, new theorem or established
algorithmic novelty is claimed.
