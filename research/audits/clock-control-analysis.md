# Paired clock-phase mechanism control

This exploratory control was specified before inspecting its aggregate outcomes. It does not amend the primary frozen protocol or select a new polling rate. The motivation is that zero-offset validation and phase validation used different physical seeds; their selected rate difference alone cannot identify a phase effect.

The fixed rates are 0.1 Hz and 1 Hz, already selected by the two original validation suites. Every comparison uses the same 40 seeds (1101–1140), three native scenarios and nominal/long/jitter latency conditions. New episodes cover zero-offset 0.1 Hz, zero-offset 1 Hz and uniform-initial-offset 0.1 Hz (1080 episodes, two workers). Uniform-initial-offset 1 Hz is reused from the fully completed main phase suite only. The phase fraction is `default_rng(seed + 72001).uniform()`, shared across scenarios, delay conditions and polling rates for that seed; seconds equal fraction/rate.

The new script verifies both original source/config/checkpoint locks and writes its own script/design/lock hashes before launching. It calls the original frozen isolation execution function with only a wrapper assigning polling phase. Runtime assertions preserve pending-margin-off and all-call-busy behavior. Phase mode is included in each new job identifier. All raw rows are retained.

Analysis is paired uniform-minus-zero within rate. For each metric, differences are averaged over the nine scenario/delay combinations within each scene seed, and 20,000 bootstrap samples resample the 40 seeds with replacement. Intervals are percentile 95% intervals. End time is physical episode duration, while wall time is execution duration under shared-machine load and is not a hardware-efficiency benchmark. Reused 1 Hz rows were run at another time under possibly different contention, so their wall-time difference is especially confounded. Success and route comparisons are not subject to this wall-clock interpretation.

Execution complete: 1080 new episodes plus 360 existing phase-suite episodes, with 360 exact pairs per rate. All original isolation/phase source, configuration and checkpoint locks and the new script lock reverified after completion. The phase suite had all 2880 unique rows before reuse. Source: `event-cbf-driving/scripts/research_clock_control.py`; data: `event-cbf-driving/runs/research-clock/`; run log: `event-cbf-driving/runs/research-clock-run.log`.

Implementation boundary: this changes the initial polling offset, not an external absolute-time clock maintained forever. The frozen controller initializes `last_rate_t = phase - 1/rate` and subsequently resets it to the actual submission time. Busy pending calls and maneuver blending can postpone submissions and shift later timing. The measured effect therefore concerns initial-offset sensitivity under the existing nonpreemptive scheduler; an ideal fixed-grid polling probability formula is only a diagnostic abstraction, not this implementation's guarantee.


## Results

| Fixed polling rate | Zero-offset success | Uniform-offset success | Uniform minus zero success (95% CI) | Zero / uniform route (%) | Zero / uniform calls |
|---|---:|---:|---:|---:|---:|
| 0.1 Hz | 360/360 (100%) | 282/360 (78.33%) | −21.67 pp [−26.67, −16.67] | 99.998 / 95.916 | 2.000 / 1.667 |
| 1 Hz | 340/360 (94.44%) | 349/360 (96.94%) | +2.50 pp [+0.83, +4.72] | 99.855 / 99.928 | 11.811 / 11.511 |

At 0.1 Hz, changing only the initial offset lost success in 78 paired episodes and gained none. Mean route completion fell 4.082 percentage points (95% CI −5.280 to −2.907), while mean episode duration rose 1.802 seconds (1.508 to 2.077). Calls fell 0.333 (−0.408 to −0.258); that apparent resource saving came with substantially worse progress.

At 1 Hz, ten pairs gained success and one lost it. Mean route rose 0.073 points (0.021 to 0.134), mean episode duration rose 0.104 seconds (0.026 to 0.179), and calls fell 0.300 (−0.378 to −0.225). Duration includes incomplete episodes; it is not a pure time-to-goal measure. No collisions occurred in either phase condition at either rate. Zero empirical collision differences and degenerate bootstrap intervals do not establish rare-event safety.

The paired experiment establishes substantial initial-clock sensitivity for the sparse 0.1 Hz reference on these exact physical seeds. The previously observed validation-selected-rate shift was confounded by differing seed sets; the present comparison removes that particular confound. It does not show that random phase is universally worse: the direction reverses for 1 Hz, and its success effect is much smaller. The defensible conclusion is rate-dependent initial-offset sensitivity under this frozen controller and nonpreemptive planner implementation. It is an exploratory mechanism result, not a novelty claim or proof of deadline-policy superiority.

Wall duration increased 0.222 seconds at 0.1 Hz (95% CI 0.096 to 0.348) and 0.055 seconds at 1 Hz (−0.238 to 0.335). Shared-machine load and reuse timing prevent an inference about hardware efficiency from those differences.

## Artifacts and reproduction

- Machine-readable results and validation checks: `event-cbf-driving/runs/research-clock/analysis.json`.
- Locked-script analysis output: `event-cbf-driving/runs/research-clock/summary.json`.
- All 1080 new raw rows: `test.jsonl` and verified lossless `test.jsonl.gz` in that directory.
- All 1440 rows including reused phase rows: `combined.jsonl` and verified lossless `combined.jsonl.gz`.
- Prelaunch lock: `lock.json`; includes script SHA, upstream lock SHAs and the full job-design hash.
- Main phase raw data was verified complete with 2880 unique IDs; every clock condition was verified to contain exactly the same 360 scenario/seed/delay keys and correct deterministic phase assignment.
- Two tests pass: phase-sensitive unique resumable IDs/design cardinality, and correct phase pairing/reset through the wrapper.

From the package directory, `PYTHONPATH=. .venv/bin/python scripts/research_clock_control.py run --workers 2` resumes without repeating completed jobs. `PYTHONPATH=. .venv/bin/python scripts/research_clock_control.py analyze` verifies locks, requires the complete phase suite and regenerates combined rows plus summary. The supplementary `analysis.json` packages the same estimates with explicit validation and interpretation metadata.
