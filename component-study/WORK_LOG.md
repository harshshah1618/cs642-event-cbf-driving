# Component study completion record

Completed November 11, 2026 on codex/component-contributions. The original dirty
harsh-testing tree and prior codex/latency-aware-research worktree were preserved.

- Separated the longitudinal CBF constraint from the stopping backup and common
  boundary guard. Kept the validity monitor fixed. Added per-stage corrections.
- Implemented a complete 16-setting controller/CBF/braking/admission factorial
  under a shared 2 Hz query policy with inflation disabled for both controllers.
- Executed 192 disjoint pilot episodes, then froze runtime source, configs,
  checkpoints, protocol, exact job design and pilot hash before 4,608 main runs.
- All 4,608 exact cells are retained, with 48 matched scene/seed initial states
  across treatments and delays. Twelve traffic-seed clusters, not independent
  policy-training replications. Achieved acceleration was recorded in every run.
- PPO full: 0/288 collision episodes and 228/288 route successes. With all three
  studied traffic protections off: 280/288 collisions and 8/288 successes.
  CBF alone: 42 collisions; braking alone: two; full minus admission: two.
- Adding CBF to braking+admission did not change observed success/crash counts,
  while improving mean achieved jerk RMS and worsening mean peak jerk. PPO also
  had lower RMS but higher peaks than PD under full protection. No blanket
  comfort superiority or universal component necessity is claimed.
- PD full: 0/288 collisions and 226/288 successes. Without traffic protections,
  PD had fewer collisions than PPO. Results do not establish a general PPO win.
- Full-PPO state-rule router: 91/96 route successes; deliberately copy-current:
  46/96. Zero collisions in both, showing that safety cannot supply useful
  maneuver decisions. Success is no collision and >=99% route; some successes
  occur at timeout, and exact-arrival/timeout counts are preserved separately.
- Eleven fixed representative re-recordings replay with zero final-state
  error and match main outcomes/physical jerk/stage metrics exactly. Validity
  monitor changed actions on zero main ticks; no emergency latches or aborts.
- Recovered teammate commit 1f16b7ff59aae20c2a9adfbeb4943762d4b1f5b4 by exact SHA
  after branch-list checks failed to reveal it. Communicated and documented the
  correction. Pinned source/results copy and independent source audit included.
- Verified saved static prediction arithmetic: 300/300 ID, 50/50 round-2,
  252/420 OOD for fine-tuned model. This is not a fresh model inference check.
- Implemented strict camera/VLM interface and opt-in exact-v1 prompt adapter.
  Both profiles completed actual-render/HTTP/async-control contract drives,
  with twelve programmed replies total, zero real-model/fallback calls. Camera
  capture leaves physics state/time unchanged. Capture/model timing is separate.
- Actual fine-tuned closed-loop arm remains pending: weights are only in the
  teammate's private Modal volume, with no published export/endpoint or local
  credentials. No cloud resources or paid training were launched. Export,
  serving, prompt caveats and ready-to-run commands are in TEAMMATE_HANDOFF.md.
- Validation: 324 pytest tests passed in 30.50 s; Ruff clean; mypy clean in
  52 package files, compatibility helper checked separately. Independent Astra
  audit recomputed every main effect/interval/interaction and checked all cells.
- Gzip rows verified lossless and unique. Original 102 Python/config/checkpoint
  files verified unchanged against snapshot50bcbde. The final archive must pass
  ZIP integrity, manifest hashes, fresh restore and frozen-lock verification.

Primary deliverables: results/EXPLAINED.md, report.md, statistics.json, all 24
condition tables, cells.csv, raw rows, lock, action traces and camera fixtures.
Original scheduler study remains separate; no real-VLM driving, official
leaderboard result, new theorem, or established algorithmic novelty is claimed.
