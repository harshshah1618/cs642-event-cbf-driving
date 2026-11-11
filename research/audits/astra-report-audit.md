# Statistical and artifact audit addendum

2026-11-11; source review of `research_report.py` and `research_freeze.py` while
studies run. No held-out outcome inspection or implementation edit was performed.

## Correct

- Paired differences use identical scenario/environment, seed, delay keys, then
  average within seed and bootstrap seeds jointly. This correctly preserves
  dependence across road environments sharing geometry and across delays.
- With complete balanced cells, the cluster mean and reported episode mean agree.
  Ten thousand seeded percentile bootstrap draws are reproducible.
- Declared counts are arithmetically consistent: validation2928; held-out
  isolation4800, roads3760, phase2880. Reference road agents occur at zero delay.
- Zero-event bootstrap degeneracy and conditional evaluation of frozen training
  are acknowledged. Native calls are not labeled measured CPU savings.

## Required report/artifact corrections

1. `Upstream road stress` currently includes zero-injected-delay rows. Separate
   standard and latency-stress results and export per-delay comparisons; the
   prospective protocol requires this separation.
2. Export the complete validation polling-rate/deadline-buffer frontier and
   per-condition results. Current figures contain selected held-out points only;
   raw compressed rows alone are not the promised frontier report. Aggregating
   delays can conceal condition-specific reversals.
3. Validate the exact expected Cartesian job set, one row per method/scene/seed/
   delay, and values against frozen selections. Counts plus unique row IDs do
   not establish this. `paired` silently overwrites duplicated cells having
   different values/IDs. Executed synthetic check: adding a second deadline
   row for one existing cell was accepted and changed call difference to49.5.
   Reject duplicate cells before dictionary construction and check expected
   methods, seed ranges, conditions, delays and parameter values explicitly.
4. Label `conditional_call_win` an **exploratory screen**, not a preregistered
   noninferiority finding. Its2pp/1point margins were not in the original
   protocol, collision criterion uses only observed mean, and intervals are
   unadjusted across comparisons. The current text does not explicitly say
   exploratory/post hoc. Present intervals and harms regardless of screen result.
5. Upcoming phase test still obtains deadline buffer from mutable isolation
   `selection.json`. Phase lock does not include that inherited value or source
   lock hash. Freeze/verify this dependency before launch. Avoid editing the
   shared freeze module already included in a running study's source hash;
   a phase-specific dependency manifest is sufficient if checked on launch.

## Provenance and presentation qualifications

- `verify` checks hashed repository files/checkpoints and validation rows. It
  records package versions but does not recheck runtime version or installed
  package contents. Scope `source_verified` accordingly, or supplement with
  runtime provenance verification. The independent earlier audit found installed
  MetaDrive Python files matched package RECORD; that is separate evidence.
- Model metadata used by `load_skill_bank` is embedded in `.pt`; auxiliary JSON
  files are not loaded there. Current checkpoint hashes cover loaded weights.
  A newly added optional RETURN_RIGHT.pt would alter loading but is absent from
  the original hash list: exact checkpoint file inventory should also be checked.
- Reference control totals live in `planner_process_s`, whereas their
  `control_process_s` is zero. Do not compare that zero to custom full-control
  CPU. Section timing is nested inside custom control totals; do not sum twice.
- Fixed seed-balanced strata support present bootstrap. If incomplete or unequal
  cells are ever admitted, equal-cluster estimands and episode means differ;
  reject incomplete data rather than silently adjusting weighting.
- Final PDF must still be rendered and visually checked; code inspection does
  not verify table wrapping or overlapping frontier labels. Citations included
  in the report were not bibliographically verified by this audit.

The prior mid-transition zero-speed recovery defect has been corrected by
removing the settlement prerequisite, with coordinator-reported regression
coverage. Faster gated polling is explicitly described as a stronger comparator,
and causal latency attribution is restricted to the matched latency ablation.
