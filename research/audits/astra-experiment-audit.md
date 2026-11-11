# Independent pre-held-out experiment audit

Audit date: 2026-11-11. Reviewed protocol, work log, scheduler/controller integration,
road geometry/controller, and all three experiment runners. No implementation was
changed by this auditor. Findings describe the pre-fix source; coordinator has
acknowledged the two necessary corrections below. No held-out results were inspected.

## Required before interpretation/freeze

1. **Matched latency ablation.** Native `research_isolation.execute` sets buffer
   0.5 for `no_latency`, while `deadline` selects buffer (currently 1). This is not
   an identical trigger with latency removed. Roads has the same buffer mismatch
   and `road_request_due` uses nonzero latency in its route-invalid branch even
   for `no_latency`. Introduce a separately identified matched ablation using the
   selected deadline buffer and zero latency in every branch. Preserve original
   validation diagnostic rows; do not relabel them. Coordinator plans this fix.
2. **Road polling frequency cap.** Runner enforces `t-last_call >= 0.5` for every
   scheduler, so nominal 10Hz polling cannot exceed 2Hz even with no delay. This
   weakens/confuses the frequency frontier. Coordinator plans uncapped periodic
   polling and a new road validation version. Preserve capped development rows.
3. **Freeze conditioned choices explicitly.** Current test runners recalculate
   per-condition polling rates from validation at launch rather than reading a
   complete frozen selection manifest. This uses validation only, so is not test
   leakage, but freeze the actual condition-to-rate mappings and hash inputs before
   launch. Verify exact expected job IDs/counts, not just unique completed IDs.

## Measurement and inference limits

- Native rows have calls and episode wall time, not planner/scheduler CPU time.
  Calls support a call-budget claim only. Road `planner_cpu_s` and `scheduler_cpu_s`
  use `perf_counter`: elapsed time inside code regions, subject to contention;
  observation, fast control and forecast backup are excluded. Do not call these
  total compute or process CPU time. Include shared fast-loop cost for practical
  efficiency claims. External reference controller timing has different boundaries.
- Native event policies share geometry access; roads share observations and fresh
  controller/backup among schedulers. Benefit is a strong necessary comparator.
  Reference ExpertPolicy/IDM do not share this control architecture or necessarily
  the same representation. Their results are external agent references, not a
  clean scheduler ablation or a camera-agent comparison. IDM is explicitly set
  to 40km/h (default class is 30), so label it as configured IDM.
- Bootstrap paired seeds across methods and delays. Safe/generalization maps are
  identical for each seed: cluster across both environments when pooled, or report
  environment strata; do not imply 80 independent road geometries from 40 seeds.
  For native scenarios conservatively resample seed jointly across families and
  delays, or justify independent family-seed scene randomization. Episodes and
  delays are not independent replications. Zero collisions does not establish
  rare-event safety; ordinary bootstrap intervals become degenerate at zero.
- Do not declare matched safety solely from a nonsignificant difference. Report
  paired differences/intervals and any prospective noninferiority margin. Selecting
  one favorable condition among many needs explicit exploratory labeling; present
  all conditions and selected opponents. A benchmark win does not establish novelty.
- Phase study is useful and uses disjoint seeds. Phase is initial polling offset;
  actual controller resets last-rate time on successful submission, so busy/blend
  periods shift future ticks. Describe this as initial-phase randomized polling,
  not a wall-clock periodic schedule with missed ticks skipped. Uniform phase
  delays the first low-rate request up to one period; document that convention.
- All experiment resume keys omit code/config hashes. Never resume into old files
  after algorithm/config changes. Version output directory and hash source,
  checkpoint files, resolved configs, selections, and installed dependency source.

## Geometry/delay observations

Road pending plans are tied to source road identity; old-road responses are
rejected and lane changes get a fresh admission check. Current-lane speed updates
remain under a fresh IDM/forecast brake controller. This is defensible delayed
high-level planning, not delayed open-loop motor actuation. Lane admission uses
constant-speed samples and nearest-front/rear abstractions; forecast backup uses
short constant-velocity footprints. Neither gives a safety certificate. Curved
road projection and extrapolation beyond lane end are approximations, common to
all schedulers; no direct array/lane-identity correctness defect was found.

An initial suspicion about native immediate same-skill replies was **withdrawn**:
`metadrive-study.yaml -> metadrive.yaml` sets `pending_margin_on_all_calls=true`,
so the compatibility shortcut is inactive in this experiment.

## Checks actually executed

- Native validation snapshot: 1,560 rows and 1,560 unique IDs.
- Road validation snapshot while running: 964 rows and 964 unique IDs; each
  environment/seed has a single map hash across observed methods/delays. Eight
  unique geometries observed; safe/generalization hashes agree by seed.
- All installed MetaDrive Python files with distribution RECORD hashes match
  those hashes; no local Python modification detected. Installed VCS metadata
  pins commit `fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`, version 0.4.3.
  This checks installed-source consistency, not remote release authenticity.
- Runners instantiate actual `MetaDriveEnv` / `SafeMetaDriveEnv` and explicitly
  set `start_seed=seed`, `num_scenarios=1`, then reset with that seed. They do not
  substitute scripted straight-road adapters in the road study. Save resolved
  defaults too: logged config currently contains overrides only.
- No held-out scene was launched by the auditor; no algorithm was edited.

Disposition: fix/version the matched ablation and capped road polling, freeze
full selections/provenance, and then proceed. Remaining items chiefly constrain
measurement/reporting. Continued work should accept a negative or mixed outcome
rather than retune against the held-out set.

## Follow-up review after coordinator fixes

Reviewed revised source before held-out launch, without algorithm edits:

- Matched no-latency methods now carry the selected deadline buffer; roads zero
  effective latency before the mandatory-route branch. Original validation rows
  retain their diagnostic identity. The original attribution defect is resolved.
- Roads periodic/conditioned/gated polling bypass the 0.5-second event cooldown.
  This resolves the unintended cap; coordinator is rerunning validation in a new
  version. Process CPU and explicitly named elapsed measurements, total custom
  control sections, and resolved environment config are now recorded. Reference
  agent `control_process_s` remains zero because its complete policy timing is in
  `planner_process_s`; reporting must account for those distinct fields.
- New stale-speed recovery handles a settled vehicle whose blocker clears.
  **Remaining counterexample:** `RoadView(0, 0, (LaneView(0),), settled=False,
  speed_reference=0)` returns false for benefit, TTC, deadline and no-latency,
  despite `plan_maneuver` returning `(0, 11.11)`. A stop selected/applied during
  lane acquisition can leave a vehicle unable to settle without forward motion.
  The recovery condition should permit speed-only replanning before settlement,
  or physical reachability of this state needs to be ruled out. Coordinator was
  notified immediately. The current settled-only unit test misses this case.
- Gated10Hz is a useful additional opponent but differs from event schedulers in
  cooldown: native event policies use 28 ticks, roads use 0.5 seconds, while gated
  polling uses 0.1 seconds (subject to pending/blending). It isolates usefulness
  from unconditional polling, not solely deadline timing from usefulness. Claims
  attributing saved calls specifically to the deadline rule need a matched-cooldown
  comparison or an explicit limitation. A fixed10Hz point is not a tuned gated
  frontier; do not call it the strongest possible gated polling opponent.
- Verified loaded config has `polling_phase_s=0` and all-call latency occupancy
  enabled. Synthetic clock checks over1,001 ticks with two blocked intervals gave
  identical submission indices for old `-1e18` and new `-1/rate` initialization
  at every candidate rate (0.1,0.25,0.5,1,2,10Hz). This supports default-phase
  behavioral equivalence; it is not a fresh physical-episode replay.

Full conditioned selections/source hashes still need the coordinator's planned
freeze. No held-out rows were examined in this follow-up.
