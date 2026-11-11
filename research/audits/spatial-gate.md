# Spatial usefulness gate: development-only exploratory revision

2026-11-11. This candidate was motivated by a primary-study failure mechanism,
not specified in the original protocol. It is not part of the primary held-out
comparison and no new spatial-gate held-out experiment was launched. The
coordinator stopped this optional replication after the independent random-phase
study met the task objective. All results below use development seeds101-108.

## Fixed formula and implementation

`event-cbf-driving/scripts/research_spatial_gate.py` injects a process-local
replacement for the controller's imported lane-opportunity function, using the
existing immediate-benefit trigger. Injection is explicitly restored before
every job and in a finally block. The controller, PPO bank, scripted router,
filter, admission checks and original experiment code are unchanged. Recorded
method/ID fields retain the true spatial-gate identity.

For each observed lead vehicle, let available gap equal center distance minus
combined half-lengths,8m clearance and1m margin. Let H equal the delay upper bound
plus2s transition time. Lane achievable-speed proxy is the minimum over leads of
20m/s, `(max(gap,0)+lead_speed*H)/(H+2s)`, and
`sqrt(lead_speed²+2*3m/s²*max(gap,0))`. This approximates terminal-headway and
stopping-distance constraints; it is not a safety certificate or a novel method.
Trigger when alternative proxy exceeds current proxy by1m/s, ego is settled,
fresh adjacent front/rear gaps are clear, and the actual scripted router would
recognize a current-lane slow lead (<15m/s within200m). Delayed responses still
undergo the existing independent fresh admission check.

Formula/constants were fixed before the120-episode development run and were
not adjusted in response to validation results. Only current observed geometry
and velocity are read; no scenario parameters or future scripts are accessed.

## Development results

Every row below covers the same120 development cells:3 scenario families,
8 seeds,5 delay conditions. Original comparator rows are reused from completed
original validation, with frozen global choices; no baseline was retuned.

| Method | Success | Crashes | Route | Calls | End time |
|---|---:|---:|---:|---:|---:|
| Spatial gate |100%|0|100%|1.333|14.083s|
| Frozen0.1Hz polling |100%|0|100%|2.000|14.850s|
| Deadline buffer1 |100%|0|100%|1.017|14.202s|
| Immediate usefulness |100%|0|100%|1.083|14.018s|
| TTC |100%|0|100%|0.958|14.459s|

The spatial candidate is competitive with periodic polling on development data,
but immediate usefulness has fewer calls and lower mean end time. These are
development comparisons, not an independent validation of improvement, a clean
held-out win, or evidence of novelty. No claim about performance on fresh
seeds5001-5040 is supported; the planned2400-episode test was not launched.

## Verification and artifacts

Five focused unit tests pass: farther/slower adjacent lead can improve spatial
progress; blocked adjacent lane rejected; empty lanes/router-known no-op do not
request; unsettled maneuver does not reverse; process-local injection restores
the original function for control jobs. Ruff passes for both new source files.
All120 expected development job IDs/metadata were checked exactly.

- Source: `event-cbf-driving/scripts/research_spatial_gate.py`.
- Tests: `event-cbf-driving/tests/test_spatial_gate.py`.
- Raw rows: `event-cbf-driving/runs/research-spatial/validation.jsonl`.
- Comparisons/constants/hashes: `runs/research-spatial/validation-analysis.json`.
- Exact candidate source snapshot: `runs/research-spatial/source.py`.

The script includes a future explicit freeze/test workflow with source/constants/
original-lock verification and paired seed-cluster analysis, but that workflow
was not invoked. The archived validation-analysis records candidate source,
validation and original lock SHA256 digests. No frozen original source was edited.
