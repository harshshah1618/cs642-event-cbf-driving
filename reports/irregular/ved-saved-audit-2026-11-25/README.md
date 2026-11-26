# Saved Qwen run audit

On November 25, access to Modal workspace `ved29022004` made the original
`cs642-irregular` volume available. All 16 final-run files were downloaded
individually and checked against their remote byte sizes. They contain exactly
400 episodes: eight scenes, each with maps 1000-1049 exactly once. The saved PPO
checkpoint is byte-identical to the checkpoint used for the local controls.
The original 20,000-patch DINOv2 bank and its metadata were also retrieved.
No cloud jobs, model inference or training were started.

## Corrected historical collision counts

The final-step crash flags are false in every historical episode, but the
original cumulative native costs support an audit under the published pinned
simulator recipe. The cost function charges one unit on a vehicle collision,
object collision or road departure; human collisions have no native cost.
Road departure and human collision terminate immediately, and the evaluator
breaks on termination. Therefore:

- Cost greater than one proves a vehicle/object collision: road departure alone
  can contribute at most one unit before termination.
- Zero cost, combined with a false final human-collision flag, rules out the
  three collision types counted in the original table under this recipe.
- Cost exactly one without a final collision flag would remain ambiguous.

There are six positive-cost episodes, with costs 64, 68, 63, 76, 76 and 2. None
has cost exactly one. Thus every historical episode has a resolved collision
count under these assumptions, although the earlier contact type and timing
cannot be recovered from the final flags.

| Scene | Local normal collisions | Local careful collisions | Historical Qwen collisions inferred |
|---|---:|---:|---:|
| Billboard | 1/50 | 0/50 | 1/50 |
| Clean | 1/50 | 1/50 | 1/50 |
| Flagger | 1/50 | 0/50 | 1/50 |
| Lane closed sign | 1/50 | 0/50 | 1/50 |
| Pedestrian in lane | 0/50 | 1/50 | 0/50 |
| Pedestrian on shoulder | 1/50 | 0/50 | 1/50 |
| Rain without added obstacle | 1/50 | 1/50 | 1/50 |
| Wet road closure | 16/50 | 2/50 | 0/50 |

All arms use one trained PPO replica and the same map IDs. The first two columns
are the new local CPU controls with per-step collision logging. The Qwen column
is inferred from historical GPU-era records using the rules above, not a fresh
GPU rerun. The original records do not independently dump the runtime environment
config; the inference assumes they used the published pinned recipe. This is
not a new fully matched hardware/runtime experiment or an unseen-map test.

The six historical collision episodes are map 1040 in billboard, clean, flagger,
lane-closed and pedestrian-shoulder, plus map 1029 in rain-clean. All 50 wet-road
Qwen episodes have zero native cost and false final human flags. All 50
pedestrian-lane Qwen episodes do as well.

The original **23/50 STOP violations** are verified directly from the restored
episode records, versus 50/50 for both local controls. This does not show perfect
compliance: 46% of the Qwen episodes still pass while STOP is showing. Wet-road
Qwen episodes average approximately 17% route completion and often wait at the
closure. Local object-scene route percentages are also partial-route measures;
the test ends after passing the object or reaching its time/termination limit.

## Limits and checks

A 24-episode saved-advice replay pilot matched all strict checks in nine cases
but drifted in the others, including collision-sensitive map 1040. Its collision
counts are not used here. No observed result was replaced by an unmatched replay.
The original positive costs and simulator termination rules provide the count
inference instead. The audit tool preserves an unresolved interval if an
ambiguous cost-one episode is supplied.

`original-episodes.json.gz` preserves all 400 original rows. `audit.json` records
the per-scene counts, collision map IDs, full input hashes and pinned native-rule
checks. `comparison.csv` contains the table above. `manifest.json` records the
checkpoint and bank hashes, retrieval provenance and replay limitation.

Reproduce with the pinned MetaDrive environment and the downloaded JSON files:

```bash
PYTHONPATH=. python tools/irregular_audit_saved.py \
  --source runs/ved-workspace-audit-20261125/final-episodes \
  --output runs/ved-workspace-audit-20261125/audit-recomputed.json
```

The shared Modal profile is `ved-shared`; the personal default profile was not
changed. GPU compute remains paused. Future fresh inference needs explicit
compute authorization; the original detector bank no longer needs rebuilding.
