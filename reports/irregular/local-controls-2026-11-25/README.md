# Local irregular scene results

**Later November 25 update:** shared-workspace access restored the 400 original
Qwen episode records and detector bank. The [saved-run audit](../ved-saved-audit-2026-11-25/README.md)
verifies the STOP result and reconstructs historical collision counts under the
pinned native rules. References below to missing access describe the earlier
state of this local-control study.

Completed November 25, 2026. This is a new local CPU comparison of two controls,
using the already-published maps 1000-1049. It does not rerun Qwen or PatchCore,
and is not a new unseen-map evaluation. Each cell below contains 50 episodes;
there are eight scenes, two controls and 800 episodes in total, with one saved
PPO tracker (training seed 0).

Both controls use the same rule planner, PPO checkpoint and collision protection.
The careful control enables the existing 0.7x speed / 1.5x headway mode from the
start. It has no learned semantic decisions or model calls. Python 3.11, NumPy
1.26.4, Torch 2.8.0 CPU, Gymnasium 0.29.1, SB3 2.3.2 and the pinned MetaDrive
source are recorded in `manifest.json`. CPU results can differ from historical
GPU runs; these counts do not retroactively replace Ved's original episodes.

| Scene | Normal collision episodes | Careful collision episodes |
|---|---:|---:|
| Billboard | 1/50 | 0/50 |
| Clean | 1/50 | 1/50 |
| Flagger | 1/50 | 0/50 |
| Lane closed sign | 1/50 | 0/50 |
| Pedestrian in lane | 0/50 | 1/50 |
| Pedestrian on shoulder | 1/50 | 0/50 |
| Rain without added obstacle | 1/50 | 1/50 |
| Wet road closure | 16/50 | 2/50 |

**The original final-step method would count 9 versus 0 wet-road collisions on
these exact new runs.** Accumulating contact flags across the episode instead
counts 16 versus 2. Careful mode therefore helps this comparison, but is not
collision-free. Its two wet collision maps (1027 and 1030) do not collide under
normal control. Repeated contact steps count as one collision episode.

Both controls pass the flagger while STOP is showing in 50/50 episodes. Constant
caution does not reproduce Ved's author-reported 23/50 STOP violations with
Qwen escalation. Those Qwen aggregates still need raw-episode verification.
Passing STOP is a cumulative event; it is separate from the broken historical
collision metric and does not establish correct stopping/restarting in every
remaining episode.

Careful mode also causes a pedestrian collision on map 1036 that normal control
avoids. Native benchmark cost is zero in that episode because this simulator's
cost function does not charge human collisions. Collision flags must therefore
be reported separately from native cost. Neither mode is uniformly safer.

On the clean/rain-only 30-second tasks, mean route progress is 65.1% with normal
control and 49.1% with careful control. Without camera inference, a rain overlay
does not enter either policy, so clean and rain-only trajectories match. This
illustrates the progress cost of cautious driving, not visual weather robustness.
Object scenes end 20 m after passing the placed object, or at termination or
the step budget; their route percentages are not destination success rates.

The separate 1,900-drive SafeMetaDrive study was recomputed from its saved raw
episodes during this audit. All 14 group summaries and paired bootstrap results
match the committed analysis exactly. That study remains separate from this
local control experiment.

## Evidence and reproduction

`episodes.json.gz` contains all 800 episode records, including cumulative
collision flags, contact timesteps and end reasons. `summary.json` and
`results.csv` contain the numbers above. `manifest.json` records source hashes,
the checkpoint hash and runtime versions. `verification.json` records coverage
and validation checks; `SHA256SUMS` protects the machine-readable evidence files.

From the repository root with the recorded runtime and restored PPO checkpoint:

```bash
PYTHONPATH=. python tools/irregular_local_eval.py \
  --checkpoint runs/safemetadrive/skills-1m-v1/skills-seed0/ppo.zip \
  --output runs/irregular/local-controls-new --workers 4
python tools/irregular_closedloop_report.py \
  runs/irregular/local-controls-new/episodes --json runs/irregular/local-controls-new/summary.json
```

This launches local CPU simulations only. No cloud jobs or training were started.
The corrected evaluator labels rows `irregular-episode-v2`. The summarizer marks
legacy collision totals unverified rather than treating final-step flags as
episode counts.

GitHub branches, release assets, pull requests/issues and Actions artifacts were
checked. The latest Actions artifact contains the older nine-case physics CI
evaluation, not the 800 irregular episodes. The only release is the November 20
SafeMetaDrive handoff. Ved's original irregular episodes remain referenced at
Modal volume `cs642-irregular`, path `closedloop/final-qwen3vl-8b`, which is not
available in the current Modal account. The linked Claude artifact requests
sign-in. No message was sent to Ved.

The volume owner can download the existing episode rows without starting jobs:

```bash
modal volume get cs642-irregular closedloop/final-qwen3vl-8b ./final-qwen3vl-8b
```

These historical episode rows can audit cost, STOP events, model calls and
coverage. Earlier collision identities require saved per-step traces or a
separately labeled rerun with the corrected evaluator; final flags alone cannot
recover them.
