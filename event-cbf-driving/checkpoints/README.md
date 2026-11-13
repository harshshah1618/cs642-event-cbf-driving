# Skill checkpoints

## What is actually trained

| Skill | Lane target | Status | Current state-dict fingerprint |
| --- | --- | --- | --- |
| `CRUISE_RIGHT` | right (`target_lane=0`, `y_target=0.0`) | distinct trained policy | `9706785d81623499e66f1b3b7d873c4f815832dff6e068feaa50317d319eb9dd` |
| `OVERTAKE_LEFT` | left (`target_lane=1`, `y_target=4.0`) | distinct trained policy | `1e44eba4c6911d19093f2cdfebd4533a02a08b3c7c910127444fec8fa4317920` |
| `RETURN_RIGHT` | right (`target_lane=0`, `y_target=0.0`) | **documented alias** of `CRUISE_RIGHT` | same as `CRUISE_RIGHT` |

`CRUISE_RIGHT` and `OVERTAKE_LEFT` are the two distinct trained skills.
`RETURN_RIGHT` is semantically "return to the right lane", which is the same
lane target as `CRUISE_RIGHT`; when no distinct `RETURN_RIGHT.pt` exists, the
loader loads `CRUISE_RIGHT.pt` and renames the returned policy to
`RETURN_RIGHT` (see `event_cbf_driving/skills/loader.py`). If a genuinely
distinct `RETURN_RIGHT.pt` is present, the loader uses it and validation flags
it if its weights duplicate another distinct skill.

## Stale `train_summary.json`

`train_summary.json` is a historical training artifact and is **not** the
source of truth for checkpoint quality. In particular, its
`OVERTAKE_LEFT.learned=false` entry is stale relative to the current
`OVERTAKE_LEFT.pt` (the summary's meta also differs from the checkpoint meta).
Do not trust that flag, and do not claim a skill is learned based on it.

The current checkpoints were re-validated without training:

```bash
event-cbf train --verify-only
```

Result (written to `validation.json` on 2026-11-05T03:22Z):

- `CRUISE_RIGHT`: revalidated `learned=true`
- `OVERTAKE_LEFT`: revalidated `learned=true` (hold `mean_abs_y=0.0078`,
  acquire `final_y=4.0118`, `t16=1.88 s`, `ay_at_other=1.100`)
- `RETURN_RIGHT`: alias of `CRUISE_RIGHT`, therefore also `learned=true`
- learned distinct skills: `2 / 2` (minimum `2`)
- duplicate fingerprints: none
- missing required skills: none
- fingerprint mismatches: none
- validation structurally valid: `true`

The validation command re-runs the learned diagnostic in inference mode. It
does not train or overwrite `CRUISE_RIGHT.pt` or `OVERTAKE_LEFT.pt`.

## Legacy duplicate

The former top-level `RETURN_RIGHT.pt` / `RETURN_RIGHT.json` were a
byte-for-byte duplicate of `CRUISE_RIGHT`, not a separately trained policy.
They were moved with `git mv` to `legacy/` to preserve history; see
`legacy/README.md`. Do not copy them back into this directory.

## Checkpoint format

New checkpoints use schema version 2 and are written atomically (temporary
file + `os.replace`, including the `.json` sidecar):

```text
<SKILL>.pt
    schema_version: 2
    state_dict: torch tensors
    meta: {name, target_lane, y_target, threshold, sigma_mean, sigma_std, spread}
    history: [mean episode returns]
    provenance:
        created_at, package_version, python, torch, numpy
        git: {commit, branch, dirty, dirty_files}
        seed, updates, history_len, final_return
        state_dict_sha256, agent, diagnostics...
    state_dict_sha256
    extra: {diagnostics, ...}

<SKILL>.json
    schema_version, created_at, name, meta, history_len
    fingerprint / state_dict_sha256, provenance
```

Loading behavior:

- Legacy v1 checkpoints (no `provenance`) still load in non-strict mode, with
  a logged warning. They are rejected by `load_skill(..., strict=True)`.
- A recorded `state_dict_sha256` is always verified. A mismatch is rejected
  even in non-strict mode because it means corruption or tampering.
- Required meta keys/types, finite `threshold`/`sigma_*`/`y_target`, and
  `spread > 0` are validated before the policy is returned.
- The state dict must load into `ActorCritic` exactly.

## Validation utilities

```python
from event_cbf_driving.skills import (
    inspect_checkpoint,
    load_skill_bank,
    validate_checkpoint_dir,
)

report = validate_checkpoint_dir("checkpoints")
bank = load_skill_bank("checkpoints")  # CRUISE, OVERTAKE, RETURN alias
```

`validate_checkpoint_dir` reports per-file fingerprints/meta, duplicate
fingerprints across distinct skills, missing required skills, alias use, and
stale or mismatched sidecar warnings. `event-cbf train --verify-only`
regenerates `validation.json`; do not hand-edit that file.

## Training

```bash
# Canonical: only the two distinct skills.
event-cbf train

# Explicitly requesting RETURN_RIGHT trains a distinct policy but warns that
# canonical deployments alias it when no distinct checkpoint exists.
event-cbf train --skills RETURN_RIGHT
```

Training writes `train_summary.json` atomically with per-skill seed,
diagnostics, learned flag, fingerprint, provenance, duplicates, and a top-level
validation block. It exits nonzero if fewer than `TrainConfig.min_learned_skills`
(default `2`) distinct skills validate. The summary is written even on failure.
