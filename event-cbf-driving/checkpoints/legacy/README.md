# Legacy checkpoints (audit only)

These files are retained for history and auditability. They are **not** loaded
by the canonical checkpoint loader and should not be used for new runs.

## Why `RETURN_RIGHT.pt` / `.json` live here

`RETURN_RIGHT` is semantically "return to the right lane", the same lane target
as `CRUISE_RIGHT` (`target_lane=0`, `y_target=0.0`). The former top-level
`RETURN_RIGHT.pt` was a byte-for-byte copy of `CRUISE_RIGHT.pt`, not a
separately trained policy:

- state-dict fingerprint from `state_dict_fingerprint` (v1 format):
  `9706785d81623499e66f1b3b7d873c4f815832dff6e068feaa50317d319eb9dd`
- same meta (`threshold`, `spread`, and all other fields)
- same 130-entry training history

Presenting that copy as a third trained skill overstated what was actually
trained. It was moved here with `git mv`, so repository history is preserved.
Canonical deployments now load `RETURN_RIGHT` as a documented alias of
`CRUISE_RIGHT` when no distinct `checkpoints/RETURN_RIGHT.pt` exists; see
`../README.md` and `event_cbf_driving/skills/loader.py`.

Do not copy these files back to the top-level `checkpoints/` directory. If a
genuinely distinct `RETURN_RIGHT` policy is trained later, save it at
`checkpoints/RETURN_RIGHT.pt`; the loader will prefer it over the alias and the
validator will flag it if its weights duplicate another distinct skill.
