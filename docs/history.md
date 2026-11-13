# Retired experiments and recovery

The working tree keeps the maintained simulator package, current reports,
proposal, checkpoint assets and final comparison. Older experiment runners,
toy projects, duplicate figures and delivery copies have been retired.

## Source snapshots

- `0dbffe7`: complete repository immediately before this trim, including all
  historical study folders and scripts.
- `2762d08`: recorded final integrated evaluation, before the PPO restructuring.
  Use this source and the full raw release when reproducing its frozen runs.

Inspect an old file with `git show 0dbffe7:path/to/file`. To recover the full
pre-trim checkout without changing the active branch:

```bash
git worktree add --detach ../cs642-event-cbf-before-trim 0dbffe7
```

## What was retired

- Earlier `research/`, `component-study/`, `maneuver-study/` and `vlm-study/`
  reports, pilot frames, raw samples and work logs. The current trained camera
  adapter, model metadata and exact pinned prompt remain under
  `event-cbf-driving/checkpoints/vlm/`.
- `wip-minimal/`, `two-trap-gauntlet/`, scratch exercises and old figures.
- Earlier one-off experiment runners, stress sweeps, data-restoration scripts,
  old report generators and wrappers duplicated by the installed CLI.
- The five test modules exclusive to the retired clock, spatial, component and
  historical cell-audit runners. Runtime and final-comparison tests remain active.
- Superseded planning notes, milestone versions and package reports.
- Generated local deliveries and earlier simulator run folders.

## Local backup

A verified full tracked-source archive, relocation manifest and retired local
files are stored next to this checkout under
`../cs642-event-cbf-archive/trim-0dbffe7/`. Its `manifest.json` records the source
commit, archive hash and moved paths. Generated raw outputs were moved there,
not discarded. The full final-study raw release is inside its retired delivery
folder. Other machines can recover tracked files from Git; raw outputs still
require the original complete release.

The recorded reports retain original experiment paths and source hashes as
historical evidence. Those strings are not current command instructions.
Use the main README and maintained script index for the current layout.
