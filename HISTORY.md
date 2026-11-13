# Source history and recovery

The maintained layout groups code, tests, tools, configuration, models and
reports by purpose. Earlier layouts and experiments remain recoverable from Git.

- `42e26f8`: the flat layout before this grouping.
- `4ea7285`: the previous nested package layout.
- `0dbffe7`: source before old experiment runners and toy projects were retired.
- `2762d08`: original frozen final-comparison source.

For example:

```bash
git worktree add --detach ../cs642-driving-original 2762d08
```

Use that frozen source and the complete raw release to re-audit the recorded
experiment. Historical paths in the saved reports describe that original source;
new experiments need fresh directories because source locks reject changed code.

Saved driver banks are in `models/native/` and `models/physical/`; the camera
adapter, prompt and metadata are in `models/camera/`. Their original bytes were
preserved. Supporting tables, audits, protocol, fonts and report-input records
remain together in `reports/study-evidence.zip`.

Local recovery archives are kept next to this checkout under
`../cs642-event-cbf-archive/`. The `regroup-42e26f8/` snapshot contains the source
before this grouping. Earlier retired run folders and the complete raw release
remain in `trim-0dbffe7/`; base model weights are external.

New experiment output goes to `runs/` and is ignored by Git. Tool caches and the
existing local Python environment remain outside the source tree. No trained
weights or reported measurements were changed by the folder organization.
