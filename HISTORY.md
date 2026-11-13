# History and recovery

The project was flattened so its code, tests, configs, models and reports live
directly in the main folder. Only GitHub's hidden workflow retains its required
`.github/workflows/` location. Git metadata and the environment are infrastructure,
not source folders; the local `.venv` link points to the existing environment
outside this checkout.

The supporting research files and fonts are preserved byte-for-byte inside
`study-evidence.zip`. Driver banks and camera assets are separate ZIP files.
`assets.py` reads these through an external temporary cache. Generated runs and
tool caches are also kept outside the project by default.

## Recover an earlier layout

- `4ea7285`: complete source immediately before flattening.
- `0dbffe7`: source before retiring old experiments and toy projects.
- `2762d08`: original frozen final comparison source.

For example, inspect an old file with `git show 4ea7285:path/to/file`, or create
an independent original-layout checkout:

```bash
git worktree add --detach ../cs642-driving-original 4ea7285
```

A verified pre-flattening source archive, old folder tree and filename map are
kept locally under `../cs642-event-cbf-archive/flat-4ea7285/`. Older retired
outputs and the complete final-study raw release are in the adjacent
`trim-0dbffe7/` archive. They were moved, not discarded.

## Reproducing results

The saved reports describe their original source versions and paths. Use the
frozen source snapshot plus the complete raw release to re-audit those runs;
source locks intentionally reject changed source. New experiments require a
fresh output directory. Large base model weights remain external.

The new folder layout changes imports and asset locations, not the driving
algorithms or saved weights. The asset archives preserve the original model
filenames and bytes after extraction.
