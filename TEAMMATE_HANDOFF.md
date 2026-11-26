# Teammate handoff: paper-ready evaluation (updated 2026-11-26)

Branch **`harsh-testing`**. Read [reports/PAPER-RESULTS.md](reports/PAPER-RESULTS.md) first: it holds
the protocol, every final table that exists so far, and the limitations. Older reports carry a
"Superseded" banner; do not quote their numbers.

## Ground rules (do not break these)

- **Final maps are evaluated once, frozen:** SafeMetaDrive **2000–2199**, irregular **3000–3099**,
  highway-env seeds **1000–1199**. Never tune anything while looking at results on them.
- Development maps: SafeMetaDrive 100–149, 500–549, 600–649; irregular 1000–1049; highway-env 100–149.
- Layer 1 (safety filter + crossing guard) and Layer 2 are frozen. Checksums of the frozen files are in
  [reports/final/evidence/code-version.txt](reports/final/evidence/code-version.txt).

## Where things stand

| Piece | Status |
|---|---|
| Crossing-traffic guard (LiDAR tracking, yields to converging traffic) | Done, frozen ([lidar_driver.py](driving/safemetadrive/lidar_driver.py) `CrossingTrafficGuard`) |
| SafeMetaDrive final, 200 maps, all methods + ablations + delay sweep | **Done** (Table 1, Figure 1, Table 2) |
| highway-env final, 600 episodes per method | **Done** (Table 4) |
| PPO-Lag / CPO retrained 5M steps, OmniSafe defaults | **Done** (Table 1) |
| VLM v3 (LiDAR backup trigger, second look at 20 m, verified hold release) | **Done**: frozen at `68dd2b2`, final run on 3000–3099 on 2026-11-26 (Ved's Modal) |
| Irregular final (VLM + 5 no-VLM arms on 3000–3099) | **Done** (Table 3; evidence `irregular-vlm-3000-3099.json.gz`, `irregular-arms-3000-3099.json.gz`) |
| Project site [site/index.html](site/index.html) | **Still shows old numbers** |

Raw evidence for finished tables is in [reports/final/evidence/](reports/final/evidence/) (gzipped JSON +
logs + SHA256SUMS). `runs/` is git-ignored; regenerate or copy evidence into `reports/final/evidence/`.

## Modal access

Both jobs run in **Harsh's Modal workspace** (`harshshah1618`):

- Baselines: app `cs642-safemetadrive`, volume `cs642-safemetadrive`, folder `baselines-5m-defaults-v2/`.
- VLM: app `cs642-irregular`, volume `cs642-irregular`. The PatchCore bank was uploaded to
  `banks/dinov2_small-v1.pt` (SHA256 `d57f63af…a5f5`, identical to the audited bank).

Either get access to that workspace, or ask Harsh to download the baseline folder for you. To run the
VLM from your own workspace, upload the bank to your `cs642-irregular` volume first (Harsh has it at
`runs/ved-workspace-audit-20261125/dinov2_small-v1.pt`). Spend so far ≈ $10–15; the final VLM run is
≈ $20–25 (800 episodes, mostly A10G time).

## Remaining steps (in order)

### 1. Collect and select the retrained baselines

```bash
.venv/bin/modal volume get cs642-safemetadrive baselines-5m-defaults-v2 runs/
PYTHONPATH=. .venv/bin/python tools/select_baselines.py --root runs/baselines-5m-defaults-v2 \
    --label-suffix "(5M, OmniSafe defaults)" --out runs/final/baselines-5m.json
```

Selection uses development maps 100–149 only (one common epoch per algorithm, by mean zero-cost success).
**Sanity check first:** these checkpoints use OmniSafe observation normalization, which
`NativeLearnedDriver` now applies ([policies.py](driving/safemetadrive/policies.py)). The dev success/cost
of the selected epoch should be in the same range as `Metrics/EpRet`/`Metrics/EpCost` in that run's
`train.log`. If the evaluated policy is far worse than training, the normalizer is wrong; fix that
before step 2.

### 2. Add them to the SafeMetaDrive final table

```bash
PYTHONPATH=. .venv/bin/python tools/final_eval.py --first 2000 --count 200 \
    --baselines runs/final/baselines-5m.json --only 5M --merge --out runs/final/safemetadrive-2000-2199.json
```

This appends the new rows to the existing results (same maps) and prints the full table. Replace Table 1 in
`PAPER-RESULTS.md` and remove its "Pending" note. Keep the 1M rows, labelled as the earlier budget.

### 3. Freeze the VLM and run its final test

Recommendation: freeze VLM v3 as is (pilot v3b: 12/49 STOP violations vs 25/50 before, 8/49 stuck, decoys
clean). More tuning on 50 dev maps risks overfitting. If the team wants one more attempt, the remaining
misreads are first looks that say `careful` while the sign shows STOP. Pilot rows:
`modal volume get cs642-irregular closedloop/dev-v3b-qwen3vl-8b runs/dev/dev-v3b`.

```bash
PYTHONPATH=. .venv/bin/modal run --detach tools/irregular_modal.py --stage final-v3
# when done (watch the Modal dashboard):
.venv/bin/modal volume get cs642-irregular closedloop/final-v3-qwen3vl-8b runs/final/irregular-vlm
```

### 4. No-VLM arms on the same final maps (local CPU, ~5 min, free)

```bash
PYTHONPATH=. .venv/bin/python tools/irregular_controls.py --seeds 3000 3100 \
    --arms normal careful oracle lidar-careful lidar-stop --out runs/final/irregular-arms-3000-3099.json
```

### 5. Build Table 3

```bash
.venv/bin/python tools/irregular_final_report.py --arm vlm=runs/final/irregular-vlm \
    --arm normal,careful,oracle,lidar-careful,lidar-stop=runs/final/irregular-arms-3000-3099.json
```

Paste both tables into Table 3 of `PAPER-RESULTS.md`. Report honestly whatever comes out: the expected
story is "the VLM is the only non-oracle arm that obeys the STOP sign at all, without the needless stops
of LiDAR→stop", not "the VLM solves it".

### 6. Archive evidence, update the site, commit

- Gzip new result JSONs into `reports/final/evidence/` and regenerate `SHA256SUMS` there. Rename `.log`
  files to `.log.txt` (`*.log` is git-ignored).
- Update [site/index.html](site/index.html): the benchmark bars (~line 740–820: old 98.0/89.3/84.0/59.3/56.0/0.0),
  the ablation widget (~line 830–900), the VLM section (~line 905–980: old 25/50, highway 80.0%/−98.4%,
  replay 76.9%), the delay section (~line 685: use `reports/final/delay-reserve.png`), and limitations (~line 986).

## Honest findings to keep in the paper

- The PPO tracker is significantly worse than the analytic tracker (likely because it was trained on
  simulator lane features but receives sensor-reconstructed ones), so the analytic tracker is the reference.
- Steering blending has no measurable effect; the turning-room reserve only matters at ≥ 4 s delay.
- The event trigger saves slow-model calls (57× fewer); it does not improve safety over a fair always-ask.
- Waymo/nuScenes (13 scenes) is too small to support a claim and uses an older, separate driver.
- highway-env uses a port of the architecture (different observation/action spaces), not the unified code.

## Optional follow-ups

- Port the replay benchmark to `UnifiedDrivingSystem` (120-ray input) and download full nuScenes for 200+
  scenes (needs a nuScenes account and licence acceptance).
- Retrain the PPO tracker on sensor-reconstructed features.

## New tools added in this round

`tools/final_eval.py` (all SafeMetaDrive methods, same maps, CIs, paired tests) ·
`tools/select_baselines.py` · `tools/irregular_controls.py` (no-VLM arms) ·
`tools/irregular_final_report.py` · `tools/plot_final_delay.py` · `tools/stats.py` ·
`tools/dev_eval.py` (development-split comparisons) · tests in `tests/test_crossing_guard.py`,
`tests/test_paper_stats.py`.

---

# Older handoff (2026-11-20, camera/maneuver study; branch `harsh-testing-further-runs`)


Use branch **`harsh-testing-further-runs`**. Start here; the older straight-road study and its camera adapter under `models/camera/` are a separate experiment.

The code, reports and checksums are in this branch. Large run files, weights, datasets and resumable training states are in this repository's **private [handoff release](https://github.com/harshshah1618/cs642-event-cbf-driving/releases/tag/safemetadrive-handoff-2026-11-20)**. Repository access is sufficient; you do not need Harsh's Modal account. No training or cloud compute starts when restoring artifacts.

## Get the saved work

Authenticate GitHub CLI with an account that has access to this private repository. After cloning and checking out this branch:

```bash
python tools/restore_safemetadrive.py --list
python tools/restore_safemetadrive.py --packages records
```

The second command downloads about 19MB of raw training/evaluation records. To also restore models, use `python tools/restore_safemetadrive.py` (about 800MB total). To restore all training data, latest recovery optimizers/RNG states, and earlier-study videos:

```bash
python tools/restore_safemetadrive.py --all --expand-datasets
```

All packages total about 2.8GB compressed. Allow 8–10GB disk space for archives, extracted files and expanded image shards. The public Qwen base model is **not** duplicated: its weight files alone are about 7.5GB, needed only for actual camera inference/training. Its exact repo/revision is in [the artifact manifest](artifacts/safemetadrive/manifest.json).

The restore tool uses `gh release download`, verifies archive and per-file SHA256 checksums, refuses to overwrite differing local files, and rejects unsafe archive paths. It recreates `runs/safemetadrive/heldout-study-v1/groups-local.json` and the local checkpoint map for your checkout directory. Original maps/plans remain preserved. It also creates local aliases for historical checkpoint commands. Already downloaded bundles can be restored offline with `--from-dir /path/to/bundles`. Package dependencies are restored automatically.

| Package | Contents |
|---|---|
| `records` | Raw final episodes/camera calls, development reports, training logs/configs/histories, selection/frozen plans, audits, local recovery diagnostics and full cloud-file inventory. |
| `models` | All exported native baseline and skill checkpoints; selected recovery camera adapters; balanced-camera comparison adapters and sidecars. |
| `training-data` | Both camera datasets, labels and teacher provenance: **20,375 images** in 40 packed shards. `--expand-datasets` expands under each dataset's `data/`. |
| `resume-0`, `resume-1`, `resume-2` | Recovery LoRA step1000 adapter, optimizer and RNG state for each seed, plus the best adapter paths required by metadata. |
| `legacy-evidence` | Earlier straight-road/camera study runs and videos. These are not SafeMetaDrive benchmark results. |

[SHA256SUMS](artifacts/safemetadrive/SHA256SUMS) and the [readable training-run index](artifacts/safemetadrive/training-runs.csv) are tracked directly in Git. The index includes setup/failed attempts; a historical manifest saying `running` does **not** mean it is active. Redundant historical camera optimizer snapshots remain on the original volume and are catalogued in the restored cloud-file inventory; the release contains selected models and latest recovery resume states, not every 50-update optimizer copy.

## What the current system actually is

Camera pictures → six-choice lane/caution planner → goal-conditioned PPO tracker → heuristic collision protection. The fast layer uses exact simulator navigation/object state; the complete system is not camera-only. The collision forecast/stopping caps are not a formal general-road CBF certificate.

The newer `manage_maneuvers=True` mode adds checked lane-transition admission, rejection/progress feedback and a checked return to the source lane after an unfinished maneuver stalls. It remains **opt-in**. Actual camera inference with the changed feedback prompt has not been rerun; there is no new held-out camera score for it. Read [the recovery implementation/validation](reports/safemetadrive-maneuver-recovery.md) and [the failure assessment](reports/safemetadrive-assessment.md).

## Training completed

Each learned family used three independent random seeds, not an ensemble.

1. **Tracking controller:** 24,000 rule demonstrations per seed, then 976,896 PPO transitions on traffic/accident-free training maps with a custom tracking reward. The copying-only (`cloned.zip`) and PPO (`ppo.zip`) versions are both saved. One shared controller accepts lane/speed targets; there are not three independently trained left/right/stay networks.
2. **Camera:** Qwen2.5-VL-3B with language-only rank 16 LoRA and frozen vision. Original collection produced 9,621 examples. Initial training reached 250–300 updates before interruption. Class-balanced continuations reached 600 cumulative updates. Delayed-teacher recovery collection produced 10,754 examples; recovery continuation with balanced sampling and mirrored examples reached 1,000 cumulative updates. Chosen image checkpoints are 1000/800/800 for seeds 0/1/2. These are continuations, not 600+1000 additional updates. Image training uses maps 0–79, image validation 80–99.
3. **Native comparisons:** three 1M-transition PPO-Lag runs and six 1M-transition CPO runs across two critic-rate configurations. Development selection chose PPO-Lag epoch250/stochastic and CPO default-critic-rate epoch150/stochastic. Interrupted attempts and short setup runs are retained separately.

Training/development/final maps are 0–99 /100–149 /500–549. The final 50 maps have already been examined and must not become the validation set for another claimed held-out result. Latest weights/RNG states allow continuation, but exact batch replay across hardware/library changes is not claimed. Native baseline checkpoints do not imply full optimizer-state resumability.

For camera training continuation, reproduce the camera environment in `tools/safemetadrive_modal.py`: Python 3.11, NumPy 1.26.4, Torch 2.8.0, torchvision 0.23.0, Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.10.1 and SB3 2.3.2. Its CUDA evaluation reports record 12.8. Base model revision is `66285546d2b821cf421d4f5eb2576359d3770cd3`. The native-baseline image/config and compatibility patch are also defined in that file. The restored resume metadata uses original `/results/...` paths: mount the restored `runs/safemetadrive` directory as `/results` in your own GPU environment when continuing those states. Mount this checkout as `/workspace`, matching the original source layout. The trainer entry is `tools/safemetadrive_train_camera.py:train`; choose a **new output run** and explicitly pass `resume_source`, a larger cumulative step count, balanced sampling and mirror augmentation. Do not start the cloud orchestrator merely to inspect results.

## Results that can be claimed

Pinned MetaDrive commit: `fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`. Native benchmark reward/cost/termination, map3, accident probability 0.8, traffic density 0.05, horizon 1000. Final study: 1,900 drives across 50 shared test maps.

| Method | Destination success | Mean native cost |
|---|---:|---:|
| Rule + copying-only tracker |100%|0.000|
| Rule + PPO tracker |100%|0.020|
| Selected camera + PPO + protection, no added delay |92%|0.333|
| PPO-Lag |59.3%|1.927|
| CPO |0.7%|21.933|

Read [the full final report](reports/safemetadrive-results.md), [frozen plan](reports/safemetadrive-results/frozen-plan.json), and [numerical results](reports/safemetadrive-results/analysis.json). Rules were strongest. PPO refinement did not improve on cloning in these controls. The weak/unstable CPO run is not an exhaustive tuned reference. Different observations, supervision and VLM pretraining prevent an equal-information/equal-total-training or official SOTA claim.

Of 150 primary camera drives,134 finished with no penalties,4 finished with penalties,11 timed out without penalties and 1 failed after vehicle collision/road departure. Native cost is not a crash percentage. Injected delay did not monotonically worsen outcomes; the single-outstanding-request design also changed query rate/frame-history spacing. Keep those limitations in any presentation.

## Verify and continue locally

The recorded local CPU environment is in [local-tested-requirements.txt](artifacts/safemetadrive/local-tested-requirements.txt). It differs from the original cloud environment; collision-sensitive numerical replays can differ. A fresh virtual environment can install that snapshot with the PyTorch CPU extra index, then install this checkout editable with `--no-deps`. The project normally uses `pip install -e '.[dev,metadrive]'`; the exact local snapshot is available when version matching matters.

```bash
python -m pytest -q
python tools/safemetadrive_analyze.py \
  --groups runs/safemetadrive/heldout-study-v1/groups-local.json \
  --output runs/safemetadrive/heldout-study-v1/recomputed
CUDA_VISIBLE_DEVICES='' python tools/safemetadrive_maneuver_fixture.py \
  --checkpoint runs/safemetadrive/heldout-study-v1/artifacts/skills-1m-v1/skills-seed0/ppo.zip \
  --output runs/safemetadrive/local-recovery-check.json
```

The last command is a controlled training-map physics fixture using a saved PPO tracker. It is not new camera inference. The handoff passes **275 tests**, including the four restoration checks, in both the original workspace and a fresh clone of the pushed branch. GitHub CI also passed. See [handoff verification](artifacts/safemetadrive/handoff-verification.json). All three saved PPO trackers recovered in the six managed fixtures with zero native cost. Fixed saved-input development checks improved five cases and worsened two; they are not fresh closed-loop camera evaluations.

Next work should focus on development scenarios, live camera response to the new feedback interface, and a same-input small-image-model comparison. A new benchmark claim needs fresh unseen maps and actual camera inference. Camera training/inference requires appropriate hardware and an agreed compute budget.

## Operational and historical notes

The Modal app `cs642-safemetadrive` (`ap-G6zs1EeG8zvf8kA7odoGCW`) was stopped after the study; the volume of the same name and HF cache `vlm-router-hf` remain. The export/restore process does not deploy or spawn functions. Do not run paid jobs without an agreed budget. Historical reported project compute was about $33.42 before credits; billing snapshots are not a current balance.

The original detached local launcher did not survive laptop disconnection; later runs used a remote coordinator. Full call IDs, attempts and chronological notes are restored under `runs/safemetadrive/`. Many historical work-log entries describe jobs as queued/running: they are historical, not instructions to resume them. The committed frozen plan and this page take precedence as descriptions of the completed study.

Earlier real-camera/straight-road work is included for completeness in `reports/real-model-study.md`, `tools/real_model_study.py`, and the optional legacy bundle. Its A/B/C actions and adapter under `models/camera/` differ from SafeMetaDrive's six choices/adapters. Set `QWEN_BASE_MODEL_PATH` to your own pinned base snapshot when using that older runner; no author's home directory is required. Some archival report links point into `runs/` and only resolve after restoration.
