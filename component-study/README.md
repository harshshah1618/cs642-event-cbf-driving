# What each driving component contributes

This work is isolated on `codex/component-contributions` in
`/home/harsh/Projects/cs642-event-cbf-components`. The original dirty checkout and
the previous frozen scheduler research remain preserved in their own worktrees.

The question is whether the learned lane driver, longitudinal CBF constraint,
stopping backup and lane-change admission guard improve actual closed-loop
driving. `PROTOCOL.md` specifies the matched factorial before the final runs.

## What the parts mean

- **Router:** chooses stay right, overtake/keep left, or return right.
- **PPO or PD:** converts the chosen lane into acceleration commands. PPO is the
  existing trained bank; PD is a fixed steering/speed rule with unchanged gains.
- **Longitudinal CBF:** caps requested forward acceleration using a headway
  inequality for relevant vehicles ahead.
- **Stopping backup:** separately limits acceleration using a stopping-distance
  envelope. It may overlap with the CBF; redundancy is an outcome, not proof that
  a component never matters.
- **Maneuver guard:** can reject a lane change using fresh front/rear traffic
  positions when the answer arrives.
- **Common protections:** road-boundary correction and invalid-data monitoring
  stay active in every treatment, including the traffic-protections-off cell.

## Actual camera integration

`router/vlm.py` implements image-plus-ego-state requests to an explicitly
configured OpenAI-compatible vLLM endpoint. The default new prompt omits the
current-skill hint, so it cannot inherit the teammate v1 prompt's accuracy.
The camera is captured on the simulation thread; only detached encoded bytes
reach the network worker. Current skill and surrounding-object ground truth are
excluded from the VLM payload. The safety mechanisms separately use simulator
traffic state; this is not a camera-only safety system.

The adapter validates a one-letter A/B/C completion, records image/prompt hashes,
measures response latency, deducts camera capture from the remaining deadline,
rejects late responses and has no scripted fallback. Camera capture and model-
request wall time are recorded separately. HTTP timeouts do not make this a hard
real-time system. The short loopback smoke verifies rendered pixels, actual HTTP,
asynchronous control and a programmed maneuver reaching the physical plant. It
is explicitly marked `actual_model=false` and is not a VLM performance result.

The supplied teammate report calls `ved-testing` unpushed. Initial branch/fork/PR
checks showed only `main` and `harsh-testing`; that did not establish commit
unavailability. A subsequent exact-commit API lookup recovered their actual code
and saved results at `1f16b7ff59aae20c2a9adfbeb4943762d4b1f5b4` from GitHub.
The verified snapshot is included under `teammate-source/vlm-router`, with hashes.
The trained weights are saved only in their private Modal volume; no model
endpoint, exported checkpoint or local Modal credentials are available here.
An actual fine-tuned arm therefore still requires model access. The controlled
factorial uses clearly named rule/fault routers and is not changed by recovering
the teammate's source. Its explicit rule geometry is inspired by that task,
not an exact reproduction of the training label generator.

## Reproduce the component experiment

Use Python 3.12 and the pinned MetaDrive/PyTorch environment from the previous
research delivery (`research/environment-freeze.txt`). The local worktree uses
an ignored symlink to the existing environment; the archive does not bundle it.

```bash
cd /home/harsh/Projects/cs642-event-cbf-components/event-cbf-driving
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check event_cbf_driving tests scripts
.venv/bin/python -m mypy event_cbf_driving

.venv/bin/python scripts/component_ablation.py pilot --workers 6
.venv/bin/python scripts/component_ablation.py freeze
.venv/bin/python scripts/component_ablation.py test --workers 6
.venv/bin/python scripts/component_ablation.py verify
.venv/bin/python scripts/component_report.py
```

For analysis of an extracted archive, run
`.venv/bin/python scripts/restore_component_data.py` first, then `verify` and
`component_report.py`. The restore helper refuses to replace differing data.

The lock cannot be overwritten. Existing rows are resumed by exact job identity;
source/checkpoint/protocol changes invalidate the main run. For a different
experiment, preserve the current evidence and create a separately versioned run.

The experiment contains 192 disjoint pilot episodes and 4,608 final episodes:
12 fresh traffic seeds x 4 scenes x 2 injected delays x 3 diagnostic routers x
2 controllers x 2 CBF settings x 2 stopping-backup settings x 2 lane guards.
Twelve seeds are twelve clusters, not 4,608 independent training replications.

## Exercise the camera interface or run a supplied model

```bash
# Programmed fixture: real pixels and HTTP, zero real-model claims.
P3D_DISPLAY=p3headlessgl .venv/bin/python scripts/vlm_closed_loop.py \
  --contract-smoke --output runs/a-new-contract-smoke

# Actual model: replace these with the teammate's explicit deployment details.
P3D_DISPLAY=p3headlessgl .venv/bin/python scripts/vlm_closed_loop.py \
  --server http://HOST:PORT/v1 --model SERVED_MODEL_NAME \
  --prompt-mode teammate-v1 \
  --seeds 5101 5102 5103 --output runs/a-new-actual-vlm-evaluation
```

The actual-model command runs full, no-CBF, no-braking, no-admission, and PD
variants on the three original scene families with asynchronous paced control.
For authenticated endpoints set `EVENT_CBF_VLM_API_KEY`; credentials are not
stored in profiles or reports. The command fails if any episode receives no
valid live-model answer. It never silently substitutes the scripted router.
The `teammate-v1` mode uses the pinned training prompt, including its current-
skill cue and parked-car/20 m/s assumptions, as an explicit compatibility
baseline. Use `--prompt-mode clean` for the hint-free alternative, and evaluate
it afresh. Neither setting inherits the saved frame-test accuracy. Model export,
serving and processor settings are documented in `TEAMMATE_HANDOFF.md`.

## Evidence

`results/report.md` is the measured interpretation once the final run completes.
`statistics.json` includes all main effects, pair interactions, conditional
removals and 24 complete scene/router/delay tables. Raw pilot/test gzip rows and
the prelaunch lock are preserved. Every episode records achieved physical
acceleration before trace reduction; jerk metrics are not guessed from commands.
Representative full traces, camera frames and a separate contract-smoke summary
document the integration. Independent Astra audits live in `audits/`.

No result should be called universal component necessity, rare-event safety,
an independently retrained PPO comparison or actual teammate VLM driving until
that specific evidence exists. Earlier scheduler results remain a separate study.
