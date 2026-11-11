# The trained AI is now available and has been run

GitHub ved-testing advanced to 082deae3f941a686400bc85a8bd9ddaca8ee892d.
It includes the trained model add-on, camera bridge, router and cloud commands.
The earlier missing-model-file blocker is resolved. Earlier study reports are
historical; their component measurements remain unchanged.

## What was actually tested here

1. Downloaded the public base AI and loaded the teammate's exact trained add-on.
   Verified every one of its 504 saved parameter tensors against the loaded
   values, including the file hash and active adapter. No base-only substitution.
2. Fed it a freshly rendered physical-simulator image showing a parked car in
   our lane and a clear left lane. It correctly answered B: overtake. This took
   12.08 seconds on the local CPU. One picture is not an accuracy benchmark.
3. Connected that actual model to the physical MetaDrive car, the existing PPO
   driver, and all three traffic protections. This used actual wall-time model
   calls in a background worker, not recorded answers or a fake server.

The driving pilot made four real predictions, all B. Three answers were applied;
the fourth finished after the episode ended. It recorded zero fallbacks and
zero crashes, but got stuck at 33.27% of the route and failed the success test.
The episode lasted 60 simulated seconds and 136.50 wall seconds including
render startup and final inference cleanup. Prediction times were 18.48, 19.52,
18.49 and 12.87 seconds. This hardware has no NVIDIA GPU.

## Why the car got stuck

The car had nearly stopped behind the parked vehicle by simulation time 6.04 s.
The first overtake answer was applied at 18.54 s. The driver then requested
forward movement and a turn, but the braking backup continued to restrict
forward movement. At almost zero speed, turning the wheels did not move the
car sideways into the other lane. A tiny negative stopping-envelope margin
persisted, and the car remained stopped. The numerical monitor did not latch
an emergency stop.

The trace supports a late-decision followed by stop/turn/braking deadlock. It
does not show a sequence of incorrect model letters. It also does not establish
that the GPU version fails: the teammate's reported GPU decisions are much
faster, and an earlier maneuver may avoid this stopped state. That requires a
separate experiment, not an assumption. No controller change was tuned on this
single failed drive.

## What the new teammate code does differently

Their new closed-loop script drives the old point-mass simulator and renders
its state into a separate MetaDrive scene for the camera. It counts model time
without render/transport time and replays pooled delays into a rule baseline.
Those are useful development experiments, but not the same as physical-camera
driving or a clean isolation of perception error. See audits/latest-integration.md.
Our pilot retained the tested physical camera and full request/response contract.

## Reproduction

The 7.51 GB base weights are cached outside the repository. Their revision is
pinned in base-model.json. Original training did not record a base revision;
the newly shared adapter was exported in BF16. Therefore this reconstruction is
not assumed to be byte-identical to the teammate's earlier merged GPU artifact.

Use the existing package environment, add PEFT 0.17.1 in a task-local directory,
and use the versions recorded in environment.json. No global packages were
changed and no cloud resources were launched.

```bash
cd /home/harsh/Projects/cs642-event-cbf-vlm
uv pip install --python event-cbf-driving/.venv/bin/python \
  --target .vlm-deps --no-deps peft==0.17.1
export PYTHONPATH="$PWD/event-cbf-driving:$PWD/.vlm-deps"
cd event-cbf-driving

# BASE is the local snapshot directory from base-model.json.
.venv/bin/python scripts/trained_vlm_cpu.py \
  --base "$BASE" --adapter ../vlm-study/model/adapter \
  --prompts ../component-study/teammate-source/vlm-router/prompts.py \
  --image ../vlm-study/frames/overtake.png --lane RIGHT --skill CRUISE_RIGHT \
  --out ../vlm-study/a-new-probe.json

P3D_DISPLAY=p3headlessgl LP_NUM_THREADS=1 \
  .venv/bin/python scripts/trained_vlm_drive.py \
  --base "$BASE" --output ../vlm-study/a-new-drive
```

The executed processes were isolated with a transient user-service limit of
10 GiB cgroup memory, no swap allocation, four CPU equivalents and 15 minutes.
File-backed page accounting differs from process RSS; the pilot's measured
process peak was 10,951,004 KiB. Full limits and observed service metrics are
retained. This is not a hard real-time controller and the simulation fell
behind wall time. It is an exploratory one-seed CPU integration check, not a
GPU benchmark, paired ablation, generalization study or safety certificate.

Evidence: probe-overtake.json; frames/manifest.json and PNGs; drive/episode.json;
drive/predictions.jsonl; drive/summary.json; adapter tensor/hash audit;
base-model.json; and the independent CPU-pilot review. The original source and
both previous research studies remain preserved in separate worktrees.
