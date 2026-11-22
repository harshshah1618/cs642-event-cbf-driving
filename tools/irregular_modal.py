"""Modal jobs for the irregular-situation experiments (T4 GPUs only).

Stages (run from the repo root):
  .venv/bin/modal run tools/irregular_modal.py --stage smoke
  .venv/bin/modal run --detach tools/irregular_modal.py --stage clean
  .venv/bin/modal run --detach tools/irregular_modal.py --stage evalset
  .venv/bin/modal run --detach tools/irregular_modal.py --stage bank --encoder dinov2_small
  .venv/bin/modal run --detach tools/irregular_modal.py --stage score --encoder dinov2_small
Download results: .venv/bin/modal volume get cs642-irregular <path> <local dir>
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("cs642-irregular")
volume = modal.Volume.from_name("cs642-irregular", create_if_missing=True)
DATA = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libegl1", "libglib2.0-0")
    .pip_install("numpy==1.26.4", "gymnasium==0.29.1", "pillow",
                 "metadrive-simulator @ git+https://github.com/metadriverse/metadrive.git@fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1")
    .run_commands("python -m metadrive.pull_asset")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.1", "scikit-learn")
    .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "NVIDIA_DRIVER_CAPABILITIES": "all",
          "P3D_DISPLAY": "p3headlessgl", "HF_HOME": f"{DATA}/hf", "TORCH_HOME": f"{DATA}/torch"})
    .add_local_dir(ROOT / "driving" / "safemetadrive", "/workspace/driving/safemetadrive",
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(ROOT / "driving" / "irregular", "/workspace/driving/irregular",
                   ignore=["**/__pycache__", "**/*.pyc"])
)
GPU = "T4"  # cheapest Modal GPU; enough for rendering and small frozen encoders


def _workspace():
    import sys
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")


@app.function(image=image, volumes={DATA: volume}, gpu=GPU, cpu=2, memory=8192,
              timeout=3 * 3600, max_containers=10)
def collect_shard(spec):
    _workspace()
    from driving.irregular.frames import (
        EVAL_PLACEMENTS,
        EVAL_SIGN_PLACEMENTS,
        SIGN_PLACEMENTS,
        collect,
    )

    placements = dict(eval=EVAL_PLACEMENTS, signs=SIGN_PLACEMENTS,
                      evalsigns=EVAL_SIGN_PLACEMENTS).get(spec["placements"])
    out = f"{DATA}/frames/{spec['run']}/shard-{spec['seeds'][0]:03d}"
    summary = collect(spec["seeds"], out, spec["split"], placements=placements,
                      every=spec["every"], max_steps=spec["max_steps"])
    _pack(out)
    volume.commit()
    return dict(out=out, frames=summary["frames"], wall_seconds=summary["wall_seconds"])


def _pack(shard_dir):
    """Bundle a shard's JPEGs into one uncompressed tar.

    Reading thousands of small files from a Modal volume costs ~140 ms each;
    one large sequential read is fast.
    """
    import io
    import tarfile

    shard = Path(shard_dir)
    rows = json.loads((shard / "labels.json").read_text())
    tmp = shard / "frames.tar.tmp"
    with tarfile.open(tmp, "w") as tar:
        for row in rows:
            data = (shard / row["file"]).read_bytes()
            info = tarfile.TarInfo(row["file"])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tmp.rename(shard / "frames.tar")
    return len(rows)


@app.function(image=image, volumes={DATA: volume}, cpu=1, memory=2048, timeout=3600,
              max_containers=16)
def pack_shard(shard_dir):
    import time
    started = time.monotonic()
    n = _pack(shard_dir)
    volume.commit()
    return dict(shard=shard_dir, frames=n, seconds=round(time.monotonic() - started))


def _frame_rows(run):
    rows = []
    for shard in sorted(Path(f"{DATA}/frames/{run}").glob("shard-*")):
        for row in json.loads((shard / "labels.json").read_text()):
            rows.append(dict(row, shard=str(shard)))
    return rows


def _shard_bytes(shard):
    import tarfile

    path = Path(shard) / "frames.tar"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run --stage pack first")
    with tarfile.open(path) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def _batches(rows, size, label="frames", log_every=1000):
    import io
    import time

    from driving.irregular.patchcore import front_view
    from PIL import Image

    started, current, cache, batch = time.monotonic(), None, None, []
    for n, row in enumerate(rows, 1):
        if row["shard"] != current:
            current, cache = row["shard"], _shard_bytes(row["shard"])
        batch.append(front_view(Image.open(io.BytesIO(cache[row["file"]]))))
        if len(batch) == size:
            yield batch
            batch = []
        if n % log_every == 0 or n == len(rows):
            print(f"[{label}] {n}/{len(rows)} frames encoded, {time.monotonic() - started:.0f} s",
                  flush=True)
    if batch:
        yield batch


@app.function(image=image, volumes={DATA: volume}, gpu=GPU, cpu=4, memory=16384, timeout=3 * 3600)
def build_bank(encoder_name, run="clean-train-v1", patches_per_image=64, coreset_size=20000,
               tag="v1"):
    import time

    _workspace()
    from driving.irregular.patchcore import Encoder, PatchCore

    started = time.monotonic()
    rows = _frame_rows(run)
    encoder = Encoder(encoder_name)
    print(f"[bank {encoder_name}] {len(rows)} clean frames", flush=True)
    model, pool = PatchCore.fit(encoder, _batches(rows, 32, f"bank {encoder_name}"),
                                patches_per_image, coreset_size)
    print(f"[bank {encoder_name}] coreset {model.bank.shape[0]} of {pool} patches, "
          f"{time.monotonic() - started:.0f} s total", flush=True)
    Path(f"{DATA}/banks").mkdir(parents=True, exist_ok=True)
    path = f"{DATA}/banks/{encoder_name}-{tag}.pt"
    model.save(path)
    meta = dict(encoder=encoder_name, frames=len(rows), pooled_patches=pool,
                bank_size=int(model.bank.shape[0]), grid=list(model.grid),
                patches_per_image=patches_per_image, source_run=run,
                wall_seconds=time.monotonic() - started)
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    volume.commit()
    return meta


@app.function(image=image, volumes={DATA: volume}, gpu=GPU, cpu=4, memory=16384, timeout=3 * 3600)
def score_frames(encoder_name, run="eval-dev-v1", tag="v1"):
    import time

    import numpy as np
    import torch

    _workspace()
    from driving.irregular.patchcore import Encoder, PatchCore

    rows = _frame_rows(run)
    encoder = Encoder(encoder_name)
    model = PatchCore.load(f"{DATA}/banks/{encoder_name}-{tag}.pt", device="cuda")
    scores, maps, per_frame_ms = [], [], []
    for images in _batches(rows, 32, f"score {encoder_name}"):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        s, m = model.score(encoder(images))
        torch.cuda.synchronize()
        per_frame_ms.append(1000 * (time.monotonic() - t0) / len(images))
        scores.extend(s.cpu().tolist())
        maps.append(m.half().cpu().numpy())
    out = Path(f"{DATA}/scores/{encoder_name}-{tag}-{run}")
    out.mkdir(parents=True, exist_ok=True)
    keep = ("file", "map_seed", "step", "scene", "placement", "truth", "speed_mps", "current_lane")
    records = [dict({k: r.get(k) for k in keep}, score=s) for r, s in zip(rows, scores, strict=True)]
    (out / "scores.json").write_text(json.dumps(dict(
        encoder=encoder_name, run=run, frames=len(rows),
        batched_ms_per_frame=float(np.median(per_frame_ms)), records=records)))
    np.save(out / "maps.npy", np.concatenate(maps))
    volume.commit()
    return dict(out=str(out), frames=len(rows), ms_per_frame=float(np.median(per_frame_ms)))


def _vlm_sample(per_episode=4, clean_frames=240, seed=0):
    """Frames for offline VLM selection: visible objects 8-35 m ahead, plus clean frames."""
    import random

    from driving.irregular.vlm import ACCEPTABLE, scene_key

    rng = random.Random(seed)
    rows = [r for r in _frame_rows("eval-dev-v1") + _frame_rows("eval-signs-dev-v1")]
    clean = [r for r in rows if r["scene"] == "clean" and "eval-dev-v1" in r["shard"]]
    picked = rng.sample(clean, min(clean_frames, len(clean)))
    episodes = {}
    for r in rows:
        t = r.get("truth")
        if r["scene"] == "clean" or not t or not t["visible"] or not 8 <= t["forward"] <= 35:
            continue
        if scene_key(r) not in ACCEPTABLE:
            continue
        episodes.setdefault((scene_key(r), r["map_seed"]), []).append(r)
    for _, eps in sorted(episodes.items()):
        eps.sort(key=lambda r: r["truth"]["forward"])
        step = max(1, len(eps) // per_episode)
        picked += eps[::step][:per_episode]
    return picked


def _vlm_eval(spec):
    import io
    import time

    import numpy as np
    from driving.irregular.patchcore import front_view
    from driving.irregular.vlm import ZeroShotVLM, anomaly_crop, option_order, scene_key
    from PIL import Image

    rows = _vlm_sample()
    if spec.get("scenes"):
        rows = [r for r in rows if scene_key(r) in spec["scenes"]]
    model = ZeroShotVLM(spec["model"], dtype=spec.get("dtype", "float16"))
    conditions_wanted = spec.get("conditions", ["canonical", "shuffled", "blank"])
    maps = {}
    if any(c.startswith("crop") for c in conditions_wanted):
        for run in ("eval-dev-v1", "eval-signs-dev-v1"):
            d = Path(f"{DATA}/scores/dinov2_small-v1-{run}")
            recs = json.loads((d / "scores.json").read_text())["records"]
            arr = np.load(d / "maps.npy")
            maps.update({rec["file"]: arr[i] for i, rec in enumerate(recs)})
    out = Path(f"{DATA}/vlm/{spec['tag']}")
    out.mkdir(parents=True, exist_ok=True)
    cache, current, results = None, None, []
    started = time.monotonic()
    blank_every = spec.get("blank_every", 4)
    for n, r in enumerate(rows):
        if r["shard"] != current:
            current, cache = r["shard"], _shard_bytes(r["shard"])
        image = front_view(Image.open(io.BytesIO(cache[r["file"]])).convert("RGB"))
        state = dict(speed_mps=r.get("speed_mps", 0.0), current_lane=r.get("current_lane", 0),
                     lane_count=r.get("lane_count", 1))
        crop = anomaly_crop(image, maps[r["file"]].astype(np.float32)) if maps else None
        conditions = [("canonical", image, option_order(state), None),
                      ("shuffled", image, option_order(state, n), None),
                      ("crop", image, option_order(state), crop),
                      ("crop-shuffled", image, option_order(state, n), crop),
                      ("crop-speed0", image, option_order(state), crop),
                      ("crop-nospeed", image, option_order(state), crop)]
        if n % blank_every == 0:
            conditions.append(("blank", Image.new("RGB", image.size, (128, 128, 128)),
                               option_order(state), None))
        conditions = [c for c in conditions if c[0] in conditions_wanted]
        for name, img, order, extra in conditions:
            st = dict(state, speed_mps=0.0) if name == "crop-speed0" else state
            action, probs, seconds = model.answer(img, st, order, crop=extra,
                                                  speed=name != "crop-nospeed")
            results.append(dict(file=r["file"], scene=scene_key(r), map_seed=r["map_seed"],
                                forward=(r.get("truth") or {}).get("forward"), condition=name,
                                order=order, action=action, probs=probs, seconds=seconds))
        if (n + 1) % 100 == 0 or n + 1 == len(rows):
            print(f"[vlm {spec['tag']}] {n + 1}/{len(rows)} frames, "
                  f"{time.monotonic() - started:.0f} s", flush=True)
    (out / "results.json").write_text(json.dumps(dict(model=spec["model"], dtype=spec.get("dtype"),
                                                     frames=len(rows), results=results)))
    volume.commit()
    return dict(tag=spec["tag"], frames=len(rows), calls=len(results),
                wall_seconds=round(time.monotonic() - started))


@app.function(image=image, volumes={DATA: volume}, gpu="T4", cpu=4, memory=24576, timeout=3 * 3600)
def vlm_eval_t4(spec):
    _workspace()
    return _vlm_eval(spec)


@app.function(image=image, volumes={DATA: volume}, gpu=["L4", "A10G"], cpu=4, memory=32768,
              timeout=4 * 3600)
def vlm_eval_l4(spec):
    _workspace()
    return _vlm_eval(spec)


@app.function(image=image, volumes={DATA: volume}, gpu="A100-80GB", cpu=8, memory=98304,
              timeout=4 * 3600)
def vlm_eval_a100(spec):
    _workspace()
    return _vlm_eval(spec)


def _closed_loop(spec):
    import time

    from driving.irregular.escalation import (
        EscalationConfig,
        PatchCoreDetector,
        VLMCaller,
        run_episode,
        scene_placements,
    )
    from driving.irregular.scene import make_env
    from driving.safemetadrive.camera_planner import camera_config
    from driving.safemetadrive.protocol import environment_config

    started = time.monotonic()
    vlm = VLMCaller(spec["model"], spec.get("dtype", "bfloat16"))
    detector = PatchCoreDetector(f"{DATA}/banks/dinov2_small-v1.pt")
    env = make_env(camera_config(environment_config(spec.get("split", "validation"))))
    out = Path(f"{DATA}/closedloop/{spec['tag']}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for seed in spec["seeds"]:
            cfg = EscalationConfig(trigger=spec["trigger"], **spec.get("config", {}))
            row = run_episode(env, seed, scene_placements(spec["scene"], seed), cfg, vlm=vlm,
                              detector=detector, render=True,
                              max_steps=300 if spec["scene"] == "clean" else 700)
            row.update(scene=spec["scene"], model=spec["model"])
            rows.append(row)
            print(json.dumps(dict(scene=spec["scene"], trigger=spec["trigger"], seed=seed,
                                  passed_stop=row["passed_stop"], calls=len(row["calls"]),
                                  stopped=row["seconds_stopped_near_object"],
                                  crash=row["crash_object"] or row["crash_human"]
                                  or row["crash_vehicle"], steps=row["steps"])), flush=True)
            name = f"{spec['scene']}-{spec['trigger']}.json"
            (out / name).write_text(json.dumps(rows, default=str))
            volume.commit()
    finally:
        env.close()
    return dict(tag=spec["tag"], scene=spec["scene"], trigger=spec["trigger"],
                episodes=len(rows), wall_seconds=round(time.monotonic() - started))


@app.function(image=image, volumes={DATA: volume}, gpu=["L4", "A10G"], cpu=4, memory=32768,
              timeout=3 * 3600, max_containers=10)
def closed_loop_l4(spec):
    _workspace()
    return _closed_loop(spec)


@app.function(image=image, volumes={DATA: volume}, gpu="A100-80GB", cpu=8, memory=98304,
              timeout=3 * 3600, max_containers=10)
def closed_loop_a100(spec):
    _workspace()
    return _closed_loop(spec)


ALL_CONDITIONS = ["canonical", "shuffled", "blank", "crop", "crop-shuffled"]
SPEED_SCENES = ["clean", "sign-billboard-shoulder", "flagger-shoulder:stop", "flagger-shoulder:slow",
                "pedestrian-shoulder", "sign-lane_closed-shoulder"]
VLM_RUNS = {
    "vlm7b": (vlm_eval_l4, dict(model="Qwen/Qwen2.5-VL-7B-Instruct", dtype="bfloat16",
                                tag="qwen25vl-7b-bf16", conditions=ALL_CONDITIONS)),
    "vlm8b-qwen3": (vlm_eval_l4, dict(model="Qwen/Qwen3-VL-8B-Instruct", dtype="bfloat16",
                                      tag="qwen3vl-8b-bf16", conditions=ALL_CONDITIONS)),
    "speedtest-32b": (vlm_eval_a100, dict(
        model="Qwen/Qwen2.5-VL-32B-Instruct", dtype="bfloat16", tag="speedtest-qwen25vl-32b",
        conditions=["crop", "crop-speed0", "crop-nospeed"], scenes=SPEED_SCENES)),
    "speedtest-8b": (vlm_eval_l4, dict(
        model="Qwen/Qwen3-VL-8B-Instruct", dtype="bfloat16", tag="speedtest-qwen3vl-8b",
        conditions=["crop", "crop-speed0", "crop-nospeed"], scenes=SPEED_SCENES)),
    "vlm32b": (vlm_eval_a100, dict(model="Qwen/Qwen2.5-VL-32B-Instruct", dtype="bfloat16",
                                   tag="qwen25vl-32b-bf16", conditions=ALL_CONDITIONS)),
}


def _shards(seeds, size):
    return [seeds[i:i + size] for i in range(0, len(seeds), size)]


@app.local_entrypoint()
def main(stage: str = "smoke", encoder: str = "dinov2_small", run: str = "eval-dev-v1"):
    if stage == "pack":
        shards = [f"{DATA}/frames/{run}/{entry.path.split('/')[-1]}"
                  for run in ("clean-train-v1", "eval-dev-v1")
                  for entry in volume.listdir(f"frames/{run}")]
        for result in pack_shard.map(shards):
            print(json.dumps(result), flush=True)
        return
    if stage == "smoke":
        specs = [dict(run="smoke-v1", split="train", seeds=[0], placements=None, every=10, max_steps=60),
                 dict(run="smoke-v1", split="validation", seeds=[100], placements="eval",
                      every=10, max_steps=120)]
    elif stage == "signsmoke":
        specs = [dict(run="signsmoke-v1", split="validation", seeds=[100, 101], placements="signs",
                      every=5, max_steps=120)]
    elif stage == "evalsigns":
        specs = [dict(run="eval-signs-dev-v1", split="validation", seeds=s, placements="evalsigns",
                      every=3, max_steps=300) for s in _shards(list(range(100, 130)), 5)]
    elif stage == "clean":
        specs = [dict(run="clean-train-v1", split="train", seeds=s, placements=None, every=5,
                      max_steps=1000) for s in _shards(list(range(100)), 10)]
    elif stage == "evalset":
        specs = [dict(run="eval-dev-v1", split="validation", seeds=s, placements="eval", every=3,
                      max_steps=300) for s in _shards(list(range(100, 130)), 5)]
    elif stage == "vlm3b":
        print(json.dumps(vlm_eval_t4.remote(dict(model="Qwen/Qwen2.5-VL-3B-Instruct",
                                                 dtype="float16", tag="qwen25vl-3b-fp16"))))
        return
    elif stage == "pilot-smoke":
        print(json.dumps(closed_loop_l4.remote(dict(
            model="Qwen/Qwen3-VL-8B-Instruct", tag="pilot-smoke", scene="flagger",
            trigger="novelty", seeds=[100]))))
        return
    elif stage in ("pilot-8b", "pilot-32b"):
        fn, model, tag = ((closed_loop_l4, "Qwen/Qwen3-VL-8B-Instruct", "pilot3-qwen3vl-8b")
                          if stage == "pilot-8b" else
                          (closed_loop_a100, "Qwen/Qwen2.5-VL-32B-Instruct", "pilot3-qwen25vl-32b"))
        specs = [dict(model=model, tag=tag, scene=scene, trigger="novelty", seeds=list(range(100, 110)))
                 for scene in ("flagger", "billboard", "pedestrian-shoulder", "lane-closed", "clean")]
        for result in fn.map(specs, return_exceptions=True):
            print(json.dumps(result if isinstance(result, dict) else repr(result)), flush=True)
        return
    elif stage in VLM_RUNS:
        fn, spec = VLM_RUNS[stage]
        print(json.dumps(fn.remote(spec)))
        return
    elif stage == "vlm3b-crop":
        print(json.dumps(vlm_eval_t4.remote(dict(
            model="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", tag="qwen25vl-3b-fp16-crop",
            conditions=["crop", "crop-shuffled"]))))
        return
    elif stage == "bank":
        print(json.dumps(build_bank.remote(encoder), indent=2))
        return
    elif stage == "score":
        print(json.dumps(score_frames.remote(encoder, run=run), indent=2))
        return
    else:
        raise ValueError(f"Unknown stage {stage!r}")
    for result in collect_shard.map(specs):
        print(json.dumps(result), flush=True)
