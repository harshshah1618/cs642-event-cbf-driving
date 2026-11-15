"""Modal pipeline for the VLM router: render data -> LoRA fine-tune -> evaluate (base vs fine-tuned).

  modal run modal/modal_app.py --smoke                # tiny end-to-end check (~a few minutes, cents)
  modal run --detach modal/modal_app.py               # full run: data, train, eval
  modal run modal/modal_app.py --stage eval           # re-run one stage (data | train | eval | all)
  modal run modal/modal_app.py --stage ood            # harder held-out test (curves, lane changes, ...)
  modal run modal/modal_app.py --stage loop --seeds 5   # closed loop: rule vs delayed rule vs VLM (two-trap)
  modal run modal/modal_app.py --stage export             # copy the trained adapter (bf16) into modal/models/
  modal run modal/modal_app.py --stage merge --adapter qwen25vl3b-lora-r16 --run-name <run>
                                                           # rebuild merged weights on YOUR volume from the repo adapter
  modal volume get vlm-router results/<run> .             # fetch results of a detached run

GPUs: training on L40S (48 GB, best bf16 throughput per dollar on Modal for a 3B LoRA; no gradient
checkpointing needed). Evaluation on L4, the same GPU as the Colab rounds, so latencies stay comparable.
Frames are rendered on T4 containers (MetaDrive camera, headless NVIDIA EGL), a few dozen in parallel.

Volumes: `vlm-router` holds /vol/data/<dataset>/<split>/ and /vol/ckpt/<run>/; `vlm-router-hf` caches HF weights.
W&B: training curves go to project `cs642-vlm-router` via the `wandb` Modal secret.
"""
import json
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))  # modal/
REPO = os.path.dirname(HERE)
ROUTER = os.path.join(REPO, "vlm-router")  # router code: frame generators, prompt, training, eval, render bridge
STACK = os.path.join(REPO, "event-cbf-driving")  # the driving stack (skills, shield, envs)
MODELS = os.path.join(HERE, "models")  # trained adapters kept in git (bf16, < GitHub's 100 MB file limit)
SRC = "/root/vlm_router"
ECD = "/root/ecd"
KINDS = ["clear", "left_only", "overtake", "blocked", "return", "keep_left"]
R2_KINDS = "clear,overtake,blocked,return,keep_left"  # round-2 test set, reproduced exactly (variety 0, seed 1)
BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
OOD_CATS = ["curve", "lanechange", "oncoming", "far", "close", "beside", "multi"]  # see gen_ood.py

app = modal.App("cs642-vlm-router")
vol = modal.Volume.from_name("vlm-router", create_if_missing=True)
hf = modal.Volume.from_name("vlm-router-hf", create_if_missing=True)
VOLS = {"/vol": vol, "/hf": hf}
# This account has no W&B secret. Keep this list empty in both the local and remote
# process. A conditional here makes the training container crash and restart.
# Inference stays on L4, then A10. A100 is left out because it costs more than twice an L4.
INFER_GPUS = ["L4", "A10G"]
SECRETS = []


def _src(img):
    for f in ("gen_frames.py", "gen_ood.py", "prompts.py", "train_lora.py", "eval_router.py", "render_bridge.py",
              "vlm_policy.py", "loop_eval.py"):
        img = img.add_local_file(os.path.join(ROUTER, f), f"{SRC}/{f}")
    return img


sim_image = _src(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libegl1", "libglib2.0-0")
    .pip_install("metadrive-simulator==0.4.3", "opencv-python-headless", "pillow")
    .run_commands("python -m metadrive.pull_asset")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all"})  # expose the NVIDIA EGL/OpenGL libraries, not just CUDA
)
train_image = _src(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.1", "peft==0.17.1",
                 "accelerate==1.10.1", "wandb", "pillow")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
)
vllm_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.33.0", "wandb", "pillow")
    # FlashInfer's sampler JIT-compiles with nvcc, which the slim image lacks; we only decode one constrained token.
    .env({"HF_HOME": "/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
vllm_image = _src(vllm_base)
# closed loop: vLLM + the driving stack (PPO skills run on CPU torch, already in the vLLM image)
loop_image = (_src(vllm_base.pip_install("gymnasium", "matplotlib"))
              .add_local_dir(os.path.join(STACK, "event_cbf_driving"), f"{ECD}/event_cbf_driving",
                             ignore=["**/__pycache__"])
              .add_local_dir(os.path.join(STACK, "checkpoints"), f"{ECD}/checkpoints"))


@app.function(image=sim_image, volumes=VOLS, gpu="T4", cpu=2, memory=8192, timeout=3600, retries=1)
def gen_shard(out_rel: str, kinds: str, n: int, seed: int, variety: int, prefix: str):
    # Headless GPU rendering through NVIDIA EGL (~0.05 s/frame). Software OpenGL on CPU took >45 s per step.
    out = f"/vol/data/{out_rel}"
    cmd = ["python", f"{SRC}/gen_frames.py", "--view", "camera", "--out", out, "--kinds", kinds, "--n", str(n),
           "--seed", str(seed), "--variety", str(variety), "--prefix", prefix]
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, P3D_DISPLAY="p3headlessgl"))
    if p.returncode != 0:
        raise RuntimeError(f"gen_frames failed for {out_rel} {prefix}:\n{(p.stderr or p.stdout)[-3000:]}")
    labels = json.load(open(f"{out}/{prefix}labels.json"))
    vol.commit()
    return out_rel, labels


@app.function(image=sim_image, volumes=VOLS, gpu="T4", cpu=2, memory=8192, timeout=3600, retries=1)
def gen_ood_shard(cat: str, n: int, seed: int, out_rel: str):
    out = f"/vol/data/{out_rel}"
    cmd = ["python", f"{SRC}/gen_ood.py", "--cat", cat, "--out", out, "--n", str(n), "--seed", str(seed)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, P3D_DISPLAY="p3headlessgl"))
    if p.returncode != 0:
        raise RuntimeError(f"gen_ood failed for {cat}:\n{(p.stderr or p.stdout)[-3000:]}")
    vol.commit()
    return cat, len(json.load(open(f"{out}/labels.json")))


@app.function(image=sim_image, volumes=VOLS, timeout=600)
def write_labels(out_rel: str, labels: list):
    vol.reload()
    json.dump(labels, open(f"/vol/data/{out_rel}/labels.json", "w"), indent=1)
    vol.commit()
    return len(labels)


@app.function(image=sim_image, volumes=VOLS, timeout=120)
def existing(paths: list):
    vol.reload()
    return [p for p in paths if os.path.exists(f"/vol/data/{p}/labels.json")]


@app.function(image=train_image, volumes=VOLS, gpu="L40S", cpu=8, memory=32768, secrets=SECRETS,
              timeout=4 * 3600)
def train(dataset: str, run_name: str, extra: list):
    vol.reload()
    sys.path.insert(0, SRC)
    import train_lora

    argv = ["--data", f"/vol/data/{dataset}", "--out", f"/vol/ckpt/{run_name}", "--run-name", run_name] + extra
    summary = train_lora.main(argv)
    vol.commit()
    return summary


@app.function(image=vllm_image, volumes=VOLS, gpu=INFER_GPUS, cpu=4, memory=16384, secrets=SECRETS, timeout=3600)
def evaluate(model: str, sets: str, tag: str, limit: int = 0, wandb_run: str = ""):
    vol.reload()
    out = "/tmp/res.json"
    p = subprocess.run([sys.executable, f"{SRC}/eval_router.py", "--model", model, "--sets", sets, "--out", out,
                        "--tag", tag, "--limit", str(limit)], cwd=SRC)
    if p.returncode != 0:
        raise RuntimeError(f"eval failed for {tag}")
    res = json.load(open(out))
    if wandb_run:
        import wandb

        run = wandb.init(project="cs642-vlm-router", name=f"eval-{wandb_run}-{tag}", job_type="eval",
                         config=dict(model=model, tag=tag, gpu=res["gpu"]))
        for name, s in res["sets"].items():
            summ = s["summary"]
            run.summary.update({f"{name}/acc": summ["acc"], f"{name}/median_ms": summ["median_ms"],
                                f"{name}/p90_ms": summ["p90_ms"]})
            run.summary.update({f"{name}/kind/{k}": v for k, v in summ["per_kind"].items()})
        run.finish()
    return res


@app.cls(image=sim_image, gpu="T4", cpu=2, memory=8192, timeout=3600, scaledown_window=300)
class Renderer:
    """Render bridge (render_bridge.py) as a service: scene dict -> PNG of the MetaDrive front camera."""

    @modal.enter()
    def start(self):
        os.environ["P3D_DISPLAY"] = "p3headlessgl"
        sys.path.insert(0, SRC)
        from render_bridge import Bridge

        self.bridge = Bridge()

    @modal.method()
    def render(self, scene: dict) -> bytes:
        return self.bridge.render_png(scene)


@app.function(image=loop_image, volumes=VOLS, gpu=INFER_GPUS, cpu=4, memory=16384, timeout=3600)
def closed_loop(run_name: str, seeds: list, tag: str):
    """VLM in this container (same L4 + vLLM setup as the evals); frames come from the Renderer on a T4."""
    vol.reload()
    # run the vLLM engine in this process: no fork/spawn of the Modal function runtime
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path[:0] = [SRC, ECD]
    import loop_eval
    from vlm_policy import VLLMPolicy

    policy = VLLMPolicy(f"/vol/ckpt/{run_name}/merged")
    renderer = Renderer()
    out = f"/vol/results/{run_name}/closed_loop_{tag}"
    import torch  # CUDA is already up inside vLLM (in-process engine), so this does not change the start method

    meta = dict(gpu=torch.cuda.get_device_name(0), model=f"/vol/ckpt/{run_name}/merged", seeds=seeds)
    res = loop_eval.run(policy, lambda scene: renderer.render.remote(scene), seeds, out, f"{ECD}/checkpoints", meta)
    vol.commit()
    return res, open(f"{out}/trajectories.png", "rb").read()


@app.function(image=train_image, volumes=VOLS, cpu=4, memory=8192, timeout=1200)
def export_adapter(run_name: str):
    """Trained adapter -> bf16 copy small enough for git. Returns {file name: bytes}."""
    import torch
    from safetensors.torch import load_file, save_file

    d = f"/vol/ckpt/{run_name}/adapter"
    files = {}
    for f in os.listdir(d):
        if f == "adapter_model.safetensors":
            tensors = {k: v.to(torch.bfloat16).contiguous() for k, v in load_file(f"{d}/{f}").items()}
            save_file(tensors, "/tmp/a.safetensors", metadata={"format": "pt"})
            files[f] = open("/tmp/a.safetensors", "rb").read()
        else:
            files[f] = open(f"{d}/{f}", "rb").read()
    summ = f"/vol/ckpt/{run_name}/train_summary.json"
    if os.path.exists(summ):
        files["train_summary.json"] = open(summ, "rb").read()
    return files


@app.function(image=train_image, volumes=VOLS, cpu=8, memory=32768, timeout=3600)
def merge(adapter_files: dict, run_name: str):
    """Base model + adapter (from the repo) -> /vol/ckpt/<run_name>/{adapter,merged}, the layout eval/loop expect."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    sys.path.insert(0, SRC)
    from prompts import MAX_PIXELS, MIN_PIXELS

    vol.reload()
    out = f"/vol/ckpt/{run_name}"
    os.makedirs(f"{out}/adapter", exist_ok=True)
    for f, b in adapter_files.items():
        open(f"{out}/{f}" if f == "train_summary.json" else f"{out}/adapter/{f}", "wb").write(b)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, f"{out}/adapter").merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(f"{out}/merged", safe_serialization=True)
    AutoProcessor.from_pretrained(BASE, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS).save_pretrained(f"{out}/merged")
    vol.commit()
    return sorted(os.listdir(f"{out}/merged"))


def data_jobs(dataset, per_kind, shard=200):
    """(out_rel, kinds, n, seed, variety, prefix) per CPU container."""
    jobs = []
    plan = [("train", per_kind, 1000), ("val", max(8, per_kind // 15), 5000), ("test", max(8, per_kind // 12), 9000)]
    for split, n, seed0 in plan:
        for ki, kind in enumerate(KINDS):
            for si, start in enumerate(range(0, n, shard)):
                jobs.append((f"{dataset}/{split}", kind, min(shard, n - start), seed0 + 100 * ki + si, 1, f"{kind}_s{si}_"))
    jobs.append(("r2test/test", R2_KINDS, 10, 1, 0, ""))
    return jobs


@app.function(image=sim_image, volumes=VOLS, timeout=8 * 3600)
def pipeline(stage: str, dataset: str, run_name: str, per_kind: int, train_extra: list, limit: int,
             skip_base: bool = False, ood_n: int = 60, ood_set: str = "ood1", also_loop: bool = False,
             seeds: int = 5, loop_tag: str = "two_trap"):
    """Runs on Modal (not the laptop), so a `modal run --detach` survives the laptop sleeping.
    Results land in /vol/results/<run>/ and are also returned."""
    rdir = f"/vol/results/{run_name}"
    os.makedirs(rdir, exist_ok=True)
    results = {}

    if stage in ("data", "all"):
        jobs = data_jobs(dataset, per_kind)
        done = set(existing.remote(sorted({j[0] for j in jobs})))
        jobs = [j for j in jobs if j[0] not in done]
        print(f"data: {len(jobs)} render shards ({'cached: ' + str(sorted(done)) if done else 'nothing cached'})", flush=True)
        merged = {}
        for out_rel, labels in gen_shard.starmap(jobs):
            merged.setdefault(out_rel, []).extend(labels)
        for out_rel, labels in merged.items():
            print(f"  {out_rel}: {write_labels.remote(out_rel, labels)} frames", flush=True)

    if stage in ("train", "all"):
        summary = train.remote(dataset, run_name, train_extra)
        results["train_summary"] = summary
        print("train: final", summary["final"], f"({summary['train_minutes']} min on {summary['gpu']})", flush=True)

    if stage in ("eval", "all"):
        sets = f"r2test=/vol/data/r2test/test,test=/vol/data/{dataset}/test"
        calls = {"finetuned": evaluate.spawn(f"/vol/ckpt/{run_name}/merged", sets, "finetuned", limit, run_name)}
        if not skip_base:
            calls["base"] = evaluate.spawn(BASE, sets, "base", limit, run_name)
        for tag, call in calls.items():
            res = call.get()
            results[f"eval_{tag}"] = res
            for name, s in res["sets"].items():
                summ = s["summary"]
                print(f"eval {tag:9s} {name:6s} acc {summ['acc']:.3f}  median {summ['median_ms']} ms  "
                      f"p90 {summ['p90_ms']} ms  per kind {summ['per_kind']}", flush=True)

    if stage == "ood":  # harder held-out frames (gen_ood.py); never used for training
        rels = {c: f"{ood_set}/{c}" for c in OOD_CATS}
        done = set(existing.remote(list(rels.values())))
        jobs = [(c, ood_n, 7000 + i, rel) for i, (c, rel) in enumerate(rels.items()) if rel not in done]
        print(f"ood: rendering {len(jobs)} categories ({'cached: ' + str(sorted(done)) if done else 'nothing cached'})",
              flush=True)
        for cat, n in gen_ood_shard.starmap(jobs):
            print(f"  {cat}: {n} frames", flush=True)
        sets = ",".join(f"{c}=/vol/data/{rel}" for c, rel in rels.items())
        calls = {"ood_finetuned": evaluate.spawn(f"/vol/ckpt/{run_name}/merged", sets, "ood-finetuned", limit, run_name)}
        if not skip_base:
            calls["ood_base"] = evaluate.spawn(BASE, sets, "ood-base", limit, run_name)
        for tag, call in calls.items():
            res = call.get()
            results[f"eval_{tag}"] = res
            for name, s in res["sets"].items():
                summ = s["summary"]
                print(f"{tag:14s} {name:10s} acc {summ['acc']:.3f}  median {summ['median_ms']} ms  "
                      f"per kind {summ['per_kind']}", flush=True)

    loop_png = None
    if also_loop and stage in ("all", "train", "eval"):
        res, loop_png = closed_loop.remote(run_name, list(range(seeds)), loop_tag)
        results["closed_loop"] = res

    vol.reload()
    os.makedirs(rdir, exist_ok=True)
    for k, v in results.items():
        json.dump(v, open(f"{rdir}/{k}.json", "w"), indent=1)
    if loop_png is not None:
        open(f"{rdir}/trajectories.png", "wb").write(loop_png)
    vol.commit()
    return results


@app.local_entrypoint()
def main(stage: str = "all", dataset: str = "v1", run_name: str = "qwen25vl3b-lora-r16", per_kind: int = 600,
         epochs: float = 2.0, smoke: bool = False, skip_base: bool = False, ood_n: int = 60, seeds: int = 5,
         loop_tag: str = "two_trap", adapter: str = "qwen25vl3b-lora-r16", also_loop: bool = False):
    if stage == "export":
        out = os.path.join(MODELS, run_name, "adapter")
        os.makedirs(out, exist_ok=True)
        for f, b in export_adapter.remote(run_name).items():
            dst = os.path.join(MODELS, run_name, f) if f == "train_summary.json" else os.path.join(out, f)
            open(dst, "wb").write(b)
            print(f"  {dst}  {len(b) / 2**20:.1f} MiB")
        return
    if stage == "merge":
        src = os.path.join(MODELS, adapter)
        files = {f: open(os.path.join(src, "adapter", f), "rb").read() for f in os.listdir(os.path.join(src, "adapter"))}
        if os.path.exists(os.path.join(src, "train_summary.json")):
            files["train_summary.json"] = open(os.path.join(src, "train_summary.json"), "rb").read()
        print("merged:", merge.remote(files, run_name), f"-> /vol/ckpt/{run_name}/merged")
        return
    if stage == "loop":
        res, png = closed_loop.remote(run_name, list(range(seeds)), loop_tag)
        out = os.path.join(ROUTER, "results", run_name, f"closed_loop_{loop_tag}")
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, "closed_loop.json"), "w"), indent=1)
        open(os.path.join(out, "trajectories.png"), "wb").write(png)
        print("saved to", out, "(frames the VLM saw: /vol/results/" + run_name + f"/closed_loop_{loop_tag}/frames)")
        return
    train_extra = ["--epochs", str(epochs)]
    limit = 0
    if smoke:
        dataset, run_name, per_kind, limit = "smoke", "smoke", 8, 12
        train_extra = ["--max-steps", "4", "--eval-every", "2", "--bs", "2", "--accum", "2"]
    if os.environ.get("CS642_NO_WANDB") == "1":
        train_extra = train_extra + ["--wandb", "0"]
    results = pipeline.remote(stage, dataset, run_name, per_kind, train_extra, limit, skip_base, ood_n,
                              also_loop=also_loop, seeds=seeds, loop_tag=loop_tag)
    out = os.path.join(ROUTER, "results", run_name)
    os.makedirs(out, exist_ok=True)
    for k, v in results.items():
        json.dump(v, open(os.path.join(out, f"{k}.json"), "w"), indent=1)
    print("results saved to", out, "(also on the volume: /vol/results/" + run_name + ")")
