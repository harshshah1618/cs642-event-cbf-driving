"""Modal pipeline for the VLM router: render data -> LoRA fine-tune -> evaluate (base vs fine-tuned).

  modal run vlm-router/modal_app.py --smoke                # tiny end-to-end check (~a few minutes, cents)
  modal run --detach vlm-router/modal_app.py               # full run: data, train, eval
  modal run vlm-router/modal_app.py --stage eval           # re-run one stage (data | train | eval | all)
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

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/root/vlm_router"
KINDS = ["clear", "left_only", "overtake", "blocked", "return", "keep_left"]
R2_KINDS = "clear,overtake,blocked,return,keep_left"  # round-2 test set, reproduced exactly (variety 0, seed 1)
BASE = "Qwen/Qwen2.5-VL-3B-Instruct"

app = modal.App("cs642-vlm-router")
vol = modal.Volume.from_name("vlm-router", create_if_missing=True)
hf = modal.Volume.from_name("vlm-router-hf", create_if_missing=True)
VOLS = {"/vol": vol, "/hf": hf}
wandb_secret = modal.Secret.from_name("wandb")


def _src(img):
    for f in ("gen_frames.py", "prompts.py", "train_lora.py", "eval_router.py"):
        img = img.add_local_file(os.path.join(HERE, f), f"{SRC}/{f}")
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
vllm_image = _src(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.33.0", "wandb", "pillow")
    # FlashInfer's sampler JIT-compiles with nvcc, which the slim image lacks; we only decode one constrained token.
    .env({"HF_HOME": "/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


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


@app.function(image=train_image, volumes=VOLS, gpu="L40S", cpu=8, memory=32768, secrets=[wandb_secret],
              timeout=4 * 3600)
def train(dataset: str, run_name: str, extra: list):
    vol.reload()
    sys.path.insert(0, SRC)
    import train_lora

    argv = ["--data", f"/vol/data/{dataset}", "--out", f"/vol/ckpt/{run_name}", "--run-name", run_name] + extra
    summary = train_lora.main(argv)
    vol.commit()
    return summary


@app.function(image=vllm_image, volumes=VOLS, gpu="L4", cpu=4, memory=16384, secrets=[wandb_secret], timeout=3600)
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
             skip_base: bool = False):
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

    vol.reload()
    for k, v in results.items():
        json.dump(v, open(f"{rdir}/{k}.json", "w"), indent=1)
    vol.commit()
    return results


@app.local_entrypoint()
def main(stage: str = "all", dataset: str = "v1", run_name: str = "qwen25vl3b-lora-r16", per_kind: int = 600,
         epochs: float = 2.0, smoke: bool = False, skip_base: bool = False):
    train_extra = ["--epochs", str(epochs)]
    limit = 0
    if smoke:
        dataset, run_name, per_kind, limit = "smoke", "smoke", 8, 12
        train_extra = ["--max-steps", "4", "--eval-every", "2", "--bs", "2", "--accum", "2"]
    results = pipeline.remote(stage, dataset, run_name, per_kind, train_extra, limit, skip_base)
    out = os.path.join(HERE, "results", run_name)
    os.makedirs(out, exist_ok=True)
    for k, v in results.items():
        json.dump(v, open(os.path.join(out, f"{k}.json"), "w"), indent=1)
    print("results saved to", out, "(also on the volume: /vol/results/" + run_name + ")")
