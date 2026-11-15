"""Retrained camera model, one GPU, inside the full MetaDrive stack.

Ten stall episodes, seeds 9001-9005. final_integrated keeps 8 m of stopping
room; no_reserve turns that room off. Same model, seeds, and controller.
The scheduler's expected wait is 0.125 s. The answer time is the measured
GPU call, not a scripted delay.
"""

import os

import modal

REPO = "/home/harsh/Projects/cs642-event-cbf-driving"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "/vol/ckpt/qwen25vl3b-lora-r16-noskill/merged"
SEEDS = (9001, 9002, 9003, 9004, 9005)
METHODS = ("final_integrated", "no_reserve")
SCENES = ("cut_in", "dense")

app = modal.App("cs642-gpu-margin")
vol = modal.Volume.from_name("vlm-router")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libegl1", "libglib2.0-0")
    .pip_install(
        "vllm==0.33.0",
        "metadrive-simulator==0.4.3",
        "opencv-python-headless",
        "pillow",
        "gymnasium",
    )
    .run_commands("python -m metadrive.pull_asset")
    .env({
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "P3D_DISPLAY": "p3headlessgl",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(os.path.join(REPO, "driving"), "/root/ecd/driving")
    .add_local_dir(os.path.join(REPO, "configs"), "/root/ecd/configs")
    .add_local_dir(os.path.join(REPO, "models", "physical"), "/root/ecd/models/physical")
    .add_local_file(os.path.join(REPO, "tools", "integrated_eval.py"), "/root/ecd/tools/integrated_eval.py")
    .add_local_file(os.path.join(HERE, "tools_init.py"), "/root/ecd/tools/__init__.py")
    .add_local_file(os.path.join(HERE, "prompts.py"), "/root/promptsrc/prompts.py")
)


@app.function(
    image=image,
    volumes={"/vol": vol},
    gpu=["L4", "A10G"],
    cpu=4,
    memory=16384,
    timeout=3600,
)
def ablation():
    import hashlib
    import json
    import sys
    import time

    import torch

    os.chdir("/root/ecd")
    sys.path[:0] = ["/root/ecd", "/root/promptsrc"]
    torch.set_num_threads(1)

    from prompts import LETTERS, MAX_PIXELS, MIN_PIXELS, messages
    from vllm import LLM, SamplingParams

    from driving.controller import DrivingStack
    from driving.env_registry import make_env
    from driving.errors import RouterError
    from driving.maneuver_reserve import ManeuverReserveShield
    from driving.router_base import RouterBackend, RouterResponse, validate_response
    import tools.integrated_eval as evaluation

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        enable_prefix_caching=True,
        gpu_memory_utilization=0.55,
        seed=0,
    )
    tok = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        allowed_token_ids=[tok.encode(c, add_special_tokens=False)[0] for c in LETTERS],
    )
    choices = {"A": "CRUISE_RIGHT", "B": "OVERTAKE_LEFT", "C": "RETURN_RIGHT"}

    class GPURouter(RouterBackend):
        name = "retrained_vlm_gpu_noskill"
        requires_camera = True
        is_live_capable = True

        def __init__(self):
            super().__init__()
            self.predictions = []

        def submit(self, req):
            if req.camera is None:
                raise RouterError("camera frame required")
            if len(self.predictions) >= 20:
                raise RouterError("twenty-call episode budget exhausted")
            import base64

            uri = "data:image/png;base64," + base64.b64encode(req.camera.image_bytes).decode()
            started = time.perf_counter()
            text = llm.chat(
                messages(uri, {"ego_lane": req.camera.lane, "current_skill": req.current_skill}),
                sampling,
                use_tqdm=False,
            )[0].outputs[0].text.strip()[:1]
            latency = time.perf_counter() - started
            if text not in choices:
                raise RouterError(f"model returned {text!r}")
            record = {
                "letter": text,
                "skill": choices[text],
                "latency_s": latency,
                "request_t": req.t,
                "ego_lane": req.camera.lane,
                "speed_mps": req.camera.speed_mps,
                "prompt_speed": "20 m/s pinned; current skill omitted",
            }
            self.predictions.append(record)
            print(
                f"  call t={req.t:.2f}s lane={req.camera.lane} letter={text} {latency * 1000:.0f} ms",
                flush=True,
            )
            response = RouterResponse(
                skill=choices[text],
                ready_at=req.t + latency,
                backend=self.name,
                raw_text=text,
                latency_s=latency,
                request_id=req.request_id,
                submitted_t=req.t,
                wall_latency_s=latency,
                fallback_used=False,
                live=True,
                model=MODEL,
                usage={"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
                finish_reason="constrained_one_token",
                image_hash=hashlib.sha256(req.camera.image_bytes).hexdigest(),
                image_captured_t=req.camera.captured_t,
                image_age_s=req.t - req.camera.captured_t,
                request_attempts=1,
            )
            validate_response(response, req)
            self._record_success(latency, usage=response.usage)
            response.health = self.health()
            return response

    evaluation.initialize()
    rows = []
    for scene in SCENES:
        for method in METHODS:
            for seed in SEEDS:
                router = GPURouter()
                job = evaluation.Job(scene, seed, 0.125, method, sigma_scale=1.0)
                print(f"start {scene} {method} seed {seed}", flush=True)
                try:
                    cfg, reserve_mode = evaluation.configs(job, live=True)
                    # vLLM is not safe on the controller's worker thread. The sim
                    # thread blocks for the measured call, and realtime pacing keeps
                    # that wait on the simulation clock.
                    cfg.stack.router_async = False
                    shield = ManeuverReserveShield(cfg.shield, mode=reserve_mode, reserve_m=8.0)
                    stack = DrivingStack(evaluation.BANK, router, shield=shield, config=cfg.stack)
                    spec = evaluation.scene_spec(job, camera=True)
                    env = make_env(spec.name, spec)
                    started = time.perf_counter()
                    try:
                        result = stack.run(env, seed=job.seed)
                    finally:
                        env.close()
                    metrics = evaluation.trace_metrics(result)
                    out = {
                        "scene": scene,
                        "method": method,
                        "seed": seed,
                        "reserve_mode": reserve_mode,
                        "pending_margin": bool(stack.last_shield.pending_margin),
                        "route_pct": result.route_pct,
                        "crashed": result.crashed,
                        "success": (not result.crashed) and result.route_pct >= 99,
                        "freeze": result.freeze,
                        "end_reason": result.end_reason,
                        "end_time_s": result.end_time,
                        "wall_s": time.perf_counter() - started,
                        "router_calls": result.router_calls,
                        "calls_applied": result.calls_applied,
                        "calls_rejected": result.calls_rejected,
                        "stopped_s": metrics["stopped_s"],
                        "terminal_stall": metrics["terminal_stall"],
                        "pending_survival": result.pending_survival,
                        "cut_in_observed": metrics["cut_in_observed"],
                        "letters": [p["letter"] for p in router.predictions],
                        "call_latency_ms": [round(p["latency_s"] * 1000, 1) for p in router.predictions],
                        "call_lanes": [p["ego_lane"] for p in router.predictions],
                    }
                except Exception:
                    import traceback
                    out = {"scene": scene, "method": method, "seed": seed, "error": traceback.format_exc(limit=8)}
                    print(f"FAIL {scene} {method} seed {seed}: {out['error']}", flush=True)
                else:
                    print(
                        f"done {scene} {method} seed {seed}: route={out['route_pct']:.1f} "
                        f"crash={out['crashed']} stopped_s={out['stopped_s']:.1f} "
                        f"letters={out['letters']}",
                        flush=True,
                    )
                rows.append(out)
    gpu = torch.cuda.get_device_name(0)
    summary = {"gpu": gpu, "model": MODEL, "episodes": rows}
    os.makedirs("/vol/results/qwen25vl3b-lora-r16-noskill", exist_ok=True)
    with open("/vol/results/qwen25vl3b-lora-r16-noskill/full_stack_cutin_dense.json", "w") as f:
        json.dump(summary, f)
    vol.commit()
    return summary


@app.local_entrypoint()
def main():
    import json

    summary = ablation.remote()
    out = os.path.join(HERE, "full_stack_cutin_dense.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print("saved", out)
