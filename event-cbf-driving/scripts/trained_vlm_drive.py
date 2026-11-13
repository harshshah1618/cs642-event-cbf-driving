"""One exploratory physical-camera drive with the actual trained CPU router.

This slow-CPU pilot is not the teammate's GPU benchmark or a paired ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.errors import RouterError
from event_cbf_driving.eval.runner import load_skill_bank
from event_cbf_driving.router.base import RouterBackend, RouterResponse, validate_response
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.stack.controller import DrivingStack

from trained_vlm_cpu import TrainedVLMCPU

ROOT = Path(__file__).resolve().parents[1]
CHOICES = {"A": "CRUISE_RIGHT", "B": "OVERTAKE_LEFT", "C": "RETURN_RIGHT"}


class ActualCPURouter(RouterBackend):
    name = "actual_trained_vlm_cpu"
    requires_camera = True
    is_live_capable = True

    def __init__(self, policy, output):
        super().__init__()
        self.policy = policy
        self.output = output
        self.predictions = []
        self.latency_s = 12.0  # Initial description only; every answer uses measured time.

    def submit(self, req):
        if req.camera is None:
            raise RouterError("actual trained router requires a captured image")
        if len(self.predictions) >= 8:
            raise RouterError("explicit eight-prediction CPU pilot budget exhausted")
        started = time.perf_counter()
        try:
            camera = req.camera
            prediction = self.policy.predict(camera.image_bytes, camera.lane, req.current_skill)
            if prediction["letter"] not in CHOICES or prediction.get("fallback_used"):
                raise RouterError("invalid trained-model prediction; no substitution")
            latency = time.perf_counter() - started
            self.latency_s = latency
            index = len(self.predictions)
            frame = self.output / "frames" / f"request-{index:03d}.png"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(camera.image_bytes)
            record = {
                **prediction,
                "request_t": req.t,
                "request_id": req.request_id,
                "current_skill": req.current_skill,
                "ego_lane": camera.lane,
                "frame": str(frame.name),
                "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
            self.predictions.append(record)
            with (self.output / "predictions.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            tokens = int(prediction["input_tokens"])
            usage = {"prompt_tokens": tokens, "completion_tokens": 1, "total_tokens": tokens + 1}
            response = RouterResponse(
                skill=CHOICES[prediction["letter"]],
                ready_at=req.t + latency,
                backend=self.name,
                raw_text=prediction["letter"],
                latency_s=latency,
                request_id=req.request_id,
                submitted_t=req.t,
                wall_latency_s=latency,
                fallback_used=False,
                live=True,
                model="Qwen2.5-VL-3B+teammate-BF16-LoRA",
                usage=usage,
                finish_reason="constrained_one_token_forward",
                image_hash=hashlib.sha256(camera.image_bytes).hexdigest(),
                image_captured_t=camera.captured_t,
                image_age_s=req.t - camera.captured_t,
                request_attempts=1,
            )
            validate_response(response, req)
            self._record_success(latency, usage=usage)
            response.health = self.health()
            return response
        except Exception as exc:
            self._record_failure(exc, latency_s=time.perf_counter() - started)
            if isinstance(exc, RouterError):
                raise
            raise RouterError(f"actual CPU inference failed: {exc}") from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "runs/camera-pilot")
    p.add_argument("--seed", type=int, default=6301)
    p.add_argument("--duration", type=float, default=60)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output directory; do not overwrite pilot evidence")
    args.output.mkdir(parents=True)
    policy = TrainedVLMCPU(
        args.base,
        ROOT / "checkpoints/vlm/adapter",
        ROOT / "checkpoints/vlm/prompts.py",
        threads=4,
    )
    cfg = load_config(ROOT / "configs/metadrive-study.yaml")
    cfg.stack.mode = "fixed_rate_shielded"
    cfg.stack.fixed_rate_hz = 2.0
    cfg.stack.router_async = cfg.stack.realtime = True
    cfg.stack.max_pending_s = 30.0
    cfg.stack.pending_margin = cfg.shield.pending_margin = False
    cfg.stack.record_interactions = True
    cfg.shield.gamma_sigma = 0.0
    cfg.validate()
    bank, checkpoints = load_skill_bank(ROOT / "checkpoints/metadrive-calibrated")
    router = ActualCPURouter(policy, args.output)
    stack = DrivingStack(
        bank, router, shield=InflatedCBFShield(config=cfg.shield), config=cfg.stack
    )
    env = make_env(
        "two_trap",
        ScenarioConfig(
            "two_trap",
            simulator="metadrive",
            duration=args.duration,
            randomize=True,
            params={"rgb_camera": True},
        ),
    )
    started = time.perf_counter()
    try:
        result = stack.run(env, seed=args.seed)
    finally:
        env.close()
    wall = time.perf_counter() - started
    (args.output / "episode.json").write_text(json.dumps(result.to_dict(), allow_nan=False) + "\n")
    summary = {
        "kind": "actual_trained_cpu_physical_camera_pilot",
        "actual_model": True,
        "seed": args.seed,
        "route_pct": result.route_pct,
        "crashed": result.crashed,
        "success": not result.crashed and result.route_pct >= 99,
        "freeze": result.freeze,
        "end_time_sim_s": result.end_time,
        "episode_wall_s_including_render_startup": wall,
        "real_model_predictions": len(router.predictions),
        "recorded_live_calls": result.router_live_calls,
        "calls_applied": result.calls_applied,
        "calls_rejected": result.calls_rejected,
        "fallbacks": result.router_fallbacks,
        "skills_used": sorted(set(result.skills)),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "prediction_times_s": [r["latency_s"] for r in router.predictions],
        "adapter_audit": policy.audit,
        "checkpoints": checkpoints,
        "limitations": [
            "One exploratory seed, CPU inference, v1 parked-car/constant20 prompt on dynamic physical scenes",
            "No paired ablation or GPU latency/safety/generalization conclusion",
            "Paced simulation can fall behind wall clock; full wall and simulator times retained",
            "Eight actual predictions maximum; reject errors/budget exhaustion without scripted substitution",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
