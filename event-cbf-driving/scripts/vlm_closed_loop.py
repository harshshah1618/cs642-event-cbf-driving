"""Run an explicitly supplied VLM server in the physical driving stack.

--contract-smoke uses programmed answers over loopback HTTP, not a model.
The server/model arguments are required for actual inference; no substitutions.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from event_cbf_driving.config import ScenarioConfig, load_config
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.eval.runner import load_skill_bank
from event_cbf_driving.router.factory import create_router
from event_cbf_driving.router.openai_compat import UrllibTransport
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.skills.classical import PDLanePolicy
from event_cbf_driving.stack.controller import DrivingStack

ROOT = Path(__file__).resolve().parents[1]


def loopback_fixture(output, permit_skill_hint=False):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            blocks = body["messages"][1]["content"]
            image = base64.b64decode(blocks[0]["image_url"]["url"].split(",", 1)[1])
            text = blocks[1]["text"]
            if ("skill" in text.lower() and not permit_skill_hint) or "occupied" in text.lower():
                raise ValueError("privileged instruction leaked to image router")
            frame = output / "frames" / f"request-{len(calls):03d}.png"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(image)
            calls.append(
                {
                    "image_sha256": hashlib.sha256(image).hexdigest(),
                    "ego_text": text,
                    "answer": "B",
                    "kind": "programmed_contract_fixture",
                    "intentional_v1_skill_hint": permit_skill_hint,
                }
            )
            time.sleep(0.04)
            response = json.dumps(
                {
                    "model": "contract-fixture-not-a-model",
                    "choices": [{"message": {"content": "B"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server")
    parser.add_argument("--model")
    parser.add_argument("--prompt-mode", choices=("clean", "teammate-v1"), default="clean")
    parser.add_argument("--contract-smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[5101])
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=["two_trap", "dense_highway", "merge"],
        default=["two_trap", "dense_highway", "merge"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["full", "no_cbf", "no_braking", "no_admission", "pd"],
        default=["full", "no_cbf", "no_braking", "no_admission", "pd"],
    )
    parser.add_argument("--timeout", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/components-vlm")
    args = parser.parse_args()
    if not args.contract_smoke and (not args.server or not args.model):
        parser.error("actual inference requires --server and --model; no trained model is bundled")
    if args.contract_smoke and (args.server or args.model):
        parser.error("contract smoke cannot be mixed with actual model configuration")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        parser.error("choose a new output directory; existing evidence is not overwritten")
    bank, checkpoint_info = load_skill_bank(ROOT / "checkpoints/metadrive-calibrated")
    server = thread = None
    fixture_calls = []
    if args.contract_smoke:
        server, thread, fixture_calls = loopback_fixture(
            args.output, args.prompt_mode == "teammate-v1"
        )
        args.server = f"http://127.0.0.1:{server.server_port}/v1"
        args.model = "contract-fixture-not-a-model"
        args.scenes, args.variants, args.seeds = ["dense_highway"], ["full"], [5101]
    summaries = []
    try:
        for scene in args.scenes:
            for seed in args.seeds:
                for variant in args.variants:
                    cfg = load_config(ROOT / "configs/metadrive-study.yaml")
                    cfg.router.backend, cfg.router.allow_live = "vlm", True
                    cfg.router.base_url, cfg.router.model = args.server, args.model
                    cfg.router.timeout_s = args.timeout
                    cfg.router.max_retries = 0
                    cfg.router.allow_fallback = False
                    cfg.router.api_key_env = cfg.router.api_key_env_fallback = (
                        "EVENT_CBF_VLM_API_KEY"
                    )
                    cfg.stack.mode = "fixed_rate_shielded"
                    cfg.stack.fixed_rate_hz = 2
                    cfg.stack.realtime = cfg.stack.router_async = True
                    cfg.stack.max_pending_s = args.timeout
                    cfg.stack.pending_margin = cfg.shield.pending_margin = False
                    cfg.stack.record_interactions = True
                    cfg.shield.gamma_sigma = 0
                    cfg.shield.cbf_constraint_enabled = variant != "no_cbf"
                    cfg.shield.braking_guard = variant != "no_braking"
                    cfg.stack.skill_switch_guard = variant != "no_admission"
                    cfg.validate()
                    overrides = (
                        {"transport": UrllibTransport(), "transport_is_live": False}
                        if args.contract_smoke
                        else {}
                    )
                    if args.prompt_mode == "teammate-v1":
                        from teammate_vlm_backend import TeammateV1VLMRouter

                        router = TeammateV1VLMRouter(cfg.router, **overrides)
                    else:
                        router = create_router(cfg.router, **overrides)
                    policies = (
                        bank if variant != "pd" else {name: PDLanePolicy(name) for name in bank}
                    )
                    spec = ScenarioConfig(
                        scene,
                        simulator="metadrive",
                        randomize=True,
                        duration=5 if args.contract_smoke else 20,
                        route_length=60 if args.contract_smoke else None,
                        params={
                            "rgb_camera": True,
                            **({"empty_road": True} if args.contract_smoke else {}),
                        },
                    )
                    env = make_env(scene, spec)
                    stack = DrivingStack(
                        policies,
                        router,
                        shield=InflatedCBFShield(config=cfg.shield),
                        config=cfg.stack,
                    )
                    try:
                        result = stack.run(env, seed=seed)
                    finally:
                        env.close()
                        router.close()
                    episode_path = args.output / f"{scene}-{variant}-{seed}.json"
                    episode_path.write_text(json.dumps(result.to_dict(), allow_nan=False) + "\n")
                    summaries.append(
                        {
                            "scene": scene,
                            "seed": seed,
                            "variant": variant,
                            "route_pct": result.route_pct,
                            "crashed": result.crashed,
                            "calls": result.router_calls,
                            "live_calls": result.router_live_calls,
                            "fallbacks": result.router_fallbacks,
                            "applied_calls": result.calls_applied,
                            "success": not result.crashed and result.route_pct >= 99,
                            "camera_capture_wall_ms": [
                                c.get("camera_capture_wall_ms") for c in result.calls
                            ],
                            "wall_latency_ms": result.wall_latency_ms,
                        }
                    )
                    print(json.dumps(summaries[-1]), flush=True)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
    record = {
        "kind": "camera_http_control_contract_smoke"
        if args.contract_smoke
        else "actual_model_integration",
        "actual_model": not args.contract_smoke,
        "episodes": summaries,
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "server": args.server,
        "checkpoint_info": checkpoint_info,
        "fixture_calls": fixture_calls,
        "config": asdict(cfg),
        "limitations": [
            "No teammate fine-tuned weights bundled or independently verified",
            "New prompt excludes current-skill shortcut and requires fresh evaluation",
            "HTTP timeout is not a hard real-time scheduling guarantee",
        ],
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.contract_smoke:
        if not fixture_calls or any(r["live_calls"] or r["fallbacks"] for r in summaries):
            raise RuntimeError(
                "contract fixture must have actual HTTP calls but no real-model claims"
            )
        if not any(r["applied_calls"] for r in summaries):
            raise RuntimeError("no programmed maneuver reached the physical stack")
    elif not all(r["live_calls"] > 0 and r["fallbacks"] == 0 for r in summaries):
        raise RuntimeError(
            "one or more episodes had no valid live-model answer; inspect failed calls"
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
