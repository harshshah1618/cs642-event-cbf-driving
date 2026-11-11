"""Matched physical worlds: exact one-frame camera aliasing, without inference."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LP_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from event_cbf_driving.config import ScenarioConfig
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.stack.controller import DrivingStack
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/maneuver-alias"
TEXT = "Our car is in the RIGHT lane. Current skill: CRUISE_RIGHT. Ego speed: 20 m/s."
FIXED_CLOCK = False


def digest(data):
    return hashlib.sha256(data).hexdigest()


def capture(lead_x, hidden_x, seed, label):
    traffic = [{"x": lead_x, "y": 0., "vx": 0.}]
    if hidden_x is not None:
        traffic.append({"x": hidden_x, "y": 4., "vx": 20.})
    spec = ScenarioConfig("two_trap", simulator="metadrive", randomize=False, duration=2,
                          params={"rgb_camera": True, "initial_traffic": traffic,
                                  "spawn_x": 0., "ego_vx": 20., "initial_y": 0.})
    env = make_env("two_trap", spec)
    try:
        env.reset(seed=seed)
        snap = env.snapshot()
        def frame():
            if FIXED_CLOCK:
                from panda3d.core import ClockObject

                clock = ClockObject.getGlobalClock()
                clock.setMode(ClockObject.M_slave)
                clock.setFrameTime(123.0)
                clock.setFrameCount(100)
                clock.setDt(1 / 60)
            return env.capture_router_camera()

        first = frame()
        camera = frame()
        after = env.snapshot()
        assert asdict(snap.ego) == asdict(after.ego)
        assert [asdict(o) for o in snap.others] == [asdict(o) for o in after.others]
        rgb = np.array(Image.open(io.BytesIO(camera.image_bytes)).convert("RGB"))
        first_rgb = np.array(Image.open(io.BytesIO(first.image_bytes)).convert("RGB"))
        proxy = SimpleNamespace(skills={"OVERTAKE_LEFT": SimpleNamespace(meta=SimpleNamespace(y_target=4.))})
        admitted = DrivingStack._switch_clear(proxy, "OVERTAKE_LEFT", snap)
        path = OUT / (label + ".png")
        path.write_bytes(camera.image_bytes)
        state = {"ego": asdict(snap.ego), "others": [asdict(o) for o in snap.others]}
        return rgb, {"frame": path.name, "png_sha256": digest(camera.image_bytes),
                     "rgb_sha256": digest(rgb.tobytes()), "rgb_shape": list(rgb.shape),
                     "state": state, "state_sha256": digest(json.dumps(state, sort_keys=True).encode()),
                     "target_lane_admitted": admitted, "text": TEXT,
                     "text_sha256": digest(TEXT.encode()), "seed": seed,
                     "repeat_capture_different_pixels": int(np.any(rgb != first_rgb, axis=2).sum()),
                     "captured_sim_t": camera.captured_t}
    finally:
        env.close()


def main():
    global OUT, FIXED_CLOCK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-visual-clock", action="store_true")
    args = parser.parse_args()
    FIXED_CLOCK = args.fixed_visual_clock
    if FIXED_CLOCK:
        OUT = OUT / "fixed-visual-clock"
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = []
    for lead_x in ((40.,) if FIXED_CLOCK else (25., 40., 65.)):
        for hidden_x in ((0.,) if FIXED_CLOCK else (-2., 0.)):
            name = f"lead{int(lead_x)}-hidden{int(hidden_x)}"
            free, a = capture(lead_x, None, 7301, name + "-free")
            blocked, b = capture(lead_x, hidden_x, 7301, name + "-occupied")
            assert a["state"]["ego"] == b["state"]["ego"]
            assert a["state_sha256"] != b["state_sha256"]
            assert a["target_lane_admitted"] and not b["target_lane_admitted"]
            delta = np.abs(free.astype(np.int16) - blocked.astype(np.int16))
            pair = {"name": name, "lead_x": lead_x, "hidden_x": hidden_x,
                    "free": a, "occupied": b, "different_pixels": int(np.any(delta != 0, axis=2).sum()),
                    "different_channels": int(np.count_nonzero(delta)), "maximum_channel_change": int(delta.max()),
                    "decoded_rgb_identical": bool(np.array_equal(free, blocked)),
                    "png_identical": a["png_sha256"] == b["png_sha256"],
                    "text_identical": a["text_sha256"] == b["text_sha256"]}
            pairs.append(pair)
            (OUT / "results.json").write_text(json.dumps({"pairs": pairs}, indent=2) + "\n")
            print(name, pair["different_pixels"], pair["maximum_channel_change"], flush=True)
    result = {"kind": "physical_front_camera_observation_aliasing", "model_inference": False,
              "fixed_visual_clock": FIXED_CLOCK,
              "script_sha256": digest(Path(__file__).read_bytes()), "pairs": pairs,
              "exact_rgb_pairs": sum(p["decoded_rgb_identical"] for p in pairs),
              "exact_png_pairs": sum(p["png_identical"] for p in pairs),
              "scope": "Test for identical one-frame sensor/text inputs with different ground-truth admission. "
                       "Does not establish unique optimal actions or inevitable collision; abstention is available."}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
