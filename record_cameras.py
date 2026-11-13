"""Record a front camera and a chase camera from one MetaDrive drive per road."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import evaluation_runner as runner
from config import ScenarioConfig, load_config
from env_registry import make_env as real_make_env
from evaluation_runner import run_episode

OUT = Path("outputs/camera-views")
FRONT_MOUNT = ((0.0, 0.8, 1.5), (0.0, -5.0, 0.0))
CHASE_MOUNT = ((0.0, -8.0, 3.2), (0.0, -16.0, 0.0))


def _bgr(image: np.ndarray) -> np.ndarray:
    frame = np.asarray(image)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    # Panda's screenshot buffer is BGR.
    return np.ascontiguousarray(frame)


def _grab(adapter, writer, scenario: str) -> None:
    sim = adapter.env
    origin = sim.vehicle.origin
    front = sim.engine.get_sensor("rgb_front").perceive(False, origin, *FRONT_MOUNT)
    chase = sim.engine.get_sensor("rgb_chase").perceive(False, origin, *CHASE_MOUNT)
    panel = np.concatenate([_bgr(chase), _bgr(front)], axis=0)
    t = adapter._step_i * 0.02
    cv2.putText(
        panel,
        f"{scenario}   chase   t={t:.1f}s",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "front camera",
        (16, panel.shape[0] // 2 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    writer.write(panel)
    if not getattr(adapter, "_saved_still", False):
        OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / f"{scenario}-frame.png"), panel)
        adapter._saved_still = True


def _camera_make(name, config):
    env = real_make_env(name, config)
    original = env.step
    holder = {"writer": None, "scenario": name}

    def step(action):
        result = original(action)
        writer = holder["writer"]
        if writer is not None and env._step_i % 2 == 0:
            _grab(env, writer, holder["scenario"])
        return result

    env.step = step
    env._camera_holder = holder
    return env


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runner.make_env = _camera_make
    cfg = load_config("metadrive-study.yaml")
    cfg.stack.record_interactions = False
    skills, _info = runner.load_skill_bank(cfg.checkpoint_dir)
    router = runner.create_router(cfg.router)
    try:
        for scenario in ("two_trap", "merge", "dense_highway"):
            spec = ScenarioConfig(
                scenario,
                simulator="metadrive",
                randomize=True,
                duration=16.0,
                params={"rgb_camera": True},
            )
            path = OUT / f"{scenario}-proposed.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25.0,
                (640, 720),
            )
            if not writer.isOpened():
                raise RuntimeError(f"could not open {path}")
            # The factory closes over holder via the env created inside run_episode.
            # Stash the writer on a list the step wrapper reads after make.
            pending = {"writer": writer, "scenario": scenario}

            real_step_make = runner.make_env

            def make_with_writer(name, config, pending=pending, inner=real_step_make):
                env = inner(name, config)
                env._camera_holder["writer"] = pending["writer"]
                env._camera_holder["scenario"] = pending["scenario"]
                return env

            runner.make_env = make_with_writer
            result = run_episode(
                "proposed",
                spec,
                0,
                skills,
                router,
                stack_config=cfg.stack,
                shield_config=cfg.shield,
            )
            writer.release()
            print(
                scenario,
                "route",
                round(result.route_pct, 1),
                "crash",
                result.crashed,
                "calls",
                result.router_calls,
                path,
                flush=True,
            )
    finally:
        runner.make_env = real_make_env
        close = getattr(router, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
