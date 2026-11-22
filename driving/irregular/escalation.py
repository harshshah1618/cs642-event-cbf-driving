"""Closed-loop escalation: triggers -> slow VLM call -> lease -> fast layer + protection.

The fast layer is Harsh's LaneCautionDriver (rule planner, optional PPO tracker,
protection). A trigger asks the VLM one question; its answer becomes a lease:

- continue: drop any lease, the rule planner drives.
- careful:  careful mode (0.7x speed, 1.5x headway) for `lease` seconds.
- stop:     hold (brake to a standstill) and ask again after `hold_recheck` s.
- left/right: one checked lane change (Harsh's gap checks), rule lane changes
            paused for `lease` seconds; a rejected change becomes careful.

Every answer still passes through protection: a hold only lowers throttle and a
lane change uses the normal admission checks, so a wrong answer cannot make the
car less safe than the fast layer alone. The answer arrives after the measured
VLM latency plus any injected delay, in simulation time.
"""

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from driving.safemetadrive.policies import LaneCautionDriver

TRIGGERS = ("none", "poll", "novelty", "cbf", "novelty+cbf", "oracle")


@dataclass
class EscalationConfig:
    trigger: str = "none"
    capture_every: int = 3        # control steps between camera frames (0.3 s at 10 Hz)
    k: int = 3                    # novelty: k of the last n frames above tau
    n: int = 5
    tau: float = 37.4             # DINOv2 PatchCore threshold from the development frames
    poll_period: float = 1.5      # seconds, as in Harsh's camera planner
    cbf_seconds: float = 1.0      # sustained protection override
    stall_seconds: float = 2.0    # stopped with route left (not on hold)
    oracle_range: float = 40.0    # oracle trigger: object visible within this distance
    cooldown: float = 3.0         # minimum seconds between calls
    lease: float = 10.0           # seconds a careful or lane-change answer holds
    hold_recheck: float = 3.0     # a stop answer is re-asked after this long
    extra_delay: float = 0.0      # injected latency on top of the measured one
    pending: str = "nominal"      # behaviour while a call is pending: nominal | careful
    use_crop: bool = True
    release_unflagged: bool = False  # release a hold when PatchCore stops flagging (pilot v2-v4)

    def __post_init__(self):
        if self.trigger not in TRIGGERS:
            raise ValueError(f"Unknown trigger {self.trigger!r}")
        if self.pending not in ("nominal", "careful"):
            raise ValueError(f"Unknown pending mode {self.pending!r}")


class EscalatingDriver(LaneCautionDriver):
    """LaneCautionDriver plus a hold that can only lower throttle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold = False

    def act(self):
        steer, throttle = super().act()
        if self.hold:
            throttle = min(throttle, float(np.clip(0.5 * (0.0 - self.vehicle.speed), -1, 1)))
        return [steer, throttle]


class Escalation:
    def __init__(self, cfg: EscalationConfig, vlm=None, detector=None, hi_capture=None):
        if cfg.trigger != "none" and vlm is None:
            raise ValueError("A trigger needs a VLM")
        if "novelty" in cfg.trigger and detector is None:
            raise ValueError("The novelty trigger needs a detector")
        self.cfg, self.vlm, self.detector = cfg, vlm, detector
        self.hi_capture = hi_capture  # renders the high-resolution front view on demand
        self.above = deque(maxlen=cfg.n)
        self.pending = None
        self.last_call = -math.inf
        self.last_poll = -math.inf
        self.lease_until = -math.inf
        self.lease_action = None
        self.recheck_due = False
        self.override_steps = 0
        self.stall_steps = 0
        self.last_interventions = 0
        self.last_map = None
        self.confirming = False
        self.calls = []

    # --- answers -------------------------------------------------------------
    def _clear(self, driver):
        driver.hold = False
        driver.careful = False
        driver.planner = "rule"
        self.lease_action = None
        self.lease_until = -math.inf

    def _apply(self, now, driver, action):
        cfg = self.cfg
        self._clear(driver)
        if action == "continue":
            return "continue"
        if action == "stop":
            driver.hold = True
            self.lease_action, self.lease_until = "stop", now + cfg.hold_recheck
            return "stop"
        if action in ("left", "right"):
            driver.planner = "external"
            delta = -1 if action == "left" else 1
            if driver.apply_intent(delta, False, driver.lane):
                self.lease_action, self.lease_until = action, now + cfg.lease
                return action
            action = "careful"  # rejected by the gap checks
        driver.careful = True
        self.lease_action, self.lease_until = "careful", now + cfg.lease
        return action

    # --- triggers ------------------------------------------------------------
    def _triggers(self, now, driver, truth, frame_step, image):
        cfg, fired = self.cfg, []
        dt = 0.1
        detector_ready = image is not None or not getattr(self.detector, "needs_image", True)
        if "novelty" in cfg.trigger and frame_step and detector_ready:
            score, patch_map = self.detector(image, truth)
            self.last_map = patch_map
            self.above.append(score > cfg.tau)
            if sum(self.above) >= cfg.k:
                fired.append("novelty")
        if "cbf" in cfg.trigger:
            overridden = driver.interventions > self.last_interventions
            self.override_steps = self.override_steps + 1 if overridden else 0
            stalled = driver.vehicle.speed < 0.5 and not driver.hold
            self.stall_steps = self.stall_steps + 1 if stalled else 0
            if self.override_steps * dt >= cfg.cbf_seconds:
                fired.append("cbf-override")
            if self.stall_steps * dt >= cfg.stall_seconds:
                fired.append("cbf-stall")
        self.last_interventions = driver.interventions
        if cfg.trigger == "poll" and frame_step and now - self.last_poll >= cfg.poll_period:
            self.last_poll = now
            fired.append("poll")
        if cfg.trigger == "oracle" and truth and truth["visible"] and \
                0 < truth["forward"] <= cfg.oracle_range:
            fired.append("oracle")
        return fired

    # --- main step -----------------------------------------------------------
    def step(self, now, driver, truth, capture, frame_step):
        """Call once per control step before driver.act().

        `capture()` returns (image or None, VLM state). `frame_step` is True on
        steps where a camera frame is taken for the novelty detector.
        """
        cfg = self.cfg
        if cfg.trigger == "none":
            return
        # 1. A pending answer becomes ready.
        if self.pending is not None and now >= self.pending["ready_at"]:
            call = self.pending
            self.pending = None
            self.recheck_due = False
            if self.lease_action == "stop" and call["action"] != "stop" and not self.confirming:
                # Stop is sticky: release only if an immediate second answer agrees.
                self.confirming, self.recheck_due = True, True
                call["applied"] = "confirming"
            else:
                self.confirming = False
                call["applied"] = self._apply(now, driver, call["action"])
            call["applied_t"] = now
        # 2. Lease expiry. An expired hold is re-asked (once) rather than released.
        if self.lease_action is not None and now >= self.lease_until and self.pending is None:
            still_flagged = (not cfg.release_unflagged or "novelty" not in cfg.trigger
                             or sum(self.above) >= cfg.k)
            if self.lease_action == "stop" and still_flagged:
                self.recheck_due = True
            else:
                self._clear(driver)  # nothing flagged any more: release the hold
        # 3. Triggers.
        image, state = capture() if frame_step or self.recheck_due else (None, None)
        fired = self._triggers(now, driver, truth, frame_step, image)
        if self.recheck_due:
            fired.append("hold-recheck")
        if not fired or self.pending is not None:
            return
        if now - self.last_call < cfg.cooldown and "hold-recheck" not in fired:
            return
        # 4. Ask the VLM (synchronous in wall time, delayed in simulation time).
        if image is None and state is None:
            image, state = capture()
        crop = None
        if cfg.use_crop and image is not None and self.last_map is not None:
            from driving.irregular.vlm import anomaly_crop
            hi = self.hi_capture() if self.hi_capture is not None else None
            crop = anomaly_crop(image, self.last_map, hi=hi)
        if getattr(self, "debug_dir", None) and crop is not None:
            crop.save(f"{self.debug_dir}/crop-t{now:05.1f}.jpg", quality=92)
        action, seconds = self.vlm(image, state, crop, truth)
        if self.lease_action == "stop":
            self.lease_until = math.inf  # keep holding until the re-check answer arrives
        self.recheck_due = False
        self.last_call = now
        self.pending = dict(t=now, triggers=fired, action=action, seconds=seconds,
                            ready_at=now + seconds + cfg.extra_delay,
                            forward=None if not truth else truth["forward"],
                            face=None if not truth else truth.get("face"))
        self.calls.append(self.pending)
        if cfg.pending == "careful" and not driver.hold:
            driver.careful = True


def run_episode(env, seed, placements, cfg, vlm=None, detector=None, render=False,
                driver_kwargs=None, max_steps=700, after_pass=20.0, hi_res=False, debug_dir=None):
    """One closed-loop episode; returns a result row."""
    from driving.irregular.scene import place, placement_record, relative

    env.reset(seed=seed)
    objs = [place(env, p) for p in placements]
    primary = next((o for o in objs if o is not None), None)
    driver = EscalatingDriver(env.agent, speed=30, **(driver_kwargs or {}))
    esc = Escalation(cfg, vlm, detector,
                     hi_capture=(lambda: capture_front_hi(env)) if render and hi_res else None)
    esc.debug_dir = debug_dir

    def capture():
        if render:
            from driving.irregular.patchcore import front_view
            from driving.safemetadrive.camera_planner import capture as cam
            image, state = cam(env, driver)
            return front_view(image), state
        lanes = env.agent.navigation.current_ref_lanes
        return None, dict(speed_mps=float(env.agent.speed), lane_count=len(lanes),
                          current_lane=int(env.agent.lane.index[-1]))

    cost, info = 0.0, {}
    stopped_near = 0
    passed_while_stop = False
    min_gap = math.inf
    prev = None
    pass_speed = None
    step = 0
    for step in range(max_steps):
        now = step * 0.1
        driver.synchronize_lane()
        truth = relative(env.agent, primary) if primary is not None else None
        esc.step(now, driver, truth, capture, step % cfg.capture_every == 0)
        _, _, terminated, truncated, info = env.step(driver.act())
        cost += float(info["cost"])
        if truth is not None:
            after = relative(env.agent, primary)
            if abs(after["lateral"]) < 6:
                min_gap = min(min_gap, after["forward"]) if after["forward"] > -2 else min_gap
            if prev is not None and prev["forward"] > 0 >= after["forward"]:
                pass_speed = float(env.agent.speed)
                passed_while_stop = passed_while_stop or after["face"] == "stop"
            if 0 < after["forward"] < 40 and env.agent.speed < 0.5:
                stopped_near += 1
            prev = after
            if after["forward"] < -after_pass:
                break
        if terminated or truncated:
            break
    return dict(
        seed=seed, placements=[placement_record(p) for p in placements], config=asdict(cfg),
        steps=step + 1, cost=cost, crash_vehicle=bool(info.get("crash_vehicle")),
        crash_object=bool(info.get("crash_object")), crash_human=bool(info.get("crash_human")),
        arrive=bool(info.get("arrive_dest")), passed_stop=passed_while_stop,
        route_completion=float(info.get("route_completion", 0.0)), pass_speed=pass_speed,
        seconds_stopped_near_object=stopped_near * 0.1,
        min_gap=None if math.isinf(min_gap) else min_gap,
        calls=[{k: v for k, v in c.items() if k != "ready_at"} for c in esc.calls],
        reached_object=prev is not None and prev["forward"] < 0)


# --- CPU stand-ins -----------------------------------------------------------

class TruthDetector:
    """Stand-in for PatchCore on CPU: fires when the object is visible within `reach` m."""
    needs_image = False

    def __init__(self, reach=25.0):
        self.reach = reach

    def __call__(self, image, truth):
        hit = bool(truth and truth["visible"] and 0 < truth["forward"] <= self.reach)
        return (100.0 if hit else 0.0), None


class OracleVLM:
    """Stand-in VLM that always gives the scene's intended answer."""

    def __init__(self, scene, latency=0.2):
        self.scene, self.latency = scene, latency

    def __call__(self, image, state, crop, truth):
        face = truth.get("face") if truth else None
        kind = self.scene
        if kind == "flagger":
            action = "stop" if face == "stop" else "careful"
        elif kind in ("billboard", "clean"):
            action = "continue"
        elif kind in ("pedestrian-lane", "cyclist-lane"):
            action = "stop"
        else:
            action = "careful"
        return action, self.latency


# --- GPU components ----------------------------------------------------------

class PatchCoreDetector:
    """Live PatchCore score for one front-camera frame."""
    needs_image = True

    def __init__(self, bank_path, encoder="dinov2_small"):
        from driving.irregular.patchcore import Encoder, PatchCore
        self.encoder = Encoder(encoder)
        self.model = PatchCore.load(bank_path, device="cuda")

    def __call__(self, image, truth):
        scores, maps = self.model.score(self.encoder([image]))
        return float(scores[0]), maps[0].float().cpu().numpy()


class VLMCaller:
    """Zero-shot VLM answering with the canonical option order."""

    def __init__(self, model_id, dtype="bfloat16", speed=False, orders=3):
        from driving.irregular.vlm import ZeroShotVLM
        self.vlm = ZeroShotVLM(model_id, dtype=dtype)
        self.orders = orders  # average answer probabilities over this many option orders
        # Stating the current speed makes a stopped car keep answering "stop" (at 0 km/h
        # Qwen3-VL-8B says stop on 90% of clean frames), so the closed loop omits it.
        self.speed = speed

    def __call__(self, image, state, crop, truth):
        from driving.irregular.vlm import option_order
        total, seconds = {}, 0.0
        for k in range(self.orders):
            order = option_order(state, None if k == 0 else k)
            _, probs, t = self.vlm.answer(image, state, order, crop=crop, speed=self.speed)
            seconds += t
            for a, p in probs.items():
                total[a] = total.get(a, 0.0) + p
        return max(total, key=total.get), seconds


def scene_placements(scene, seed):
    """Pilot scenes: one object 60 m ahead (the flagger shows STOP for 15-30 s)."""
    import random

    from driving.irregular.scene import Placement
    if scene == "flagger":
        return [Placement("flagger", 60, "shoulder", stop_seconds=15 + 15 * random.Random(seed).random())]
    if scene == "billboard":
        return [Placement("sign", 60, "shoulder", face="billboard")]
    if scene == "lane-closed":
        return [Placement("sign", 60, "shoulder", face="lane_closed")]
    if scene == "pedestrian-shoulder":
        return [Placement("pedestrian", 60, "shoulder")]
    if scene == "clean":
        return []
    raise ValueError(f"Unknown scene {scene!r}")


HI_RES = (1344, 756)  # 3x the 448 x 252 front camera


def hi_res_camera_config(config):
    """Harsh's camera setup plus a high-resolution copy of the front camera."""
    from metadrive.component.sensors.rgb_camera import RGBCamera

    from driving.safemetadrive.camera_planner import camera_config
    config = camera_config(config)
    config["sensors"] = dict(config["sensors"], rgb_front_hi=(RGBCamera, *HI_RES))
    return config


def capture_front_hi(env):
    """Render the front view at HI_RES with the same pose as Harsh's front camera."""
    from PIL import Image
    front = env.engine.get_sensor("rgb_front_hi").perceive(False, env.agent.origin,
                                                           (0, 0.8, 1.5), (0, -5, 0))
    return Image.fromarray(front[..., :3][..., ::-1].astype(np.uint8), "RGB")
