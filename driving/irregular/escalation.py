"""Closed-loop escalation: triggers -> slow VLM call -> lease -> fast layer + protection.

The fast layer is Harsh's LaneCautionDriver (rule planner, optional PPO tracker,
protection). A trigger asks the VLM one question; its answer becomes a lease:

- continue: drop any lease, the rule planner drives.
- careful:  careful mode (0.7x speed, 1.5x headway) for `lease` seconds.
- stop:     hold (brake to a standstill) and ask again after `hold_recheck` s.
- left/right: one checked lane change (Harsh's gap checks), rule lane changes
            paused for `lease` seconds; a rejected change becomes careful.

The existing protection remains active: a hold only lowers throttle and a lane
change uses the normal admission checks. These are heuristic checks, not a
guarantee that an incorrect answer cannot increase risk. The answer arrives
after the measured VLM latency plus any injected delay, in simulation time.
"""

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from driving.safemetadrive.policies import LaneCautionDriver

TRIGGERS = ("none", "poll", "novelty", "cbf", "novelty+cbf", "oracle", "novelty+lidar", "lidar")


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
    # Upgrade #2: Weather-invariant local contrast trigger (suppresses uniform rain false alarms)
    weather_invariant: bool = False
    contrast_min: float = 1.50       # max_patch / median_patch required for localized hazard trigger
    median_weather_tau: float = 24.0 # median_patch above this without local spike -> 1-shot weather check
    # Upgrade #1: Grounded 3D constraint compilation (VLM hazard -> 3D stop wall / wet friction / detour)
    grounded_constraints: bool = False
    # v3: LiDAR backup trigger -- ask once about each new static object near the path ahead
    lidar_min_fwd: float = 10.0
    lidar_max_fwd: float = 40.0
    lidar_max_lateral: float = 20.0
    camera_fov_deg: float = 65.0
    # v3: verified release -- a hold is released only on a confident "proceed" to a hold question
    verified_release: bool = False
    release_threshold: float = 0.47  # P(proceed) to release; chosen on dev maps 1000-1049 (tau=0.53)
    lidar_second_look: float = 20.0  # ask again about an object once it is this close (m); 0 = off
    release_to: str = "careful"     # what a released hold becomes

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


class UnifiedEscalatingDriver:
    """Adapter connecting Escalation (Layer 3 VLM) to UnifiedDrivingSystem (Layers 1-2).

    Strictly observation-only: UnifiedDrivingSystem receives ONLY the 259-dim
    LidarStateObservation vector (zero access to simulator object lists or road graphs).
    """

    def __init__(self, observation, *, skill_checkpoint=None, initial_careful=False,
                 max_speed_kmh=30.0, **kwargs):
        from driving.safemetadrive.unified import UnifiedDrivingSystem
        tracker = "ppo" if skill_checkpoint else "analytic"
        self.system = UnifiedDrivingSystem(
            observation,
            tracker=tracker,
            skill_checkpoint=skill_checkpoint,
            max_speed_kmh=max_speed_kmh,
            **kwargs,
        )
        self.hold = False
        self.careful = bool(initial_careful)
        self.planner = "rule"
        self.lane = None

    @property
    def interventions(self):
        return self.system.interventions

    def planner_feedback(self):
        """Lane state for the VLM prompt, from the 259-dim observation (no simulator lane graph)."""
        from driving.safemetadrive.lidar_driver import decode_observation
        d = decode_observation(self.system.observation)
        centers = self.system._core._lane_centers_from_left(d)
        lane = self.system._core._current_lane_idx(d, centers)
        return dict(current_lane=lane, target_lane=lane, lane_count=d.n_lanes)

    def synchronize_lane(self):
        pass

    def on_observation(self, obs):
        self.system.on_observation(obs)

    def apply_intent(self, delta, careful, capture_lane):
        from driving.safemetadrive.lidar_driver import decode_observation
        d = decode_observation(self.system.observation)
        centers = self.system._core._lane_centers_from_left(d)
        cur = self.system._core.target_lane_idx
        if cur is None:
            cur = self.system._core._current_lane_idx(d, centers)
        target = cur + int(delta)
        if not (0 <= target < d.n_lanes):
            return False
        if delta:
            _, _, s_lane, d_right = self.system._core._lidar_to_road_points(d)
            gaps = self.system._core._lane_gaps(d, centers, s_lane, d_right)
            if not self.system._core._can_admit_lane(target, gaps, d.speed_mps):
                return False
            self.system._core.target_lane_idx = target
            self.system._core.changes += 1
        self.careful = bool(careful)
        return True

    def act(self):
        self.system._core.allow_lane_change = (self.planner == "rule")
        self.system.set_semantic_lease(stop=self.hold, careful=self.careful)
        return self.system.act()

    def close(self):
        pass


class Escalation:
    def __init__(self, cfg: EscalationConfig, vlm=None, detector=None, hi_capture=None):
        if cfg.trigger != "none" and vlm is None:
            raise ValueError("A trigger needs a VLM")
        if "novelty" in cfg.trigger and detector is None:
            raise ValueError("The novelty trigger needs a detector")
        self.cfg, self.vlm, self.detector = cfg, vlm, detector
        self.hi_capture = hi_capture  # renders the high-resolution front view on demand
        self.above: deque[bool] = deque(maxlen=cfg.n)
        self.pending: dict[str, Any] | None = None
        self.last_call = -math.inf
        self.last_poll = -math.inf
        self.lease_until = -math.inf
        self.lease_action: str | None = None
        self.recheck_due = False
        self.override_steps = 0
        self.stall_steps = 0
        self.last_interventions = 0
        self.last_map = None
        self.last_cx_norm = 0.5
        self.weather_checked = False
        self.confirming = False
        self.asked_tracks: set = set()  # track ids (first look) and (id, 2) (second look)
        self.lidar_target: tuple[float, float] | None = None
        self.lidar_candidate: int | None = None
        self.crop_track: int | None = None   # LiDAR track the current hold is about
        self.calls: list[dict[str, Any]] = []
        self.debug_dir: str | None = None

    # --- answers -------------------------------------------------------------
    def _clear(self, driver):
        driver.hold = False
        driver.careful = False
        driver.planner = "rule"
        if hasattr(driver, "system") and hasattr(driver.system, "set_grounded_constraints"):
            driver.system.set_grounded_constraints(clear_stop_wall=True, force_detour=False)
        self.lease_action = None
        self.lease_until = -math.inf

    def _apply(self, now, driver, action):
        cfg = self.cfg
        # Upgrade #1: Grounded 3D semantic constraint compilation into Layer 1
        if cfg.grounded_constraints and hasattr(driver, "system"):

            if action == "benign_clear" or action == "continue":
                self._clear(driver)
                return "continue"
            if action == "wet_weather_only":
                self._clear(driver)
                driver.system.set_grounded_constraints(friction_scale=0.70)
                self.weather_checked = True
                self.lease_action, self.lease_until = "wet_weather_only", now + 60.0
                return "wet_weather_only"
            if action in ("stop_flagger", "pedestrian_lane", "stop"):
                # Immediately hold full brake so a thin shoulder flagger is never passed
                # while waiting on an uncertain LiDAR return; release only on VLM re-check.
                driver.hold = True
                driver.careful = True
                driver.system.set_grounded_constraints(clear_stop_wall=True, force_detour=False)
                self.lease_action, self.lease_until = "stop", now + cfg.hold_recheck
                return action
            if action == "road_closure":
                # Keep lane-change replanning active to detour around the closure at safe wet friction!
                driver.hold = False
                driver.careful = False
                driver.planner = "rule"
                driver.system.set_grounded_constraints(
                    clear_stop_wall=True, friction_scale=0.40, force_detour=True
                )
                self.lease_action, self.lease_until = "road_closure", now + cfg.lease
                return "road_closure"
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
            if cfg.weather_invariant and patch_map is not None:
                from driving.irregular.patchcore import local_contrast_metrics
                m = local_contrast_metrics(patch_map)
                self.last_cx_norm = m["cx_norm"]
                is_local = (m["max_score"] > cfg.tau) and (m["contrast_ratio"] >= cfg.contrast_min)
                is_global_weather = (
                    (not is_local)
                    and (not self.weather_checked)
                    and (m["median_score"] >= cfg.median_weather_tau)
                )
                self.above.append(is_local)
                if sum(self.above) >= cfg.k:
                    fired.append("novelty-local")
                elif is_global_weather:
                    self.weather_checked = True
                    fired.append("weather-global")
            else:
                if patch_map is not None:
                    from driving.irregular.patchcore import local_contrast_metrics
                    self.last_cx_norm = local_contrast_metrics(patch_map)["cx_norm"]
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
        if "lidar" in cfg.trigger:
            guard = getattr(getattr(getattr(driver, "system", None), "_core", None), "crossing", None)
            if guard is not None:
                for t in guard.static_objects():
                    x, y = float(t["p"][0]), float(t["p"][1])
                    if not (cfg.lidar_min_fwd <= x <= cfg.lidar_max_fwd) or abs(y) > cfg.lidar_max_lateral:
                        continue
                    # First look on detection; one second look once the object is close
                    # (more pixels), unless we are already holding for it.
                    second = (cfg.lidar_second_look > 0 and x <= cfg.lidar_second_look
                              and (t["id"], 2) not in self.asked_tracks and self.lease_action != "stop")
                    if t["id"] in self.asked_tracks and not second:
                        continue
                    key = (t["id"], 2) if t["id"] in self.asked_tracks else t["id"]
                    # PatchCore already raised this object: that call covers this look
                    if "novelty" in fired or "novelty-local" in fired:
                        self.asked_tracks.add(key)
                        continue
                    self.lidar_candidate = key
                    self.lidar_target = (x, y)
                    fired.append("lidar-object")
                    break
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
        second_look = "lidar-object" in fired and isinstance(self.lidar_candidate, tuple)
        if now - self.last_call < cfg.cooldown and "hold-recheck" not in fired and not second_look:
            return
        # 4. Ask the VLM (synchronous in wall time, delayed in simulation time).
        if image is None and state is None:
            image, state = capture()
        guard = getattr(getattr(getattr(driver, "system", None), "_core", None), "crossing", None)
        if "lidar-object" in fired:
            self.asked_tracks.add(self.lidar_candidate)
            key = self.lidar_candidate
            self.crop_track = key[0] if isinstance(key, tuple) else key
        elif "hold-recheck" not in fired:
            self.crop_track = None
        if self.crop_track is not None and guard is not None:
            live = [t for t in guard.tracks if t["id"] == self.crop_track]
            self.lidar_target = (float(live[0]["p"][0]), float(live[0]["p"][1])) if live else None
        crop = None
        if cfg.use_crop and image is not None and self.crop_track is not None and self.lidar_target:
            from driving.irregular.vlm import bearing_crop
            hi = self.hi_capture() if self.hi_capture is not None else None
            bearing = math.degrees(math.atan2(self.lidar_target[1], self.lidar_target[0]))
            crop = bearing_crop(image, bearing, cfg.camera_fov_deg, hi=hi)
        elif cfg.use_crop and image is not None and self.last_map is not None:
            from driving.irregular.vlm import anomaly_crop
            hi = self.hi_capture() if self.hi_capture is not None else None
            crop = anomaly_crop(image, self.last_map, hi=hi)
        if getattr(self, "debug_dir", None) and crop is not None:
            crop.save(f"{self.debug_dir}/crop-t{now:05.1f}.jpg", quality=92)
        keep_p = None
        if cfg.verified_release and self.lease_action == "stop" and hasattr(self.vlm, "verify_hold"):
            keep_p, seconds = self.vlm.verify_hold(image, state, crop, truth)
            action = "stop" if (1.0 - keep_p) < cfg.release_threshold else cfg.release_to
        else:
            action, seconds = self.vlm(image, state, crop, truth)
        if self.lease_action == "stop":
            self.lease_until = math.inf  # keep holding until the re-check answer arrives
        self.recheck_due = False
        self.last_call = now
        self.pending = dict(t=now, triggers=fired, action=action, seconds=seconds,
                            ready_at=now + seconds + cfg.extra_delay,
                            keep_p=keep_p,
                            forward=None if not truth else truth["forward"],
                            face=None if not truth else truth.get("face"))
        self.calls.append(self.pending)
        if cfg.pending == "careful" and not driver.hold:
            driver.careful = True


def run_episode(env, seed, placements, cfg, vlm=None, detector=None, render=False,
                driver_kwargs=None, max_steps=700, after_pass=20.0, hi_res=False, debug_dir=None,
                driver_kind="legacy"):
    """One closed-loop episode; returns a result row."""
    from driving.irregular.scene import place, placement_record, relative

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    reset_out = env.reset(seed=seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    objs = [place(env, p) for p in placements]
    primary = next((o for o in objs if o is not None), None)
    lidar_obs = None
    if driver_kind == "unified":
        if isinstance(obs, dict) or np.asarray(obs).shape != (259,):
            from metadrive.obs.state_obs import LidarStateObservation
            lidar_obs = LidarStateObservation(env.config)
            obs_259 = lidar_obs.observe(env.agent)
        else:
            obs_259 = obs
        driver = UnifiedEscalatingDriver(obs_259, **(driver_kwargs or {}))
    else:
        driver = EscalatingDriver(env.agent, speed=30, **(driver_kwargs or {}))
    esc = Escalation(cfg, vlm, detector,
                     hi_capture=(lambda: weather(capture_front_hi(env), salt=env.engine.episode_step))
                     if render and hi_res else None)
    esc.debug_dir = debug_dir

    overlay = next((p.overlay for p in placements if p.overlay), None)

    def weather(image, salt=0):
        if overlay is None:
            return image
        from driving.irregular.weather import OVERLAYS
        return OVERLAYS[overlay](image, seed=seed * 100000 + salt)

    def capture():
        if render:
            from driving.irregular.patchcore import front_view
            from driving.safemetadrive.camera_planner import capture as cam
            image, state = cam(env, driver)
            return weather(front_view(image), salt=env.engine.episode_step), state
        lanes = env.agent.navigation.current_ref_lanes
        return None, dict(speed_mps=float(env.agent.speed), lane_count=len(lanes),
                          current_lane=int(env.agent.lane.index[-1]))

    cost, info = 0.0, {}
    crash_keys = ("crash_vehicle", "crash_object", "crash_human")
    crash_steps = dict.fromkeys(crash_keys, 0)
    collision_events = []
    end_reason = "step_budget"
    stopped_near = 0
    passed_while_stop = False
    min_gap = math.inf
    prev = None
    pass_speed = None
    step = 0
    try:
        for step in range(max_steps):
            now = step * 0.1
            driver.synchronize_lane()
            truth = relative(env.agent, primary) if primary is not None else None
            esc.step(now, driver, truth, capture, step % cfg.capture_every == 0)
            obs, _, terminated, truncated, info = env.step(driver.act())
            if driver_kind == "unified":
                obs_259 = lidar_obs.observe(env.agent) if lidar_obs is not None else obs
                driver.on_observation(obs_259)
            cost += float(info["cost"])
            # MetaDrive resets crash flags before every step; final-step flags alone
            # miss collisions when the episode continues after contact.
            hits = [key for key in crash_keys if info.get(key, False)]
            for key in hits:
                crash_steps[key] += 1
            if hits:
                collision_events.append(dict(step=step + 1, seconds=(step + 1) * 0.1, kinds=hits))
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
                    end_reason = "passed_object"
                    break
            if terminated or truncated:
                end_reason = "terminated" if terminated else "truncated"
                break
    finally:
        driver.close()
    return dict(
        metrics_schema="irregular-episode-v2", end_reason=end_reason,
        seed=seed, placements=[placement_record(p) for p in placements], config=asdict(cfg),
        steps=step + 1, cost=cost, **{key: count > 0 for key, count in crash_steps.items()},
        crash_steps=crash_steps, collision_events=collision_events,
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

    def verify_hold(self, image, state, crop, truth):
        face = truth.get("face") if truth else None
        keep = (self.scene == "flagger" and face == "stop") or self.scene in ("pedestrian-lane",
                                                                             "cyclist-lane")
        return (1.0 if keep else 0.0), self.latency


class FixedVLM:
    """Non-semantic baseline: every trigger gets the same answer (e.g. careful or stop)."""

    def __init__(self, action, latency=0.2):
        self.action, self.latency = action, latency

    def __call__(self, image, state, crop, truth):
        return self.action, self.latency

    def verify_hold(self, image, state, crop, truth):
        return (1.0 if self.action == "stop" else 0.0), self.latency


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
        total: dict[str, float] = {}
        seconds = 0.0
        for k in range(self.orders):
            order = option_order(state, None if k == 0 else k)
            _, probs, t = self.vlm.answer(image, state, order, crop=crop, speed=self.speed)
            seconds += t
            for a, p in probs.items():
                total[a] = total.get(a, 0.0) + p
        return max(total, key=lambda action: total[action]), seconds

    def verify_hold(self, image, state, crop, truth):
        """P(keep waiting), averaged over both option orders."""
        import time
        started = time.monotonic()
        keep = 0.0
        for order in (("keep", "proceed"), ("proceed", "keep")):
            keep += self.vlm.hold_answer(image, order, crop=crop)["keep"] / 2.0
        return keep, time.monotonic() - started


class GroundedVLMCaller:
    """Upgrade #1: Order-debiased 5-class semantic hazard classifier for 3D constraint compilation."""

    def __init__(self, model_id, dtype="bfloat16", orders=3):
        from driving.irregular.vlm import GroundedConstraintVLM
        self.vlm = GroundedConstraintVLM(model_id, dtype=dtype)
        self.orders = orders

    def __call__(self, image, state, crop, truth):
        import random

        from driving.irregular.vlm import HAZARD_CLASSES

        total: dict[str, float] = {}
        seconds = 0.0
        for k in range(self.orders):
            order = list(HAZARD_CLASSES)
            if k > 0:
                random.Random(k).shuffle(order)
            _, probs, t = self.vlm.classify_hazard(image, state, order, crop=crop)
            seconds += t
            for h, p in probs.items():
                total[h] = total.get(h, 0.0) + p
        return max(total, key=lambda h: total[h]), seconds


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
    if scene == "pedestrian-lane":
        return [Placement("pedestrian", 60, "lane")]
    if scene == "wet-closed":
        # Road closed 60 m ahead, braking ~2 m/s^2 (below protection's 4 m/s^2), rain on camera.
        return [Placement("closure", 60, "lane", brake_scale=0.10, overlay="rain")]
    if scene == "rain-clean":
        return [Placement("none", 60, overlay="rain")]
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


IRREGULAR_TEST = (1000, 50)  # first test maps; used for development since 2026-11-25
IRREGULAR_FINAL = (3000, 100)  # untouched maps for the frozen v3 final test


def irregular_config(split):
    """Environment config: Harsh's partitions plus the fresh irregular test maps."""
    from driving.safemetadrive.protocol import environment_config
    if split == "irregular-test":
        config = environment_config("validation")
        config.update(start_seed=IRREGULAR_TEST[0], num_scenarios=IRREGULAR_TEST[1])
        return config
    if split == "irregular-final":
        config = environment_config("validation")
        config.update(start_seed=IRREGULAR_FINAL[0], num_scenarios=IRREGULAR_FINAL[1])
        return config
    return environment_config(split)
