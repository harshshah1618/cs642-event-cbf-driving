"""3-tier driving stack: event router + PPO skills + inflated CBF.

Production changes over the research prototype:

* an explicit :class:`PendingCall` state machine (deadline, expiry, delivery,
  same-skill answers still count as in-flight, no lost interrupt edges);
* router responses are validated before they can touch the skill bank;
* the delay margin is active for the whole call, not only when the answer
  changes skill;
* a final :class:`SafetyMonitor` gate after the CBF, with optional abort;
* every metric is accumulated by one shared :class:`RolloutTracker`.
"""

from __future__ import annotations

import copy
import math
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Union

import numpy as np

from blend import blend_actions, sigmoid_blend_alpha
from config import StackConfig, StackMode
from env_types import WorldSnapshot
from errors import RouterError, SafetyError
from logging_utils import get_logger
from pending import FuturePendingCall, PendingCall
from policy import SkillPolicy
from result import RolloutResult, RolloutTracker
from router_base import RouterBackend, RouterRequest, response_record_fields, validate_response
from safety import InflatedCBFShield
from safety_monitor import SafetyMonitor, SafetyStatus
from scheduling import lane_opportunity, opportunity_due

logger = get_logger(__name__)

_VARIANT_MODES = {"proposed_no_margin", "proposed_no_blend"}


class DrivingStack:
    """Closed-loop controller shared by the proposed stack and all baselines."""

    def __init__(
        self,
        skills: Dict[str, SkillPolicy],
        router: RouterBackend,
        shield: Optional[InflatedCBFShield] = None,
        config: Optional[StackConfig] = None,
    ):
        self.skills = dict(skills)
        self.alias_used = False
        if "RETURN_RIGHT" not in self.skills and "CRUISE_RIGHT" in self.skills:
            self.skills["RETURN_RIGHT"] = self.skills["CRUISE_RIGHT"]
            self.alias_used = True
        missing = [name for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT") if name not in self.skills]
        if missing:
            raise SafetyError(f"missing required skill policies: {missing}")
        self.router = router
        self.shield = shield or InflatedCBFShield()
        self.config = config or StackConfig()
        self.config.validate("stack")
        self.last_shield: Optional[InflatedCBFShield] = None
        self.last_monitor: Optional[SafetyMonitor] = None
        self.last_router_health: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, env, seed: Optional[int] = None) -> RolloutResult:
        if self.config.router_async:
            # One bounded worker; cleanup joins the backend's bounded request
            # before a subsequent episode can reset or close that backend.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="skill-router") as executor:
                return self._run(env, seed, executor=executor)
        return self._run(env, seed)

    def _run(self, env, seed=None, *, executor=None) -> RolloutResult:
        cfg = copy.deepcopy(self.config)
        mode_name = str(cfg.mode)
        base_mode = self._resolve_mode(mode_name)
        if base_mode == StackMode.PROPOSED and cfg.trigger_policy in ("legacy", "uncertainty"):
            if any(
                not getattr(policy, "supports_uncertainty", True) for policy in self.skills.values()
            ):
                raise SafetyError(
                    "This trigger needs dual-critic disagreement. For single-critic PPO, "
                    "choose a traffic-based trigger such as 'deadline', 'benefit', or 'barrier'."
                )
        matched_polling = mode_name in ("fixed_rate_shielded", "always_on_shielded")
        shield = copy.deepcopy(self.shield)
        self._configure_mode(shield, base_mode, cfg, variant=mode_name)
        monitor = SafetyMonitor(
            config=None,
            ax_min=float(shield.ax_min),
            ax_max=float(shield.ax_max),
            emergency_decel=float(shield.emergency_decel),
        )

        reset = getattr(self.router, "reset", None)
        if callable(reset):
            reset()
        run_token = secrets.token_hex(3)

        _obs, reset_info = env.reset(seed=seed)
        snap: WorldSnapshot = env.snapshot()
        scenario = str(snap.info.get("scenario") or getattr(env, "scenario", ""))
        tracker = RolloutTracker(mode_name, cfg, scenario=scenario, seed=seed)
        if isinstance(reset_info, dict) and reset_info.get("params"):
            tracker.params = dict(reset_info["params"])

        skill = cfg.initial_skill
        if skill not in self.skills:
            raise SafetyError(f"initial skill {skill!r} is not loaded")
        pending: Optional[Union[PendingCall, FuturePendingCall]] = None
        worker_future = None
        simulated_busy_until = -1.0
        background_calls = []
        blend_from: Optional[str] = None
        blend_tick = 0
        blending = False
        override_streak = 0
        override_off = 10_000
        shield_armed = True
        shield_episode_called = False
        since_call = 10_000
        sigma_latch: Dict[str, bool] = {name: False for name in self.skills}
        latch_reason: Optional[str] = None
        rate = cfg.always_on_hz if base_mode == StackMode.ALWAYS_ON else cfg.fixed_rate_hz
        last_rate_t = cfg.polling_phase_s - 1.0 / rate
        crashed = False
        crash_time: Optional[float] = None
        end_reason = "timeout"
        call_index = 0
        wall_start = time.perf_counter()
        sim_start = float(snap.t)

        while True:
            snap = env.snapshot()
            t = float(snap.t)
            if cfg.realtime:
                time.sleep(max(0.0, wall_start + t - sim_start - time.perf_counter()))

            # ---- Deliver an in-flight answer exactly once. -----------------
            pending_this_tick = False
            if isinstance(pending, FuturePendingCall) and pending.response is None:
                if pending.future.done():
                    try:
                        response = pending.future.result()
                        validate_response(response, pending.request)
                        # The worker measures actual inference/retry time. Keep
                        # the configured delay as a lower bound for ablations.
                        response.ready_at = (
                            pending.submitted_t
                            + pending.request.camera_capture_wall_s
                            + max(float(response.latency_s), float(response.wall_latency_s))
                        )
                        response.latency_s = response.ready_at - pending.submitted_t
                        pending.response = response
                        pending.record["to"] = response.skill
                        pending.record.update(response_record_fields(response))
                        tracker.record_router_response(pending.record)
                        if response.ready_at > pending.deadline_t:
                            raise RouterError("measured router latency exceeded pending deadline")
                    except Exception as exc:
                        pending.record.update(rejected=True, applied=False, error=str(exc))
                        tracker.calls_rejected += 1
                        latch_reason = pending.request.reason
                        pending = None
            if pending is not None:
                if pending.is_expired(t):
                    if cfg.simulate_deadline_wait:
                        simulated_busy_until = max(simulated_busy_until, pending.ready_at)
                    pending.expired = True
                    pending.record["rejected"] = True
                    pending.record["error"] = "pending call expired before its answer"
                    pending.record["applied"] = False
                    tracker.calls_rejected += 1
                    latch_reason = pending.request.reason
                    pending = None
                elif pending.is_ready(t):
                    if (
                        cfg.skill_switch_guard
                        and pending.response is not None
                        and pending.response.skill != skill
                        and not self._switch_clear(pending.response.skill, snap)
                    ):
                        # Re-check fresh geometry at delivery, never just the
                        # older snapshot seen by the slow decision layer.
                        pending.record.update(
                            rejected=True,
                            applied=False,
                            error="skill switch blocked by fresh target-lane gap",
                        )
                        tracker.calls_rejected += 1
                        latch_reason = pending.request.reason
                        pending = None
                    else:
                        new_skill = pending.deliver(skill)
                        pending.record["applied"] = True
                        pending.record["switched"] = bool(new_skill)
                        tracker.calls_applied += 1
                        if new_skill is not None:
                            old_skill = skill
                            skill = self._switch_skill(skill, new_skill, cfg, base_mode)
                            if (
                                cfg.hard_switch
                                or (
                                    not matched_polling
                                    and base_mode
                                    in (
                                        StackMode.FIXED_VLM,
                                        StackMode.ALWAYS_ON,
                                    )
                                )
                                or not cfg.use_blend
                            ):
                                blending, blend_from, blend_tick = False, None, 0
                            else:
                                blending, blend_from, blend_tick = True, old_skill, 0
                            sigma_latch[skill] = False
                            if cfg.rearm_on_skill_change:
                                # A newly accepted intent owns a new intervention
                                # episode, even if the previous shield episode
                                # never cleared. Blending/rate limits still gate
                                # submission; same-skill replies never re-arm it.
                                shield_armed = True
                                shield_episode_called = False
                                override_streak = 0
                        pending = None

            # ---- Active skill action (+ optional crossfade). ---------------
            a_new, sigma_new, _ = self._skill_action(skill, snap)
            blend_alpha = 1.0
            sigma_skill = skill
            if blending and blend_from is not None:
                a_old, sigma_old, _ = self._skill_action(blend_from, snap)
                alpha = sigmoid_blend_alpha(blend_tick, cfg.blend_ticks)
                blend_alpha = alpha
                a_rl = blend_actions(a_old, a_new, alpha)
                if alpha >= 0.5:
                    sigma, sigma_skill = sigma_new, skill
                else:
                    sigma, sigma_skill = sigma_old, blend_from
                blend_tick += 1
                if blend_tick >= cfg.blend_ticks:
                    blending = False
                    blend_from = None
            else:
                a_rl = a_new
                sigma = sigma_new

            vlm_pending = pending is not None
            target_y = getattr(self.skills[skill].meta, "y_target", None)
            if target_y is not None:
                snap.info["target_y"] = float(target_y)
            sr = shield.filter(
                a_rl,
                snap,
                sigma=sigma if base_mode == StackMode.PROPOSED or matched_polling else 0.0,
                vlm_pending=vlm_pending,
            )

            # ---- Event trigger (proposed) or timer (baselines). -----------
            trigger_reason: Optional[str] = None
            fixed_due = False
            if base_mode == StackMode.PROPOSED and cfg.trigger_policy not in (
                "legacy",
                "uncertainty",
                "barrier",
            ):
                opportunity = lane_opportunity(snap, skill, transition_s=cfg.trigger_transition_s)
                if opportunity_due(
                    opportunity,
                    cfg.trigger_policy,
                    cfg.trigger_latency_bound_s,
                    cfg.trigger_buffer_s,
                    cfg.trigger_threshold_s,
                ):
                    trigger_reason = cfg.trigger_policy
            elif base_mode == StackMode.PROPOSED:
                threshold = self.skills[sigma_skill].meta.threshold * cfg.sigma_threshold_scale
                sigma_high = bool(sigma > threshold) and cfg.trigger_policy != "barrier"
                if not sigma_high:
                    sigma_latch[sigma_skill] = False

                # Shield-override trigger is a falling-edge-with-hysteresis
                # event, not a level.  The legacy controller required
                # ``shield_clear_ticks`` consecutive clear ticks before a new
                # shield interrupt; without that hysteresis the pending delay
                # margin itself keeps the shield overriding and produces an
                # endless call/margin feedback loop (or dense-lane chattering).
                if sr.overridden:
                    override_streak += 1
                    override_off = 0
                    # Stay armed for the whole override episode: the hold guard
                    # must not consume the arm on the first eligible tick.
                    shield_rise = (
                        shield_armed
                        and not shield_episode_called
                        and override_streak >= max(1, cfg.shield_hold_ticks)
                    )
                else:
                    override_streak = 0
                    override_off += 1
                    shield_rise = False
                    shield_episode_called = False
                    if override_off >= cfg.shield_clear_ticks:
                        shield_armed = True
                    if latch_reason in ("shield", "shield+sigma"):
                        latch_reason = None

                if shield_rise and cfg.trigger_policy != "uncertainty":
                    trigger_reason = "shield+sigma" if sigma_high else "shield"
                    # Fire at most once per override episode; a new episode
                    # requires the full clear window above.
                    shield_episode_called = True
                    shield_armed = False
                elif sigma_high and not sigma_latch.get(sigma_skill, False):
                    trigger_reason = "sigma"
                if trigger_reason is None and latch_reason is not None:
                    if latch_reason == "shield+sigma" and not sigma_high and sr.overridden:
                        latch_reason = "shield"
                    elif latch_reason == "shield+sigma" and sigma_high and not sr.overridden:
                        latch_reason = "sigma"
                    elif latch_reason in ("sigma", "shield+sigma") and not sigma_high:
                        latch_reason = None
                    elif latch_reason == "shield" and not sr.overridden:
                        latch_reason = None
                    if latch_reason is not None:
                        trigger_reason = latch_reason
            elif base_mode == StackMode.FIXED_VLM:
                fixed_due = t - last_rate_t >= (1.0 / cfg.fixed_rate_hz) - 1e-12
                trigger_reason = "fixed_rate" if fixed_due else None
            elif base_mode == StackMode.ALWAYS_ON:
                fixed_due = t - last_rate_t >= (1.0 / cfg.always_on_hz) - 1e-12
                trigger_reason = "always_on" if fixed_due else None

            if (
                cfg.polling_usefulness_gate
                and trigger_reason is not None
                and base_mode in (StackMode.FIXED_VLM, StackMode.ALWAYS_ON)
            ):
                if not lane_opportunity(snap, skill, transition_s=cfg.trigger_transition_s).useful:
                    trigger_reason = None
                    last_rate_t = t

            can_call = pending is None and not blending
            can_call = can_call and t + 1e-12 >= simulated_busy_until
            if worker_future is not None and not worker_future.done():
                can_call = False
            if base_mode == StackMode.PROPOSED:
                can_call = can_call and since_call >= cfg.min_call_ticks
            elif base_mode in (StackMode.FIXED_VLM, StackMode.ALWAYS_ON):
                can_call = can_call
            else:
                can_call = False

            if trigger_reason is not None:
                if can_call:
                    call_index += 1
                    request = RouterRequest(
                        t=t,
                        current_skill=skill,
                        snapshot=snap,
                        sigma=float(sigma),
                        reason=trigger_reason,
                        request_id=f"{run_token}-{call_index:03d}",
                        residual=float(sr.residual),
                        shield_h=float(sr.h) if math.isfinite(float(sr.h)) else None,
                        vlm_pending=vlm_pending,
                        deadline_s=cfg.max_pending_s,
                    )
                    if getattr(self.router, "requires_camera", False):
                        capture = getattr(env, "capture_router_camera", None)
                        if not callable(capture):
                            raise SafetyError("camera router requires an environment camera")
                        # Panda3D capture belongs on the simulation thread;
                        # the worker receives detached bytes only.
                        capture_started = time.perf_counter()
                        request.camera = capture()
                        request.camera_capture_wall_s = time.perf_counter() - capture_started
                        request.deadline_s = max(
                            1e-6, cfg.max_pending_s - request.camera_capture_wall_s
                        )
                    pending, pending_this_tick = self._submit(
                        request,
                        trigger_reason,
                        tracker,
                        cfg,
                        skill,
                        base_mode,
                        t,
                        executor=executor,
                    )
                    if isinstance(pending, FuturePendingCall):
                        worker_future = pending.future
                        background_calls.append(pending)
                    since_call = 0
                    last_rate_t = t
                    latch_reason = None
                    if trigger_reason in ("sigma", "shield+sigma"):
                        sigma_latch[sigma_skill] = True
                    if trigger_reason in ("shield", "shield+sigma"):
                        shield_episode_called = True
                    if pending is not None:
                        # The call is already in flight at time t: apply the
                        # pending margin on this very tick, not one tick later.
                        sr = shield.filter(
                            a_rl,
                            snap,
                            sigma=sigma
                            if base_mode == StackMode.PROPOSED or matched_polling
                            else 0.0,
                            vlm_pending=True,
                        )
                else:
                    # Latch the edge instead of dropping it. Count only the
                    # first suppressed edge of a burst.
                    if latch_reason is None and base_mode == StackMode.PROPOSED:
                        latch_reason = trigger_reason
                        if trigger_reason in ("shield", "shield+sigma"):
                            shield_episode_called = True
                        tracker.suppress_call(
                            {
                                "t": t,
                                "from": skill,
                                "to": skill,
                                "reason": trigger_reason,
                                "sigma": float(sigma),
                                "backend": getattr(self.router, "name", "unknown"),
                                "applied": False,
                            }
                        )

            # ---- Final safety gate. ---------------------------------------
            status = monitor.observe(sr)
            action = monitor.safe_action(sr.action, sr)
            if status == SafetyStatus.EMERGENCY:
                if cfg.abort_on_emergency:
                    # No plant step follows an abort, so no unapplied action
                    # may enter the replay log.
                    end_reason = "safety_abort"
                    break

            # ---- Metrics, then advance the plant. -------------------------
            tracker.record_tick(
                snap,
                action,
                sr,
                skill,
                sigma=sigma,
                vlm_pending=vlm_pending or pending_this_tick,
                pending_age_s=pending.age_s(t) if pending is not None else 0.0,
            )
            _obs, _reward, term, trunc, info = env.step(action)
            if cfg.record_interactions:
                tracker.interactions.append(
                    {
                        "t": t,
                        "skill": skill,
                        "sigma": float(sigma),
                        "blend_alpha": float(blend_alpha),
                        "requested_action": a_rl.tolist(),
                        "shield_action": sr.action.tolist(),
                        "applied_action": action.tolist(),
                        "shield_override": bool(sr.overridden),
                        "shield_feasible": bool(sr.feasible),
                        "lateral_override": bool(getattr(sr, "lateral_overridden", False)),
                        "lateral_h": getattr(sr, "lateral_h", None),
                        "shield_reason": sr.reason,
                        "braking_h": getattr(sr, "braking_h", None),
                        "braking_feasible": getattr(sr, "braking_feasible", True),
                        "cbf_override": bool(getattr(sr, "cbf_overridden", False)),
                        "braking_override": bool(getattr(sr, "braking_overridden", False)),
                        "barrier_h": float(sr.h) if math.isfinite(float(sr.h)) else None,
                        "inflation": float(sr.inflation),
                        "monitor": str(status),
                        "router_trigger": trigger_reason,
                        "router_pending": pending is not None or pending_this_tick,
                        "request_id": pending.request_id if pending is not None else None,
                        "plant": copy.deepcopy(info.get("plant", {})),
                        "ego": snap.ego.as_array().tolist(),
                        "obstacles": [other.as_array().tolist() for other in snap.others],
                    }
                )
            snap2 = info.get("snapshot") or env.snapshot()
            since_call += 1

            if snap2.crashed or info.get("crashed"):
                crashed = True
                crash_time = float(snap2.t)
                end_reason = "crash"
                if pending is not None or pending_this_tick:
                    tracker.pending_survival = False
                break
            if term:
                end_reason = info.get("end_reason") or (
                    "route_complete"
                    if float(snap2.ego.x) >= float(snap2.route_length)
                    else "terminated"
                )
                break
            if trunc:
                end_reason = "timeout"
                break

        if pending is not None:
            pending.record["aborted"] = True
            pending.record["applied"] = False

        # Control has ended. Join outstanding bounded work before finalizing
        # telemetry, including paid replies that arrived after expiry/route
        # completion. These answers are accounted for but can never be applied.
        for call in background_calls:
            if call.response is not None:
                continue
            try:
                response = call.future.result()
                validate_response(response, call.request)
                response.ready_at = call.submitted_t + max(
                    response.latency_s, response.wall_latency_s
                )
                response.latency_s = response.ready_at - call.submitted_t
                fields = response_record_fields(response)
                if call.record.get("error"):
                    fields.pop("error", None)
                call.record.update(fields)
                call.record["to"] = response.skill
                tracker.record_router_response(call.record)
                call.response = response
            except Exception as exc:
                if not call.record.get("rejected"):
                    call.record.update(rejected=True, applied=False, error=str(exc))
                    tracker.calls_rejected += 1

        result = tracker.finalize(
            env,
            crashed=crashed,
            crash_time=crash_time,
            end_reason=end_reason,
        )
        result.scenario = scenario
        result.seed = seed
        result.params = dict(tracker.params)
        self.last_shield = shield
        self.last_monitor = monitor
        health = getattr(self.router, "health", None)
        self.last_router_health = health().to_dict() if callable(health) else None
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_mode(mode_name: str) -> StackMode:
        if mode_name in _VARIANT_MODES:
            return StackMode.PROPOSED
        if mode_name == "fixed_rate_shielded":
            return StackMode.FIXED_VLM
        if mode_name == "always_on_shielded":
            return StackMode.ALWAYS_ON
        if mode_name == "always_on_vlm":
            return StackMode.ALWAYS_ON
        try:
            return StackMode(mode_name)
        except ValueError as exc:
            raise SafetyError(f"unknown stack mode {mode_name!r}") from exc

    @staticmethod
    def _configure_mode(
        shield: InflatedCBFShield,
        base_mode: StackMode,
        cfg: StackConfig,
        *,
        variant: str,
    ) -> None:
        if base_mode in (StackMode.UNSHIELDED, StackMode.FIXED_VLM, StackMode.ALWAYS_ON):
            shield.enabled = False
            shield.inflate = False
        elif base_mode == StackMode.STATIC_CBF:
            shield.enabled = True
            shield.inflate = False
        else:
            shield.enabled = True
            shield.inflate = True
            shield.pending_margin = bool(cfg.pending_margin)
        if variant == "proposed_no_margin":
            shield.pending_margin = False
            cfg.pending_margin = False
        if variant == "proposed_no_blend":
            cfg.use_blend = False
            cfg.hard_switch = True
        if variant in ("fixed_rate_shielded", "always_on_shielded"):
            shield.enabled = True
            shield.inflate = True
            shield.pending_margin = bool(cfg.pending_margin)
            # Matched polling changes scheduling/protection only. Respect the
            # caller's explicit switching/blending choice for fair ablations.

    def _skill_action(self, name: str, snap: WorldSnapshot):
        if name not in self.skills:
            raise SafetyError(f"router selected unknown skill {name!r}")
        try:
            ax, ay, sigma = self.skills[name].act(
                snap.ego.y, snap.ego.vy, snap.ego.vx, deterministic=True
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise SafetyError(f"skill {name!r} failed: {exc}") from exc
        action = np.array([ax, ay], dtype=np.float32)
        if not np.all(np.isfinite(action)) or not math.isfinite(float(sigma)):
            raise SafetyError(f"skill {name!r} produced a non-finite action or sigma")
        return action, float(sigma), self.skills[name]

    def _switch_skill(
        self,
        current: str,
        new: str,
        cfg: StackConfig,
        base_mode: StackMode,
    ) -> str:
        if new not in self.skills:
            raise SafetyError(f"router selected unknown skill {new!r}")
        return new

    def _switch_clear(self, new_skill: str, snap: WorldSnapshot) -> bool:
        """Conservative straight-road gap check before accepting new intent.

        Covers current and 1.5 s projected longitudinal gaps in the target
        corridor, including vehicles behind. This is a maneuver admission
        guard; the control-rate barrier still filters every accepted action.
        """
        target_y = float(self.skills[new_skill].meta.y_target)
        ego = snap.ego
        for other in snap.others:
            if abs(other.y - target_y) > 0.5 * (ego.width + other.width) + 0.35:
                continue
            dx = other.x - ego.x
            projected = dx + 1.5 * (other.vx - ego.vx)
            clearance = 0.5 * (ego.length + other.length) + 6.0
            if min(abs(dx), abs(projected)) < clearance or dx * projected <= 0:
                return False
        return True

    def _submit(
        self,
        request: RouterRequest,
        reason: str,
        tracker: RolloutTracker,
        cfg: StackConfig,
        current_skill: str,
        base_mode: StackMode,
        t: float,
        executor=None,
    ):
        record = {
            "t": float(t),
            "from": current_skill,
            "to": current_skill,
            "reason": reason,
            "sigma": float(request.sigma),
            "backend": getattr(self.router, "name", "unknown"),
            "ready_at": None,
            "applied": False,
            "rejected": False,
            "fallback_used": False,
            "wall_latency_ms": None,
            "prompt_hash": "",
            "error": None,
            "request_id": request.request_id,
            "camera_capture_wall_ms": 1000.0 * request.camera_capture_wall_s,
        }
        if request.camera_capture_wall_s >= cfg.max_pending_s:
            record.update(
                rejected=True, applied=False, error="camera capture exceeded pending deadline"
            )
            tracker.add_call(record)
            return None, False
        if executor is not None:

            def invoke():
                started = time.perf_counter()
                response = self.router.submit(copy.deepcopy(request))
                response.wall_latency_s = max(
                    float(response.wall_latency_s), time.perf_counter() - started
                )
                return response

            future = executor.submit(invoke)
            tracker.add_call(record)
            return FuturePendingCall(
                request=request,
                future=future,
                submitted_t=t,
                deadline_t=t + cfg.max_pending_s,
                record=record,
            ), True
        try:
            response = self.router.submit(request)
            validate_response(response, request)
            response.ready_at += request.camera_capture_wall_s
            response.latency_s += request.camera_capture_wall_s
        except RouterError as exc:
            record["rejected"] = True
            record["error"] = str(exc)
            tracker.add_call(record)
            logger.warning("router call rejected: %s", exc)
            return None, False

        record["to"] = response.skill
        record.update(response_record_fields(response))

        if response.skill == current_skill and not cfg.pending_margin_on_all_calls:
            # Legacy compatibility: same-skill answers do not lock the skill.
            record["applied"] = True
            tracker.add_call(record)
            return None, False

        deadline = float(t) + float(cfg.max_pending_s)
        if float(response.ready_at) > deadline and not cfg.simulate_deadline_wait:
            record["rejected"] = True
            record["error"] = (
                f"router ready_at={response.ready_at:.3f}s exceeds pending deadline={deadline:.3f}s"
            )
            tracker.add_call(record)
            return None, False

        tracker.add_call(record)
        return (
            PendingCall(
                request=request,
                response=response,
                submitted_t=float(t),
                deadline_t=deadline,
                record=record,
            ),
            True,
        )
