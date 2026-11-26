"""Unified 3-layer, strictly observation-only driving stack for SafeMetaDrive.

Architecture (matches the original 3-layer proposal, zero simulator peeking):
    Layer 3 (Slow Brain / Router):
        Event-triggered high-level maneuver router (stay / left / right / careful / stop).
        Supports deterministic LiDAR routing (with configurable decision delay),
        always-on polling (ablation), static-lane (ablation), or external VLM leases.
    Layer 2 (Fast Driver / Tracker):
        Control-rate lateral + longitudinal tracker. Supports either the trained
        PPO skill checkpoint (``tracker="ppo"``) or the analytic lane tracker
        (``tracker="analytic"``), with smooth steering blending across lane switches.
    Layer 1 (Fast Safety Filter):
        Strictly 259-dim LiDAR + car-sensor safety shield (no simulator object list):
        - Stopping-distance speed cap & direct-nose emergency braking
        - Road-boundary steering guard
        - Option-preserving maneuver reserve while Layer 3 is pending/blocked
        - Fresh LiDAR gap admission gate for delayed Layer 3 replies
        - Hard semantic lease enforcement (STOP / CAREFUL from Layer 3 VLM)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from driving.safemetadrive.lidar_driver import (
    DecodedObs,
    LidarOptionReserveDriver,
    decode_observation,
)


def _load_sb3_mlp_actor(zip_path: str):
    """Load the deterministic actor MLP from an SB3 .zip checkpoint using pure PyTorch."""
    import io
    import zipfile

    import torch

    torch.set_num_threads(1)
    with zipfile.ZipFile(zip_path) as zf:
        sd = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
    w0, b0 = sd["mlp_extractor.policy_net.0.weight"], sd["mlp_extractor.policy_net.0.bias"]
    w1, b1 = sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"]
    w2, b2 = sd["action_net.weight"], sd["action_net.bias"]
    net = torch.nn.Sequential(
        torch.nn.Linear(w0.shape[1], w0.shape[0]),
        torch.nn.Tanh(),
        torch.nn.Linear(w1.shape[1], w1.shape[0]),
        torch.nn.Tanh(),
        torch.nn.Linear(w2.shape[1], w2.shape[0]),
    )
    net[0].weight.data.copy_(w0)
    net[0].bias.data.copy_(b0)
    net[2].weight.data.copy_(w1)
    net[2].bias.data.copy_(b1)
    net[4].weight.data.copy_(w2)
    net[4].bias.data.copy_(b2)
    net.eval()

    def predict(feats: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return net(torch.as_tensor(feats, dtype=torch.float32)).numpy()

    return predict


@dataclass
class UnifiedConfig:
    # Layer 1: Fast Safety Filter
    use_safety_filter: bool = True
    option_reserve_m: float = 6.5
    crossing_guard: bool = True           # yield to converging traffic (LiDAR tracking)
    use_viability_reserve: bool = False  # Upgrade #3: closed-form Option-Viability Barrier
    expected_delay_s: float = 0.5        # Expected slow-planner latency for viability reserve
    headway: float = 1.15
    clearance: float = 3.2
    lateral_accel: float = 2.0
    max_speed_kmh: float = 36.0

    # Layer 2: Fast Driver
    tracker: str = "analytic"  # "analytic" or "ppo"
    skill_checkpoint: str | None = None
    blend_steps: int = 6       # 0 = hard switch ablation

    # Layer 3: Slow Router
    router: str = "event_lidar"  # "event_lidar", "always_poll", "none"
    decision_delay_steps: int = 0
    change_period: int = 10


class UnifiedDrivingSystem:
    """Single canonical 3-layer system evaluated consistently across all tables.

    Receives ONLY the 259-dim ``LidarStateObservation`` array — never ``vehicle`` or ``env``.
    """

    def __init__(self, observation: np.ndarray, config: UnifiedConfig | None = None, **overrides: Any):
        if config is None:
            config = UnifiedConfig(**overrides)
        elif overrides:
            data = {**config.__dict__, **overrides}
            config = UnifiedConfig(**data)
        self.config = config
        self.observation = np.asarray(observation, dtype=np.float32)

        # Reuse the verified pure-observation LiDAR perception & safety primitives
        self._core = LidarOptionReserveDriver(
            observation=self.observation,
            max_speed_kmh=config.max_speed_kmh,
            lateral_accel=config.lateral_accel,
            headway=config.headway,
            clearance=config.clearance,
            allow_lane_change=(config.router != "none"),
            option_reserve_m=config.option_reserve_m,
            decision_delay_steps=config.decision_delay_steps,
            protection=config.use_safety_filter,
            change_period=config.change_period,
            crossing_guard=config.crossing_guard,
        )

        # Optional trained PPO tracker (Layer 2) loaded directly from SB3 zip in pure PyTorch
        self._ppo = None
        if config.tracker == "ppo":
            if not config.skill_checkpoint:
                raise ValueError("tracker='ppo' requires skill_checkpoint")
            self._ppo = _load_sb3_mlp_actor(config.skill_checkpoint)

        # Telemetry & Layer 3 / Layer 2 state
        self.router_calls: int = 0
        self.semantic_stop: bool = False
        self.semantic_careful: bool = False
        # Upgrade #1: Grounded 3D semantic constraints compiled from Layer 3 VLM
        self.stop_wall_s: float | None = None   # Virtual stop wall distance ahead (m), tracked via odometry
        self.friction_scale: float = 1.0        # Estimated road friction multiplier mu in (0.15, 1.0]
        self.force_detour: bool = False         # Actively seek adjacent open lane around road closure
        self._prev_target_lane: int | None = None
        self._blend_rem: int = 0
        self._blend_from_steer: float = 0.0
        self._prev_steer: float = 0.0
        self.steer_deltas: list[float] = []

    @property
    def changes(self) -> int:
        return self._core.changes

    @property
    def interventions(self) -> int:
        return self._core.interventions

    @property
    def reserve_active_steps(self) -> int:
        return self._core.reserve_active_steps

    def set_semantic_lease(self, *, stop: bool = False, careful: bool = False) -> None:
        """Apply a high-level Layer 3 VLM semantic constraint (enforced by Layer 1)."""
        self.semantic_stop = bool(stop)
        self.semantic_careful = bool(careful)

    def set_grounded_constraints(
        self,
        *,
        stop_wall_s: float | None = None,
        friction_scale: float | None = None,
        force_detour: bool | None = None,
        clear_stop_wall: bool = False,
    ) -> None:
        """Compile grounded 3D physical constraints from Layer 3 directly into Layer 1."""
        if clear_stop_wall:
            self.stop_wall_s = None
        elif stop_wall_s is not None:
            self.stop_wall_s = max(0.0, float(stop_wall_s))
        if friction_scale is not None:
            self.friction_scale = float(np.clip(friction_scale, 0.2, 1.0))
        if force_detour is not None:
            self.force_detour = bool(force_detour)

    def on_observation(self, observation: np.ndarray) -> None:
        self.observation = np.asarray(observation, dtype=np.float32)
        self._core.on_observation(self.observation)

    def _build_ppo_features(self, d: DecodedObs, target_offset_m: float, desired_speed: float) -> np.ndarray:
        """Construct the exact 6-dim tracker input strictly from the 259-dim obs."""
        # In policies.py:
        #   alpha = atan2(delta_y, delta_x) - heading_theta (+ = target is LEFT of ego nose)
        #   lane_lat = local_coordinates(pos)[1] (+ = ego is RIGHT of lane center)
        #   lane_heading = lane_heading - ego_heading (+ = lane points LEFT of ego)
        #   curvature = 1 / radius
        # From our verified DecodedObs:
        #   target_offset_m = target_center_from_left - dist_left (+ = target is RIGHT of ego)
        #   d.heading_err > 0 = ego points RIGHT of lane -> lane points LEFT of ego (+)!
        look = max(3.0, 0.6 * d.speed_mps)
        # Target point in ego frame: fwd ~= look, left ~= -target_offset_m
        # Angle to target relative to lane = atan2(-target_offset_m, look); add lane_heading (+d.heading_err)
        alpha = math.atan2(-target_offset_m, look) + d.heading_err
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        lane_lat = -target_offset_m  # ego right of target => target_offset_m < 0 => lane_lat > 0
        lane_heading = d.heading_err
        curvature = (1.0 / d.bend1_radius_m) if math.isfinite(d.bend1_radius_m) else 0.0
        return np.array(
            [alpha, d.speed_mps / 15.0, desired_speed / 15.0, lane_lat / 5.0, lane_heading, curvature * 10.0],
            dtype=np.float32,
        )

    def act(self) -> list[float]:
        c = self.config
        d = decode_observation(self.observation)

        # Upgrade #3: dynamically update Option-Viability Reserve from (v, mu, tau)
        if c.use_viability_reserve:
            from driving.safemetadrive.viability import option_viability_reserve_m
            delay_s = max(c.expected_delay_s, c.decision_delay_steps * 0.1)
            self._core.option_reserve_m = option_viability_reserve_m(
                d.speed_mps,
                delay_s,
                lateral_accel=c.lateral_accel,
                friction_scale=self.friction_scale,
            )

        # Upgrade #1: adjust Layer 1 physics limits for wet/low-friction road
        self._core.lateral_accel = c.lateral_accel * math.sqrt(self.friction_scale)
        self._core.headway = c.headway / math.sqrt(self.friction_scale)

        # Advance virtual 3D stop wall via ego odometry (pure obs[3] speed * 0.1 s)
        if self.stop_wall_s is not None:
            self.stop_wall_s = max(0.0, self.stop_wall_s - d.speed_mps * 0.1)

        # Count Layer 3 router calls cleanly
        prev_pending = self._core.pending_target
        prev_changes = self._core.changes
        if c.router == "always_poll":
            self.router_calls += 1

        # Run core LiDAR perception, Layer 3 scheduling, and Layer 1 safety caps
        analytic_steer, safe_throttle = self._core.act()

        # Upgrade #1: Enforce virtual 3D stop wall smoothly via kinematic braking curve
        if c.use_safety_filter and self.stop_wall_s is not None:
            b_eff = max(0.8, 4.2 * self.friction_scale)
            avail = max(0.0, self.stop_wall_s - 2.5)
            wall_cap = math.sqrt(2.0 * b_eff * avail) if avail > 0.2 else 0.0
            wall_throttle = float(np.clip(0.85 * (wall_cap - d.speed_mps), -1.0, 0.85))
            safe_throttle = min(safe_throttle, wall_throttle)

        # Upgrade #1: When wet/low-friction road is active, apply gentle cruise cap without freezing
        if c.use_safety_filter and self.friction_scale < 0.85:
            wet_cruise = (c.max_speed_kmh / 3.6) * max(0.55, math.sqrt(self.friction_scale))
            if d.speed_mps > wet_cruise:
                safe_throttle = min(safe_throttle, float(np.clip(0.75 * (wet_cruise - d.speed_mps), -1.0, 0.0)))

        # Upgrade #1: Active detour around road closure (keep lane-change seeking enabled!)
        if self.force_detour and self._core.allow_lane_change:
            centers = self._core._lane_centers_from_left(d)
            cur = self._core.target_lane_idx if self._core.target_lane_idx is not None else self._core._current_lane_idx(d, centers)
            _, _, s_lane, d_right = self._core._lidar_to_road_points(d)
            gaps = self._core._lane_gaps(d, centers, s_lane, d_right)
            if gaps[cur][0] < 30.0:
                for cand in (cur - 1, cur + 1):
                    if 0 <= cand < d.n_lanes and gaps[cand][0] > gaps[cur][0] + 6.0:
                        if self._core._can_admit_lane(cand, gaps, d.speed_mps, relaxed=True):
                            self._core.target_lane_idx = cand
                            self._core.changes += 1
                            break

        if c.router == "event_lidar":
            # Count a call whenever a new delayed request was queued or an instant replan fired
            if (prev_pending is None and self._core.pending_target is not None) or (
                c.decision_delay_steps <= 0 and self._core.changes > prev_changes
            ):
                self.router_calls += 1

        # Trigger smooth blending window when target lane changes
        cur_target = self._core.target_lane_idx
        if self._prev_target_lane is not None and cur_target != self._prev_target_lane and c.blend_steps > 0:
            self._blend_rem = c.blend_steps
            self._blend_from_steer = self._prev_steer
        self._prev_target_lane = cur_target

        # Layer 2: choose analytic or trained PPO tracker
        steer = analytic_steer
        throttle = safe_throttle
        if self._ppo is not None:
            centers = self._core._lane_centers_from_left(d)
            target_offset_m = centers[cur_target] - d.dist_left
            # Reconstruct safe desired speed implied by safe_throttle
            desired_est = max(0.0, d.speed_mps + (safe_throttle / 0.45 if safe_throttle >= 0 else safe_throttle / 0.75))
            if self.semantic_careful:
                desired_est = min(desired_est, 0.7 * (c.max_speed_kmh / 3.6))
            if self.semantic_stop:
                desired_est = 0.0
            feats = self._build_ppo_features(d, target_offset_m, desired_est)
            learned = self._ppo(feats)
            steer = float(np.clip(learned[0], -1.0, 1.0))
            # Layer 1 longitudinal protection always caps Layer 2 throttle!
            throttle = min(float(np.clip(learned[1], -1.0, 1.0)), safe_throttle) if c.use_safety_filter else float(
                np.clip(learned[1], -1.0, 1.0)
            )
            # Layer 1 road-boundary guard always protects Layer 2 steering!
            if c.use_safety_filter:
                border_margin = 1.65
                if d.dist_left < border_margin:
                    steer = min(steer, -(border_margin - d.dist_left) * 1.35)
                if d.dist_right < border_margin:
                    steer = max(steer, (border_margin - d.dist_right) * 1.35)
                steer = float(np.clip(steer, -1.0, 1.0))
        else:
            if self.semantic_careful and d.speed_mps > 0.7 * (c.max_speed_kmh / 3.6):
                throttle = min(throttle, -0.35)
            if self.semantic_stop:
                throttle = -1.0 if d.speed_mps > 0.2 else 0.0

        # Smooth transition blending across lane changes (ablatable via blend_steps=0)
        if self._blend_rem > 0 and c.blend_steps > 0:
            w = 1.0 - (self._blend_rem / float(c.blend_steps + 1))
            steer = float(np.clip((1.0 - w) * self._blend_from_steer + w * steer, -1.0, 1.0))
            self._blend_rem -= 1

        self.steer_deltas.append(abs(steer - self._prev_steer))
        self._prev_steer = steer
        return [float(steer), float(throttle)]
