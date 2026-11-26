"""Strictly observation-only (259-dim) SafeMetaDrive driver with option-preserving reserve.

Fairness guarantee by construction:
    ``LidarOptionReserveDriver.act()`` uses ONLY the standard 259-float
    MetaDrive ``LidarStateObservation`` vector (identical to PPO-Lag, CPO, and
    the upstream PPO expert). It never receives or accesses ``env``, ``vehicle``,
    ``navigation``, road-network graphs, or simulator object lists.

Observation layout (verified against MetaDrive commit fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1):
    obs[0]      dist_to_left_side / 18.0         (total_width = (MAX_LANE_NUM + 1) * MAX_LANE_WIDTH = 4 * 4.5)
    obs[1]      dist_to_right_side / 18.0
    obs[2]      heading_diff in [0, 1]           (0.5 = aligned; <0.5 = pointing left of lane; >0.5 = right)
    obs[3]      (speed_km_h + 1) / 81.0          (max_speed_km_h = 80.0)
    obs[4]      (steering / MAX_STEERING + 1)/2
    obs[5]      (last_action_steer + 1)/2
    obs[6]      (last_action_throttle + 1)/2
    obs[7]      yaw_rate (clipped to [0, 1])
    obs[8]      (lateral_in_lane * 2 / 4.5 + 1)/2 (0.5 = centered; <0.5 = left; >0.5 = right)
    obs[9:14]   checkpoint 1: fwd, left_offset, bend_radius, bend_dir, bend_angle
    obs[14:19]  checkpoint 2: fwd, left_offset, bend_radius, bend_dir, bend_angle
    obs[19:259] 240 LiDAR rays in [0, 1] scaled by 50.0 m, CCW from ego heading (1.5 deg step)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TOTAL_WIDTH_SCALE = 18.0
MAX_LANE_WIDTH = 4.5
STANDARD_LANE_WIDTH = 3.5
MAX_SPEED_KMH = 80.0
LIDAR_RANGE = 50.0
NUM_LIDAR = 240
NAVI_POINT_DIST = 50.0
CURVE_RADIUS_SCALE = 55.5

# Precompute LiDAR ray unit vectors in ego frame (+x forward, +y left).
_RAY_ANGLES = np.arange(NUM_LIDAR, dtype=np.float64) * (2.0 * math.pi / NUM_LIDAR)
_RAY_COS = np.cos(_RAY_ANGLES)
_RAY_SIN = np.sin(_RAY_ANGLES)


@dataclass
class DecodedObs:
    dist_left: float
    dist_right: float
    road_width: float
    n_lanes: int
    heading_err: float  # rad: >0 pointing right of lane, <0 pointing left
    speed_mps: float
    steering_norm: float
    yaw_rate: float
    lat_in_lane: float  # m: >0 right of lane center, <0 left
    ckpt1_fwd: float
    ckpt1_left: float
    bend1_radius_m: float  # inf if straight
    bend1_dir: float       # +1 clockwise (right turn), -1 ccw (left turn), 0 straight
    ckpt2_fwd: float
    ckpt2_left: float
    bend2_radius_m: float
    bend2_dir: float
    lidar_m: np.ndarray    # shape (240,) in meters


def decode_observation(obs: np.ndarray) -> DecodedObs:
    """Decode the 259-dim SafeMetaDrive observation vector into physical units."""
    arr = np.asarray(obs, dtype=np.float64)
    if arr.shape != (259,):
        raise ValueError(f"Expected 259-dim SafeMetaDrive observation, got {arr.shape}")

    dist_left = float(arr[0]) * TOTAL_WIDTH_SCALE
    dist_right = float(arr[1]) * TOTAL_WIDTH_SCALE
    road_width = max(STANDARD_LANE_WIDTH, dist_left + dist_right)
    n_lanes = max(1, min(4, int(round(road_width / STANDARD_LANE_WIDTH))))

    heading_err = (float(arr[2]) - 0.5) * math.pi
    speed_kmh = max(0.0, float(arr[3]) * (MAX_SPEED_KMH + 1.0) - 1.0)
    speed_mps = speed_kmh / 3.6
    steering_norm = float(arr[4]) * 2.0 - 1.0
    yaw_rate = float(arr[7])
    lat_in_lane = (float(arr[8]) * 2.0 - 1.0) * (MAX_LANE_WIDTH / 2.0)

    ckpt1_fwd = (float(arr[9]) * 2.0 - 1.0) * NAVI_POINT_DIST
    ckpt1_left = (float(arr[10]) * 2.0 - 1.0) * NAVI_POINT_DIST
    r1_norm = float(arr[11])
    bend1_radius_m = (r1_norm * CURVE_RADIUS_SCALE) if r1_norm > 0.01 else math.inf
    bend1_dir = (float(arr[12]) * 2.0 - 1.0) if r1_norm > 0.01 else 0.0

    ckpt2_fwd = (float(arr[14]) * 2.0 - 1.0) * NAVI_POINT_DIST
    ckpt2_left = (float(arr[15]) * 2.0 - 1.0) * NAVI_POINT_DIST
    r2_norm = float(arr[16])
    bend2_radius_m = (r2_norm * CURVE_RADIUS_SCALE) if r2_norm > 0.01 else math.inf
    bend2_dir = (float(arr[17]) * 2.0 - 1.0) if r2_norm > 0.01 else 0.0

    lidar_m = arr[19:259] * LIDAR_RANGE
    return DecodedObs(
        dist_left=dist_left,
        dist_right=dist_right,
        road_width=road_width,
        n_lanes=n_lanes,
        heading_err=heading_err,
        speed_mps=speed_mps,
        steering_norm=steering_norm,
        yaw_rate=yaw_rate,
        lat_in_lane=lat_in_lane,
        ckpt1_fwd=ckpt1_fwd,
        ckpt1_left=ckpt1_left,
        bend1_radius_m=bend1_radius_m,
        bend1_dir=bend1_dir,
        ckpt2_fwd=ckpt2_fwd,
        ckpt2_left=ckpt2_left,
        bend2_radius_m=bend2_radius_m,
        bend2_dir=bend2_dir,
        lidar_m=lidar_m,
    )


def lidar_clusters(d: DecodedObs, gap_m: float = 1.2) -> list[np.ndarray]:
    """Group adjacent LiDAR hits into object clusters; returns centroids (x fwd, y left)."""
    valid = d.lidar_m < (LIDAR_RANGE - 0.6)
    if not np.any(valid):
        return []
    px = d.lidar_m * _RAY_COS
    py = d.lidar_m * _RAY_SIN
    clusters: list[list[int]] = []
    current: list[int] = []
    for i in range(NUM_LIDAR):
        if not valid[i]:
            if current:
                clusters.append(current)
                current = []
            continue
        if current and math.hypot(px[i] - px[current[-1]], py[i] - py[current[-1]]) > gap_m:
            clusters.append(current)
            current = []
        current.append(i)
    if current:
        clusters.append(current)
    # The scan is circular: merge the last cluster into the first if they touch at ray 0.
    if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == NUM_LIDAR - 1:
        a, b = clusters[0][0], clusters[-1][-1]
        if math.hypot(px[a] - px[b], py[a] - py[b]) <= gap_m:
            clusters[0] = clusters.pop() + clusters[0]
    return [np.array([px[c].mean(), py[c].mean()]) for c in clusters]


class CrossingTrafficGuard:
    """Yield to converging moving objects using only consecutive LiDAR scans.

    Tracks LiDAR clusters frame to frame, compensating ego motion with speed and
    the observed yaw rate (|obs[7]| signed by the steering command), estimates
    each object's ground velocity, and predicts whether a moving object will
    enter the ego's swept corridor (along its current arc) within ``horizon_s``.
    Lead vehicles already inside the corridor are left to the headway filter.
    """

    def __init__(self, horizon_s: float = 3.0, min_speed: float = 1.5, hold_s: float = 0.5,
                 dt: float = 0.1):
        self.horizon_s = horizon_s
        self.min_speed = min_speed
        self.hold_steps = int(round(hold_s / dt))
        self.dt = dt
        self.tracks: list[dict] = []
        self.hold = 0
        self.last_conflict_s: float | None = None
        self.next_id = 0
        self.window = 8  # frames in the least-squares velocity fit

    @staticmethod
    def ego_yaw_rate(d: DecodedObs) -> float:
        return math.copysign(min(1.0, d.yaw_rate), d.steering_norm) if d.steering_norm != 0 else 0.0

    def update(self, d: DecodedObs) -> bool:
        """Advance one control step; True when the ego should yield (stop)."""
        dt, v, w = self.dt, d.speed_mps, self.ego_yaw_rate(d)
        cos_w, sin_w = math.cos(-w * dt), math.sin(-w * dt)
        cents = lidar_clusters(d)
        new_tracks: list[dict] = []
        used: set[int] = set()
        rot = np.array([[cos_w, -sin_w], [sin_w, cos_w]])
        for tr in self.tracks:
            # Re-express the track history in the current ego frame (ego moved v*dt, turned w*dt):
            # a static object keeps the same compensated coordinates, so the history's slope is
            # the object's ground velocity.
            hist = (tr["hist"] - np.array([v * dt, 0.0])) @ rot.T
            pred = hist[-1] + tr["v"] * dt
            best, best_d = -1, 2.5
            for j, c in enumerate(cents):
                if j in used:
                    continue
                dist = float(np.hypot(*(c - pred)))
                if dist < best_d:
                    best, best_d = j, dist
            if best < 0:
                continue
            used.add(best)
            c = cents[best]
            hist = np.vstack([hist, c])[-self.window:]
            if len(hist) >= 3:
                t = np.arange(len(hist)) * dt
                t = t - t.mean()
                vel = (t @ (hist - hist.mean(axis=0))) / float(t @ t)  # least-squares slope
            else:
                vel = (hist[-1] - hist[0]) / dt
            new_tracks.append(dict(p=c, v=vel, hist=hist, age=tr["age"] + 1, id=tr["id"]))
        for j, c in enumerate(cents):
            if j not in used:
                new_tracks.append(dict(p=c, v=np.zeros(2), hist=c[None, :], age=0, id=self.next_id))
                self.next_id += 1
        self.tracks = new_tracks

        conflict = self._first_conflict(v, w)
        self.last_conflict_s = conflict
        if conflict is not None:
            self.hold = self.hold_steps
            return True
        if self.hold > 0:
            self.hold -= 1
            return True
        return False

    def static_objects(self, min_age: int = 3, max_speed: float = 1.0) -> list[dict]:
        """Tracked objects that are not moving over the ground (signs, people, cones, stopped cars)."""
        return [t for t in self.tracks if t["age"] >= min_age and float(np.hypot(*t["v"])) < max_speed]

    def _first_conflict(self, v: float, w: float) -> float | None:
        corridor_half_w, front, rear = 2.2, 6.5, -4.0  # front margin: stop clear of the crossing path
        steps = int(round(self.horizon_s / self.dt))
        for tr in self.tracks:
            if tr["age"] < 3 or float(np.hypot(*tr["v"])) < self.min_speed:
                continue
            p0 = tr["p"]
            if p0[0] < -1.0 or abs(p0[1]) <= corridor_half_w or float(np.hypot(*p0)) > 35.0:
                continue  # behind us, already in our corridor, or far away
            for k in range(1, steps + 1):
                t = k * self.dt
                psi = w * t
                if abs(w) > 1e-3:
                    ex, ey = v * math.sin(psi) / w, v * (1.0 - math.cos(psi)) / w
                else:
                    ex, ey = v * t, 0.0
                ox, oy = p0[0] + tr["v"][0] * t - ex, p0[1] + tr["v"][1] * t - ey
                c, s = math.cos(-psi), math.sin(-psi)
                rx, ry = c * ox - s * oy, s * ox + c * oy
                if rear < rx < front and abs(ry) < corridor_half_w:
                    return t
        return None


class LidarOptionReserveDriver:
    """Pure observation-only driver with LiDAR safety filter and option-preserving reserve."""

    def __init__(
        self,
        observation: np.ndarray,
        *,
        max_speed_kmh: float = 36.0,
        lateral_accel: float = 2.0,
        headway: float = 1.15,
        clearance: float = 3.2,
        allow_lane_change: bool = True,
        option_reserve_m: float = 6.5,
        decision_delay_steps: int = 0,
        protection: bool = True,
        change_period: int = 10,
        crossing_guard: bool = True,
    ):
        self.observation = np.asarray(observation, dtype=np.float32)
        self.crossing = CrossingTrafficGuard() if crossing_guard else None
        self.crossing_yield_steps = 0
        self.max_speed_kmh = float(max_speed_kmh)
        self.lateral_accel = float(lateral_accel)
        self.headway = float(headway)
        self.clearance = float(clearance)
        self.allow_lane_change = bool(allow_lane_change)
        self.option_reserve_m = float(option_reserve_m)
        self.decision_delay_steps = int(decision_delay_steps)
        self.protection = bool(protection)
        self.change_period = int(change_period)

        self.tick = 0
        self.changes = 0
        self.interventions = 0
        self.target_lane_idx: int | None = None
        self.prev_n_lanes: int | None = None
        self.prev_front_gap: float | None = None
        self.est_front_speed: float = max_speed_kmh / 3.6
        self.prev_steer: float = 0.0
        self.last_change_tick: int = -100
        self.stuck_steps: int = 0

        self.pending_target: int | None = None
        self.pending_ready_tick: int = -1
        self.reserve_active_steps: int = 0

    def on_observation(self, observation: np.ndarray) -> None:
        self.observation = np.asarray(observation, dtype=np.float32)

    def _lane_centers_from_left(self, d: DecodedObs) -> list[float]:
        lw = d.road_width / d.n_lanes
        return [(i + 0.5) * lw for i in range(d.n_lanes)]

    def _current_lane_idx(self, d: DecodedObs, centers: list[float]) -> int:
        return int(min(range(len(centers)), key=lambda i: abs(d.dist_left - centers[i])))

    def _lidar_to_road_points(self, d: DecodedObs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert valid LiDAR hits to ego (x_ego, y_ego_right) and road (s_lane, d_right)."""
        mask = d.lidar_m < (LIDAR_RANGE - 0.6)
        if not np.any(mask):
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty, empty

        r = d.lidar_m[mask]
        x_ego = r * _RAY_COS[mask]
        y_ego_right = -r * _RAY_SIN[mask]

        ch = math.cos(d.heading_err)
        sh = math.sin(d.heading_err)
        s_lane = x_ego * ch - y_ego_right * sh
        d_right = x_ego * sh + y_ego_right * ch

        if math.isfinite(d.bend1_radius_m) and d.bend1_radius_m > 5.0:
            s_clip = np.clip(s_lane, -15.0, 35.0)
            curve_offset = d.bend1_dir * (s_clip * s_clip) / (2.0 * d.bend1_radius_m)
            d_right = d_right - curve_offset

        return x_ego, y_ego_right, s_lane, d_right

    def _lane_gaps(
        self,
        d: DecodedObs,
        centers: list[float],
        s_lane: np.ndarray,
        d_right: np.ndarray,
    ) -> list[tuple[float, float]]:
        gaps: list[tuple[float, float]] = []
        ego_half_len = 2.25
        for c_left in centers:
            lane_rel_right = c_left - d.dist_left
            in_lane = np.abs(d_right - lane_rel_right) < 1.50
            if not np.any(in_lane):
                gaps.append((LIDAR_RANGE, LIDAR_RANGE))
                continue
            s_hits = s_lane[in_lane]
            fwd_hits = s_hits[s_hits > 0.5]
            rear_hits = -s_hits[s_hits < -0.5]
            front = max(0.0, float(np.min(fwd_hits)) - ego_half_len) if fwd_hits.size else LIDAR_RANGE
            rear = max(0.0, float(np.min(rear_hits)) - ego_half_len) if rear_hits.size else LIDAR_RANGE
            gaps.append((front, rear))
        return gaps

    def _ego_nose_hazard_dist(self, d: DecodedObs, x_ego: np.ndarray, y_ego_right: np.ndarray) -> float:
        """Immediate ego-body forward swept box in ego frame (catches direct collisions & cut-ins)."""
        if x_ego.size == 0:
            return LIDAR_RANGE
        look_x = max(7.0, min(26.0, d.speed_mps * 2.2 + 4.0))
        # Ego half-width is ~0.925 m; widen gently with distance to catch crossing/cut-in vehicles
        # while staying inside adjacent-lane centers (3.5 m away).
        half_w = 1.15 + 0.075 * np.clip(x_ego, 0.0, look_x)
        in_box = (x_ego > 0.3) & (x_ego < look_x) & (np.abs(y_ego_right) < half_w)
        if not np.any(in_box):
            return LIDAR_RANGE
        return max(0.0, float(np.min(x_ego[in_box])) - 2.25)

    def _compute_curve_speed_cap(self, d: DecodedObs) -> float:
        cap = self.max_speed_kmh / 3.6
        if math.isfinite(d.bend1_radius_m):
            cap = min(cap, math.sqrt(self.lateral_accel * max(6.0, d.bend1_radius_m)))
        dist_ckpt1 = math.hypot(d.ckpt1_fwd, d.ckpt1_left)
        if math.isfinite(d.bend2_radius_m) and dist_ckpt1 < max(30.0, d.speed_mps * 3.5):
            next_cap = math.sqrt(self.lateral_accel * max(6.0, d.bend2_radius_m))
            avail = max(0.0, dist_ckpt1 - 6.0)
            entry_cap = math.sqrt(next_cap * next_cap + 2.0 * 2.6 * avail)
            cap = min(cap, entry_cap)
        if d.ckpt1_fwd > 1.0:
            lateral_ratio = abs(d.ckpt1_left) / max(8.0, d.ckpt1_fwd)
            if lateral_ratio > 0.30:
                cap = min(cap, max(4.2, 10.0 - 6.5 * lateral_ratio))
        return cap

    def _can_admit_lane(self, idx: int, gaps: list[tuple[float, float]], speed: float, *, relaxed: bool = False) -> bool:
        front, rear = gaps[idx]
        if relaxed:
            min_front = max(6.0, speed * 1.0 + 2.5)
            min_rear = max(7.0, speed * 0.7 + 2.0)
        else:
            min_front = max(9.5, speed * 1.2 + 3.0)
            min_rear = max(9.5, speed * 0.85 + 2.0)
        return front >= min_front and rear >= min_rear

    def act(self) -> list[float]:
        d = decode_observation(self.observation)
        centers = self._lane_centers_from_left(d)
        cur_idx = self._current_lane_idx(d, centers)

        if self.target_lane_idx is None or self.prev_n_lanes != d.n_lanes:
            self.target_lane_idx = cur_idx
            self.pending_target = None
            self.prev_n_lanes = d.n_lanes
        else:
            self.target_lane_idx = max(0, min(d.n_lanes - 1, self.target_lane_idx))

        if d.speed_mps < 0.8:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        x_ego, y_ego_right, s_lane, d_right = self._lidar_to_road_points(d)
        gaps = self._lane_gaps(d, centers, s_lane, d_right)

        cur_front, _ = gaps[self.target_lane_idx]
        if self.prev_front_gap is not None and cur_front < LIDAR_RANGE - 1.0 and self.prev_front_gap < LIDAR_RANGE - 1.0:
            raw_front_v = d.speed_mps + (cur_front - self.prev_front_gap) / 0.1
            raw_front_v = float(np.clip(raw_front_v, 0.0, self.max_speed_kmh / 3.6))
            self.est_front_speed = 0.65 * self.est_front_speed + 0.35 * raw_front_v
        elif cur_front >= LIDAR_RANGE - 1.0:
            self.est_front_speed = self.max_speed_kmh / 3.6
        self.prev_front_gap = cur_front

        # --- High-level lane planner (with optional simulated decision latency) ---
        target_offset_m = centers[self.target_lane_idx] - d.dist_left
        # If nearly stopped and stuck, allow replanning even if mid-transition
        settled = abs(target_offset_m) < (1.9 if self.stuck_steps >= 10 else 0.90)

        if self.pending_target is not None and self.tick >= self.pending_ready_tick:
            cand = max(0, min(d.n_lanes - 1, self.pending_target))
            if cand != self.target_lane_idx and settled and self._can_admit_lane(cand, gaps, d.speed_mps, relaxed=self.stuck_steps >= 10):
                self.target_lane_idx = cand
                self.changes += 1
                self.last_change_tick = self.tick
                self.prev_front_gap = None
            self.pending_target = None

        obstacle_ahead = cur_front < max(26.0, d.speed_mps * 2.8) or self.stuck_steps >= 10
        if (
            self.allow_lane_change
            and settled
            and self.pending_target is None
            and (self.tick - self.last_change_tick) >= (5 if self.stuck_steps >= 15 else self.change_period)
            and obstacle_ahead
        ):
            best = self.target_lane_idx
            best_score = min(cur_front, 45.0)
            # Check adjacent lanes first; if stuck, also consider steering toward a clearer 2-lanes-away lane one step at a time
            candidates = [c for c in (self.target_lane_idx - 1, self.target_lane_idx + 1) if 0 <= c < d.n_lanes]
            for cand in candidates:
                if not self._can_admit_lane(cand, gaps, d.speed_mps, relaxed=self.stuck_steps >= 10):
                    continue
                cand_front, _ = gaps[cand]
                # Bonus if cand leads toward an even clearer outer lane
                bonus = 0.0
                if self.stuck_steps >= 10:
                    far = cand + (cand - self.target_lane_idx)
                    if 0 <= far < d.n_lanes and gaps[far][0] > 20.0:
                        bonus = 8.0
                gain_req = 2.0 if self.stuck_steps >= 10 else 6.0
                if cand_front + bonus > best_score + gain_req:
                    best = cand
                    best_score = cand_front + bonus
            if best != self.target_lane_idx:
                if self.decision_delay_steps <= 0:
                    self.target_lane_idx = best
                    self.changes += 1
                    self.last_change_tick = self.tick
                    self.prev_front_gap = None
                else:
                    self.pending_target = best
                    self.pending_ready_tick = self.tick + self.decision_delay_steps

        # --- Option-Preserving Maneuver Reserve ---
        target_offset_m = centers[self.target_lane_idx] - d.dist_left
        in_transition = abs(target_offset_m) >= 0.90
        has_adjacent_lane = d.n_lanes > 1
        reserve_m = 0.0
        if (
            self.protection
            and self.allow_lane_change
            and has_adjacent_lane
            and not in_transition
            and self.option_reserve_m > 0.0
            and (self.pending_target is not None or (cur_front < 24.0 and self.est_front_speed < 3.0))
        ):
            reserve_m = self.option_reserve_m
            self.reserve_active_steps += 1

        # --- Longitudinal Safety Filter (Pure LiDAR) ---
        cruise_cap = self._compute_curve_speed_cap(d)
        desired_speed = cruise_cap

        if self.protection:
            tgt_front = gaps[self.target_lane_idx][0]
            nose_d = self._ego_nose_hazard_dist(d, x_ego, y_ego_right)
            eff_front = min(tgt_front, nose_d)

            avail = max(0.0, eff_front - (self.clearance + reserve_m))
            brake_a = 4.2
            fv = self.est_front_speed if eff_front == tgt_front else 0.0
            cap = -brake_a * self.headway + math.sqrt(
                (brake_a * self.headway) ** 2 + fv * fv + 2.0 * brake_a * avail
            )
            desired_speed = min(desired_speed, max(0.0, cap))

            # Yield to converging traffic (roundabout entries, intersections, cut-ins)
            yield_now = self.crossing.update(d) if self.crossing is not None else False
            if yield_now:
                desired_speed = 0.0
                self.crossing_yield_steps += 1
            # Emergency close-range stop only for true direct-nose collision risk
            if nose_d < max(2.8, d.speed_mps * 0.65):
                desired_speed = 0.0
            # Anti-deadlock creep: if transitioning into a clear target lane and nose has >2.8 m, allow slow creep
            elif in_transition and not yield_now and tgt_front > 8.0 and nose_d >= 2.8 and desired_speed < 2.2:
                desired_speed = 2.2

            if desired_speed < cruise_cap - 0.15:
                self.interventions += 1

        # --- Lateral Lane Tracking + Road-Boundary Guard (Pure Obs) ---
        speed_gain = 1.0 / (1.0 + 0.05 * d.speed_mps)
        k_lat = 0.36 * speed_gain
        k_head = 1.45

        # Gentle curve feedforward (kept modest so it never oversteers into inside curbs)
        ff_steer = 0.0
        if math.isfinite(d.bend1_radius_m) and d.bend1_radius_m > 5.0:
            ff_steer = float(np.clip(-d.bend1_dir * (0.9 / d.bend1_radius_m), -0.12, 0.12))

        # Slight inward bias on tight curves to stay well clear of the outer/inner curb
        curve_bias = 0.0
        if math.isfinite(d.bend1_radius_m) and d.bend1_radius_m < 25.0:
            if d.dist_right < 2.2:
                curve_bias = -0.35  # bias target slightly left (away from right curb)
            elif d.dist_left < 2.2:
                curve_bias = +0.35  # bias target slightly right (away from left curb)

        raw_steer = k_head * d.heading_err - k_lat * np.clip(target_offset_m + curve_bias, -4.0, 4.0) + ff_steer

        # Smooth steering rate limit for normal driving
        max_dsteer = 0.32 if d.speed_mps > 6.0 else 0.45
        steer = float(np.clip(raw_steer, self.prev_steer - max_dsteer, self.prev_steer + max_dsteer))

        # Road-boundary CBF-style guard (bypasses rate limit when near road edge!)
        if self.protection:
            border_margin = 1.65
            if d.dist_left < border_margin:
                push = (border_margin - d.dist_left) * 1.35
                steer = min(steer, -push)
            if d.dist_right < border_margin:
                push = (border_margin - d.dist_right) * 1.35
                steer = max(steer, push)

        steer = float(np.clip(steer, -1.0, 1.0))
        self.prev_steer = steer

        speed_err = desired_speed - d.speed_mps
        if speed_err >= 0:
            throttle = float(np.clip(0.45 * speed_err, 0.0, 0.85))
        else:
            throttle = float(np.clip(0.75 * speed_err, -1.0, 0.0))

        self.tick += 1
        return [steer, throttle]
