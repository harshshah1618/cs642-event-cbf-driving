"""Sensor-range rule control and upstream baselines for the initial feasibility study.

The candidate is explicitly a rule controller, not a learned PPO or camera model.
Navigation geometry and objects returned by the upstream 50 m lidar are used.
"""

import math
from typing import Any

import numpy as np


class LaneCautionDriver:
    def __init__(self, vehicle, speed=40.0, headway=1.0, lateral_accel=2.5,
                 clearance=3.0, change_period=15, preview=True, prediction=True,
                 planner="rule", skill_checkpoint=None, protection=True, initial_careful=False,
                 manage_maneuvers=False, control_dt=0.1):
        from metadrive.policy.idm_policy import IDMPolicy

        self.vehicle = vehicle
        self.tracker = IDMPolicy(vehicle, random_seed=0)
        self.speed = speed
        self.headway = headway
        self.lateral_accel = lateral_accel
        self.clearance = clearance
        self.change_period = change_period
        self.lane: Any = None
        self.tick = 0
        self.changes = 0
        self.interventions = 0
        self.preview = preview
        self.prediction = prediction
        self.planner = planner
        self.protection = protection
        self.careful = initial_careful
        self.last_features = None
        self.last_teacher_action = None
        self.last_caution_label = False
        self.last_intent_reason = "no_previous_request"
        self.last_caution_applied = False
        self.maneuvers = None
        if manage_maneuvers:
            from driving.safemetadrive.maneuvers import ManeuverSupervisor
            self.maneuvers = ManeuverSupervisor(dt=control_dt)
        self.skill = None
        if skill_checkpoint:
            from stable_baselines3 import PPO
            self.skill = PPO.load(skill_checkpoint, device="cpu")

    def apply_intent(self, delta, careful, capture_lane):
        """Admit a lane target captured before inference, using fresh gap checks."""
        lanes = self.vehicle.navigation.current_ref_lanes
        self.last_caution_applied = False
        if capture_lane not in lanes or self.lane is not capture_lane:
            self.last_intent_reason = "stale_lane_context"
            return False
        supervisor = getattr(self, "maneuvers", None)
        # A current-context request to be more cautious can survive a rejected
        # lane change. A rejected normal-speed request never removes caution.
        if supervisor is not None and careful:
            self.careful = True
            self.last_caution_applied = True
        current = lanes.index(capture_lane)
        target = current + int(delta)
        if not 0 <= target < len(lanes):
            self.last_intent_reason = "invalid_adjacent_lane"
            return False
        if delta:
            _, lat = self.lane.local_coordinates(self.vehicle.position)
            if abs(lat) > 0.65:
                self.last_intent_reason = "transition_unfinished_or_off_center"
                return False
            objs = self.vehicle.lidar.get_surrounding_objects(self.vehicle)
            f, r, _ = self.lane_objects(lanes[target], objs)
            if self.protection and (f < max(8, self.vehicle.speed*1.1) or r < max(10, self.vehicle.speed*0.8)):
                self.last_intent_reason = "front_or_rear_gap_too_small"
                return False
            if supervisor is not None:
                reason = supervisor.admission_reason(self, lanes[target], objs)
                if reason is not None:
                    self.last_intent_reason = reason
                    return False
                supervisor.start(self.lane, lanes[target], self.vehicle.position)
            self.lane = lanes[target]
            self.changes += 1
        returning = supervisor is not None and supervisor.phase == "returning"
        self.careful = bool(careful) or returning
        self.last_caution_applied = self.careful == bool(careful)
        self.last_intent_reason = "accepted"
        return True

    def planner_feedback(self):
        if self.maneuvers is None:
            return {}
        return dict(**self.maneuvers.feedback(), last_intent_reason=self.last_intent_reason,
                    caution_applied=self.last_caution_applied)

    def path_point(self, distance):
        """Continue the selected lane across the next navigation road seam."""
        lane = self.lane
        s, _ = lane.local_coordinates(self.vehicle.position)
        s += distance
        nxt = self.vehicle.navigation.next_ref_lanes
        if s > lane.length and nxt:
            end = lane.position(lane.length, 0)
            successor = min(nxt, key=lambda n: float(np.linalg.norm(n.position(0, 0) - end)))
            s, lane = s - lane.length, successor
        return lane.position(min(s, lane.length), 0), lane.heading_theta_at(min(s, lane.length))

    def conflict_time(self, objects, *, include_rear=False):
        """Sample oriented footprints under constant-velocity object predictions.

        This is a heuristic collision forecast, not an invariant-set certificate.
        It also checks crossing traffic that is outside the current lane corridor.
        """
        v = self.vehicle
        speed = max(v.speed, 2.0)
        horizon = 3.0
        forward = np.array([math.cos(v.heading_theta), math.sin(v.heading_theta)])
        _, lat = self.lane.local_coordinates(v.position)
        paths = []
        for t in np.arange(0.2, horizon + 0.01, 0.2):
            point, heading = self.path_point(speed * t)
            h = np.array([math.cos(heading), math.sin(heading)])
            side = np.array([-h[1], h[0]])
            # MetaDrive's positive lateral coordinate points to the right.
            point = point - side * lat * math.exp(-t / 1.2)
            paths.append((t, point, h, side))
        earliest = math.inf
        for obj in objects:
            delta = obj.position - v.position
            oh = float(getattr(obj, "heading_theta", 0))
            axis = np.array([math.cos(oh), math.sin(oh)])
            other_side = np.array([-axis[1], axis[0]])
            velocity = np.asarray(getattr(obj, "velocity", [0, 0]))
            # Rear following traffic is not made safer by braking in front of it.
            if not include_rear and delta @ forward < -v.LENGTH and axis @ forward > 0.5:
                continue
            if np.linalg.norm(delta) > (speed + np.linalg.norm(velocity))*horizon + 10:
                continue
            length, width = float(getattr(obj, "LENGTH", 1)), float(getattr(obj, "WIDTH", 1))
            for t, point, h, side in paths:
                if t >= earliest:
                    break
                difference = obj.position + velocity*t - point
                longitudinal = v.LENGTH/2 + length/2*abs(axis@h) + width/2*abs(other_side@h) + 0.8
                transverse = v.WIDTH/2 + length/2*abs(axis@side) + width/2*abs(other_side@side) + 0.35
                other_long = length/2 + v.LENGTH/2*abs(h@axis) + v.WIDTH/2*abs(side@axis) + 0.8
                other_lat = width/2 + v.LENGTH/2*abs(h@other_side) + v.WIDTH/2*abs(side@other_side) + 0.35
                if (abs(difference@h) < longitudinal and abs(difference@side) < transverse
                        and abs(difference@axis) < other_long
                        and abs(difference@other_side) < other_lat):
                    earliest = t
                    break
        return earliest

    def lane_objects(self, lane, objects):
        ego_s, _ = lane.local_coordinates(self.vehicle.position)
        front, rear, front_v = 80.0, 80.0, self.speed / 3.6
        for obj in objects:
            s, d = lane.local_coordinates(obj.position)
            width = float(getattr(obj, "WIDTH", 1.0))
            length = float(getattr(obj, "LENGTH", 1.0))
            if abs(d) > (self.vehicle.WIDTH + width) / 2 + 0.25:
                continue
            gap = s - ego_s
            half_length = (self.vehicle.LENGTH + length) / 2
            if gap >= 0 and gap - half_length < front:
                front = max(0, gap - half_length)
                heading = lane.heading_theta_at(s)
                vel = np.asarray(getattr(obj, "velocity", [0, 0]))
                front_v = max(0, float(vel @ [math.cos(heading), math.sin(heading)]))
            elif gap < 0:
                rear = min(rear, max(0, -gap - half_length))
        return front, rear, front_v

    def synchronize_lane(self):
        v = self.vehicle
        lanes = v.navigation.current_ref_lanes
        if not lanes:
            raise RuntimeError("Navigation has no current lanes")
        if self.lane not in lanes:
            network = v.navigation.map.road_network
            connected = [x for x in lanes if self.lane is not None and
                         network.has_connection(self.lane.index, x.index)]
            candidates = connected if connected else lanes
            # Several parallel lanes can share a road-network connection. Taking
            # the first one makes an unintended multi-lane change at every seam.
            self.lane = min(candidates, key=lambda lane: abs(lane.local_coordinates(v.position)[1]))
            supervisor = getattr(self, "maneuvers", None)
            if supervisor is not None:
                supervisor.reset_road()
        return lanes

    def act(self):
        v = self.vehicle
        lanes = self.synchronize_lane()
        objects = v.lidar.get_surrounding_objects(v)
        if self.maneuvers is not None:
            self.maneuvers.update(self, objects)
        current = lanes.index(self.lane)
        ego_speed = v.speed
        gaps = [self.lane_objects(lane, objects) for lane in lanes]
        s, lat = self.lane.local_coordinates(v.position)
        # Keep the selected target until the lane transition has finished.
        if self.planner == "rule" and self.tick % self.change_period == 0 and abs(lat) < 0.65:
            front, _, front_v = gaps[current]
            best = current
            best_score = min(front, 50) + 1.5 * front_v
            for i in (current - 1, current + 1):
                if not 0 <= i < len(lanes):
                    continue
                f, r, fv = gaps[i]
                if f < max(8, ego_speed * 1.1) or r < max(10, ego_speed * 0.8):
                    continue
                score = min(f, 50) + 1.5 * fv - 8
                if front < max(25, ego_speed * 3) and score > best_score:
                    best, best_score = i, score
            # Align with available successor lanes before a merge.
            nxt = v.navigation.next_ref_lanes
            if nxt and self.lane.length - s < max(30, ego_speed * 4):
                network = v.navigation.map.road_network
                valid = [i for i, lane in enumerate(lanes) if any(
                    network.has_connection(lane.index, n.index) for n in nxt)]
                if valid and best not in valid:
                    toward = current + int(np.sign(min(valid, key=lambda i: abs(i-current))-current))
                    if 0 <= toward < len(lanes) and gaps[toward][0] > 8 and gaps[toward][1] > 10:
                        best = toward
            if best != current:
                if self.maneuvers is not None:
                    reason = self.maneuvers.admission_reason(self, lanes[best], objects)
                    if reason is not None:
                        best = current
                    else:
                        self.maneuvers.start(self.lane, lanes[best], v.position)
            if best != current:
                self.lane = lanes[best]
                self.changes += 1
                current = best
        self.tick += 1
        s, _ = self.lane.local_coordinates(v.position)
        look = max(8, ego_speed * 2)
        headings = ([self.path_point(d)[1] for d in (0, look / 2, look)] if self.preview else
                    [self.lane.heading_theta_at(min(self.lane.length, s + d))
                     for d in (0, look / 2, look)])
        angles = np.unwrap(headings)
        curvature = max(abs(np.diff(angles))) / (look / 2)
        cruise = self.speed / 3.6 * (0.7 if self.careful else 1.0)
        if self.maneuvers is not None and self.maneuvers.phase == "returning":
            cruise = min(cruise, 2.0)
        target = min(cruise, math.sqrt(self.lateral_accel / max(curvature, 1e-4)))
        # A stopping-distance speed cap, with the front object's speed retained.
        front, _, fv = gaps[current]
        available = max(0, front - self.clearance)
        # b*T^2 relation solves v*T + (v^2-v_front^2)/(2b) <= available.
        brake = 4.0
        headway = self.headway * (1.5 if self.careful else 1.0)
        cap = -brake * headway + math.sqrt(
            (brake * headway)**2 + fv**2 + 2 * brake * available)
        desired = min(target, max(0, cap)) if self.protection else target
        conflict = math.inf
        if self.prediction and self.protection:
            conflict = self.conflict_time(objects)
            if conflict < max(1.2, ego_speed / 4.0 + 0.6):
                desired = 0.0
            elif conflict < 3.0:
                desired = min(desired, max(0, ego_speed - 1.0))
        point, _ = self.path_point(max(3.0, 0.6 * ego_speed))
        delta = point - v.position
        alpha = math.atan2(delta[1], delta[0]) - v.heading_theta
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        _, lane_lat = self.lane.local_coordinates(v.position)
        lane_heading = self.lane.heading_theta_at(s) - v.heading_theta
        lane_heading = math.atan2(math.sin(lane_heading), math.cos(lane_heading))
        self.last_features = np.array([alpha, ego_speed/15, desired/15,
                                       lane_lat/5, lane_heading, curvature*10], dtype=np.float32)
        self.last_caution_label = bool(curvature > 0.015 or conflict < 3.0 or front < max(20, ego_speed*3))
        if self.preview:
            wheelbase = v.FRONT_WHEELBASE + v.REAR_WHEELBASE
            max_steer = math.radians(v.get_dynamics_parameters()["max_steering"])
            steer = math.atan2(2 * wheelbase * math.sin(alpha), max(float(np.linalg.norm(delta)), 1.0)) / max_steer
        else:
            steer = self.tracker.steering_control(self.lane)
        throttle = float(np.clip(0.5 * (desired - ego_speed), -1, 1))
        self.last_teacher_action = np.array([float(np.clip(steer, -1, 1)), throttle], dtype=np.float32)
        if self.skill is not None:
            learned, _ = self.skill.predict(self.last_features, deterministic=True)
            steer = float(learned[0])
            # Runtime longitudinal protection is applied after the skill action.
            throttle = min(float(learned[1]), throttle) if self.protection else float(learned[1])
        if desired < target - 0.1:
            self.interventions += 1
        return [float(np.clip(steer, -1, 1)), throttle]

    def close(self):
        self.tracker.destroy()


class NativeLearnedDriver:
    """OmniSafe Gaussian actor on the unmodified native observation."""
    def __init__(self, checkpoint, observation, deterministic=True, policy_seed=0):
        import torch

        self.torch = torch
        torch.set_num_threads(1)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved["pi"]
        # OmniSafe obs_normalize=True saves its running normalizer; apply it exactly as
        # omnisafe.common.normalizer.Normalizer.normalize does: clamp((x - mean) / std, -clip, clip).
        norm = saved.get("obs_normalizer")
        self.norm = None if norm is None else (norm["_mean"].float(), norm["_std"].float(),
                                               norm["_clip"].float())
        parameters = {k.removeprefix("mean."): v for k, v in state.items() if k.startswith("mean.")}
        weights = sorted((int(k.split(".")[0]), value) for k, value in parameters.items() if k.endswith("weight"))
        layers: list[Any] = []
        for index, (_, weight) in enumerate(weights):
            layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
            layers.append(torch.nn.Tanh() if index < len(weights)-1 else torch.nn.Identity())
        self.actor = torch.nn.Sequential(*layers)
        self.actor.load_state_dict(parameters)
        self.actor.eval()
        self.observation = observation
        self.deterministic = deterministic
        self.std = state["log_std"].exp()
        self.generator = torch.Generator().manual_seed(policy_seed)
        self.policy_seed = policy_seed

    def on_observation(self, observation):
        self.observation = observation

    def act(self):
        with self.torch.inference_mode():
            x = self.torch.as_tensor(self.observation, dtype=self.torch.float32)
            if self.norm is not None:
                mean, std, clip = self.norm
                x = self.torch.maximum(self.torch.minimum((x - mean) / std, clip), -clip)
            action = self.actor(x)
            if not self.deterministic:
                action = action+self.torch.randn(action.shape, generator=self.generator)*self.std
            return action.numpy()


def make_driver(name, vehicle, parameters, observation=None, episode_seed=0):
    if name == "unified_system":
        from driving.safemetadrive.unified import UnifiedDrivingSystem
        return UnifiedDrivingSystem(observation=observation, **parameters)
    if name == "lidar_option_reserve":
        from driving.safemetadrive.lidar_driver import LidarOptionReserveDriver
        return LidarOptionReserveDriver(observation=observation, **parameters)
    if name == "native_learned":
        options = dict(parameters)
        options["policy_seed"] = options.get("policy_seed", 0)+episode_seed
        return NativeLearnedDriver(observation=observation, **options)
    if name == "lane_caution_rule":
        return LaneCautionDriver(vehicle, **parameters)
    if name == "idm":
        from metadrive.policy.idm_policy import IDMPolicy
        policy = IDMPolicy(vehicle, random_seed=0)
        for key, value in parameters.items():
            setattr(policy, key, value)
        return policy
    if name == "upstream_expert":
        from metadrive.examples.ppo_expert.numpy_expert import expert

        class Expert:
            def act(self):
                return expert(vehicle, deterministic=True)

        return Expert()
    raise ValueError(f"Unknown policy: {name}")
