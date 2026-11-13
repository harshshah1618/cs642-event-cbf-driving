"""Physical MetaDrive plant for the shared three-layer controller.

Native scenario classes generate seeded initial conditions only. All subsequent
motion and collisions are computed by MetaDrive/Bullet, including scripted
traffic. The supported road is straight, with two 4 m lanes; curved-road support
requires a different observation and barrier model and is deliberately rejected.
MetaDrive imports are lazy so the base package works without its optional extra.
"""

from __future__ import annotations

import copy
import io
import math
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from driving import DT
from driving.config import ScenarioConfig
from driving.env_base import detached_snapshot, validate_action
from driving.env_types import ACTION_HIGH, ACTION_LOW, VehicleState, WorldSnapshot
from driving.errors import ConfigError, SafetyError, ValidationError


class AccelerationTracker:
    """Track physical acceleration using steering and wheel forces.

    Feedforward uses the selected vehicle's mass, wheelbase and force limits.
    A bicycle-model feedforward tracks requested lateral acceleration. This is
    an approximate tracking controller, not an identity action conversion or a
    safety certificate. Its achieved acceleration and saturation are logged.
    """

    def __init__(self):
        self.previous_velocity: Optional[np.ndarray] = None
        self.previous_request: Optional[float] = None
        self.disturbance = 0.0

    def command(self, action, vehicle, state: VehicleState, *, feedback: bool = True):
        ax, ay = validate_action(action)
        p = vehicle.get_dynamics_parameters()
        mass = float(p["mass"])
        engine_force = 4.0 * float(p["max_engine_force"])
        # Bullet's wheel brake value limits per-step impulse, whereas engine
        # force is a force. Convert the brake impulse to force using physics dt.
        brake_force = 4.0 * float(p["max_brake_force"]) / DT
        steer_limit = math.radians(float(p["max_steering"]))
        wheelbase = float(vehicle.FRONT_WHEELBASE + vehicle.REAR_WHEELBASE)
        if not all(
            math.isfinite(v) and v > 0
            for v in (mass, engine_force, brake_force, steer_limit, wheelbase)
        ):
            raise SafetyError("MetaDrive vehicle has invalid dynamics parameters")
        heading = float(vehicle.heading_theta)
        if not math.isfinite(heading) or not all(math.isfinite(v) for v in (state.vx, state.vy)):
            raise SafetyError("MetaDrive vehicle has a non-finite heading or velocity")
        c, s = math.cos(heading), math.sin(heading)
        velocity = np.array([state.vx, state.vy])
        measured = (
            np.zeros(2)
            if self.previous_velocity is None
            else (velocity - self.previous_velocity) / DT
        )
        self.previous_velocity = velocity
        # The skill/shield use road-frame acceleration. Wheel force and
        # curvature act in the body frame. Braking/accelerating while turning
        # couples both axes, so convert the complete acceleration vector.
        forward = float(ax) * c + float(ay) * s
        lateral = float(ay) * c - float(ax) * s
        measured_forward = float(measured[0]) * c + float(measured[1]) * s
        # Slowly compensate drag using velocity tracking rather than noisy
        # acceleration differentiation. Disturbance is intentionally bounded.
        if feedback and self.previous_request is not None:
            error = float(self.previous_request) - measured_forward
            self.disturbance = float(np.clip(self.disturbance + 0.03 * error, -1.0, 1.0))
        self.previous_request = forward
        effort_ax = forward + self.disturbance
        if ax <= -6.5:
            # The emergency path must not be softened by a tracking correction.
            effort_ax = min(forward, float(ax))
        throttle = mass * effort_ax / (engine_force if effort_ax >= 0 else brake_force)
        # Do not add another ay-derived heading correction here: doing so
        # applies the same acceleration twice and amplifies it with speed.
        steer = math.atan(wheelbase * lateral / max(float(velocity @ velocity), 1.0)) / steer_limit
        raw = np.array([steer, throttle], dtype=float)
        command = np.clip(raw, -1.0, 1.0)
        return command, {
            "command": command.tolist(),
            "saturated": bool(np.any(np.abs(raw) > 1.0)),
            "measured_ax": float(measured[0]),
        }


def _environment_class():
    """Create an upstream environment with lifecycle-managed physical traffic."""
    try:
        from metadrive import MetaDriveEnv
        from metadrive.manager.base_manager import BaseManager
    except ImportError as exc:
        raise ConfigError("MetaDrive is unavailable; install the pinned [metadrive] extra") from exc

    class ScenarioTrafficManager(BaseManager):
        def before_reset(self):
            super().before_reset()
            self.entries = []

        def before_step(self, *args, **kwargs):
            for vehicle, tracker, adapter, target_y, target_speed in self.entries:
                state = adapter._vehicle_state(vehicle)
                # Scripted lane/speed tracking; no state resets or teleporting.
                ax = float(np.clip(1.5 * (target_speed - state.vx), -5.0, 2.0))
                # Following traffic reacts to vehicles ahead in its corridor.
                for other in adapter._all_vehicles():
                    if other is vehicle:
                        continue
                    obs = adapter._vehicle_state(other)
                    gap = obs.x - state.x - 0.5 * (state.length + obs.length)
                    if gap > 0 and abs(obs.y - state.y) < 1.8:
                        ax = min(ax, (obs.vx - state.vx + 0.5 * (gap - 5 - state.vx)) / 1.5)
                ay = float(np.clip(1.5 * (target_y - state.y) - 2.5 * state.vy, -3.0, 3.0))
                command, _ = tracker.command([ax, ay], vehicle, state)
                vehicle.before_step(command)
            return {}

        def after_step(self, *args, **kwargs):
            for vehicle, *_rest in self.entries:
                vehicle.after_step()
            return {}

    class StackMetaDriveEnv(MetaDriveEnv):
        def setup_engine(self):
            super().setup_engine()
            self.engine.register_manager("stack_traffic", ScenarioTrafficManager())

        def _is_out_of_road(self, vehicle):
            # Straight-lane geometry is authoritative here. Upstream's vertical
            # ray localization can miss the thin road mesh at segment seams;
            # that should not end an otherwise on-road episode. Collision
            # flags remain upstream physical contacts and are never suppressed.
            for lane in vehicle.navigation.current_ref_lanes:
                longitudinal, lateral = lane.local_coordinates(vehicle.position)
                if -0.5 <= longitudinal <= lane.length + 0.5 and abs(lateral) <= lane.width / 2:
                    return False
            return True

    return StackMetaDriveEnv


class MetaDriveAdapter:
    """DrivingEnv contract using actual MetaDrive vehicle dynamics at 50 Hz."""

    def __init__(self, config: ScenarioConfig):
        self.config = copy.deepcopy(config)
        self._scenario_spec = copy.deepcopy(config)
        self.config.validate()
        self.scenario = self.SCENARIO_NAME = config.name
        self.route_length = float(
            config.route_length or (200 if config.name == "two_trap" else 220)
        )
        self.duration = float(config.duration or 20.0)
        self.action_space = spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.env: Any = None
        self._closed = False
        self._step_i = 0
        self._done = False
        self._crashed = False
        self._last_info: dict[str, Any] = {}
        self._origin = np.zeros(2)
        self._params: dict[str, Any] = {}
        self._tracker = AccelerationTracker()
        self._last_velocity = np.zeros(2)
        self._acceleration = np.zeros(2)
        self._cut_in_spawned = False
        self._empty_road = self.config.params.pop("empty_road", False)
        self._initial_y = self.config.params.pop("initial_y", None)
        self._initial_vy = float(self.config.params.pop("initial_vy", 0.0))
        traffic = self.config.params.pop("initial_traffic", None)
        self._initial_traffic = None
        if traffic is not None:
            from driving.env_base import validate_vehicle

            if not isinstance(traffic, list) or len(traffic) > 32:
                raise ConfigError("initial_traffic must be a list of at most 32 vehicles")
            self._initial_traffic = []
            for index, entry in enumerate(traffic):
                if not isinstance(entry, dict):
                    raise ConfigError("initial_traffic entries must be vehicle mappings")
                try:
                    state = VehicleState(**entry)
                    validate_vehicle(state, name=f"initial_traffic[{index}]")
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"invalid initial traffic: {exc}") from exc
                self._initial_traffic.append(state)
        self._events = self.config.params.pop("traffic_events", [])
        self._applied_events: list[int] = []
        if not isinstance(self._empty_road, bool):
            raise ConfigError("empty_road must be boolean")
        if self._initial_y is not None:
            self._initial_y = float(self._initial_y)
            if not math.isfinite(self._initial_y) or not -0.5 <= self._initial_y <= 4.5:
                raise ConfigError("initial_y must be finite and inside the two driving lanes")
        if not math.isfinite(self._initial_vy) or abs(self._initial_vy) > 1:
            raise ConfigError("initial_vy must be finite and in [-1, 1]")
        if not isinstance(self._events, list) or len(self._events) > 64:
            raise ConfigError("traffic_events must be a list of at most 64 events")
        for event in self._events:
            if (not isinstance(event, dict) or set(event)-{"at", "vehicle_index", "target_speed", "target_y"}
                    or "at" not in event or "vehicle_index" not in event
                    or not ({"target_speed", "target_y"} & set(event))):
                raise ConfigError("invalid traffic event fields")
            if not isinstance(event["vehicle_index"], int) or isinstance(event["vehicle_index"], bool) or event["vehicle_index"] < 0:
                raise ConfigError("traffic event vehicle_index must be a non-negative integer")
            for key in ("at", "target_speed", "target_y"):
                if key in event and (not isinstance(event[key], (int, float)) or isinstance(event[key], bool) or not math.isfinite(float(event[key]))):
                    raise ConfigError("traffic event values must be finite numbers")
            if not 0 <= float(event["at"]) < self.duration:
                raise ConfigError("traffic event time must lie inside the episode")
            if "target_speed" in event and not 0 <= float(event["target_speed"]) <= 35:
                raise ConfigError("traffic target speed must be in [0, 35]")
            if "target_y" in event and float(event["target_y"]) not in (0.0, 4.0):
                raise ConfigError("traffic target_y must select a supported driving lane")
        self._render = self.config.params.pop("render", False)
        self._rgb_camera = self.config.params.pop("rgb_camera", False)
        if not isinstance(self._render, bool):
            raise ConfigError("MetaDrive params.render must be a boolean")
        if not isinstance(self._rgb_camera, bool):
            raise ConfigError("MetaDrive params.rgb_camera must be a boolean")
        # Reuse scenario sampling/validation, never its dynamics or collision code.
        from driving.env_registry import make_env

        source_config = copy.deepcopy(self.config)
        source_config.simulator = "native"
        self._source = make_env(source_config.name, source_config)

    def _require_reset(self):
        if self.env is None or self._closed:
            raise ValidationError("MetaDrive environment must be reset before use")

    def _vehicle_state(self, vehicle) -> VehicleState:
        pos = np.asarray(vehicle.position, dtype=float) - self._origin
        velocity = np.asarray(vehicle.velocity, dtype=float)
        return VehicleState(
            x=float(pos[0]),
            y=float(pos[1]),
            vx=float(velocity[0]),
            vy=float(velocity[1]),
            ax=float(self._acceleration[0]) if vehicle is self.env.vehicle else 0.0,
            ay=float(self._acceleration[1]) if vehicle is self.env.vehicle else 0.0,
            length=float(vehicle.LENGTH),
            width=float(vehicle.WIDTH),
            lane_id=int(np.clip(round(float(pos[1]) / 4.0), 0, 1)),
        )

    def _all_vehicles(self):
        return [self.env.vehicle] + [e[0] for e in self.env.engine.stack_traffic.entries]

    def _spawn(self, state: VehicleState, *, target_y=None):
        from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle

        cfg = copy.deepcopy(dict(self.env.config["vehicle_config"]))
        cfg.update(
            {
                "spawn_position_heading": [(self._origin + [state.x, state.y]).tolist(), 0.0],
                "spawn_velocity": [state.vx, state.vy],
                "spawn_velocity_car_frame": False,
                "show_navi_mark": False,
                "show_dest_mark": False,
                # These cars are controlled by our lane/speed manager and do
                # not need upstream route planning/ray-based spawn localization.
                "navigation_module": None,
            }
        )
        manager = self.env.engine.stack_traffic
        vehicle = manager.spawn_object(TrafficDefaultVehicle, vehicle_config=cfg)
        manager.entries.append(
            (
                vehicle,
                AccelerationTracker(),
                self,
                float(state.y if target_y is None else target_y),
                float(state.vx),
            )
        )

    def reset(self, *, seed=None, options=None):
        if options:
            raise ConfigError("MetaDrive reset options are unsupported; use ScenarioConfig")
        seed = self.config.seed if seed is None else int(seed)
        _, source_info = self._source.reset(seed=seed)
        initial = self._source.snapshot()
        if self.env is None:
            cls = _environment_class()
            env_config = {
                "use_render": self._render,
                "map_config": {
                    "type": "block_sequence",
                    "config": "S",
                    "lane_width": 4.0,
                    "lane_num": 3 if self.scenario == "merge" else 2,
                    "exit_length": self.route_length + 150.0,
                },
                "traffic_density": 0.0,
                "random_spawn_lane_index": False,
                "start_seed": 0,
                "num_scenarios": 1,
                "agent_configs": {
                    "default_agent": {
                        "spawn_lane_index": (">", ">>", 1),
                        "spawn_longitude": 5.0,
                        "spawn_velocity": [initial.ego.vx, 0.0],
                    }
                },
                "vehicle_config": {"enable_reverse": False},
                "decision_repeat": 1,
                "physics_world_step_size": DT,
                "horizon": int(math.ceil(self.duration / DT)),
                "log_level": 50,
                "show_terrain": False,
                "show_skybox": False,
                "preload_models": False,
                "on_continuous_line_done": False,
            }
            if self._rgb_camera:
                from metadrive.component.sensors.rgb_camera import RGBCamera

                # image_observation keeps the cameras from being stripped.
                # The adapter still feeds the controller state, not pixels.
                env_config["image_observation"] = True
                env_config["norm_pixel"] = False
                env_config["image_on_cuda"] = False
                env_config["show_terrain"] = True
                env_config["show_skybox"] = True
                env_config["vehicle_config"]["image_source"] = "rgb_front"
                env_config["sensors"] = {
                    "rgb_front": (RGBCamera, 640, 360),
                    "rgb_chase": (RGBCamera, 640, 360),
                }
            self.env = cls(env_config)
        self._closed = False
        try:
            self.env.reset(seed=0)
            ego = self.env.vehicle
            ego.set_velocity([initial.ego.vx, 0.0])  # initial condition only
            self._origin = np.asarray(ego.position, dtype=float) - [initial.ego.x, initial.ego.y]
            if self._initial_y is not None:
                ego.set_position(self._origin + [initial.ego.x, self._initial_y])
            ego.set_velocity([initial.ego.vx, self._initial_vy])
            self._step_i = 0
            self._done = self._crashed = self._cut_in_spawned = False
            self._applied_events = []
            self._tracker = AccelerationTracker()
            self._last_velocity = np.asarray(ego.velocity, dtype=float)
            self._acceleration = np.zeros(2)
            self._params = copy.deepcopy(source_info.get("params", {}))
            self._params.update(
                {
                    "simulator": "metadrive",
                    "scenario_seed": seed,
                    "physics_dt": DT,
                    "lane_width": 4.0,
                    "map": "straight",
                    "lane_num": 3 if self.scenario == "merge" else 2,
                }
            )
            self._params["scenario_config"] = asdict(self._scenario_spec)
            traffic = initial.others if self._initial_traffic is None else self._initial_traffic
            for state in ([] if self._empty_road else traffic):
                self._spawn(state, target_y=0.0 if state.y < 0 else state.y)
            self._last_info = {}
            return self.snapshot().skill_obs(), self._info()
        except Exception:
            self.close()
            raise

    def snapshot(self) -> WorldSnapshot:
        self._require_reset()
        return detached_snapshot(
            t=self._step_i * DT,
            ego=self._vehicle_state(self.env.vehicle),
            others=[self._vehicle_state(e[0]) for e in self.env.engine.stack_traffic.entries],
            route_length=self.route_length,
            done=self._done,
            crashed=self._crashed,
            info={
                "scenario": self.scenario,
                "simulator": "metadrive",
                "params": self._params,
                "plant": copy.deepcopy(self._last_info),
                "road_bounds": [-6.0 if self.scenario == "merge" else -2.0, 6.0],
                "heading": float(self.env.vehicle.heading_theta),
            },
        )

    def capture_router_camera(self):
        """Capture a detached front-camera PNG for an explicit VLM request."""
        from PIL import Image

        from driving.router_base import CameraObservation

        self._require_reset()
        if not self._rgb_camera:
            raise ConfigError("VLM requests require scenario.params.rgb_camera=true")
        sensor = self.env.engine.get_sensor("rgb_front")
        pixels = np.asarray(sensor.perceive(
            False, self.env.vehicle.origin, (0.0, 0.8, 1.5), (0.0, -5.0, 0.0)
        ))
        # MetaDrive's CPU screenshot path exposes BGR channels.
        rgb = np.ascontiguousarray(pixels[..., :3][..., ::-1], dtype=np.uint8)
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        state = self._vehicle_state(self.env.vehicle)
        return CameraObservation(
            image_bytes=stream.getvalue(), mime_type="image/png",
            captured_t=self._step_i * DT,
            lane="LEFT" if state.y >= 2.0 else "RIGHT",
            speed_mps=float(state.vx), width=int(rgb.shape[1]), height=int(rgb.shape[0]),
        )

    def _info(self):
        snap = self.snapshot()
        return {
            "snapshot": snap,
            "scenario": self.scenario,
            "params": copy.deepcopy(self._params),
            "crashed": self._crashed,
            "route_pct": float(np.clip(100 * snap.ego.x / self.route_length, 0, 100)),
            "end_reason": "out_of_road" if self._last_info.get("out_of_road") else None,
            "plant": copy.deepcopy(self._last_info),
        }

    def step(self, action):
        self._require_reset()
        if self._done:
            raise ValidationError("cannot step a finished MetaDrive episode; reset it")
        requested = validate_action(action)
        for index, event in enumerate(self._events):
            if index not in self._applied_events and self._step_i * DT >= float(event["at"]) - 1e-12:
                entries = self.env.engine.stack_traffic.entries
                vehicle_index = event["vehicle_index"]
                if vehicle_index >= len(entries):
                    raise ConfigError("traffic event references a vehicle that has not spawned")
                vehicle, tracker, adapter, target_y, target_speed = entries[vehicle_index]
                entries[vehicle_index] = (vehicle, tracker, adapter,
                    float(event.get("target_y", target_y)), float(event.get("target_speed", target_speed)))
                self._applied_events.append(index)
        before = self.snapshot()
        command, telemetry = self._tracker.command(requested, self.env.vehicle, before.ego)
        _, reward, terminated, truncated, info = self.env.step(command)
        self._step_i += 1
        velocity = np.asarray(self.env.vehicle.velocity, dtype=np.float64)
        self._acceleration = (velocity - self._last_velocity) / DT
        self._last_velocity = velocity.copy()
        self._crashed = bool(
            any(
                info.get(k, False)
                for k in (
                    "crash_vehicle",
                    "crash_object",
                    "crash_building",
                    "crash_human",
                    "crash_sidewalk",
                )
            )
        )
        self._last_info = {
            **telemetry,
            "requested_acceleration": requested.tolist(),
            "achieved_acceleration": self._acceleration.tolist(),
            "out_of_road": bool(info.get("out_of_road", False)),
            "upstream_on_lane": bool(self.env.vehicle.on_lane),
            "traffic_events_applied": list(self._applied_events),
        }
        current = self._vehicle_state(self.env.vehicle)
        if self.scenario == "two_trap" and not self._cut_in_spawned and not self._empty_road:
            if current.x >= self._params["cut_in_trigger_x"]:
                self._spawn(
                    VehicleState(
                        current.x + self._params["cut_in_gap"], 4.0, self._params["cut_in_speed"]
                    )
                )
                self._cut_in_spawned = True
        terminated = bool(terminated or self._crashed or current.x >= self.route_length)
        truncated = bool(truncated or self._step_i * DT >= self.duration - 1e-12)
        self._done = terminated or truncated
        return self.snapshot().skill_obs(), float(reward), terminated, truncated, self._info()

    def close(self):
        if self.env is not None and not self._closed:
            self.env.close()
        self.env = None
        self._closed = True
        self._source.close()
