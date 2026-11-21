"""Place extra objects along the ego route in SafeMetaDrive and log ground truth.

Objects are spawned through a MetaDrive manager, so the engine removes them on
every reset like the built-in accident objects.
"""

import math
from dataclasses import asdict, dataclass

KINDS = ("pedestrian", "cyclist", "cone", "warning", "barrier")
# Kinds the clean training maps already contain (accident_prob=0.8 places them).
FAMILIAR_KINDS = ("cone", "warning", "barrier")


@dataclass(frozen=True)
class Placement:
    kind: str
    ahead: float              # metres along the route from the ego's spawn point
    where: str = "shoulder"   # "shoulder" (right of the rightmost lane) or "lane" (ego lane)
    offset: float | None = None  # metres beyond the lane edge ("shoulder", default 1.2)
                                 # or from the lane centre ("lane", default 0)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown kind {self.kind!r}")
        if self.where not in ("shoulder", "lane"):
            raise ValueError(f"Unknown placement {self.where!r}")
        if not self.ahead > 0:
            raise ValueError("Objects must be ahead of the ego")


def route_lanes(agent, lane_index=None):
    """Lanes along the navigation route, keeping one lane index (clamped per road)."""
    nav = agent.navigation
    graph = nav.map.road_network.graph
    current = agent.lane.index[-1] if lane_index is None else lane_index
    lanes = []
    for start, end in zip(nav.checkpoints[:-1], nav.checkpoints[1:], strict=True):
        road = graph[start][end]
        lanes.append(road[min(current, len(road) - 1)] if current >= 0 else road[-1])
    return lanes


def locate(agent, ahead, where="shoulder", offset=None):
    """Return (lane, longitudinal, lateral) for a point `ahead` metres along the route."""
    ego_lane = agent.lane
    start_s, _ = ego_lane.local_coordinates(agent.position)
    lanes = route_lanes(agent, -1 if where == "shoulder" else None)
    # Skip route lanes before the ego's current road.
    road = ego_lane.index[:2]
    names = [lane.index[:2] for lane in lanes]
    first = names.index(road) if road in names else 0
    remaining = ahead + start_s
    for lane in lanes[first:]:
        if remaining <= lane.length - 2.0:
            if where == "shoulder":
                lateral = lane.width_at(remaining) / 2 + (1.2 if offset is None else offset)
            else:
                lateral = 0.0 if offset is None else offset
            return lane, remaining, lateral
        remaining -= lane.length
    raise ValueError("Route is shorter than the requested distance")


def make_manager_class():
    from metadrive.component.static_object.traffic_object import (
        TrafficBarrier,
        TrafficCone,
        TrafficWarning,
    )
    from metadrive.component.traffic_participants.cyclist import Cyclist
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    from metadrive.manager.base_manager import BaseManager

    classes = dict(pedestrian=Pedestrian, cyclist=Cyclist, cone=TrafficCone,
                   warning=TrafficWarning, barrier=TrafficBarrier)

    class IrregularObjectManager(BaseManager):
        PRIORITY = 9

        def place(self, agent, placement: Placement):
            lane, s, lateral = locate(agent, placement.ahead, placement.where, placement.offset)
            position = lane.position(s, lateral)
            heading = lane.heading_theta_at(s)
            cls = classes[placement.kind]
            kwargs = dict(position=position, heading_theta=heading)
            if placement.kind in FAMILIAR_KINDS:
                kwargs.update(lane=lane, static=True)
            obj = self.spawn_object(cls, **kwargs)
            if placement.kind in ("pedestrian", "cyclist"):
                # Stand still and cannot be shoved by traffic; collisions still register.
                obj.set_velocity([math.cos(heading), math.sin(heading)], 0.0)
                obj.set_static(True)
            return obj

    return IrregularObjectManager


def make_env(config):
    """SafeMetaDriveEnv with the extra object manager registered."""
    from metadrive import SafeMetaDriveEnv

    manager_class = make_manager_class()

    class IrregularSafeMetaDriveEnv(SafeMetaDriveEnv):
        def setup_engine(self):
            super().setup_engine()
            self.engine.register_manager("irregular_manager", manager_class())

    return IrregularSafeMetaDriveEnv(config)


def relative(agent, obj, fov_degrees=65.0, max_range=120.0):
    """Ground truth for one object: forward/lateral offset and front-camera visibility."""
    dx = obj.position[0] - agent.position[0]
    dy = obj.position[1] - agent.position[1]
    heading = agent.heading_theta
    forward = dx * math.cos(heading) + dy * math.sin(heading)
    lateral = -dx * math.sin(heading) + dy * math.cos(heading)
    bearing = math.degrees(math.atan2(lateral, forward)) if forward > 0 else 180.0
    visible = forward > 0 and abs(bearing) < fov_degrees / 2 and math.hypot(dx, dy) < max_range
    return dict(forward=forward, lateral=lateral, bearing=bearing, visible=visible,
                distance=math.hypot(dx, dy))


def placement_record(placement: Placement):
    return asdict(placement)


def place(env, placement: Placement):
    """Spawn one object for the current episode; removed automatically on reset."""
    return env.engine.irregular_manager.place(env.agent, placement)
