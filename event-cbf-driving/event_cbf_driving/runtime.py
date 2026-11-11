"""Episode records, strict validation, and deterministic open-loop replay.

The evaluation runner can write one JSON document per episode.  Those
documents contain the full :class:`~event_cbf_driving.stack.result.RolloutResult`
including per-tick actions.  This module turns them into a first-class,
auditable artifact:

* :func:`load_episode` validates the schema and rejects non-finite traces.
* :func:`write_episode` writes atomically with provenance metadata.
* :func:`replay_open_loop` resets the recorded scenario with the recorded seed
  and reapplies the action log.  It is an artifact audit, not a second
  controller: if replay diverges, either the world is non-deterministic or the
  recording is corrupt.

Replay is intentionally *open loop*.  Re-running the router would test the
router, not the recording; the action log is the exact sequence that reached
the plant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

import numpy as np

from event_cbf_driving import DT
from event_cbf_driving.config import ScenarioConfig
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.errors import ArtifactError, ValidationError
from event_cbf_driving.eval.metrics import json_safe
from event_cbf_driving.io_utils import atomic_write_json, read_json, utc_now
from event_cbf_driving.stack.result import RolloutResult

EPISODE_SCHEMA_VERSION = 1
_REQUIRED_KEYS = (
    "mode",
    "scenario",
    "crashed",
    "route_pct",
    "avg_speed",
    "router_calls",
    "peak_lateral_jerk",
    "peak_sigma",
    "freeze",
    "x",
    "y",
    "vx",
    "end_time",
    "xs",
    "ys",
    "skills",
)
_METADATA_KEYS = {
    "schema_version",
    "recorded",
    "recorded_at",
    "scenario_config",
    "source",
    "repeat",
}
_RESULT_KEYS = set(RolloutResult.__dataclass_fields__)
_KNOWN_KEYS = _RESULT_KEYS | _METADATA_KEYS

EpisodeLike = Union["Episode", Mapping[str, Any], str, Path]


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _check_finite(values: Any, name: str, problems: List[str]) -> List[float]:
    if not isinstance(values, (list, tuple)):
        problems.append(f"{name} must be a list")
        return []
    out: List[float] = []
    for index, value in enumerate(values):
        number = _number(value)
        if number is None:
            problems.append(f"{name}[{index}] is not a finite number")
        else:
            out.append(number)
    return out


def validate_episode(payload: Mapping[str, Any]) -> List[str]:
    """Return human-readable schema problems (empty means valid)."""

    if not isinstance(payload, Mapping):
        return [f"episode must be a JSON object, got {type(payload).__name__}"]
    problems: List[str] = []
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        problems.append(f"missing required keys: {missing}")
    unknown = sorted(set(payload) - _KNOWN_KEYS)
    if unknown:
        problems.append(f"unknown keys: {unknown}")

    mode = payload.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        problems.append("mode must be a non-empty string")
    scenario = payload.get("scenario")
    if not isinstance(scenario, str) or not scenario.strip():
        problems.append("scenario must be a non-empty string")

    xs = _check_finite(payload.get("xs", []), "xs", problems)
    ys = _check_finite(payload.get("ys", []), "ys", problems)
    skills = payload.get("skills", [])
    if not isinstance(skills, (list, tuple)):
        problems.append("skills must be a list")
        skills = []
    if not (len(xs) == len(ys) == len(skills)):
        problems.append(
            "xs, ys, and skills must have equal length "
            f"(got {len(xs)}, {len(ys)}, {len(skills)})"
        )

    actions_present = "actions" in payload
    if actions_present:
        actions = payload.get("actions") or []
        if not isinstance(actions, (list, tuple)):
            problems.append("actions must be a list")
        elif actions:
            # An empty action list on a non-empty trace is a legacy record
            # without an action log, not a corrupt modern record.
            if xs and len(actions) != len(xs):
                problems.append(
                    f"actions length {len(actions)} does not match trace length {len(xs)}"
                )
            for index, action in enumerate(actions):
                if not isinstance(action, (list, tuple)) or len(action) != 2:
                    problems.append(f"actions[{index}] must be [ax, ay]")
                    continue
                if any(_number(component) is None for component in action):
                    problems.append(f"actions[{index}] contains a non-finite value")

    interactions = payload.get("interactions", [])
    if not isinstance(interactions, (list, tuple)):
        problems.append("interactions must be a list")
    elif interactions:
        if len(interactions) != len(xs):
            problems.append("interactions and recorded states must have equal length")
        previous_t = -1.0
        for index, row in enumerate(interactions):
            prefix = f"interactions[{index}]"
            if not isinstance(row, Mapping):
                problems.append(f"{prefix} must be an object")
                continue
            t = _number(row.get("t"))
            if t is None or t <= previous_t:
                problems.append(f"{prefix}.t must be finite and strictly increasing")
            else:
                previous_t = t
            for field in ("requested_action", "shield_action", "applied_action"):
                value = row.get(field)
                if (not isinstance(value, (list, tuple)) or len(value) != 2
                        or any(_number(v) is None for v in value)):
                    problems.append(f"{prefix}.{field} must be a finite two-component action")

    route = _number(payload.get("route_pct"))
    if route is None:
        problems.append("route_pct must be finite")
    elif route < -1e-6 or route > 100.0 + 1e-6:
        problems.append(f"route_pct {route} is outside [0, 100]")

    avg_speed = _number(payload.get("avg_speed"))
    if avg_speed is None:
        problems.append("avg_speed must be finite")
    elif avg_speed < -1e-6:
        problems.append(f"avg_speed {avg_speed} must be non-negative")

    end_time = _number(payload.get("end_time"))
    if end_time is None:
        problems.append("end_time must be finite")
    elif end_time < 0.0:
        problems.append(f"end_time {end_time} must be non-negative")

    for key in ("x", "y", "vx", "peak_lateral_jerk", "peak_sigma"):
        if _number(payload.get(key)) is None:
            problems.append(f"{key} must be finite")

    if payload.get("min_barrier_h") is not None and _number(payload.get("min_barrier_h")) is None:
        problems.append("min_barrier_h must be finite or null")

    calls = payload.get("calls", [])
    if calls is not None and not isinstance(calls, (list, tuple)):
        problems.append("calls must be a list")

    crashed = payload.get("crashed")
    if not isinstance(crashed, bool):
        problems.append("crashed must be a boolean")
    crash_time = payload.get("crash_time")
    if crash_time is not None and _number(crash_time) is None:
        problems.append("crash_time must be finite or null")

    config = payload.get("scenario_config")
    if config is not None:
        if not isinstance(config, Mapping):
            problems.append("scenario_config must be an object or null")
        else:
            try:
                ScenarioConfig(**dict(config)).validate("scenario_config")
            except Exception as exc:
                problems.append(f"scenario_config is invalid: {exc}")
    return problems


@dataclass
class TrajectorySample:
    """One recorded pre-step state.

    ``t`` is derived from the 50 Hz clock. ``vx`` is a finite-difference
    estimate from adjacent recorded positions; it is ``None`` for a one-tick
    trace.
    """

    index: int
    t: float
    x: float
    y: float
    skill: str
    vx: Optional[float] = None


def _scenario_for(result: RolloutResult, config: Optional[Mapping[str, Any]]) -> ScenarioConfig:
    if not config and isinstance(result.params.get("scenario_config"), Mapping):
        config = result.params["scenario_config"]
    if config:
        scenario = ScenarioConfig(**dict(config))
    else:
        scenario = ScenarioConfig(
            name=result.scenario or "two_trap",
            seed=int(result.seed or 0),
            params=dict(result.params or {}),
        )
    scenario.seed = int(result.seed if result.seed is not None else scenario.seed)
    scenario.validate("scenario")
    return scenario


@dataclass
class Episode:
    """A validated episode record plus the scenario needed to replay it."""

    result: RolloutResult
    scenario: ScenarioConfig
    repeat: int = 0
    recorded_at: Optional[str] = None
    schema_version: int = EPISODE_SCHEMA_VERSION
    source: Optional[Path] = None

    @property
    def ticks(self) -> int:
        return len(self.result.xs)

    @property
    def actions(self) -> List[List[float]]:
        return self.result.actions

    def samples(self) -> Iterator[TrajectorySample]:
        xs = self.result.xs
        for index, (x, y, skill) in enumerate(
            zip(xs, self.result.ys, self.result.skills, strict=True)
        ):
            if index + 1 < len(xs):
                vx = (float(xs[index + 1]) - float(x)) / DT
            elif index > 0:
                vx = (float(x) - float(xs[index - 1])) / DT
            else:
                vx = None
            yield TrajectorySample(
                index=index, t=index * DT, x=x, y=y, skill=skill, vx=vx
            )

    def to_payload(self) -> Dict[str, Any]:
        payload = json_safe(self.result.to_dict())
        payload["scenario"] = self.result.scenario or self.scenario.name
        payload["schema_version"] = int(self.schema_version)
        payload["recorded"] = True
        payload["recorded_at"] = self.recorded_at or utc_now()
        payload["repeat"] = int(self.repeat)
        payload["scenario_config"] = json_safe(self.scenario.__dict__)
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, source: Optional[Path] = None
    ) -> "Episode":
        problems = validate_episode(payload)
        if problems:
            raise ValidationError("invalid episode: " + "; ".join(problems))
        result_payload = {key: payload[key] for key in _RESULT_KEYS if key in payload}
        result = RolloutResult(**result_payload)
        scenario = _scenario_for(result, payload.get("scenario_config"))
        return cls(
            result=result,
            scenario=scenario,
            repeat=int(payload.get("repeat", 0) or 0),
            recorded_at=payload.get("recorded_at"),
            schema_version=int(payload.get("schema_version", EPISODE_SCHEMA_VERSION) or 0),
            source=source,
        )


def write_episode(
    path: Union[str, Path],
    result: RolloutResult,
    *,
    scenario: Optional[ScenarioConfig] = None,
    repeat: int = 0,
    recorded_at: Optional[str] = None,
) -> Path:
    """Atomically write ``result`` as a replayable episode document."""

    if not isinstance(result, RolloutResult):
        raise ValidationError(
            f"write_episode expected RolloutResult, got {type(result).__name__}"
        )
    scenario_cfg = scenario if scenario is not None else _scenario_for(result, None)
    episode = Episode(
        result=result,
        scenario=scenario_cfg,
        repeat=int(repeat),
        recorded_at=recorded_at or utc_now(),
    )
    target = Path(path)
    atomic_write_json(target, episode.to_payload())
    return target


def load_episode(path: Union[str, Path]) -> Episode:
    """Load and strictly validate an episode JSON document."""

    target = Path(path)
    try:
        payload = read_json(target)
    except ArtifactError:
        raise
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"{target}: expected a JSON object")
    return Episode.from_payload(payload, source=target)


def _coerce_episode(value: EpisodeLike) -> Episode:
    if isinstance(value, Episode):
        return value
    if isinstance(value, (str, Path)):
        return load_episode(value)
    if isinstance(value, Mapping):
        return Episode.from_payload(value)
    raise ValidationError(f"unsupported episode type {type(value).__name__}")


@dataclass
class ReplayReport:
    ok: bool
    ticks: int
    expected_ticks: int
    expected_x: float
    replayed_x: float
    expected_y: float
    replayed_y: float
    final_state_error: float
    expected_crashed: bool
    replayed_crashed: bool
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(self.__dict__)


def replay_open_loop(
    episode: EpisodeLike,
    *,
    env: Any = None,
    scenario: Optional[ScenarioConfig] = None,
    tolerance: float = 1e-6,
) -> ReplayReport:
    """Reapply the recorded action log and compare final state.

    The environment is reset with the recorded seed, then every recorded
    action is applied in order. A successful replay proves that the artifact
    is sufficient to reproduce the trajectory on this build. Divergence is
    reported, never silently tolerated.
    """

    recorded = _coerce_episode(episode)
    if not recorded.actions:
        raise ValidationError(
            f"episode {recorded.source or recorded.result.scenario!r} has no action log; "
            "record it with record_episodes=True on the current version"
        )

    created = env is None
    active_env = env
    if active_env is None:
        scenario_cfg = scenario if scenario is not None else recorded.scenario
        active_env = make_env(scenario_cfg.name, scenario_cfg)

    errors: List[str] = []
    ticks = 0
    crashed = False
    try:
        active_env.reset(seed=int(recorded.result.seed or 0))
        total = len(recorded.actions)
        for index, action in enumerate(recorded.actions):
            _obs, _reward, terminated, truncated, info = active_env.step(
                np.asarray(action, dtype=np.float32)
            )
            ticks += 1
            crashed = bool(info.get("crashed", False))
            if (terminated or truncated) and index < total - 1:
                errors.append(
                    f"environment terminated after {ticks} ticks but {total} actions were recorded"
                )
                break
        final = active_env.snapshot()
    finally:
        if created:
            close = getattr(active_env, "close", None)
            if callable(close):
                close()

    expected_x = float(recorded.result.x)
    expected_y = float(recorded.result.y)
    replayed_x = float(final.ego.x)
    replayed_y = float(final.ego.y)
    error = max(abs(replayed_x - expected_x), abs(replayed_y - expected_y))
    if error > float(tolerance):
        errors.append(
            f"final state differs by {error:.6g} m (tolerance {float(tolerance):.6g} m)"
        )
    if crashed != bool(recorded.result.crashed):
        errors.append(
            f"crash mismatch: replayed={crashed}, recorded={bool(recorded.result.crashed)}"
        )
    if ticks != len(recorded.actions):
        errors.append(f"replayed {ticks} of {len(recorded.actions)} actions")

    return ReplayReport(
        ok=not errors,
        ticks=ticks,
        expected_ticks=len(recorded.actions),
        expected_x=expected_x,
        replayed_x=replayed_x,
        expected_y=expected_y,
        replayed_y=replayed_y,
        final_state_error=float(error),
        expected_crashed=bool(recorded.result.crashed),
        replayed_crashed=crashed,
        errors=errors,
    )


__all__ = [
    "EPISODE_SCHEMA_VERSION",
    "Episode",
    "ReplayReport",
    "TrajectorySample",
    "load_episode",
    "replay_open_loop",
    "validate_episode",
    "write_episode",
]
