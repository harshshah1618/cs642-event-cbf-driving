"""Typed, validated configuration for the whole stack.

Dataclasses with explicit validation and YAML inheritance.

"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import yaml

from driving.errors import ConfigError

T = TypeVar("T")

SKILL_NAMES = ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")
SCENARIO_NAMES = ("two_trap", "merge", "dense_highway")
EVAL_MODES = (
    "unshielded_skill",
    "static_cbf",
    "fixed_rate_vlm",
    "always_on_vlm",
    "proposed",
    "proposed_no_margin",
    "proposed_no_blend",
    "fixed_rate_shielded",
    "always_on_shielded",
)


class _Enum(str, Enum):
    """String enum that serializes as its value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class StackMode(_Enum):
    UNSHIELDED = "unshielded_skill"
    STATIC_CBF = "static_cbf"
    FIXED_VLM = "fixed_rate_vlm"
    ALWAYS_ON = "always_on_vlm"
    PROPOSED = "proposed"


def _fail(path: str, message: str) -> None:
    raise ConfigError(f"{path}: {message}")


def _finite(path: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        _fail(path, f"expected a number, got {value!r}")
    if not math.isfinite(out):
        _fail(path, f"expected a finite number, got {value!r}")
    return out


def _positive(path: str, value: Any) -> float:
    out = _finite(path, value)
    if out <= 0.0:
        _fail(path, f"must be > 0, got {out}")
    return out


def _non_negative(path: str, value: Any) -> float:
    out = _finite(path, value)
    if out < 0.0:
        _fail(path, f"must be >= 0, got {out}")
    return out


def _choice(path: str, value: Any, options: Sequence[str]) -> str:
    out = str(value)
    if out not in options:
        _fail(path, f"must be one of {list(options)}, got {out!r}")
    return out


def _bool(path: str, value: Any) -> bool:
    if not isinstance(value, bool):
        _fail(path, f"expected a boolean, got {value!r}")
    return value


@dataclass
class ShieldConfig:
    """Longitudinal safety-filter parameters.

    ``lateral_mode="lane"`` reproduces the original research behavior: only
    obstacles inside a fixed corridor around the ego's current lane are seen.
    ``lateral_mode="footprint"`` uses the actual vehicle widths plus a margin,
    and ``lookahead_s`` additionally considers where the ego will be laterally
    after that horizon. The production profile enables both.
    """

    standoff: float = 6.0
    headway_s: float = 1.1
    class_k: float = 2.5
    lane_half_width: float = 1.85
    gamma_sigma: float = 2.8
    sigma_cap: float = 10.0
    eta_latency: float = 1.0
    vlm_delay_s: float = 0.30
    ax_min: float = -7.5
    ax_max: float = 3.5
    enabled: bool = True
    cbf_constraint_enabled: bool = True
    inflate: bool = True
    pending_margin: bool = True
    finite_checks: bool = True
    fail_closed: bool = True
    lateral_mode: str = "lane"
    lateral_margin: float = 0.0
    lookahead_s: float = 0.0
    emergency_decel: float = -7.5
    min_gap_m: float = 0.0
    road_boundary_guard: bool = False
    boundary_gain: float = 1.5
    boundary_margin: float = 0.15
    braking_guard: bool = False
    braking_decel: float = 6.0
    braking_clearance: float = 2.0
    braking_response_s: float = 0.15
    intent_corridor: bool = False

    def validate(self, path: str = "shield") -> None:
        self.standoff = _non_negative(f"{path}.standoff", self.standoff)
        self.headway_s = _positive(f"{path}.headway_s", self.headway_s)
        self.class_k = _non_negative(f"{path}.class_k", self.class_k)
        self.lane_half_width = _positive(f"{path}.lane_half_width", self.lane_half_width)
        self.gamma_sigma = _non_negative(f"{path}.gamma_sigma", self.gamma_sigma)
        self.sigma_cap = _non_negative(f"{path}.sigma_cap", self.sigma_cap)
        self.eta_latency = _non_negative(f"{path}.eta_latency", self.eta_latency)
        self.vlm_delay_s = _non_negative(f"{path}.vlm_delay_s", self.vlm_delay_s)
        self.ax_min = _finite(f"{path}.ax_min", self.ax_min)
        self.ax_max = _finite(f"{path}.ax_max", self.ax_max)
        self.emergency_decel = _finite(f"{path}.emergency_decel", self.emergency_decel)
        self.lateral_margin = _non_negative(f"{path}.lateral_margin", self.lateral_margin)
        self.lookahead_s = _non_negative(f"{path}.lookahead_s", self.lookahead_s)
        self.min_gap_m = _non_negative(f"{path}.min_gap_m", self.min_gap_m)
        _bool(f"{path}.road_boundary_guard", self.road_boundary_guard)
        _bool(f"{path}.cbf_constraint_enabled", self.cbf_constraint_enabled)
        self.boundary_gain = _positive(f"{path}.boundary_gain", self.boundary_gain)
        self.boundary_margin = _non_negative(f"{path}.boundary_margin", self.boundary_margin)
        _bool(f"{path}.braking_guard", self.braking_guard)
        _bool(f"{path}.intent_corridor", self.intent_corridor)
        self.braking_decel = _positive(f"{path}.braking_decel", self.braking_decel)
        self.braking_clearance = _non_negative(f"{path}.braking_clearance", self.braking_clearance)
        self.braking_response_s = _non_negative(
            f"{path}.braking_response_s", self.braking_response_s
        )
        if self.braking_decel > abs(self.ax_min):
            _fail(path, "braking_decel must fit inside the actuator deceleration limit")
        self.lateral_mode = _choice(
            f"{path}.lateral_mode", self.lateral_mode, ("lane", "footprint")
        )
        if self.ax_min >= self.ax_max:
            _fail(f"{path}.ax_min", "must be strictly less than ax_max")
        if self.emergency_decel > self.ax_min:
            _fail(f"{path}.emergency_decel", "must be <= ax_min")
        if not self.enabled and self.fail_closed:
            # An explicitly disabled shield is valid for ablations, but the
            # caller must acknowledge that it is a deliberate unsafe mode.
            self.fail_closed = False


@dataclass
class RouterConfig:
    backend: str = "scripted"
    allow_live: bool = False
    latency_s: float = 0.30
    latency_jitter_s: float = 0.0
    latency_schedule_s: List[float] = field(default_factory=list)
    timeout_s: float = 30.0
    max_retries: int = 2
    backoff_s: float = 0.25
    max_consecutive_failures: int = 5
    model: str = "camera-router"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key_env_fallback: str = "EVENT_CBF_API_KEY"
    seed: int = 0

    def validate(self, path: str = "router") -> None:
        self.backend = _choice(f"{path}.backend", self.backend, ("scripted", "vlm"))
        self.allow_live = _bool(f"{path}.allow_live", self.allow_live)
        self.latency_s = _non_negative(f"{path}.latency_s", self.latency_s)
        self.latency_jitter_s = _non_negative(f"{path}.latency_jitter_s", self.latency_jitter_s)
        if not isinstance(self.latency_schedule_s, list) or len(self.latency_schedule_s) > 10000:
            _fail(path, "latency_schedule_s must be a bounded list")
        self.latency_schedule_s = [
            _non_negative(f"{path}.latency_schedule_s", v) for v in self.latency_schedule_s
        ]
        if self.latency_schedule_s and self.latency_jitter_s:
            _fail(path, "recorded latency schedules cannot also use random jitter")
        self.seed = int(_finite(f"{path}.seed", self.seed))
        self.timeout_s = _positive(f"{path}.timeout_s", self.timeout_s)
        self.max_retries = int(_non_negative(f"{path}.max_retries", self.max_retries))
        self.backoff_s = _non_negative(f"{path}.backoff_s", self.backoff_s)
        self.max_consecutive_failures = int(
            _positive(f"{path}.max_consecutive_failures", self.max_consecutive_failures)
        )


@dataclass
class ScenarioConfig:
    name: str = "two_trap"
    route_length: Optional[float] = None
    duration: Optional[float] = None
    seed: int = 0
    randomize: bool = False
    params: Dict[str, Any] = field(default_factory=dict)
    simulator: str = "native"

    def validate(self, path: str = "scenario") -> None:
        self.simulator = _choice(f"{path}.simulator", self.simulator, ("native", "metadrive"))
        self.name = _choice(f"{path}.name", self.name, SCENARIO_NAMES)
        self.seed = int(_finite(f"{path}.seed", self.seed))
        if self.route_length is not None:
            self.route_length = _positive(f"{path}.route_length", self.route_length)
        if self.duration is not None:
            self.duration = _positive(f"{path}.duration", self.duration)
        if not isinstance(self.params, dict):
            _fail(f"{path}.params", "must be a mapping")


@dataclass
class StackConfig:
    mode: str = StackMode.PROPOSED.value
    initial_skill: str = "CRUISE_RIGHT"
    blend_ticks: int = 24
    fixed_rate_hz: float = 1.0
    always_on_hz: float = 10.0
    shield_clear_ticks: int = 16
    shield_hold_ticks: int = 1
    min_call_ticks: int = 28
    sigma_threshold_scale: float = 1.55
    use_blend: bool = True
    hard_switch: bool = False
    pending_margin: bool = True
    pending_margin_on_all_calls: bool = False
    abort_on_emergency: bool = False
    freeze_speed: float = 0.5
    freeze_window_s: float = 1.0
    route_success_pct: float = 90.0
    max_pending_s: float = 2.0
    router_latency_s: Optional[float] = None
    record_interactions: bool = False
    router_async: bool = False
    realtime: bool = False
    skill_switch_guard: bool = False
    rearm_on_skill_change: bool = False
    simulate_deadline_wait: bool = False
    trigger_policy: str = "legacy"
    trigger_latency_bound_s: float = 0.3
    trigger_transition_s: float = 2.0
    trigger_buffer_s: float = 0.5
    trigger_threshold_s: float = 3.0
    polling_phase_s: float = 0.0
    polling_usefulness_gate: bool = False

    def validate(self, path: str = "stack") -> None:
        # StackMode covers the primary ablations; the extra entries are
        # controller variants (always-on, no-margin, no-blend) handled by the
        # same DrivingStack implementation.
        self.mode = _choice(f"{path}.mode", self.mode, EVAL_MODES)
        self.initial_skill = _choice(f"{path}.initial_skill", self.initial_skill, SKILL_NAMES)
        self.blend_ticks = int(_non_negative(f"{path}.blend_ticks", self.blend_ticks))
        self.fixed_rate_hz = _positive(f"{path}.fixed_rate_hz", self.fixed_rate_hz)
        self.always_on_hz = _positive(f"{path}.always_on_hz", self.always_on_hz)
        self.shield_clear_ticks = int(
            _non_negative(f"{path}.shield_clear_ticks", self.shield_clear_ticks)
        )
        self.shield_hold_ticks = int(
            _non_negative(f"{path}.shield_hold_ticks", self.shield_hold_ticks)
        )
        self.min_call_ticks = int(_non_negative(f"{path}.min_call_ticks", self.min_call_ticks))
        self.sigma_threshold_scale = _positive(
            f"{path}.sigma_threshold_scale", self.sigma_threshold_scale
        )
        self.freeze_speed = _non_negative(f"{path}.freeze_speed", self.freeze_speed)
        self.freeze_window_s = _positive(f"{path}.freeze_window_s", self.freeze_window_s)
        self.route_success_pct = _non_negative(f"{path}.route_success_pct", self.route_success_pct)
        self.max_pending_s = _positive(f"{path}.max_pending_s", self.max_pending_s)
        if self.route_success_pct > 100.0:
            _fail(f"{path}.route_success_pct", "must be <= 100")
        if self.router_latency_s is not None:
            self.router_latency_s = _non_negative(f"{path}.router_latency_s", self.router_latency_s)
        _bool(f"{path}.record_interactions", self.record_interactions)
        _bool(f"{path}.router_async", self.router_async)
        _bool(f"{path}.realtime", self.realtime)
        _bool(f"{path}.skill_switch_guard", self.skill_switch_guard)
        _bool(f"{path}.rearm_on_skill_change", self.rearm_on_skill_change)
        _bool(f"{path}.simulate_deadline_wait", self.simulate_deadline_wait)
        self.trigger_policy = _choice(
            f"{path}.trigger_policy",
            self.trigger_policy,
            ("legacy", "deadline", "no_latency", "ttc", "benefit", "uncertainty", "barrier"),
        )
        for key in ("trigger_latency_bound_s", "trigger_buffer_s"):
            setattr(self, key, _non_negative(f"{path}.{key}", getattr(self, key)))
        self.polling_phase_s = _non_negative(f"{path}.polling_phase_s", self.polling_phase_s)
        _bool(f"{path}.polling_usefulness_gate", self.polling_usefulness_gate)
        for key in ("trigger_transition_s", "trigger_threshold_s"):
            setattr(self, key, _positive(f"{path}.{key}", getattr(self, key)))
        if self.router_async and not self.realtime:
            _fail(
                path,
                "router_async requires realtime pacing so inference deadlines use the same clock",
            )


@dataclass
class EvalConfig:
    scenarios: List[ScenarioConfig] = field(default_factory=lambda: [ScenarioConfig("two_trap")])
    modes: List[str] = field(default_factory=lambda: ["proposed"])
    seeds: List[int] = field(default_factory=lambda: [0])
    out_dir: str = "runs"

    def validate(self, path: str = "eval") -> None:
        self.scenarios = [
            s if isinstance(s, ScenarioConfig) else _construct(ScenarioConfig, s, path)
            for s in self.scenarios
        ]
        if not self.scenarios or not self.modes or not self.seeds:
            _fail(path, "scenarios, modes and seeds must be nonempty")
        for scenario in self.scenarios:
            scenario.validate(path)
        self.modes = [_choice(f"{path}.modes", str(m), EVAL_MODES) for m in self.modes]
        self.seeds = [int(_finite(f"{path}.seeds", s)) for s in self.seeds]


@dataclass
class ProjectConfig:
    name: str = "event-cbf"
    seed: int = 0
    checkpoint_dir: str = "models/native"
    log_level: str = "INFO"
    router: RouterConfig = field(default_factory=RouterConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    stack: StackConfig = field(default_factory=StackConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> "ProjectConfig":
        if not isinstance(self.name, str) or not self.name.strip():
            _fail("name", "must be a non-empty string")
        self.seed = int(_finite("seed", self.seed))
        self.log_level = _choice(
            "log_level", self.log_level.upper(), ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        )
        self.router.validate("router")
        self.shield.validate("shield")
        self.stack.validate("stack")
        self.eval.validate("eval")
        if not self.checkpoint_dir:
            _fail("checkpoint_dir", "must be a non-empty path")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(asdict(self))


def _to_plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _construct(cls: Type[T], data: Mapping[str, Any], path: str) -> T:
    """Construct a dataclass, rejecting unknown keys (typo protection)."""

    if not isinstance(data, Mapping):
        _fail(path, f"expected a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cast(Any, cls))}
    unknown = sorted(set(data) - known)
    if unknown:
        _fail(path, f"unknown keys {unknown}; allowed keys are {sorted(known)}")
    kwargs = dict(data)
    if cls is ProjectConfig:
        for nested, nested_cls in (
            ("router", RouterConfig),
            ("shield", ShieldConfig),
            ("stack", StackConfig),
            ("eval", EvalConfig),
        ):
            if nested in kwargs and isinstance(kwargs[nested], Mapping):
                kwargs[nested] = _construct(nested_cls, kwargs[nested], f"{path}.{nested}")
    return cls(**kwargs)  # type: ignore[return-value]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_mapping(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            raise ConfigError(f"unsupported config extension {path.suffix!r}; use .yaml/.yml/.json")
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"cannot parse config {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"config {path} must contain a mapping at the top level")
    return dict(data)


def _read_config_chain(path: Path, seen: Optional[set[Path]] = None) -> Dict[str, Any]:
    """Read a config and resolve ``extends`` recursively, child wins."""

    config_path = Path(path).expanduser().resolve()
    visited = set() if seen is None else set(seen)
    if config_path in visited:
        raise ConfigError(f"config extends cycle involving {config_path}")
    visited.add(config_path)
    data = _read_mapping(config_path)
    parent = data.pop("extends", None)
    if parent is not None:
        parent_path = (config_path.parent / str(parent)).resolve()
        data = _deep_merge(_read_config_chain(parent_path, visited), data)
    data.pop("profile", None)  # metadata only; not a dataclass field
    return data


def load_config(path: Union[str, Path, None] = None) -> ProjectConfig:
    """Load and validate a config file.

    ``extends: ../base.yaml`` is resolved recursively relative to each file,
    with the child mapping taking precedence.
    """

    data: Dict[str, Any] = {}
    if path is not None:
        data = _read_config_chain(Path(path))

    config = _construct(ProjectConfig, data, "config")
    return config.validate()
