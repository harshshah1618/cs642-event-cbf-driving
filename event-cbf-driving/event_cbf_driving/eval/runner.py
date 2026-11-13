"""Production evaluation runner: every mode x scenario x seed, honestly aggregated."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from event_cbf_driving.config import (
    EVAL_MODES,
    BaselineMode,
    ProjectConfig,
    ScenarioConfig,
    ShieldConfig,
    StackConfig,
)
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.errors import ArtifactError, ConfigError
from event_cbf_driving.eval.metrics import (
    aggregate_rollouts,
    json_safe,
    result_to_episode_record,
)
from event_cbf_driving.eval.reporting import render_markdown
from event_cbf_driving.io_utils import (
    RunDirectory,
    atomic_write_json,
    atomic_write_text,
    new_run_id,
    slugify,
)
from event_cbf_driving.provenance import (
    build_manifest,
    git_info,
    validate_manifest,
)
from event_cbf_driving.router.factory import create_router
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.skills.checkpoints import load_skill
from event_cbf_driving.skills.model import ActorCritic
from event_cbf_driving.skills.policy import SkillMeta, SkillPolicy
from event_cbf_driving.stack.baselines import M3BaselineStack
from event_cbf_driving.stack.controller import DrivingStack, RolloutResult

SCHEMA_VERSION = 1
# Proposed makes at most one call per ``min_call_ticks`` (28 ticks at 50 Hz).
# A generous absolute ceiling catches runaway chattering without rejecting
# legitimate short scenarios.
MAX_PROPOSED_ROUTER_CALLS = 100

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_MODES = {
    "fasionad_wakeup": BaselineMode.FASIONAD,
    "cert_replan": BaselineMode.CERT,
}


@dataclass
class EvaluationResult:
    """Result of one multi-mode, multi-seed evaluation run."""

    run_id: str
    run_dir: Optional[Path]
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    aggregates: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "run_dir": None if self.run_dir is None else str(self.run_dir),
                "episodes": self.episodes,
                "aggregates": self.aggregates,
                "manifest": self.manifest,
                "warnings": self.warnings,
            }
        )

    def results_payload(self) -> Dict[str, Any]:
        manifest = self.manifest or {}
        extra = manifest.get("extra") or {}
        return json_safe(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "created_at": manifest.get("created_at"),
                "config_sha256": manifest.get("config_sha256"),
                "environment": manifest.get("environment"),
                "git": manifest.get("git"),
                "aggregates": self.aggregates,
                "episodes": self.episodes,
                "warnings": self.warnings,
                "aliases": extra.get("aliases") or {},
                "router": extra.get("router") or {},
                "router_telemetry": extra.get("router_telemetry") or {},
            }
        )

    def write(self) -> Path:
        """Atomically (re)write results.json, report.md, and manifest.json."""

        if self.run_dir is None:
            raise ArtifactError("cannot write an EvaluationResult with run_dir=None")
        target = Path(self.run_dir)
        for sub in ("logs", "figures", "episodes"):
            (target / sub).mkdir(parents=True, exist_ok=True)
        run = RunDirectory(root=target, run_id=self.run_id)
        run.write_results(self.results_payload())
        run.write_report(render_markdown(self))
        run.write_manifest(self.manifest)
        return target


def _resolve_checkpoint_dir(config: ProjectConfig, checkpoint_dir: Optional[Path | str]) -> Path:
    raw = Path(checkpoint_dir if checkpoint_dir is not None else config.checkpoint_dir).expanduser()
    if not raw.is_absolute():
        raw = _REPO_ROOT / raw
    return raw


def _fallback_policy(name: str) -> SkillPolicy:
    """Documented smoke-test fallback: an untrained, deterministic lane holder."""

    from event_cbf_driving.env.skill_lane import LANE_Y

    target_lane = 1 if name == "OVERTAKE_LEFT" else 0
    try:
        import torch

        with torch.random.fork_rng():
            torch.manual_seed(0)
            net = ActorCritic()
    except Exception:  # pragma: no cover - torch is a hard dependency
        net = ActorCritic()
    meta = SkillMeta(
        name=name,
        target_lane=target_lane,
        y_target=float(LANE_Y[target_lane]),
        threshold=1e9,
        sigma_mean=0.0,
        sigma_std=1.0,
        spread=1.0,
    )
    return SkillPolicy(name, net, meta)


def load_skill_bank(
    checkpoint_dir: Path | str,
    *,
    allow_missing: bool = False,
) -> Tuple[Dict[str, SkillPolicy], Dict[str, Any]]:
    """Load CRUISE_RIGHT + OVERTAKE_LEFT, aliasing RETURN_RIGHT if absent.

    ``RETURN_RIGHT.pt`` is optional: when missing, a shallow alias of the
    CRUISE_RIGHT policy (same network weights, explicit ``RETURN_RIGHT``
    name/meta) is used. Missing required skills raise ``FileNotFoundError``
    unless ``allow_missing=True``, which installs the documented untrained
    smoke-test fallback and records it in the returned metadata.
    """

    root = Path(checkpoint_dir)
    policies: Dict[str, SkillPolicy] = {}
    aliases: Dict[str, str] = {}
    fallbacks: List[str] = []
    missing: List[str] = []
    sources: Dict[str, Any] = {}
    warnings: List[str] = []

    required = ("CRUISE_RIGHT", "OVERTAKE_LEFT")
    for name in required:
        path = root / f"{name}.pt"
        if path.exists():
            policies[name] = load_skill(path)
            sources[name] = {"path": str(path)}
            continue
        missing.append(str(path))
        if not allow_missing:
            raise FileNotFoundError(
                f"Missing required skill checkpoint {path}; run event-cbf train first "
                "(or pass allow_missing_checkpoints=True for a smoke-only fallback)"
            )
        policies[name] = _fallback_policy(name)
        fallbacks.append(name)
        sources[name] = {"fallback": True, "path": None}
        warnings.append(
            f"Missing required checkpoint {path}; allow_missing_checkpoints=True installed "
            f"an untrained fallback for {name} (smoke tests only, not a trained result)."
        )

    return_path = root / "RETURN_RIGHT.pt"
    if return_path.exists():
        policies["RETURN_RIGHT"] = load_skill(return_path)
        sources["RETURN_RIGHT"] = {"path": str(return_path)}
    elif "CRUISE_RIGHT" in policies:
        base = policies["CRUISE_RIGHT"]
        alias = copy.copy(base)
        alias.name = "RETURN_RIGHT"
        meta = copy.copy(base.meta)
        meta.name = "RETURN_RIGHT"
        alias.meta = meta
        policies["RETURN_RIGHT"] = alias
        aliases["RETURN_RIGHT"] = "CRUISE_RIGHT"
        sources["RETURN_RIGHT"] = {"alias_of": "CRUISE_RIGHT", "path": str(root / "CRUISE_RIGHT.pt")}
        warnings.append(
            f"RETURN_RIGHT.pt is missing in {root}; using CRUISE_RIGHT weights as an "
            "explicit RETURN_RIGHT alias."
        )
    else:  # pragma: no cover - only reachable with allow_missing + corrupt state
        policies["RETURN_RIGHT"] = _fallback_policy("RETURN_RIGHT")
        fallbacks.append("RETURN_RIGHT")
        sources["RETURN_RIGHT"] = {"fallback": True, "path": None}
        warnings.append("RETURN_RIGHT fallback policy installed (smoke tests only).")

    info = {
        "directory": str(root),
        "required": list(required),
        "aliases": dict(aliases),
        "fallbacks": list(fallbacks),
        "missing": list(missing),
        "sources": sources,
        "allow_missing_checkpoints": bool(allow_missing),
        "warnings": warnings,
    }
    return policies, info


def _copy_stack_config(stack_config: Optional[StackConfig], mode: str) -> StackConfig:
    if stack_config is None:
        config = StackConfig()
    elif isinstance(stack_config, StackConfig):
        config = copy.deepcopy(stack_config)
    else:
        raise ConfigError(f"stack_config must be a StackConfig, got {type(stack_config).__name__}")
    config.mode = mode
    return config


def _copy_shield_config(shield_config: Optional[ShieldConfig]) -> ShieldConfig:
    if shield_config is None:
        return ShieldConfig()
    if not isinstance(shield_config, ShieldConfig):
        raise ConfigError(f"shield_config must be a ShieldConfig, got {type(shield_config).__name__}")
    return copy.deepcopy(shield_config)


def run_episode(
    mode: str,
    scenario: ScenarioConfig,
    seed: int,
    skills: Mapping[str, SkillPolicy],
    router: Any,
    *,
    stack_config: Optional[StackConfig] = None,
    shield_config: Optional[ShieldConfig] = None,
) -> RolloutResult:
    """Run exactly one closed-loop episode for ``mode``.

    Unknown modes raise :class:`ConfigError`; there is no silent fallback. The
    caller's ``ScenarioConfig``, ``StackConfig``, and ``ShieldConfig`` are not
    mutated. The router is supplied already-built; ``DrivingStack.run`` resets
    it, while the baseline stacks do not.
    """

    mode_name = str(mode)
    if mode_name not in EVAL_MODES:
        raise ConfigError(f"unknown evaluation mode {mode_name!r}; choose from {list(EVAL_MODES)}")
    if router is None:
        raise ConfigError("run_episode requires an already-built router backend")

    if isinstance(scenario, Mapping):
        scenario = ScenarioConfig(**dict(scenario))
    if not isinstance(scenario, ScenarioConfig):
        raise ConfigError(f"scenario must be a ScenarioConfig, got {type(scenario).__name__}")
    scenario_copy = copy.deepcopy(scenario)
    scenario_copy.validate("scenario")
    scenario_copy.seed = int(seed)

    config = _copy_stack_config(stack_config, mode_name)
    shield_cfg = _copy_shield_config(shield_config)

    if mode_name == "unshielded_skill":
        config.use_blend = False
        shield_cfg.enabled = False
        shield_cfg.inflate = False
    elif mode_name == "static_cbf":
        config.use_blend = False
        shield_cfg.enabled = True
        shield_cfg.inflate = False
    elif mode_name == "fixed_rate_vlm":
        config.hard_switch = True
        config.use_blend = False
        shield_cfg.enabled = False
        shield_cfg.inflate = False
    elif mode_name == "always_on_vlm":
        config.hard_switch = True
        config.use_blend = False
        shield_cfg.enabled = False
        shield_cfg.inflate = False
    elif mode_name == "proposed":
        shield_cfg.enabled = True
        shield_cfg.inflate = True
    elif mode_name == "proposed_no_margin":
        shield_cfg.enabled = True
        shield_cfg.inflate = True
        shield_cfg.pending_margin = False
    elif mode_name == "proposed_no_blend":
        config.hard_switch = True
        config.use_blend = False
        shield_cfg.enabled = True
        shield_cfg.inflate = True
    elif mode_name in _BASELINE_MODES:
        shield_cfg.enabled = True
        shield_cfg.inflate = False

    config.validate("stack")
    shield_cfg.validate("shield")
    shield = InflatedCBFShield(config=shield_cfg)
    env = make_env(scenario_copy.name, scenario_copy)

    try:
        stack: Union[DrivingStack, M3BaselineStack]
        if mode_name in _BASELINE_MODES:
            stack = M3BaselineStack(
                skills=dict(skills),
                router=router,
                shield=shield,
                mode=_BASELINE_MODES[mode_name],
                config=config,
            )
        else:
            stack = DrivingStack(
                skills=dict(skills),
                router=router,
                shield=shield,
                config=config,
            )
        result = stack.run(env, seed=int(seed))
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    result.scenario = scenario_copy.name
    result.seed = int(seed)
    return result


def _router_info(router: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "name": getattr(router, "name", type(router).__name__),
        "live_capable": bool(getattr(router, "is_live_capable", False)),
        "allow_live": bool(getattr(router, "allow_live", False)),
    }
    model = getattr(router, "model", None)
    if model:
        info["model"] = str(model)
    describe = getattr(router, "describe", None)
    try:
        info["describe"] = describe() if callable(describe) else str(describe or info["name"])
    except Exception as exc:  # pragma: no cover - defensive
        info["describe"] = info["name"]
        info["describe_error"] = str(exc)
    health = getattr(router, "health", None)
    if callable(health):
        try:
            state = health()
            info["health"] = state.to_dict() if hasattr(state, "to_dict") else json_safe(state)
        except Exception as exc:  # pragma: no cover - defensive
            info["health_error"] = str(exc)
    return json_safe(info)


def _router_telemetry(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate live-call and token telemetry across episode records."""

    def total(key: str) -> int:
        return sum(int(record.get(key) or 0) for record in episodes)

    calls = total("router_calls")
    live_calls = total("router_live_calls")
    models = sorted(
        {
            str(model)
            for record in episodes
            for model in (record.get("router_models") or [])
            if str(model)
        }
    )
    return {
        "router_calls": calls,
        "router_live_calls": live_calls,
        "router_fallbacks": total("router_fallbacks"),
        "router_live_fraction": (float(live_calls) / float(calls)) if calls else 0.0,
        "prompt_tokens": total("router_prompt_tokens"),
        "completion_tokens": total("router_completion_tokens"),
        "total_tokens": total("router_tokens"),
        "models": models,
    }


def _mean_value(container: Any) -> Optional[float]:
    if isinstance(container, Mapping):
        container = container.get("mean")
    if container is None or isinstance(container, bool):
        if isinstance(container, bool):
            return 1.0 if container else 0.0
        return None
    try:
        value = float(container)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metric_mean(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    if key in metrics:
        value = _mean_value(metrics[key])
        if value is not None:
            return value
    for alias in (f"{key}_mean", f"{key}_rate"):
        if alias in metrics:
            value = _mean_value(metrics[alias])
            if value is not None:
                return value
    return None


def _regression_failures(
    aggregates: Mapping[str, Any],
    *,
    min_route_pct: float,
    require_no_crash: bool,
    require_no_freeze: bool,
) -> List[str]:
    failures: List[str] = []
    for scenario in sorted(aggregates):
        by_mode = aggregates[scenario]
        if not isinstance(by_mode, Mapping) or "proposed" not in by_mode:
            continue
        metrics = by_mode["proposed"]
        for incident in ("out_of_road", "safety_abort"):
            if (_metric_mean(metrics, incident) or 0.0) > 0.0:
                failures.append(f"{scenario}/proposed: {incident} occurred")
        n = _metric_mean(metrics, "n")
        if n is not None and n <= 0:
            failures.append(f"{scenario}/proposed: no episodes were evaluated")
            continue

        crash_rate = _metric_mean(metrics, "crash_rate")
        if crash_rate is None:
            crash_rate = _metric_mean(metrics, "crashed") or 0.0
        if require_no_crash and crash_rate > 0.0:
            failures.append(
                f"{scenario}/proposed: crashed in {crash_rate:.0%} of episodes "
                "(require_no_crash=True)"
            )

        freeze_rate = _metric_mean(metrics, "freeze_rate")
        if freeze_rate is None:
            freeze_rate = _metric_mean(metrics, "freeze") or 0.0
        if require_no_freeze and freeze_rate > 0.0:
            failures.append(
                f"{scenario}/proposed: froze in {freeze_rate:.0%} of episodes "
                "(require_no_freeze=True)"
            )

        route = _metric_mean(metrics, "route_pct")
        if route is None:
            failures.append(f"{scenario}/proposed: missing route_pct aggregate")
        elif route < float(min_route_pct):
            failures.append(
                f"{scenario}/proposed: mean route {route:.1f}% is below the "
                f"{float(min_route_pct):.1f}% gate"
            )

        calls = _metric_mean(metrics, "router_calls")
        if calls is None:
            failures.append(f"{scenario}/proposed: missing finite router_calls aggregate")
        elif calls > MAX_PROPOSED_ROUTER_CALLS:
            failures.append(
                f"{scenario}/proposed: mean router calls {calls:.1f} exceed the "
                f"documented bound {MAX_PROPOSED_ROUTER_CALLS}"
            )
    return failures


def check_regressions(
    result: EvaluationResult,
    *,
    min_route_pct: float = 60.0,
    require_no_crash: bool = True,
    require_no_freeze: bool = True,
) -> List[str]:
    """Return human-readable proposed-stack gate failures (never raises).

    The router-call bound is :data:`MAX_PROPOSED_ROUTER_CALLS` (100): a
    generous absolute ceiling that catches runaway chattering. Scenarios
    without a ``proposed`` aggregate are skipped, so baseline-only runs do not
    produce spurious failures.
    """

    return _regression_failures(
        getattr(result, "aggregates", {}) or {},
        min_route_pct=min_route_pct,
        require_no_crash=require_no_crash,
        require_no_freeze=require_no_freeze,
    )


def _write_figures(
    representatives: Mapping[str, Mapping[str, RolloutResult]],
    modes: Sequence[str],
    run: RunDirectory,
    warnings: List[str],
) -> None:
    for scenario, by_mode in representatives.items():
        proposed = by_mode.get("proposed")
        if proposed is not None and proposed.interactions:
            try:
                from event_cbf_driving.eval.interactions import plot_interactions

                plot_interactions(proposed, run.figure_path(f"{scenario}-interactions.png"))
            except Exception as exc:
                warnings.append(f"interaction figure for {scenario} failed: {exc}")
    scenario_results = representatives.get("two_trap")
    if not scenario_results:
        return
    try:
        from event_cbf_driving.eval.figures import plot_ablation_bars, plot_gauntlet_rollouts
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.append(f"figures skipped: matplotlib/figures unavailable ({exc})")
        return

    ordered = [mode for mode in modes if mode in scenario_results]
    ordered.extend(sorted(mode for mode in scenario_results if mode not in ordered))
    results = [scenario_results[mode] for mode in ordered]

    # The existing helpers use a fixed layout (four colors / 2x2 panels), so
    # chunk rather than dropping modes. The first chunk keeps canonical names.
    for index in range(0, len(results), 4):
        chunk = results[index : index + 4]
        suffix = "" if index == 0 else f"-{index // 4 + 1}"
        try:
            plot_ablation_bars(chunk, run.figure_path(f"ablation-bars{suffix}.png"))
        except Exception as exc:
            warnings.append(f"figure ablation-bars{suffix}.png failed: {exc}")
        try:
            plot_gauntlet_rollouts(
                chunk, run.figure_path(f"gauntlet-rollouts{suffix}.png")
            )
        except Exception as exc:
            warnings.append(f"figure gauntlet-rollouts{suffix}.png failed: {exc}")


def _legacy_rows(aggregates: Mapping[str, Any], scenario: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_mode = aggregates.get(scenario) or {}
    for mode in sorted(by_mode):
        metrics = by_mode[mode]
        route = _metric_mean(metrics, "route_pct")
        speed = _metric_mean(metrics, "avg_speed")
        calls = _metric_mean(metrics, "router_calls")
        jerk = _metric_mean(metrics, "peak_lateral_jerk")
        sigma = _metric_mean(metrics, "peak_sigma")
        crash = _metric_mean(metrics, "crash_rate") or 0.0
        freeze = _metric_mean(metrics, "freeze_rate") or 0.0
        rows.append(
            json_safe(
                {
                    "mode": mode,
                    "crash_time": None,
                    "crashed": crash > 0.0,
                    "route_pct": None if route is None else round(route, 2),
                    "avg_speed": None if speed is None else round(speed, 3),
                    "router_calls": 0 if calls is None else int(round(calls)),
                    "peak_lateral_jerk": None if jerk is None else round(jerk, 2),
                    "freeze": freeze > 0.0,
                    "peak_sigma": None if sigma is None else round(sigma, 3),
                    "x": None,
                    "y": None,
                    "vx": None,
                    "end_time": None,
                    "n": metrics.get("n", 0),
                    "crash_rate": crash,
                    "freeze_rate": freeze,
                    "aggregate": True,
                }
            )
        )
    return rows


def _update_docs(result: EvaluationResult) -> None:
    manifest = result.manifest or {}
    extra = manifest.get("extra") or {}
    router = extra.get("router") or {}
    payload = {
        "router": router.get("describe") or router.get("name") or "unknown",
        "router_name": router.get("name") or "unknown",
        "milestone": "M3-production",
        "gauntlet": _legacy_rows(result.aggregates, "two_trap"),
        "merge": _legacy_rows(result.aggregates, "merge"),
        "dense_highway": _legacy_rows(result.aggregates, "dense_highway"),
        "production": {
            "schema_version": result.schema_version,
            "run_id": result.run_id,
            "aggregates": result.aggregates,
            "warnings": result.warnings,
            "aliases": extra.get("aliases") or {},
            "router": router,
            "router_telemetry": extra.get("router_telemetry") or {},
        },
    }
    atomic_write_json(_REPO_ROOT / "results.json", json_safe(payload))
    atomic_write_text(_REPO_ROOT / "RESULTS.md", render_markdown(result))


def run_evaluation(
    config: ProjectConfig,
    *,
    checkpoint_dir: Optional[Path | str] = None,
    router: Any = None,
    out_dir: Optional[Path | str] = None,
    run_id: Optional[str] = None,
    write: bool = True,
    update_docs: bool = False,
    record_episodes: Optional[bool] = None,
    allow_missing_checkpoints: bool = False,
) -> EvaluationResult:
    """Evaluate every configured mode/scenario/seed and aggregate honestly.

    The config is deep-copied before validation. ``checkpoint_dir`` defaults to
    ``config.checkpoint_dir`` (relative paths are resolved against the package
    parent, not the process cwd). One router is built and reused; a supplied
    router belongs to the caller and is not closed. ``DrivingStack`` resets the
    router per episode, baseline stacks may not.
    """

    if not isinstance(config, ProjectConfig):
        raise ConfigError(f"run_evaluation expected ProjectConfig, got {type(config).__name__}")
    cfg = config.copy()
    cfg.validate()

    ckpt_dir = _resolve_checkpoint_dir(cfg, checkpoint_dir)
    skills, loader_info = load_skill_bank(
        ckpt_dir, allow_missing=bool(allow_missing_checkpoints)
    )
    created_router = router is None
    active_router = router if router is not None else create_router(cfg.router)

    warnings: List[str] = []
    for message in loader_info.get("warnings") or []:
        if message not in warnings:
            warnings.append(message)

    effective_record = bool(
        cfg.eval.record_episodes if record_episodes is None else record_episodes
    )
    effective_update_docs = bool(update_docs or cfg.eval.update_docs)
    effective_run_id = run_id or cfg.eval.run_id or new_run_id("eval")
    representatives: Dict[str, Dict[str, RolloutResult]] = {}
    episode_records: List[Tuple[Dict[str, Any], RolloutResult, int, Dict[str, Any]]] = []
    router_info: Dict[str, Any] = {}
    router_telemetry: Dict[str, Any] = {}

    try:
        for scenario in cfg.eval.scenarios:
            scenario_name = scenario.name
            for seed in cfg.eval.seeds:
                for repeat in range(cfg.eval.repeats):
                    scenario_copy = copy.deepcopy(scenario)
                    scenario_copy.seed = int(seed)
                    scenario_payload = json_safe(asdict(scenario_copy))
                    for mode in cfg.eval.modes:
                        rollout = run_episode(
                            mode,
                            scenario_copy,
                            int(seed),
                            skills,
                            active_router,
                            stack_config=cfg.stack,
                            shield_config=cfg.shield,
                        )
                        record = result_to_episode_record(
                            rollout, include_calls=effective_record
                        )
                        record["scenario"] = scenario_name
                        record["mode"] = str(mode)
                        record["seed"] = int(seed)
                        record["repeat"] = int(repeat)
                        episode_records.append(
                            (record, rollout, repeat, dict(scenario_payload))
                        )
                        representatives.setdefault(scenario_name, {}).setdefault(mode, rollout)

        router_info = _router_info(active_router)
        router_info["allow_live"] = bool(cfg.router.allow_live)
        router_telemetry = _router_telemetry(
            [record for record, _result, _repeat, _scenario in episode_records]
        )
        router_info["telemetry"] = dict(router_telemetry)
        aggregates = aggregate_rollouts(
            record for record, _result, _repeat, _scenario in episode_records
        )
        regression_failures = _regression_failures(
            aggregates,
            min_route_pct=float(cfg.eval.min_route_pct),
            require_no_crash=True,
            require_no_freeze=True,
        )
        for failure in regression_failures:
            warnings.append(f"regression gate: {failure}")
    finally:
        if created_router:
            close = getattr(active_router, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    run: Optional[RunDirectory] = None
    if write:
        run = RunDirectory.create(out_dir or cfg.eval.out_dir, effective_run_id)
        if effective_record:
            for record, result, repeat, scenario_payload in episode_records:
                name = (
                    f"{slugify(result.scenario or record['scenario'])}-"
                    f"{slugify(record['mode'])}-seed{record['seed']}"
                )
                if cfg.eval.repeats > 1:
                    name += f"-rep{int(repeat)}"
                payload = json_safe(result.to_dict())
                payload["repeat"] = int(repeat)
                payload["recorded"] = True
                payload["scenario_config"] = scenario_payload
                atomic_write_json(run.episode_path(f"{name}.json"), payload)
        if cfg.eval.figures:
            _write_figures(representatives, cfg.eval.modes, run, warnings)

    config_dict = cfg.to_dict()
    scenario_configs = (config_dict.get("eval") or {}).get("scenarios") or []
    manifest = build_manifest(
        config=cfg,
        command="run_evaluation",
        run_id=effective_run_id,
        checkpoint_dir=ckpt_dir,
        extra={
            "modes": list(cfg.eval.modes),
            "scenarios": [scenario.name for scenario in cfg.eval.scenarios],
            "scenario_configs": scenario_configs,
            "seeds": list(cfg.eval.seeds),
            "repeats": int(cfg.eval.repeats),
            "router": router_info,
            "router_telemetry": router_telemetry,
            "aliases": dict(loader_info.get("aliases") or {}),
            "checkpoint_loading": loader_info,
            "warnings": list(warnings),
            "record_episodes": effective_record,
            "result_schema_version": SCHEMA_VERSION,
            "regression_gate": {
                "min_route_pct": float(cfg.eval.min_route_pct),
                "max_router_calls": MAX_PROPOSED_ROUTER_CALLS,
                "failures": regression_failures,
            },
        },
    )
    try:
        manifest["git"] = git_info(_REPO_ROOT)
    except Exception:  # pragma: no cover - provenance is best-effort
        pass
    problems = validate_manifest(manifest)
    if problems:
        raise ArtifactError("internal manifest validation failed: " + "; ".join(problems))

    evaluation = EvaluationResult(
        run_id=effective_run_id,
        run_dir=run.root if run is not None else None,
        episodes=[record for record, _episode, _repeat, _scenario in episode_records],
        aggregates=aggregates,
        manifest=manifest,
        warnings=warnings,
        schema_version=SCHEMA_VERSION,
    )
    if write:
        evaluation.write()
    if effective_update_docs:
        _update_docs(evaluation)
    return evaluation


__all__ = [
    "EvaluationResult",
    "MAX_PROPOSED_ROUTER_CALLS",
    "SCHEMA_VERSION",
    "check_regressions",
    "load_skill_bank",
    "run_episode",
    "run_evaluation",
]
