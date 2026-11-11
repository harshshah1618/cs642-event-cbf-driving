"""Human-readable, honest reporting for production evaluations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from event_cbf_driving.eval.metrics import aggregate_rollouts, result_to_episode_record

if TYPE_CHECKING:  # pragma: no cover - typing only
    from event_cbf_driving.eval.runner import EvaluationResult


def aggregate_episodes(episodes: Sequence[Any]) -> Dict[str, Any]:
    """Aggregate episode dicts (or ``RolloutResult`` objects) by scenario/mode."""

    return aggregate_rollouts(episodes)


def aggregate_by_mode(episodes: Sequence[Any]) -> Dict[str, Any]:
    """Aggregate episode dicts (or ``RolloutResult`` objects) by mode only."""

    return aggregate_rollouts(episodes, group_by=("mode",))


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        return _number(value.get("mean"))
    return _number(value)


def _ci(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        return _number(value.get("ci95"))
    return None


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}{suffix}"


def _fmt_summary(value: Any, digits: int = 1, suffix: str = "") -> str:
    mean = _mean(value)
    if mean is None:
        return "-"
    ci = _ci(value)
    if ci is None:
        return f"{mean:.{digits}f}{suffix}"
    return f"{mean:.{digits}f} ± {ci:.{digits}f}{suffix}"


def _rate(metrics: Mapping[str, Any], prefix: str) -> float:
    value = _number(metrics.get(f"{prefix}_rate"))
    if value is not None:
        return value
    count = _number(metrics.get(f"{prefix}_count"))
    n = _number(metrics.get("n"))
    if count is not None and n:
        return count / n
    return 0.0


def _flags(metrics: Mapping[str, Any]) -> str:
    flags: List[str] = []
    crash_rate = _rate(metrics, "crash")
    freeze_rate = _rate(metrics, "freeze")
    if crash_rate > 0.0:
        flags.append(f"crash {100.0 * crash_rate:.0f}%")
    if freeze_rate > 0.0:
        flags.append(f"freeze {100.0 * freeze_rate:.0f}%")
    for incident in ("out_of_road", "safety_abort"):
        value = metrics.get(incident, {})
        if isinstance(value, dict) and (value.get("mean") or 0) > 0:
            flags.append(incident)

    safety = _mean(metrics.get("safety_events"))
    if safety is not None and safety > 0.0:
        flags.append(f"safety_events={safety:.1f}")
    infeasible = _mean(metrics.get("infeasible_ticks"))
    if infeasible is not None and infeasible > 0.0:
        flags.append(f"infeasible={infeasible:.1f}")
    barrier = _mean(metrics.get("min_barrier_h"))
    if barrier is not None and barrier < 0.0:
        flags.append(f"min_h={barrier:.2f}")

    survival = _number(metrics.get("pending_survival_rate"))
    if survival is not None and survival < 0.999:
        flags.append(f"pending_survival={100.0 * survival:.0f}%")
    pending = _mean(metrics.get("pending_ticks"))
    if pending is not None and pending > 0.0:
        flags.append(f"pending_ticks={pending:.1f}")

    fallbacks = _mean(metrics.get("router_fallbacks"))
    if fallbacks is not None and fallbacks > 0.0:
        flags.append(f"fallbacks={fallbacks:.1f}")
    live_calls = _mean(metrics.get("router_live_calls"))
    if live_calls is not None and live_calls > 0.0:
        flags.append(f"live_calls={live_calls:.1f}")
    tokens = _mean(metrics.get("router_tokens"))
    if tokens is not None and tokens > 0.0:
        flags.append(f"tokens={tokens:.0f}")
    rejected = _mean(metrics.get("calls_rejected"))
    if rejected is not None and rejected > 0.0:
        flags.append(f"rejected={rejected:.1f}")
    return ", ".join(flags) if flags else "-"


def _scenario_table(scenario: str, by_mode: Mapping[str, Mapping[str, Any]]) -> List[str]:
    lines = [
        f"### `{scenario}`",
        "",
        "| Mode | n | Route % (mean ± ci95) | Crash % | Freeze % | Calls | Peak jerk | Flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for mode in sorted(by_mode):
        metrics = by_mode[mode]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{mode}`",
                    str(metrics.get("n", 0)),
                    _fmt_summary(metrics.get("route_pct")),
                    _fmt_summary(metrics.get("crash_pct")),
                    _fmt_summary(metrics.get("freeze_pct")),
                    _fmt_summary(metrics.get("router_calls")),
                    _fmt_summary(metrics.get("peak_lateral_jerk")),
                    _flags(metrics),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _mode_table(episodes: Sequence[Any]) -> List[str]:
    by_mode = aggregate_by_mode(episodes) if episodes else {}
    lines = [
        "## Per-mode summary (all scenarios)",
        "",
        "| Mode | n | Route % | Crash % | Freeze % | Calls | Peak jerk |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not by_mode:
        lines.append("| _no episodes_ | 0 | - | - | - | - | - |")
    for mode in sorted(by_mode):
        metrics = by_mode[mode]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{mode}`",
                    str(metrics.get("n", 0)),
                    _fmt_summary(metrics.get("route_pct")),
                    _fmt_summary(metrics.get("crash_pct")),
                    _fmt_summary(metrics.get("freeze_pct")),
                    _fmt_summary(metrics.get("router_calls")),
                    _fmt_summary(metrics.get("peak_lateral_jerk")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _appendix(episodes: Sequence[Any]) -> List[str]:
    lines = [
        "## Per-episode appendix",
        "",
        "| Scenario | Mode | Seed | Repeat | Route % | Crash | Freeze | Calls | End reason |",
        "| --- | --- | ---: | ---: | ---: | :---: | :---: | ---: | --- |",
    ]
    if not episodes:
        lines.append("| _no episodes_ | - | - | - | - | - | - | - | - |")
    rows = []
    for episode in episodes:
        record = episode if isinstance(episode, Mapping) else result_to_episode_record(episode)
        rows.append(record)
    rows.sort(
        key=lambda r: (
            str(r.get("scenario") or ""),
            str(r.get("mode") or ""),
            int(r.get("seed") or 0),
            int(r.get("repeat") or 0),
        )
    )
    for record in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record.get('scenario') or '-'}`",
                    f"`{record.get('mode') or '-'}`",
                    str(record.get("seed") if record.get("seed") is not None else "-"),
                    str(record.get("repeat") if record.get("repeat") is not None else 0),
                    _fmt(record.get("route_pct")),
                    "yes" if record.get("crashed") else "no",
                    "yes" if record.get("freeze") else "no",
                    str(record.get("router_calls") if record.get("router_calls") is not None else "-"),
                    str(record.get("end_reason") or "-"),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _provenance(result: Any) -> List[str]:
    manifest = result.manifest or {}
    git = manifest.get("git") or {}
    environment = manifest.get("environment") or {}
    extra = manifest.get("extra") or {}
    router = extra.get("router") or {}
    config_sha = manifest.get("config_sha256")
    commit = git.get("commit") or "unknown"
    branch = git.get("branch") or "unknown"
    dirty = "yes" if git.get("dirty") else "no"
    describe = router.get("describe") or router.get("name") or "unknown"
    lines = [
        f"# Evaluation report: `{result.run_id}`",
        "",
        f"- Run id: `{result.run_id}`",
        f"- Created: `{manifest.get('created_at', 'unknown')}`",
        f"- Git commit: `{commit}` (branch `{branch}`, dirty: {dirty})",
        f"- Python: `{environment.get('python', 'unknown')}` "
        f"({environment.get('platform', 'unknown')})",
        f"- Router: `{describe}`",
        f"- Live capable: {'yes' if router.get('live_capable') else 'no'}",
        f"- Config SHA-256: `{config_sha or 'unknown'}`",
        f"- Schema version: `{result.schema_version}`",
    ]
    telemetry = extra.get("router_telemetry") or {}
    if telemetry:
        lines.append(
            "- Router telemetry: "
            f"live calls `{int(telemetry.get('router_live_calls') or 0)}` / "
            f"calls `{int(telemetry.get('router_calls') or 0)}`, "
            f"fallbacks `{int(telemetry.get('router_fallbacks') or 0)}`, "
            f"tokens `{int(telemetry.get('total_tokens') or 0)}`"
        )
        models = telemetry.get("models") or []
        if models:
            label = (
                "Live model(s)"
                if int(telemetry.get("router_live_calls") or 0) > 0
                else "Router model(s)"
            )
            lines.append(f"- {label}: " + ", ".join(f"`{m}`" for m in models))
    if extra.get("seeds") is not None:
        lines.append(f"- Eval seeds: `{extra.get('seeds')}` (repeats: {extra.get('repeats', 1)})")
    if extra.get("modes") is not None:
        lines.append(f"- Modes: {', '.join(f'`{m}`' for m in extra.get('modes') or [])}")
    lines.append("")
    return lines


def _warnings_section(result: Any) -> List[str]:
    warnings = list(result.warnings or [])
    lines = ["## Warnings", ""]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.append("")
    return lines


def _checkpoint_section(result: Any) -> List[str]:
    manifest = result.manifest or {}
    extra = manifest.get("extra") or {}
    aliases = extra.get("aliases") or {}
    loading = extra.get("checkpoint_loading") or {}
    checkpoints = manifest.get("checkpoints") or {}
    lines = ["## Checkpoints, aliases, and fingerprints", ""]
    if aliases:
        lines.append("Checkpoint aliases:")
        lines.append("")
        for name in sorted(aliases):
            source = aliases[name]
            if isinstance(source, Mapping):
                source = source.get("source", source)
            lines.append(f"- `{name}` -> `{source}`")
        lines.append("")
    else:
        lines.append("No checkpoint aliases were used.")
        lines.append("")

    fallbacks = loading.get("fallbacks") or []
    if fallbacks:
        lines.append(
            "**Smoke-test fallback policies were used for missing required checkpoints:** "
            + ", ".join(f"`{name}`" for name in fallbacks)
            + ". These numbers are not trained-policy results."
        )
        lines.append("")

    if checkpoints:
        lines.append("| Checkpoint file | SHA-256 | Bytes |")
        lines.append("| --- | --- | ---: |")
        for name in sorted(checkpoints):
            info = checkpoints[name] or {}
            if info.get("missing"):
                lines.append(f"| `{name}` | missing | - |")
            else:
                sha = str(info.get("sha256") or "")
                lines.append(
                    f"| `{name}` | `{sha[:16]}` | {info.get('bytes', '-')} |"
                )
    else:
        lines.append("No checkpoint fingerprints were available.")
    lines.append("")
    return lines


def _notes(result: Any) -> List[str]:
    manifest = result.manifest or {}
    extra = manifest.get("extra") or {}
    router = extra.get("router") or {}
    config = manifest.get("config") or {}
    eval_cfg = config.get("eval") or {}
    repeats = int(eval_cfg.get("repeats") or extra.get("repeats") or 1)
    n_episodes = len(result.episodes or [])
    lines = [
        "## Honest notes",
        "",
        f"- This report summarizes {n_episodes} measured episode(s). Confidence intervals use "
        "`1.96 * sample_std(ddof=1) / sqrt(n)`; they are `0` for `n < 2` and are not "
        "meaningful for tiny samples.",
        "- The default offline router is the deterministic **scripted** lane-threat rule. "
        "These results characterize the controller + shield loop; they are not evidence about "
        "general language-model behavior unless a live backend was explicitly configured. "
        f"Backend used here: `{router.get('name') or router.get('describe') or 'unknown'}`.",
    ]
    if repeats > 1:
        lines.append(
            f"- Repeats ({repeats}) re-run the same seed by design. They are not independent "
            "samples; treating them as such can understate uncertainty."
        )
    if not result.episodes:
        lines.append("- No episode records were produced; aggregate tables may be empty.")
    lines.append("")
    return lines


def render_markdown(result: "EvaluationResult") -> str:
    """Render a deterministic Markdown report for an :class:`EvaluationResult`."""

    lines: List[str] = []
    lines.extend(_provenance(result))
    lines.extend(_warnings_section(result))

    lines.extend(["## Per-scenario results", ""])
    aggregates = result.aggregates or {}
    if not aggregates:
        lines.extend(["_No aggregate results._", ""])
    for scenario in sorted(aggregates):
        lines.extend(_scenario_table(scenario, aggregates[scenario]))

    lines.extend(_mode_table(result.episodes or []))
    physical = [record for record in result.episodes or []
                if record.get("plant_peak_lateral_jerk") is not None]
    if physical:
        lines.extend(["## Physical tracking diagnostics", "",
                      "The main table's jerk measures requested acceleration. Achieved motion is measured separately below.", "",
                      "| Scenario | Mode | Achieved peak lateral jerk (m/s³) | Acceleration RMSE (m/s²) | Saturation fraction |",
                      "| --- | --- | ---: | ---: | ---: |"])
        for scenario in sorted(aggregates):
            for mode, metrics in sorted(aggregates[scenario].items()):
                lines.append(f"| `{scenario}` | `{mode}` | "
                             f"{_fmt_summary(metrics.get('plant_peak_lateral_jerk'))} | "
                             f"{_fmt_summary(metrics.get('acceleration_tracking_rmse'), 3)} | "
                             f"{_fmt_summary(metrics.get('actuator_saturation_fraction'), 3)} |")
        lines.append("")
    lines.extend(_appendix(result.episodes or []))
    lines.extend(_checkpoint_section(result))
    lines.extend(_notes(result))
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "aggregate_by_mode",
    "aggregate_episodes",
    "render_markdown",
]
