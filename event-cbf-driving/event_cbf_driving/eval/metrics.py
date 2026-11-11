"""Measured metrics helpers - no fabricated numbers.

The legacy :func:`result_to_row` / :func:`ablation_table` API is preserved.
Production aggregation lives in :func:`summarize`, :func:`aggregate_rollouts`,
and :func:`result_to_episode_record`; all of them return strict-JSON-safe
values (non-finite numbers become ``None`` rather than ``NaN``/``Infinity``).
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from event_cbf_driving.stack.result import RolloutResult


def result_to_row(r: RolloutResult) -> Dict:
    return {
        "mode": r.mode,
        "crash_time": None if r.crash_time is None else round(r.crash_time, 2),
        "crashed": r.crashed,
        "route_pct": round(r.route_pct, 2),
        "avg_speed": round(r.avg_speed, 3),
        "router_calls": r.router_calls,
        "peak_lateral_jerk": round(r.peak_lateral_jerk, 2),
        "freeze": r.freeze,
        "peak_sigma": round(r.peak_sigma, 3),
        "x": round(r.x, 2),
        "y": round(r.y, 3),
        "vx": round(r.vx, 3),
        "end_time": round(r.end_time, 2),
    }


def ablation_table(results: List[RolloutResult]) -> str:
    headers = ["Mode", "Crash (s)", "Route %", "Avg speed", "Freeze", "Router calls", "Peak jerk"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    labels = {
        "unshielded_skill": "1 Unshielded skill",
        "static_cbf": "2 Static CBF",
        "fixed_rate_vlm": "3 Fixed-rate VLM",
        "proposed": "4 Proposed",
        "fasionad_wakeup": "5 FASIONAD-style",
        "cert_replan": "6 CERT-Replan-style",
    }
    for r in results:
        crash = "none" if not r.crashed else f"{r.crash_time:.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    labels.get(r.mode, r.mode),
                    crash,
                    f"{r.route_pct:.1f}",
                    f"{r.avg_speed:.2f}",
                    "yes" if r.freeze else "no",
                    str(r.router_calls),
                    f"{r.peak_lateral_jerk:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strict JSON / statistics helpers
# ---------------------------------------------------------------------------
def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_int(value: Any) -> Optional[int]:
    number = _finite_float(value)
    if number is None:
        return None
    return int(round(number))


def json_safe(value: Any) -> Any:
    """Recursively convert a value to something ``strict_json_dumps`` accepts."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Enums, numpy scalars, and similar objects.
    if hasattr(value, "value") and hasattr(value, "name"):
        return json_safe(value.value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


def summarize(values: Iterable[Any]) -> Dict[str, Any]:
    """Mean / sample std / 95% CI / min / max over finite values.

    ``std`` uses ``ddof=1`` when there are at least two values (otherwise
    ``0.0``). ``ci95`` is ``1.96 * std / sqrt(n)`` and is ``0.0`` when
    ``n < 2``. If every input is missing or non-finite, ``n`` is zero and all
    summary fields are ``None``.
    """

    finite = [number for number in (_finite_float(v) for v in values) if number is not None]
    n = len(finite)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95": None,
            "min": None,
            "max": None,
        }
    mean = float(statistics.fmean(finite))
    std = float(statistics.stdev(finite)) if n > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _record_number(record: Mapping[str, Any], key: str, aliases: Sequence[str] = ()) -> Optional[float]:
    for name in (key, *aliases):
        if name not in record:
            continue
        value = record[name]
        if isinstance(value, Mapping):
            value = value.get("mean")
        number = _finite_float(value)
        if number is not None:
            return number
    return None


def _record_bool(record: Mapping[str, Any], key: str, default: bool = False) -> bool:
    if key in record:
        return bool(record[key])
    return default


def _record_incident(record: Mapping[str, Any], key: str) -> bool:
    """Read a boolean incident flag, falling back to its aggregate rate.

    Aggregate rows use ``crash_rate``/``freeze_rate`` while episode rows use
    ``crashed``/``freeze``; both spellings are accepted so aggregates can be
    re-aggregated without silently losing incidents.
    """

    if key in record:
        return bool(record[key])
    rate_keys = [f"{key}_rate"]
    if key.endswith("ed"):
        rate_keys.append(f"{key[:-2]}_rate")
    for rate_key in rate_keys:
        rate = _finite_float(record.get(rate_key))
        if rate is not None:
            return rate > 0.0
    return False


def result_to_episode_record(
    r: RolloutResult,
    *,
    include_calls: bool = False,
    include_traces: bool = False,
) -> Dict[str, Any]:
    """Compact, strict-JSON-safe episode metadata (no huge traces by default)."""

    mode = str(_get(r, "mode", "") or "")
    scenario = str(_get(r, "scenario", "") or "")
    crashed = bool(_get(r, "crashed", False))
    freeze = bool(_get(r, "freeze", False))
    route_pct = _finite_float(_get(r, "route_pct"))
    avg_speed = _finite_float(_get(r, "avg_speed"))
    router_calls = _finite_int(_get(r, "router_calls"))
    call_attempts = _finite_int(_get(r, "call_attempts"))
    record: Dict[str, Any] = {
        "mode": mode,
        "scenario": scenario,
        "seed": _finite_int(_get(r, "seed")),
        "repeat": _finite_int(_get(r, "repeat", 0)),
        "crashed": crashed,
        "crash_time": _finite_float(_get(r, "crash_time")),
        "route_pct": route_pct,
        "avg_speed": avg_speed,
        "freeze": freeze,
        "router_calls": router_calls,
        "call_attempts": call_attempts if call_attempts is not None else router_calls,
        "calls_applied": _finite_int(_get(r, "calls_applied")) or 0,
        "calls_rejected": _finite_int(_get(r, "calls_rejected")) or 0,
        "calls_suppressed": _finite_int(_get(r, "calls_suppressed")) or 0,
        "router_fallbacks": _finite_int(_get(r, "router_fallbacks")) or 0,
        "router_prompt_tokens": _finite_int(_get(r, "router_prompt_tokens")) or 0,
        "router_completion_tokens": _finite_int(_get(r, "router_completion_tokens")) or 0,
        "router_tokens": _finite_int(_get(r, "router_tokens")) or 0,
        "router_live_calls": _finite_int(_get(r, "router_live_calls")) or 0,
        "router_models": json_safe(_get(r, "router_models", []) or []),
        "peak_lateral_jerk": _finite_float(_get(r, "peak_lateral_jerk")),
        "mean_lateral_jerk": _finite_float(_get(r, "mean_lateral_jerk")),
        "plant_peak_lateral_jerk": _finite_float(_get(r, "plant_peak_lateral_jerk")),
        "acceleration_tracking_rmse": _finite_float(_get(r, "acceleration_tracking_rmse")),
        "actuator_saturation_fraction": _finite_float(_get(r, "actuator_saturation_fraction")),
        "peak_sigma": _finite_float(_get(r, "peak_sigma")),
        "override_fraction": _finite_float(_get(r, "override_fraction")),
        "pending_ticks": _finite_int(_get(r, "pending_ticks")) or 0,
        "pending_survival": bool(_get(r, "pending_survival", True)),
        "safety_events": _finite_int(_get(r, "safety_events")) or 0,
        "infeasible_ticks": _finite_int(_get(r, "infeasible_ticks")) or 0,
        "min_barrier_h": _finite_float(_get(r, "min_barrier_h")),
        "min_braking_h": _finite_float(_get(r, "min_braking_h")),
        "braking_infeasible_ticks": _finite_int(_get(r, "braking_infeasible_ticks")) or 0,
        "max_pending_s": _finite_float(_get(r, "max_pending_s")) or 0.0,
        "end_reason": str(_get(r, "end_reason", "unknown") or "unknown"),
        "out_of_road": _get(r, "end_reason") == "out_of_road",
        "safety_abort": _get(r, "end_reason") == "safety_abort",
        "x": _finite_float(_get(r, "x")),
        "y": _finite_float(_get(r, "y")),
        "vx": _finite_float(_get(r, "vx")),
        "end_time": _finite_float(_get(r, "end_time")),
        "params": json_safe(_get(r, "params", {}) or {}),
    }
    if include_calls:
        record["calls"] = json_safe(_get(r, "calls", []) or [])
    if include_traces:
        record["xs"] = json_safe(_get(r, "xs", []) or [])
        record["ys"] = json_safe(_get(r, "ys", []) or [])
        record["skills"] = json_safe(_get(r, "skills", []) or [])
    return record


def _aggregate_group(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(records)}

    summary_fields: Dict[str, Tuple[str, Sequence[str]]] = {
        "route_pct": ("route_pct", ()),
        "avg_speed": ("avg_speed", ()),
        "router_calls": ("router_calls", ("call_attempts",)),
        "call_attempts": ("call_attempts", ("router_calls",)),
        "calls_applied": ("calls_applied", ()),
        "calls_rejected": ("calls_rejected", ()),
        "calls_suppressed": ("calls_suppressed", ()),
        "router_fallbacks": ("router_fallbacks", ()),
        "router_prompt_tokens": ("router_prompt_tokens", ()),
        "router_completion_tokens": ("router_completion_tokens", ()),
        "router_tokens": ("router_tokens", ()),
        "router_live_calls": ("router_live_calls", ()),
        "peak_lateral_jerk": ("peak_lateral_jerk", ()),
        "mean_lateral_jerk": ("mean_lateral_jerk", ()),
        "plant_peak_lateral_jerk": ("plant_peak_lateral_jerk", ()),
        "acceleration_tracking_rmse": ("acceleration_tracking_rmse", ()),
        "actuator_saturation_fraction": ("actuator_saturation_fraction", ()),
        "peak_sigma": ("peak_sigma", ()),
        "override_fraction": ("override_fraction", ()),
        "pending_ticks": ("pending_ticks", ()),
        "safety_events": ("safety_events", ()),
        "infeasible_ticks": ("infeasible_ticks", ()),
        "max_pending_s": ("max_pending_s", ()),
        "min_braking_h": ("min_braking_h", ()),
        "braking_infeasible_ticks": ("braking_infeasible_ticks", ()),
        "out_of_road": ("out_of_road", ()),
        "safety_abort": ("safety_abort", ()),
    }
    for name, (key, aliases) in summary_fields.items():
        values = [
            number
            for number in (_record_number(record, key, aliases) for record in records)
            if number is not None
        ]
        out[name] = summarize(values)

    models = sorted(
        {
            str(model)
            for record in records
            for model in (record.get("router_models") or [])
            if str(model)
        }
    )
    out["router_models"] = models
    total_calls = sum(
        int(_record_number(record, "router_calls", ("call_attempts",)) or 0.0)
        for record in records
    )
    live_calls = sum(
        int(_record_number(record, "router_live_calls") or 0.0) for record in records
    )
    out["router_live_fraction"] = (
        float(live_calls) / float(total_calls) if total_calls else 0.0
    )

    crash_values = [1.0 if _record_incident(record, "crashed") else 0.0 for record in records]
    freeze_values = [1.0 if _record_incident(record, "freeze") else 0.0 for record in records]
    survival_values = [
        1.0 if _record_bool(record, "pending_survival", True) else 0.0 for record in records
    ]
    out["crash_count"] = int(sum(crash_values))
    out["freeze_count"] = int(sum(freeze_values))
    out["crash_rate"] = float(sum(crash_values) / len(records)) if records else 0.0
    out["freeze_rate"] = float(sum(freeze_values) / len(records)) if records else 0.0
    out["crash_pct"] = summarize(100.0 * value for value in crash_values)
    out["freeze_pct"] = summarize(100.0 * value for value in freeze_values)
    out["pending_survival_rate"] = (
        float(sum(survival_values) / len(records)) if records else 1.0
    )

    barrier = [
        number
        for number in (_finite_float(record.get("min_barrier_h")) for record in records)
        if number is not None
    ]
    out["min_barrier_h"] = summarize(barrier) if barrier else None

    end_reason_counts: Dict[str, int] = {}
    for record in records:
        reason = str(record.get("end_reason") or "unknown")
        end_reason_counts[reason] = end_reason_counts.get(reason, 0) + 1
    out["end_reason"] = {key: end_reason_counts[key] for key in sorted(end_reason_counts)}
    return json_safe(out)


def aggregate_rollouts(
    rows: Iterable[Any],
    *,
    group_by: Sequence[str] = ("scenario", "mode"),
) -> Dict[str, Any]:
    """Aggregate episode rows (dicts or ``RolloutResult`` objects).

    The default return shape is ``aggregates[scenario][mode] = metrics``.
    Passing ``group_by=("mode",)`` returns ``aggregates[mode] = metrics``.
    """

    group_keys = tuple(group_by)
    if not group_keys:
        raise ValueError("group_by must contain at least one key")

    records: List[Dict[str, Any]] = []
    for row in rows:
        record = dict(row) if isinstance(row, Mapping) else result_to_episode_record(row)
        records.append(record)

    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for record in records:
        key = tuple(str(record.get(name) or "") for name in group_keys)
        groups.setdefault(key, []).append(record)

    aggregates: Dict[str, Any] = {}
    for key in sorted(groups):
        node = aggregates
        for part in key[:-1]:
            node = node.setdefault(part, {})
        node[key[-1]] = _aggregate_group(groups[key])
    return aggregates


__all__ = [
    "ablation_table",
    "aggregate_rollouts",
    "json_safe",
    "result_to_episode_record",
    "result_to_row",
    "summarize",
]
