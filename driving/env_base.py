"""Environment contract, validation helpers, and deterministic sampling utils.

Every scenario in :mod:`environments` implements the small
:class:`DrivingEnv` protocol below.  Keeping the contract explicit makes the
stack simulator-agnostic: the native scenarios, highway-env wrappers, and the
CARLA integration skeleton are interchangeable from the controller's point of
view.

Two invariants are enforced here:

* actions are two-dimensional, finite ``[ax, ay]`` vectors;
* snapshots contain only finite vehicle state and are detached from live
  environment state before being returned.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from driving.env_types import VehicleState, WorldSnapshot
from driving.errors import ValidationError


@runtime_checkable
class DrivingEnv(Protocol):
    """Minimal Gymnasium-style contract used by the whole control stack."""

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and return ``(observation, info)``."""

    def step(
        self, action: np.ndarray
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Apply ``[ax, ay]`` and return the Gymnasium five-tuple."""

    def snapshot(self) -> WorldSnapshot:
        """Return a detached, validated world snapshot."""

    def close(self) -> None:
        """Release resources (a no-op for the lightweight native scenarios)."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name}: expected a number, got a boolean")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValidationError(f"{name}: expected a finite number, got {value!r}")
    return out


def validate_vehicle(state: VehicleState, *, name: str = "vehicle") -> VehicleState:
    """Validate one vehicle state, raising :class:`ValidationError` on failure."""

    if not isinstance(state, VehicleState):
        raise ValidationError(f"{name}: expected VehicleState, got {type(state).__name__}")
    for field_name in (
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "length",
        "width",
        "speed_limit",
    ):
        _finite(getattr(state, field_name), f"{name}.{field_name}")
    if float(state.length) <= 0.0:
        raise ValidationError(f"{name}.length: must be > 0, got {state.length!r}")
    if float(state.width) <= 0.0:
        raise ValidationError(f"{name}.width: must be > 0, got {state.width!r}")
    if float(state.speed_limit) < 0.0:
        raise ValidationError(
            f"{name}.speed_limit: must be >= 0, got {state.speed_limit!r}"
        )
    return state


def validate_action(
    action: Any,
    *,
    low: Optional[Sequence[float]] = None,
    high: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Validate an ``[ax, ay]`` action and return it as a float ndarray.

    Shape and finiteness are always checked.  Optional ``low``/``high`` bounds
    can be supplied by callers that want the full action box enforced.
    """

    try:
        arr = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"action: expected a numeric array, got {action!r}") from exc
    if arr.shape != (2,):
        raise ValidationError(f"action: expected shape (2,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"action: contains non-finite values: {arr.tolist()}")
    if low is not None:
        low_arr = np.asarray(low, dtype=np.float64)
        if low_arr.shape != (2,):
            raise ValidationError(f"action low bound: expected shape (2,), got {low_arr.shape}")
        if np.any(arr < low_arr):
            raise ValidationError(f"action: {arr.tolist()} below lower bound {low_arr.tolist()}")
    if high is not None:
        high_arr = np.asarray(high, dtype=np.float64)
        if high_arr.shape != (2,):
            raise ValidationError(
                f"action high bound: expected shape (2,), got {high_arr.shape}"
            )
        if np.any(arr > high_arr):
            raise ValidationError(
                f"action: {arr.tolist()} above upper bound {high_arr.tolist()}"
            )
    return arr


def validate_snapshot(snapshot: WorldSnapshot) -> WorldSnapshot:
    """Validate a world snapshot and return it unchanged."""

    if not isinstance(snapshot, WorldSnapshot):
        raise ValidationError(
            f"snapshot: expected WorldSnapshot, got {type(snapshot).__name__}"
        )
    _finite(snapshot.t, "snapshot.t")
    if float(snapshot.t) < 0.0:
        raise ValidationError(f"snapshot.t: must be >= 0, got {snapshot.t!r}")
    route_length = _finite(snapshot.route_length, "snapshot.route_length")
    if route_length <= 0.0:
        raise ValidationError(
            f"snapshot.route_length: must be > 0, got {snapshot.route_length!r}"
        )
    validate_vehicle(snapshot.ego, name="snapshot.ego")
    if not isinstance(snapshot.others, (list, tuple)):
        raise ValidationError(
            f"snapshot.others: expected a list, got {type(snapshot.others).__name__}"
        )
    for i, other in enumerate(snapshot.others):
        validate_vehicle(other, name=f"snapshot.others[{i}]")
    if not isinstance(snapshot.info, Mapping):
        raise ValidationError(
            f"snapshot.info: expected a mapping, got {type(snapshot.info).__name__}"
        )
    return snapshot


def detached_snapshot(
    *,
    t: float,
    ego: VehicleState,
    others: Sequence[VehicleState],
    route_length: float,
    done: bool = False,
    crashed: bool = False,
    info: Optional[Mapping[str, Any]] = None,
) -> WorldSnapshot:
    """Build a validated snapshot whose vehicles are independent copies."""

    snap = WorldSnapshot(
        t=float(t),
        ego=replace(ego),
        others=[replace(other) for other in others],
        route_length=float(route_length),
        done=bool(done),
        crashed=bool(crashed),
        info=copy.deepcopy(dict(info or {})),
    )
    return validate_snapshot(snap)


def copy_snapshot(snapshot: WorldSnapshot) -> WorldSnapshot:
    """Return a detached copy of an existing snapshot."""

    return snapshot.copy()


def normalize_range(value: Any, *, name: str) -> Tuple[float, float]:
    """Normalize a scalar or ``(low, high)`` pair into a finite range."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes, Mapping)):
        raise ValidationError(f"{name}: expected a number or (low, high) pair")
    if np.isscalar(value):
        out = _finite(value, name)
        return (out, out)
    try:
        seq = list(value)
    except TypeError as exc:
        raise ValidationError(
            f"{name}: expected a number or (low, high) pair, got {value!r}"
        ) from exc
    if len(seq) != 2:
        raise ValidationError(f"{name}: expected exactly two bounds, got {len(seq)}")
    low = _finite(seq[0], f"{name}[0]")
    high = _finite(seq[1], f"{name}[1]")
    if low > high:
        raise ValidationError(f"{name}: low bound {low} must be <= high bound {high}")
    return (low, high)


def validate_lateral_bounds(
    bounds: Optional[Sequence[float]], *, name: str = "lateral_bounds"
) -> Optional[Tuple[float, float]]:
    """Validate optional road lateral limits."""

    if bounds is None:
        return None
    low, high = normalize_range(bounds, name=name)
    if low >= high:
        raise ValidationError(f"{name}: low bound must be strictly below high bound")
    return (low, high)


def sample_range(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    """Draw one deterministic sample from ``bounds`` (inclusive for a point)."""

    low, high = bounds
    if low == high:
        return float(low)
    return float(rng.uniform(low, high))


def require_positive(value: Any, *, name: str) -> float:
    """Validate a strictly positive finite scalar."""

    out = _finite(value, name)
    if out <= 0.0:
        raise ValidationError(f"{name}: must be > 0, got {out}")
    return out


def require_non_negative(value: Any, *, name: str) -> float:
    """Validate a finite non-negative scalar."""

    out = _finite(value, name)
    if out < 0.0:
        raise ValidationError(f"{name}: must be >= 0, got {out}")
    return out
