"""Strict scenario registry — all eval scenarios go through the same stack."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping

from config import SCENARIO_NAMES, ScenarioConfig
from env_dense import DenseHighwayEnv
from env_merge import MergeScenarioEnv
from env_two_trap import TwoTrapGauntletEnv
from errors import ConfigError

SCENARIO_IDS = tuple(SCENARIO_NAMES)

_ENV_CLASSES = {
    "two_trap": TwoTrapGauntletEnv,
    "merge": MergeScenarioEnv,
    "dense_highway": DenseHighwayEnv,
}

_SCENARIO_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "two_trap": {
        "route_length": 200.0,
        "duration": 12.0,
        "randomize": False,
        "params": copy.deepcopy(TwoTrapGauntletEnv.DEFAULT_PARAMS),
    },
    "merge": {
        "route_length": 180.0,
        "duration": 14.0,
        "randomize": False,
        "params": copy.deepcopy(MergeScenarioEnv.DEFAULT_PARAMS),
    },
    "dense_highway": {
        "route_length": 220.0,
        "duration": 16.0,
        "n_vehicles": DenseHighwayEnv.DEFAULT_N_VEHICLES,
        "randomize": False,
        "params": copy.deepcopy(DenseHighwayEnv.DEFAULT_PARAMS),
    },
}


def _canonical_name(name: str) -> str:
    if not isinstance(name, str):
        raise KeyError(f"Unknown scenario: {name!r}. Choose from {list(SCENARIO_IDS)}")
    key = name.strip().lower().replace("-", "_")
    # Strict lookup: no prefix matching and no legacy aliases.
    if key not in _ENV_CLASSES:
        raise KeyError(f"Unknown scenario: {name!r}. Choose from {list(SCENARIO_IDS)}")
    return key


def list_scenarios() -> List[str]:
    """Return the canonical scenario names accepted by :func:`make_env`."""

    return list(SCENARIO_NAMES)


def scenario_defaults(name: str) -> Dict[str, Any]:
    """Return a copy of the constructor defaults for a canonical scenario."""

    key = _canonical_name(name)
    return copy.deepcopy(_SCENARIO_DEFAULTS[key])


def make_env(
    name: str,
    config: ScenarioConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
):
    """Build a scenario environment.

    The signature remains backward compatible::

        make_env("two_trap", route_length=..., duration=..., seed=...)

    A :class:`~config.ScenarioConfig` can be supplied either
    positionally or by keyword.  Explicit keyword arguments override values
    from the config.  Scenario names are strict (canonical names only).
    """

    key = _canonical_name(name)
    if isinstance(config, Mapping):
        config = ScenarioConfig(**dict(config))
    if isinstance(config, ScenarioConfig) and config.simulator == "metadrive":
        from env_physics import MetaDriveAdapter

        config.validate()
        if config.name != key:
            raise ConfigError(f"config.name={config.name!r} does not match requested scenario {key!r}")
        if kwargs:
            raise ConfigError("MetaDrive overrides belong in ScenarioConfig, not extra kwargs")
        return MetaDriveAdapter(config)
    if config is not None:
        if isinstance(config, Mapping):
            try:
                config = ScenarioConfig(**dict(config))
            except TypeError as exc:
                raise ConfigError(f"invalid ScenarioConfig mapping: {exc}") from exc
        if not isinstance(config, ScenarioConfig):
            raise ConfigError(
                f"config: expected ScenarioConfig, got {type(config).__name__}"
            )
        config.validate()
        if config.name != key:
            raise ConfigError(
                f"config.name={config.name!r} does not match requested scenario {key!r}"
            )
        config_kwargs: Dict[str, Any] = {
            "seed": config.seed,
            "randomize": config.randomize,
        }
        if config.route_length is not None:
            config_kwargs["route_length"] = config.route_length
        if config.duration is not None:
            config_kwargs["duration"] = config.duration
        merged_params = None
        if config.params:
            merged_params = copy.deepcopy(dict(config.params))
            if isinstance(kwargs.get("params"), Mapping):
                merged_params.update(copy.deepcopy(dict(kwargs["params"])))
        config_kwargs.update(kwargs)
        if merged_params is not None:
            config_kwargs["params"] = merged_params
        kwargs = config_kwargs
    return _ENV_CLASSES[key](**kwargs)
