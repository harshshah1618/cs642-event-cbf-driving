"""Frozen simulator configuration and disjoint map partitions."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

METADRIVE_REVISION = "fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1"
SPLITS = {"train": (0, 100), "validation": (100, 50), "test": (500, 50)}


def environment_config(split="validation"):
    start, count = SPLITS[split]
    return dict(
        start_seed=start, num_scenarios=count, map=3, horizon=1000,
        accident_prob=0.8, traffic_density=0.05,
        crash_vehicle_done=False, crash_object_done=False,
        use_render=False, log_level=50,
    )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def manifest(split, policy, parameters):
    return dict(
        simulator_revision=METADRIVE_REVISION,
        protocol="safemetadrive-current-v1",
        config=environment_config(split), split=split, policy=policy,
        parameters=parameters, parameter_sha256=digest(parameters),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(Path(__file__).parent.glob("*.py"))},
        installed_packages={name: importlib.metadata.version(name)
                            for name in ("metadrive-simulator", "numpy", "gymnasium", "panda3d")},
        note="Current pinned simulator; historical paper tables are context, not matched reruns.",
    )
