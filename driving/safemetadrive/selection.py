"""Select one baseline checkpoint and inference mode per algorithm on development maps."""

import re
from collections import defaultdict
from typing import Any

import numpy as np

from driving.safemetadrive.protocol import SPLITS, environment_config


def select_baselines(reports):
    start, count = SPLITS["validation"]
    expected_maps = set(range(start, start+count))
    candidates: dict[tuple[str,int,bool], dict[int,list[dict[str,Any]]]] = defaultdict(lambda: defaultdict(list))
    checkpoints: dict[tuple[tuple[str,int,bool], int], str] = {}
    for report in reports:
        if report["split"] != "validation" or report["config"] != environment_config("validation"):
            raise ValueError("Baseline selection requires the reserved development configuration")
        if report["policy"] != "native_learned":
            raise ValueError("Expected native baseline actor reports")
        parameters = report["parameters"]
        checkpoint = parameters["checkpoint"]
        match = re.search(r"/(PPOLag|CPO)-seed(\d+)/.*epoch-(\d+)\.pt$", checkpoint)
        if not match:
            raise ValueError(f"Unrecognized baseline artifact: {checkpoint}")
        algorithm, seed, epoch = match.group(1), int(match.group(2)), int(match.group(3))
        key = (algorithm, epoch, bool(parameters["deterministic"]))
        candidates[key][seed].extend(report["episodes"])
        if (key, seed) in checkpoints and checkpoints[key, seed] != checkpoint:
            raise ValueError("One candidate cannot combine different model artifacts")
        checkpoints[key, seed] = checkpoint

    selection: dict[str,Any] = {}
    table: list[dict[str,Any]] = []
    for algorithm in ("PPOLag", "CPO"):
        required = {(algorithm, epoch, deterministic) for epoch in (50,100,150,200,250)
                    for deterministic in (True,False)}
        if {key for key in candidates if key[0] == algorithm} != required:
            raise ValueError(f"{algorithm}: expected all five checkpoints and both inference modes")
        choices: list[dict[str,Any]] = []
        for key in sorted(required):
            replicas = candidates[key]
            if set(replicas) != {0,1,2}:
                raise ValueError(f"{key}: expected three independent training seeds")
            metrics = []
            for seed in range(3):
                rows = replicas[seed]
                if len(rows) != count or {row["seed"] for row in rows} != expected_maps:
                    raise ValueError(f"{key}, seed{seed}: development maps must appear exactly once")
                metrics.append(dict(seed=seed,
                    success_rate=float(np.mean([row["success"] for row in rows])),
                    mean_cost=float(np.mean([row["cost"] for row in rows])),
                    mean_reward=float(np.mean([row["reward"] for row in rows]))))
            row = dict(algorithm=algorithm, epoch=key[1], transitions=key[1]*4000,
                       deterministic=key[2], replicas=metrics,
                       checkpoints={str(seed): checkpoints[key, seed] for seed in range(3)},
                       **{metric: float(np.mean([m[metric] for m in metrics]))
                          for metric in ("success_rate", "mean_cost", "mean_reward")})
            choices.append(row)
        # A common choice across training seeds avoids selecting a different
        # favorable checkpoint/mode independently for every trained replica.
        selection[algorithm] = max(choices, key=lambda row: (
            row["success_rate"], -row["mean_cost"], row["mean_reward"],
            -row["epoch"], row["deterministic"]))
        table.extend(choices)
    return dict(split="validation", selection=selection, candidates=table,
                criterion="Maximize mean destination success across three seeds; ties minimize "
                          "native cost, maximize return, prefer earlier checkpoints, then mean actions. "
                          "Use one checkpoint epoch and inference mode per algorithm.")
