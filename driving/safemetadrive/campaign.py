"""Freeze development-selected artifacts and construct the untouched-map study."""

import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from driving.safemetadrive.camera_planner import BASE_MODEL_REVISION
from driving.safemetadrive.protocol import METADRIVE_REVISION, environment_config
from driving.safemetadrive.selection import select_baselines


def read_reports(root, run):
    summaries = json.loads((root/run/"summaries.json").read_text())
    return [json.loads((root/f"{tag}.json").read_text()) for tag, _ in summaries]


def camera_development(root, run, through_step):
    reports = read_reports(root, run+"/development")
    replicas: dict[int,list[dict[str,Any]]] = {seed: [] for seed in range(3)}
    for report in reports:
        if report["split"] != "validation" or report["config"] != environment_config("validation"):
            raise ValueError("Camera selection requires development maps only")
        if report["parameters"]["extra_delay"] != 0 or not report["parameters"]["protection"]:
            raise ValueError("Camera selection requires the standard protected configuration")
        if not report["parameters"].get("lane_action_mask"):
            raise ValueError("Camera selection requires the current lane-action decoder")
        match = re.fullmatch(r"seed([012])-through"+str(through_step), Path(report["parameters"]["adapter"]).name)
        if not match or Path(report["parameters"]["adapter"]).parent.parent.name != run:
            raise ValueError("Camera report does not use the declared immutable snapshot")
        replicas[int(match.group(1))].extend(report["episodes"])
    for seed, rows in replicas.items():
        if len(rows) != 50 or {row["seed"] for row in rows} != set(range(100,150)):
            raise ValueError(f"Camera seed{seed} lacks complete development coverage")
    rows = [row for replica in replicas.values() for row in replica]
    return dict(run=run, through_step=through_step,
                success_rate=float(np.mean([row["success"] for row in rows])),
                mean_cost=float(np.mean([row["cost"] for row in rows])),
                mean_reward=float(np.mean([row["reward"] for row in rows])),
                per_seed_success=[float(np.mean([row["success"] for row in replicas[seed]])) for seed in range(3)])


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_final(root, run):
    root = Path(root)
    destination = root/run/"frozen-plan.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        payload = {key:value for key,value in previous.items() if key != "plan_sha256"}
        if hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest() != previous["plan_sha256"]:
            raise ValueError("Frozen plan content no longer matches its digest")
        for path,digest in previous["artifact_sha256"].items():
            if file_digest(path) != digest:
                raise ValueError(f"Frozen artifact changed: {path}")
        return previous
    if not json.loads((root/"camera-config-parity.json").read_text())["matched"]:
        raise ValueError("Camera/native environment parity must pass before final testing")
    baseline = select_baselines(read_reports(root,"native-baseline-checkpoints-development-v1"))
    cameras = [camera_development(root,"camera-balanced-v2",600),
               camera_development(root,"camera-recovery-v5",1000)]
    # One common camera family across training seeds; ties prefer the earlier
    # family after success, cost and return, avoiding per-seed cherry-picking.
    camera = max(cameras, key=lambda row: (row["success_rate"], -row["mean_cost"], row["mean_reward"]))
    plan: dict[str,Any] = dict(protocol="safemetadrive-current-v1", split="test", seeds=list(range(500,550)),
                frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                simulator_revision=METADRIVE_REVISION, base_model_revision=BASE_MODEL_REVISION,
                baseline_selection=baseline, camera_selection=camera, camera_candidates=cameras,
                camera_criterion="Common family across three seeds: maximize development destination success, "
                                 "then lower native cost, higher return, earlier family.",
                cpu_specs=[], camera_specs=[], groups={}, artifact_sha256={})
    groups = plan["groups"]
    artifacts: set[str] = set()

    def cpu(label, replica, policy, parameters):
        tag = f"{run}/episodes/{label}-seed{replica}"
        plan["cpu_specs"].append((tag,policy,parameters,"test",list(range(500,550))))
        groups.setdefault(label,[]).append([str(root/f"{tag}.json")])
        if "skill_checkpoint" in parameters:
            artifacts.add(parameters["skill_checkpoint"])
        if "checkpoint" in parameters:
            artifacts.add(parameters["checkpoint"])

    cpu("IDM",0,"idm",{})
    cpu("Rule",0,"lane_caution_rule",dict(speed=30.0))
    for seed in range(3):
        checkpoint = str(root/f"skills-1m-v1/skills-seed{seed}/ppo.zip")
        cpu("Rule-PPO",seed,"lane_caution_rule",dict(speed=30.0,skill_checkpoint=checkpoint))
        cloned = str(root/f"skills-1m-v1/skills-seed{seed}/cloned.zip")
        cpu("Rule-BC",seed,"lane_caution_rule",dict(speed=30.0,skill_checkpoint=cloned))
        for careful in (False,True):
            label = "Fixed-careful" if careful else "Fixed-normal"
            cpu(label,seed,"lane_caution_rule",dict(speed=30.0,skill_checkpoint=checkpoint,
                planner="external",initial_careful=careful))
        for algorithm in ("PPOLag","CPO"):
            selection = baseline["selection"][algorithm]
            cpu(algorithm,seed,"native_learned",dict(checkpoint=selection["checkpoints"][str(seed)],
                deterministic=selection["deterministic"],policy_seed=1000003*seed))
        adapter = str(root/camera["run"]/"snapshots"/f"seed{seed}-through{camera['through_step']}")
        snapshot = json.loads((Path(adapter)/"snapshot.json").read_text())
        adapter_file = str(Path(adapter)/"adapter_model.safetensors")
        if file_digest(adapter_file) != snapshot["sha256"]:
            raise ValueError("Selected camera adapter differs from its immutable snapshot")
        artifacts.add(adapter_file)
        for delay, protection in [(d,True) for d in (0.0,0.5,1.0,2.0,4.0)]+[(0.0,False)]:
            label = f"Camera-delay{delay:g}" if protection else "Camera-unprotected"
            paths = []
            for start in (500,525):
                tag = f"{run}/episodes/{label}-seed{seed}-{start}"
                plan["camera_specs"].append((tag,adapter,checkpoint,"test",list(range(start,start+25)),delay,protection))
                paths.append(str(root/f"{tag}.json"))
            groups.setdefault(label,[]).append(paths)
    plan["artifact_sha256"] = {path: file_digest(path) for path in sorted(artifacts)}
    source = Path(__file__).resolve().parents[2]
    plan["source_sha256"] = {str(path.relative_to(source)):file_digest(path)
        for path in sorted([*(source/"driving"/"safemetadrive").glob("*.py"),
                            *(source/"tools").glob("safemetadrive*.py")])}
    encoded_plan = json.dumps(plan,sort_keys=True).encode()
    plan["plan_sha256"] = hashlib.sha256(encoded_plan).hexdigest()
    destination.parent.mkdir(parents=True,exist_ok=True)
    # A single remote coordinator owns this immutable study directory.
    temporary = destination.with_suffix(".writing")
    with temporary.open("w") as stream:
        json.dump(plan,stream,indent=2)
    temporary.replace(destination)
    return plan


def completed_spec(root, spec, camera=False):
    path = Path(root)/f"{spec[0]}.json"
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    seeds = spec[4]
    if "summary" not in report:
        return False
    if (report["split"] != "test" or len(report["episodes"]) != len(seeds) or
            {row["seed"] for row in report["episodes"]} != set(seeds)):
        raise ValueError("Existing study report disagrees with the frozen plan")
    if camera:
        expected = dict(adapter=spec[1],skill_checkpoint=spec[2],extra_delay=spec[5],protection=spec[6])
        if any(report["parameters"][key] != value for key,value in expected.items()):
            raise ValueError("Existing camera report uses different frozen parameters")
    elif report["policy"] != spec[1] or report["parameters"] != spec[2]:
        raise ValueError("Existing actor report uses different frozen parameters")
    return True
