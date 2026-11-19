"""Bounded, checkpointed SafeMetaDrive cloud experiments.

Run: .venv/bin/modal run --detach tools/safemetadrive_modal.py --stage pilot
"""

import json
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("cs642-safemetadrive")
volume = modal.Volume.from_name("cs642-safemetadrive", create_if_missing=True)
hf_volume = modal.Volume.from_name("vlm-router-hf")
def simulator_image(python_version):
    return (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install("git", "libgl1", "libegl1", "libglib2.0-0")
        .pip_install("numpy==1.26.4", "gymnasium==0.29.1", "pillow",
                     "metadrive-simulator @ git+https://github.com/metadriverse/metadrive.git@fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1")
        .run_commands("python -m metadrive.pull_asset")
        .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    )


base_image = simulator_image("3.11")


def sources(img):
    return (img.add_local_dir(ROOT / "driving" / "safemetadrive", "/workspace/driving/safemetadrive",
                              ignore=["**/__pycache__", "**/*.pyc"])
            .add_local_file(ROOT / "tools" / "safemetadrive_eval.py", "/workspace/tools/safemetadrive_eval.py")
            .add_local_file(ROOT / "tools" / "safemetadrive_analyze.py", "/workspace/tools/safemetadrive_analyze.py")
            .add_local_file(ROOT / "tools" / "safemetadrive_train_skills.py", "/workspace/tools/safemetadrive_train_skills.py")
            .add_local_file(ROOT / "tools" / "safemetadrive_camera_data.py", "/workspace/tools/safemetadrive_camera_data.py")
            .add_local_file(ROOT / "tools" / "safemetadrive_train_camera.py", "/workspace/tools/safemetadrive_train_camera.py")
            .add_local_file(ROOT / "tools" / "safemetadrive_camera_eval.py", "/workspace/tools/safemetadrive_camera_eval.py")
            .add_local_file(ROOT / "tools" / "safemetadrive_train.py", "/workspace/tools/safemetadrive_train.py"))


image = sources(base_image)
training_image = sources(
    base_image
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gymnasium==0.28.1", "mujoco==2.3.3", "xmltodict", "tensorboard",
                 "wandb", "pyyaml", "moviepy==1.0.3", "typer", "seaborn",
                 "pandas==2.0.3", "gdown")
    .pip_install("safety-gymnasium==1.0.0", "gymnasium-robotics==1.2.2", "omnisafe==0.5.0", extra_options="--no-deps")
    .pip_install("pettingzoo==1.24.3", "scipy==1.11.4", extra_options="--no-deps")
    .add_local_file(ROOT / "tools" / "safemetadrive_dependency_compat.py", "/opt/dependency_compat.py", copy=True)
    .run_commands("python /opt/dependency_compat.py")
    .pip_install("stable-baselines3==2.3.2")
    .run_commands("python -c 'import omnisafe; import stable_baselines3'")
)
camera_training_image = sources(
    base_image.pip_install("numpy==1.26.4", "torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.1",
                           "peft==0.17.1", "accelerate==1.10.1", "stable-baselines3==2.3.2")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false", "NVIDIA_DRIVER_CAPABILITIES": "all",
          "P3D_DISPLAY": "p3headlessgl"})
)


@app.function(image=image, volumes={"/results": volume}, cpu=2, memory=4096,
              timeout=3600, max_containers=12)
def evaluate_shard(spec):
    import sys
    sys.path.insert(0, "/workspace/tools")
    from safemetadrive_eval import evaluate

    tag, policy, parameters, split, seeds = spec
    result = evaluate(policy, parameters, split, seeds, f"/results/{tag}.json")
    volume.commit()
    return tag, result["summary"]


@app.function(image=training_image, volumes={"/results": volume}, cpu=2, memory=8192,
              timeout=3600, max_containers=12)
def evaluate_learned_shard(spec):
    import sys
    sys.path.insert(0, "/workspace/tools")
    from safemetadrive_eval import evaluate

    volume.reload()
    tag, policy, parameters, split, seeds = spec
    result = evaluate(policy, parameters, split, seeds, f"/results/{tag}.json")
    volume.commit()
    return tag, result["summary"]


@app.function(image=training_image, volumes={"/results": volume}, cpu=2, memory=8192, timeout=900)
def audit_native_actor(seed=2):
    import sys
    sys.path.insert(0, "/workspace")
    import numpy as np
    import torch
    from driving.safemetadrive.policies import NativeLearnedDriver
    from driving.safemetadrive.protocol import environment_config
    from metadrive import SafeMetaDriveEnv
    from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor

    volume.reload()
    root = Path(f"/results/baselines-1m-v1/PPOLag-seed{seed}")
    checkpoint = max(root.rglob("epoch-250.pt"), key=lambda p: p.stat().st_mtime)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["pi"]
    report = dict(checkpoint=str(checkpoint), action_std=state["log_std"].exp().tolist(), episodes=[])
    torch.set_num_threads(2)
    for split, seeds in [("train", [0,1,2]), ("validation", [100,101,102])]:
        env = SafeMetaDriveEnv(environment_config(split))
        try:
            reference = GaussianLearningActor(env.observation_space, env.action_space, [256,256], activation="tanh")
            reference.load_state_dict(state)
            reference.eval()
            for deterministic in [True, False]:
                for map_seed in seeds:
                    obs, _ = env.reset(seed=map_seed)
                    torch.manual_seed(1000+map_seed)
                    manual = NativeLearnedDriver(str(checkpoint), obs)
                    reward = cost = difference = 0.0
                    while True:
                        tensor = torch.as_tensor(obs, dtype=torch.float32)
                        with torch.inference_mode():
                            expected = reference.predict(tensor, deterministic=True).numpy()
                            difference = max(difference, float(np.max(np.abs(manual.act()-expected))))
                            action = reference.predict(tensor, deterministic=deterministic).numpy()
                        obs, r, tm, tc, info = env.step(action)
                        manual.on_observation(obs)
                        reward += r
                        cost += info["cost"]
                        if tm or tc:
                            break
                    row = dict(split=split, seed=map_seed, deterministic=deterministic,
                               reward=float(reward), cost=float(cost), success=bool(info["arrive_dest"]),
                               route_completion=float(info["route_completion"]), max_action_difference=difference)
                    print(json.dumps(row), flush=True)
                    report["episodes"].append(row)
        finally:
            env.close()
    out = Path(f"/results/native-actor-audit-seed{seed}.json")
    out.write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


def _train_baseline(spec):
    import subprocess
    import sys

    run, algorithm, seed, steps, *options = spec
    critic_lr = options[0] if options else 5e-5
    out = f"/results/{run}/{algorithm}-seed{seed}"
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(f"{out}/train.log", "w") as log:
        proc = subprocess.Popen([
            sys.executable, "/workspace/tools/safemetadrive_train.py",
            "--algorithm", algorithm, "--seed", str(seed), "--steps", str(steps),
            "--output", out,"--critic-lr",str(critic_lr)], stdout=log, stderr=subprocess.STDOUT)
        import time
        while proc.poll() is None:
            time.sleep(30)
            volume.commit()
    volume.commit()
    if proc.returncode:
        raise RuntimeError(Path(f"{out}/train.log").read_text()[-8000:])
    return json.loads(Path(f"{out}/manifest.json").read_text())


@app.function(image=training_image, volumes={"/results": volume}, cpu=8, memory=16384,
              timeout=6 * 3600, max_containers=6)
def train_baseline(spec):
    return _train_baseline(spec)


@app.function(image=training_image, volumes={"/results": volume}, cpu=4, memory=8192,
              timeout=6*3600, max_containers=4, nonpreemptible=True)
def train_baseline_reserved(spec):
    return _train_baseline(spec)


@app.function(image=training_image, volumes={"/results": volume}, cpu=8, memory=16384,
              timeout=6 * 3600, max_containers=3)
def train_skills(spec):
    import subprocess
    import sys
    import time

    run, seed, steps = spec
    out = f"/results/{run}/skills-seed{seed}"
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(f"{out}/train.log", "w") as log:
        proc = subprocess.Popen([sys.executable, "/workspace/tools/safemetadrive_train_skills.py",
                                 "--seed", str(seed), "--steps", str(steps),
                                 "--demos", "400" if steps < 10000 else "24000", "--output", out],
                                stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(30)
            volume.commit()
    volume.commit()
    if proc.returncode:
        raise RuntimeError(Path(f"{out}/train.log").read_text()[-8000:])
    return json.loads(Path(f"{out}/manifest.json").read_text())


@app.function(image=sources(base_image.env({"NVIDIA_DRIVER_CAPABILITIES": "all", "P3D_DISPLAY": "p3headlessgl"})),
              volumes={"/results": volume}, gpu="T4", cpu=2, memory=8192,
              timeout=3600, max_containers=8)
def camera_data(spec):
    import sys
    sys.path.insert(0, "/workspace/tools")
    from safemetadrive_camera_data import generate

    run, start, count, max_steps = spec
    result = generate(list(range(start, start+count)), f"/results/{run}/data/shard-{start}", max_steps)
    volume.commit()
    return result


@app.function(image=sources(base_image.env({"NVIDIA_DRIVER_CAPABILITIES": "all", "P3D_DISPLAY": "p3headlessgl"})),
              volumes={"/results": volume}, gpu="T4", cpu=2, memory=8192,
              timeout=3600, max_containers=8)
def camera_recovery_data(spec):
    import sys
    sys.path.insert(0, "/workspace/tools")
    from safemetadrive_camera_data import generate_recovery

    run, start, count, delay = spec
    result = generate_recovery(list(range(start, start+count)),
                               f"/results/{run}/data/shard-{start}", delay=delay)
    volume.commit()
    return result


@app.function(image=sources(base_image.env({"NVIDIA_DRIVER_CAPABILITIES": "all", "P3D_DISPLAY": "p3headlessgl"})),
              volumes={"/results": volume}, gpu="T4", cpu=2, memory=8192, timeout=900)
def audit_camera_parity():
    import sys
    sys.path.insert(0,"/workspace")
    import numpy as np
    from driving.safemetadrive.camera_planner import camera_config
    from driving.safemetadrive.policies import LaneCautionDriver
    from driving.safemetadrive.protocol import environment_config
    from metadrive import SafeMetaDriveEnv

    traces: list[list[dict[str,Any]]] = []
    for rendered in (False,True):
        config = environment_config("train")
        env = SafeMetaDriveEnv(camera_config(config) if rendered else config)
        current = []
        try:
            for seed in (0,1):
                env.reset(seed=seed)
                driver = LaneCautionDriver(env.agent,speed=30.0)
                objects = sorted([type(o).__name__,*np.asarray(o.position).tolist()]
                                 for o in env.agent.lidar.get_surrounding_objects(env.agent))
                trajectory = []
                for _ in range(1000):
                    action = driver.act()
                    _,reward,terminated,truncated,info = env.step(action)
                    trajectory.append([*env.agent.position.tolist(),float(env.agent.speed),
                                       float(reward),float(info["cost"]),*map(float,action)])
                    if terminated or truncated:
                        break
                current.append(dict(seed=seed,objects=objects,trajectory=trajectory))
                driver.close()
        finally:
            env.close()
        traces.append(current)
    matched = all(a["objects"] == b["objects"] and len(a["trajectory"]) == len(b["trajectory"]) and
                  np.allclose(a["trajectory"],b["trajectory"],rtol=0,atol=1e-6)
                  for a,b in zip(*traces,strict=True))
    result = dict(matched=bool(matched),seeds=[0,1],absolute_tolerance=1e-6,traces=traces)
    Path("/results/camera-config-parity.json").write_text(json.dumps(result,indent=2))
    volume.commit()
    if not matched:
        raise RuntimeError("Camera configuration changed scene or driving dynamics")
    return dict(matched=True,seeds=[0,1],steps=[len(row["trajectory"]) for row in traces[0]])


@app.function(image=image, volumes={"/results": volume}, cpu=2, memory=4096,
              timeout=1800, max_containers=8)
def pack_camera_shard(start, dataset="camera-data-v1"):
    import concurrent.futures
    import hashlib
    import io
    import tarfile

    volume.reload()
    root = Path(f"/results/{dataset}/data/shard-{start}")
    files = sorted(root.iterdir())
    archive = io.BytesIO()
    def read(path):
        return path.name, path.read_bytes()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name, data in concurrent.futures.ThreadPoolExecutor(32).map(read, files):
            info = tarfile.TarInfo(f"shard-{start}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    content = archive.getvalue()
    out = Path(f"/results/{dataset}/archives")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"shard-{start}.tar").write_bytes(content)
    volume.commit()
    return dict(shard=start, files=len(files), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


@app.function(image=camera_training_image, volumes={"/results": volume, "/hf": hf_volume},
              gpu="L40S", cpu=4, memory=16384, timeout=4*3600, max_containers=3)
def train_camera(run, dataset="camera-data-v1", steps=600, seed=0,
                 sampling="sqrt", resume_source_run=None, mirror_augmentation=False):
    import concurrent.futures
    import shutil
    import sys
    import tarfile
    sys.path.insert(0, "/workspace/tools")
    from safemetadrive_train_camera import train

    volume.reload()
    local = Path("/tmp/camera-data")
    local.mkdir(parents=True, exist_ok=True)
    def copy(start):
        source = Path(f"/results/{dataset}/archives/shard-{start}.tar")
        destination = local / source.name
        shutil.copyfile(source, destination)
        return destination
    archives = list(concurrent.futures.ThreadPoolExecutor(8).map(copy, range(0, 100, 5)))
    for archive in archives:
        with tarfile.open(archive) as tar:
            tar.extractall(local, filter="data")
    print("Loaded packed camera dataset onto local disk", flush=True)
    source = f"/results/{resume_source_run}/camera-seed{seed}" if resume_source_run else None
    summary = train(str(local), f"/results/{run}/camera-seed{seed}", steps, seed,
                    commit=volume.commit, sampling=sampling, resume_source=source,
                    mirror_augmentation=mirror_augmentation)
    volume.commit()
    return {k: v for k, v in summary.items() if k != "source_data_manifests"}


@app.function(image=image, volumes={"/results": volume}, cpu=1, memory=2048, timeout=600)
def snapshot_camera_adapter(run, seed, through_step):
    import hashlib
    import shutil
    import uuid

    volume.reload()
    root = Path(f"/results/{run}")
    source = root / f"camera-seed{seed}"
    target = root / "snapshots" / f"seed{seed}-through{through_step}"
    if (target / "snapshot.json").exists():
        return json.loads((target / "snapshot.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    history = [row for row in summary["history"] if row["step"] > 0]
    if not history or max(row["step"] for row in history) != through_step:
        raise ValueError(f"Expected validation through step {through_step}; found {[r['step'] for r in history]}")
    best = max(history, key=lambda row: row["balanced_accuracy"])
    temporary = target.with_name(target.name+f"-tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(summary.get("best_adapter", str(source / "best-adapter"))), temporary)
    data = (temporary / "adapter_model.safetensors").read_bytes()
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8+size])
    expected = max(value["data_offsets"][1] for key, value in header.items() if key != "__metadata__")
    if len(data) != 8+size+expected:
        raise RuntimeError("Incomplete adapter snapshot")
    record = dict(seed=seed, validation_through_step=through_step, selected_step=best["step"],
                  validation=best, sha256=hashlib.sha256(data).hexdigest(), adapter=str(target),
                  base_model_revision=summary.get("base_model_revision"))
    (temporary / "snapshot.json").write_text(json.dumps(record, indent=2))
    temporary.rename(target)
    volume.commit()
    return record


@app.function(image=camera_training_image, volumes={"/results": volume, "/hf": hf_volume},
              gpu=["A10G","L40S"], cpu=4, memory=24576, timeout=4*3600, max_containers=6)
def evaluate_camera_shard(spec):
    import sys
    sys.path.insert(0, "/workspace/tools")
    from safemetadrive_camera_eval import evaluate_camera

    volume.reload()
    tag, adapter, checkpoint, split, seeds, delay, protection = spec
    result = evaluate_camera(adapter, f"/results/{tag}.json", seeds, checkpoint, delay,
                             split, protection=protection, commit=volume.commit)
    volume.commit()
    return tag, result["summary"]


def _save_summaries(run, summaries):
    root = ROOT / "runs" / "safemetadrive" if modal.is_local() else Path("/results")
    path = root / run / "summaries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, indent=2))
    if not modal.is_local():
        volume.commit()
    print(json.dumps(summaries, indent=2))
    return summaries


def run_stage(stage="pilot", run="pilot-v1", steps=1000000, camera_step=200, previous_call=""):
    # Resolve the active deployment at phase launch. Captured handles can point
    # to an older definition after a deployment while the coordinator waits.
    assert app.name is not None
    evaluate_shard = modal.Function.from_name(app.name,"evaluate_shard")
    evaluate_learned_shard = modal.Function.from_name(app.name,"evaluate_learned_shard")
    evaluate_camera_shard = modal.Function.from_name(app.name,"evaluate_camera_shard")
    train_camera = modal.Function.from_name(app.name,"train_camera")
    snapshot_camera_adapter = modal.Function.from_name(app.name,"snapshot_camera_adapter")
    camera_recovery_data = modal.Function.from_name(app.name,"camera_recovery_data")
    camera_data = modal.Function.from_name(app.name,"camera_data")
    pack_camera_shard = modal.Function.from_name(app.name,"pack_camera_shard")
    train_baseline_reserved = modal.Function.from_name(app.name,"train_baseline_reserved")
    train_baseline = modal.Function.from_name(app.name,"train_baseline")
    train_skills = modal.Function.from_name(app.name,"train_skills")
    specs: list[Any]
    if stage == "camera-masked-development":
        import time
        sources = [("camera-balanced-v2",600),("camera-recovery-v5",1000)]
        while True:
            volume.reload()
            records = []
            for source,_ in sources:
                for seed in range(3):
                    path = Path(f"/results/{source}/camera-seed{seed}/summary.json")
                    if path.exists():
                        records.append(json.loads(path.read_text()))
            if len(records) == 6 and all(record["status"] == "completed" for record in records):
                break
            time.sleep(15)
        if previous_call:
            call = modal.FunctionCall.from_id(previous_call)
            stale = {child.function_call_id for parent in call.get_call_graph() for child in parent.children
                     if child.function_name in ("snapshot_camera_adapter","evaluate_camera_shard")}
            call.cancel(terminate_containers=True)
            for identifier in stale:
                modal.FunctionCall.from_id(identifier).cancel()
        specs = []
        for source,through in sources:
            snapshots = list(snapshot_camera_adapter.starmap([(source,seed,through) for seed in range(3)]))
            specs.extend((f"{source}/masked-development/camera-skill{seed}-{start}",snapshots[seed]["adapter"],
                          f"/results/skills-1m-v1/skills-seed{seed}/ppo.zip","validation",
                          list(range(start,start+25)),0.0,True)
                         for seed in range(3) for start in (100,125))
        results = list(evaluate_camera_shard.map(specs))
        volume.reload()
        for source,_ in sources:
            original = Path(f"/results/{source}/development/summaries.json")
            backup = Path(f"/results/{source}/unmasked-development/summaries.json")
            if original.exists() and not backup.exists():
                backup.parent.mkdir(parents=True,exist_ok=True)
                backup.write_bytes(original.read_bytes())
            _save_summaries(source+"/development",[row for row in results if row[0].startswith(source+"/")])
        return _save_summaries(run,results)
    if stage == "cpo-critic-campaign":
        records = list(train_baseline_reserved.map([(run,"CPO",seed,steps,1e-3) for seed in range(3)]))
        _save_summaries(run+"/training",records)
        volume.reload()
        specs = []
        for seed in range(3):
            root = Path(f"/results/{run}/CPO-seed{seed}")
            for epoch in (50,100,150,200,250):
                checkpoint = max(root.rglob(f"epoch-{epoch}.pt"),key=lambda p:p.stat().st_mtime)
                for deterministic in (True,False):
                    mode = "mean" if deterministic else "stochastic"
                    specs.append((f"{run}/development/CPO-seed{seed}-epoch{epoch}-{mode}","native_learned",
                        dict(checkpoint=str(checkpoint),deterministic=deterministic,policy_seed=1000003*seed),
                        "validation",list(range(100,150))))
        results = list(evaluate_learned_shard.map(specs))
        _save_summaries(run+"/development",results)
        volume.reload()
        canonical = "native-baseline-checkpoints-development-v1"
        combined = dict(json.loads(Path(f"/results/{canonical}/summaries.json").read_text()))
        combined.update(dict(results))
        return _save_summaries(canonical,list(combined.items()))
    if stage in ("resume-balanced-development","resume-recovery-development"):
        import time
        source, through = (("camera-balanced-v2",600) if stage == "resume-balanced-development"
                           else ("camera-recovery-v5",1000))
        # Preserve in-flight training. Only replace its stale evaluation fan-out
        # after all three final optimizer snapshots have been committed.
        while True:
            volume.reload()
            records = []
            for seed in range(3):
                path = Path(f"/results/{source}/camera-seed{seed}/summary.json")
                if path.exists():
                    records.append(json.loads(path.read_text()))
            if len(records) == 3 and all(record["status"] == "completed" for record in records):
                break
            time.sleep(15)
        if previous_call:
            call = modal.FunctionCall.from_id(previous_call)
            stale = {child.function_call_id for parent in call.get_call_graph() for child in parent.children
                     if child.function_name in ("snapshot_camera_adapter","evaluate_camera_shard")}
            call.cancel(terminate_containers=True)
            for identifier in stale:
                modal.FunctionCall.from_id(identifier).cancel()
        snapshots = list(snapshot_camera_adapter.starmap([(source,seed,through) for seed in range(3)]))
        specs = [(f"{source}/development/camera-skill{seed}-{start}",snapshots[seed]["adapter"],
                  f"/results/skills-1m-v1/skills-seed{seed}/ppo.zip","validation",
                  list(range(start,start+25)),0.0,True)
                 for seed in range(3) for start in (100,125)]
        return _save_summaries(source+"/development",list(evaluate_camera_shard.map(specs)))
    if stage == "final-study":
        import sys
        sys.path.insert(0,"/workspace")
        sys.path.insert(0,"/workspace/tools")
        from driving.safemetadrive.campaign import completed_spec, freeze_final

        from safemetadrive_analyze import markdown, summarize

        volume.reload()
        plan = freeze_final("/results",run)
        volume.commit()  # Freeze selection and artifact hashes BEFORE any test episode.
        cpu = [spec for spec in plan["cpu_specs"] if not completed_spec("/results",spec)]
        if cpu:
            _save_summaries(run+"/cpu",list(evaluate_learned_shard.map(cpu)))
        volume.reload()
        camera = [spec for spec in plan["camera_specs"] if not completed_spec("/results",spec,camera=True)]
        if camera:
            _save_summaries(run+"/camera",list(evaluate_camera_shard.map(camera)))
        volume.reload()
        result = summarize(plan["groups"],split="test")
        result["frozen_plan_sha256"] = plan["plan_sha256"]
        out = Path("/results")/run
        (out/"analysis.json").write_text(json.dumps(result,indent=2))
        note = ("\nFast control uses privileged simulator geometry and object states. Camera planning uses RGB "
                "and declared ego/navigation state. Teacher data and Qwen pretraining add resources beyond "
                "the one-million-transition control training budget. These are matched-environment reruns, "
                "not an equal-information or historical state-of-the-art claim.\n\n"
                "Latency is simulated holding of the old intent for measured GPU inference plus injected delay; "
                "this is not an asynchronous real-time hardware demonstration. The protection is a sampled "
                "forecast and stopping cap, without a formal general-road safety certificate.\n")
        (out/"results.md").write_text(markdown(result)+note)
        volume.commit()
        return result
    if stage == "camera-recovery-data":
        results = list(camera_recovery_data.map([(run, start, 5, 2.0) for start in range(0,100,5)]))
        _save_summaries(run, results)
        return _save_summaries(run+"/packing", list(pack_camera_shard.starmap(
            [(start,run) for start in range(0,100,5)])))
    if stage == "camera-recovery-campaign":
        volume.reload()
        previous = json.loads(Path("/results/camera-balanced-v2/training/summaries.json").read_text())
        if len(previous) != 3 or any(record["status"] != "completed" for record in previous):
            raise ValueError("All three source camera trainings must complete before continuation")
        records = list(train_camera.starmap(
            [(run, "camera-recovery-data-v1", 1000, seed, "balanced", "camera-balanced-v2", True) for seed in range(3)]))
        _save_summaries(run+"/training", records)
        snapshots = list(snapshot_camera_adapter.starmap([(run, seed, 1000) for seed in range(3)]))
        specs = [(f"{run}/development/camera-skill{seed}-{start}", snapshots[seed]["adapter"],
                  f"/results/skills-1m-v1/skills-seed{seed}/ppo.zip", "validation",
                  list(range(start, start+25)), 0.0, True)
                 for seed in range(3) for start in (100,125)]
        return _save_summaries(run+"/development", list(evaluate_camera_shard.map(specs)))
    if stage == "disconnect-probe":
        import time
        time.sleep(15)
        return _save_summaries(run, [{"remote_probe": "completed_after_launcher_exit"}])
    if stage == "camera-balanced-campaign":
        records = list(train_camera.starmap(
            [(run, "camera-data-v1", 600, seed, "balanced", "camera-lora-v1") for seed in range(3)]))
        _save_summaries(run+"/training", records)
        snapshots = list(snapshot_camera_adapter.starmap([(run, seed, 600) for seed in range(3)]))
        specs = [(f"{run}/development/camera-skill{seed}-{start}", snapshots[seed]["adapter"],
                  f"/results/skills-1m-v1/skills-seed{seed}/ppo.zip", "validation",
                  list(range(start,start+5)), 0.0, True)
                 for seed in range(3) for start in range(100,150,5)]
        results = list(evaluate_camera_shard.map(specs))
        return _save_summaries(run+"/development", results)
    if stage == "baseline-finish":
        specs = [(run, alg, seed, steps) for alg, seed in
                 [("PPOLag", 1), ("CPO", 0), ("CPO", 1), ("CPO", 2)]]
        return _save_summaries(run, list(train_baseline_reserved.map(specs)))
    if stage == "baseline-checkpoint-development":
        volume.reload()
        specs = []
        for algorithm in ("PPOLag", "CPO"):
            for seed in range(3):
                source = ("baselines-1m-v1" if algorithm == "PPOLag" and seed in (0, 2)
                          else "baselines-completion-v2")
                root = Path(f"/results/{source}/{algorithm}-seed{seed}")
                if json.loads((root/"manifest.json").read_text())["status"] != "completed":
                    raise ValueError(f"{algorithm} seed {seed} has not completed")
                for epoch in (50,100,150,200):
                    checkpoint = max(root.rglob(f"epoch-{epoch}.pt"), key=lambda p: p.stat().st_mtime)
                    for deterministic in (True,False):
                        mode = "mean" if deterministic else "stochastic"
                        specs.append((f"{run}/{algorithm}-seed{seed}-epoch{epoch}-{mode}", "native_learned",
                                      dict(checkpoint=str(checkpoint), deterministic=deterministic,
                                           policy_seed=1000003*seed), "validation", list(range(100,150))))
        # Final epoch250 reports already cover all maps in the earlier run.
        summaries = json.loads(Path("/results/native-baseline-all-development-v1/summaries.json").read_text())
        summaries.extend(evaluate_learned_shard.map(specs))
        return _save_summaries(run,summaries)
    if stage == "baseline-all-development":
        volume.reload()
        specs = []
        # Retain the already completed two-seed inference-mode comparison.
        previous = Path("/results/native-baseline-modes-development-v1/summaries.json")
        summaries = json.loads(previous.read_text())
        for algorithm in ("PPOLag", "CPO"):
            for seed in range(3):
                if algorithm == "PPOLag" and seed in (0, 2):
                    continue
                source = ("baselines-1m-v1" if algorithm == "PPOLag" and seed in (0, 2)
                          else "baselines-completion-v2")
                root = Path(f"/results/{source}/{algorithm}-seed{seed}")
                record = json.loads((root / "manifest.json").read_text())
                if record["status"] != "completed":
                    raise ValueError(f"{algorithm} seed {seed} has not completed")
                checkpoint = max(root.rglob("epoch-250.pt"), key=lambda p: p.stat().st_mtime)
                for deterministic in (True, False):
                    mode = "mean" if deterministic else "stochastic"
                    for start in range(100, 150, 5):
                        specs.append((f"{run}/{algorithm}-seed{seed}-{mode}-{start}", "native_learned",
                                      dict(checkpoint=str(checkpoint), deterministic=deterministic,
                                           policy_seed=1000003*seed), "validation", list(range(start, start+5))))
        summaries.extend(evaluate_learned_shard.map(specs))
        return _save_summaries(run, summaries)
    if stage in ("baseline-completed-development", "baseline-completed-modes-development"):
        volume.reload()
        specs = []
        for seed in (0, 2):
            root = Path(f"/results/baselines-1m-v1/PPOLag-seed{seed}")
            record = json.loads((root / "manifest.json").read_text())
            if record["status"] != "completed":
                raise ValueError(f"Baseline seed {seed} has not completed")
            checkpoint = max(root.rglob("epoch-250.pt"), key=lambda p: p.stat().st_mtime)
            modes = [True, False] if stage.endswith("modes-development") else [True]
            for deterministic in modes:
                mode = "mean" if deterministic else "stochastic"
                for start in range(100, 150, 5):
                    specs.append((f"{run}/PPOLag-seed{seed}-{mode}-{start}", "native_learned",
                                  dict(checkpoint=str(checkpoint), deterministic=deterministic,
                                       policy_seed=1000003*seed), "validation", list(range(start,start+5))))
        return _save_summaries(run, list(evaluate_learned_shard.map(specs)))
    if stage in ("camera-eval-pilot", "camera-eval-development"):
        snapshots = list(snapshot_camera_adapter.starmap(
            [("camera-lora-v1", seed, camera_step) for seed in range(3)]))
        starts = [100] if stage == "camera-eval-pilot" else list(range(100, 150, 5))
        specs = [(f"{run}/camera-skill{seed}-{start}", snapshots[seed]["adapter"],
                  f"/results/skills-1m-v1/skills-seed{seed}/ppo.zip", "validation",
                  list(range(start, start+5)), 0.0, True)
                 for seed in range(3) for start in starts]
        summaries = list(evaluate_camera_shard.map(specs, order_outputs=False))
        return _save_summaries(run, summaries)
    if stage == "baseline-replacements":
        specs = [(run, alg, seed, steps) for alg, seed in [("PPOLag", 1), ("CPO", 0)]]
        for record in train_baseline_reserved.map(specs, order_outputs=False):
            print(json.dumps(record))
        return
    if stage == "camera-pack":
        for record in pack_camera_shard.map(range(0, 100, 5), order_outputs=False):
            print(json.dumps(record))
        return
    if stage in ("tracker-develop", "tracker-final-develop"):
        variants = ([("final", "ppo.zip")] if stage == "tracker-final-develop" else
                    [("cloned", "cloned.zip"), ("200k", "checkpoints/rl_model_200000_steps.zip")])
        policies = [(f"rule-skill{seed}-{variant}", "lane_caution_rule",
                     dict(speed=30.0, skill_checkpoint=f"/results/skills-1m-v1/skills-seed{seed}/{checkpoint}"))
                    for seed in range(3) for variant, checkpoint in
                    variants]
        specs = [(f"{run}/{tag}-{start}", policy, parameters, "validation", list(range(start, start+5)))
                 for tag, policy, parameters in policies for start in range(100, 150, 5)]
        summaries = list(evaluate_learned_shard.map(specs, order_outputs=False))
        return _save_summaries(run, summaries)
    if stage == "planner-ablation-develop":
        policies = [(f"{mode}-skill{seed}", "lane_caution_rule", dict(
            speed=30.0, planner="external", initial_careful=(mode == "careful"),
            skill_checkpoint=f"/results/skills-1m-v1/skills-seed{seed}/ppo.zip"))
                    for seed in range(3) for mode in ("normal", "careful")]
        specs = [(f"{run}/{tag}-{start}", policy, parameters, "validation", list(range(start, start+5)))
                 for tag, policy, parameters in policies for start in range(100, 150, 5)]
        summaries = list(evaluate_learned_shard.map(specs, order_outputs=False))
        return _save_summaries(run, summaries)
    if stage in ("camera-train", "camera-train-smoke"):
        specs = [(run, "camera-data-v1", 2 if stage == "camera-train-smoke" else 600, seed)
                 for seed in ([0] if stage == "camera-train-smoke" else [0, 1, 2])]
        for record in train_camera.starmap(specs, order_outputs=False):
            print(json.dumps(record))
        return
    if stage in ("camera-data", "camera-smoke"):
        specs = ([(run, 0, 1, 20)] if stage == "camera-smoke" else
                 [(run, start, 5, 1000) for start in range(0, 100, 5)])
        for record in camera_data.map(specs, order_outputs=False):
            print(json.dumps(record))
        return
    if stage in ("skills", "skills-smoke"):
        for record in train_skills.map([(run, seed, 4000 if stage == "skills-smoke" else steps)
                                        for seed in ([0] if stage == "skills-smoke" else [0, 1, 2])],
                                       order_outputs=False):
            print(json.dumps(record))
        return
    if stage in ("train", "train-smoke"):
        specs = [(run, alg, seed, 4000 if stage == "train-smoke" else steps)
                 for alg in ("PPOLag", "CPO") for seed in ([0] if stage == "train-smoke" else [0, 1, 2])]
        for record in train_baseline.map(specs, order_outputs=False):
            print(json.dumps(record))
        return
    if stage == "develop":
        policies = [
            ("idm", "idm", {}),
            ("rule25", "lane_caution_rule", {"speed": 25.0}),
            ("rule30", "lane_caution_rule", {"speed": 30.0}),
            ("rule35", "lane_caution_rule", {"speed": 35.0}),
            ("rule30room", "lane_caution_rule", {"speed": 30.0, "headway": 1.5, "clearance": 5.0}),
            ("rule30curve", "lane_caution_rule", {"speed": 30.0, "lateral_accel": 1.5}),
        ]
        specs = [(f"{run}/{tag}-{start}", policy, params, "validation", list(range(start, start+5)))
                 for tag, policy, params in policies for start in range(100, 150, 5)]
        summaries = list(evaluate_shard.map(specs, order_outputs=False))
        return _save_summaries(run, summaries)
    if stage != "pilot":
        raise ValueError(stage)
    policies = [
        ("idm", "idm", {}),
        ("expert", "upstream_expert", {}),
        ("rule30", "lane_caution_rule", {"speed": 30.0}),
        ("rule40", "lane_caution_rule", {"speed": 40.0}),
        ("rule50", "lane_caution_rule", {"speed": 50.0}),
    ]
    specs = [(f"{run}/{tag}-{start}", policy, params, "validation", list(range(start, start+5)))
             for tag, policy, params in policies for start in (100, 105)]
    summaries = list(evaluate_shard.map(specs, order_outputs=False))
    return _save_summaries(run, summaries)


@app.function(image=image, volumes={"/results": volume}, cpu=0.125, memory=512,
              timeout=6*3600, nonpreemptible=True)
def cloud_run(stage: str = "pilot", run: str = "pilot-v1", steps: int = 1000000,
              camera_step: int = 200, after_call: str = "", after_calls: list[str] | None = None,
              previous_call: str = ""):
    """Own the entire fan-out remotely; a disconnected laptop is not the parent."""
    import datetime
    import traceback

    path = Path(f"/results/{run}/coordinator.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = dict(stage=stage, run=run, status="running",
                  started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dependencies = after_calls or ([after_call] if after_call else [])
    if dependencies:
        record.update(status="waiting", after_calls=dependencies)
    path.write_text(json.dumps(record, indent=2))
    volume.commit()
    try:
        if dependencies:
            for dependency in dependencies:
                modal.FunctionCall.from_id(dependency).get()
            volume.reload()
            record.update(status="running",
                          dependency_finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            path.write_text(json.dumps(record, indent=2))
            volume.commit()
        result = run_stage(stage, run, steps, camera_step,previous_call)
        record["status"] = "completed"
        return result
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        volume.reload()
        record["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        path.write_text(json.dumps(record, indent=2))
        volume.commit()


@app.local_entrypoint()
def main(stage: str = "pilot", run: str = "pilot-v1", steps: int = 1000000,
         camera_step: int = 200):
    return cloud_run.remote(stage, run, steps, camera_step)
