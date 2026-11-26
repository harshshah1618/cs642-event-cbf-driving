"""Re-evaluate irregular scenes on CPU without camera inference or cloud jobs.

The `none` arm repeats the published rule + PPO control. `careful` keeps the
same rule + PPO + protection and enables careful mode from the episode start.
This is a new local control comparison, not a rerun of Qwen or PatchCore.
"""

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCENES = ("flagger", "billboard", "pedestrian-shoulder", "pedestrian-lane", "lane-closed",
          "wet-closed", "rain-clean", "clean")


def evaluate(spec):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    from driving.irregular.escalation import (
        EscalationConfig,
        irregular_config,
        run_episode,
        scene_placements,
    )
    from driving.irregular.scene import make_env

    torch.set_num_threads(1)
    scene, mode, seeds, checkpoint, output, source_hash = spec
    path = Path(output) / f"{scene}-{mode}-{seeds[0]}.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    if any(row.get("source_sha256") != source_hash for row in rows):
        raise ValueError(f"Source changed since {path}; use a new output directory")
    completed = {r["seed"] for r in rows}
    started = time.monotonic()
    env = make_env(irregular_config("irregular-test"))
    try:
        for seed in seeds:
            if seed in completed:
                continue
            row = run_episode(env, seed, scene_placements(scene, seed), EscalationConfig(),
                              driver_kwargs=dict(skill_checkpoint=checkpoint,
                                                 initial_careful=mode == "careful"),
                              max_steps=300 if scene in ("clean", "rain-clean") else 600)
            row.update(scene=scene, model=f"local-PPO-{mode}", control=mode,
                       source_sha256=source_hash)
            rows.append(row)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
    finally:
        env.close()
    return dict(scene=scene, control=mode, episodes=len(rows),
                seconds=round(time.monotonic() - started, 1), path=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1050)))
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--controls", nargs="+", choices=("none", "careful"),
                        default=["none", "careful"])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 1000 <= s < 1050 for s in args.seeds):
        parser.error("Use unique seeds from the already-published 1000-1049 map set")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in (root / "driving/irregular", root / "driving/safemetadrive")
               for p in sorted(folder.glob("*.py"))}
    sources["tools/irregular_local_eval.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    manifest = dict(
        scope="Local CPU control comparison on previously evaluated maps; no VLM or PatchCore",
        seeds=args.seeds, scenes=args.scenes, controls=args.controls, source_sha256=source_hash,
        source_files=sources,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        packages={name: importlib.metadata.version(name) for name in
                  ("numpy", "torch", "gymnasium", "stable-baselines3", "metadrive-simulator", "panda3d")},
    )
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error("Output manifest differs; choose a new directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    episodes = args.output / "episodes"
    episodes.mkdir(exist_ok=True)
    specs = [(scene, mode, args.seeds, str(args.checkpoint.resolve()), str(episodes), source_hash)
             for scene in args.scenes for mode in args.controls]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in as_completed([pool.submit(evaluate, spec) for spec in specs]):
            print(json.dumps(result.result()), flush=True)


if __name__ == "__main__":
    main()
