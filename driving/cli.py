"""Small CLI for the maintained driver, simulator and camera integration."""

import argparse
import importlib.util
import json
import logging
from pathlib import Path

from driving import __version__
from driving.config import EVAL_MODES, load_config
from driving.errors import EventCBFError
from driving.evaluation import load_skill_bank, run_episode
from driving.io_utils import atomic_write_json, json_safe
from driving.model_loader import validate_checkpoint_dir
from driving.planner import create_router
from driving.policy import evaluate_skill
from driving.runtime import write_episode

ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"event-cbf {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("eval", "verify", "doctor"):
        sub = commands.add_parser(name)
        sub.add_argument("--profile", default="default")
        sub.add_argument("--config", type=Path)
        sub.add_argument("--ckpt-dir", type=Path)
        if name == "eval":
            sub.add_argument("--modes", nargs="+", choices=EVAL_MODES)
            sub.add_argument("--seeds", nargs="+", type=int)
            sub.add_argument("--out-dir", type=Path)
            sub.add_argument("--run-id", default="evaluation")
            sub.add_argument("--router", choices=("scripted", "vlm"))
            sub.add_argument("--allow-live", action="store_true")
    train = commands.add_parser("train", help="Train a new single-critic PPO driver")
    train.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv=None):
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        from driving.train import main as train

        return train(argv[1:]) or 0
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config or ROOT / "configs" / f"{args.profile}.yaml")
        logging.basicConfig(level=cfg.log_level)
        checkpoint_dir = args.ckpt_dir or ROOT / cfg.checkpoint_dir
        if args.command in ("verify", "doctor"):
            report = validate_checkpoint_dir(checkpoint_dir)
            if args.command == "verify" and report["valid"]:
                bank, _ = load_skill_bank(checkpoint_dir)
                report["diagnostics"] = {
                    name: evaluate_skill(policy.net, policy.meta.y_target)
                    for name, policy in bank.items()
                    if name != "RETURN_RIGHT"
                }
            if args.command == "doctor":
                report["physics_available"] = importlib.util.find_spec("metadrive") is not None
                report["package_version"] = __version__
            print(json.dumps(json_safe(report), indent=2))
            return 0 if report["valid"] else 1
        if args.router:
            cfg.router.backend = args.router
        cfg.router.allow_live = args.allow_live or cfg.router.allow_live
        bank, _ = load_skill_bank(checkpoint_dir)
        output = args.out_dir or ROOT / cfg.eval.out_dir
        target = output / args.run_id
        router = create_router(cfg.router)
        summaries = []
        try:
            for scene in cfg.eval.scenarios:
                for seed in args.seeds or cfg.eval.seeds:
                    for mode in args.modes or cfg.eval.modes:
                        result = run_episode(
                            mode,
                            scene,
                            seed,
                            bank,
                            router,
                            stack_config=cfg.stack,
                            shield_config=cfg.shield,
                        )
                        row = {
                            key: getattr(result, key)
                            for key in (
                                "scenario",
                                "seed",
                                "mode",
                                "route_pct",
                                "crashed",
                                "freeze",
                                "router_calls",
                            )
                        }
                        summaries.append(row)
                        write_episode(
                            target / f"{scene.name}-{mode}-{seed}.json", result, scenario=scene
                        )
            atomic_write_json(target / "results.json", summaries)
        finally:
            close = getattr(router, "close", None)
            if callable(close):
                close()
        print(json.dumps(json_safe(summaries), indent=2))
        return 0
    except (EventCBFError, ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return getattr(exc, "exit_code", 2)


def eval_entry():
    import sys

    return main(["eval", *sys.argv[1:]])


def train_entry():
    from driving.train import main as train

    return train() or 0


def verify_entry():
    import sys

    return main(["verify", *sys.argv[1:]])


def doctor_entry():
    import sys

    return main(["doctor", *sys.argv[1:]])
