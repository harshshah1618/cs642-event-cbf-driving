"""Command-line interface for training, evaluation, validation, and diagnostics.

The CLI is intentionally thin: all logic lives in importable package modules so
the same operations are available from Python. Every failure maps to a stable
exit code from :mod:`event_cbf_driving.errors`.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from event_cbf_driving import __version__
from event_cbf_driving.config import (
    EVAL_MODES,
    SCENARIO_NAMES,
    ProjectConfig,
    ScenarioConfig,
    apply_overrides,
    default_config,
    load_config,
)
from event_cbf_driving.errors import (
    ArtifactError,
    CheckpointError,
    ConfigError,
    EvaluationFailure,
    EventCBFError,
    RouterError,
)
from event_cbf_driving.eval.runner import check_regressions, run_evaluation
from event_cbf_driving.io_utils import strict_json_dumps
from event_cbf_driving.logging_utils import configure_logging, json_logs_requested
from event_cbf_driving.router.check import check_router
from event_cbf_driving.router.factory import create_router
from event_cbf_driving.skills.loader import validate_checkpoint_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "configs"
DEFAULT_PROFILE = PROFILE_DIR / "default.yaml"
PROFILE_CHOICES = (
    "default",
    "production",
    "ci",
    "metadrive",
    "metadrive-live",
    "metadrive-study",
    "metadrive-local",
    "live-openai",
    "live-ollama",
    "live-local",
)
logger = logging.getLogger("event_cbf_driving.cli")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML/JSON config file (default: configs/default.yaml when present)",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=None,
        help="named config profile; mutually exclusive with --config",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. stack.min_call_ticks=20 (repeatable)",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="emit structured JSON logs on stderr",
    )


def _load_project_config(args: argparse.Namespace) -> ProjectConfig:
    if args.config is not None and args.profile is not None:
        raise ConfigError("--config and --profile are mutually exclusive")
    if args.config is not None:
        path = Path(args.config).expanduser()
    elif args.profile is not None:
        path = PROFILE_DIR / f"{args.profile}.yaml"
    elif DEFAULT_PROFILE.is_file():
        path = DEFAULT_PROFILE
    else:
        config = default_config()
        apply_overrides(config, args.override)
        return config
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    return load_config(path, overrides=args.override)


def _configure_logging(args: argparse.Namespace, config: ProjectConfig) -> None:
    level = (args.log_level or config.log_level or "INFO").upper()
    configure_logging(
        level=level,
        json_logs=bool(args.json_logs or config.json_logs or json_logs_requested()),
        force=True,
    )


def _print_json(payload: Any) -> None:
    sys.stdout.write(strict_json_dumps(payload, indent=2, sort_keys=True) + "\n")


def _short_number(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number or number in (float("inf"), float("-inf")):
        return "n/a"
    return f"{number:.{digits}f}"


def _mean(metric: Any) -> Optional[float]:
    if isinstance(metric, dict):
        metric = metric.get("mean")
    try:
        value = float(metric)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _print_eval_summary(result) -> None:
    print(f"run_id: {result.run_id}")
    if result.run_dir is not None:
        print(f"artifacts: {result.run_dir}")
    print()
    header = f"{'scenario':<14} {'mode':<22} {'route%':>7} {'crash%':>7} {'freeze%':>8} {'calls':>6} {'jerk':>7}"
    print(header)
    print("-" * len(header))
    for scenario in sorted(result.aggregates):
        modes = result.aggregates[scenario]
        for mode in sorted(modes):
            metrics = modes[mode]
            route = _mean(metrics.get("route_pct"))
            crash = _mean(metrics.get("crashed")) or _mean(metrics.get("crash_rate"))
            freeze = _mean(metrics.get("freeze")) or _mean(metrics.get("freeze_rate"))
            calls = _mean(metrics.get("router_calls"))
            jerk = _mean(metrics.get("peak_lateral_jerk"))
            print(
                f"{scenario:<14} {mode:<22} "
                f"{_short_number(route):>7} "
                f"{_short_number(100.0 * crash if crash is not None else None):>7} "
                f"{_short_number(100.0 * freeze if freeze is not None else None):>8} "
                f"{_short_number(calls, 1):>6} "
                f"{_short_number(jerk):>7}"
            )
    telemetry = ((result.manifest or {}).get("extra") or {}).get(
        "router_telemetry"
    ) or {}
    if telemetry:
        print(
            "\nrouter telemetry: "
            f"live calls {int(telemetry.get('router_live_calls') or 0)}"
            f"/{int(telemetry.get('router_calls') or 0)}, "
            f"fallbacks {int(telemetry.get('router_fallbacks') or 0)}, "
            f"tokens {int(telemetry.get('total_tokens') or 0)}"
        )
        models = telemetry.get("models") or []
        if models:
            print("  models: " + ", ".join(str(model) for model in models))
    if result.warnings:
        print("\nwarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


def _scenario_configs(names: Optional[Sequence[str]], seeds: Sequence[int]) -> List[ScenarioConfig]:
    if not names:
        return []
    return [ScenarioConfig(name=name, seed=int(seeds[0]) if seeds else 0) for name in names]


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    config = _load_project_config(args)
    _configure_logging(args, config)

    if args.router is not None:
        config.router.backend = args.router
    if args.allow_live:
        config.router.allow_live = True
    if args.ckpt_dir is not None:
        config.checkpoint_dir = str(Path(args.ckpt_dir).expanduser())
    if args.seeds is not None:
        config.eval.seeds = [int(seed) for seed in args.seeds]
    if args.scenarios:
        configured = {scenario.name: scenario for scenario in config.eval.scenarios}
        config.eval.scenarios = [
            configured.get(name, ScenarioConfig(name=name)) for name in args.scenarios
        ]
    if args.modes:
        config.eval.modes = list(dict.fromkeys(args.modes))
    if args.repeats is not None:
        config.eval.repeats = int(args.repeats)
    if args.no_figures:
        config.eval.figures = False
    if args.record_episodes:
        config.eval.record_episodes = True
    if args.fail_on_regression is not None:
        config.eval.fail_on_regression = bool(args.fail_on_regression)
    if args.min_route_pct is not None:
        # Keep the manifest and the gate consistent when the floor is
        # overridden on the command line.
        config.eval.min_route_pct = float(args.min_route_pct)
    config.router.validate("router")
    config.eval.validate("eval")
    config.validate()

    live_backend = config.router.backend in ("auto", "openai", "local")
    live_requested = bool(
        args.require_live or (config.router.allow_live and live_backend)
    )
    if live_requested and not args.no_live_preflight:
        probe = check_router(
            config.router,
            allow_live=config.router.allow_live,
            require_live=True,
        )
        if not probe.ok:
            detail = probe.error or (
                "no live model answered (fallback or offline backend)"
            )
            if args.config:
                command = f"event-cbf router-check --config {args.config}"
            else:
                command = (
                    f"event-cbf router-check --profile {args.profile or 'default'}"
                )
            raise RouterError(
                f"live preflight failed: {detail}. Run `{command}` for details."
            )

    # The runner owns router construction so the same gated factory path is
    # used by the CLI and by programmatic callers.
    result = run_evaluation(
        config,
        checkpoint_dir=args.ckpt_dir,
        router=None,
        out_dir=args.out_dir,
        run_id=args.run_id,
        write=not args.no_write,
        update_docs=args.update_docs,
        record_episodes=True if args.record_episodes else None,
        allow_missing_checkpoints=args.allow_missing_checkpoints,
    )

    if args.json:
        _print_json(
            {
                "run_id": result.run_id,
                "run_dir": None if result.run_dir is None else str(result.run_dir),
                "aggregates": result.aggregates,
                "warnings": result.warnings,
            }
        )
    else:
        _print_eval_summary(result)

    if args.require_live:
        telemetry = ((result.manifest or {}).get("extra") or {}).get(
            "router_telemetry"
        ) or {}
        live_calls = int(telemetry.get("router_live_calls") or 0)
        fallbacks = int(telemetry.get("router_fallbacks") or 0)
        if live_calls <= 0:
            raise RouterError(
                "--require-live: the evaluation made no live model calls "
                "(check the router backend and selected modes)"
            )
        if fallbacks > 0:
            raise RouterError(
                f"--require-live: {fallbacks} router call(s) used a fallback; "
                "inspect results.json and the run manifest"
            )

    if not config.eval.fail_on_regression:
        return 0
    min_route_pct = (
        float(config.eval.min_route_pct)
        if args.min_route_pct is None
        else float(args.min_route_pct)
    )
    failures = check_regressions(
        result,
        min_route_pct=min_route_pct,
        require_no_crash=not args.allow_crash,
        require_no_freeze=not args.allow_freeze,
    )
    if failures:
        for failure in failures:
            logger.error("regression: %s", failure)
        raise EvaluationFailure("; ".join(failures))
    return 0


# ---------------------------------------------------------------------------
# router-check
# ---------------------------------------------------------------------------


def cmd_router_check(args: argparse.Namespace) -> int:
    config = _load_project_config(args)
    _configure_logging(args, config)
    if args.router is not None:
        config.router.backend = args.router
    if args.allow_live:
        config.router.allow_live = True
    if args.timeout is not None:
        config.router.timeout_s = float(args.timeout)
    config.router.validate("router")

    check = check_router(
        config.router,
        allow_live=config.router.allow_live,
        require_live=bool(args.require_live),
    )
    payload = check.to_dict()

    if args.json:
        _print_json(payload)
    else:
        status = "ok" if check.ok else "FAIL"
        print(f"router-check: {status}")
        print(f"  configured backend: {check.configured_backend}")
        print(f"  backend: {check.name} ({check.describe})")
        print(f"  live capable: {'yes' if check.live_capable else 'no'}")
        print(f"  live answer: {'yes' if check.live else 'no'}")
        if check.model:
            print(f"  model: {check.model}")
        if check.skill:
            print(f"  skill: {check.skill}")
        print(f"  latency: {check.wall_latency_s * 1000.0:.1f} ms")
        if check.usage:
            print(
                "  tokens: "
                f"prompt={check.usage.get('prompt_tokens', 0)} "
                f"completion={check.usage.get('completion_tokens', 0)} "
                f"total={check.usage.get('total_tokens', 0)}"
            )
        if check.fallback_used:
            print(f"  fallback: yes ({check.error or 'unknown reason'})")
        elif check.error:
            print(f"  error: {check.error}")
        health = check.health or {}
        print(
            "  health: "
            f"{health.get('status', 'unknown')} "
            f"(calls={health.get('total_calls', 0)}, "
            f"failures={health.get('total_failures', 0)}, "
            f"tokens={health.get('total_tokens', 0)})"
        )

    if check.ok:
        return 0
    logger.error("router check failed: %s", check.error or "no live answer")
    return int(RouterError.exit_code)


# ---------------------------------------------------------------------------
# train / verify
# ---------------------------------------------------------------------------


def _training_argv(
    args: argparse.Namespace,
    config: ProjectConfig,
    *,
    verify_only: bool,
) -> List[str]:
    argv: List[str] = []
    ckpt_dir = Path(args.ckpt_dir).expanduser() if args.ckpt_dir else Path(config.checkpoint_dir)
    argv += ["--ckpt-dir", str(ckpt_dir)]
    if args.fig_dir is not None:
        argv += ["--fig-dir", str(Path(args.fig_dir).expanduser())]
    if verify_only:
        argv.append("--verify-only")
        if args.min_learned_skills is not None:
            argv += ["--min-learned-skills", str(int(args.min_learned_skills))]
        return argv

    skills = args.skills or list(config.train.skills)
    argv += ["--skills", *skills]
    updates = args.updates if args.updates is not None else int(config.train.updates)
    argv += ["--updates", str(int(updates))]
    seed = args.seed if args.seed is not None else config.train.seed
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    min_learned = (
        args.min_learned_skills
        if args.min_learned_skills is not None
        else config.train.min_learned_skills
    )
    if min_learned is not None:
        argv += ["--min-learned-skills", str(int(min_learned))]
    return argv


def cmd_train(args: argparse.Namespace) -> int:
    from event_cbf_driving import training

    config = _load_project_config(args)
    _configure_logging(args, config)
    config.train.skills = list(args.skills) if args.skills else list(config.train.skills)
    config.train.validate("train")
    return int(training.main(_training_argv(args, config, verify_only=False)))


def cmd_verify(args: argparse.Namespace) -> int:
    from event_cbf_driving import training

    config = _load_project_config(args)
    _configure_logging(args, config)
    rc = int(training.main(_training_argv(args, config, verify_only=True)))
    if rc != 0:
        raise CheckpointError("checkpoint validation failed; see the output above")
    return 0


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def cmd_replay(args: argparse.Namespace) -> int:
    from event_cbf_driving.runtime import load_episode, replay_open_loop

    episode = load_episode(args.episode)
    payload: Dict[str, Any] = {
        "episode": str(args.episode),
        "mode": episode.result.mode,
        "scenario": episode.result.scenario,
        "seed": episode.result.seed,
        "ticks": episode.ticks,
        "has_actions": bool(episode.actions),
    }
    if not args.validate_only:
        report = replay_open_loop(episode, tolerance=float(args.tolerance))
        payload["replay"] = report.to_dict()
        if not report.ok:
            if args.json:
                _print_json(payload)
            raise ArtifactError("replay divergence: " + "; ".join(report.errors))

    if args.json:
        _print_json(payload)
    else:
        print(
            f"{payload['scenario']}/{payload['mode']} seed={payload['seed']} "
            f"ticks={payload['ticks']}"
        )
        if args.validate_only:
            print("schema: ok")
        else:
            replay = payload["replay"]
            print(
                "replay: ok"
                if replay["ok"]
                else f"replay: FAIL ({'; '.join(replay['errors'])})"
            )
            print(
                f"  final state error {replay['final_state_error']:.3g} m "
                f"(x={replay['replayed_x']:.3f}, y={replay['replayed_y']:.3f})"
            )
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _check_import(name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "version": getattr(module, "__version__", None)}


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _load_project_config(args)
    _configure_logging(args, config)
    checks: Dict[str, Any] = {
        "package_version": __version__,
        "python": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 10),
        "platform": platform.platform(),
        "imports": {
            name: _check_import(name)
            for name in ("numpy", "torch", "gymnasium", "matplotlib", "yaml")
        },
        "config": {"ok": True, "name": config.name},
        "checkpoints": {},
        "router": {},
        "errors": [],
    }

    if any(scenario.simulator == "metadrive" for scenario in config.eval.scenarios):
        checks["imports"]["metadrive"] = _check_import("metadrive")
        checks["imports"]["panda3d.bullet"] = _check_import("panda3d.bullet")

    ckpt_dir = Path(args.ckpt_dir or config.checkpoint_dir).expanduser()
    try:
        checkpoint_report = validate_checkpoint_dir(ckpt_dir)
        checks["checkpoints"] = {
            "ok": bool(checkpoint_report.get("valid")),
            "directory": str(ckpt_dir),
            "aliases": checkpoint_report.get("aliases", {}),
            "duplicates": checkpoint_report.get("duplicates", []),
            "problems": checkpoint_report.get("problems", []),
            "learned_distinct": checkpoint_report.get("learned_distinct"),
        }
    except Exception as exc:
        checks["checkpoints"] = {"ok": False, "error": str(exc)}

    try:
        router = create_router(config.router, allow_live=config.router.allow_live)
        checks["router"] = {
            "ok": True,
            "name": getattr(router, "name", type(router).__name__),
            "describe": router.describe() if hasattr(router, "describe") else None,
            "live_capable": bool(getattr(router, "is_live_capable", False)),
            "allow_live": bool(getattr(router, "allow_live", False)),
            "model": getattr(router, "model", None),
        }
        close = getattr(router, "close", None)
        if callable(close):
            close()
    except Exception as exc:
        checks["router"] = {"ok": False, "error": str(exc)}

    checks["ok"] = bool(
        checks["python_ok"]
        and all(item.get("ok") for item in checks["imports"].values())
        and checks["checkpoints"].get("ok")
        and checks["router"].get("ok")
    )
    if args.json:
        _print_json(checks)
    else:
        print(f"event-cbf {__version__}  python {checks['python']}")
        print(f"config: {config.name}")
        for name, item in checks["imports"].items():
            status = "ok" if item.get("ok") else f"FAIL ({item.get('error')})"
            print(f"  import {name:<10} {status}")
        ckpt = checks["checkpoints"]
        print(f"  checkpoints {ckpt.get('directory')} ok={ckpt.get('ok')}")
        for problem in ckpt.get("problems", []):
            print(f"    - {problem}")
        print(
            f"  router {checks['router'].get('name')} "
            f"ok={checks['router'].get('ok')} "
            f"live_capable={checks['router'].get('live_capable')}"
        )
        print(f"overall: {'ok' if checks['ok'] else 'FAIL'}")
    return 0 if checks["ok"] else 1


# ---------------------------------------------------------------------------
# Parser / entries
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="event-cbf",
        description="Event-triggered skill router + PPO skills + CBF safety shield",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval", help="run the full multi-mode, multi-seed evaluation")
    _add_common(eval_parser)
    eval_parser.add_argument(
        "--router",
        choices=("auto", "scripted", "dryrun", "openai", "local"),
        default=None,
    )
    eval_parser.add_argument(
        "--allow-live",
        action="store_true",
        help="opt in to live router backends (sets router.allow_live=true)",
    )
    eval_parser.add_argument(
        "--require-live",
        action="store_true",
        help="preflight and fail unless real live model calls produced the results",
    )
    eval_parser.add_argument(
        "--no-live-preflight",
        action="store_true",
        help="skip the one-request live connectivity probe (diagnostics only)",
    )
    eval_parser.add_argument("--ckpt-dir", type=Path, default=None)
    eval_parser.add_argument("--out-dir", type=Path, default=None)
    eval_parser.add_argument("--run-id", default=None)
    eval_parser.add_argument("--seeds", nargs="+", type=int, default=None)
    eval_parser.add_argument("--repeats", type=int, default=None)
    eval_parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_NAMES, default=None)
    eval_parser.add_argument("--modes", nargs="+", choices=EVAL_MODES, default=None)
    eval_parser.add_argument("--no-figures", action="store_true")
    eval_parser.add_argument("--record-episodes", action="store_true")
    eval_parser.add_argument("--update-docs", action="store_true")
    eval_parser.add_argument("--no-write", action="store_true")
    eval_parser.add_argument(
        "--allow-missing-checkpoints",
        action="store_true",
        help="smoke tests only: install untrained fallback policies",
    )
    eval_parser.add_argument(
        "--fail-on-regression",
        dest="fail_on_regression",
        action="store_true",
        default=None,
    )
    eval_parser.add_argument(
        "--no-fail-on-regression",
        dest="fail_on_regression",
        action="store_false",
    )
    eval_parser.add_argument(
        "--min-route-pct",
        type=float,
        default=None,
        help="route floor for the proposed stack (default: eval.min_route_pct)",
    )
    eval_parser.add_argument("--allow-crash", action="store_true")
    eval_parser.add_argument("--allow-freeze", action="store_true")
    eval_parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    eval_parser.set_defaults(func=cmd_eval)

    check_parser = sub.add_parser(
        "router-check",
        help="send one sample interrupt and report live/fallback/usage/health",
    )
    _add_common(check_parser)
    check_parser.add_argument(
        "--router",
        choices=("auto", "scripted", "dryrun", "openai", "local"),
        default=None,
    )
    check_parser.add_argument(
        "--allow-live",
        action="store_true",
        help="opt in to live router backends (sets router.allow_live=true)",
    )
    check_parser.add_argument(
        "--require-live",
        action="store_true",
        help="exit nonzero unless a real model answered without fallback",
    )
    check_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="override router.timeout_s for this probe",
    )
    check_parser.add_argument("--json", action="store_true")
    check_parser.set_defaults(func=cmd_router_check)

    train_parser = sub.add_parser("train", help="train distinct PPO skills")
    _add_common(train_parser)
    train_parser.add_argument("--ckpt-dir", type=Path, default=None)
    train_parser.add_argument("--fig-dir", type=Path, default=None)
    train_parser.add_argument("--skills", nargs="+", default=None)
    train_parser.add_argument("--updates", type=int, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--min-learned-skills", type=int, default=None)
    train_parser.set_defaults(func=cmd_train)

    verify_parser = sub.add_parser("verify", help="validate checkpoints without training")
    _add_common(verify_parser)
    verify_parser.add_argument("--ckpt-dir", type=Path, default=None)
    verify_parser.add_argument("--fig-dir", type=Path, default=None)
    verify_parser.add_argument("--min-learned-skills", type=int, default=None)
    verify_parser.set_defaults(func=cmd_verify)

    replay_parser = sub.add_parser(
        "replay", help="validate an episode artifact and replay its action log"
    )
    replay_parser.add_argument("episode", type=Path, help="episode JSON written by --record-episodes")
    replay_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="schema-check the artifact without re-running the environment",
    )
    replay_parser.add_argument("--tolerance", type=float, default=1e-6)
    replay_parser.add_argument("--json", action="store_true")
    replay_parser.set_defaults(func=cmd_replay)

    doctor_parser = sub.add_parser("doctor", help="check dependencies, config, and checkpoints")
    _add_common(doctor_parser)
    doctor_parser.add_argument("--ckpt-dir", type=Path, default=None)
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except EventCBFError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return int(getattr(exc, "exit_code", 1))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - unexpected bug
        logger.exception("unexpected failure")
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1


def eval_entry() -> int:
    return main(["eval", *sys.argv[1:]])


def train_entry() -> int:
    return main(["train", *sys.argv[1:]])


def verify_entry() -> int:
    return main(["verify", *sys.argv[1:]])


def doctor_entry() -> int:
    return main(["doctor", *sys.argv[1:]])


__all__ = [
    "build_parser",
    "cmd_doctor",
    "cmd_eval",
    "cmd_replay",
    "cmd_router_check",
    "cmd_train",
    "cmd_verify",
    "doctor_entry",
    "eval_entry",
    "main",
    "train_entry",
    "verify_entry",
]


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
