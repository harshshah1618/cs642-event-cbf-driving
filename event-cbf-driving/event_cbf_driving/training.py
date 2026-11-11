#!/usr/bin/env python3
"""Train the distinct PPO lane skills and validate their checkpoints.

Canonical training covers ``CRUISE_RIGHT`` and ``OVERTAKE_LEFT``. A request
for ``RETURN_RIGHT`` is still honoured (it trains a distinct policy), but this
module warns that canonical deployments load ``RETURN_RIGHT`` as a documented
alias of ``CRUISE_RIGHT`` when no distinct checkpoint exists.

``--verify-only`` never runs the training loop. It validates the existing
checkpoint directory, re-runs the learned diagnostic on each policy, and
writes ``validation.json``.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

from event_cbf_driving.config import SKILL_NAMES, TrainConfig
from event_cbf_driving.env.skill_lane import LANE_Y
from event_cbf_driving.io_utils import atomic_write_json, utc_now
from event_cbf_driving.provenance import environment_info, git_info
from event_cbf_driving.skills.loader import load_skill_bank, validate_checkpoint_dir
from event_cbf_driving.skills.ppo import (
    PPOTrainer,
    SkillMeta,
    calibrate_threshold,
    evaluate_skill,
    load_skill,
    save_skill,
)

logger = logging.getLogger(__name__)

SKILL_SPECS = [
    ("CRUISE_RIGHT", 0, LANE_Y[0], 0),
    ("OVERTAKE_LEFT", 1, LANE_Y[1], 1),
    ("RETURN_RIGHT", 0, LANE_Y[0], 2),
]
DEFAULT_SKILLS = ("CRUISE_RIGHT", "OVERTAKE_LEFT")
# Seeds known to converge on CPU within ~160 updates (plus fallbacks).
SEED_TRIES = {
    "CRUISE_RIGHT": [0, 3, 5, 9],
    "OVERTAKE_LEFT": [1, 7, 11, 21, 42],
    "RETURN_RIGHT": [2, 4, 8, 12],
}


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _revalidate_checkpoints(checkpoint_dir: Path, validation: Dict[str, Any]) -> Dict[str, Any]:
    """Load each checkpoint and re-run the learned diagnostic (inference only)."""

    revalidation: Dict[str, Any] = {}
    names = sorted(set(validation.get("skills", {})) | set(validation.get("required", ())))
    for name in names:
        path = Path(checkpoint_dir) / f"{name}.pt"
        entry: Dict[str, Any] = {"path": _display_path(path), "present": path.is_file()}
        if not path.is_file():
            entry["status"] = "missing"
            revalidation[name] = entry
            continue
        try:
            policy = load_skill(path)
            diagnostics = evaluate_skill(policy.net, policy.meta.y_target)
            entry.update(
                {
                    "status": "ok",
                    "learned": bool(diagnostics.get("learned")),
                    "diagnostics": diagnostics,
                    "meta": asdict(policy.meta),
                    "fingerprint": policy.checkpoint_info.get("fingerprint"),
                    "legacy": policy.checkpoint_info.get("legacy"),
                }
            )
        except Exception as exc:
            entry.update({"status": "error", "error": str(exc)})
        revalidation[name] = entry

    for name, target in (validation.get("aliases") or {}).items():
        source = revalidation.get(target)
        if source and source.get("status") == "ok":
            revalidation[name] = {
                "status": "alias",
                "alias_of": target,
                "path": source["path"],
                "learned": source["learned"],
                "diagnostics": source["diagnostics"],
                "meta": source["meta"],
                "fingerprint": source["fingerprint"],
                "legacy": source["legacy"],
            }
        else:
            revalidation[name] = {
                "status": "alias_target_unavailable",
                "alias_of": target,
            }
    return revalidation


def _distinct_required_names(validation: Dict[str, Any]) -> List[str]:
    aliases = set((validation.get("aliases") or {}).keys())
    return [name for name in validation.get("required", ()) if name not in aliases]


def _verify_only(args: argparse.Namespace) -> int:
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_checkpoint_dir(ckpt_dir)
    revalidation = _revalidate_checkpoints(ckpt_dir, validation)
    distinct = _distinct_required_names(validation)
    learned_distinct = sum(
        1
        for name in distinct
        if revalidation.get(name, {}).get("status") == "ok"
        and revalidation[name].get("learned") is True
    )
    min_learned = (
        args.min_learned_skills
        if args.min_learned_skills is not None
        else TrainConfig().min_learned_skills
    )
    validation.update(
        {
            "mode": "verify-only",
            "revalidation": revalidation,
            "learned_distinct": learned_distinct,
            "min_learned_skills": int(min_learned),
            "learned_ok": learned_distinct >= int(min_learned),
        }
    )
    try:
        bank = load_skill_bank(ckpt_dir)
        validation["bank"] = {
            "skills": sorted(bank),
            "aliases": bank.metadata.get("aliases", {}),
            "duplicates": bank.metadata.get("duplicates", []),
        }
    except Exception as exc:
        validation["bank"] = {"error": str(exc)}

    out_path = ckpt_dir / "validation.json"
    atomic_write_json(out_path, validation)
    _print_validation(validation, out_path)
    return 0 if validation.get("valid") else 1


def _print_validation(validation: Dict[str, Any], out_path: Path) -> None:
    print(f"Validation: {validation.get('checkpoint_dir')}")
    for name, info in sorted((validation.get("skills") or {}).items()):
        revalidated = (validation.get("revalidation") or {}).get(name, {})
        learned = revalidated.get("learned", info.get("learned"))
        fingerprint = (info.get("fingerprint") or "-")[:12]
        print(
            f"  {name:15s} fp={fingerprint:12s} legacy={info.get('legacy')!s:5s} "
            f"learned={learned}"
        )
    for name, alias in sorted((validation.get("aliases") or {}).items()):
        info = (validation.get("revalidation") or {}).get(name, {})
        print(f"  {name:15s} alias of {alias} (learned={info.get('learned')})")
    duplicates = validation.get("duplicates") or []
    print(f"  duplicates: {duplicates if duplicates else 'none'}")
    missing = validation.get("missing_required") or []
    print(f"  missing required: {missing if missing else 'none'}")
    mismatches = validation.get("fingerprint_mismatches") or []
    print(f"  fingerprint mismatches: {mismatches if mismatches else 'none'}")
    sidecar_warnings = validation.get("sidecar_warnings") or []
    print(f"  sidecar warnings: {len(sidecar_warnings)}")
    for warning in sidecar_warnings:
        print(f"    - {warning}")
    print(
        f"  learned distinct: {validation.get('learned_distinct')} "
        f"(min {validation.get('min_learned_skills')}, "
        f"ok={validation.get('learned_ok')})"
    )
    print(f"  valid: {validation.get('valid')}  -> {out_path}")


def _train(args: argparse.Namespace) -> int:
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_info()
    git = git_info()

    histories: Dict[str, List[float]] = {}
    entries: Dict[str, Dict[str, Any]] = {}
    for name, lane, y_t, _offset in SKILL_SPECS:
        if name not in args.skills:
            continue
        if name == "RETURN_RIGHT":
            message = (
                "RETURN_RIGHT targets the same right lane as CRUISE_RIGHT; canonical "
                "deployments load it as a documented alias. Training a distinct policy "
                "because it was explicitly requested."
            )
            logger.warning(message)
            print(f"WARNING: {message}")

        seeds = [args.seed] if args.seed is not None else SEED_TRIES[name]
        seeds_tried: List[int] = []
        best: Optional[Dict[str, Any]] = None
        for seed in seeds:
            seeds_tried.append(seed)
            print(f"\n=== Training {name} (y*={y_t}, seed={seed}) ===")
            trainer = PPOTrainer(name=name, y_target=y_t, target_lane=lane, seed=seed)
            net = trainer.train(n_updates=args.updates, min_updates=40)
            diag = evaluate_skill(net, y_t)
            thr, mean, std, spread = calibrate_threshold(net, y_t, seed=seed + 99)
            meta = SkillMeta(
                name=name,
                target_lane=lane,
                y_target=y_t,
                threshold=float(thr),
                sigma_mean=float(mean),
                sigma_std=float(std),
                spread=float(spread),
            )
            candidate = {
                "seed": seed,
                "net": net,
                "meta": meta,
                "history": list(trainer.history),
                "diag": diag,
                "learned": bool(diag.get("learned")),
            }
            if best is None:
                best = candidate
            elif candidate["learned"] and not best["learned"]:
                best = candidate
            elif (
                candidate["learned"]
                and best["learned"]
                and diag["hold"]["mean_abs_y"] < best["diag"]["hold"]["mean_abs_y"]
            ):
                best = candidate
            if diag["learned"]:
                break

        if best is None:  # pragma: no cover - seeds is never empty
            raise RuntimeError(f"no training candidate produced for {name}")
        diag = best["diag"]
        meta = best["meta"]
        history = best["history"]
        path = args.ckpt_dir / f"{name}.pt"
        provenance = {
            "agent": "event_cbf_driving.training",
            "caller": "event_cbf_driving.training:main",
            "skill": name,
            "seed": best["seed"],
            "seeds_tried": seeds_tried,
            "updates": len(history),
            "updates_requested": args.updates,
            "learned": bool(diag.get("learned")),
            "diagnostics": diag,
            "target_lane": lane,
            "y_target": y_t,
        }
        checkpoint_info = save_skill(
            path,
            best["net"],
            meta,
            history=history,
            provenance=provenance,
            extra={"diagnostics": diag},
        )
        histories[name] = history
        entries[name] = {
            "path": _display_path(path),
            "seed": best["seed"],
            "seeds_tried": seeds_tried,
            "updates_requested": args.updates,
            "updates_run": len(history),
            "history_len": len(history),
            "meta": asdict(meta),
            "diag": diag,
            "diagnostics": diag,
            "learned": bool(diag.get("learned")),
            "final_return": _finite_or_none(history[-1] if history else None),
            "fingerprint": checkpoint_info.get("fingerprint"),
            "provenance": checkpoint_info.get("provenance"),
        }
        print(
            f"  saved {path}  seed={best['seed']} thr={meta.threshold:.3f} "
            f"spread={meta.spread:.3f} learned={diag['learned']} "
            f"hold_y={diag['hold']['mean_abs_y']:.3f}"
        )

    validation = validate_checkpoint_dir(args.ckpt_dir)
    distinct = _distinct_required_names(validation)
    learned_count = sum(
        1
        for name, entry in entries.items()
        if entry.get("learned") is True
        and name in validation.get("fingerprints", {})
        and name not in (validation.get("aliases") or {})
    )
    min_learned = (
        args.min_learned_skills
        if args.min_learned_skills is not None
        else TrainConfig().min_learned_skills
    )
    summary: Dict[str, Any] = {
        "schema_version": 2,
        "created_at": utc_now(),
        "command": "event_cbf_driving.training",
        "config": {
            "skills": list(args.skills),
            "updates": args.updates,
            "seed": args.seed,
            "ckpt_dir": _display_path(args.ckpt_dir),
        },
        "environment": environment,
        "git": git,
        "skills": entries,
        "validation": validation,
        "distinct_skills": distinct,
        "duplicates": validation.get("duplicates", []),
        "learned_count": learned_count,
        "min_learned_skills": int(min_learned),
        "meets_min_learned": learned_count >= int(min_learned),
    }
    summary_path = args.ckpt_dir / "train_summary.json"
    atomic_write_json(summary_path, summary)

    try:
        from event_cbf_driving.eval.figures import plot_training_curves

        plot_training_curves(histories, args.fig_dir / "training-curves.png")
        print(f"\nWrote figures to {args.fig_dir / 'training-curves.png'}")
    except Exception as exc:  # pragma: no cover - plotting must not hide the summary
        logger.warning("could not write training-curve figure: %s", exc)
        summary["figure_error"] = str(exc)
        atomic_write_json(summary_path, summary)

    print(f"Checkpoints in {args.ckpt_dir}")
    print(f"Summary: {summary_path}")
    print(
        f"Learned distinct skills: {learned_count}/{len(distinct)} "
        f"(min {min_learned}, ok={summary['meets_min_learned']})"
    )
    if not validation.get("valid"):
        for problem in validation.get("problems", []):
            print(f"  VALIDATION PROBLEM: {problem}")
    return 0 if summary["meets_min_learned"] and validation.get("valid") else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="event-cbf train",
        description="Train and validate event-CBF driving skills",
    )
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--seed", type=int, default=None, help="Force a single seed (skip sweep)")
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "media" / "full-stack",
    )
    parser.add_argument(
        "--skills",
        nargs="+",
        choices=list(SKILL_NAMES),
        default=list(DEFAULT_SKILLS),
        help=(
            "skills to train; defaults to the two distinct policies "
            "(CRUISE_RIGHT and OVERTAKE_LEFT)"
        ),
    )
    parser.add_argument(
        "--min-learned-skills",
        type=int,
        default=None,
        help="minimum learned distinct skills required for success (default: TrainConfig)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate existing checkpoints and write validation.json; never train",
    )
    args = parser.parse_args(argv)
    if args.updates <= 0:
        parser.error("--updates must be > 0")
    if args.min_learned_skills is not None and args.min_learned_skills < 0:
        parser.error("--min-learned-skills must be >= 0")
    if args.verify_only:
        return _verify_only(args)
    return _train(args)


if __name__ == "__main__":
    raise SystemExit(main())
