"""Calibrate critic disagreement on nominal physical rollouts, with held-out seeds.

The policy weights stay fixed. The threshold is an empirical nominal-exposure
quantile; neither sigma nor the exceedance rate is a collision probability.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from event_cbf_driving.config import ScenarioConfig, ShieldConfig
from event_cbf_driving.env.registry import make_env
from event_cbf_driving.io_utils import atomic_write_json
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.skills.checkpoints import save_skill, state_dict_fingerprint
from event_cbf_driving.skills.loader import load_skill_bank


def collect(policy, seeds):
    samples, episodes = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for acquire in (False, True):
            start_y = (4.0 - policy.meta.y_target) if acquire else policy.meta.y_target
            cfg = ScenarioConfig(
                "two_trap",
                simulator="metadrive",
                duration=6.0,
                seed=seed,
                params={
                    "empty_road": True,
                    "initial_y": start_y,
                    "initial_vy": float(rng.uniform(-0.2, 0.2)),
                    "ego_vx": float(rng.uniform(14, 22)),
                },
            )
            env = make_env(cfg.name, cfg)
            shield = InflatedCBFShield(config=ShieldConfig(road_boundary_guard=True))
            episode_samples = []
            try:
                env.reset(seed=seed)
                while True:
                    snap = env.snapshot()
                    ax, ay, sigma = policy.act(snap.ego.y, snap.ego.vy, snap.ego.vx)
                    raw = sigma * policy.meta.spread
                    episode_samples.append(raw)
                    result = shield.filter(np.array([ax, ay]), snap, sigma=0)
                    _, _, term, trunc, info = env.step(result.action)
                    if term or trunc:
                        break
                final = env.snapshot()
                episodes.append(
                    {
                        "seed": seed,
                        "acquire": acquire,
                        "ticks": len(episode_samples),
                        "crashed": final.crashed,
                        "final_lane_error": abs(final.ego.y - policy.meta.y_target),
                        "out_of_road": bool(info.get("plant", {}).get("out_of_road")),
                    }
                )
                if final.crashed or episodes[-1]["out_of_road"]:
                    raise RuntimeError(
                        "nominal calibration rollout failed; investigate before calibrating"
                    )
                samples.extend(episode_samples)
            finally:
                env.close()
    return np.array(samples, dtype=float), episodes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("checkpoints"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/metadrive-calibrated"))
    args = p.parse_args()
    train, validation = [101, 103, 107, 109], [211, 223, 227]
    assert not set(train) & set(validation)
    bank = load_skill_bank(args.source)
    artifact = {
        "schema_version": 1,
        "train_seeds": train,
        "validation_seeds": validation,
        "quantile": 0.99,
        "weights_updated": False,
        "skills": {},
    }
    for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT"):
        policy = bank[name]
        raw, episodes = collect(policy, train)
        held_out, held_episodes = collect(policy, validation)
        spread = max(float(np.std(raw)), 1e-6)
        sigma = raw / spread
        threshold = float(np.quantile(sigma, 0.99))
        meta = replace(
            policy.meta,
            spread=spread,
            threshold=threshold,
            sigma_mean=float(sigma.mean()),
            sigma_std=float(sigma.std()),
        )
        report = {
            "source_fingerprint": state_dict_fingerprint(policy.net.state_dict()),
            "original_meta": asdict(policy.meta),
            "calibrated_meta": asdict(meta),
            "samples": len(raw),
            "validation_samples": len(held_out),
            "train_exceedance": float(np.mean(sigma > threshold)),
            "validation_exceedance": float(np.mean(held_out / spread > threshold)),
            "original_validation_exceedance": float(
                np.mean(held_out / policy.meta.spread > policy.meta.threshold)
            ),
            "episodes": episodes,
            "validation_episodes": held_episodes,
        }
        artifact["skills"][name] = report
        save_skill(
            args.output / f"{name}.pt",
            policy.net,
            meta,
            provenance={"calibration": {"method": "nominal_99th_quantile", "train_seeds": train}},
            extra={"metadrive_calibration": report},
        )
        print(
            name,
            "threshold",
            threshold,
            "held-out exceedance",
            report["validation_exceedance"],
            flush=True,
        )
    atomic_write_json(args.output / "calibration.json", artifact)


if __name__ == "__main__":
    main()
