"""Audit saved VLM drives with fixed recorded advice, without model inference.

This is a deterministic reconstruction, not a fresh VLM evaluation. Every
request time, recorded object offset, applied answer and saved episode endpoint
is checked. Failed checks leave the replay explicitly unverified.
"""

import argparse
import json
import math
from pathlib import Path

from driving.irregular import escalation


class RecordedAdvice:
    def __init__(self, calls):
        self.records = calls
        self.index = 0
        self.now = None
        self.offset_errors = []
        self.face_matches = []

    def triggers(self, now):
        self.now = now
        if self.index == len(self.records):
            return []
        call = self.records[self.index]
        if now > call["t"] + 1e-6:
            raise ValueError(f"Missed recorded request at {call['t']}; now {now}")
        return call["triggers"] if abs(now - call["t"]) < 1e-6 else []

    def __call__(self, image, state, crop, truth):
        if self.index == len(self.records):
            raise ValueError("Replay requested an unrecorded answer")
        call = self.records[self.index]
        if not math.isclose(self.now, call["t"], abs_tol=1e-6):
            raise ValueError(f"Request timing changed: {self.now} vs {call['t']}")
        recorded_forward = call.get("forward")
        forward = truth["forward"] if truth else None
        if recorded_forward is not None and forward is not None:
            self.offset_errors.append(abs(forward - recorded_forward))
        elif recorded_forward != forward:
            raise ValueError("Recorded and replay object context differ")
        self.face_matches.append(call.get("face") == (truth.get("face") if truth else None))
        self.index += 1
        return call["action"], call["seconds"]


def compare(original, replay, advice, tolerance=1e-6):
    exact = ("steps", "cost", "arrive", "passed_stop", "seconds_stopped_near_object",
             "reached_object")
    matches = {key: original[key] == replay[key] for key in exact}
    errors = {}
    for key in ("route_completion", "pass_speed", "min_gap"):
        a, b = original[key], replay[key]
        errors[key] = abs(a - b) if a is not None and b is not None else (0. if a == b else None)
    matches["continuous_endpoints"] = all(v is not None and v <= tolerance for v in errors.values())
    matches["all_requests_consumed"] = advice.index == len(advice.records)
    matches["request_offsets"] = max(advice.offset_errors, default=0.) <= tolerance
    matches["sign_faces"] = all(advice.face_matches)
    matches["applied_answers"] = len(original["calls"]) == len(replay["calls"]) and all(
        a.get("applied") == b.get("applied") and a.get("applied_t") == b.get("applied_t")
        for a, b in zip(original["calls"], replay["calls"], strict=False))
    return dict(verified=all(matches.values()), matches=matches, endpoint_errors=errors,
                max_request_offset_error=max(advice.offset_errors, default=0.),
                checked_requests=advice.index)


def replay_type(base_class, advice):
    class ReplayEscalation(base_class):
        def _triggers(self, now, driver, truth, frame_step, image):
            return advice.triggers(now)
    return ReplayEscalation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--scenes", nargs="+")
    args = parser.parse_args()
    from driving.irregular.scene import Placement, make_env

    originals = [r for p in sorted(args.source.glob("*.json")) for r in json.loads(p.read_text())]
    originals = [r for r in originals if (not args.seeds or r["seed"] in args.seeds)
                 and (not args.scenes or r["scene"] in args.scenes)]
    args.output.mkdir(parents=True, exist_ok=True)
    base_class = escalation.Escalation
    results = []
    env = make_env(escalation.irregular_config("irregular-test"))
    try:
        for original in originals:
            advice = RecordedAdvice(original["calls"])

            escalation.Escalation = replay_type(base_class, advice)
            row = escalation.run_episode(
                env, original["seed"], [Placement(**p) for p in original["placements"]],
                escalation.EscalationConfig(**original["config"]), vlm=advice, detector=object(),
                driver_kwargs=dict(skill_checkpoint=str(args.checkpoint)),
                max_steps=300 if original["scene"] in ("clean", "rain-clean") else 600)
            row.update(scene=original["scene"], model="recorded-Qwen-advice-replay",
                       replay_audit=compare(original, row, advice))
            results.append(row)
            (args.output / "episodes.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(dict(scene=row["scene"], seed=row["seed"],
                                  audit=row["replay_audit"], collision=bool(row["collision_events"]))),
                  flush=True)
    finally:
        escalation.Escalation = base_class
        env.close()


if __name__ == "__main__":
    main()
