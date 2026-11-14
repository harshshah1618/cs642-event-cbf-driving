"""First closed loop with the VLM router: stalled-car (two-trap) scenario, a few seeds, three routers.

  rule_instant   label rule on simulator ground truth, answer available at once
  rule_delayed   same rule, each answer delayed by a latency the VLM actually took in this run (replayed in order)
  vlm            render bridge (MetaDrive front frame) + fine-tuned Qwen2.5-VL-3B; ready_at = t + measured latency

Everything else is the proposed stack unchanged: event-triggered calls (sigma or shield rising edge), PPO skills
with blending, inflated CBF shield. rule_instant vs rule_delayed = cost of the delay; rule_delayed vs vlm = cost
of perception errors.
"""
import json
import os
import statistics as st
from pathlib import Path

from event_cbf_driving.env.two_trap import TwoTrapGauntletEnv
from event_cbf_driving.router.vlm import LabelRuleRouter, VLMRouter, snapshot_to_scene
from event_cbf_driving.shield.cbf import InflatedCBFShield
from event_cbf_driving.skills.ppo import load_skill
from event_cbf_driving.stack.controller import DrivingStack, StackConfig, StackMode


def load_skills(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    skills = {}
    for name in ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT"):
        p = ckpt_dir / f"{name}.pt"
        if not p.exists():  # some branches keep only CRUISE_RIGHT/OVERTAKE_LEFT; RETURN_RIGHT = CRUISE_RIGHT then
            p = ckpt_dir / "CRUISE_RIGHT.pt"
        skills[name] = load_skill(p)
        skills[name].name = name
    return skills


class GapTracker:
    """Env proxy that records the smallest bumper gap to a car ahead in the ego corridor (same test as the shield)."""

    def __init__(self, env, lane_half_width=1.85):
        self.env, self.w, self.min_gap = env, lane_half_width, float("inf")

    def __getattr__(self, k):
        return getattr(self.env, k)

    def _track(self, snap):
        for o in snap.obstacles_ahead(self.w):
            self.min_gap = min(self.min_gap, o.x - snap.ego.x)

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.min_gap = float("inf")
        self._track(self.env.snapshot())
        return out

    def step(self, a):
        out = self.env.step(a)
        self._track(out[-1].get("snapshot") or self.env.snapshot())
        return out


def episode(skills, router, seed, config=None, shield=None):
    env = GapTracker(TwoTrapGauntletEnv(randomize=True))
    stack = DrivingStack(skills=skills, router=router, shield=shield or InflatedCBFShield(enabled=True, inflate=True),
                         config=config or StackConfig(mode=StackMode.PROPOSED, use_blend=True, hard_switch=False))
    r = stack.run(env, seed=seed)
    layout = env.snapshot().info["layout"]
    k = max(1, len(r.xs) // 150)  # keep ~150 trajectory points for plots
    return dict(seed=seed, layout={a: round(b, 2) for a, b in layout.items()}, crashed=r.crashed,
                crash_time=r.crash_time, route_pct=round(r.route_pct, 1), avg_speed=round(r.avg_speed, 2),
                freeze=r.freeze, router_calls=r.router_calls, peak_lateral_jerk=round(r.peak_lateral_jerk, 1),
                min_gap=round(env.min_gap, 2), end_time=round(r.end_time, 2), calls=r.calls,
                xs=[round(v, 2) for v in r.xs[::k]], ys=[round(v, 2) for v in r.ys[::k]], skills=r.skills[::k])


def summarize(eps):
    n = len(eps)
    return dict(n=n, crashes=sum(e["crashed"] for e in eps), freezes=sum(e["freeze"] for e in eps),
                mean_route_pct=round(sum(e["route_pct"] for e in eps) / n, 1),
                mean_avg_speed=round(sum(e["avg_speed"] for e in eps) / n, 2),
                mean_router_calls=round(sum(e["router_calls"] for e in eps) / n, 2),
                mean_peak_jerk=round(sum(e["peak_lateral_jerk"] for e in eps) / n, 1),
                min_gap=round(min(e["min_gap"] for e in eps), 2),
                route_pct_sd=round(st.pstdev(e["route_pct"] for e in eps), 1))


def plot(results, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    seeds = [e["seed"] for e in results[names[0]]["episodes"]][:5]  # first 5 seeds only
    fig, axes = plt.subplots(len(seeds), 1, figsize=(10, 1.6 * len(seeds)), sharex=True, squeeze=False)
    colors = {"rule_instant": "#1f77b4", "rule_delayed": "#ff7f0e", "vlm": "#2ca02c", "vlm_diag": "#d62728"}
    for ax, seed in zip(axes[:, 0], seeds):
        for name in names:
            e = next(x for x in results[name]["episodes"] if x["seed"] == seed)
            ax.plot(e["xs"], e["ys"], color=colors.get(name), lw=1.4, label=name)
            if e["crashed"]:
                ax.scatter([e["xs"][-1]], [e["ys"][-1]], marker="x", color=colors.get(name), s=60, zorder=6)
        L = e["layout"]
        ax.scatter([L["stalled_x"]], [0.0], c="k", s=40, zorder=5)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axhline(4, color="0.85", lw=0.8)
        ax.set_ylim(-1.5, 5.5)
        ax.set_ylabel(f"seed {seed}\ny (m)")
    axes[0, 0].legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("x (m)   black dot = stalled car (right lane, y=0); x = crash")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def run(policy, renderer, seeds, out_dir, ckpt_dir, meta=None, vlm_variants=(("vlm", None),)):
    """vlm_variants: (name, right_lane_skill) pairs. ("vlm_diag", "CRUISE_RIGHT") is the skill-text diagnostic:
    the VLM is told CRUISE_RIGHT whenever the ego is in the right lane (the only pairing it saw in training).
    rule_delayed replays the latencies of the first variant."""
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    skills = load_skills(ckpt_dir)

    # warm-up: first render resets MetaDrive (~5 s); first model calls compile kernels and fill the prefix cache
    env = TwoTrapGauntletEnv(randomize=True)
    env.reset(seed=0)
    png = renderer(snapshot_to_scene(env.snapshot()))
    for _ in range(3):
        policy(png, "RIGHT", "CRUISE_RIGHT")

    results = {}
    lat = None
    for vname, right_lane_skill in vlm_variants:
        def sink(name, data, vname=vname):
            (out_dir / "frames" / f"{vname}_{name}").write_bytes(data)

        vlm = VLMRouter(renderer, policy, frame_sink=sink, right_lane_skill=right_lane_skill)
        eps = []
        for s in seeds:
            vlm.start_episode(f"seed{s}", car_seed=s)
            eps.append(episode(skills, vlm, s))
            print(f"{vname} seed {s}: crash={eps[-1]['crashed']} route={eps[-1]['route_pct']} "
                  f"calls={eps[-1]['router_calls']}", flush=True)
        v_lat = [c["infer_ms"] / 1000 for c in vlm.log]
        lat = v_lat if lat is None else lat
        results[vname] = dict(episodes=eps, summary=summarize(eps))
        calls_ok = [c["ok"] for c in vlm.log]
        results[vname]["summary"].update(
            right_lane_skill=right_lane_skill,
            call_agreement_with_rule=round(sum(calls_ok) / max(1, len(calls_ok)), 3), n_calls=len(vlm.log),
            median_infer_ms=round(st.median(v_lat) * 1000, 1) if v_lat else None,
            p90_infer_ms=round(sorted(v_lat)[int(0.9 * (len(v_lat) - 1))] * 1000, 1) if v_lat else None,
            median_render_ms=round(st.median(c["render_ms"] for c in vlm.log), 1) if vlm.log else None)
        results[vname]["call_log"] = vlm.log

    for name, router in (("rule_instant", LabelRuleRouter(0.0)), ("rule_delayed", LabelRuleRouter(lat or 0.125))):
        eps = [episode(skills, router, s) for s in seeds]
        results[name] = dict(episodes=eps, summary=summarize(eps))
    results["rule_delayed"]["summary"]["latency_source"] = \
        f"{len(lat or [])} measured VLM latencies ({vlm_variants[0][0]}), replayed in order"

    results = {k: results[k] for k in ("rule_instant", "rule_delayed", *[v for v, _ in vlm_variants])}
    for k, v in results.items():
        print(f"{k:13s} {json.dumps(v['summary'])}", flush=True)
    plot(results, out_dir / "trajectories.png")
    results["meta"] = meta or {}
    print("meta", json.dumps(results["meta"]), flush=True)
    json.dump(results, open(out_dir / "closed_loop.json", "w"), indent=1)
    return results
