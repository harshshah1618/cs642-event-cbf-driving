"""Delay sweep and ablations with the ground-truth rule router (no GPU, no VLM).

Same scenario as loop_eval.py (two-trap, randomized layout per seed), same PPO skills, same proposed stack.
The router is LabelRuleRouter: the rule the VLM was trained to imitate, applied to simulator ground truth, with
a chosen answer delay. So any change in the numbers comes from the delay or from the stack variant, never from
perception errors.

  delay sweep   proposed stack, delay in DELAYS (+ "vlm" = the VLM's measured latencies, replayed in order)
  ablations     each VARIANT at each delay in ABLATION_DELAYS

Variants (everything not named is the proposed stack):
  proposed               event-triggered calls, blend, CBF inflated by sigma margin + pending-call margin
  no_pending_margin      pending-call margin off (eta_latency = 0)
  no_sigma_margin        uncertainty margin off (gamma_sigma = 0)
  no_margins             both off: plain CBF, skills still switched by the router
  matched_pending_margin pending margin uses the true delay instead of the default 0.30 s
  constant_margin        a fixed margin on every tick, equal to the proposed stack's mean margin at that delay
  hard_switch            no blend: the new skill takes over at once
  periodic_1hz/2hz       router called on a timer instead of on sigma/shield events (shield + blend unchanged)
  fixed_1hz_baseline     the repo's FIXED_VLM baseline: 1 Hz timer, hard switch, shield OFF
"""
import statistics as st

DELAYS = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, "vlm"]
ABLATION_DELAYS = ["vlm", 0.5, 1.0]
VARIANTS = ["proposed", "no_pending_margin", "no_sigma_margin", "no_margins", "matched_pending_margin",
            "constant_margin", "hard_switch", "periodic_1hz", "periodic_2hz", "fixed_1hz_baseline"]


def _shields():
    from event_cbf_driving.shield.cbf import InflatedCBFShield

    class RecordingShield(InflatedCBFShield):
        """Proposed shield that also records the margin it applied on every tick."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.margins = []

        def inflation(self, sigma, vlm_pending, vx):
            m = super().inflation(sigma, vlm_pending, vx)
            self.margins.append(m)
            return m

    class ConstantShield(InflatedCBFShield):
        def __init__(self, margin, **kw):
            super().__init__(**kw)
            self.margin = margin

        def inflation(self, sigma, vlm_pending, vx):
            return self.margin

    return RecordingShield, ConstantShield


def _parts(variant, delay_s, const_margin=None):
    from event_cbf_driving.stack.controller import StackConfig, StackMode

    RecordingShield, ConstantShield = _shields()
    cfg = dict(mode=StackMode.PROPOSED, use_blend=True, hard_switch=False)
    sh = dict(enabled=True, inflate=True)
    if variant == "no_pending_margin":
        sh["eta_latency"] = 0.0
    elif variant == "no_sigma_margin":
        sh["gamma_sigma"] = 0.0
    elif variant == "no_margins":  # DrivingStack forces inflate=True in PROPOSED mode, so zero both gains
        sh.update(eta_latency=0.0, gamma_sigma=0.0)
    elif variant == "matched_pending_margin":
        sh["vlm_delay_s"] = delay_s
    elif variant == "hard_switch":
        cfg["hard_switch"] = True
    elif variant == "periodic_1hz":
        cfg["periodic_hz"] = 1.0
    elif variant == "periodic_2hz":
        cfg["periodic_hz"] = 2.0
    elif variant == "fixed_1hz_baseline":
        cfg = dict(mode=StackMode.FIXED_VLM, fixed_rate_hz=1.0)
    if variant == "constant_margin":
        shield = ConstantShield(const_margin, **sh)
    else:
        shield = RecordingShield(**sh)
    return StackConfig(**cfg), shield


def _slim(e):
    return {k: v for k, v in e.items() if k not in ("xs", "ys", "skills", "calls")}


def summarize(eps):
    n = len(eps)
    r = [e["route_pct"] for e in eps]
    return dict(n=n, crashes=sum(e["crashed"] for e in eps), freezes=sum(e["freeze"] for e in eps),
                mean_route_pct=round(st.mean(r), 1), sd_route_pct=round(st.pstdev(r), 1),
                se_route_pct=round(st.pstdev(r) / n ** 0.5, 2),
                mean_calls=round(st.mean(e["router_calls"] for e in eps), 2),
                mean_peak_jerk=round(st.mean(e["peak_lateral_jerk"] for e in eps), 1),
                min_gap=round(min(e["min_gap"] for e in eps), 2),
                mean_min_gap=round(st.mean(min(e["min_gap"], 200.0) for e in eps), 2))


def job(variant, delay, seeds, ckpt_dir, vlm_latencies):
    """One (variant, delay) cell over all seeds. delay is seconds or "vlm"."""
    from event_cbf_driving.router.vlm import LabelRuleRouter
    from loop_eval import episode, load_skills

    skills = load_skills(ckpt_dir)
    lat = list(vlm_latencies) if delay == "vlm" else float(delay)
    delay_s = st.median(vlm_latencies) if delay == "vlm" else float(delay)
    const = None
    extra = {}
    if variant == "constant_margin":  # mean margin the proposed stack used at this delay, over all ticks
        margins = []
        router = LabelRuleRouter(lat)
        for s in seeds:
            cfg, sh = _parts("proposed", delay_s)
            episode(skills, router, s, cfg, sh)
            margins += sh.margins
        const = st.mean(margins)
        extra["constant_margin_m"] = round(const, 3)
    router = LabelRuleRouter(lat)
    eps = []
    for s in seeds:
        cfg, sh = _parts(variant, delay_s, const)
        eps.append(_slim(episode(skills, router, s, cfg, sh)))
    return dict(variant=variant, delay=delay, delay_s=round(delay_s, 4), summary={**summarize(eps), **extra},
                episodes=eps)


def plot(cells, out):  # runs on Modal (matplotlib is not installed locally)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = sorted([c for c in cells if c["variant"] == "proposed" and c["delay"] != "vlm"], key=lambda c: c["delay_s"])
    vlm = next((c for c in cells if c["variant"] == "proposed" and c["delay"] == "vlm"), None)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = [c["delay_s"] for c in sweep]
    m = [c["summary"]["mean_route_pct"] for c in sweep]
    se = [c["summary"]["se_route_pct"] for c in sweep]
    ax[0].errorbar(x, m, yerr=[2 * v for v in se], marker="o", capsize=3, color="#1f77b4", label="rule, fixed delay")
    if vlm:
        ax[0].errorbar([vlm["delay_s"]], [vlm["summary"]["mean_route_pct"]], yerr=[2 * vlm["summary"]["se_route_pct"]],
                       marker="s", color="#2ca02c", capsize=3, ls="none", label="rule, VLM's measured latencies")
    ax[0].set_xlabel("router answer delay (s)")
    ax[0].set_ylabel("route completion (%)  mean ± 2 SE")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, [c["summary"]["crashes"] for c in sweep], marker="o", label="crashes", color="#d62728")
    ax[1].plot(x, [c["summary"]["freezes"] for c in sweep], marker="o", label="freezes", color="#7f7f7f")
    ax[1].set_xlabel("router answer delay (s)")
    ax[1].set_ylabel(f"episodes (of {sweep[0]['summary']['n']})")
    ax[1].legend(fontsize=8)
    fig.suptitle("Two-trap, proposed stack, ground-truth rule router: effect of answer delay")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
