#!/usr/bin/env python3
"""Minimal WIP sketch for the 3-tier driving loop (v0).

  pip install -r requirements.txt
  python run_minimal.py

PD stand-ins for skills (not PPO yet). Runs interrupt → skill →
pending-margin CBF → short blend on the Two-Trap Gauntlet. Meant to finish
in under ~2 min on CPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

STORE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"
FIG = STORE / "media" / "wip-minimal-rollout.png"
RESULTS = HERE / "results.json"

DT = 0.02
AX_LO, AX_HI = -6.0, 3.5
AY_LO, AY_HI = -5.0, 5.0
AX_SAFE_LO, AX_SAFE_HI = -7.5, 3.5

HEADWAY, STANDOFF, CBF_ALPHA = 1.1, 6.0, 2.5
LANE_CORRIDOR = 1.85
SIGMA_INFL = 0.0  # PD has no critic σ; pending margin still active
PENDING_INFL = 0.30  # η · v · Δt with Δt=0.30 s → 0.30 * vx while pending

COLLIDE_DX, COLLIDE_DY = 4.5, 1.6
ROAD_M, GAUNTLET_STEPS = 200.0, 600
ROUTER_DELAY, BLEND_TICKS, ROUTER_PERIOD = 15, 20, 50  # 300 ms, blend, 1 Hz
SLOW_OBS = 15.0

CRUISE, OVERTAKE = "CRUISE_RIGHT", "OVERTAKE_LEFT"
LANE_Y = {CRUISE: 0.0, OVERTAKE: 4.0}

MODE_ORDER = ("unshielded_cruise", "static_cbf", "fixed_1hz", "proposed")
MODE_TITLE = {
    "unshielded_cruise": "1 Unshielded cruise",
    "static_cbf": "2 Static CBF (no router)",
    "fixed_1hz": "3 Fixed-rate hard switch",
    "proposed": "4 Proposed interrupt+margin+blend",
}


# ---------------------------------------------------------------------------
# Dynamics / CBF / router
# ---------------------------------------------------------------------------

def kinematics(x, y, vx, vy, ax, ay):
    x2 = x + vx * DT + 0.5 * ax * DT * DT
    y2 = y + vy * DT + 0.5 * ay * DT * DT
    vx2 = max(0.0, vx + ax * DT)
    vy2 = vy + ay * DT
    return x2, y2, vx2, vy2


def collide(x, y, ox, oy):
    return abs(ox - x) < COLLIDE_DX and abs(oy - y) < COLLIDE_DY


def cbf_shield(ax_rl, x, y, vx, obstacles, inflation):
    ax = float(ax_rl)
    for ox, oy, ov in obstacles:
        dx = ox - x
        if dx <= 0.0 or abs(oy - y) >= LANE_CORRIDOR:
            continue
        h = (dx - STANDOFF) - HEADWAY * (vx - ov) - inflation
        ax = min(ax, ((ov - vx) + CBF_ALPHA * h) / HEADWAY)
    ax_out = float(np.clip(ax, AX_SAFE_LO, AX_SAFE_HI))
    return ax_out, ax_out < float(ax_rl) - 1e-3


def router_decide(skill, x, obstacles):
    """Fake slow router: lane-threat rule. Delay is applied elsewhere (300 ms)."""
    lane = LANE_Y[skill]
    threat = any(
        0.0 < (ox - x) <= ROAD_M and abs(oy - lane) < LANE_CORRIDOR and ov < SLOW_OBS
        for ox, oy, ov in obstacles
    )
    if skill == CRUISE and threat:
        return OVERTAKE
    if skill == OVERTAKE and threat:
        return CRUISE
    return skill


def blend_alpha(i: int) -> float:
    u = i / max(BLEND_TICKS - 1, 1)
    return float(1.0 / (1.0 + np.exp(-10.0 * (u - 0.5))))


# ---------------------------------------------------------------------------
# PD skills (+ tiny torch wrapper so checkpoints are real .pt files)
# ---------------------------------------------------------------------------

class PDSkill(nn.Module):
    """Simple lane PD toward (y*, vx*=20). Placeholder until PPO skills exist."""

    def __init__(self, y_target: float):
        super().__init__()
        self.y_target = float(y_target)
        # Registered so state_dict() round-trips.
        self.ky = nn.Parameter(torch.tensor(2.8))
        self.kdy = nn.Parameter(torch.tensor(2.2))
        self.kv = nn.Parameter(torch.tensor(1.4))

    @torch.no_grad()
    def act(self, y, vy, vx):
        ay = float(self.ky) * (self.y_target - y) - float(self.kdy) * vy
        ax = float(self.kv) * (20.0 - vx)
        ax = float(np.clip(ax, AX_LO, AX_HI))
        ay = float(np.clip(ay, AY_LO, AY_HI))
        return ax, ay, 0.0  # σ=0 for PD


def train_and_save_skills():
    """No real training — just fix PD gains, dump .pt files, run hold/acquire."""
    CKPT.mkdir(parents=True, exist_ok=True)
    skills = {CRUISE: PDSkill(0.0), OVERTAKE: PDSkill(4.0)}
    meta = {}
    t0 = time.time()
    for name, net in skills.items():
        torch.save(net.state_dict(), CKPT / f"{name}.pt")
        # Hold at setpoint + acquire from other lane (deterministic PD).
        y, vy, vx = net.y_target, 0.0, 20.0
        for _ in range(200):
            ax, ay, _ = net.act(y, vy, vx)
            _, y, vx, vy = kinematics(0.0, y, vx, vy, ax, ay)
        hold_err = abs(y - net.y_target)
        other = 4.0 if net.y_target < 2 else 0.0
        y, vy, vx = other, 0.0, 20.0
        t16 = None
        for i in range(250):
            ax, ay, _ = net.act(y, vy, vx)
            _, y, vx, vy = kinematics(0.0, y, vx, vy, ax, ay)
            if t16 is None and abs(y - other) >= 1.6:
                t16 = (i + 1) * DT
        meta[name] = {
            "y_target": net.y_target,
            "kind": "PD_stand_in",
            "hold_err": round(hold_err, 4),
            "acquire_t16": None if t16 is None else round(t16, 3),
            "final_y": round(y, 3),
            "threshold": 1.0,  # unused (σ=0); interrupt uses shield rise
            "spread": 1.0,
        }
        (CKPT / f"{name}.json").write_text(json.dumps(meta[name], indent=2))
        print(
            f"  {name} PD saved  hold_err={hold_err:.3f}  acquire_t16={t16}  "
            f"final_y={y:.2f}",
            flush=True,
        )
    print(f"skill setup {time.time() - t0:.2f}s", flush=True)
    return skills, meta


def load_skills():
    skills = {}
    meta = {}
    for name, yt in ((CRUISE, 0.0), (OVERTAKE, 4.0)):
        net = PDSkill(yt)
        try:
            state = torch.load(CKPT / f"{name}.pt", map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(CKPT / f"{name}.pt", map_location="cpu")
        net.load_state_dict(state)
        net.eval()
        skills[name] = net
        meta[name] = json.loads((CKPT / f"{name}.json").read_text())
    return skills, meta


# ---------------------------------------------------------------------------
# Gauntlet
# ---------------------------------------------------------------------------

def run_gauntlet(mode, skills):
    x = y = vy = 0.0
    vx = 20.0
    committed = CRUISE
    delay = 0
    delay_to = None
    blend = None
    calls = 0
    call_log = []
    override_off = 100
    since_call = 1000
    spawned = False
    cut = None
    prev_ay = 0.0
    peak_jerk = 0.0
    hist = {k: [] for k in ("t", "x", "y", "vx", "ay", "skill", "shield", "cutx")}
    crashed = False
    end_step = GAUNTLET_STEPS

    def obstacles():
        obs = [(75.0, 0.0, 0.0)]
        if cut is not None:
            obs.append((cut[0], cut[1], cut[2]))
        return obs

    def log(t, skill, shield):
        hist["t"].append(t)
        hist["x"].append(x)
        hist["y"].append(y)
        hist["vx"].append(vx)
        hist["ay"].append(prev_ay)
        hist["skill"].append(skill)
        hist["shield"].append(shield)
        hist["cutx"].append(np.nan if cut is None else cut[0])

    log(0.0, committed, False)

    for step in range(GAUNTLET_STEPS):
        t = step * DT
        obs_list = obstacles()
        if (not spawned) and x >= 92.0:
            spawned = True
            cut = [x + 23.0, 4.0, 9.5]
            obs_list = obstacles()

        pending = delay > 0
        if delay > 0:
            delay -= 1
            if delay == 0:
                if mode == "fixed_1hz":
                    committed = delay_to
                else:
                    blend = [0, committed, delay_to]
                    committed = delay_to

        def act_of(skill):
            return skills[skill].act(y, vy, vx)

        if mode in ("unshielded_cruise", "static_cbf"):
            ax, ay, _ = act_of(CRUISE)
            skill_label = CRUISE
        elif blend is not None:
            src, dst = blend[1], blend[2]
            ax0, ay0, _ = act_of(src)
            ax1, ay1, _ = act_of(dst)
            a = blend_alpha(blend[0])
            ax = (1.0 - a) * ax0 + a * ax1
            ay = (1.0 - a) * ay0 + a * ay1
            skill_label = "BLEND"
            blend[0] += 1
            if blend[0] >= BLEND_TICKS:
                blend = None
        else:
            ax, ay, _ = act_of(committed)
            skill_label = committed

        ax_rl = float(ax)
        ay = float(np.clip(ay, AY_LO, AY_HI))
        overriding = False
        if mode in ("static_cbf", "proposed"):
            infl = 0.0
            if mode == "proposed":
                # Pending margin: locked skill intent during router delay — NOT stale sensors.
                infl = SIGMA_INFL + (PENDING_INFL * vx if pending else 0.0)
            ax, overriding = cbf_shield(ax_rl, x, y, vx, obs_list, infl)
        else:
            ax = float(np.clip(ax_rl, AX_LO, AX_HI))

        if mode == "fixed_1hz" and step % ROUTER_PERIOD == 0:
            calls += 1
            dec = router_decide(committed, x, obs_list)
            call_log.append({"t": round(t, 3), "from": committed, "to": dec, "reason": "1hz"})
            if dec != committed and delay == 0 and blend is None:
                delay = ROUTER_DELAY
                delay_to = dec
        elif mode == "proposed":
            shield_rise = overriding and override_off >= 8
            can_call = delay == 0 and blend is None and since_call >= 10
            if can_call and shield_rise:
                calls += 1
                since_call = 0
                dec = router_decide(committed, x, obs_list)
                call_log.append({"t": round(t, 3), "from": committed, "to": dec, "reason": "shield"})
                if dec != committed:
                    delay = ROUTER_DELAY
                    delay_to = dec
            override_off = 0 if overriding else override_off + 1
            since_call += 1

        peak_jerk = max(peak_jerk, abs(ay - prev_ay) / DT)
        prev_ay = ay
        x, y, vx, vy = kinematics(x, y, vx, vy, ax, ay)
        if cut is not None:
            cut[0] += cut[2] * DT
        log((step + 1) * DT, skill_label, overriding)

        hit = any(collide(x, y, ox, oy) for ox, oy, _ in obstacles())
        if hit or x >= ROAD_M:
            crashed = hit
            end_step = step + 1
            break

    end_t = end_step * DT
    route = 100.0 * min(x, ROAD_M) / ROAD_M
    freeze = (not crashed) and vx < 0.5 and route < 90.0
    for k, v in list(hist.items()):
        if k != "skill":
            hist[k] = np.asarray(v, dtype=np.float64)
    return {
        "mode": mode,
        "crash_time": round(end_t, 4) if crashed else None,
        "crashed": crashed,
        "route_pct": round(route, 2),
        "avg_speed": round(float(x / end_t) if end_t > 0 else 0.0, 3),
        "freeze": freeze,
        "router_calls": calls,
        "peak_lateral_jerk": round(peak_jerk, 2),
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "vx": round(float(vx), 3),
        "end_time": round(end_t, 4),
        "calls": call_log,
        "hist": hist,
    }


# ---------------------------------------------------------------------------
# Figure / table / README
# ---------------------------------------------------------------------------

def _cars(ax, stalled=True, cut=None):
    if stalled:
        ax.add_patch(
            FancyBboxPatch(
                (75.0 - 2.25, -0.8), 4.5, 1.6,
                boxstyle="round,pad=0.02,rounding_size=0.3",
                facecolor="#4a4a4a", edgecolor="black", lw=0.4, zorder=3,
            )
        )
    if cut is not None and np.isfinite(cut):
        ax.add_patch(
            Rectangle((cut - 2.25, 4.0 - 0.8), 4.5, 1.6,
                      facecolor="#b03a2e", edgecolor="black", lw=0.4, zorder=3)
        )


def plot_rollouts(results, path):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), dpi=140, sharex=True, sharey=True)
    for ax, mode in zip(axes.ravel(), MODE_ORDER):
        r = results[mode]
        h = r["hist"]
        ax.axhspan(-1.6, 5.6, color="#eef2f6", zorder=0)
        ax.axhline(0.0, color="0.55", ls="--", lw=0.7)
        ax.axhline(4.0, color="0.55", ls="--", lw=0.7)
        ax.plot(h["x"], h["y"], color="#1f4e79", lw=1.6, zorder=2)
        _cars(ax, True, None)
        cut = h["cutx"]
        finite = np.isfinite(cut)
        if finite.any():
            ax.plot(cut[finite], np.full(finite.sum(), 4.0), color="#b03a2e", lw=1.0, alpha=0.8)
            _cars(ax, False, float(cut[finite][-1]))
        if r["crashed"]:
            ax.scatter([r["x"]], [r["y"]], marker="x", s=46, color="#9b1c1c", zorder=4)
        crash = "no crash" if r["crash_time"] is None else f"crash {r['crash_time']:.2f}s"
        freeze = " · freeze" if r["freeze"] else ""
        ax.set_title(
            f"{MODE_TITLE[mode]}\n{crash}{freeze} · route {r['route_pct']:.1f}%",
            fontsize=10,
        )
        ax.set_xlim(0, 210)
        ax.set_ylim(-2.2, 6.6)
        ax.grid(True, color="0.88", lw=0.5)
    axes[1, 0].set_xlabel("x (m)")
    axes[1, 1].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")
    fig.suptitle("WIP minimal — Two-Trap Gauntlet (PD skills)", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def print_table(results):
    headers = ["Mode", "Crash (s)", "Route %", "Avg speed", "Freeze", "Calls", "Peak jerk"]
    print("| " + " | ".join(headers) + " |", flush=True)
    print("| " + " | ".join("---" for _ in headers) + " |", flush=True)
    rows = []
    for mode in MODE_ORDER:
        r = results[mode]
        crash = "none" if r["crash_time"] is None else f"{r['crash_time']:.2f}"
        line = (
            f"| {MODE_TITLE[mode]} | {crash} | {r['route_pct']:.1f} | {r['avg_speed']:.2f} | "
            f"{'yes' if r['freeze'] else 'no'} | {r['router_calls']} | {r['peak_lateral_jerk']:.1f} |"
        )
        print(line, flush=True)
        rows.append(line)
    return rows


def write_readme(results, meta, wall_s):
    def row(mode):
        r = results[mode]
        crash = "none" if r["crash_time"] is None else f"{r['crash_time']:.2f}"
        return (
            f"| {MODE_TITLE[mode]} | {crash} | {r['route_pct']:.1f} | {r['avg_speed']:.2f} | "
            f"{'yes' if r['freeze'] else 'no'} | {r['router_calls']} | {r['peak_lateral_jerk']:.1f} |"
        )

    m1, m2, m3, m4 = [results[m] for m in MODE_ORDER]
    text = f"""# WIP minimal — 3-tier driving sketch (v0)

Throwaway demo that the interrupt → skill → pending-margin CBF → blend loop
actually runs on the Two-Trap Gauntlet. Rough on purpose; not paper material.
Standalone — does not import `event-cbf-driving`.

Wall time this run: **{wall_s:.1f}s** on CPU.

## How to run

```bash
cd wip-minimal
pip install -r requirements.txt
python run_minimal.py
```

Should finish in under 2 min on CPU (usually a few seconds).

## Real vs stub

| Piece | Status |
| --- | --- |
| `CRUISE_RIGHT` / `OVERTAKE_LEFT` | **Stub** — PD stand-ins for now. Swap for PPO later. Checkpoints are real `.pt` files. |
| Two-Trap Gauntlet (stalled + cut-in) | **Real** closed-loop kinematics |
| Static / pending-inflated CBF | **Real** analytic filter |
| Short sigmoid blend | **Real** |
| Skill router | **Stub / scripted** — lane-threat rule + 300 ms delay (fine for WIP) |
| Sensors | **Stub** — ground-truth state every tick |

## Pending margin

While a router call is in flight, skill intent is locked — we keep running the
old skill until the delayed answer arrives. Pending-call CBF margin is
`0.30 · vx` during that window so we keep extra standoff. Not a stale-sensor
model; sensors here are ground truth every tick.

## Scenario

- Road 200 m, 12 s, dt = 0.02 s. Ego starts x=0, y=0, vx=20.
- Trap 1: stalled car at x=75, y=0.
- Trap 2: when ego x≥92, cut-in at y=4, 9.5 m/s, 23 m ahead. Without pending
  inflation, a hard switch can hit it during the 300 ms delay.
- Collision: \\|dx\\| < 4.5 m and \\|dy\\| < 1.6 m.

## Four modes

1. **Unshielded cruise** — cruise PD only. Hits the stalled car.
2. **Static CBF** — cruise + CBF, inflation=0, no router. Freezes.
3. **Fixed-rate hard switch** — 1 Hz scripted router, hard switch after 300 ms, no shield. Can crash on the cut-in.
4. **Proposed** — shield-rise interrupt + pending margin + blend + CBF.

## Measured gauntlet (this run)

| Mode | Crash (s) | Route % | Avg speed | Freeze | Router calls | Peak jerk |
| --- | --- | --- | --- | --- | --- | --- |
{row('unshielded_cruise')}
{row('static_cbf')}
{row('fixed_1hz')}
{row('proposed')}

Unshielded crashed={m1['crashed']} @ x={m1['x']:.1f}. Static freeze={m2['freeze']} @ x={m2['x']:.1f}, vx={m2['vx']:.2f}.
Fixed-rate crashed={m3['crashed']} crash_time={m3['crash_time']}. Proposed crash={m4['crashed']} freeze={m4['freeze']} route={m4['route_pct']:.1f}%.

PD checks: cruise hold_err={meta[CRUISE]['hold_err']}, acquire_t16={meta[CRUISE]['acquire_t16']};
overtake hold_err={meta[OVERTAKE]['hold_err']}, acquire_t16={meta[OVERTAKE]['acquire_t16']}.

Figure: `../media/wip-minimal-rollout.png`
"""
    (HERE / "README.md").write_text(text)


def self_check():
    x, y, vx, vy = kinematics(0.0, 0.0, 20.0, 0.0, 1.0, 2.0)
    assert abs(x - (20 * DT + 0.5 * DT * DT)) < 1e-9
    # Static shield must stop before collision.
    x = y = vy = 0.0
    vx = 20.0
    for _ in range(600):
        ax, _ = cbf_shield(0.0, x, y, vx, [(75.0, 0.0, 0.0)], 0.0)
        x, y, vx, vy = kinematics(x, y, vx, vy, ax, 0.0)
        if collide(x, y, 75.0, 0.0):
            raise RuntimeError(f"static shield collided x={x:.2f}")
    if vx > 0.3 or not (60.0 < x < 70.5):
        raise RuntimeError(f"static shield did not hold x={x:.2f} vx={vx:.2f}")
    print("self-check ok", flush=True)


def main():
    t0 = time.time()
    self_check()
    print("saving PD skill checkpoints", flush=True)
    train_and_save_skills()
    skills, meta = load_skills()

    results = {}
    for mode in MODE_ORDER:
        results[mode] = run_gauntlet(mode, skills)
        r = results[mode]
        print(
            f"{mode:20s} crash={r['crash_time']} route={r['route_pct']:.1f} "
            f"freeze={r['freeze']} calls={r['router_calls']} "
            f"x={r['x']:.1f} y={r['y']:.2f} vx={r['vx']:.2f}",
            flush=True,
        )
        for c in r["calls"]:
            print(f"    call {c}", flush=True)

    print("\n=== Measured table ===", flush=True)
    print_table(results)
    plot_rollouts(results, FIG)
    wall = time.time() - t0
    write_readme(results, meta, wall)
    public = []
    for mode in MODE_ORDER:
        row = dict(results[mode])
        row.pop("hist", None)
        public.append(row)
    RESULTS.write_text(json.dumps({"rollouts": public, "meta": meta, "wall_s": round(wall, 2)}, indent=2))
    print(f"\ndone in {wall:.1f}s", flush=True)
    print("wrote", FIG, HERE / "README.md", RESULTS, flush=True)


if __name__ == "__main__":
    main()
