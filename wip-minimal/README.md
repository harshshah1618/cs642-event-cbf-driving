# WIP minimal — 3-tier driving sketch (v0)

Throwaway demo that the interrupt → skill → pending-margin CBF → blend loop
actually runs on the Two-Trap Gauntlet. Rough on purpose; not paper material.
Standalone — does not import `event-cbf-driving`.

Wall time this run: **2.3s** on CPU.

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
- Collision: \|dx\| < 4.5 m and \|dy\| < 1.6 m.

## Four modes

1. **Unshielded cruise** — cruise PD only. Hits the stalled car.
2. **Static CBF** — cruise + CBF, inflation=0, no router. Freezes.
3. **Fixed-rate hard switch** — 1 Hz scripted router, hard switch after 300 ms, no shield. Can crash on the cut-in.
4. **Proposed** — shield-rise interrupt + pending margin + blend + CBF.

## Measured gauntlet (this run)

| Mode | Crash (s) | Route % | Avg speed | Freeze | Router calls | Peak jerk |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Unshielded cruise | 3.54 | 35.4 | 20.00 | no | 0 | 0.0 |
| 2 Static CBF (no router) | none | 34.5 | 5.75 | yes | 0 | 0.0 |
| 3 Fixed-rate hard switch | none | 100.0 | 20.00 | no | 10 | 252.8 |
| 4 Proposed interrupt+margin+blend | none | 100.0 | 17.05 | no | 2 | 32.9 |

Unshielded crashed=True @ x=70.8. Static freeze=True @ x=69.0, vx=0.00.
Fixed-rate crashed=False crash_time=None. Proposed crash=False freeze=False route=100.0%.

PD checks: cruise hold_err=0.0, acquire_t16=0.84;
overtake hold_err=0.0, acquire_t16=0.84.

Figure: `../media/wip-minimal-rollout.png`
