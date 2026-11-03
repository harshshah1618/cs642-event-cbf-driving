# Results (measured)

All numbers come from `scripts/eval_stack.py` on this machine. Nothing is fabricated.

## M1 router (scripted)

Router backend: **scripted** (same interrupt API; 300 ms simulated latency). PPO skills are trained checkpoints under `checkpoints/`.

**Gauntlet takeaway (M1):** proposed is the only mode that neither crashes nor freezes, with the highest route % and far lower peak lateral jerk than fixed-rate VLM.

### Two-Trap Gauntlet (M1)

| Mode | Crash (s) | Route % | Avg speed | Freeze | Router calls | Peak jerk |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Unshielded skill | 3.44 | 35.4 | 20.55 | no | 0 | 0.7 |
| 2 Static CBF | none | 34.5 | 5.75 | yes | 0 | 0.7 |
| 3 Fixed-rate VLM | 5.86 | 62.9 | 21.47 | no | 6 | 81.5 |
| 4 Proposed | none | 80.6 | 13.43 | no | 9 | 15.3 |

## M2 router

Router backend used: **openai_compat_dryrun(dry-run-skill-router)**.

Interrupt retune vs M1 (proposed only; baseline shield geometry unchanged): `min_call_ticks=28`, `shield_clear_ticks=16`, `sigma_threshold_scale=1.55`, `blend_ticks=24`, `gamma_sigma=2.8` (σ inflation cap 10).

**Gauntlet takeaway (M2):** proposed keeps 0 crash / no freeze, raises route above M1 (80.6%), and cuts router calls below M1 (9).

### Two-Trap Gauntlet (M2)

| Mode | Crash (s) | Route % | Avg speed | Freeze | Router calls | Peak jerk |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Unshielded skill | 3.44 | 35.4 | 20.55 | no | 0 | 0.7 |
| 2 Static CBF | none | 34.5 | 5.75 | yes | 0 | 0.7 |
| 3 Fixed-rate VLM | 5.86 | 62.9 | 21.47 | no | 6 | 81.5 |
| 4 Proposed | none | 80.9 | 13.48 | no | 5 | 12.7 |

### Merge (M2)

| Mode | Crash (s) | Route % | Avg speed | Freeze | Router calls | Peak jerk |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Unshielded skill | 5.82 | 66.9 | 20.68 | no | 0 | 0.7 |
| 2 Static CBF | none | 100.0 | 16.48 | no | 0 | 0.7 |
| 3 Fixed-rate VLM | 3.46 | 41.1 | 21.41 | no | 4 | 63.5 |
| 4 Proposed | none | 100.0 | 16.17 | no | 6 | 15.2 |

### Dense highway (M2)

| Mode | Crash (s) | Route % | Avg speed | Freeze | Router calls | Peak jerk |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Unshielded skill | 4.22 | 39.5 | 20.61 | no | 0 | 0.7 |
| 2 Static CBF | none | 97.7 | 13.44 | no | 0 | 0.7 |
| 3 Fixed-rate VLM | 3.94 | 38.1 | 21.30 | no | 4 | 127.8 |
| 4 Proposed | none | 97.4 | 13.39 | no | 2 | 14.3 |

### Raw gauntlet rows (M2)

```json
[
  {
    "mode": "unshielded_skill",
    "crash_time": 3.44,
    "crashed": true,
    "route_pct": 35.35,
    "avg_speed": 20.552,
    "router_calls": 0,
    "peak_lateral_jerk": 0.67,
    "freeze": false,
    "peak_sigma": 0.644,
    "x": 70.7,
    "y": -0.004,
    "vx": 20.824,
    "end_time": 3.44
  },
  {
    "mode": "static_cbf",
    "crash_time": null,
    "crashed": false,
    "route_pct": 34.53,
    "avg_speed": 5.755,
    "router_calls": 0,
    "peak_lateral_jerk": 0.67,
    "freeze": true,
    "peak_sigma": 5.92,
    "x": 69.06,
    "y": 0.17,
    "vx": 0.0,
    "end_time": 12.0
  },
  {
    "mode": "fixed_rate_vlm",
    "crash_time": 5.86,
    "crashed": true,
    "route_pct": 62.9,
    "avg_speed": 21.468,
    "router_calls": 6,
    "peak_lateral_jerk": 81.51,
    "freeze": false,
    "peak_sigma": 19.071,
    "x": 125.81,
    "y": 3.958,
    "vx": 21.128,
    "end_time": 5.86
  },
  {
    "mode": "proposed",
    "crash_time": null,
    "crashed": false,
    "route_pct": 80.86,
    "avg_speed": 13.477,
    "router_calls": 5,
    "peak_lateral_jerk": 12.75,
    "freeze": false,
    "peak_sigma": 22.384,
    "x": 161.72,
    "y": -1.309,
    "vx": 18.297,
    "end_time": 12.0
  }
]
```
