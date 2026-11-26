> **Superseded (2026-11-26).** These numbers predate the crossing-traffic guard and were measured on map
> splits that have since been used for development (500-549, 600-649, 1000-1049). Use
> [PAPER-RESULTS.md](PAPER-RESULTS.md) for the frozen final evaluation on untouched maps.

# Unified 3-Layer System: Benchmark & Single-Switch Ablations (2026-11-25)

Every row below runs the **exact same canonical 3-layer class**
(`UnifiedDrivingSystem` in [driving/safemetadrive/unified.py](../driving/safemetadrive/unified.py)):
- **Layer 1 (Fast Safety Filter):** Strictly 259-dim LiDAR + car sensors (zero simulator peeking),
  stopping-distance speed cap, road-boundary guard, fresh lane-admission gate, and option-preserving reserve.
- **Layer 2 (Fast Driver):** Trained PPO tracker (`skills-seed{0,1,2}/ppo.zip`) with smooth steering blending
  (or analytic tracker as an explicit ablation).
- **Layer 3 (Slow Event-Triggered Router):** Wakes up only when blocked/needed, with pluggable delay and
  semantic STOP/CAREFUL lease override hook.

---

## Table 1: Fresh Untouched Maps (`seeds 600–649`, 50 maps, exact same 259-dim sensors)

| Method | Same 259-dim sensors? | Finished | Zero-cost finished | Mean safety cost | Slow-AI calls | Speed |
|---|---|---:|---:|---:|---:|---:|
| **Unified System (PPO, 3-seed mean)** | **Yes (no peeking)** | **89.3%** | **82.0%** | **0.747** | **1.78** | **0.131 ms** |
| **Unified System (Analytic Layer 2)** | **Yes (no peeking)** | **98.0%** | **90.0%** | 1.400 | 1.70 | **0.065 ms** |
| IDM *(MetaDrive built-in)* | No (peeks at objects) | 84.0% | 72.0% | 1.280 | — | 0.045 ms |
| PPO-Lag *(3-seed mean)* | Yes | 59.3% | 47.3% | 2.200 | — | 0.070 ms |
| MetaDrive Official PPO-Expert | Yes | 56.0% | 46.0% | 190.840 | — | 2.183 ms |
| CPO *(3-seed mean)* | Yes | 0.0% | 0.0% | 9.193 | — | — |

Per-seed PPO on `600–649`: seed 0 = 88% / 0.680 cost; seed 1 = 88% / 0.700 cost; seed 2 = 92% / 0.860 cost.

---

## Table 2: One-Switch-at-a-Time Ablations (`seeds 500–549`, 50 held-out maps)

All rows use the exact same `UnifiedDrivingSystem` (PPO seed 0), turning off **one component at a time**:

| Variant | Finished | Zero-cost finished | Mean safety cost | Router calls/drive | Peak steering jump | What broke? |
|---|---:|---:|---:|---:|---:|---|
| **Full Unified System** | **94.0%** | **84.0%** | **1.300** | **2.1** | **0.536** | *(Reference)* |
| **No Safety Filter (Layer 1 OFF)** | 84.0% | **56.0%** | **17.000** | 2.0 | 0.465 | **13× higher safety cost** |
| **No Replanning (Layer 3 OFF)** | **40.0%** | **34.0%** | 1.620 | 0.0 | 0.413 | **Freezes (-54 pts)** |
| **Always-Poll 10 Hz (No Event Trigger)** | 94.0% | 84.0% | 1.300 | **512.2** | 0.536 | **244× more calls** |
| **Hard Switch (No Blending)** | 94.0% | 84.0% | 1.420 | 2.2 | **0.735** | **+37% sharper steering snap** |
| **4.0 s Delay — WITH Option Reserve** | **92.0%** | **82.0%** | **1.640** | 1.8 | 0.527 | Survives 4 s delay cleanly |
| **4.0 s Delay — WITHOUT Option Reserve** | **72.0%** | **62.0%** | 1.680 | 1.5 | 0.481 | **Trapped (-20 pts!)** |
| *Analytic Layer 2 (No PPO)* | *96.0%* | *84.0%* | *2.460* | *2.2* | *0.330* | *Higher cost than PPO (2.46 vs 1.30)* |

---

## Reproduce (CPU, $0)

```bash
.venv/bin/python -m pytest -q tests/test_lidar_driver.py tests/test_unified_system.py
.venv/bin/python tools/run_unified_suite.py
```

Raw per-episode JSON: `runs/benchmarks/unified-system-suite-2026-11-25.json`.
