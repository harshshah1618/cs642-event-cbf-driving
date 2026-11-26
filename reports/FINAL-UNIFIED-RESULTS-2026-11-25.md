# Final Unified 3-Layer System: Complete Verified Results (2026-11-25)

Every table below evaluates **one single canonical 3-layer system**
(`UnifiedDrivingSystem` in [driving/safemetadrive/unified.py](../driving/safemetadrive/unified.py)):
- **Layer 1 (Fast Safety Filter):** Strictly 259-dim LiDAR + car sensors (zero simulator peeking),
  stopping-distance speed cap, road-boundary guard, fresh lane-admission check, option-preserving
  turning-room reserve, and semantic `STOP`/`CAREFUL` enforcement.
- **Layer 2 (Fast Driver):** Trained PPO tracker (`skills-seed{0,1,2}/ppo.zip`) with smooth steering blending.
- **Layer 3 (Slow Brain / Event-Triggered Router):** Wakes up only when blocked or when PatchCore
  visual novelty fires (`Qwen3-VL-8B-Instruct` on irregular visual scenes).

---

## Table 1: SafeMetaDrive Equal-Sensor Benchmark (`500–549` & Fresh `600–649`)

All methods ran on the exact same environment, same maps, and same CPU runner.

### 1A. Brand-New Untouched Maps (`600–649`, 50 maps)

| Method | Same 259-dim sensors? | Finished | Zero-cost finished | Mean safety cost | Speed |
|---|---|---:|---:|---:|---:|
| **Unified System (PPO, 3-seed avg)** | **Yes (no peeking)** | **89.3%** | **82.0%** | **0.747** | **0.131 ms** |
| **Unified System (Analytic Layer 2)** | **Yes (no peeking)** | **98.0%** | **90.0%** | 1.400 | **0.065 ms** |
| IDM *(MetaDrive built-in)* | No (peeks at objects) | 84.0% | 72.0% | 1.280 | 0.045 ms |
| PPO-Lag *(3-seed avg)* | Yes | 59.3% | 47.3% | 2.200 | 0.070 ms |
| MetaDrive Official PPO-Expert | Yes | 56.0% | 46.0% | 190.840 | 2.183 ms |
| CPO *(3-seed avg)* | Yes | 0.0% | 0.0% | 9.193 | — |

### 1B. Standard Held-Out Split (`500–549`, 50 maps — Same-Runner Parity)

| Method | Same 259-dim sensors? | Finished | Zero-cost finished | Mean safety cost |
|---|---|---:|---:|---:|
| **Unified System (PPO, 3-seed avg)** | **Yes (no peeking)** | **90.0%** | **82.0%** | **1.600** |
| **Unified System (Analytic Layer 2)** | **Yes (no peeking)** | **96.0%** | **84.0%** | 2.460 |
| IDM *(MetaDrive built-in)* | No (peeks at objects) | 88.0% | 76.0% | 1.180 |
| PPO-Lag *(3-seed avg)* | Yes | 54.0% | 43.3% | 2.193 |
| MetaDrive Official PPO-Expert | Yes | 44.0% | 32.0% | 154.380 |
| CPO *(3-seed avg)* | Yes | 0.7% | 0.7% | 11.687 |

---

## Table 2: One-Switch-at-a-Time Ablations (`500–549`, ONE `UnifiedDrivingSystem`, PPO seed 0)

| Variant (1 switch toggled) | Finished | Zero-cost | Safety cost | Router calls | Peak steer snap | What broke? |
|---|---:|---:|---:|---:|---:|---|
| **Full Unified System (PPO)** | **94.0%** | **84.0%** | **1.300** | **2.1** | **0.536** | *(Reference)* |
| **Layer 1 OFF (No Safety Filter)** | 84.0% | **56.0%** | **17.000** | 2.0 | 0.465 | **13× worse safety cost** |
| **Layer 3 OFF (No Replanning)** | **40.0%** | **34.0%** | 1.620 | 0.0 | 0.413 | **Freezes (-54 pts)** |
| **Event Trigger OFF (10 Hz Poll)** | 94.0% | 84.0% | 1.300 | **512.2** | 0.536 | **244× more calls** |
| **Blending OFF (Hard Switch)** | 94.0% | 84.0% | 1.420 | 2.2 | **0.735** | **+37% sharper snap** |
| **4.0 s Delay — WITH Reserve (6.5 m)** | **92.0%** | **82.0%** | **1.640** | 1.8 | 0.527 | Survives 4 s delay |
| **4.0 s Delay — NO Reserve (0 m)** | **72.0%** | **62.0%** | 1.680 | 1.5 | 0.481 | **Trapped (-20 pts)** |

---

## Table 3: Controlled Delay Tipping-Point Sweep (`0 s → 18 s`, 576 physical runs)

See [delay-tipping-point.png](delay-tipping-point.png) (exact 8 milestone seeds × 3 scenes = 24 runs/point, 0 crashes):
- **0.0 s – 3.0 s:** Both With Room (8 m) and Without Room (0 m) finish **24/24 (100%)**.
- **4.0 s *(Tipping point)*:** With Room = **24/24 (100%)**, Without Room = **22/24 (91.7%, 2 stalled)**.
- **5.0 s – 18.0 s *(Cliff)*:** With Room = **24/24 (100%)**, Without Room = **8/24 (33.3%, 16/16 blocked runs trapped)**.

---

## Table 4: Unified Irregular Visual Scenes (`1000–1049`, 1,200 episodes — ALL on `UnifiedDrivingSystem` + `irregular-episode-v2`)

All three arms use the **exact same `UnifiedDrivingSystem`** (zero-peeking 259-dim LiDAR Layer 1 + PPO seed 0 Layer 2)
and the **exact same per-step collision logger** (`irregular-episode-v2`, Modal run `ap-L1zovXOxSvHEinhasQgFea` for Qwen3-VL-8B):

| Scene (n=50 each) | Unified Normal crashes | Unified Careful crashes | **Unified + Qwen3-VL-8B crashes** | Normal STOP viol. | **Qwen STOP viol.** | Qwen calls/ep | Route % (N / C / Q) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Flagger (STOP sign)** | 1/50 | 0/50 | 1/50 | **50/50** | **25/50** | 5.0 | 25 / 25 / 25% |
| **Wet road closure** | **2/50** | **0/50** | **0/50** | 0/50 | 0/50 | 21.5 | 17 / 16 / 16% |
| Pedestrian in lane | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 1.6 | 25 / 25 / 25% |
| Lane closed sign | 1/50 | 0/50 | 1/50 | 0/50 | 0/50 | 1.6 | 25 / 25 / 25% |
| Billboard *(decoy)* | 1/50 | 0/50 | 1/50 | 0/50 | 0/50 | 1.5 | 25 / 25 / 25% |
| Pedestrian shoulder *(decoy)* | 1/50 | 0/50 | 1/50 | 0/50 | 0/50 | 0.9 | 25 / 25 / 25% |
| Rain clean *(weather)* | 1/50 | 1/50 | 1/50 | 0/50 | 0/50 | 9.6 | 54 / 47 / 47% |
| **Clean road** | 1/50 | 1/50 | 1/50 | 0/50 | 0/50 | **0.1** | **54 / 47 / 54%** |
| **Total (400 eps / arm)** | **8/400** | **2/400** | **6/400** | **50/50** | **25/50** | **5.24** | — |

---

## Table 5: Cross-Simulator Generalization (Waymo/nuScenes & `highway-env`)

- **Real-World Replays (Waymo + nuScenes, 13 scenes):** LiDAR safety filter raises crash-free driving from **61.5% → 76.9%** (**60% → 80%** on nuScenes) at **0.045 ms/step**.
- **`highway-env` (750 episodes across `highway-v0`, `merge-v1`, `roundabout-v1`, 1.5 s delay):** Event-triggered + fast safety filter reaches **80.0% crash-free** (**100% on `merge-v1`**, **82% on `highway-v0`**) vs **34.0%** (always-ask) and **68.0%** (static filter), while cutting slow-AI calls by **98.4%** (**128.0 → 2.1 calls/ep**) at **0.007 ms/step**.

---

## Raw Evidence Paths
- `runs/benchmarks/unified-system-suite-2026-11-25.json`
- `runs/benchmarks/parity-500-549-2026-11-25.json`
- `runs/benchmarks/controlled-delay-sweep-2026-11-25.json`
- `runs/benchmarks/unified-irregular-controls-2026-11-25.json`
- `runs/benchmarks/final-unified-qwen3vl-8b/` (all 16 Modal shards, 400 `irregular-episode-v2` episodes)
- `runs/benchmarks/realworld-waymo-nuscenes-2026-11-25.json`
- `runs/benchmarks/highway-env-2026-11-25.json`
