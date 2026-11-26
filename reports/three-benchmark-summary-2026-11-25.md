# Three-Benchmark Fair Evaluation Summary (2026-11-25)

All three benchmarks below were run locally on CPU (**$0 cloud compute**) using **strictly observation-only**
inputs (zero simulator object-list peeking).

---

## Benchmark 1: SafeMetaDrive (Curved Roads, Intersections, Roundabouts, Accidents)

Exact same 259-float LiDAR + car sensor input as PPO-Lag, CPO, and MetaDrive's official PPO-Expert.
Tuned only on dev maps (`100–149`), then frozen and tested once on **50 brand-new untouched maps (`600–649`)**
and **50 standard held-out maps (`500–549`)**.

### Fresh Untouched Maps (`600–649`, 50 maps)

| Method | Same sensors? | Finished | Zero-cost finished | Mean safety cost | Speed (ms/step) |
|---|---|---:|---:|---:|---:|
| **Ours (LiDAR + Option Reserve)** | **Yes** | **98.0%** | **90.0%** | **0.880** | **0.058 ms** |
| IDM *(MetaDrive built-in)* | No (peeks) | 84.0% | 72.0% | 1.280 | 0.045 ms |
| PPO-Lag *(3-seed average)* | Yes | 59.3% | 47.3% | 2.200 | 0.070 ms |
| Official PPO-Expert | Yes | 56.0% | 46.0% | 190.840 | 2.183 ms |
| CPO *(3-seed average)* | Yes | 0.0% | 0.0% | 9.193 | — |

On held-out `500–549`: **Ours = 96.0%** vs **42.0%** static-lane ablation; under 1.5 s delay, adding Option Reserve
improved finishes (**92.0% → 94.0%**) and clean zero-cost finishes (**82.0% → 86.0%**).

---

## Benchmark 2: Real-World Traffic Replays — Waymo & nuScenes (`ScenarioEnv`, 13 scenes)

Real human-recorded traffic trajectories (3 Waymo + 10 nuScenes mini scenes). Strictly 161-dim sensor input.

| Method | Crash-free (all 13) | nuScenes crash-free (10) | Waymo crash-free (3) | Route arrival | Speed (ms/step) |
|---|---:|---:|---:|---:|---:|
| **Ours (LiDAR Safety Filter + Tracker)** | **76.9%** (10/13) | **80.0%** (8/10) | **66.7%** (2/3) | 92.3% (12/13) | **0.045 ms** |
| Unshielded Tracker *(no safety filter)* | 61.5% (8/13) | 60.0% (6/10) | 66.7% (2/3) | 100.0% (13/13) | 0.022 ms |
| Naive constant throttle | 30.8% (4/13) | 30.0% (3/10) | 33.3% (1/3) | 53.8% (7/13) | — |

*Takeaway:* Adding the fast LiDAR safety filter cuts real-world crashes by **40%** (5/13 → 3/13, raising nuScenes
crash-free from 60% → 80%) while keeping 12/13 arrivals at 0.045 ms/step.

---

## Benchmark 3: `highway-env` (Completely Separate Simulator, 750 episodes)

50 seeds × 3 standard environments (`highway-v0`, `merge-v1`, `roundabout-v1`) = **150 episodes per method**,
10 Hz control, **1.5 s slow-AI delay**, strictly standard `Kinematics` observation.

| Method | Overall crash-free (150) | `highway-v0` (50) | `merge-v1` (50) | `roundabout-v1` (50) | Slow-AI calls/ep | Speed (ms/step) |
|---|---:|---:|---:|---:|---:|---:|
| **Ours (Event + Safety + Reserve)** | **80.0%** | **82.0%** | **100.0%** | 58.0% | **2.1** | **0.007 ms** |
| Event + Safety *(no reserve)* | 80.0% | 82.0% | 100.0% | 58.0% | 2.2 | 0.007 ms |
| Static Safety Filter *(no replan)* | 68.0% | 72.0% | 74.0% | 58.0% | 0.0 | 0.007 ms |
| Unshielded Fast Driver | 42.0% | 8.0% | 60.0% | 58.0% | 0.0 | 0.006 ms |
| Always-Ask Slow AI *(10 Hz, 1.5 s)* | 34.0% | 0.0% | 44.0% | 58.0% | 128.0 | 0.011 ms |

*Takeaway:* In a completely separate simulator, our event-triggered + fast safety stack lifts crash-free driving from
**34% (always-ask)** and **68% (static filter)** to **80%** (**100% on `merge-v1`**, **82% on `highway-v0`**) while
cutting slow-AI calls by **98.4%** (128.0 → 2.1 calls/episode) at **0.007 ms/step**.

---

## Raw Evidence & Reproduction

- SafeMetaDrive: `tools/safemetadrive_fair_benchmark.py` → `runs/safemetadrive/fair-lidar-benchmark-2026-11-25.json`
- Waymo + nuScenes: `tools/benchmark_realworld_replays.py` → `runs/benchmarks/realworld-waymo-nuscenes-2026-11-25.json`
- `highway-env`: `tools/benchmark_highway_env.py` → `runs/benchmarks/highway-env-2026-11-25.json`
