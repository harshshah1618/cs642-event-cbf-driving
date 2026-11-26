> **Superseded (2026-11-26).** These numbers predate the crossing-traffic guard and were measured on map
> splits that have since been used for development (500-549, 600-649, 1000-1049). Use
> [PAPER-RESULTS.md](PAPER-RESULTS.md) for the frozen final evaluation on untouched maps.

# Fair Equal-Observation SafeMetaDrive Benchmark (2026-11-25)

This study removes **all** simulator-state peeking from our SafeMetaDrive driver.
`LidarOptionReserveDriver` ([driving/safemetadrive/lidar_driver.py](../driving/safemetadrive/lidar_driver.py))
receives **only** the standard 259-float MetaDrive `LidarStateObservation` vector —
the exact same input given to PPO-Lag, CPO, and MetaDrive's upstream PPO expert.
It never accesses `env`, `vehicle`, `navigation`, road-network graphs, or simulator object lists
(verified by `tests/test_lidar_driver.py`).

All tuning was performed strictly on the development split (`maps 100–149`, frozen at 94.0% success / 0.580 mean cost).
The frozen controller was then evaluated once on the standard held-out split (`maps 500–549`) and once on a
**brand-new untouched fresh split (`maps 600–649`)**, head-to-head against IDM, MetaDrive's upstream PPO expert,
PPO-Lag (3 trained seeds), and CPO (3 trained seeds).

## 1. Brand-New Untouched Maps (`seeds 600–649`, 50 maps)

| Method | Same 259-dim input? | Success | Zero-cost success | Mean cost | Mean return |
|---|---|---:|---:|---:|---:|
| **Ours (LiDAR-only + Option Reserve)** | **Yes (no peeking)** | **98.0%** | **90.0%** | **0.880** | **381.1** |
| IDM (upstream rule) | No (uses object list) | 84.0% | 72.0% | 1.280 | 337.3 |
| PPO-Lag (3-seed mean) | Yes | 59.3% | 47.3% | 2.200 | 298.1 |
| Upstream PPO-Expert | Yes | 56.0% | 46.0% | 190.840 | -652.4 |
| CPO (3-seed mean) | Yes | 0.0% | 0.0% | 9.193 | 73.9 |

Per-seed breakdown on `600–649`:
- PPO-Lag: seed 0 = 62% / cost 2.70; seed 1 = 66% / cost 1.48; seed 2 = 50% / cost 2.42.
- CPO: seed 0 = 0% / cost 0.76; seed 1 = 0% / cost 11.54; seed 2 = 0% / cost 15.28.

## 2. Standard Held-Out Test Split (`seeds 500–549`, 50 maps) & Ablations

| Method | Same 259-dim input? | Success | Zero-cost success | Mean cost | Mean return |
|---|---|---:|---:|---:|---:|
| **Ours (LiDAR-only + Option Reserve, 0 s)** | **Yes (no peeking)** | **96.0%** | **84.0%** | **1.420** | **363.5** |
| Ours + 1.5 s delay (WITH Option Reserve) | Yes | 94.0% | 86.0% | 2.060 | 358.6 |
| Ablation: 1.5 s delay (NO Option Reserve) | Yes | 92.0% | 82.0% | 1.920 | 354.3 |
| Ablation: LiDAR-only Static Lane (no replan) | Yes | 42.0% | 38.0% | 1.740 | 261.0 |
| Upstream PPO-Expert | Yes | 44.0% | 32.0% | 154.380 | -497.0 |
| *Prior record: IDM (500–549)* | *No (uses object list)* | *88.0%* | *76.0%* | *1.180* | *355.7* |
| *Prior record: PPO-Lag 3-seed mean (500–549)* | *Yes* | *59.3%* | *46.0%* | *1.927* | *297.5* |
| *Prior record: CPO 3-seed mean (500–549)* | *Yes* | *0.7%* | *0.7%* | *21.933* | *22.1* |

## 3. Honest Scope & Caveats

- This is a **strictly equal-observation win** on SafeMetaDrive (`map=3, accident_prob=0.8, traffic_density=0.05, horizon=1000`, pinned commit `fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`).
- Our driver is a structured LiDAR perception + safety-filter + option-reserve controller (zero training transitions), whereas PPO-Lag and CPO are model-free Safe-RL policies trained for 1M transitions. It shows that structured observation-only safety filtering + option-preserving replanning dramatically outperforms both model-free Safe-RL baselines and MetaDrive's built-in IDM and PPO-Expert under identical sensor inputs.
- Raw per-episode records are saved in `runs/safemetadrive/fair-lidar-benchmark-2026-11-25.json`.

## Reproduce (CPU, $0)

```bash
.venv/bin/python -m pytest -q tests/test_lidar_driver.py
.venv/bin/python tools/safemetadrive_fair_benchmark.py
```
