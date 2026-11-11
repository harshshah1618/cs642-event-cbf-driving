# Complete held-out condition tables

## isolation: dense_highway, fast

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Deadline-aware | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 20.48 | 17.21 |
| Matched no-latency | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.00 |
| Condition-tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.00 |
| Gated polling (10Hz) | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| TTC threshold | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |

## isolation: dense_highway, jitter

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Deadline-aware | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 11.25 | 17.41 |
| Matched no-latency | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.09 |
| Condition-tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.09 |
| Gated polling (10Hz) | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| TTC threshold | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |

## isolation: dense_highway, long

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.87 |
| Deadline-aware | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.87 |
| Original event trigger | 40 | 95.0 | 0 | 99.9 | 5.50 | 17.41 |
| Matched no-latency | 40 | 95.0 | 0 | 99.9 | 0.28 | 16.86 |
| Tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.25 |
| Condition-tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.25 |
| Gated polling (10Hz) | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.87 |
| TTC threshold | 40 | 95.0 | 0 | 99.9 | 0.28 | 16.86 |

## isolation: dense_highway, nominal

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Deadline-aware | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 16.77 | 17.33 |
| Matched no-latency | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.03 |
| Condition-tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.03 |
| Gated polling (10Hz) | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| TTC threshold | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |

## isolation: dense_highway, slow

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Deadline-aware | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Original event trigger | 40 | 97.5 | 0 | 99.9 | 6.62 | 17.74 |
| Matched no-latency | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| Tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.11 |
| Condition-tuned polling | 40 | 97.5 | 0 | 100.0 | 2.00 | 17.11 |
| Gated polling (10Hz) | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |
| TTC threshold | 40 | 95.0 | 0 | 99.9 | 0.25 | 16.86 |

## isolation: merge, fast

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.39 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 0.78 | 12.92 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 3.70 | 11.64 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.78 | 12.93 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.39 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.39 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.39 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.78 | 12.93 |

## isolation: merge, jitter

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.77 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 1.00 | 12.10 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 3.15 | 12.01 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.78 | 13.17 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.77 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.77 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.77 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.78 | 13.17 |

## isolation: merge, long

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 1.00 | 12.25 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 1.00 | 12.50 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 2.38 | 12.53 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.78 | 13.62 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.25 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.25 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 1.00 | 12.25 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.78 | 13.62 |

## isolation: merge, nominal

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.56 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 0.90 | 12.50 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 3.75 | 11.78 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.78 | 13.02 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.56 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.56 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.56 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.78 | 13.02 |

## isolation: merge, slow

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.84 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 0.97 | 12.35 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 2.67 | 12.12 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.78 | 13.25 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.84 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.84 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.84 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.78 | 13.25 |

## isolation: two_trap, fast

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.82 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.07 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 5.05 | 13.36 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.10 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 14.17 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 14.17 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 2.00 | 11.84 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.10 |

## isolation: two_trap, jitter

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.88 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.89 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 5.58 | 14.87 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.28 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 14.90 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 14.89 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.91 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.28 |

## isolation: two_trap, long

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.18 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.21 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 4.00 | 16.75 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.72 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 16.36 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 16.36 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.21 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.72 |

## isolation: two_trap, nominal

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.13 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.29 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 5.15 | 13.90 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.44 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 14.39 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 14.39 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.15 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.44 |

## isolation: two_trap, slow

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.25 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.27 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 5.53 | 15.24 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.67 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.14 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.14 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.28 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.67 |

## roads: generalization, jitter

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 70.0 | 6 | 83.7 | 2.65 | 39.95 |
| Deadline-aware | 40 | 67.5 | 5 | 83.1 | 1.65 | 40.04 |
| No replanning | 40 | 70.0 | 4 | 83.3 | 0.00 | 39.67 |
| Matched no-latency | 40 | 72.5 | 5 | 83.7 | 1.27 | 40.01 |
| Tuned polling | 40 | 65.0 | 4 | 83.8 | 20.50 | 40.05 |
| Condition-tuned polling | 40 | 72.5 | 6 | 85.3 | 39.20 | 41.01 |
| Gated polling (10Hz) | 40 | 70.0 | 6 | 83.7 | 2.67 | 39.95 |
| Intervention trigger | 40 | 75.0 | 4 | 85.0 | 7.75 | 42.17 |
| TTC threshold | 40 | 72.5 | 5 | 83.7 | 1.27 | 40.01 |

## roads: generalization, long

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 67.5 | 5 | 82.1 | 2.75 | 39.08 |
| Deadline-aware | 40 | 67.5 | 5 | 83.0 | 1.80 | 40.11 |
| No replanning | 40 | 70.0 | 4 | 83.3 | 0.00 | 39.67 |
| Matched no-latency | 40 | 70.0 | 5 | 83.2 | 1.07 | 39.60 |
| Tuned polling | 40 | 70.0 | 5 | 84.5 | 20.98 | 40.94 |
| Condition-tuned polling | 40 | 70.0 | 5 | 84.5 | 20.98 | 40.94 |
| Gated polling (10Hz) | 40 | 67.5 | 5 | 82.1 | 2.75 | 39.08 |
| Intervention trigger | 40 | 67.5 | 5 | 83.6 | 5.08 | 42.38 |
| TTC threshold | 40 | 70.0 | 5 | 83.2 | 1.07 | 39.60 |

## roads: generalization, nominal

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 70.0 | 7 | 84.5 | 2.98 | 40.27 |
| Deadline-aware | 40 | 70.0 | 6 | 82.6 | 1.45 | 39.62 |
| No replanning | 40 | 70.0 | 4 | 83.3 | 0.00 | 39.67 |
| Matched no-latency | 40 | 70.0 | 6 | 82.6 | 1.43 | 39.56 |
| Tuned polling | 40 | 72.5 | 4 | 85.2 | 20.93 | 40.95 |
| Condition-tuned polling | 40 | 70.0 | 4 | 83.5 | 133.60 | 39.99 |
| Gated polling (10Hz) | 40 | 70.0 | 7 | 84.5 | 3.77 | 40.20 |
| Intervention trigger | 40 | 72.5 | 5 | 84.2 | 9.05 | 42.02 |
| TTC threshold | 40 | 70.0 | 6 | 82.6 | 1.43 | 39.56 |

## roads: generalization, slow

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 67.5 | 8 | 84.3 | 2.58 | 40.25 |
| Deadline-aware | 40 | 72.5 | 4 | 83.1 | 1.60 | 39.85 |
| No replanning | 40 | 70.0 | 4 | 83.3 | 0.00 | 39.67 |
| Matched no-latency | 40 | 72.5 | 4 | 83.1 | 1.15 | 39.61 |
| Tuned polling | 40 | 70.0 | 3 | 85.4 | 20.95 | 41.02 |
| Condition-tuned polling | 40 | 70.0 | 3 | 85.4 | 20.95 | 41.02 |
| Gated polling (10Hz) | 40 | 67.5 | 8 | 84.3 | 2.58 | 40.25 |
| Intervention trigger | 40 | 72.5 | 4 | 84.7 | 9.62 | 41.98 |
| TTC threshold | 40 | 72.5 | 4 | 83.1 | 1.15 | 39.61 |

## roads: generalization, standard

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 70.0 | 5 | 82.3 | 2.70 | 39.03 |
| Deadline-aware | 40 | 70.0 | 5 | 82.6 | 1.15 | 39.25 |
| Released PPO expert | 40 | 75.0 | 5 | 88.1 | 0.00 | 42.58 |
| Upstream IDM (40km/h) | 40 | 40.0 | 13 | 72.9 | 0.00 | 29.05 |
| No replanning | 40 | 70.0 | 4 | 83.3 | 0.00 | 39.67 |
| Matched no-latency | 40 | 70.0 | 5 | 82.6 | 1.15 | 39.25 |
| Tuned polling | 40 | 72.5 | 3 | 84.9 | 20.65 | 40.37 |
| Condition-tuned polling | 40 | 70.0 | 4 | 83.4 | 39.65 | 39.20 |
| Gated polling (10Hz) | 40 | 72.5 | 4 | 84.4 | 5.03 | 39.97 |
| Intervention trigger | 40 | 72.5 | 4 | 83.0 | 8.57 | 40.66 |
| TTC threshold | 40 | 70.0 | 5 | 82.6 | 1.15 | 39.25 |

## roads: safe, jitter

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 67.5 | 2 | 85.5 | 2.58 | 44.97 |
| Deadline-aware | 40 | 75.0 | 2 | 87.3 | 1.55 | 42.49 |
| No replanning | 40 | 37.5 | 2 | 68.5 | 0.00 | 59.95 |
| Matched no-latency | 40 | 75.0 | 2 | 87.2 | 1.18 | 42.46 |
| Tuned polling | 40 | 67.5 | 3 | 84.9 | 22.95 | 44.95 |
| Condition-tuned polling | 40 | 67.5 | 3 | 84.9 | 22.95 | 44.95 |
| Gated polling (10Hz) | 40 | 67.5 | 2 | 85.5 | 2.58 | 44.97 |
| Intervention trigger | 40 | 75.0 | 2 | 86.6 | 13.35 | 47.02 |
| TTC threshold | 40 | 75.0 | 2 | 87.2 | 1.18 | 42.46 |

## roads: safe, long

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 72.5 | 2 | 88.6 | 2.65 | 44.80 |
| Deadline-aware | 40 | 67.5 | 2 | 86.6 | 1.45 | 43.87 |
| No replanning | 40 | 37.5 | 2 | 68.5 | 0.00 | 59.95 |
| Matched no-latency | 40 | 72.5 | 2 | 87.7 | 1.20 | 45.20 |
| Tuned polling | 40 | 75.0 | 2 | 87.9 | 22.70 | 44.51 |
| Condition-tuned polling | 40 | 70.0 | 2 | 85.3 | 12.25 | 47.14 |
| Gated polling (10Hz) | 40 | 72.5 | 2 | 88.6 | 2.65 | 44.80 |
| Intervention trigger | 40 | 70.0 | 2 | 83.9 | 7.80 | 44.41 |
| TTC threshold | 40 | 72.5 | 2 | 87.7 | 1.20 | 45.20 |

## roads: safe, nominal

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 67.5 | 2 | 86.3 | 2.95 | 43.16 |
| Deadline-aware | 40 | 72.5 | 2 | 87.1 | 1.30 | 42.25 |
| No replanning | 40 | 37.5 | 2 | 68.5 | 0.00 | 59.95 |
| Matched no-latency | 40 | 72.5 | 2 | 87.1 | 1.12 | 42.06 |
| Tuned polling | 40 | 75.0 | 2 | 87.2 | 22.77 | 44.57 |
| Condition-tuned polling | 40 | 67.5 | 2 | 84.4 | 149.05 | 44.63 |
| Gated polling (10Hz) | 40 | 67.5 | 2 | 85.5 | 3.30 | 42.60 |
| Intervention trigger | 40 | 72.5 | 2 | 85.6 | 13.00 | 45.03 |
| TTC threshold | 40 | 72.5 | 2 | 87.1 | 1.12 | 42.06 |

## roads: safe, slow

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 70.0 | 2 | 86.8 | 2.50 | 45.02 |
| Deadline-aware | 40 | 72.5 | 2 | 86.8 | 1.38 | 43.67 |
| No replanning | 40 | 37.5 | 2 | 68.5 | 0.00 | 59.95 |
| Matched no-latency | 40 | 75.0 | 2 | 87.8 | 1.18 | 44.53 |
| Tuned polling | 40 | 70.0 | 2 | 84.9 | 23.00 | 45.02 |
| Condition-tuned polling | 40 | 70.0 | 2 | 86.0 | 46.40 | 45.87 |
| Gated polling (10Hz) | 40 | 70.0 | 2 | 86.8 | 2.50 | 45.02 |
| Intervention trigger | 40 | 72.5 | 2 | 85.7 | 14.85 | 46.74 |
| TTC threshold | 40 | 75.0 | 2 | 87.8 | 1.18 | 44.53 |

## roads: safe, standard

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 70.0 | 2 | 87.5 | 2.90 | 42.34 |
| Deadline-aware | 40 | 70.0 | 2 | 86.9 | 1.12 | 41.92 |
| Released PPO expert | 40 | 47.5 | 15 | 77.3 | 0.00 | 62.33 |
| Upstream IDM (40km/h) | 40 | 50.0 | 6 | 81.2 | 0.00 | 32.16 |
| No replanning | 40 | 37.5 | 2 | 68.5 | 0.00 | 59.95 |
| Matched no-latency | 40 | 70.0 | 2 | 86.9 | 1.12 | 41.92 |
| Tuned polling | 40 | 72.5 | 2 | 86.2 | 22.38 | 43.92 |
| Condition-tuned polling | 40 | 72.5 | 2 | 86.2 | 46.10 | 45.70 |
| Gated polling (10Hz) | 40 | 70.0 | 2 | 88.3 | 5.62 | 42.75 |
| Intervention trigger | 40 | 72.5 | 3 | 86.1 | 14.25 | 46.66 |
| TTC threshold | 40 | 70.0 | 2 | 86.9 | 1.12 | 41.92 |

## phase: dense_highway, jitter

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 0.15 | 17.40 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 0.12 | 17.40 |
| Original event trigger | 40 | 90.0 | 0 | 99.8 | 12.30 | 17.90 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.12 | 17.40 |
| Tuned polling | 40 | 92.5 | 0 | 99.8 | 15.22 | 17.81 |
| Condition-tuned polling | 40 | 97.5 | 0 | 99.9 | 1.73 | 17.33 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 0.15 | 17.40 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.12 | 17.40 |

## phase: dense_highway, long

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 0.15 | 17.40 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 0.15 | 17.38 |
| Original event trigger | 40 | 92.5 | 0 | 99.7 | 6.22 | 17.57 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.17 | 17.36 |
| Tuned polling | 40 | 87.5 | 0 | 99.7 | 8.18 | 18.07 |
| Condition-tuned polling | 40 | 97.5 | 0 | 100.0 | 1.73 | 17.30 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 0.17 | 17.38 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.15 | 17.36 |

## phase: dense_highway, nominal

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 0.15 | 17.41 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 0.12 | 17.40 |
| Original event trigger | 40 | 90.0 | 0 | 99.7 | 18.38 | 17.83 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.12 | 17.40 |
| Tuned polling | 40 | 92.5 | 0 | 99.8 | 17.68 | 17.73 |
| Condition-tuned polling | 40 | 97.5 | 0 | 99.9 | 1.73 | 17.36 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 0.15 | 17.41 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.12 | 17.40 |

## phase: merge, jitter

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.78 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 0.95 | 12.34 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 3.23 | 12.07 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.72 | 13.60 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 11.15 | 12.05 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 1.27 | 13.89 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.80 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.72 | 13.61 |

## phase: merge, long

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 1.00 | 12.15 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 0.97 | 12.63 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 2.58 | 12.56 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.72 | 13.97 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 6.15 | 12.53 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 1.32 | 14.38 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 1.00 | 12.17 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.72 | 13.97 |

## phase: merge, nominal

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.64 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 0.90 | 12.59 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 3.90 | 11.90 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 0.72 | 13.45 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 11.85 | 11.92 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 1.23 | 13.69 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 1.00 | 11.66 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 0.72 | 13.45 |

## phase: two_trap, jitter

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.96 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.96 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 5.72 | 14.84 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.29 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 12.75 | 13.83 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 4.00 | 16.41 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.06 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 2.00 | 13.29 |

## phase: two_trap, long

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.27 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.30 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 4.00 | 16.73 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.70 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 7.90 | 16.23 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 7.92 | 15.76 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.37 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 2.00 | 15.70 |

## phase: two_trap, nominal

| Method | n | Success % | Crashes | Route % | Calls | End time s |
|---|---:|---:|---:|---:|---:|---:|
| Immediate usefulness | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.17 |
| Deadline-aware | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.29 |
| Original event trigger | 40 | 100.0 | 0 | 100.0 | 5.35 | 13.88 |
| Matched no-latency | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.43 |
| Tuned polling | 40 | 100.0 | 0 | 100.0 | 12.72 | 12.88 |
| Condition-tuned polling | 40 | 100.0 | 0 | 100.0 | 4.00 | 15.85 |
| Gated polling (10Hz) | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.25 |
| TTC threshold | 40 | 100.0 | 0 | 100.0 | 2.00 | 12.43 |
