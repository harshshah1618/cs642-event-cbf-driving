# Evaluation report: `exploratory-single-seed`

- Run id: `exploratory-single-seed`
- Created: `2026-11-11T10:00:50.648Z`
- Git commit: `01ba8e426940ef0a07020c24c07c296c48386ec4` (branch `codex/latency-aware-research`, dirty: yes)
- Python: `3.12.13` (Linux-7.3.3-1-cachyos-x86_64-with-glibc2.44)
- Router: `local_transformers(Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306, device=cpu)`
- Live capable: yes
- Config SHA-256: `20a3b35660221fcd2c14c1f6f72abca10e98ae09e3e5322edb2a5d21e90ea80b`
- Schema version: `1`
- Router telemetry: live calls `9` / calls `9`, fallbacks `0`, tokens `4169`
- Live model(s): `Qwen/Qwen2.5-1.5B-Instruct`
- Eval seeds: `[3001]` (repeats: 1)
- Modes: `proposed`, `fixed_rate_shielded`

## Warnings

- RETURN_RIGHT.pt is missing in /home/harsh/Projects/cs642-event-cbf-research/event-cbf-driving/checkpoints/metadrive-calibrated; using CRUISE_RIGHT weights as an explicit RETURN_RIGHT alias.
- regression gate: two_trap/proposed: froze in 100% of episodes (require_no_freeze=True)

## Per-scenario results

### `dense_highway`

| Mode | n | Route % (mean ± ci95) | Crash % | Freeze % | Calls | Peak jerk | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fixed_rate_shielded` | 1 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.0 ± 0.0 | 0.7 ± 0.0 | safety_events=139.0, min_h=-2.47, pending_ticks=870.0, live_calls=2.0, tokens=1232, rejected=1.0 |
| `proposed` | 1 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 | safety_events=139.0, min_h=-2.47 |

### `merge`

| Mode | n | Route % (mean ± ci95) | Crash % | Freeze % | Calls | Peak jerk | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fixed_rate_shielded` | 1 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.0 ± 0.0 | 9.0 ± 0.0 | safety_events=18.0, min_h=-1.42, pending_ticks=709.0, live_calls=2.0, tokens=958 |
| `proposed` | 1 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.0 ± 0.0 | 9.0 ± 0.0 | safety_events=18.0, min_h=-1.42, pending_ticks=478.0, live_calls=1.0, tokens=479 |

### `two_trap`

| Mode | n | Route % (mean ± ci95) | Crash % | Freeze % | Calls | Peak jerk | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fixed_rate_shielded` | 1 | 85.2 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 2.0 ± 0.0 | 13.4 ± 0.0 | freeze 100%, safety_events=224.0, infeasible=222.0, min_h=-8.50, pending_ticks=724.0, live_calls=2.0, tokens=731 |
| `proposed` | 1 | 83.1 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 2.0 ± 0.0 | 13.4 ± 0.0 | freeze 100%, safety_events=248.0, infeasible=246.0, min_h=-8.50, pending_ticks=704.0, live_calls=2.0, tokens=769 |

## Per-mode summary (all scenarios)

| Mode | n | Route % | Crash % | Freeze % | Calls | Peak jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_rate_shielded` | 3 | 95.1 ± 9.7 | 0.0 ± 0.0 | 33.3 ± 65.3 | 2.0 ± 0.0 | 7.7 ± 7.3 |
| `proposed` | 3 | 94.4 ± 11.0 | 0.0 ± 0.0 | 33.3 ± 65.3 | 1.0 ± 1.1 | 7.7 ± 7.3 |

## Physical tracking diagnostics

The main table's jerk measures requested acceleration. Achieved motion is measured separately below.

| Scenario | Mode | Achieved peak lateral jerk (m/s³) | Acceleration RMSE (m/s²) | Saturation fraction |
| --- | --- | ---: | ---: | ---: |
| `dense_highway` | `fixed_rate_shielded` | 0.2 ± 0.0 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| `dense_highway` | `proposed` | 0.2 ± 0.0 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| `merge` | `fixed_rate_shielded` | 31.0 ± 0.0 | 0.043 ± 0.000 | 0.087 ± 0.000 |
| `merge` | `proposed` | 31.0 ± 0.0 | 0.043 ± 0.000 | 0.088 ± 0.000 |
| `two_trap` | `fixed_rate_shielded` | 190.4 ± 0.0 | 2.050 ± 0.000 | 0.276 ± 0.000 |
| `two_trap` | `proposed` | 190.3 ± 0.0 | 2.209 ± 0.000 | 0.276 ± 0.000 |

## Per-episode appendix

| Scenario | Mode | Seed | Repeat | Route % | Crash | Freeze | Calls | End reason |
| --- | --- | ---: | ---: | ---: | :---: | :---: | ---: | --- |
| `dense_highway` | `fixed_rate_shielded` | 3001 | 0 | 100.0 | no | no | 2 | route_complete |
| `dense_highway` | `proposed` | 3001 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `fixed_rate_shielded` | 3001 | 0 | 100.0 | no | no | 2 | route_complete |
| `merge` | `proposed` | 3001 | 0 | 100.0 | no | no | 1 | route_complete |
| `two_trap` | `fixed_rate_shielded` | 3001 | 0 | 85.2 | no | yes | 2 | timeout |
| `two_trap` | `proposed` | 3001 | 0 | 83.1 | no | yes | 2 | timeout |

## Checkpoints, aliases, and fingerprints

Checkpoint aliases:

- `RETURN_RIGHT` -> `CRUISE_RIGHT`

| Checkpoint file | SHA-256 | Bytes |
| --- | --- | ---: |
| `CRUISE_RIGHT.pt` | `94eb1250bf10384b` | 63393 |
| `OVERTAKE_LEFT.pt` | `452afd2e196e6d16` | 63393 |

## Honest notes

- This report summarizes 6 measured episode(s). Confidence intervals use `1.96 * sample_std(ddof=1) / sqrt(n)`; they are `0` for `n < 2` and are not meaningful for tiny samples.
- The default offline router is the deterministic **scripted** lane-threat rule. These results characterize the controller + shield loop; they are not evidence about general language-model behavior unless a live backend was explicitly configured. Backend used here: `local_llm`.
