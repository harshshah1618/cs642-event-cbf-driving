# Evaluation report: `production-v1`

- Run id: `production-v1`
- Created: `2026-11-05T05:19:07.701Z`
- Git commit: `2a190a0bc659ee2dab5de1cd7d776f3eec9b2935` (branch `harsh-testing`, dirty: yes)
- Python: `3.12.13` (Linux-7.3.0-4-cachyos-x86_64-with-glibc2.44)
- Router: `scripted(slow_speed=15, look_ahead=200, latency_s=0.3)`
- Live capable: no
- Config SHA-256: `4c0076182b784c73977adb9fe15af05f7033f0f13fb10772083f578dfda8b38e`
- Schema version: `1`
- Router telemetry: live calls `0` / calls `510`, fallbacks `0`, tokens `0`
- Router model(s): `scripted`
- Eval seeds: `[0, 1, 2, 3, 4]` (repeats: 1)
- Modes: `unshielded_skill`, `static_cbf`, `fixed_rate_vlm`, `always_on_vlm`, `fasionad_wakeup`, `cert_replan`, `proposed`, `proposed_no_margin`, `proposed_no_blend`

## Warnings

- RETURN_RIGHT.pt is missing in /home/harsh/Projects/cs642-event-cbf-driving/event-cbf-driving/checkpoints; using CRUISE_RIGHT weights as an explicit RETURN_RIGHT alias.

## Per-scenario results

### `dense_highway`

| Mode | n | Route % (mean ± ci95) | Crash % | Freeze % | Calls | Peak jerk | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `always_on_vlm` | 5 | 38.2 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 14.0 ± 0.0 | 122.5 ± 0.0 | crash 100%, pending_survival=0%, pending_ticks=198.0 |
| `cert_replan` | 5 | 96.1 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 139.2 ± 0.0 | - |
| `fasionad_wakeup` | 5 | 97.7 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 | - |
| `fixed_rate_vlm` | 5 | 38.1 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 4.0 ± 0.0 | 127.8 ± 0.0 | crash 100%, pending_ticks=60.0 |
| `proposed` | 5 | 97.3 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.0 ± 0.0 | 14.4 ± 0.0 | safety_events=176.0, infeasible=47.0, min_h=-11.25, pending_ticks=30.0 |
| `proposed_no_blend` | 5 | 96.9 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.0 ± 0.0 | 114.5 ± 0.0 | safety_events=253.0, infeasible=43.0, min_h=-10.15, pending_ticks=30.0 |
| `proposed_no_margin` | 5 | 97.3 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.0 ± 0.0 | 14.4 ± 0.0 | safety_events=165.0, infeasible=33.0, min_h=-7.62, pending_ticks=30.0 |
| `static_cbf` | 5 | 97.7 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 | - |
| `unshielded_skill` | 5 | 39.5 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 | crash 100% |

### `merge`

| Mode | n | Route % (mean ± ci95) | Crash % | Freeze % | Calls | Peak jerk | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `always_on_vlm` | 5 | 41.1 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 12.0 ± 0.0 | 63.5 ± 0.0 | crash 100%, pending_survival=0%, pending_ticks=173.0 |
| `cert_replan` | 5 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 54.3 ± 0.0 | - |
| `fasionad_wakeup` | 5 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 | - |
| `fixed_rate_vlm` | 5 | 41.1 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 4.0 ± 0.0 | 63.5 ± 0.0 | crash 100%, pending_ticks=60.0 |
| `proposed` | 5 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.0 ± 0.0 | 13.2 ± 0.0 | safety_events=119.0, infeasible=90.0, min_h=-10.84, pending_ticks=90.0 |
| `proposed_no_blend` | 5 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.0 ± 0.0 | 131.0 ± 0.0 | safety_events=136.0, infeasible=98.0, min_h=-9.25, pending_ticks=90.0 |
| `proposed_no_margin` | 5 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.0 ± 0.0 | 13.0 ± 0.0 | safety_events=86.0, infeasible=60.0, min_h=-7.80, pending_ticks=90.0 |
| `static_cbf` | 5 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 | - |
| `unshielded_skill` | 5 | 66.9 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 | crash 100% |

### `two_trap`

| Mode | n | Route % (mean ± ci95) | Crash % | Freeze % | Calls | Peak jerk | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `always_on_vlm` | 5 | 62.8 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 20.0 ± 0.0 | 67.0 ± 0.0 | crash 100%, pending_survival=0%, pending_ticks=293.0 |
| `cert_replan` | 5 | 34.5 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 147.1 ± 0.0 | freeze 100%, safety_events=495.0, infeasible=113.0, min_h=-4.20 |
| `fasionad_wakeup` | 5 | 62.7 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 1.0 ± 0.0 | 125.0 ± 0.0 | freeze 100%, safety_events=197.0, infeasible=113.0, min_h=-4.20 |
| `fixed_rate_vlm` | 5 | 62.9 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 6.0 ± 0.0 | 81.5 ± 0.0 | crash 100%, pending_ticks=90.0 |
| `proposed` | 5 | 71.5 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 5.0 ± 0.0 | 14.5 ± 0.0 | safety_events=177.0, infeasible=157.0, min_h=-15.08, pending_ticks=75.0 |
| `proposed_no_blend` | 5 | 76.5 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 7.0 ± 0.0 | 131.3 ± 0.0 | safety_events=157.0, infeasible=123.0, min_h=-15.10, pending_ticks=105.0 |
| `proposed_no_margin` | 5 | 71.6 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 5.0 ± 0.0 | 14.5 ± 0.0 | safety_events=172.0, infeasible=144.0, min_h=-12.39, pending_ticks=75.0 |
| `static_cbf` | 5 | 34.5 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 | freeze 100%, safety_events=495.0, infeasible=113.0, min_h=-4.20 |
| `unshielded_skill` | 5 | 35.4 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 | crash 100% |

## Per-mode summary (all scenarios)

| Mode | n | Route % | Crash % | Freeze % | Calls | Peak jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_on_vlm` | 15 | 47.4 ± 5.8 | 100.0 ± 0.0 | 0.0 ± 0.0 | 15.3 ± 1.8 | 84.3 ± 14.2 |
| `cert_replan` | 15 | 76.9 ± 15.7 | 0.0 ± 0.0 | 33.3 ± 24.7 | 0.0 ± 0.0 | 113.5 ± 22.0 |
| `fasionad_wakeup` | 15 | 86.8 ± 8.9 | 0.0 ± 0.0 | 33.3 ± 24.7 | 0.3 ± 0.2 | 42.1 ± 30.7 |
| `fixed_rate_vlm` | 15 | 47.4 ± 5.8 | 100.0 ± 0.0 | 0.0 ± 0.0 | 4.7 ± 0.5 | 90.9 ± 14.2 |
| `proposed` | 15 | 89.6 ± 6.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.3 ± 0.9 | 14.1 ± 0.3 |
| `proposed_no_blend` | 15 | 91.2 ± 5.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 5.0 ± 1.1 | 125.6 ± 4.1 |
| `proposed_no_margin` | 15 | 89.6 ± 6.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.3 ± 0.9 | 13.9 ± 0.4 |
| `static_cbf` | 15 | 77.4 ± 15.9 | 0.0 ± 0.0 | 33.3 ± 24.7 | 0.0 ± 0.0 | 0.7 ± 0.0 |
| `unshielded_skill` | 15 | 47.3 ± 7.3 | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.0 |

## Per-episode appendix

| Scenario | Mode | Seed | Repeat | Route % | Crash | Freeze | Calls | End reason |
| --- | --- | ---: | ---: | ---: | :---: | :---: | ---: | --- |
| `dense_highway` | `always_on_vlm` | 0 | 0 | 38.2 | yes | no | 14 | crash |
| `dense_highway` | `always_on_vlm` | 1 | 0 | 38.2 | yes | no | 14 | crash |
| `dense_highway` | `always_on_vlm` | 2 | 0 | 38.2 | yes | no | 14 | crash |
| `dense_highway` | `always_on_vlm` | 3 | 0 | 38.2 | yes | no | 14 | crash |
| `dense_highway` | `always_on_vlm` | 4 | 0 | 38.2 | yes | no | 14 | crash |
| `dense_highway` | `cert_replan` | 0 | 0 | 96.1 | no | no | 0 | timeout |
| `dense_highway` | `cert_replan` | 1 | 0 | 96.1 | no | no | 0 | timeout |
| `dense_highway` | `cert_replan` | 2 | 0 | 96.1 | no | no | 0 | timeout |
| `dense_highway` | `cert_replan` | 3 | 0 | 96.1 | no | no | 0 | timeout |
| `dense_highway` | `cert_replan` | 4 | 0 | 96.1 | no | no | 0 | timeout |
| `dense_highway` | `fasionad_wakeup` | 0 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `fasionad_wakeup` | 1 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `fasionad_wakeup` | 2 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `fasionad_wakeup` | 3 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `fasionad_wakeup` | 4 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `fixed_rate_vlm` | 0 | 0 | 38.1 | yes | no | 4 | crash |
| `dense_highway` | `fixed_rate_vlm` | 1 | 0 | 38.1 | yes | no | 4 | crash |
| `dense_highway` | `fixed_rate_vlm` | 2 | 0 | 38.1 | yes | no | 4 | crash |
| `dense_highway` | `fixed_rate_vlm` | 3 | 0 | 38.1 | yes | no | 4 | crash |
| `dense_highway` | `fixed_rate_vlm` | 4 | 0 | 38.1 | yes | no | 4 | crash |
| `dense_highway` | `proposed` | 0 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `proposed` | 1 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `proposed` | 2 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `proposed` | 3 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `proposed` | 4 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_blend` | 0 | 0 | 96.9 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_blend` | 1 | 0 | 96.9 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_blend` | 2 | 0 | 96.9 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_blend` | 3 | 0 | 96.9 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_blend` | 4 | 0 | 96.9 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_margin` | 0 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_margin` | 1 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_margin` | 2 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_margin` | 3 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `proposed_no_margin` | 4 | 0 | 97.3 | no | no | 2 | timeout |
| `dense_highway` | `static_cbf` | 0 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `static_cbf` | 1 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `static_cbf` | 2 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `static_cbf` | 3 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `static_cbf` | 4 | 0 | 97.7 | no | no | 0 | timeout |
| `dense_highway` | `unshielded_skill` | 0 | 0 | 39.5 | yes | no | 0 | crash |
| `dense_highway` | `unshielded_skill` | 1 | 0 | 39.5 | yes | no | 0 | crash |
| `dense_highway` | `unshielded_skill` | 2 | 0 | 39.5 | yes | no | 0 | crash |
| `dense_highway` | `unshielded_skill` | 3 | 0 | 39.5 | yes | no | 0 | crash |
| `dense_highway` | `unshielded_skill` | 4 | 0 | 39.5 | yes | no | 0 | crash |
| `merge` | `always_on_vlm` | 0 | 0 | 41.1 | yes | no | 12 | crash |
| `merge` | `always_on_vlm` | 1 | 0 | 41.1 | yes | no | 12 | crash |
| `merge` | `always_on_vlm` | 2 | 0 | 41.1 | yes | no | 12 | crash |
| `merge` | `always_on_vlm` | 3 | 0 | 41.1 | yes | no | 12 | crash |
| `merge` | `always_on_vlm` | 4 | 0 | 41.1 | yes | no | 12 | crash |
| `merge` | `cert_replan` | 0 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `cert_replan` | 1 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `cert_replan` | 2 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `cert_replan` | 3 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `cert_replan` | 4 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `fasionad_wakeup` | 0 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `fasionad_wakeup` | 1 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `fasionad_wakeup` | 2 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `fasionad_wakeup` | 3 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `fasionad_wakeup` | 4 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `fixed_rate_vlm` | 0 | 0 | 41.1 | yes | no | 4 | crash |
| `merge` | `fixed_rate_vlm` | 1 | 0 | 41.1 | yes | no | 4 | crash |
| `merge` | `fixed_rate_vlm` | 2 | 0 | 41.1 | yes | no | 4 | crash |
| `merge` | `fixed_rate_vlm` | 3 | 0 | 41.1 | yes | no | 4 | crash |
| `merge` | `fixed_rate_vlm` | 4 | 0 | 41.1 | yes | no | 4 | crash |
| `merge` | `proposed` | 0 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed` | 1 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed` | 2 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed` | 3 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed` | 4 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_blend` | 0 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_blend` | 1 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_blend` | 2 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_blend` | 3 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_blend` | 4 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_margin` | 0 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_margin` | 1 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_margin` | 2 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_margin` | 3 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `proposed_no_margin` | 4 | 0 | 100.0 | no | no | 6 | route_complete |
| `merge` | `static_cbf` | 0 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `static_cbf` | 1 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `static_cbf` | 2 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `static_cbf` | 3 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `static_cbf` | 4 | 0 | 100.0 | no | no | 0 | route_complete |
| `merge` | `unshielded_skill` | 0 | 0 | 66.9 | yes | no | 0 | crash |
| `merge` | `unshielded_skill` | 1 | 0 | 66.9 | yes | no | 0 | crash |
| `merge` | `unshielded_skill` | 2 | 0 | 66.9 | yes | no | 0 | crash |
| `merge` | `unshielded_skill` | 3 | 0 | 66.9 | yes | no | 0 | crash |
| `merge` | `unshielded_skill` | 4 | 0 | 66.9 | yes | no | 0 | crash |
| `two_trap` | `always_on_vlm` | 0 | 0 | 62.8 | yes | no | 20 | crash |
| `two_trap` | `always_on_vlm` | 1 | 0 | 62.8 | yes | no | 20 | crash |
| `two_trap` | `always_on_vlm` | 2 | 0 | 62.8 | yes | no | 20 | crash |
| `two_trap` | `always_on_vlm` | 3 | 0 | 62.8 | yes | no | 20 | crash |
| `two_trap` | `always_on_vlm` | 4 | 0 | 62.8 | yes | no | 20 | crash |
| `two_trap` | `cert_replan` | 0 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `cert_replan` | 1 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `cert_replan` | 2 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `cert_replan` | 3 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `cert_replan` | 4 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `fasionad_wakeup` | 0 | 0 | 62.7 | no | yes | 1 | timeout |
| `two_trap` | `fasionad_wakeup` | 1 | 0 | 62.7 | no | yes | 1 | timeout |
| `two_trap` | `fasionad_wakeup` | 2 | 0 | 62.7 | no | yes | 1 | timeout |
| `two_trap` | `fasionad_wakeup` | 3 | 0 | 62.7 | no | yes | 1 | timeout |
| `two_trap` | `fasionad_wakeup` | 4 | 0 | 62.7 | no | yes | 1 | timeout |
| `two_trap` | `fixed_rate_vlm` | 0 | 0 | 62.9 | yes | no | 6 | crash |
| `two_trap` | `fixed_rate_vlm` | 1 | 0 | 62.9 | yes | no | 6 | crash |
| `two_trap` | `fixed_rate_vlm` | 2 | 0 | 62.9 | yes | no | 6 | crash |
| `two_trap` | `fixed_rate_vlm` | 3 | 0 | 62.9 | yes | no | 6 | crash |
| `two_trap` | `fixed_rate_vlm` | 4 | 0 | 62.9 | yes | no | 6 | crash |
| `two_trap` | `proposed` | 0 | 0 | 71.5 | no | no | 5 | timeout |
| `two_trap` | `proposed` | 1 | 0 | 71.5 | no | no | 5 | timeout |
| `two_trap` | `proposed` | 2 | 0 | 71.5 | no | no | 5 | timeout |
| `two_trap` | `proposed` | 3 | 0 | 71.5 | no | no | 5 | timeout |
| `two_trap` | `proposed` | 4 | 0 | 71.5 | no | no | 5 | timeout |
| `two_trap` | `proposed_no_blend` | 0 | 0 | 76.5 | no | no | 7 | timeout |
| `two_trap` | `proposed_no_blend` | 1 | 0 | 76.5 | no | no | 7 | timeout |
| `two_trap` | `proposed_no_blend` | 2 | 0 | 76.5 | no | no | 7 | timeout |
| `two_trap` | `proposed_no_blend` | 3 | 0 | 76.5 | no | no | 7 | timeout |
| `two_trap` | `proposed_no_blend` | 4 | 0 | 76.5 | no | no | 7 | timeout |
| `two_trap` | `proposed_no_margin` | 0 | 0 | 71.6 | no | no | 5 | timeout |
| `two_trap` | `proposed_no_margin` | 1 | 0 | 71.6 | no | no | 5 | timeout |
| `two_trap` | `proposed_no_margin` | 2 | 0 | 71.6 | no | no | 5 | timeout |
| `two_trap` | `proposed_no_margin` | 3 | 0 | 71.6 | no | no | 5 | timeout |
| `two_trap` | `proposed_no_margin` | 4 | 0 | 71.6 | no | no | 5 | timeout |
| `two_trap` | `static_cbf` | 0 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `static_cbf` | 1 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `static_cbf` | 2 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `static_cbf` | 3 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `static_cbf` | 4 | 0 | 34.5 | no | yes | 0 | timeout |
| `two_trap` | `unshielded_skill` | 0 | 0 | 35.4 | yes | no | 0 | crash |
| `two_trap` | `unshielded_skill` | 1 | 0 | 35.4 | yes | no | 0 | crash |
| `two_trap` | `unshielded_skill` | 2 | 0 | 35.4 | yes | no | 0 | crash |
| `two_trap` | `unshielded_skill` | 3 | 0 | 35.4 | yes | no | 0 | crash |
| `two_trap` | `unshielded_skill` | 4 | 0 | 35.4 | yes | no | 0 | crash |

## Checkpoints, aliases, and fingerprints

Checkpoint aliases:

- `RETURN_RIGHT` -> `CRUISE_RIGHT`

| Checkpoint file | SHA-256 | Bytes |
| --- | --- | ---: |
| `CRUISE_RIGHT.pt` | `a0a83816e0338b6e` | 62829 |
| `OVERTAKE_LEFT.pt` | `f06f84cfa544f349` | 62729 |

## Honest notes

- This report summarizes 135 measured episode(s). Confidence intervals use `1.96 * sample_std(ddof=1) / sqrt(n)`; they are `0` for `n < 2` and are not meaningful for tiny samples.
- The default offline router is the deterministic **scripted** lane-threat rule. These results characterize the controller + shield loop; they are not evidence about general language-model behavior unless a live backend was explicitly configured. Backend used here: `scripted`.
