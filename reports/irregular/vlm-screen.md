# Bigger-VLM screen (2026-11-26, development frames only)

Question: Harsh's v3 pilot fails the flagger in 12/50 runs, all because the first look answers
`careful` while the sign shows STOP. Does a larger Qwen3-VL model read the sign better?

Setup: the 739 saved development frames (maps 100-149; flagger STOP/SLOW, billboard, shoulder
pedestrian, lane-closed sign, clean road), objects 8-35 m ahead. The first-look prompt is the v3
speed-free prompt with the PatchCore crop, voted over two option orders. The hold question is
Harsh's v3 release question, averaged over both orders. Crops come from the 448x252 frames,
not the 1344x756 render used in closed loop. 8B ran on an L4 and the others on an A100, so
the seconds-per-call figures are not comparable across models.
Stage `screen` in tools/irregular_modal.py; table from tools/irregular_screen_report.py.

| scene (acceptable) | Qwen3-VL-8B-Instruct | Qwen3-VL-30B-A3B-Instruct | Qwen3-VL-32B-Instruct |
|---|---:|---:|---:|
| clean (continue) | 53% (stop 1%, n=240) | 13% (stop 0%, n=240) | 50% (stop 1%, n=240) |
| flagger-shoulder:slow (careful) | 89% (stop 8%, n=76) | 99% (stop 0%, n=76) | 88% (stop 7%, n=76) |
| flagger-shoulder:stop (stop) | 67% (stop 67%, n=98) | 26% (stop 26%, n=98) | 58% (stop 58%, n=98) |
| pedestrian-shoulder (careful) | 92% (stop 2%, n=109) | 94% (stop 0%, n=109) | 74% (stop 11%, n=109) |
| sign-billboard-shoulder (continue) | 29% (stop 0%, n=109) | 8% (stop 0%, n=109) | 25% (stop 2%, n=109) |
| sign-lane_closed-shoulder (careful/left) | 93% (stop 0%, n=107) | 98% (stop 0%, n=107) | 88% (stop 3%, n=107) |
| **all scenes** | 67% | 47% | 60% |
| hold: releases while STOP shows (P(proceed) >= 0.47) | 4% of 98 | 0% of 98 | 5% of 98 |
| hold: releases once SLOW shows | 16% of 76 | 0% of 76 | 11% of 76 |
| hold: STOP-vs-SLOW separation (AUROC) | 0.85 | 0.90 | 0.92 |
| median seconds per call | 0.21 | 0.19 | 0.17 |

STOP recall by distance (first look, voted):

| Model | 15-25 m | 25-35 m | Wrong answers |
|---|---:|---:|---|
| Qwen3-VL-8B | 47/57 (82%) | 19/41 (46%) | careful 29, continue 3 |
| Qwen3-VL-30B-A3B | 19/57 (33%) | 6/41 (15%) | careful 72, continue 1 |
| Qwen3-VL-32B | 35/57 (61%) | 22/41 (54%) | careful 36, continue 4, right 1 |

**Conclusion:** neither larger model reads the STOP sign better at the distance where it
matters. The 8B is best at 15-25 m, which is where the v3 second look happens. The 30B-A3B
mostly answers `careful`. The 32B is slightly better far away and worse up close. We keep
Qwen3-VL-8B. The hold question separates STOP from SLOW better with the larger models
(AUROC 0.90-0.92 vs 0.85). But at the 0.47 threshold, every model releases rarely even once the
sign shows SLOW (0-16% of frames). The stuck runs come from that conservative threshold, not
from the model's size.
