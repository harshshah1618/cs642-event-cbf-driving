# The result in simple terms

The safety system matters. Some protections overlap. The tests have not established that the fast driver needs to use PPO.

## What happened when we removed protections?

These are the same 288 matched simulated drives with the PPO driver, including intentionally faulty router decisions. They are diagnostic challenges, not a real-world crash-rate estimate. Road-boundary and invalid-data checks remain active throughout.

| Traffic protections active | Drives with a crash |
|---|---:|
| None of the three | 280 / 288 |
| Longitudinal CBF only | 42 / 288 |
| Braking backup only | 2 / 288 |
| All three, including lane-change guard | 0 / 288 |

The CBF prevented many crashes on its own. The braking backup handled many of the same hazards. With that backup already active, removing the CBF did not change crash/success counts on this sample. That is overlapping coverage, not evidence that the CBF never helps. Removing only the braking backup from the full PPO system caused 36 crashes; removing only the lane-change guard caused two.
A lane-change guard alone is not enough: it rejects an unsafe maneuver but does not supply braking. With both longitudinal protections absent it increased PPO collision episodes from 280 to 286 in this mix. Protection mechanisms need to be assessed together, not assumed helpful in every combination.

## What did PPO add?

With full protections, PPO met the route-success criterion in 228 of 288 drives and simple PD in 226 of 288; both recorded zero crashes. Success means no crash and at least 99% route, including near-finish timeouts; it is not always exact arrival. PPO had lower average achieved jerk RMS but higher peaks. Without traffic protection, PPO crashed more often than PD. These particular lane-regulation policies do not demonstrate a decisive learning advantage.

PPO is producing the driving commands. A simpler controller can fill the same role on these tasks. This experiment tests those existing policies and fixed gains; it does not prove what independently retrained policies could achieve.

## Does a safe car still need a good decision maker?

Yes. With all protections and PPO, the ground-truth state-rule router met the same success criterion in 91/96 drives. A deliberately stubborn router that always repeated its current maneuver did so in only 46/96. Both had zero crashes: safety could stop the car but could not supply missing maneuver decisions. These routers are diagnostic rules, not the teammate's VLM.

## What about the teammate's camera model?

The exact GitHub commit and saved results were recovered and checked. The code now supports a camera-to-vLLM connection, both the exact original prompt and a hint-free prompt, and measured capture/request timing. Both paths were exercised with actual camera frames and HTTP using programmed replies, not a model.

Actual driving with the fine-tuned VLM is still pending: its trained weights are in the teammate's private Modal volume. Source code and saved prediction rows cannot substitute for those weights. TEAMMATE_HANDOFF.md gives the exact export/serving commands and the ready-to-run evaluation command.

Full evidence: report.md, statistics.json, all-conditions.md, raw pilot/test rows, source lock and eleven exact action replays. Twelve shared traffic-seed clusters; no real-vehicle safety claim or universal PPO verdict.
