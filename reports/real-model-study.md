# Real camera model, final stack

The router on every call is the trained camera model. No scripted lane rule drove the car.
The system text is the training prompt. The only difference between the two arms is the speed number in the state line: always 20 m/s, or the measured speed.

A finish is no crash and at least 99% of the route. Intervals are 95% Wilson intervals.

| Scene | State line | Finished | Crashed | Stalled | Mean route | Calls | Letter agreement |
|---|---|---:|---:|---:|---:|---:|---:|
| stall | pinned | 4/4 (51%–100%) | 0 | 0 | 100% | 1.0 | 100% |
| stall | measured | 4/4 (51%–100%) | 0 | 0 | 100% | 1.0 | 100% |
| cut_in | pinned | 4/4 (51%–100%) | 0 | 0 | 100% | 2.0 | 100% |
| cut_in | measured | 4/4 (51%–100%) | 0 | 0 | 100% | 2.0 | 100% |
| pass_return | pinned | 4/4 (51%–100%) | 0 | 0 | 100% | 2.0 | 100% |
| pass_return | measured | 4/4 (51%–100%) | 0 | 0 | 100% | 2.0 | 100% |
| dense | pinned | 2/2 (34%–100%) | 0 | 0 | 100% | 0.5 | 0% |
| dense | measured | 2/2 (34%–100%) | 0 | 0 | 100% | 0.5 | 0% |

Paired seeds: 14. The letter sequence changed on 0 of them.

Letter agreement is an offline score against a geometric reading of the trained prompt (right-lane blocker within 100 m and slower than 15 m/s; left lane occupied within 20 m). That score did not choose a skill.

## What the drives actually did

28 episodes, 0 crashes, 0 stalls. Every route finished. Each forward pass took about 14–15 seconds. There were 4 seeds per scene, so a 4/4 finish still has a wide interval: about 51% to 100%. This is a fixed set, not a large test.

The model was the only router. On the stall scenes it said pass (B) once, at the start, while still near 20 m/s, and that pass was used. On the pass-then-return road it said pass, then return (C). All 8 of those returns were used, and the car ended in the right lane. That is the return the earlier camera-model sample never completed.

On the cut-in scenes the model also said pass, then return. The return was still being computed when the 200 m route ended, so it was not used. The car finished in the left lane. The cut-in occurred during the real wait, and the car did not hit it.

On dense traffic, one seed never asked the camera: neither lane was a useful move, and the car finished in the right lane. The other seed asked once. The model said stay (A). A car was 29 m ahead in the left lane, outside the 20 m window used by the offline score, so the score said pass and the agreement for that call is 0%. The stay was not a lane change. The car finished.

The measured-speed line did change the text. On the second call the car was at about 12 m/s, and the training line still said 20 m/s. The letter was the same on all 14 paired seeds. No call in this set was made while the car was stopped, so this does not test the case where the text says 20 m/s and the car is already at rest.

Records: `runs/real-model-study/episodes.jsonl`. Protocol, fixed before the runs: `runs/real-model-study/protocol.json`.
