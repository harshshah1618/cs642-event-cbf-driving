# CS642 milestone report

Akash Ravi, Harsh Shah, Ved Chadderwala  
November 11, 2026

## What we said we would do

Build one closed-loop driving stack and measure it. A slow model only picks a skill (cruise, pass, return). Frozen PPO skills drive at 50 Hz. A safety filter can brake from the current scene, and can add extra gap while a skill call is still out. We call the model when the active skill looks unsure, or when the filter has been overriding for a while. We compare that with no filter, a filter that never changes skill, and asking on a timer. Same skills, same seeds, same roads.

## What we built

The skills were trained on an empty road to hold a lane at 20 m/s, then frozen. Cruise is the right lane. Overtake is the left lane. Return reuses cruise.

The measured study runs in MetaDrive, so the car and the other traffic move under physics. The car is given true positions. There is no camera. CARLA was left as a stub.

We also ran a real local model, Qwen2.5-1.5B on CPU, on 18 drives. Most comparison numbers use a rule router so the delay is controlled.

## Results

15 normal drives, 5 seeds, three roads (stalled car plus a cut-in, a merge, and denser traffic). Crashes and stalls are counts. Route and calls are averages.

| Method | Route | Crashes | Stalls | Calls |
| --- | ---: | ---: | ---: | ---: |
| Skill only | 37.9% | 15 | 0 | 0 |
| Filter, no skill change | 79.0% | 0 | 5 | 0 |
| Ask all the time, with filter | 100% | 0 | 0 | 36.5 |
| Ask on a timer, with filter | 99.8% | 0 | 0 | 14.5 |
| Ours (ask only when needed) | 99.8% | 0 | 0 | 9.2 |
| Ours, no extra gap while waiting | 100% | 0 | 0 | 9.7 |

On 42 harder drives (a lead car braking, cut-ins, tighter gaps, small timing noise) our stack had 0 crashes and 0 stalls. A few drives ended short of the full route. The worst case still finished about 92%.

The filter is what stops the crashes. Changing skill is what stops the stalls. Asking only when needed finishes about the same route as asking all the time, with about a quarter of the calls.

## What did not work

The 1.5B model made 62 real calls and did not beat a rule that waited the same amount of time. Answers took about 3 to 7 seconds.

Turning on the extra gap while a call is in flight did not clearly help. On the normal drives, leaving it off finished slightly more of the route. With the slow model, one no-gap drive stalled.

The gap target still goes negative on some ticks. Zero crashes in this sample is not a safety proof. The FASIONAD-style and CERT-style rows are our own stand-ins on this simulator. They are not the published systems, and we are not claiming to beat those papers.

## Where we are

The stack, the comparison, the local-model check, and the write-up are done. We are not doing CARLA.
