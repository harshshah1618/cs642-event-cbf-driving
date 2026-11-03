# Two-trap gauntlet

> **Pointer:** The full 3-tier research package (install / train / eval / multi-scenario stack) lives at [`../event-cbf-driving/`](../event-cbf-driving/). This directory is the earlier kinematic proof-of-concept only; do not treat it as the product.

Closed-loop proof of concept. Two continuous-action PPO policies are trained on an empty road, saved, loaded back, and driven through the gauntlet. A kinematic tracker is not used in the rollout.

## What is trained

Both skills are separate actor-critics. The observation is ego `y`, `vy`, `vx` only. Nothing in training is an obstacle.

- `CRUISE_RIGHT` regulates y = 0 m and vx = 20 m/s.
- `OVERTAKE_LEFT` regulates y = 4 m and vx = 20 m/s.

The action is longitudinal and lateral acceleration. The policy box is ax in [-6, 3.5] m/s² and ay in [-5, 5] m/s². The step is dt = 0.02 s:

```
x  += vx dt + 0.5 ax dt²
y  += vy dt + 0.5 ay dt²
vx  = max(0, vx + ax dt)
vy += ay dt
```

Episodes last 8 s. Initial vy is uniform on [-1.5, 1.5] m/s. Resets are a mixture: 55% within 1.5 m of the target lane at vx in [12, 22] m/s, 25% within 1 m of the lane at vx in [0, 8] m/s (the speeds the shield creates), and 20% up to 4.5 m from the lane at vx in [12, 22] so the skill can acquire the lane when the router engages it. The reward is an alive bonus minus (y − y_target)² minus 0.05 (vx − 20)² minus an action penalty (0.02 ax² + 0.35 ay²) and a small lateral-rate penalty (0.05 vy²).

The actor is a small tanh MLP plus a linear skip, with a squashed Gaussian whose zero maps to zero acceleration (the box [-6, 3.5] × [-5, 5] is applied after that). Two value heads sit beside it and are trained with bootstrap masks (keep probability 0.8), so they are not the same fit. Epistemic uncertainty is

```
σ = |V1 − V2| / spread
```

where `spread` is the standard deviation of |V1 − V2| on states the stochastic policy visits during a calibration rollout after training. The interrupt threshold is mean(σ) + 3 std(σ) on that same training distribution. It is computed once per skill and frozen. It is not adjusted to the gauntlet crash.

This run: cruise threshold 3.806 (mean 0.806, std 1.000); overtake threshold 3.135 (mean 1.000, std 0.712).

## What is scripted

The router does not call a language model. It is a lane rule with a 300 ms delay (15 ticks at 50 Hz): a vehicle ahead in the active lane traveling under 15 m/s requests the other skill. `CRUISE_RIGHT` switches toward `OVERTAKE_LEFT` for the stalled car. `OVERTAKE_LEFT` switches toward `CRUISE_RIGHT` for the cut-in.

- Fixed baseline: the rule is queried at 1 Hz and the new skill is applied with a hard switch after the delay. No shield.
- Proposed: the rule runs only when an interrupt fires, then the two trained policies are mixed with a 20-tick sigmoid. An interrupt is σ above that skill's training threshold, or the shield overriding ax.

The shield is a CBF on obstacles ahead in the lane corridor |y_obs − y| < 1.85 m:

```
h = (x_obs − x − 6) − 1.1 (vx − v_obs) − inflation
ax = clip(min(ax_rl, ((v_obs − vx) + 2.5 h) / 1.1), −7.5, 3.5)
inflation = 4 σ + (0.30 vx if the router delay is still pending else 0)
```

The headway in that expression is 1.1 s, not 0.8 s. With a 0.8 s headway and ax limited to −7.5 m/s², the barrier is not control-invariant from 20 m/s: a static shield still enters the |dx| < 4.5 m collision box (impact near 4.1 s at about 6 m/s). 1.1 s is the smallest tenth of a second that holds a stop on the 6 m standoff from 18–22 m/s. The standoff, the class-K gain 2.5, the inflation formula, and the shield clip are unchanged. Static CBF mode sets inflation to 0 and does not run the router. Proposed mode uses the inflation formula.

The road is 200 m and 12 s. Ego starts at x = 0, y = 0, vx = 20. A stalled car sits at x = 75, y = 0. Once ego reaches x = 92, a cut-in appears 23 m ahead in the left lane (y = 4) at 9.5 m/s. A collision is |dx| < 4.5 m and |dy| < 1.6 m.

## How to rerun

From this directory, with numpy, matplotlib, and torch:

```bash
python train_and_run.py
```

The script trains both skills on CPU, writes `checkpoints/`, reloads those weights, runs the four closed-loop modes, and overwrites the figures and this README with the numbers from that run. A few minutes is typical.

## Figures

`media/training-curves.png` is the mean undiscounted episode return against PPO update, one curve per skill. The points are the returns collected on that update.

`media/gauntlet-rollout.png` is the four closed-loop paths. Dashed lines are the lane centers y = 0 and y = 4. The dark car is the stalled obstacle. The red car and its trail are the cut-in. A red cross is a collision.

`media/gauntlet-proposed.gif` is the proposed rollout only.

## Empty-road check (10 s, deterministic)

| Skill | Mean \|y − target\| (m) | Mean \|vx − 20\| (m/s) | Final y (m) | Final vx (m/s) | Time to move 1.6 m from the other lane (s) | Learned |
| --- | --- | --- | --- | --- | --- | --- |
| CRUISE_RIGHT | 0.002 | 0.781 | -0.001 | 20.92 | 1.34 | yes |
| OVERTAKE_LEFT | 0.008 | 0.990 | 4.010 | 21.18 | 1.88 | yes |

Cruise is started on y = 0 and, for the lane-change time, from y = 4. Overtake is started on y = 4 and from y = 0. Both targets are vx = 20.

## Measured gauntlet

Route % is 100 × min(x, 200) / 200 at the end of the rollout. Average speed is distance traveled divided by time until collision, the 200 m mark, or 12 s. Lateral jerk is |Δay| / dt. Router calls count every query, including queries that do not change skill.

| Mode | Crash time (s) | Route % | Avg speed (m/s) | Router calls | Peak lateral jerk (m/s³) |
| --- | --- | --- | --- | --- | --- |
| 1  Unshielded cruise | 3.44 | 35.4 | 20.55 | 0 | 0.7 |
| 2  Cruise + static CBF | none | 34.5 | 5.75 | 0 | 0.7 |
| 3  Fixed 1 Hz router | 5.86 | 62.9 | 21.47 | 6 | 81.5 |
| 4  Proposed interrupt + CBF | none | 77.7 | 12.95 | 10 | 15.3 |

These measurements are what the closed loop did on this run. They do not all match the intended mechanism split (unshielded crash, static shield stops, 1 Hz router crashes, proposed clears both traps).

Proposed peak σ was 22.35. Static CBF ended at x = 69.06 m, y = 0.17 m, vx = 0.00 m/s.
