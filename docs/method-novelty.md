# Is this driving stack a new method?

This note compares the 3-tier highway stack with published work. The stack uses a slow vision-language model that only picks a skill (cruise, overtake, or return), separate fast reinforcement-learning skills that do not see that decision, and a control barrier function that runs at the same rate as the skills. The barrier’s safe distance grows with the critic’s uncertainty, and, while a model call is in flight, by a term proportional to speed times that call’s delay. The slow model is not polled on a timer. It is called when critic uncertainty crosses a threshold or when the barrier has been overriding the skill’s acceleration. A skill change is mixed in over a short window. The running demo is a 20-tick sigmoid mix of the two actions. The writeup calls the intended mix advantage-gated smooth blending.

LeapAD and DriveVLM-Dual already pair a slow language model with a fast controller. They are not the closest match on the three hooks below, so they are not re-argued here.

## When the slow model is called

[FASIONAD](https://arxiv.org/abs/2411.18013) (Qian et al.; the later writeup is [FASIONAD++](https://arxiv.org/abs/2503.08162)) already refuses to run the slow model on every frame. A fast end-to-end planner stays in charge when a waypoint reward is high and the uncertainty of those waypoints is low. Otherwise a vision-language model is woken to comment on the scene, and that comment is fed back into the same planner. The uncertainty is a Laplace fit on waypoints, not a critic standard deviation, and nothing in the switch watches a safety filter.

[CERT-Replan](https://arxiv.org/abs/2610.09302) (Suganda and Hu) is the closest match on the interrupt itself. A one-step barrier filter still corrects the next control. A second monitor scores how likely the current plan is to violate the barrier over a horizon. When that risk exceeds a budget, the robot drops the plan and picks another speed profile, corridor, or path class, instead of letting the filter keep patching the same plan. The new plan is not chosen by a vision-language model, and the trigger is predicted barrier risk, not “the filter has been overriding acceleration” and not a critic threshold.

[Chang, Renganathan, and Ahmed](https://arxiv.org/abs/2510.10442) switch from a relaxed barrier to a more cautious one when the barrier residual is poor or the filter becomes infeasible. That is a residual trigger. It changes the filter, not the driving skill.

No paper found fires a vision-language skill choice when critic uncertainty crosses a line, or when the barrier has been overriding the reinforcement-learning acceleration.

## How the barrier grows

[Yu, Li, Xiong, Han, and Leng](https://arxiv.org/abs/2505.19939) put an ensemble critic in front of a high-order barrier filter for an intersection. Epistemic uncertainty is the spread of the critics. As that joint uncertainty rises, the barrier inequality gets stricter, and the filter is more willing to replace the reinforcement-learning action. The tightening is not a headway grown by the critic’s standard deviation, and there is no term for a slow model’s delay.

[Periotto, Ferizbegovic, Barbosa, and Sundin](https://arxiv.org/abs/2403.18650) grow the barrier margin by speed times delay, plus a term for delay jitter: the margin is \(\lVert v\rVert\) times the round-trip time, plus max speed times a gain times the delay’s standard deviation. The delay is the network round trip of a teleoperated robot, and they note that it also covers time spent computing the control. The margin is on whenever the delay is present. It is not added only while a vision-language call is in flight, and it does not depend on a critic.

[Mandaokar and Rinner](https://arxiv.org/abs/2512.00462) enlarge a UAV’s safe clearance by max speed times perception-to-actuation latency, plus a short braking distance, and then widen it again with a Cantelli bound on position noise. An early-warning layer expands the margin further when risk is already high. The latency is sensing and actuation, not a vision-language call, and the uncertainty is sensor noise, not a critic.

[Kumar, Liu, Fisac, Adams, and Ramadge](https://arxiv.org/abs/2112.12210) fold the variance of a learned barrier residual into the safety constraint, so the filter demands more slack where the model is unsure. That variance is of the barrier dynamics, not of an action-value critic.

Two other delay treatments do not inflate a distance. [Wabersich and Zeilinger](https://arxiv.org/abs/1812.05506) keep a backup trajectory and change the learned input when following it would leave the safe set. [Ushio](https://doi.org/10.1587/transfun.2024MAP0003) predicts the state across a computation delay, then applies a high-order barrier shield to that prediction. [Peng et al.](https://arxiv.org/abs/2606.20458) do handle a stale vision-language answer: the fast planner keeps running, the language model’s pick is faded out as it ages, and the robot can stop if the answer is too old. They do not grow a barrier by speed times that delay.

No paper found uses one barrier whose safe distance grows with critic uncertainty and, only while a vision-language call is in flight, by a speed-times-delay term.

## Blending two skills

[De Cooman and Suykens](https://arxiv.org/abs/2510.24674) give a highway car separate options for speed and for lane changes. The high-level policy picks an option. It does not mix two option actions over a short window.

[Kuang et al.](https://arxiv.org/abs/2506.09366) freeze several robot skills and train a gating network whose weights mix their actions. The weights come from the task, not from an advantage, and a humanoid is not choosing cruise versus overtake.

[Hou et al.](https://arxiv.org/abs/2002.02829) do use the advantage. Inside one soft actor-critic policy, mixture weights depend on the advantage of each component, so components specialize. That mix is the policy itself. It is not a short crossfade between two already trained driving skills after an outside model picks one.

No paper found under the name advantage-gated smooth blending, and no paper found that mixes two driving skills with a fixed sigmoid over a few dozen control ticks. The demo’s 20-tick sigmoid is a crossfade. The advantage gate in the writeup was not found as a published method.

## Stopping, and never changing lanes

[Trautman and Krause](https://doi.org/10.1109/iros.2010.5654369) named the freezing robot problem in 2010. In a dense crowd the planner decides every forward path is unsafe, so the robot stops even though a path exists. Their fix is to plan as if the other people will make room. It is not a skill switch, and it is not a barrier filter.

[Grover, Liu, and Sycara](https://arxiv.org/abs/1911.09146) showed that a barrier quadratic program can stall on the boundary of the safe set: the only control that stays safe also makes no progress. Their fix detects that equilibrium and steers the robots out while keeping them apart. [Xie, Zhan, and Chin](https://arxiv.org/abs/2604.04409) detect the same clash between the nominal plan and the barrier and nudge the nominal reference so the robots do not sit still. [He, Zeng, Zhang, and Sreenath](https://hybrid-robotics.berkeley.edu/publications/ACC2021_Lane-Change-CBF.pdf) use a state machine on a lane change: the maneuver starts only when the barrier problem is feasible, and it is aborted back into the original lane if it stops being feasible.

[CERT-Replan](https://arxiv.org/abs/2610.09302), cited above, states the related failure in plain terms. A filter that keeps correcting one plan does not change the plan. Their fix is to switch the plan when barrier risk stays high.

No paper found whose published failure is specifically “the shield brakes to zero and the nominal policy therefore never changes lanes,” with the fix being a vision-language skill interrupt. The stall, and the idea of answering it by changing the nominal behavior, are published. That exact pair is not.

Already published: a slow vision-language model woken by planner uncertainty, a barrier-risk interrupt that changes the plan, a barrier tightened by critic uncertainty, a barrier margin grown by speed times delay, advantage-weighted mixing inside one policy, and the stall that happens when a barrier filter keeps correcting one nominal behavior.

Not found: a vision-language skill interrupt on critic uncertainty or on barrier override; one barrier grown by critic uncertainty and, only during the model call, by speed times that call’s delay; or an advantage-gated blend of two driving skills. The demo’s sigmoid mix was not found as a published method either.

A novelty claim is not supportable. Each hook already has a close published neighbor, and the unpublished details are specifics of wiring, not a new method.
