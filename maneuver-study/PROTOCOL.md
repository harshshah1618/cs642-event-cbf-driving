# Maneuver-reserve test protocol and claim boundary

2026-11-11. Development and held-out condition grids are deterministic physical MetaDrive driving experiments, not tests of VLM accuracy. The same scripted router eventually says OVERTAKE_LEFT at the specified delay. PPO and PD use identical respective checkpoints/parameters across policies. Ordinary CBF, braking, road-boundary protection and fresh maneuver admission remain enabled. Optional inflation is disabled in every arm. Success means no collision and at least99% of the160m route within60simulated seconds. Freeze, route fraction and end time are retained. End time includes60s timeouts; successful-only time must not hide failures.

## Development and selection

Source/design hashes are frozen in runs/maneuver-matrix/development/lock.json before execution. The full36-condition grid crosses PPO/PD, steering15/25/40degrees, speed/gap14m/s/45m or26m/s/90m, and delay0.125/12/18s. Seven policies are baseline; constant optional reserve2/4/8/12/20m, retained until an admitted target change; and the curvature-derived geometric reserve at scale1. The first12 one-condition pilot episodes are exploratory and separate.

After development outcomes, choose the smallest swept constant reserve with36/36 success and zero crashes. This selects8m. This is an explicit development choice, not a preregistered tuning rule. Geometric reserve is not claimed superior: compare its held-out outcomes to this tuned constant with the SAME release policy.

## Held-out grid, frozen before starting held-out episodes

The48conditions cross PPO/PD, steering18/30/40degrees, speed/gap18m/s/55m or24m/s/80m, and delay0.125/6/18/30s. These combinations were not in development;18s and40degrees individually overlap. Four policies: baseline, selected8m admitted-target-release reserve, geometric reserve scale1, and static8m reserve never released. Static is a mechanism control, not the strongest competitor. Do not retune after viewing these outcomes. Each condition is run once with matched physical seed across policies. These are deterministic condition coverage, NOT48independent stochastic scene draws; do not attach an independent-seed confidence interval. Different delays/controllers share initial geometry. All initial states and actual actuator geometry are saved and checked.

## Policy meaning and limits

A reserve arms when a request is pending and there is a slow lead in the target corridor. It persists through stopping, no-op replies and rejected replies; only a changed target admitted by the existing fresh gap guard releases it. No rearming during a lane transition. The original stopping clearance remains2m. The geometric reserve is a symmetric two-arc curvature heuristic, not a swept-footprint safety/reachability certificate. Parameterized steering changes actual MetaDrive vehicle.max_steering and config together; snapshots report actual wheelbase and steering limit.

This setup isolates the stopping/turning interaction. It does not prove safety under arbitrary traffic, delayed wrong instructions, perception error, actuator uncertainty or curves. A correct-intent stream is favorable and must not be described as a learned-VLM benchmark. Full factorial safety layers and prior scheduling results remain separate. No result establishes physical impossibility of escape from a stalled state.

## Separate actual-model exploratory pair

After choosing8m, rerun the prior failure seed6301 with the actual trained BF16 adapter and physical camera, once with baseline and once with8m release reserve. Same initial world must be verified. Timing uses each measured local CPU prediction plus camera capture; it varies between arms, so this pair is supporting integration evidence, not the controlled causal comparison or a GPU/generalization result. Sources, prompts and adapter are locked before model loading; all real images, answers, delays, guard rejections and traces are retained. No fallback replies.
