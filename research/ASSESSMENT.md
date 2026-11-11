# What this project actually built

The original system is a hierarchy for choosing and executing a small set of
driving maneuvers. A slow router chooses a maneuver, a fast controller produces
acceleration commands, and a safety filter plus monitor can change or reject
those commands. It is substantial executable engineering, with physical
MetaDrive integration and real local-model calls available.

The learned part is narrower than the architecture diagram suggests. Two frozen
PPO policies regulate lane position and speed from `[y, vy, vx]`; they do not
observe surrounding traffic. RETURN_RIGHT reuses CRUISE_RIGHT. The router and
fresh traffic guards supply the traffic awareness. Critic disagreement measures
model disagreement and cannot, by itself, tell whether another car is coming.
The bulk controlled study uses a rule-based maneuver router. The real language
model is a separate small integration check, not the source of the main win.

A CBF-style filter means the program solves a constrained action correction.
It does not mean the entire sampled, delayed, approximate physical system has a
proved safety certificate. The common guards do substantial work. The new
curved-road study uses a separate classical controller, not the original PPO
skills magically generalizing to curves.

# What the research work added

The testable question is when to spend a slow planning call: poll on a clock,
react to immediate usefulness/TTC, or predict an opportunity closing before the
planner can answer. The new code implements that deadline hypothesis and compares
it against tuned polling, condition-tuned polling, gated polling and matched
latency ablations. Choices are frozen on development data before held-out runs;
raw rows, checkpoint/source hashes and exact experimental cells are audited.

The best result is a scoped benchmark win. With randomized initial polling
offsets, deadline scheduling uses 1.025 calls/episode versus 2.769 for condition-
tuned polling: 63.0% fewer. Observed native success is 360/360 versus 357/360;
both groups have zero observed crashes. Native success requires no crash and at
least 99% route completion. The original zero-offset suite does not meet the
report's exploratory success-tolerance screen, so this is not universal dominance.

The strongest mechanistic result changes only the initial polling offset on
matched physical seeds. At 0.1 Hz, success falls from 100% to 78.33%; at 1 Hz the
effect reverses and is much smaller. This shows rate-dependent timing sensitivity
in this implementation. It is useful evidence, not an established novel discovery.

# The uncomfortable results

- Simple TTC and no-latency triggers remain competitive. The expensive-sounding
  latency term has no established incremental benefit here.
- The fixed PD and frozen PPO banks have the same observed success and aggregate
  call count in the 1,200-run controlled subset. This benchmark has not shown
  that learning is necessary for these lane-regulation tasks.
- Procedural roads still produce crashes and timeouts. Lower query counts cannot
  turn that into a safety claim.
- Actual Qwen inference takes seconds on this CPU. Both live systems stall in
  one scenario; six episodes cannot establish that an LLM helps.
- Bulk native runs lack achieved-jerk records. Nine full replays validate a
  representative subset, not comfort across the whole population.
- Close prior work already covers event-triggered MPC, risk-aware replanning and
  robot metareasoning. There is no honest basis for a broad "first" claim.

# What is worth presenting

Present an audited scheduling study and a timing-sensitive benchmark failure,
with a conditional call-efficiency win. The contribution is stronger than a
three-box architecture demo, but it is not yet a new autonomous-driving algorithm
or a top-conference result. The next research step would need a fresh benchmark
where prediction demonstrably beats simple opportunity/TTC rules, faithful
published opponents, multiple independently trained controllers, and an explicit
compute/progress objective. Those are future experiments, not results obtained
in this delivery.

Full numerical evidence: `results/research-report.pdf`, `results/statistics.json`,
`results/condition-tables.md`, compressed raw data and the independent audits.
