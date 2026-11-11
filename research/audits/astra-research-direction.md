# Independent research-direction audit

Date: 2026-11-11. Scope: prospective protocol, current code and primary literature. No implementation changes and no new physical benchmark run performed by this audit. Existing experimental claims must come from the main experiment logs.

## Assessment

The current deadline trigger is a useful ablation, not yet a defensible algorithmic novelty. In `scheduling.py:95-103`, it is a usefulness gate followed by a threshold on min(time to front constraint, alternative expiry), with threshold latency + transition + buffer. A win over TTC=3 seconds can arise simply from tuning that threshold. The strongest cheap opponent is immediate benefit with the same observations, admission and pending-call semantics. Earlier requests cannot be worse on a single monotonically closing opportunity if they incur no extra physical cost and produce the same useful decision; a deadline win needs an explicit mechanism such as transient false opportunities, stale decisions, commitment cost, or occupied planner capacity. Establish that mechanism instead of interpreting any win as proof of predictive reasoning.

The most defensible current contribution is an empirical and mechanistic finding: planner request timing, occupancy and opportunity lifetime interact, so nominal polling frequency is an inadequate measure of responsiveness. This is a plausible project contribution, not established field novelty. Random-clock-phase results and counterfactual occupancy ablations are valuable even if the deadline heuristic loses.

## Code findings requiring care

1. **Confirmed trigger/planner action-space mismatch on upstream roads.** `plan_maneuver` chooses both lane and speed, including zero. `road_request_due` considers lane-change benefit and route validity only. A stale slow speed reference can remain indefinitely when the obstruction clears. Executed directly against the current package:

```python
blocked = RoadView(8, 0, (LaneView(0, front_gap=8, front_speed=0),))
clear = RoadView(0, 0, (LaneView(0),))
plan_maneuver(blocked)  # (0, 0.0)
plan_maneuver(clear)    # (0, 11.11)
# road_request_due(clear, p, 2) is False for all four:
# benefit, ttc, deadline, no_latency
```

This is a functional counterexample, not proof that existing physical failures have this cause. Inspect traces for reduced speed reference plus unconstrained forward road plus no pending request. If relevant, retain frozen results as the initial system study and version a correction. The smallest clean isolation is lane-only high-level decisions with fresh IDM speed regulation common to every scheduler; alternatively add a shared speed-plan-obsolescence trigger to all event methods. Do not give only the proposed method recovery behavior.

2. **Latency ablation is not identical except for latency.** Road `no_latency` retains latency in the route-invalid branch (`road_planning.py:67-69`). Also `research_roads.py` assigns the validation-selected buffer only to deadline; no_latency always uses 0.5. The native script similarly defaults to 0.5 and changes only deadline. Unless selected buffer is exactly 0.5, this is a latency-plus-buffer comparison. A correct mechanistic ablation copies the frozen selected buffer and changes latency alone. Keep existing rows but label accurately.

3. **Finite windows are not modeled symmetrically.** Native expiry includes front and rear closing vehicles. Road expiry only includes rear closing vehicles. Road choices must be admissible now for a full two-second transition, but need not remain admissible after latency. The predicate triggers even when the window is already too short to survive latency. Count late-at-request and rejected-at-arrival calls separately.

4. **Multi-lane aggregation can spuriously trigger.** Road expiry is the minimum across useful alternatives. If one alternative closes imminently but another equally good one stays open, the trigger still treats the opportunity as expiring. The relevant object is loss of the best attainable useful alternative, not earliest expiry of any alternative. This is a plausible small algorithmic refinement, not demonstrated novelty. A three-lane paired case can isolate it.

5. **Same-skill occupancy must be verified through resolved configuration.** `_submit` has a legacy fast path applying same-skill replies immediately when `pending_margin_on_all_calls=False`. The research config extends `metadrive.yaml`, which sets this True, so inspection suggests research already avoids that fast path. Record the resolved flag in provenance. Roads always occupy the pending slot even for unchanged decisions. The causal study must use consistent semantics.

6. **Calls are not total computation.** Compare scheduler CPU + planner CPU, and include common observation/prediction cost when claiming end-to-end efficiency. A cheap deterministic router with injected delay establishes latency sensitivity, not actual expensive-planner compute savings. Report injected delay and measured runtime separately.

## Closest verified literature and what it rules out

- [CERT-Replan, 2026](https://arxiv.org/html/2610.09302v1): calibrated finite-horizon barrier risk decides when to replace a local planning mode, with an independent one-step safety filter. Its Algorithm 1 solves MPC/filter before assessing mode replacement. It is not interchangeable with a cheap lane-opportunity trigger, and locally naming a baseline after it does not reproduce its calibration, CVaR, or MPC. It blocks a broad claim that forecasting risk to trigger semantic mode changes is new. A proper source reproduction would require its actual uncertainty and candidate-mode machinery, not threshold substitution.
- [Dang et al., 2022](https://arxiv.org/abs/2208.10302), [released source](https://github.com/DangFengying/RL-based-event-triggered-MPC): learned event-triggered vehicle path-following MPC. Inspected upstream commit `3c1c1f232510e270fa03fb3ae2ef30a2f442fcbc`, `veh_env2.py:192-226`: binary action solves MPC or advances the existing control sequence; reward is negative tracking cost minus rho times trigger action. The 12-dimensional observation concatenates physical state and predicted state. The simulation advances by fixed dt=0.2 after solving; this code does not itself model solver wall time as physical delay. It is a relevant learned scheduling baseline, but transferring its network to lane planning without retraining is invalid. Adaptation should be named inspired-by, trained across multiple seeds and compared at matched observation access.
- [Learning Robot Metareasoning, 2026](https://arxiv.org/html/2610.23662v1): learns adaptive computation/commitment durations as an SMDP, conditioning on state, executing plan and cheap solver statistics. It includes time-to-infeasibility among statistics and evaluates navigation, quadrotor and manipulation. This is a particularly strong novelty obstacle for generic latency-aware adaptive scheduling or value-of-computation claims. The local system differs because fresh low-level feedback continues while only semantic choice is delayed, and injected latency does not improve planner quality. That distinction defines the research question; it does not by itself establish novelty.
- [Zhou et al., 2025 real-vehicle event-triggered MPC](https://jchen2020.net/pubs/j_ietcta2025-a.pdf): full-size-vehicle path tracking and discussion of improved performance with fewer solves under computation delay. Therefore “replanning more often can hurt” is already known. A local result needs a more specific mechanism than that slogan.
- [VLALagBench, Nov 2026](https://arxiv.org/abs/2611.03103): studies latency sensitivity across tasks, object speeds and synchronous/asynchronous execution. This blocks claims that a latency sweep or asynchronous execution evaluation is itself new. It concerns VLA manipulation, not this maneuver hierarchy; cross-domain numerical comparisons would be misleading.

## Strongest practical baseline set

For the current frozen planner, prioritize immediate benefit; validation-tuned TTC thresholds using the same tuning budget as deadline; deadline with the identical chosen buffer and zero latency; all periodic rates with random phase and per-condition validation selection; and an always-free/high-rate planner diagnostic that is explicitly an oracle/upper-reference for occupancy, not a fair compute baseline. Existing upstream IDM and Expert agents test absolute driving competence, not scheduler superiority because their control/planning stacks differ.

If introducing a learned trigger, use a small binary ask/wait model on the same opportunity features plus pending state and elapsed request age, with a transparent safety/progress/call objective and several training seeds. This is stronger than a new large model and directly tests whether the analytic trigger adds anything. It is an adaptation of established learned scheduling, not a novel baseline invention.

## Small next experiment: separate timing from server occupancy

First finish and report the frozen results. Then register a disjoint mechanistic experiment before viewing its outcomes:

- Factor A: useful-window width relative to fixed latency + measured maneuver duration.
- Factor B: initially idle planner versus a same-skill request occupying it when the window appears.
- Factor C: periodic clock phase; use several phases for the same physical seed, not only one random phase for each different seed.
- Methods: immediate benefit, deadline, conditioned periodic, and periodic plus cheap usefulness gate. The latter tests whether avoiding irrelevant work explains the gain without a deadline predictor.
- Hold the planner and fresh admission/backup fixed. Log opportunity appearance, request submission, completion, accepted maneuver, initial occupied duration, and whether expiry happened before submission or before application.

An elementary diagnostic: with one exogenously appearing window of width W, idle server, polling period P, fixed latency L, required maneuver duration T, and uniform residual polling phase, the completion probability is clamp((W-L-T)/P, 0, 1). This is a simple derived reference, not a claimed theorem about the coupled vehicle simulation. Event-triggered immediate dispatch succeeds under these assumptions whenever W >= L+T. Any empirical departure should be explained by occupancy, trajectory coupling, prediction error or admission. It gives the study a falsifiable mechanism rather than another aggregate table.

## If a method is still needed

Test **remaining useful-option reserve** instead of the minimum expiry. For each useful adjacent lane estimate its last feasible request time after subtracting latency and transition; combine alternatives through the best surviving value, not minimum time. Ask when delaying one more decision interval loses a useful option that cannot be replaced by an equally valuable surviving option. Include identical logic with latency zero and the immediate-benefit baseline. Start with the three-lane case where one option closes while another remains open. This is an unvalidated hypothesis; do not call it novel without a narrower related-work search, and do not enlarge the benchmark until it beats the strongest cheap alternatives for an understood reason.

If immediate benefit ties or dominates throughout, the honest result is that under a fresh low-level safety loop and inexpensive deterministic maneuver selection, deadline sophistication is unnecessary in these regimes. A well-supported boundary result is more defensible than tuning until some baseline loses.
