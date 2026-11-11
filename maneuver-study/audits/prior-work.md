# Maneuver reserve: prior-work and falsifiable contribution audit

Date: 2026-11-11. Exploratory research proposal, not a novelty or safety claim. Primary papers were inspected through author manuscripts/arXiv and the publisher review. No experiment was run for this audit. Scope: a delayed but correct high-level intent can arrive after a brake-safe stop has lost the space needed by the installed forward-driving lane-change controller.

## Verdict

“Brake earlier to preserve a maneuver” is not defensibly new by itself. Future feasibility, recoverability, backup-controller selection, safety-induced deadlocks, delay compensation, and shielding-aware planning all have substantial prior art. The strongest plausible narrow contribution is an **asynchronous maneuver-admission contract**: retain a measurable lane-change execution reserve while high-level intent is unresolved; allow that extra reserve to be consumed only after a fresh state-based guard admits the maneuver; retain the original collision envelope throughout. This remains a hypothesis. Merely replacing a constant braking clearance with another constant and releasing it when B arrives is an engineering heuristic, not a new viability method.

A useful publishable empirical result could instead be a reproducible distinction between correct maneuver classification, timely intent admission, and successful physical execution, with a causal account of why a safe low-level filter defeats late correct decisions. That would be a scoped systems finding rather than discovery of CBF deadlock.

## Closest primary literature

| Primary source | Established contribution relevant here | Implication for our claim |
|---|---|---|
| [Hu, Nakamura, Fisac, SHARP, RA-L 2022](https://arxiv.org/abs/2110.00843) | Plans while accounting for future shielding; driving/overtaking examples show that reactive last-resort safety can degrade performance. | Closest conceptual baseline. “Reason about the shield to avoid bad stops” is already established. Our potentially distinct object is pending asynchronous semantic intent and its admission lifecycle. |
| [Wabersich and Zeilinger, Linear Model Predictive Safety Certification, CDC 2018](https://arxiv.org/abs/1803.08552) | Certifies an input through a feasible trajectory to a safe terminal set, with robust constraints. | Preserving a feasible continuation is standard. A maneuver-reserve heuristic is weaker than this and must not be presented as a general feasibility theorem. |
| [Janwani et al., Multiple Backup Control Barrier Functions, ICRA 2024, author manuscript](https://tamasmolnar.com/publication/2024_Janwani-et-al_safe%20human-robot%20collaboration_ICRA_conf.pdf) | Chooses among backup controllers using learned driver intention, with bounded-input safety and reachable-set improvements. | Strong counter to “intent-conditioned safety reserve is new.” Distinction would require pending/completed/rejected intent timing and preserving options before intent becomes available. |
| [Goncalves et al., Using Circulation to Mitigate Spurious Equilibria, 2023](https://arxiv.org/abs/2307.09901) | Analyzes undesirable CBF equilibria and adds circulation constraints to avoid them. | Safe nonprogress and CBF-induced equilibria are known. Do not call a stalled rollout a new deadlock phenomenon. |
| [Rule-Based Safety-Critical Control with Application to Autonomous Lane Change, 2021](https://arxiv.org/abs/2103.12382) | FSM plus CLF/CBF control selects safe lane-change opportunities using traffic and driver commands. | Hybrid maneuver guards combined with CBF lane changes are prior art. |
| [Frauenfelder, Wiltz, Dimarogonas, Decentralized Vehicle Coordination and Lane Switching, 2023](https://arxiv.org/abs/2305.04885) | Maneuver indications change vehicles' safe regions through coordination functions. | Intent-conditioned safe regions are established, though this cooperative synchronous setting differs from delayed model output. |
| [Van Koevering et al., Probabilistic Safety and Feasibility-Assured eCBF Control, 2022](https://arxiv.org/abs/2205.03744) | Runtime optimization ensures feasibility while treating motion uncertainty; lane change and intersections. | Constraint feasibility is established. Distinguish feasible QP input, safe stopping, and reaching the lane-change goal. |
| [Sudvarg, Clark, Gill, Integrated Real-Time Control and Scheduling, RTAS 2025](https://www.sudvarg.com/publications/RTAS2025_Integrated_Control_and_Scheduling.pdf) | State-dependent timing requirements and safe handoffs between performance and backup controllers. | Broad control/scheduling co-design or delay-aware handoff claims are not new. |
| [Kim et al., Robust Safety-Critical Control for Input-Delayed Systems with Delay Estimation, ECC 2024](https://paperhost.org/proceedings/controls/ECC24/files/0872.pdf) | Predicts the state at actuation under estimated bounded input delay to enforce robust safety. | Adding speed times latency is not a new delay-handling principle. Our high-level decision latency differs from actuator delay, but still requires an explicit model. |
| [Hsu, Hu, Fisac, The Safety Filter, 2024 review](https://doi.org/10.1146/annurev-control-071723-102940) | Unifies reachability, CBF, backup and predictive filters, distinguishing safe constraints from controlled-invariant sets. | Use its terminology: collision safety does not imply mission progress. It is a taxonomy source, not independent experimental evidence. |

The audit is not exhaustive and cannot establish absence of an earlier identical method. Particularly close follow-up: timed reach-avoid backup CBFs and backup-controller switching, which already formalize preserving reachability to a target while avoiding obstacles. [Author manuscript on timed reach-avoid](https://personal.hkust-gz.edu.cn/jiyiding/files/publications/journal/26LCSS_BackupCBF.pdf).

## Actual code mechanism and limits

At the inherited source inspected here, `event-cbf-driving/event_cbf_driving/shield/cbf.py:263` defines a discrete stopping envelope. It subtracts a fixed `braking_clearance`, relative braking distance, and a response allowance from longitudinal separation, then limits longitudinal acceleration. This certifies neither lane-change completion nor progress. `env/metadrive.py` uses a physical steering/force adapter with bicycle-model feedforward; the car cannot translate laterally in place like a point-mass model. These facts make the proposed failure plausible, not yet established.

There are three different diagnoses:

1. **Controller-relative failure:** installed lane skill plus shield cannot escape the stopped state. A different steering trajectory might escape.
2. **Restricted maneuver infeasibility:** no path exists within a specified forward-only, bounded-curvature, road-constrained maneuver family.
3. **Physical impossibility:** no admissible control trajectory exists under the full dynamics. Failed rollouts or grid search do not prove this.

Use the first label unless stronger evidence is supplied. A lateral controller failing at zero speed can also be an actuator-adapter issue. Test direct steering creep or a trajectory-following lane-change controller from the same saved state before calling it unrecoverable.

## Precise hypothesis and reserve definition

Hypothesis H1: for paired scenes with a fixed, eventually correct intent stream, a reserve tied to current maneuver reachability increases late-intent completion without increasing collision rate relative to validation-selected constant clearance, at comparable benign-scene progress cost.

Hypothesis H2: fresh admission, rather than the extra distance alone, explains the benefit when intent can be stale, rejected, absent, or contradicted by a newly blocked target lane.

Let `C_stop` denote the existing stopping-safe set. Let `R_m` denote states from which a specified bounded-input lane-change policy can reach its target lane without collision and while retaining a valid stop fallback. While intent is pending, aim to remain in `C_stop ∩ R_m` for at least one admissible useful maneuver; after admission of m, execute a checked continuation in `R_m` while maintaining the original collision constraints. This is a proposed contract, not a proved invariant. If no maneuver is admissible, stop safely rather than force progress.

A rollout-based sufficient reserve can estimate the forward distance consumed by an explicitly defined lane-change trajectory until footprints separate. Include current speed, heading, lane width, steering/rate limits, acceleration, obstacle footprints, and any remaining decision delay. Reject trajectories that violate constraints; do not infer feasibility solely from `v * transition_time`. A scalar `gap - estimated_required_distance` is an empirical predictor unless the model error and reachable tube are bounded. At zero speed, the required geometric travel distance need not vanish.

The reserve needs to remain active before a request if opportunity can be lost before request launch. A “pending only” margin cannot recover distance already spent. Release extra reserve only when the guard admits a trajectory, not on any syntactically valid response. The original collision clearance remains enforced. Rejections/timeouts must not release it. A long unresolved request must not prevent a cheap local safe-stop action.

## Small decisive experiment

First isolate control mechanics without additional VLM calls: replay identical intended letters at controlled completion times in the physical simulator. This is explicitly a decision-delay experiment, not evidence of trained-model accuracy. Save the state at the first stop and at intent admission.

Use paired seeds, the same nominal skill and base safety envelope, and a prespecified delay grid spanning early, near-threshold and late admission. Include varied obstacle gaps, ego speeds, lane widths/curvature or steering limits. Separate development scenes from held-out scenes. Essential conditions:

- Existing clearance.
- Constant larger clearance, with at least a small validation sweep and locked selection; show the entire success/progress Pareto curve so an undertuned constant cannot be defeated artificially.
- Same reserve magnitude/profile, but released on a timer or any response, irrespective of fresh admission. Match release timing in the all-correct case; use stale/rejected responses to distinguish semantics.
- Fresh-admission reserve (candidate).
- Immediate correct intent, and a cheap geometry-based lane-change policy, as attainable upper/reference bounds.

Controls that matter: zero latency; no obstacles; target lane permanently blocked; target lane becomes blocked during inference; stale B followed by a fresh B; request timeout; no pending request; repeated B after an existing maneuver. An always-retained reserve may itself prevent forward creep, so include a version with the same release logic and constant reserve: otherwise apparent dynamic-method gains merely compare a released margin to an unreleased one.

Measure crash, success, completion time, route fraction, stopped duration, minimum gap, interventions, admission/rejection timestamps, speed/gap/heading at admission, and successful lane-change continuation from the stopped state. Add total wall-clock planner/camera latency only for actual model calls. Report paired seed-cluster confidence intervals and per-delay curves, not only pooled averages. Do not tune on held-out outcomes.

Mechanism evidence requires a temporal chain: the original envelope remains positive; the continuation-success indicator crosses from recoverable to nonrecoverable before late intent; reserve prevents that crossing; identical later intent then succeeds. A saved-state rescue with direct steering or another controller localizes whether failure is policy-relative. Disabling safety entirely is diagnostic only, not a competitive baseline.

## What would support or kill the contribution

Support: geometry-dependent reserve adapts across speeds/curvatures/vehicle limits; dominates the locked constant-clearance Pareto frontier; admission coupling prevents failures on stale/blocked-lane cases; result survives another physical controller and fresh seeds. Stronger theory could give a sufficient reach-avoid contract under bounded latency and model error, but would need to connect explicitly to existing backup/reachability methods.

Kill/reframe: a fixed larger clearance plus identical admission release matches it; direct steering resolves every stop without earlier braking; gains occur only under a broken low-speed adapter; or reserve consumes unacceptable benign-scene progress. Then report an implementation interaction and repair, not a new planning algorithm. The failure remains useful evidence that offline A/B/C accuracy and safe stopping are insufficient evaluation criteria for a delayed physical planner.
