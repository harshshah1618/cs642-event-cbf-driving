# Architecture

This document describes the runtime path from a world snapshot to a plant
action, the evaluation pipeline, and the provenance guarantees. The design goal
is narrow: every decision that affects a measured number should be explicit,
typed, logged, and reproducible.

The current physical study is documented in [FINAL_REPORT.md](FINAL_REPORT.md).
In `metadrive-study`, the router's accepted intent determines the shield's
current-to-target corridor, and a changed intent re-arms shield interrupts.
The PPO action and critic disagreement feed the shield; shield intervention
feeds the router. A separate stopping-envelope cap and road-edge guard act
before the final monitor and physical acceleration tracker. Every tick logs
the action chain through to the plant. New guards default off in historical
profiles. The local-model profile uses a bounded worker and wall-clock pacing;
delay-replayed controls also preserve expiration and backend-busy time.

## Layer map

| Layer | Module | Responsibility |
| --- | --- | --- |
| Configuration | `event_cbf_driving/config.py` | Typed dataclasses, YAML loading, dotted overrides, strict validation. Every run field is serialized into the manifest. |
| Environment | `event_cbf_driving/env/` | Canonical scenarios (`two_trap`, `merge`, `dense_highway`), snapshot/step contract, strict registry, detached snapshots. |
| Skills | `event_cbf_driving/skills/` | Frozen dual-critic PPO lane policies, checkpoint I/O, aliases, integrity validation. |
| Router | `event_cbf_driving/router/` | Event-triggered skill selection: scripted rule, OpenAI-compatible transport, local transformers/Ollama, deterministic dry-run, live gating/check/telemetry. |
| Shield | `event_cbf_driving/shield/` | Latency/uncertainty-inflated longitudinal CBF and the final `SafetyMonitor` gate. |
| Control stack | `event_cbf_driving/stack/` | `DrivingStack` (proposed) and `M3BaselineStack` (local style approximations of named baselines), shared result schema and metrics accumulator. |
| Evaluation | `event_cbf_driving/eval/` | Multi-mode × scenario × seed runner, strict aggregation, Markdown report, figures, regression gate. |
| Replay | `event_cbf_driving/runtime.py` | Episode schema validation, atomic recording, deterministic open-loop replay. |
| CLI | `event_cbf_driving/cli.py` | `eval`, `router-check`, `train`, `verify`, `doctor`, `replay`; stable exit codes; JSON output. |

## Control loop (one tick at 50 Hz)

```
snapshot ──► deliver pending router answer (once)
         ──► active skill action [ax, ay] (+ optional crossfade from old skill)
         ──► CBF shield.filter(action, snapshot, sigma, vlm_pending)
         ──► event trigger (sigma and/or shield override)
         ──► optional router submission + pending latency margin
         ──► SafetyMonitor.safe_action(...)
         ──► env.step(action) and metric accumulation
```

The controller never acts on a raw router answer. An answer is delivered only
after its simulated `ready_at` time, and only once. Answers are validated for
skill name, backend, request id, finite timing fields, and allowed skill set;
invalid or expired answers are recorded as rejected and do not switch skills.

## Event trigger state machine

There are two interrupts in the proposed mode:

1. **Uncertainty interrupt.** The active skill's critic dispersion `sigma` is
   compared with `meta.threshold * sigma_threshold_scale`. A crossing fires at
   most once until `sigma` returns below threshold (`sigma_latch`).
2. **Shield interrupt.** The CBF override signal is treated as a
   falling-edge-with-hysteresis event, not a level:

   - the shield must be clear for `shield_clear_ticks` consecutive ticks
     before the trigger re-arms;
   - once armed, the override must persist for `shield_hold_ticks` ticks;
   - at most one call is made per override episode;
   - a suppressed edge (pending call, blend, or call-rate limit) is latched and
     retried when the call slot opens, instead of being silently dropped.

This is deliberately not a level trigger. The pending latency margin itself can
push the shield into override; a level trigger would then call the router again,
which extends the margin, producing a call/margin feedback loop. The hysteresis
makes the event edge-triggered and keeps dense traffic from chattering.

## Shield

For each obstacle ahead in the ego's lateral interval:

```
dx = x_obs - x_ego - standoff
h  = dx - headway_s * (vx_ego - vx_obs) - inflation
ax_safe = ((vx_obs - vx_ego) + class_k * h) / headway_s
ax      = clip(min(ax_rl, ax_safe), ax_min, ax_max)
```

`inflation` combines the calibrated critic term and the pending-call delay
margin:

```
inflation = min(gamma_sigma * sigma, sigma_cap)
          + eta_latency * I(vlm_pending) * vx_ego * vlm_delay_s
```

`lateral_mode="lane"` reproduces the research corridor. `lateral_mode="footprint"`
uses actual vehicle widths plus `lateral_margin`; `lookahead_s` projects the ego
laterally over that horizon. Infeasibility is reported explicitly
(`feasible=False`) rather than saturated and called safe.

## Baseline stacks

The optional MetaDrive path (`ScenarioConfig.simulator: metadrive`) implements
the same plant contract through an acceleration tracker and real Bullet
vehicle dynamics. Its profile records per-tick interactions and adds a fresh
target-lane/rear-gap admission check before accepting a router's skill change.
See [METADRIVE.md](METADRIVE.md) for the complete data path and restricted
straight-road scope.

`stack.router_async: true` runs inference in one bounded background worker;
it requires `stack.realtime: true`. The fast loop polls readiness and continues
using fresh state. Actual inference/retry time determines answer readiness,
with configured latency as a lower bound. Late answers cannot change the skill.
The default inline execution path preserves deterministic delay experiments.

`fixed_rate_shielded` and `always_on_shielded` use the same inflated shield,
skill blending, and configured switch guard as the proposed controller. Their
routing schedules differ. Historical polling modes below remain unshielded.

`M3BaselineStack` implements the published comparison modes on the same result
accumulator:

- `unshielded_skill`: skill only;
- `static_cbf`: barrier with no sigma/delay inflation;
- `fixed_rate_vlm`: timer-driven routing with hard switching;
- `fasionad_wakeup` / `cert_replan`: the M3 imitation baselines.

The proposed stack adds inflation, blending, and the event trigger. Variants
(`always_on_vlm`, `proposed_no_margin`, `proposed_no_blend`) isolate each
mechanism.

## Evaluation pipeline

`run_evaluation` performs:

1. deep-copy and validate the config;
2. load the checkpoint bank once (documenting `RETURN_RIGHT` aliasing);
3. build one router for the whole run;
4. for each scenario, seed, repeat, and mode: run one episode;
5. aggregate per `(scenario, mode)` with sample standard deviation (`ddof=1`)
   and a 95% normal CI (`1.96 * std / sqrt(n)`);
6. run the regression gate and append failures to warnings;
7. write artifacts atomically.

Artifacts (`runs/<run_id>/`):

```
manifest.json    code/config/environment/git/checkpoint provenance
results.json     schema version, aggregates, compact episode records
report.md        human-readable tables and caveats
figures/         optional ablation/rollout plots
episodes/        full traces + action logs when --record-episodes is enabled
```

`--update-docs` additionally writes the repository-level `results.json` and
`RESULTS.md` from the latest run. JSON is strict: NaN and Infinity are rejected
on write and read.

## Live router path

`RouterConfig.allow_live` is the single opt-in gate. `create_router` rejects
`backend: openai`/`local` when it is false (except an explicitly unloaded local
diagnostic object), so the CLI, evaluation runner, and programmatic callers all
use the same policy. `auto` chooses live OpenAI-compatible calls only when live
is allowed and a key is present; otherwise it uses the deterministic dry-run.

Live responses extend the router contract with `live`, `model`, `finish_reason`,
and normalized `usage`. `response_record_fields` flattens those into each call
record; `RolloutTracker`, `result_to_episode_record`, aggregation, reports, and
manifests carry `router_live_calls`, token totals, and model names. Scripted,
dry-run, and fallback responses set `live: false` and do not invent tokens.

`event-cbf router-check` builds the configured backend, sends one synthetic
interrupt, and reports live/fallback status, latency, usage, and health. The
`eval` command uses the same check as a one-request preflight when live is
requested, then optionally (`--require-live`) verifies that the run actually
made live calls without fallback. Canonical results remain offline and do not
depend on this path.

## Determinism and provenance

- Scenario randomness is opt-in (`randomize`); the published scenarios are
  deterministic, so repeated seeds produce identical episodes and zero variance.
  The runner still records `seed` and `repeat` for every episode.
- The run manifest includes package version, git commit/branch/dirty flag,
  Python/platform, dependency versions, config hash, router description/health,
  checkpoint fingerprints, and aliases.
- Checkpoints use a stable state-dict SHA-256. A mismatch is rejected even in
  compatibility mode. New checkpoints carry schema-v2 provenance.

## Extension points

| Need | Extend |
| --- | --- |
| New scenario | Add a class in `env/`, register it in `env/registry.py`, add it to `SCENARIO_NAMES`. |
| New router backend | Implement `RouterBackend.submit()` and register it in `router/factory.py`. |
| New controller variant | Add an `EVAL_MODES` entry and map it in `eval/runner.run_episode`. |
| New safety filter | Keep the `ShieldResult` contract; the monitor and controller depend on it. |
| New metric | Add it to `RolloutTracker`, then to `result_to_episode_record` and aggregation. |

## Related documents

- [`SAFETY.md`](SAFETY.md) — threat model, fail-closed behavior, and limits.
- [`OPERATIONS.md`](OPERATIONS.md) — install, commands, artifacts, exit codes,
  troubleshooting, and release checklist.
- [`RESULTS.md`](RESULTS.md) — generated measured results.
