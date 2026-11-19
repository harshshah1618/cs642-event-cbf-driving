# SafeMetaDrive held-out study — November 19, 2026

The rule planner with the cloned tracking controller achieved **100% destination success and zero recorded native cost** on all 50 held-out maps, for each of three trained controller replicas. The selected camera planner with PPO tracking and collision protection achieved **92% success and mean native cost 0.333**. It outperformed the adapted PPO-Lag and CPO runs, but the rule controller remained stronger. These results establish a successful system on this test suite; they do not establish an official state-of-the-art result or an advantage from camera reasoning over simulator-state rules.

All **1,900 final drives** are complete: 1,000 CPU episodes and 900 camera episodes. The selected models, source hashes and configurations were frozen at 20:10:35 UTC before final evaluation. No model or controller was changed using final-test outcomes. The [frozen plan](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results/frozen-plan.json), [full numerical results and failures](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results/analysis.json), and [verification record](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results/verification.json) preserve the audit trail.

| Method | Trained replicas | Success | Zero-cost success | Mean native cost | Return |
|---|---:|---:|---:|---:|---:|
| Analytic rule | — | 100.0% | 100.0% | 0.000 | 388.13 |
| Rule + cloned tracker | 3 | 100.0% | 100.0% | 0.000 | 388.00 |
| Rule + PPO tracker | 3 | 100.0% | 99.3% | 0.020 | 387.99 |
| **Camera + PPO + protection, no added delay** | **3** | **92.0%** | **89.3%** | **0.333** | **370.77** |
| IDM | — | 88.0% | 76.0% | 1.180 | 355.65 |
| Fixed lane, normal | 3 | 46.0% | 45.3% | 0.020 | 285.22 |
| Fixed lane, careful | 3 | 46.0% | 44.0% | 0.067 | 285.08 |
| PPO-Lag | 3 | 59.3% | 46.0% | 1.927 | 297.46 |
| CPO | 3 | 0.7% | 0.7% | 21.933 | 22.05 |

Each replica covers the same 50 maps, seeds 500–549. “Zero-cost success” means destination success with zero summed benchmark cost, called collision-free success in the raw reports. Native cost is not dollars or a calibrated collision probability. Fixed-lane controllers timed out on 54% of drives. Camera planning raised completion from 46% to 92%, with higher cost than fixed-normal control. PPO refinement did not improve on cloning in the rule controls.

![Success and cost comparison; whiskers show training-seed minimum and maximum](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results/benchmark-comparison.png)

A paired bootstrap resampled the 50 maps 10,000 times after averaging the supplied trained replicas. Camera minus PPO-Lag success was +32.7 percentage points (95% interval +22.7 to +42.7); cost was −1.593 (−2.453 to −0.787). Against fixed-normal control, success improved by 46.0 points (+33.3 to +59.3), while cost increased by 0.313 (0.000 to 0.747). Against rule plus cloning, camera success was 8.0 points lower (−13.3 to −3.3). These intervals condition on the trained models; they do not cover full retraining uncertainty. With only 50 unique maps, zero observed failures still gives a one-sided 95% failure-probability upper bound of 5.8%, assuming independent representative maps. Zero observed cost does not bound future cost severity.

The simulator is MetaDrive 0.4.3 at commit `fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`: three-block procedural roads, accident probability 0.8, traffic density 0.05, horizon 1,000 commands, 10 Hz control. Upstream reward, cost and termination remain unchanged; vehicle/object crashes do not terminate immediately. Training used maps 0–99 and closed-loop development used 100–149. Full settings are in the [protocol](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-protocol.md).

The planner chooses stay/left/right and normal/careful. Normal targets 30 km/h; careful targets 21 km/h with 1.5 times the headway. A shared learned tracker follows the target lane. The fast layer uses exact navigation geometry and simulator object state within a 50 m broad phase, sampled constant-velocity footprint forecasts, stopping caps and fresh lane-gap checks. This is heuristic collision protection, not a formally certified general-road CBF. The camera system as a whole therefore still relies on privileged simulator information.

Each tracker used 24,000 cloned demonstrations followed by 976,896 PPO transitions. The camera is Qwen2.5-VL-3B with language-only rank-16 LoRA and a frozen vision encoder. It sees two front/rear RGB mosaics plus speed and lane metadata. Impossible adjacent lanes are masked. Initial data contained 9,621 labeled images; recovery collection added 10,754 images on the same 100 training maps. Balanced sampling and mirrored training examples improved rare lane-change learning. Image checkpoint selection used maps 80–99, separate from image-training maps 0–79; family selection used closed-loop development maps. Earlier failed models remain recorded.

PPO-Lag and CPO use OmniSafe 0.5 implementations and native low-level observations/actions. Nine baseline runs completed one million transitions each: three PPO-Lag and two three-seed CPO configurations. Development selection compared five checkpoints and deterministic/stochastic inference. PPO-Lag selected one million transitions with stochastic actions. CPO selected the 600,000-transition checkpoint from the default-critic-rate configuration, also stochastic. CPO test costs varied sharply across seeds: 0.14, 64.72 and 0.94. This weak, unstable reference is not an exhaustive tuned CPO benchmark or an exact reproduction of published results. Teacher supervision, custom tracker training, external VLM pretraining and observation privileges prevent equal-information or equal-total-training comparisons.

The predeclared delay and protection ablations produced:

| Added response delay | Success | Zero-cost success | Mean native cost |
|---|---:|---:|---:|
| 0 s | 92.0% | 89.3% | 0.333 |
| 0.5 s | 94.0% | 94.0% | 0.020 |
| 1 s | 95.3% | 94.0% | 0.207 |
| 2 s | 96.0% | 96.0% | 0.440 |
| 4 s | 99.3% | 98.7% | 0.020 |
| 0 s, collision protection disabled | 95.3% | 56.7% | 5.993 |

![Delay and protection ablations](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results/delay-ablation.png)

**The delay results contradict the proposed monotonic-harm story.** Success increased with added delay and cost was nonmonotonic. Only one planner request can be outstanding. Above the nominal 1.5 s query period, delays also reduce request frequency and change frame-history spacing. Median query intervals were 1.5/1.5/1.5/2.3/4.3 s, and applied-response fractions were 90.9/81.7/72.9/59.6/31.6%. This is combined latency/throughput sensitivity, not a causal test of staleness at fixed query frequency. Four seconds remains an ablation, not a newly selected primary model.

Median measured inference was about 0.228 s; response age includes inference and 0.1 s simulation quantization. Evaluation used NVIDIA A10/A10G, BF16, Torch 2.8.0/CUDA 12.8. Inference runs synchronously, then simulation logically holds the prior intent for the measured and injected delay; asynchronous real-time deployment was not demonstrated. The [latency metadata](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results/latency-metadata.json) records hardware and response distributions. Disabling collision protection raised mean cost from 0.333 to 5.993 while increasing completion. That ablation retains route/lane validity and was tested only at zero added delay; no delay-by-protection interaction is established.

All 15 frozen model hashes were verified. Selected weights and adapter files are backed up locally; the [checkpoint manifest](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results/checkpoint-manifest.json) maps their locations, and [checksums](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results/model-checksums.json) record the pinned public Qwen base revision. All report source hashes and map coverage passed verification. Independently recomputed aggregates and paired intervals matched remote results within 1e-10. Repository validation passed 257 tests; lint and type checks passed for the SafeMetaDrive modules.

Reported project compute totals **approximately $33.42 before credits**, including retries. Account-wide billed usage after credits was **$5.70** at the final snapshot; credits cannot be attributed solely to this project, and billing may settle later. The deployed app was stopped at 13:19:56 Pacific on November 19 and verified at zero tasks. Result and model volumes remain retained. See the [cost snapshot](/home/harsh/Projects/cs642-event-cbf-driving/reports/safemetadrive-results/cost-summary.json). No additional training is running.

To recompute the numerical summary from downloaded episode reports without cloud compute:

```bash
.venv/bin/python tools/safemetadrive_analyze.py \
  --groups runs/safemetadrive/heldout-study-v1/groups-local.json \
  --output runs/safemetadrive/heldout-study-v1/analysis-recomputed
```
