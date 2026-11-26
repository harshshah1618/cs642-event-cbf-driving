# Irregular-situation escalation: final test (2026-11-21)

**Workspace access restored November 25:** all 400 original Qwen episode rows
are now downloaded and audited. Under the pinned native cost/termination rules,
the historical run contains six collision episodes, including zero in wet-closed;
the original final-step flag table missed all six. The directly recorded 23/50
STOP violations are verified. See the [saved-run audit and comparison with local
controls](ved-saved-audit-2026-11-25/README.md) for derivation and limitations.
This audit does not run new model inference.

**Initial November 25 audit, before workspace access:** the historical evaluator recorded collision flags only
from the final timestep. MetaDrive clears those flags each step and vehicle/object
collisions do not immediately end these episodes. The crash column below is
therefore an unvalidated final-step count, not an episode collision count. In
particular, the wet-road 9-to-0 comparison cannot yet support a collision-reduction
claim. The original table is preserved as reported. The corrected evaluator marks
new rows `irregular-episode-v2`, accumulates all collision flags, and saves contact
timesteps. Historical rows cannot recover earlier contact types from final flags;
raw timestep traces or a separately labeled rerun are needed. The other numbers
remain author-reported aggregates until the underlying episodes are available.

Fresh test maps 1000-1049, 8 scenes x 50 maps. Row 3 = Harsh's rule + PPO (skills-seed0) + protection; row 5 = the same stack with Qwen3-VL-8B called on PatchCore novelty (high-resolution crop, no speed in the prompt, 3 option orders, sticky stop). Raw episode rows: Modal volume cs642-irregular, closedloop/final-qwen3vl-8b, and runs/irregular/closedloop/final-none (local). Regenerate with tools/irregular_closedloop_report.py. Full write-up: https://claude.ai/artifact/2qcT24NJ8SsRStgfwjpE8q

| model | scene | trigger | n | passed STOP | crashes | calls/episode | s stopped near object | episode s | route % | pass speed m/s | VLM answers |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-VL-8B-Instruct | billboard | novelty | 50 | 0 | 0 | 1.3 | 0.0 | 13.6 | 25 | 6.5 | careful 52, continue 14 |
| none (rule + PPO + protection) | billboard | none | 50 | 0 | 0 | 0.0 | 0.0 | 12.2 | 25 | 7.7 |  |
| Qwen3-VL-8B-Instruct | clean | novelty | 50 | 0 | 0 | 0.1 | 0.0 | 29.8 | 65 | nan | careful 3, continue 1 |
| none (rule + PPO + protection) | clean | none | 50 | 0 | 0 | 0.0 | 0.0 | 29.8 | 63 | nan |  |
| Qwen3-VL-8B-Instruct | flagger | novelty | 50 | 23 | 0 | 5.5 | 10.3 | 24.6 | 25 | 5.9 | stop 169, careful 99, continue 6 |
| none (rule + PPO + protection) | flagger | none | 50 | 50 | 0 | 0.0 | 0.0 | 12.1 | 25 | 7.8 |  |
| Qwen3-VL-8B-Instruct | lane-closed | novelty | 50 | 0 | 0 | 1.7 | 0.1 | 14.2 | 25 | 5.9 | careful 74, left 7, continue 2, stop 1 |
| none (rule + PPO + protection) | lane-closed | none | 50 | 0 | 0 | 0.0 | 0.0 | 12.1 | 25 | 7.7 |  |
| Qwen3-VL-8B-Instruct | pedestrian-lane | novelty | 50 | 0 | 0 | 2.0 | 1.2 | 15.8 | 25 | 5.7 | careful 94, stop 3, continue 3 |
| none (rule + PPO + protection) | pedestrian-lane | none | 50 | 0 | 0 | 0.0 | 1.0 | 14.3 | 25 | 7.4 |  |
| Qwen3-VL-8B-Instruct | pedestrian-shoulder | novelty | 50 | 0 | 0 | 0.8 | 0.0 | 13.2 | 25 | 6.1 | careful 41, continue 1 |
| none (rule + PPO + protection) | pedestrian-shoulder | none | 50 | 0 | 0 | 0.0 | 0.0 | 11.9 | 25 | 7.7 |  |
| Qwen3-VL-8B-Instruct | rain-clean | novelty | 50 | 0 | 0 | 10.0 | 0.0 | 30.0 | 49 | nan | careful 500 |
| none (rule + PPO + protection) | rain-clean | none | 50 | 0 | 0 | 0.0 | 0.0 | 29.8 | 62 | nan |  |
| Qwen3-VL-8B-Instruct | wet-closed | novelty | 50 | 0 | 0 | 22.0 | 47.7 | 60.0 | 17 | nan | careful 749, stop 349 |
| none (rule + PPO + protection) | wet-closed | none | 50 | 0 | 9 | 0.0 | 40.9 | 54.3 | 18 | 4.7 |  |
