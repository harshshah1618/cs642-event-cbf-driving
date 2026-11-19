# SafeMetaDrive study protocol

This study uses upstream SafeMetaDrive, pinned to commit
`fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`. It is a new benchmark run,
separate from the straight-road scenarios in the existing final report.

Maps: procedural three-block roads; train seeds 0–99, development seeds 100–149,
final test seeds 500–549. The 50-map final test is reserved for frozen candidates.
The original EGPO evaluation also used seeds 500–549, but the simulator version
differs, so comparisons to old paper tables are contextual only.

Use upstream reward, cost and termination without overrides. Accident probability
is 0.8, traffic density 0.05, horizon 1000. Collisions with vehicles and objects
do not end the episode. Report destination success, collision-free success,
episodic cost, return, route completion, timeout and all failed episodes.
Hard braking, jerk, requests and latency are auxiliary system metrics.

Development sequence:

1. Run upstream IDM, upstream pretrained expert, and a sensor-range lane/caution
   rule controller on development maps. The expert is an external pretrained
   reference with unknown training-map overlap; it cannot establish held-out
   generalization or a matched training-budget win.
2. Establish curved-road control and a useful high-level action interface.
3. Train learned controls and camera planning; compare against rules and fixed
   caution, with explicit observation privileges and full data accounting.
4. Rerun PPO-Lag and CPO under the same benchmark configuration with one million
   control-training transitions per seed, preserving their standard low-level
   observations. Disclose the candidate's additional camera teacher rollouts,
   supervised updates, external pretraining and stronger state information;
   these prevent an equal-total-compute or equal-information claim. Compare
   mean and stochastic actions and checkpoints at200k,400k,600k,800k,1M using
   development maps. Select one common epoch and inference mode per algorithm
   across its three seeds: maximize mean destination success, break ties by
   lower cost, higher return, earlier checkpoint, then mean actions.
5. Freeze selection using development maps; evaluate all 50 final maps across
   independent training seeds; report paired intervals, cost and success jointly.
6. Evaluate image-to-application delay, including expired/rejected instructions,
   with matched controller/protection ablations.

All pilot runs are feasibility measurements, not benchmark victories. No
camera-model, learned-skill, CBF-certificate or state-of-the-art claim is earned
by the rule controller. Failed experiments remain in the run records.

Sources: [MetaDrive source](https://github.com/metadriverse/metadrive),
[EGPO evaluation configuration](https://github.com/decisionforce/EGPO/blob/main/egpo_utils/common.py),
[original EGPO PPO-Lag training](https://github.com/decisionforce/EGPO/blob/main/training_script/train_ppo_lag_native.py).
