# Frozen-weight MetaDrive calibration

These v2 checkpoints contain the original CRUISE_RIGHT and OVERTAKE_LEFT
weights unchanged. Only critic disagreement metadata is recalibrated for
nominal physical lane holding/acquisition. RETURN_RIGHT remains an explicit
CRUISE_RIGHT alias through the loader.

Regenerate with `make calibrate`. `calibration.json` records fit seeds
101,103,107,109, validation seeds 211,223,227, rollout conditions, sample
counts and original/calibrated exceedance. The threshold is the empirical
99th nominal-exposure percentile, not a collision probability or safety
certificate. Correlated physics ticks are not independent trials.

Tests verify disjoint fit/validation seeds and exact equality of all policy
weights with the original bank. No policy was retrained in MetaDrive.
