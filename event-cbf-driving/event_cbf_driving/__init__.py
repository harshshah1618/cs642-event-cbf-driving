"""Event-triggered skill router + PPO skills + inflated CBF shield.

Layer map:
    config      typed, validated experiment configuration
    env         deterministic scenarios and the environment contract
    skills      frozen PPO policies + checkpoint registry
    router      event-triggered skill selection backends
    shield      latency/uncertainty-inflated safety filter
    stack       3-tier closed-loop controller and baselines
    eval        multi-seed runner, metrics, reporting
    runtime     episode recording and replay
"""

__version__ = "1.0.0"

SKILLS = ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
