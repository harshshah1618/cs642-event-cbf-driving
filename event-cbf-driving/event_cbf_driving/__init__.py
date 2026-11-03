"""Event-triggered skill router + PPO skills + inflated CBF shield."""

__version__ = "0.1.0"

SKILLS = ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
