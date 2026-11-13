"""Shared driving constants: skill names and simulation tick length."""

__version__ = "1.0.0"

SKILLS = ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
