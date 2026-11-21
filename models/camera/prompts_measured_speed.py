"""Same skill text as prompts.py. The state line reports measured ego speed.

The pinned training prompt always writes "Ego speed: 20 m/s", including when the
car is stopped. This module keeps that system text byte-for-byte and changes
only the speed number, and only when the caller supplies ego_speed_mps.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_path = Path(__file__).with_name("prompts.py")
_spec = importlib.util.spec_from_file_location("camera_prompts_pinned", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load pinned prompts from {_path}")
_pinned = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pinned)

SYSTEM = _pinned.SYSTEM
LETTERS = list(_pinned.LETTERS)
MIN_PIXELS, MAX_PIXELS = _pinned.MIN_PIXELS, _pinned.MAX_PIXELS


def state_text(lab):
    if "ego_speed_mps" not in lab:
        return _pinned.state_text(lab)
    speed = float(lab["ego_speed_mps"])
    return (
        f"Our car is in the {lab['ego_lane']} lane. Current skill: {lab['current_skill']}. "
        f"Ego speed: {speed:.1f} m/s."
    )


def messages(image, lab, answer=None):
    img = (
        {"type": "image", "image": image}
        if not isinstance(image, str)
        else {"type": "image_url", "image_url": {"url": image}}
    )
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [img, {"type": "text", "text": state_text(lab)}]},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs
