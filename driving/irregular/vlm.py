"""Zero-shot VLM escalation: prompt, action set and constrained one-letter answers.

The VLM sees the forward camera and picks one of five actions. The prompt is
scenario-neutral (it never names pedestrians, signs or construction) and the
option order can be shuffled, so answers can be checked against prompt-order
bias. Answers are read from the logits of the option letters, so every call is
one forward pass with no free-text parsing.
"""

import random

ACTIONS = ("continue", "careful", "stop", "left", "right")
DESCRIPTIONS = {
    "continue": "continue driving normally in the current lane",
    "careful": "slow down and drive carefully, staying in the current lane",
    "stop": "stop before the hazard and wait",
    "left": "change to the lane on the left",
    "right": "change to the lane on the right",
}
LETTERS = "ABCDE"

# Acceptable answers per evaluation scene (frames where the object is visible and
# close enough to matter). Keys are scene tags from frames.py; flagger depends on
# the face shown at capture time.
ACCEPTABLE = {
    "clean": {"continue"},
    "pedestrian-shoulder": {"careful"},
    "cyclist-shoulder": {"careful"},
    "pedestrian-lane": {"stop", "left"},
    "cyclist-lane": {"stop", "left"},
    "barrier-lane": {"stop", "left"},
    "warning-lane": {"stop", "left"},
    "flagger-shoulder:stop": {"stop"},
    "flagger-shoulder:slow": {"careful"},
    "sign-lane_closed-shoulder": {"left", "careful"},
    "sign-billboard-shoulder": {"continue"},
}


def scene_key(row):
    face = (row.get("truth") or {}).get("face")
    return f"{row['scene']}:{face}" if row["scene"].startswith("flagger") else row["scene"]


def valid_actions(state):
    """Lane changes off the edge of the road are not offered (as in Harsh's planner)."""
    lane, count = state.get("current_lane", 0), state.get("lane_count", 1)
    return [a for a in ACTIONS
            if not (a == "left" and lane == 0) and not (a == "right" and lane >= count - 1)]


def option_order(state, seed=None):
    """Valid actions in the order shown to the model; seed=None keeps the canonical order."""
    order = valid_actions(state)
    if seed is not None:
        random.Random(seed).shuffle(order)
    return order


def anomaly_crop(image, patch_map, patch=14, size=(128, 72), hi=None):
    """Enlarge the region around the highest-scoring PatchCore patch (same aspect as the frame).

    With `hi` (the same view rendered at higher resolution) the crop is cut from it,
    so the VLM sees real detail instead of upscaled pixels.
    """
    import numpy as np
    from PIL import Image

    r, c = np.unravel_index(int(np.argmax(patch_map)), patch_map.shape)
    cx, cy = (c + 0.5) * patch, (r + 0.5) * patch
    w, h = size
    x0 = int(min(max(cx - w / 2, 0), image.width - w))
    y0 = int(min(max(cy - h / 2, 0), image.height - h))
    if hi is not None:
        k = hi.width / image.width
        return hi.crop((int(x0 * k), int(y0 * k), int((x0 + w) * k), int((y0 + h) * k))).resize(
            image.size, Image.LANCZOS)
    return image.crop((x0, y0, x0 + w, y0 + h)).resize(image.size, Image.LANCZOS)


def prompt(state, order, crop=False, speed=True):
    lane = state.get("current_lane", 0)
    count = state.get("lane_count", 1)
    side = ("the rightmost lane" if lane == count - 1 else
            "the leftmost lane" if lane == 0 else f"lane {lane + 1} from the left")
    options = "\n".join(f"{LETTERS[i]}. {DESCRIPTIONS[a]}" for i, a in enumerate(order))
    return (
        "You are assisting the driver of a car. The image is the view from the car's "
        "forward-facing camera.\n"
        + ("The second image is an enlarged view of one region of the first image.\n" if crop else "")
        + (f"The car is travelling at {state.get('speed_mps', 0.0) * 3.6:.0f} km/h in {side} "
           f"of a {count}-lane road.\n" if speed else f"The car is in {side} of a {count}-lane road.\n")
        + "Look carefully at the road and its surroundings, including anything at the side "
        "of the road, and choose the best action for the next few seconds.\n"
        f"{options}\n"
        "Answer with the single letter of the best option."
    )


class ZeroShotVLM:
    """Qwen VL model answering with one constrained letter."""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", dtype="float16", max_pixels=448 * 252):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        # Qwen3-VL uses a different image-size interface; our 448x252 frames are within its defaults.
        kwargs = {} if "Qwen3" in model_id else dict(min_pixels=56 * 56, max_pixels=max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa").cuda().eval()
        ids = [self.processor.tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
        if any(len(x) != 1 for x in ids):
            raise RuntimeError("Option letters must be single tokens")
        self.letter_ids = torch.tensor([x[0] for x in ids], device="cuda")

    def answer(self, image, state, order, crop=None, speed=True):
        """Return (action, probabilities by action, seconds). `crop` adds an enlarged view."""
        import time

        import torch

        started = time.monotonic()
        images = [image] if crop is None else [image, crop]
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt(state, order, crop is not None, speed)})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1, self.letter_ids[:len(order)]].float()
        probs = logits.softmax(-1).cpu().tolist()
        if any(p != p for p in probs):  # NaN from fp16 overflow
            raise FloatingPointError("NaN logits; use bfloat16 on a GPU that supports it")
        by_action = {a: probs[i] for i, a in enumerate(order)}
        best = max(by_action, key=by_action.get)
        return best, by_action, time.monotonic() - started
