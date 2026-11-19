"""Camera planner contract: one token for lane intent and caution."""

import math

LETTERS = "ABCDEF"
INTENTS = [(0, False), (-1, False), (1, False), (0, True), (-1, True), (1, True)]
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
BASE_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def reflect_example(state):
    """Mirror camera-example lane indices and exchange left/right targets."""
    reflected = dict(state)
    for key in ("current_lane", "target_lane"):
        reflected[key] = state["lane_count"]-1-state[key]
    delta, careful = decode(state["label"])
    reflected["label"] = LETTERS[INTENTS.index((-delta, careful))]
    return reflected


def messages(image, state, previous_image=None):
    text = (
        "Plan the next short stretch of driving. You see an earlier camera mosaic followed by the current one. "
        "In each mosaic the top image is the forward camera and the bottom image is the rear camera. "
        f"The earlier mosaic was captured {state.get('history_dt', 0):.1f} seconds before the current one. "
        "Use the image history to judge motion; other vehicles can move. "
        f"Ego speed: {state['speed_mps']:.1f} m/s. "
        f"Current lane index: {state['current_lane']}; committed target lane: {state['target_lane']}; "
        f"number of route lanes: {state['lane_count']}. Lane indices increase to the right. "
        "Continue the navigation route. A lane change moves one lane from the committed target; "
        "continue an unfinished maneuver before requesting another. "
        "Use careful mode for curves, junction conflicts, obstacles or crowded traffic ahead. "
        "A=continue target, normal; B=left, normal; C=right, normal; "
        "D=continue target, careful; E=left, careful; F=right, careful. "
        "Return exactly one letter. A fast safety layer separately checks each command."
    )
    previous_image = image if previous_image is None else previous_image
    return [{"role": "user", "content": [{"type": "image", "image": previous_image},
                                          {"type": "image", "image": image},
                                          {"type": "text", "text": text}]}]


def camera_config(config):
    from metadrive.component.sensors.rgb_camera import RGBCamera

    config = dict(config)
    config.update(image_observation=True, image_on_cuda=False,
                  sensors=dict(rgb_front=(RGBCamera, 448, 252), rgb_rear=(RGBCamera, 448, 126)),
                  vehicle_config=dict(image_source="rgb_front"), show_interface=False,
                  show_logo=False, show_fps=False)
    return config


def capture(env, driver):
    import numpy as np
    from PIL import Image

    front = env.engine.get_sensor("rgb_front").perceive(False, env.agent.origin,
                                                       (0, 0.8, 1.5), (0, -5, 0))
    rear = env.engine.get_sensor("rgb_rear").perceive(False, env.agent.origin,
                                                     (0, -1.5, 1.5), (180, -5, 0))
    rgb = np.concatenate([front[..., :3], rear[..., :3]], axis=0)[..., ::-1].copy()
    image = Image.fromarray(rgb.astype(np.uint8), "RGB")
    lanes = env.agent.navigation.current_ref_lanes
    state = dict(speed_mps=float(env.agent.speed), current_lane=int(env.agent.lane.index[-1]),
                 target_lane=int(driver.lane.index[-1]) if driver.lane in lanes else int(env.agent.lane.index[-1]),
                 lane_count=len(lanes))
    return image, state


def decode(letter):
    if letter not in LETTERS or len(letter) != 1:
        raise ValueError("Invalid planner response")
    return INTENTS[LETTERS.index(letter)]


class PendingIntent:
    """Bind a delayed relative action to its capture-time target lane."""
    def __init__(self, letter, captured_at, delay, lane, ttl=4.5):
        if not all(math.isfinite(x) for x in (captured_at, delay, ttl)) or delay < 0:
            raise ValueError("Invalid intent timing")
        self.delta, self.careful = decode(letter)
        self.captured_at, self.ready_at = captured_at, captured_at+delay
        self.lane, self.ttl = lane, ttl

    def apply(self, now, driver):
        if now < self.ready_at:
            return "pending"
        if now - self.captured_at > self.ttl:
            return "expired"
        return "applied" if driver.apply_intent(self.delta, self.careful, self.lane) else "rejected"


class QwenCameraPlanner:
    def __init__(self, adapter=None):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION,
                                                       min_pixels=56*56, max_pixels=448*392)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, revision=BASE_MODEL_REVISION, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        ids = [self.processor.tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
        if any(len(x) != 1 for x in ids):
            raise RuntimeError("Planner labels must be single tokens")
        self.ids = torch.tensor([x[0] for x in ids], device="cuda")

    def predict(self, image, state, previous_image=None):
        import time

        import torch

        started = time.monotonic()
        previous_image = image if previous_image is None else previous_image
        text = self.processor.apply_chat_template(messages(image, state, previous_image), tokenize=False,
                                                   add_generation_prompt=True)
        x = self.processor(text=[text], images=[previous_image, image], return_tensors="pt").to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**x).logits[0, -1, self.ids].float()
        probabilities = logits.softmax(-1).cpu().tolist()
        predicted = max(range(6), key=lambda i: probabilities[i])
        return LETTERS[predicted], dict(probabilities=probabilities,
                                       inference_seconds=time.monotonic()-started)
