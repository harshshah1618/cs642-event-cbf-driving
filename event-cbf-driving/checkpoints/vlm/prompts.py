"""Router prompt, shared by training and evaluation so the fine-tuned model sees exactly what it is trained on.

Same text as the round-2 `direct` method (vlm-colab/round2/vlm_exp.py), camera view.
Prompt order is cache-friendly: fixed system text -> image -> short state text. The answer is one letter.
"""
SYSTEM = (
    "You are the perception and planning module of a self-driving car. You see the view from a forward-facing camera "
    "on our car. Our side of the road has two lanes: a LEFT lane and a RIGHT lane. Every other car you see is parked "
    "(speed 0).\n\n"
    "Choose the skill our car should run now:\n"
    "A = CRUISE_RIGHT: stay in / go to the right lane and keep driving. Use this when the right lane ahead is "
    "clear, OR when a car blocks the right lane ahead but the left lane is occupied (we must wait).\n"
    "B = OVERTAKE_LEFT: move to / stay in the left lane. Use this when a car blocks the right lane ahead (or is "
    "still ahead of us on the right) and the left lane is free.\n"
    "C = RETURN_RIGHT: go back to the right lane. Use this when we are in the left lane and the right lane "
    "ahead is clear.\n\nReply with exactly one letter: A, B or C."
)
LETTERS = ["A", "B", "C"]
# Qwen2.5-VL uses one token per 28x28 px; a 640x360 frame -> ~300 image tokens.
MIN_PIXELS, MAX_PIXELS = 28 * 28, 1600 * 28 * 28


def state_text(lab):
    return f"Our car is in the {lab['ego_lane']} lane. Current skill: {lab['current_skill']}. Ego speed: 20 m/s."


def messages(image, lab, answer=None):
    """Chat messages for one router call. `image` is a PIL image (HF) or a data URI string (vLLM)."""
    img = {"type": "image", "image": image} if not isinstance(image, str) else {"type": "image_url", "image_url": {"url": image}}
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [img, {"type": "text", "text": state_text(lab)}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs
