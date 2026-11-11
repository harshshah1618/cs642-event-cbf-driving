"""The fine-tuned router model behind a single call: (PNG frame, ego lane, current skill) -> (letter, seconds).

Runs exactly like eval_router.py: vLLM, batch size 1, prefix caching, one constrained letter, bf16.
The timed part is the model call only (image preprocessing + prefill + one decode step).
"""
import base64
import time

from prompts import LETTERS, MAX_PIXELS, MIN_PIXELS, messages


class VLLMPolicy:
    def __init__(self, model, gpu_memory_utilization=0.85):
        # Do not initialise CUDA before LLM() (vLLM would switch to "spawn").
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, limit_mm_per_prompt={"image": 1},
                       mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                       enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization, seed=0)
        tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0, max_tokens=1,
                                 allowed_token_ids=[tok.encode(c, add_special_tokens=False)[0] for c in LETTERS])
        self.model = model

    def __call__(self, png, ego_lane, current_skill):
        uri = "data:image/png;base64," + base64.b64encode(png).decode()
        m = messages(uri, dict(ego_lane=ego_lane, current_skill=current_skill))
        t = time.perf_counter()
        o = self.llm.chat(m, self.sp, use_tqdm=False)[0]
        return o.outputs[0].text.strip()[:1], time.perf_counter() - t
