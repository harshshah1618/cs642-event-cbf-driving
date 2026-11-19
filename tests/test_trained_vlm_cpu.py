"""CPU-policy contracts without downloading or loading model weights."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Qwen2VLImageProcessor

PATH = Path(__file__).resolve().parents[1] / "tools/trained_vlm_cpu.py"
spec = importlib.util.spec_from_file_location("trained_cpu_tests", PATH)
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


class Tokenizer:
    def encode(self, letter, add_special_tokens=False):
        return [{"A": 1, "B": 2, "C": 3}[letter]]

    def apply_chat_template(self, messages, **kwargs):
        assert messages["lane"] == "RIGHT" and messages["skill"] == "RETURN_RIGHT"
        return "prefix<|image_pad|>suffix"

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[9] + [42] * text.count("<|image_pad|>") + [10]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def convert_tokens_to_ids(self, text):
        return 42


def test_official_slow_processor_and_exact_placeholder_expansion():
    image = Image.new("RGB", (640, 360), (10, 20, 30))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    prompts = SimpleNamespace(
        messages=lambda image, lab: {"lane": lab["ego_lane"], "skill": lab["current_skill"]},
        state_text=lambda lab: f"{lab['ego_lane']} {lab['current_skill']}",
    )
    inputs, metadata = cpu.prepare_inputs(
        stream.getvalue(),
        "RIGHT",
        "RETURN_RIGHT",
        tokenizer=Tokenizer(),
        image_processor=Qwen2VLImageProcessor(min_pixels=784, max_pixels=1254400),
        prompts=prompts,
    )
    assert metadata["width"] == 640 and metadata["height"] == 360
    assert metadata["image_grid_thw"] == [[1, 26, 46]] and metadata["image_tokens"] == 299
    assert inputs["input_ids"].dtype == torch.int64
    assert inputs["pixel_values"].shape == (1196, 1176)
    assert metadata["input_tokens"] == 301 and len(metadata["image_hash"]) == 64


def test_constrained_choice_ignores_larger_disallowed_logit():
    logits = torch.tensor([[[100.0, -3.0, 2.0, 1.0]]])
    result = cpu.constrained_result(logits, cpu.letter_token_ids(Tokenizer()))
    assert result["letter"] == "B" and result["allowed_token_ids"] == [1, 2, 3]
    assert sum(result["probs"]) == pytest.approx(1.0)
    assert result["logits"] == [-3.0, 2.0, 1.0]


def test_invalid_tokenization_and_nonfinite_logits_rejected():
    class BadTokenizer:
        def encode(self, *args, **kwargs):
            return [1, 2]

    with pytest.raises(ValueError, match="single token"):
        cpu.letter_token_ids(BadTokenizer())
    with pytest.raises(ValueError, match="Nonfinite"):
        cpu.constrained_result(torch.tensor([[[0.0, float("nan"), 2.0, 3.0]]]), [1, 2, 3])


@pytest.mark.parametrize("fault", [None, "missing", "value", "zero"])
def test_adapter_all504_tensors_checked(tmp_path, fault):
    state = {
        f"layer{i}.lora_{side}.weight": torch.tensor([1.0], dtype=torch.bfloat16)
        for i in range(252)
        for side in ("A", "B")
    }
    if fault == "zero":
        state[next(iter(state))] = torch.zeros(1, dtype=torch.bfloat16)
    save_file(state, str(tmp_path / "adapter_model.safetensors"))
    loaded = dict(state)
    if fault == "missing":
        loaded.pop(next(iter(loaded)))
    if fault == "value":
        loaded[next(iter(loaded))] = torch.tensor([2.0], dtype=torch.bfloat16)
    model = SimpleNamespace(active_adapters=["default"])
    if fault:
        with pytest.raises(ValueError):
            cpu.verify_adapter(model, tmp_path, state_reader=lambda _: loaded)
    else:
        audit = cpu.verify_adapter(model, tmp_path, state_reader=lambda _: loaded)
        assert audit["verified_tensor_count"] == 504 and audit["all_keys_shapes_values_equal"]
