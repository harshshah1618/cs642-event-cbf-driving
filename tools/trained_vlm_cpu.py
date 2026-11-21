"""Actual exported teammate LoRA inference on CPU; no base-only fallback.

Run under an external process memory/time limit. A slow Torch forward cannot be
safely cancelled by a Python thread. Imports/model loading occur only on init.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import threading
import time
from pathlib import Path

from driving.assets import asset_directory

LETTERS = ("A", "B", "C")
SKILLS = ("CRUISE_RIGHT", "OVERTAKE_LEFT", "RETURN_RIGHT")
MIN_PIXELS, MAX_PIXELS = 784, 1254400
EXPECTED_ADAPTER_SHA256 = "e0b999e98601d58bf8a180d1296d4d075fd40058dec5f55e05fb71ae8ea803ab"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompts(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("trained_vlm_pinned_prompts", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import explicit prompt source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.LETTERS) != LETTERS or (module.MIN_PIXELS, module.MAX_PIXELS) != (
        MIN_PIXELS,
        MAX_PIXELS,
    ):
        raise ValueError("Prompt choices or processor bounds differ from teammate v1")
    return module


def letter_token_ids(tokenizer):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(tokens) != 1 for tokens in ids) or len({tokens[0] for tokens in ids}) != 3:
        raise ValueError("A/B/C must each be a distinct single token")
    return [tokens[0] for tokens in ids]


def prepare_inputs(png, lane, current_skill, *, tokenizer, image_processor, prompts, speed_mps=None):
    """Image-only equivalent of official Qwen2.5VLProcessor's expansion path."""
    from PIL import Image

    if lane not in ("LEFT", "RIGHT") or current_skill not in SKILLS:
        raise ValueError("Invalid lane or current skill")
    if not isinstance(png, bytes) or not png:
        raise ValueError("Encoded image bytes required")
    if speed_mps is not None and not math.isfinite(float(speed_mps)):
        raise ValueError("Invalid ego speed")
    with Image.open(io.BytesIO(png)) as source:
        image = source.convert("RGB")
    lab = {"ego_lane": lane, "current_skill": current_skill}
    if speed_mps is not None:
        lab["ego_speed_mps"] = float(speed_mps)
    rendered_state = prompts.state_text(lab)
    messages = prompts.messages(image, lab)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = image_processor(images=image, return_tensors="pt")
    grid = image_inputs["image_grid_thw"]
    if tuple(grid.shape) != (1, 3):
        raise ValueError("Expected exactly one image grid")
    image_tokens = int(grid[0].prod().item()) // (image_processor.merge_size**2)
    token = "<|image_pad|>"
    if text.count(token) != 1 or image_tokens < 1:
        raise ValueError("Expected exactly one unexpanded image placeholder")
    expanded = text.replace(token, token * image_tokens, 1)
    inputs = dict(tokenizer(expanded, return_tensors="pt", padding=False, add_special_tokens=False))
    inputs.update(dict(image_inputs))
    image_id = tokenizer.convert_tokens_to_ids(token)
    if int((inputs["input_ids"] == image_id).sum().item()) != image_tokens:
        raise ValueError("Image placeholder/token-grid mismatch")
    metadata = {
        "width": image.width,
        "height": image.height,
        "image_tokens": image_tokens,
        "image_grid_thw": grid.tolist(),
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "prompt_hash": hashlib.sha256(text.encode()).hexdigest(),
        "expanded_prompt_hash": hashlib.sha256(expanded.encode()).hexdigest(),
        "image_hash": hashlib.sha256(png).hexdigest(),
        "prompt_mode": (
            "teammate-v1-current-skill-literal20"
            if rendered_state.endswith("Ego speed: 20 m/s.")
            else "teammate-v1-current-skill-measured-speed"
        ),
        "state_text": rendered_state,
        "ego_speed_mps_supplied": None if speed_mps is None else float(speed_mps),
    }
    return inputs, metadata


def constrained_result(logits, allowed_ids):
    import torch

    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != 1:
        raise ValueError("Expected final-token logits [1,1,vocabulary]")
    selected = logits[0, 0, allowed_ids].float()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("Nonfinite A/B/C logits")
    probs = torch.softmax(selected, dim=0)
    index = int(torch.argmax(selected).item())
    return {
        "letter": LETTERS[index],
        "allowed_token_ids": list(allowed_ids),
        "logits": selected.tolist(),
        "probs": probs.tolist(),
        "probability_scope": "conditional on allowed A/B/C, not global confidence",
    }


def verify_adapter(model, adapter_path, *, state_reader=None):
    import torch
    from safetensors.torch import load_file

    if state_reader is None:
        from peft import get_peft_model_state_dict

        def state_reader(wrapped):
            return get_peft_model_state_dict(wrapped, save_embedding_layers=False)
    expected = load_file(str(Path(adapter_path) / "adapter_model.safetensors"), device="cpu")
    actual = state_reader(model)
    if len(expected) != 504 or set(actual) != set(expected):
        raise ValueError(
            f"Adapter key mismatch: expected504/source{len(expected)}/loaded{len(actual)}"
        )
    nonzero = 0
    for key, saved in expected.items():
        loaded = actual[key].detach().cpu()
        if loaded.shape != saved.shape or not torch.equal(loaded, saved.to(loaded.dtype)):
            raise ValueError(f"Adapter tensor mismatch: {key}")
        if not bool(torch.isfinite(loaded).all()) or not bool(torch.count_nonzero(loaded)):
            raise ValueError(f"Nonfinite or all-zero adapter tensor: {key}")
        nonzero += int(torch.count_nonzero(loaded))
    active = getattr(model, "active_adapters", None)
    if active != ["default"]:
        raise ValueError(f"Unexpected active adapter: {active}")
    return {
        "verified_tensor_count": len(expected),
        "nonzero_elements": nonzero,
        "active_adapters": active,
        "all_keys_shapes_values_equal": True,
    }


class TrainedVLMCPU:
    def __init__(self, base_path, adapter_path, prompts_path, threads=4):
        print("stage: importing CPU inference libraries", flush=True)
        import torch
        from peft import PeftModel
        from transformers import (
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2VLImageProcessor,
        )

        base_path, adapter_path, prompts_path = map(Path, (base_path, adapter_path, prompts_path))
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self._lock = threading.Lock()
        adapter_path = asset_directory(adapter_path)
        self.prompts = load_prompts(prompts_path)
        adapter_sha = sha256(adapter_path / "adapter_model.safetensors")
        if adapter_sha != EXPECTED_ADAPTER_SHA256:
            raise ValueError("Exported trained adapter SHA256 mismatch")
        config = json.loads((adapter_path / "adapter_config.json").read_text())
        if (
            config.get("base_model_name_or_path") != "Qwen/Qwen2.5-VL-3B-Instruct"
            or config.get("r") != 16
            or config.get("lora_alpha") != 32
        ):
            raise ValueError("Unexpected teammate adapter configuration")
        print("stage: loading local tokenizer and slow image processor", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path, local_files_only=True, trust_remote_code=False
        )
        template_path = base_path / "chat_template.json"
        if template_path.exists():
            template = json.loads(template_path.read_text())
            self.tokenizer.chat_template = (
                template["chat_template"] if isinstance(template, dict) else template
            )
        elif (base_path / "chat_template.jinja").exists():
            self.tokenizer.chat_template = (base_path / "chat_template.jinja").read_text()
        if not self.tokenizer.chat_template:
            raise ValueError("Exact local model chat template is missing")
        self.allowed_token_ids = letter_token_ids(self.tokenizer)
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            base_path, local_files_only=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        print("stage: loading bf16 base on CPU (no full-model copy/merge)", flush=True)
        started = time.perf_counter()
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        print("stage: applying trained LoRA", flush=True)
        self.model = PeftModel.from_pretrained(
            base,
            adapter_path,
            is_trainable=False,
            local_files_only=True,
            low_cpu_mem_usage=True,
            autocast_adapter_dtype=False,
        )
        self.model.set_adapter("default")
        self.model.eval()
        self.model.requires_grad_(False)
        print("stage: verifying all504 adapter tensors", flush=True)
        self.audit = verify_adapter(self.model, adapter_path)
        if any(p.device.type != "cpu" for p in self.model.parameters()):
            raise ValueError("Model is not entirely on CPU")
        self.audit.update(
            {
                "adapter_sha256": adapter_sha,
                "adapter_config_sha256": sha256(adapter_path / "adapter_config.json"),
                "prompt_source_sha256": sha256(prompts_path),
                "base_config_sha256": sha256(base_path / "config.json"),
                "base_path": str(base_path.resolve()),
                "adapter_path": str(adapter_path.resolve()),
                "base_revision_original_training": None,
                "dtype": "bfloat16",
                "device": "cpu",
                "threads": threads,
                "load_and_adapter_verify_s": time.perf_counter() - started,
            }
        )
        print("stage: trained model ready", flush=True)

    def predict(self, png: bytes, lane: str, current_skill: str, speed_mps=None) -> dict:
        import torch

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent CPU inference refused")
        try:
            start = time.perf_counter()
            inputs, metadata = prepare_inputs(
                png,
                lane,
                current_skill,
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                prompts=self.prompts,
                speed_mps=speed_mps,
            )
            preprocess_s = time.perf_counter() - start
            print("stage: actual trained-model forward", flush=True)
            inference_start = time.perf_counter()
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                result = constrained_result(output.logits, self.allowed_token_ids)
            inference_s = time.perf_counter() - inference_start
            result.update(metadata)
            result.update(
                {
                    "latency_s": time.perf_counter() - start,
                    "preprocessing_s": preprocess_s,
                    "model_forward_s": inference_s,
                    "audit": dict(self.audit),
                    "trained_adapter_applied": True,
                    "fallback_used": False,
                }
            )
            if not math.isfinite(result["latency_s"]):
                raise ValueError("Invalid inference timing")
            return result
        finally:
            self._lock.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "adapter", "prompts", "image", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--lane", choices=("LEFT", "RIGHT"), required=True)
    p.add_argument("--skill", choices=SKILLS, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    policy = TrainedVLMCPU(a.base, a.adapter, a.prompts, threads=a.threads)
    result = policy.predict(a.image.read_bytes(), a.lane, a.skill)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
