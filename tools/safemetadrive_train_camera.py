"""Train a six-intent camera planner using frozen Qwen vision and language LoRA."""

import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def train(data, output, steps=600, seed=0, commit=None):
    import numpy as np
    import torch
    from driving.safemetadrive.camera_planner import BASE_MODEL, LETTERS, messages
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    manifests = []
    for path in sorted(Path(data).glob("shard-*/labels.json")):
        records.extend([(path.parent / row["file"], row) for row in json.loads(path.read_text())])
        manifests.append(json.loads((path.parent / "manifest.json").read_text()))
    covered = sorted(s for m in manifests for s in m["seeds"])
    if covered != list(range(100)):
        raise ValueError(f"Expected exactly 100 training-map shards, got seeds {covered}")
    training = [r for r in records if r[1]["map_seed"] < 80]
    validation = [r for r in records if r[1]["map_seed"] >= 80]
    if steps < 10:
        # A smoke test checks every available class without paying for a full
        # development evaluation before the first optimizer update.
        selected = []
        for letter in LETTERS:
            selected.extend([r for r in validation if r[1]["label"] == letter][:8])
        validation = selected
    by_time = {(row["map_seed"], row["step"]): path for path, row in records}
    for path, row in records:
        previous = by_time.get((row["map_seed"], row["step"]-15), path)
        row["history_file"] = str(previous)
        row["history_dt"] = 1.5 if previous != path else 0.0
    processor = AutoProcessor.from_pretrained(BASE_MODEL, min_pixels=56*56, max_pixels=448*392)
    processor.tokenizer.padding_side = "right"
    ids = [processor.tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
    assert all(len(x) == 1 for x in ids)
    letter_ids = torch.tensor([x[0] for x in ids], device="cuda")

    def collate(batch):
        imgs, texts = [], []
        for path, row in batch:
            previous = Image.open(row["history_file"]).convert("RGB")
            image = Image.open(path).convert("RGB")
            imgs.extend([previous, image])
            texts.append(processor.apply_chat_template(messages(image, row, previous), tokenize=False,
                                                        add_generation_prompt=True))
        encoded = processor(text=texts, images=imgs, padding=True, return_tensors="pt")
        target = torch.tensor([LETTERS.index(row["label"]) for _, row in batch])
        return encoded, target

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    resume_path = output / "optimizer.pt"
    if resume_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, output / "last-adapter", is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.print_trainable_parameters()
    counts = Counter(row["label"] for _, row in training)
    weights = [min(10, math.sqrt(len(training)/max(counts[row["label"]], 1))) for _, row in training]
    loader = torch.utils.data.DataLoader(training, batch_size=4, collate_fn=collate,
        sampler=torch.utils.data.WeightedRandomSampler(weights, num_samples=len(training), replacement=True))
    val_loader = torch.utils.data.DataLoader(validation, batch_size=4, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    meta = dict(base_model=BASE_MODEL, seed=seed, steps=steps, train_maps=list(range(80)),
                validation_maps=list(range(80, 100)), training_examples=len(training),
                validation_examples=len(validation), train_counts=dict(counts),
                source_data_manifests=manifests, constrained_labels=LETTERS,
                input="two front/rear RGB mosaics, 1.5 s history, and declared ego/navigation state",
                base_model_revision=model.config._commit_hash,
                history=[], status="running")

    def logits(encoded):
        encoded = {k: v.cuda() for k, v in encoded.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**encoded).logits
        pos = encoded["attention_mask"].sum(1)-1
        return output[torch.arange(len(pos), device="cuda"), pos][:, letter_ids].float()

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        conf = torch.zeros(6, 6, dtype=torch.int64)
        for encoded, target in val_loader:
            predicted = logits(encoded).argmax(-1).cpu()
            for y, p in zip(target, predicted, strict=True):
                conf[y, p] += 1
        supports = conf.sum(1)
        per_class = [float(conf[i, i]/supports[i]) if supports[i] else None for i in range(6)]
        result = dict(step=step, accuracy=float(conf.diag().sum()/conf.sum()),
                      balanced_accuracy=float(np.mean([x for x in per_class if x is not None])),
                      per_class_recall=dict(zip(LETTERS, per_class, strict=True)), confusion=conf.tolist())
        meta["history"].append(result)
        (output_path / "summary.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(result), flush=True)
        model.train()
        return result["balanced_accuracy"]

    output_path = output
    resume = torch.load(resume_path, map_location="cuda", weights_only=False) if resume_path.exists() else None
    if resume:
        opt.load_state_dict(resume["optimizer"])
        meta = resume["metadata"]
        meta["resumptions"] = meta.get("resumptions", 0)+1
        torch.set_rng_state(resume["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in resume["cuda_rng"]])
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        print(f"Resuming camera training at optimizer step {resume['step']}", flush=True)
    else:
        best = evaluate(0)
        meta["base_balanced_accuracy"] = best
    # Always save a trained adapter, even when base classification is stronger.
    best_trained = meta.get("best_balanced_accuracy", -1.0)
    started = time.monotonic()
    model.train()
    opt.zero_grad(set_to_none=True)
    step = resume["step"] if resume else 0
    micro = step*4
    while step < steps:
        for encoded, target in loader:
            loss = torch.nn.functional.cross_entropy(logits(encoded), target.cuda()) / 4
            loss.backward()
            micro += 1
            if micro % 4:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(json.dumps(dict(step=step, loss=float(loss)*4,
                                      samples_per_second=micro*4/(time.monotonic()-started))), flush=True)
            if step % 50 == 0 or step == steps:
                model.save_pretrained(output / "last-adapter")
                processor.save_pretrained(output / "last-adapter")
                state = dict(step=step, optimizer=opt.state_dict(), metadata=meta,
                             cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                             python_rng=random.getstate(), numpy_rng=np.random.get_state())
                torch.save(state, output / "optimizer.tmp")
                (output / "optimizer.tmp").replace(resume_path)
                if commit:
                    commit()
            if step % 200 == 0 or step == steps:
                score = evaluate(step)
                model.save_pretrained(output / "last-adapter")
                processor.save_pretrained(output / "last-adapter")
                if score > best_trained:
                    best_trained = score
                    model.save_pretrained(output / "best-adapter")
                    processor.save_pretrained(output / "best-adapter")
                    meta["best_step"] = step
                    meta["best_balanced_accuracy"] = best_trained
                if commit:
                    commit()
            if step >= steps:
                break
    meta.update(status="completed", wall_seconds=time.monotonic()-started,
                best_balanced_accuracy=best_trained)
    (output / "summary.json").write_text(json.dumps(meta, indent=2))
    return meta
