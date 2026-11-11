"""LoRA fine-tune of Qwen2.5-VL on MetaDrive router frames: image + state text -> one letter (A/B/C).

Only the language model gets LoRA adapters; the vision encoder stays frozen. The loss is cross-entropy on the
single answer token, so this is effectively a 3-way classifier that keeps the VLM's prompt interface (and its
~1-token decode cost). Validation accuracy uses the logits of A/B/C at the answer position (teacher-forced
prefix = the exact inference prompt), so it equals greedy constrained decoding.

usage: python train_lora.py --data /vol/data/v1 --out /vol/ckpt/run1 [--epochs 2] [--max-steps N] [--wandb 1]
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import torch
from PIL import Image

from prompts import LETTERS, MAX_PIXELS, MIN_PIXELS, messages


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True, help="dir with train/ and val/ (each has labels.json + PNGs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--grad-ckpt", type=int, default=0)
    ap.add_argument("--wandb", type=int, default=1)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def load_split(d):
    return [(os.path.join(d, lab["file"]), lab) for lab in json.load(open(os.path.join(d, "labels.json")))]


def main(argv=None):
    a = parse(argv)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

    proc = AutoProcessor.from_pretrained(a.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    proc.tokenizer.padding_side = "right"
    tok = proc.tokenizer
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    letter_ids_t = torch.tensor(letter_ids)

    train, val = load_split(os.path.join(a.data, "train")), load_split(os.path.join(a.data, "val"))
    print(f"train {len(train)}  val {len(val)}", flush=True)

    def collate(batch):
        texts, imgs = [], []
        for path, lab in batch:
            img = Image.open(path).convert("RGB")
            prompt = proc.apply_chat_template(messages(img, lab), tokenize=False, add_generation_prompt=True)
            texts.append(prompt + lab["label"] + "<|im_end|>")
            imgs.append(img)
        enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
        n = enc["attention_mask"].sum(1)
        enc["answer_pos"] = n - 3  # logits here predict the letter (sequence ends: ..., letter, <|im_end|>)
        enc["target"] = torch.tensor([LETTERS.index(lab["label"]) for _, lab in batch])
        enc["kinds"] = [lab["kind"] for _, lab in batch]
        return enc

    # Sanity check: the token right after answer_pos must be the letter.
    b = collate(train[:2])
    for i in range(2):
        got = b["input_ids"][i, b["answer_pos"][i] + 1].item()
        assert got == letter_ids[b["target"][i]], (tok.decode([got]), train[i][1]["label"])
    print("prompt tokens per sample:", b["attention_mask"].sum(1).tolist(), flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                               attn_implementation="sdpa").cuda()
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                      # language-model projections only; the vision tower (".visual.") has gate/up/down_proj too
                      target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    model = get_peft_model(model, lcfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    dl = torch.utils.data.DataLoader(train, batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=6,
                                     drop_last=True, persistent_workers=True)
    vdl = torch.utils.data.DataLoader(val, batch_size=8, shuffle=False, collate_fn=collate, num_workers=4)
    steps_per_epoch = len(dl) // a.accum
    total = a.max_steps or math.ceil(a.epochs * steps_per_epoch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, min(a.warmup, total // 10 + 1), total)

    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project="cs642-vlm-router", name=a.run_name, config=vars(a) | dict(
            n_train=len(train), n_val=len(val), steps=total, gpu=torch.cuda.get_device_name(0)))

    MODEL_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")

    def forward(batch):
        x = {k: batch[k].cuda(non_blocking=True) for k in MODEL_KEYS}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**x).logits
        rows = torch.arange(logits.shape[0], device=logits.device)
        return logits[rows, batch["answer_pos"].cuda()].float()  # (B, vocab)

    @torch.no_grad()
    def evaluate():
        model.eval()
        ok, per_kind, conf = 0, defaultdict(list), [[0] * 3 for _ in range(3)]
        t = time.time()
        for batch in vdl:
            pred = forward(batch)[:, letter_ids_t.cuda()].argmax(-1).cpu()
            for p, y, k in zip(pred.tolist(), batch["target"].tolist(), batch["kinds"]):
                ok += p == y
                per_kind[k].append(p == y)
                conf[y][p] += 1
        model.train()
        res = {"val/acc": ok / len(val), "val/sec": time.time() - t}
        res.update({f"val_kind/{k}": sum(v) / len(v) for k, v in per_kind.items()})
        return res, conf

    res, conf = evaluate()
    print(f"step 0 (base model) {json.dumps(res)} confusion(rows=true A/B/C) {conf}", flush=True)
    if run:
        run.log(res, step=0)

    model.train()
    step, micro, t0, seen, history = 0, 0, time.time(), 0, [dict(step=0, **res)]
    while step < total:
        for batch in dl:
            logits = forward(batch)
            target_ids = letter_ids_t[batch["target"]].cuda()
            loss = torch.nn.functional.cross_entropy(logits, target_ids) / a.accum
            loss.backward()
            micro += 1
            seen += len(batch["target"])
            if micro % a.accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            with torch.no_grad():
                acc3 = (logits[:, letter_ids_t.cuda()].argmax(-1).cpu() == batch["target"]).float().mean().item()
            log = {"train/loss": loss.item() * a.accum, "train/acc_last_microbatch": acc3, "train/grad_norm": gn.item(),
                   "train/lr": sched.get_last_lr()[0], "train/samples_per_s": seen / (time.time() - t0),
                   "train/epoch": seen / len(train)}
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{total} loss {log['train/loss']:.4f} lr {log['train/lr']:.2e} "
                      f"{log['train/samples_per_s']:.1f} samples/s", flush=True)
            if step % a.eval_every == 0 or step == total:
                res, conf = evaluate()
                log.update(res)
                history.append(dict(step=step, **res))
                print(f"step {step} {json.dumps(res)} confusion {conf}", flush=True)
            if run:
                run.log(log, step=step)
            if step >= total:
                break

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(os.path.join(a.out, "adapter"))
    merged = model.merge_and_unload().to(torch.bfloat16)
    merged.save_pretrained(os.path.join(a.out, "merged"), safe_serialization=True)
    proc.save_pretrained(os.path.join(a.out, "merged"))
    summary = dict(args=vars(a), final=history[-1], history=history, train_minutes=round((time.time() - t0) / 60, 1),
                   gpu=torch.cuda.get_device_name(0))
    json.dump(summary, open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)
    if run:
        run.summary.update({"final_val_acc": history[-1]["val/acc"], "base_val_acc": history[0]["val/acc"]})
        run.finish()
    print("saved", a.out, flush=True)
    return summary


if __name__ == "__main__":
    main()
