"""Evaluate a router model with vLLM exactly as it would run in the loop: batch size 1, one call per frame,
constrained one-letter answer, prefix caching on. Reports accuracy (overall, per scene kind, confusion) and latency.

usage: python eval_router.py --model Qwen/Qwen2.5-VL-3B-Instruct --sets r2test=/vol/data/r2test/test,test=/vol/data/v1/test --out res.json
"""
import argparse
import base64
import io
import json
import os
import random
import statistics as st
import time
from collections import defaultdict

from PIL import Image

from prompts import LETTERS, MAX_PIXELS, MIN_PIXELS, messages


def uri(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def fresh(img):
    """Change one pixel so every call misses vLLM's image cache, like a new camera frame."""
    img = img.copy()
    img.putpixel((random.randrange(img.width), random.randrange(img.height)), (random.randrange(256),) * 3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", required=True, help="name=dir,name=dir (each dir has labels.json + PNGs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    # Do not initialise CUDA before LLM(): vLLM would switch to "spawn" and re-import this script.
    from vllm import LLM, SamplingParams

    random.seed(0)
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
              enable_prefix_caching=True, gpu_memory_utilization=0.85, seed=0)
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=[tok.encode(c, add_special_tokens=False)[0]
                                                                        for c in LETTERS])

    import torch
    out = dict(model=a.model, tag=a.tag, gpu=torch.cuda.get_device_name(0), load_s=round(load_s, 1), sets={})
    for spec in a.sets.split(","):
        name, d = spec.split("=", 1)
        labs = json.load(open(os.path.join(d, "labels.json")))
        if a.limit:
            labs = labs[: a.limit]
        load = lambda f: Image.open(os.path.join(d, f)).convert("RGB")  # noqa: E731
        for _ in range(3):  # warm-up: kernels, CUDA graphs, prefix cache of the system prompt
            llm.chat(messages(uri(fresh(load(labs[0]["file"]))), labs[0]), sp, use_tqdm=False)
        rows = []
        for lab in labs:
            m = messages(uri(fresh(load(lab["file"]))), lab)
            t = time.perf_counter()
            o = llm.chat(m, sp, use_tqdm=False)[0]
            ms = (time.perf_counter() - t) * 1000
            pred = o.outputs[0].text.strip()[:1]
            rows.append(dict(file=lab["file"], kind=lab["kind"], want=lab["label"], pred=pred, ok=pred == lab["label"],
                             ms=round(ms, 1), prompt_tok=len(o.prompt_token_ids), obstacles=lab.get("obstacles")))
        per_kind = defaultdict(list)
        conf = {y: {p: 0 for p in LETTERS} for y in LETTERS}
        for r in rows:
            per_kind[r["kind"]].append(r["ok"])
            if r["pred"] in LETTERS:
                conf[r["want"]][r["pred"]] += 1
        ms = sorted(r["ms"] for r in rows)
        summ = dict(n=len(rows), acc=round(sum(r["ok"] for r in rows) / len(rows), 3),
                    per_kind={k: round(sum(v) / len(v), 2) for k, v in per_kind.items()}, confusion=conf,
                    median_ms=round(st.median(ms), 1), p90_ms=ms[int(0.9 * (len(ms) - 1))], max_ms=ms[-1],
                    prompt_tok=st.median(r["prompt_tok"] for r in rows))
        out["sets"][name] = dict(summary=summ, rows=rows)
        print(f"[{a.tag or a.model}] {name}: {json.dumps(summ)}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
