"""Round-2 VLM router experiments. One model per process (GPU memory is freed on exit).

usage: python vlm_exp.py --model Qwen/Qwen2.5-VL-3B-Instruct --frames /content/frames2 --out results_3b.json

For each view (topdown, camera) found under --frames/<view>/{test,shots}:
  direct      zero-shot: image -> one letter A/B/C
  fewshot     5 labelled example images (1 per class) in the fixed prefix -> one letter. The prefix is cached.
  perceive    two yes/no questions about the scene (Q1 car ahead in RIGHT lane? Q2 other car in LEFT lane?)
              -> a fixed rule maps the answers to a skill. Q2 reuses Q1's cached image.
"""
import argparse
import base64
import io
import json
import math
import os
import random
import statistics as st
import time

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-model-len", type=int, default=8192)
args = ap.parse_args()

# Do not touch CUDA before LLM(): if CUDA is already initialised, vLLM starts its engine with "spawn", which
# re-imports this script and crashes. Read the GPU's compute capability from nvidia-smi instead.
import subprocess  # noqa: E402
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

random.seed(0)
CC = float(subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).decode().split()[0])
DTYPE = "bfloat16" if CC >= 8 else "half"
kw = dict(model=args.model, dtype=DTYPE, max_model_len=args.max_model_len, limit_mm_per_prompt={"image": 6},
          enable_prefix_caching=True, gpu_memory_utilization=0.88, seed=0)
if "Qwen2.5-VL" in args.model:
    kw["mm_processor_kwargs"] = {"min_pixels": 28 * 28, "max_pixels": 1600 * 28 * 28}
t0 = time.time()
llm = LLM(**kw)
LOAD_S = time.time() - t0
tok = llm.get_tokenizer()


def first_id(s):
    return tok.encode(s, add_special_tokens=False)[0]


LETTERS = ["A", "B", "C"]
SP_LETTER = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=[first_id(c) for c in LETTERS], logprobs=5)
SP_YESNO = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=[first_id("Yes"), first_id("No")], logprobs=5)

SCENE = {
    "topdown": ("You see a bird's-eye view of the road. It shows our side of a highway: the yellow line on the left "
                "is the road centre, then the LEFT lane, a dashed white divider, then the RIGHT lane, then the road "
                "edge. Our car is GREEN, near the bottom, driving UP the image. Every other car is parked (speed 0)."),
    "camera": ("You see the view from a forward-facing camera on our car. Our side of the road has two lanes: a LEFT "
               "lane and a RIGHT lane. Every other car you see is parked (speed 0)."),
}
SKILLS = ("Choose the skill our car should run now:\n"
          "A = CRUISE_RIGHT: stay in / go to the right lane and keep driving. Use this when the right lane ahead is "
          "clear, OR when a car blocks the right lane ahead but the left lane is occupied (we must wait).\n"
          "B = OVERTAKE_LEFT: move to / stay in the left lane. Use this when a car blocks the right lane ahead (or is "
          "still ahead of us on the right) and the left lane is free.\n"
          "C = RETURN_RIGHT: go back to the right lane. Use this when we are in the left lane and the right lane "
          "ahead is clear.\n\nReply with exactly one letter: A, B or C.")
PERCEIVE = "Answer the question about the image with exactly one word: Yes or No."
Q1 = "Question: Is there a car ahead of us in the RIGHT lane?"
Q2 = "Question: Is there another car (not ours) in the LEFT lane?"


def system(view, task):
    return f"You are the perception and planning module of a self-driving car. {SCENE[view]}\n\n" + (
        SKILLS if task == "skill" else PERCEIVE)


def uri(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def user(img, text):
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri(img)}}, {"type": "text", "text": text}]}


def state(lab):
    return f"Our car is in the {lab['ego_lane']} lane. Current skill: {lab['current_skill']}. Ego speed: 20 m/s."


def run(msgs, sp):
    t = time.perf_counter()
    out = llm.chat(msgs, sp, use_tqdm=False)[0]
    ms = (time.perf_counter() - t) * 1000
    o = out.outputs[0]
    conf = None
    if o.logprobs:
        lp = {v.decoded_token.strip(): v.logprob for v in o.logprobs[0].values()}
        z = {k: math.exp(v) for k, v in lp.items()}
        conf = round(z.get(o.text.strip(), 0.0) / (sum(z.values()) or 1.0), 3)
    return o.text.strip(), ms, len(out.prompt_token_ids), conf


def fresh(img):
    """Change one pixel so every test frame misses the image cache, like a new camera frame."""
    img = img.copy()
    img.putpixel((random.randrange(img.width), random.randrange(img.height)), (random.randrange(256),) * 3)
    return img


def rule(ego_lane, right_ahead, left_car):
    if ego_lane == "RIGHT":
        return "B" if (right_ahead and not left_car) else "A"
    return "B" if right_ahead else "C"


results = dict(model=args.model, gpu=torch.cuda.get_device_name(0), load_s=round(LOAD_S, 1), views={})
for view in ("topdown", "camera"):
    vdir = os.path.join(args.frames, view)
    if not os.path.exists(os.path.join(vdir, "test", "labels.json")):
        print(f"[{view}] no frames, skipping", flush=True)
        continue
    test = json.load(open(os.path.join(vdir, "test", "labels.json")))
    shots = json.load(open(os.path.join(vdir, "shots", "labels.json")))
    load = lambda d, f: Image.open(os.path.join(vdir, d, f)).convert("RGB")
    sys_skill, sys_perc = system(view, "skill"), system(view, "perceive")
    # Few-shot prefix: one labelled example per class, identical on every call -> served from the prefix cache.
    seen, fs_prefix = set(), [{"role": "system", "content": sys_skill}]
    for s in shots:
        if s["kind"] in seen:
            continue
        seen.add(s["kind"])
        fs_prefix += [user(load("shots", s["file"]), state(s)), {"role": "assistant", "content": s["label"]}]

    # Warm-up (kernels, CUDA graphs, and caching the fixed prefixes).
    w = load("test", test[0]["file"])
    for _ in range(2):
        run([{"role": "system", "content": sys_skill}, user(fresh(w), state(test[0]))], SP_LETTER)
        run(fs_prefix + [user(fresh(w), state(test[0]))], SP_LETTER)
        run([{"role": "system", "content": sys_perc}, user(fresh(w), Q1)], SP_YESNO)

    rows = []
    for lab in test:
        img = fresh(load("test", lab["file"]))
        r = dict(file=lab["file"], kind=lab["kind"], want=lab["label"], right_ahead=lab["right_ahead"], left_car=lab["left_car"])
        r["direct"], r["direct_ms"], r["direct_tok"], r["direct_conf"] = run(
            [{"role": "system", "content": sys_skill}, user(img, state(lab))], SP_LETTER)
        # Each method gets its own pixel-perturbed copy, so no call reuses another's vision-encoder output.
        r["fewshot"], r["fewshot_ms"], r["fewshot_tok"], r["fewshot_conf"] = run(
            fs_prefix + [user(fresh(img), state(lab))], SP_LETTER)
        img2 = fresh(img)  # new pixels again so Q1 pays for the image; Q2 then reuses Q1's cached image
        a1, r["q1_ms"], r["q1_tok"], r["q1_conf"] = run([{"role": "system", "content": sys_perc}, user(img2, Q1)], SP_YESNO)
        a2, r["q2_ms"], _, r["q2_conf"] = run([{"role": "system", "content": sys_perc}, user(img2, Q2)], SP_YESNO)
        r["q1"], r["q2"] = a1.lower().startswith("y"), a2.lower().startswith("y")
        r["perceive"] = rule(lab["ego_lane"], r["q1"], r["q2"])
        for m in ("direct", "fewshot", "perceive"):
            r[f"{m}_ok"] = r[m][:1] == lab["label"]
        r["q1_ok"], r["q2_ok"] = r["q1"] == lab["right_ahead"], r["q2"] == lab["left_car"]
        rows.append(r)
        print(f"[{view}] {lab['file']:15s} want {lab['label']}  direct {r['direct']} fewshot {r['fewshot']} "
              f"perceive {r['perceive']} (Q1 {int(r['q1'])}/{int(lab['right_ahead'])} Q2 {int(r['q2'])}/{int(lab['left_car'])})  "
              f"ms {r['direct_ms']:.0f}/{r['fewshot_ms']:.0f}/{r['q1_ms']:.0f}+{r['q2_ms']:.0f}", flush=True)

    acc = lambda k: round(sum(r[k] for r in rows) / len(rows), 3)
    med = lambda k: round(st.median(r[k] for r in rows), 1)
    summ = dict(
        n=len(rows),
        direct=dict(acc=acc("direct_ok"), ms=med("direct_ms"), prompt_tok=med("direct_tok")),
        fewshot=dict(acc=acc("fewshot_ok"), ms=med("fewshot_ms"), prompt_tok=med("fewshot_tok")),
        perceive=dict(acc=acc("perceive_ok"), ms=round(st.median(r["q1_ms"] + r["q2_ms"] for r in rows), 1),
                      q1_acc=acc("q1_ok"), q2_acc=acc("q2_ok"), q1_ms=med("q1_ms"), q2_ms_cached_image=med("q2_ms")),
    )
    results["views"][view] = dict(summary=summ, rows=rows)
    print(f"[{view}] SUMMARY {json.dumps(summ)}", flush=True)

json.dump(results, open(args.out, "w"), indent=1)
print("saved", args.out)
