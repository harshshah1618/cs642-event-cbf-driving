"""Summarise an offline zero-shot VLM evaluation (results.json from vlm_eval).

Usage: .venv/bin/python tools/irregular_vlm_report.py <results.json> [...]
"""

import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, ".")
from driving.irregular.vlm import ACCEPTABLE, ACTIONS  # noqa: E402


def summarise(path):
    data = json.load(open(path))
    res = data["results"]
    by = defaultdict(list)
    for r in res:
        by[(r["condition"], r["scene"])].append(r)
    conds = sorted({c for c, _ in by})
    main, alt = ("canonical", "shuffled") if "canonical" in conds else ("crop", "crop-shuffled")
    scenes = [s for s in ACCEPTABLE if (main, s) in by]
    print(f"\n## {data['model']} ({data.get('dtype')}), {data['frames']} frames, "
          f"median {np.median([r['seconds'] for r in res]):.3f} s per call")
    print("| scene | acceptable | n | correct (canonical) | correct (shuffled) | answers (canonical) |")
    print("|---|---|---:|---:|---:|---|")
    overall = defaultdict(list)
    for s in scenes:
        rows = {"canonical": by[(main, s)], "shuffled": by[(alt, s)]}
        acc = {c: np.mean([r["action"] in ACCEPTABLE[s] for r in rows[c]]) for c in rows}
        for c in rows:
            overall[c] += [r["action"] in ACCEPTABLE[s] for r in rows[c]]
        dist = Counter(r["action"] for r in rows["canonical"])
        print(f"| {s} | {'/'.join(sorted(ACCEPTABLE[s]))} | {len(rows['canonical'])} | "
              f"{acc['canonical']:.0%} | {acc['shuffled']:.0%} | "
              + ", ".join(f"{a} {dist[a]}" for a in ACTIONS if dist[a]) + " |")
    print(f"| **all** | | | {np.mean(overall['canonical']):.0%} | {np.mean(overall['shuffled']):.0%} | |")
    # Per-action recall on frames whose acceptable set is a single action.
    print("\nRecall by required action (single-answer scenes, canonical order):")
    for a in ACTIONS:
        rows = [r for s in scenes if ACCEPTABLE[s] == {a} for r in by[(main, s)]]
        if rows:
            print(f"  {a}: {np.mean([r['action'] == a for r in rows]):.0%} of {len(rows)}")
    # Order sensitivity: same frame, canonical vs shuffled.
    pair = defaultdict(dict)
    for r in res:
        if r["condition"] in (main, alt):
            pair[r["file"]]["canonical" if r["condition"] == main else "shuffled"] = r["action"]
    same = [p["canonical"] == p["shuffled"] for p in pair.values() if len(p) == 2]
    print(f"Same answer when only the option order changes: {np.mean(same):.0%} of {len(same)} frames")
    # Order-debiased answer: average each action's probability over both option orders.
    probs = defaultdict(dict)
    scene_of = {}
    for r in res:
        if r["condition"] in (main, alt):
            probs[r["file"]][r["condition"]] = r["probs"]
            scene_of[r["file"]] = r["scene"]
    debiased = defaultdict(list)
    for f, pp in probs.items():
        if len(pp) == 2:
            mean = {a: (pp[main].get(a, 0) + pp[alt].get(a, 0)) / 2 for a in pp[main]}
            debiased[scene_of[f]].append(max(mean, key=mean.get))
    print("Order-debiased (mean of both orders): " + ", ".join(
        f"{s} {np.mean([x in ACCEPTABLE[s] for x in debiased[s]]):.0%}" for s in scenes)
        + f"; all {np.mean([x in ACCEPTABLE[s] for s in scenes for x in debiased[s]]):.0%}")
    blank = Counter(r["action"] for r in res if r["condition"] == "blank")
    real = Counter(r["action"] for r in res if r["condition"] == main)
    print(f"Conditions: {main} / {alt}")
    print("Blank grey image answers: " + ", ".join(f"{a} {blank[a]}" for a in ACTIONS if blank[a]))
    print("Real image answers:       " + ", ".join(f"{a} {real[a]}" for a in ACTIONS if real[a]))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarise(p)
