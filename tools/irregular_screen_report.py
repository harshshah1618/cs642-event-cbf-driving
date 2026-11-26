"""Compare models from the bigger-VLM screen (stage 'screen' in irregular_modal.py).

First look: the speed-free crop prompt, probabilities summed over two option orders
(as the closed loop votes). Hold: the v3 release question on flagger frames, averaged
over both option orders; release needs P(proceed) >= 0.47 (Harsh's dev threshold for 8B).

Usage: .venv/bin/python tools/irregular_screen_report.py runs/irregular/vlm/screen-*/results.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driving.irregular.vlm import ACCEPTABLE  # noqa: E402

RELEASE = 0.47


def auroc(neg, pos):
    """P(score of a positive > score of a negative), ties count half."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    if not len(neg) or not len(pos):
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def summarise(path):
    data = json.loads(Path(path).read_text())
    res = data["results"]
    first = defaultdict(lambda: defaultdict(float))
    hold = defaultdict(list)
    scene = {}
    for r in res:
        scene[r["file"]] = r["scene"]
        if r["condition"] in ("crop-nospeed", "crop-nospeed-shuffled"):
            for a, p in r["probs"].items():
                first[r["file"]][a] += p
        elif r["condition"] in ("hold", "hold-swapped"):
            hold[r["file"]].append(r["probs"]["proceed"])
    voted = {f: max(p, key=p.get) for f, p in first.items()}
    by_scene = defaultdict(list)
    for f, a in voted.items():
        by_scene[scene[f]].append(a)
    seconds = np.median([r["seconds"] for r in res if r["condition"] == "crop-nospeed"])
    out = dict(model=data["model"].split("/")[-1], seconds=float(seconds))
    out["scenes"] = {s: dict(n=len(a), correct=float(np.mean([x in ACCEPTABLE[s] for x in a])),
                             stop=float(np.mean([x == "stop" for x in a])))
                     for s, a in sorted(by_scene.items())}
    out["all"] = float(np.mean([a in ACCEPTABLE[scene[f]] for f, a in voted.items()]))
    stop_p = {k: [np.mean(v) for f, v in hold.items() if scene[f] == k]
              for k in ("flagger-shoulder:stop", "flagger-shoulder:slow")}
    out["hold"] = dict(
        release_on_stop=float(np.mean(np.array(stop_p["flagger-shoulder:stop"]) >= RELEASE)),
        release_on_slow=float(np.mean(np.array(stop_p["flagger-shoulder:slow"]) >= RELEASE)),
        auroc=auroc(stop_p["flagger-shoulder:stop"], stop_p["flagger-shoulder:slow"]),
        n_stop=len(stop_p["flagger-shoulder:stop"]), n_slow=len(stop_p["flagger-shoulder:slow"]))
    return out


def main(paths):
    rows = [summarise(p) for p in paths]
    scenes = sorted({s for r in rows for s in r["scenes"]})
    print("| scene (acceptable) | " + " | ".join(r["model"] for r in rows) + " |")
    print("|---|" + "---:|" * len(rows))
    for s in scenes:
        cells = []
        for r in rows:
            c = r["scenes"].get(s)
            cells.append(f"{c['correct']:.0%} (stop {c['stop']:.0%}, n={c['n']})" if c else "–")
        print(f"| {s} ({'/'.join(sorted(ACCEPTABLE[s]))}) | " + " | ".join(cells) + " |")
    print("| **all scenes** | " + " | ".join(f"{r['all']:.0%}" for r in rows) + " |")
    print(f"| hold: releases while STOP shows (P(proceed) >= {RELEASE}) | "
          + " | ".join(f"{r['hold']['release_on_stop']:.0%} of {r['hold']['n_stop']}" for r in rows) + " |")
    print(f"| hold: releases once SLOW shows | "
          + " | ".join(f"{r['hold']['release_on_slow']:.0%} of {r['hold']['n_slow']}" for r in rows) + " |")
    print("| hold: STOP-vs-SLOW separation (AUROC) | "
          + " | ".join(f"{r['hold']['auroc']:.2f}" for r in rows) + " |")
    print("| median seconds per call | " + " | ".join(f"{r['seconds']:.2f}" for r in rows) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
