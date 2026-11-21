"""Summarise PatchCore scores on the development evaluation frames.

Usage (after `modal volume get cs642-irregular scores/<enc>-v1-eval-dev-v1 <dir>`):
  .venv/bin/python tools/irregular_patchcore_report.py <dir> [<dir> ...] --out <report dir>

Positives: frames where the placed object is visible in the front camera and
5-60 m ahead. Negatives: every frame of the clean episodes on the same maps.
Threshold tau = 99th percentile of clean frame scores (1% of clean frames above).
"""

import argparse
import json
from pathlib import Path

import numpy as np

SCENES = ["pedestrian-shoulder", "pedestrian-lane", "cyclist-shoulder", "cyclist-lane",
          "barrier-lane", "warning-lane"]
NOVEL = SCENES[:4]
BINS = [(5, 15), (15, 30), (30, 45), (45, 60), (60, 80)]
# Fixed categorical order (dataviz reference palette, light mode).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
FRAME_SECONDS = 0.3  # eval frames every 3 control steps of 0.1 s


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def persistent_alarms(scores, tau, k=3, n=5):
    """Count trigger events: k of the last n frames above tau, then a 10-frame cooldown."""
    above, events, cooldown = [], 0, 0
    for s in scores:
        above = (above + [s > tau])[-n:]
        if cooldown:
            cooldown -= 1
        elif sum(above) >= k:
            events, cooldown = events + 1, 10
    return events


def analyse(path):
    data = json.loads((Path(path) / "scores.json").read_text())
    recs = data["records"]
    clean = [r for r in recs if r["scene"] == "clean"]
    neg = np.array([r["score"] for r in clean])
    tau = float(np.percentile(neg, 99))
    out = dict(encoder=data["encoder"], frames=len(recs), clean_frames=len(neg), tau=tau,
               ms_per_frame=data["batched_ms_per_frame"], scenes={})
    for scene in SCENES:
        rows = [r for r in recs if r["scene"] == scene and r["truth"] and r["truth"]["visible"]]
        pos = np.array([r["score"] for r in rows if 5 <= r["truth"]["forward"] <= 60])
        by_bin = {}
        for lo, hi in BINS:
            b = [r["score"] for r in rows if lo <= r["truth"]["forward"] < hi]
            by_bin[f"{lo}-{hi}"] = dict(frames=len(b), detected=float(np.mean(np.array(b) > tau)) if b else None)
        # Farthest distance at which each episode's first persistent alarm fires.
        first = []
        for seed in sorted({r["map_seed"] for r in rows}):
            ep = sorted((r for r in recs if r["scene"] == scene and r["map_seed"] == seed),
                        key=lambda r: r["step"])
            above = []
            hit = None
            for r in ep:
                above = (above + [r["score"] > tau])[-5:]
                if sum(above) >= 3 and r["truth"]["forward"] > 0:
                    hit = r["truth"]["forward"]
                    break
            first.append(hit)
        found = [d for d in first if d is not None]
        out["scenes"][scene] = dict(
            novel=scene in NOVEL, positive_frames=len(pos), auroc=auroc(pos, neg),
            detected_at_tau=float(np.mean(pos > tau)) if len(pos) else None,
            by_distance=by_bin, episodes=len(first), episodes_alarmed=len(found),
            median_first_alarm_m=float(np.median(found)) if found else None)
    # False trigger events per minute on clean episodes (3-of-5 rule, cooldown).
    events, minutes = 0, 0.0
    for seed in sorted({r["map_seed"] for r in clean}):
        ep = sorted((r for r in clean if r["map_seed"] == seed), key=lambda r: r["step"])
        events += persistent_alarms([r["score"] for r in ep], tau)
        minutes += len(ep) * FRAME_SECONDS / 60
    out["clean_false_alarms_per_minute"] = events / minutes if minutes else None
    return out, recs


def plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.2), squeeze=False)
    centres = [(lo + hi) / 2 for lo, hi in BINS]
    for ax, (res, _) in zip(axes[0], results, strict=True):
        for scene, color in zip(SCENES, COLORS, strict=True):
            ys = [res["scenes"][scene]["by_distance"][f"{lo}-{hi}"]["detected"] for lo, hi in BINS]
            ax.plot(centres, [np.nan if y is None else 100 * y for y in ys], color=color, lw=2,
                    marker="o", ms=5, ls="-" if scene in NOVEL else "--", label=scene)
        ax.set_title(f"{res['encoder']}: frames above τ (1% of clean frames)", fontsize=11)
        ax.set_xlabel("Object distance ahead (m)")
        ax.set_ylabel("Frames flagged (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][-1].legend(fontsize=8.5, frameon=False, loc="upper right",
                       title="solid = novel, dashed = familiar", title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "detection-by-distance.png", dpi=150)
    plt.close(fig)


def contact_sheet(score_dir, recs, out_dir, volume_name="cs642-irregular", per_scene=2):
    """Overlay patch maps on a few frames per scene (images read from the Modal volume)."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import modal
    from PIL import Image

    maps = np.load(Path(score_dir) / "maps.npy").astype(np.float32)
    vol = modal.Volume.from_name(volume_name)
    run = json.loads((Path(score_dir) / "scores.json").read_text())["run"]
    chosen = []
    for scene in ["clean"] + SCENES:
        idx = [i for i, r in enumerate(recs) if r["scene"] == scene
               and (scene == "clean" or (r["truth"]["visible"] and 12 <= r["truth"]["forward"] <= 22))]
        chosen += [(scene, i) for i in idx[: per_scene * 7: 7][:per_scene]]
    # Colour only the top patches: below the 99th percentile of all patch scores is left clear.
    vmin, vmax = float(np.percentile(maps, 99)), float(np.percentile(maps, 99.97))
    encoder = json.loads((Path(score_dir) / "scores.json").read_text())["encoder"]
    fig, axes = plt.subplots(len(chosen) // per_scene, per_scene, figsize=(5.2 * per_scene, 1.75 * len(chosen) // per_scene))
    for ax, (scene, i) in zip(axes.flat, chosen, strict=False):
        r = recs[i]
        path = f"frames/{run}/shard-{(r['map_seed'] // 5) * 5:03d}/{r['file']}"
        img = Image.open(io.BytesIO(b"".join(vol.read_file(path)))).crop((0, 0, 448, 252))
        ax.imshow(img)
        ax.imshow(np.ma.masked_less(maps[i], vmin), cmap="Reds", alpha=0.6, vmin=vmin, vmax=vmax,
                  extent=(0, 448, 252, 0), interpolation="nearest")
        dist = "" if scene == "clean" else f", {r['truth']['forward']:.0f} m"
        ax.set_title(f"{scene}{dist}: score {r['score']:.2f}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.suptitle(f"{encoder}: patches above the 99th percentile of patch scores", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"heatmaps-{encoder}.png", dpi=130)
    plt.close(fig)


def fmt(value, pattern="{:.3f}"):
    return "-" if value is None else pattern.format(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", action="store_true", help="also render heat-map overlays")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [analyse(d) for d in args.score_dirs]
    (out_dir / "summary.json").write_text(json.dumps([r for r, _ in results], indent=2))
    plot(results, out_dir)
    if args.sheet:
        for d, (_, recs) in zip(args.score_dirs, results, strict=True):
            contact_sheet(d, recs, out_dir)
    for res, _ in results:
        print(f"\n## {res['encoder']}  tau={res['tau']:.3f}  {res['ms_per_frame']:.1f} ms/frame  "
              f"clean false alarms/min={res['clean_false_alarms_per_minute']:.2f}")
        print("| scene | AUROC | frames flagged (5-60 m) | episodes alarmed | median first alarm (m) |")
        print("|---|---:|---:|---:|---:|")
        for scene in SCENES:
            s = res["scenes"][scene]
            print(f"| {scene} | {fmt(s['auroc'])} | {fmt(s['detected_at_tau'], '{:.0%}')} | "
                  f"{s['episodes_alarmed']}/{s['episodes']} | {fmt(s['median_first_alarm_m'], '{:.0f}')} |")


if __name__ == "__main__":
    main()
