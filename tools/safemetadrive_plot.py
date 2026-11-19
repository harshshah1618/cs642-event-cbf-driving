"""Render benchmark and latency figures from a complete study analysis."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(analysis, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    groups = analysis["groups"]
    names = {"Camera-delay0":"Camera planner + PPO", "Rule-PPO":"Rule planner + PPO",
             "Rule-BC":"Rule planner + cloning", "Rule":"Rule controller", "PPOLag":"PPO-Lag",
             "CPO":"CPO", "IDM":"IDM", "Fixed-normal":"Fixed normal", "Fixed-careful":"Fixed careful"}
    labels = [key for key in names if key in groups]
    fig,axes = plt.subplots(1,2,figsize=(10,0.47*len(labels)+1.8),sharey=True,layout="constrained")
    for axis,metric,replica_metric,scale,title in [
            (axes[0],"success_rate","per_replica_success",100,"Destination success (%)"),
            (axes[1],"mean_cost","per_replica_cost",1,"Mean native episodic cost")]:
        values = np.array([groups[key][metric]*scale for key in labels])
        low = np.array([min(groups[key][replica_metric])*scale for key in labels])
        high = np.array([max(groups[key][replica_metric])*scale for key in labels])
        colors = ["#226e9d" if key == "Camera-delay0" else "#8c9ca5" for key in labels]
        bars = axis.barh(np.arange(len(labels)),values,color=colors,
                         xerr=np.stack([np.maximum(0,values-low),np.maximum(0,high-values)]),capsize=3)
        axis.bar_label(bars,labels=[f"{value:.1f}" if scale == 100 else f"{value:.3f}" for value in values],padding=4,fontsize=9)
        axis.set_xlabel(title)
        axis.set_axisbelow(True)
        axis.grid(axis="x",alpha=.2)
        axis.spines[["top","right"]].set_visible(False)
        axis.set_xlim(0,110 if scale == 100 else max(float(high.max())*1.25,.1))
    axes[0].set_yticks(np.arange(len(labels)),[names[key] for key in labels])
    axes[0].invert_yaxis()
    fig.suptitle(f"SafeMetaDrive {analysis['split']}: 50 shared maps")
    fig.supxlabel("Whiskers show the range across supplied training seeds; they are not confidence intervals.",fontsize=9)
    fig.savefig(output/"benchmark-comparison.png",dpi=220)
    plt.close(fig)
    delays = [value for value in (0,.5,1,2,4) if f"Camera-delay{value:g}" in groups]
    if len(delays) > 1:
        fig,axes = plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
        for axis,metric,replica_metric,scale,title in [
                (axes[0],"success_rate","per_replica_success",100,"Destination success (%)"),
                (axes[1],"mean_cost","per_replica_cost",1,"Mean native cost")]:
            rows = [groups[f"Camera-delay{delay:g}"] for delay in delays]
            values = np.array([row[metric]*scale for row in rows])
            lo = np.array([min(row[replica_metric])*scale for row in rows])
            hi = np.array([max(row[replica_metric])*scale for row in rows])
            axis.errorbar(delays,values,yerr=np.stack([np.maximum(0,values-lo),np.maximum(0,hi-values)]),
                          marker="o",color="#226e9d",capsize=3,label="Protection enabled")
            if "Camera-unprotected" in groups:
                axis.scatter([0],[groups["Camera-unprotected"][metric]*scale],marker="x",color="#a24632",label="Protection disabled (0 s)",s=65)
            axis.set_xlabel("Injected response delay (s)")
            axis.set_ylabel(title)
            axis.set_xticks(delays)
            axis.set_ylim(bottom=0)
            if scale == 100:
                axis.set_ylim(0,105)
            axis.grid(alpha=.2)
            axis.spines[["top","right"]].set_visible(False)
        axes[0].legend(fontsize=8)
        fig.supxlabel("Measured GPU inference is added to injected delay. Whiskers: range across training seeds.",fontsize=8)
        fig.savefig(output/"delay-ablation.png",dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    render(json.loads(Path(args.analysis).read_text()),args.output)
