"""Regenerate the paper figures from the public turn and panel data. Requires matplotlib."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

docs = Path(__file__).resolve().parents[1] / "docs"
with (docs / "trajectory.csv").open(encoding="utf8") as stream:
    rows = list(csv.DictReader(stream))
fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True, layout="constrained")
styles = [(0, "#205951", "o", "-"), (0.5, "#a65c18", "s", "--"), (1, "#65498e", "^", ":")]
for col, task in enumerate(("Community radio", "Garden restoration")):
    axes[0, col].set_title(task)
    for ix, metric in enumerate(("hostility", "task_progress")):
        ax = axes[ix, col]
        for dose, color, marker, line in styles:
            series = sorted((r for r in rows if r["task"] == task and float(r["dose"]) == dose), key=lambda r: int(r["turn"]))
            if series:
                ax.plot([int(r["turn"]) for r in series], [float(r[metric]) for r in series],
                        marker=marker, linestyle=line, color=color, label=f"Dose {dose:g}")
        ax.set(xticks=range(1, 7), yticks=range(5), ylim=(-0.15,4.15))
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        if col == 0:
            ax.set_ylabel(metric.replace("_", " ").title())
        if ix == 1:
            ax.set_xlabel("Conversation turn")
axes[0,0].legend(loc="center", fontsize=8, ncols=3, bbox_to_anchor=(0.5,-0.17))
fig.savefig(docs / "trajectories.png", dpi=180)

with (docs / "panel.csv").open(encoding="utf8") as stream:
    fixed = [row for row in csv.DictReader(stream) if row["panel"] == "fixed"]

fig, ax = plt.subplots(figsize=(8.1, 4.45), layout="constrained")
for model, marker, line in [("4b", "o", "-"), ("27b", "s", "--")]:
    series = sorted((row for row in fixed if row["model"] == model), key=lambda row: int(row["dose"]))
    for metric, color in [("hostility", "#a53f23"), ("coherence", "#18635c")]:
        ax.plot([int(row["dose"]) / 100 for row in series], [float(row[metric]) for row in series],
                marker=marker, linestyle=line, linewidth=2, markersize=7,
                markerfacecolor="white" if model == "27b" else color, color=color,
                label=f"{model.upper()} {metric}")
ax.set(xlabel="Steering dose", ylabel="Mean judge rating", xticks=[0, .5, .75, 1],
       yticks=range(5), ylim=(-.13, 4.12))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=.2)
ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=4, frameon=False)
fig.savefig(docs / "dose-response.png", dpi=170, bbox_inches="tight")
