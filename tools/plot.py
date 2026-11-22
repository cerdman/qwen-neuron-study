"""Regenerate the paper figure from the public turn data. Requires matplotlib."""
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
