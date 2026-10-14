"""Plot hourly nodal prices of one or more `run.py market` outputs side by side.
Nodes whose price series agree within 1e-6 EUR/MWh share one line.

python scripts/diagnostics/plot_prices.py OUT.png RUN_DIR "Panel title" [RUN_DIR "Panel title" ...]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # reference categorical order


def read_prices(run_dir):
    prices = defaultdict(dict)
    with (Path(run_dir) / "prices.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prices[row["node"]][int(row["hour"])] = float(row["price_eur_per_mwh"])
    return prices


def group_nodes(prices, tolerance=1e-6):
    """[(label, hourly prices)], nodes with identical series merged."""
    groups = []
    for node, series in prices.items():
        for members, reference in groups:
            if all(abs(series[h] - reference[h]) <= tolerance for h in series):
                members.append(node)
                break
        else:
            groups.append(([node], series))
    return [(" = ".join(members), series) for members, series in groups]


out = Path(sys.argv[1])
panels = list(zip(sys.argv[2::2], sys.argv[3::2]))  # (run directory, panel title)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.4), sharey=True, facecolor=SURFACE)
axes = [axes] if len(panels) == 1 else list(axes)
colors = {}
for ax, (run_dir, title) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    for k, (label, series) in enumerate(group_nodes(read_prices(run_dir))):
        color = colors.setdefault(label, SERIES[len(colors) % len(SERIES)])
        hours = sorted(series)
        values = [series[h] for h in hours]
        # Earlier groups on top: where a pocket equals the core price, the core line shows.
        ax.step(hours, values, where="mid", color=color, linewidth=2, label=label, zorder=20 - k,
                path_effects=[pe.Stroke(linewidth=4, foreground=SURFACE), pe.Normal()])
        # Direct label: the top line (drawn first) at its peak, the others at their trough.
        h = (max if k == 0 else min)(hours, key=lambda t: series[t])
        ax.annotate(label, (h, series[h]), xytext=(0, 7 if k == 0 else -14), textcoords="offset points",
                    ha="center", color=INK, fontsize=9)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0.5, 24.5)
    ax.set_xlabel("Hour of day")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Nodal price (EUR/MWh)")
handles = {h.get_label(): h for ax in axes for h in ax.get_lines() if not h.get_label().startswith("_")}
fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(out, dpi=160, facecolor=SURFACE)
print(out)
