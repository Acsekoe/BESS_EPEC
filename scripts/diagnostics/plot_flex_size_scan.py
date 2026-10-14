"""Plot flex_size_scan.py results: pocket price, the battery's own (merchant)
profit, and the RES owner's gain against battery size, one line per flexible
demand scale; plus hourly prices without storage per scale. --per-scale adds one
figure per listed scale: pocket price against battery size and hourly pocket
prices at selected battery sizes.

python scripts/diagnostics/plot_flex_size_scan.py model/output/flex_scan --output model/output/flex_scan --per-scale 0.5 1

The MW labels assume the scan's base size N2 20 / N5 50 MW (the data until the
switch to N2 10 / N5 25 MW on 2026-10-14). With the current data, scan scales
0, 0.2, 0.5, 1, 2, 4 to reproduce the same sizes.
"""
import argparse
import collections
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("scan_dir", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--per-scale", nargs="*", default=[])
args = parser.parse_args()

# Reference palette (dataviz skill): flexible demand size is ordinal, so one blue
# ramp light -> dark; "none" is the neutral reference in muted ink.
SCALES = ["0", "0.1", "0.25", "0.5", "1", "2"]
COLOR = dict(zip(SCALES, ["#898781", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]))
SIZE = {"0": "none", "0.1": "0.1: N2 2, N5 5 MW", "0.25": "0.25: N2 5, N5 12.5 MW", "0.5": "0.5 (chosen): N2 10, N5 25 MW",
        "1": "1: N2 20, N5 50 MW", "2": "2: N2 40, N5 100 MW"}
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SURPLUS_HOURS = {"N5": list(range(11, 17)), "N2": list(range(18, 25)) + list(range(1, 8))}
OWNER = {"N5": "I4", "N2": "I3"}  # 80 % owner of the pocket's RES
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(node, scale):
    price = collections.defaultdict(dict)
    with open(args.scan_dir / f"{node}_s{scale}_prices.csv") as f:
        for r in csv.DictReader(f):
            price[float(r["power_mw"])][(r["node"], int(r["hour"]))] = float(r["price_eur_per_mwh"])
    profit = collections.defaultdict(dict)
    with open(args.scan_dir / f"{node}_s{scale}_profits.csv") as f:
        for r in csv.DictReader(f):
            profit[float(r["power_mw"])][r["investor"]] = float(r["profit_eur_per_day"])
    return price, profit


def style(ax, title, ylabel, xlabel=None):
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.tick_params(length=0)


data = {(node, s): load(node, s) for node in ("N5", "N2") for s in SCALES}
fig, axes = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
for col, (node, duration) in enumerate((("N5", 4.027), ("N2", 6))):
    hours = SURPLUS_HOURS[node]
    for s in SCALES:
        price, profit = data[node, s]
        powers = sorted(price)
        mean_price = [sum(price[p][node, t] for t in hours) / len(hours) for p in powers]
        merchant = [profit[p]["I1"] for p in powers]
        owner = OWNER[node]
        owner_gain = [profit[p][owner] - profit[0.0][owner] + profit[p]["I1"] for p in powers]
        kw = dict(color=COLOR[s], linewidth=1.5, label=SIZE[s], zorder=3 if s == "1" else 2)
        axes[0, col].plot(powers, mean_price, **kw)
        axes[1, col].plot(powers, merchant, **kw)
        axes[2, col].plot(powers, owner_gain, **kw)
    span = f"hours {hours[0]}-{hours[-1]}" if node == "N5" else "hours 18-24 and 1-7"
    xlabel = f"Battery power at {node} [MW] ({duration:g} h duration; no other storage)"
    style(axes[0, col], f"{node}: mean price in the surplus {span}", "EUR/MWh")
    style(axes[1, col], f"{node}: the battery's own profit (merchant I1, after capex)", "EUR/day")
    style(axes[2, col], f"{node}: gain of the 80 % RES owner ({OWNER[node]}) if it holds the battery", "EUR/day",
          xlabel)
    axes[1, col].axhline(0, color=AXIS, linewidth=0.8, zorder=1)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title="Flexible demand size (scale)", loc="outside upper center", ncol=3, frameon=False,
           labelcolor=INK, title_fontsize=9)
fig.suptitle("5-bus system, eps 1e-4: pocket price and profits against battery size, by flexible demand size",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.savefig(args.output / "flex_size_vs_battery.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True, sharey=True)
for ax, node in zip(axes, ("N2", "N5", "N1")):
    for s in SCALES:
        price, _ = data["N5", s]
        hours = range(1, 25)
        ax.plot(hours, [price[0.0][node, t] for t in hours], color=COLOR[s], linewidth=1.5, label=SIZE[s])
    style(ax, f"{node}{' (hub)' if node == 'N1' else ''}", "EUR/MWh" if node == "N2" else "", "Hour")
    ax.set_xticks([1, 6, 12, 18, 24])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Flexible demand size (scale)", loc="outside upper center", ncol=6, frameon=False,
           labelcolor=INK, title_fontsize=9, fontsize=8)
fig.suptitle("Hourly prices without storage, by flexible demand size", color=INK, fontsize=11, x=0.01, ha="left")
fig.savefig(args.output / "flex_size_no_storage_prices.png", dpi=150)
print("wrote", args.output / "flex_size_vs_battery.png", args.output / "flex_size_no_storage_prices.png")

# Per scale: battery size is ordinal too, so the same ramp; 0 MW is the neutral reference.
SELECTED = {"N5": [0.0, 60.0, 100.0, 120.0, 140.0, 160.0], "N2": [0.0, 20.0, 40.0, 60.0, 72.0, 100.0]}
BATTERY_COLOR = list(COLOR.values())
for s in args.per_scale:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for col, (node, duration) in enumerate((("N5", 4.027), ("N2", 6))):
        price, _ = data[node, s]
        powers = sorted(price)
        hours = SURPLUS_HOURS[node]
        axes[0, col].plot(powers, [sum(price[p][node, t] for t in hours) / len(hours) for p in powers],
                          color=COLOR[s], linewidth=1.5)
        for p in SELECTED[node]:
            axes[0, col].plot(p, sum(price[p][node, t] for t in hours) / len(hours), "o", markersize=6,
                              color=BATTERY_COLOR[SELECTED[node].index(p)], markeredgecolor=SURFACE, zorder=3)
        span = f"hours {hours[0]}-{hours[-1]}" if node == "N5" else "hours 18-24 and 1-7"
        style(axes[0, col], f"{node}: mean price in the surplus {span}", "EUR/MWh",
              f"Battery power at {node} [MW] ({duration:g} h duration; no other storage)")
        axes[0, col].set_ylim(-15, 70)
        for k, p in enumerate(SELECTED[node]):
            axes[1, col].plot(range(1, 25), [price[p][node, t] for t in range(1, 25)], color=BATTERY_COLOR[k],
                              linewidth=1.5, label=f"{p:g} MW")
        style(axes[1, col], f"{node}: hourly price at selected battery sizes (dots above)", "EUR/MWh", "Hour")
        axes[1, col].set_xticks([1, 6, 12, 18, 24])
        axes[1, col].set_ylim(-15, 70)
        axes[1, col].legend(title="Battery at " + node, frameon=False, fontsize=8, title_fontsize=8, ncol=2,
                            labelcolor=INK)
    fig.suptitle(f"Flexible demand scale {s} ({SIZE[s].split(': ')[1]}), eps 1e-4: pocket prices against "
                 "battery size", color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(args.output / f"flex_scale{s}_prices.png", dpi=150)
    print("wrote", args.output / f"flex_scale{s}_prices.png")
