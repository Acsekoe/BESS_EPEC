"""Plot nodal demand, served demand, prices, and system generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "model" / "output" / "single_investor_mpec"
DEFAULT_OUTPUT = ROOT / "plots" / "output" / "node_market_overview.png"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], column: str) -> float:
    return float(row[column])


def build_plot(results_dir: Path, output_path: Path) -> None:
    node_rows = read_csv(results_dir / "node_hour_balance_prices.csv")
    gen_rows = read_csv(results_dir / "generator_hour_dispatch_duals.csv")

    nodes = sorted({row["node"] for row in node_rows})
    hours = sorted({int(row["hour"]) for row in node_rows})

    by_node = defaultdict(dict)
    for row in node_rows:
        demand = f(row, "demand_mw")
        shed = max(0.0, min(demand, f(row, "load_shed_mw")))
        hour = int(row["hour"])
        by_node[row["node"]][hour] = {
            "demand": demand,
            "served": demand - shed,
            "price": f(row, "lambda_eur_per_mwh"),
        }

    generation_by_unit = defaultdict(lambda: defaultdict(float))
    for row in gen_rows:
        generation_by_unit[row["generator"]][int(row["hour"])] += max(0.0, f(row, "dispatch_mw"))

    fig, axes = plt.subplots(
        len(nodes) + 1,
        2,
        figsize=(16, 2.2 * len(nodes) + 3.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0] * len(nodes) + [1.25]},
        constrained_layout=True,
    )
    if len(nodes) == 0:
        raise ValueError("No nodes found in node_hour_balance_prices.csv.")

    for row_idx, node in enumerate(nodes):
        demand_ax = axes[row_idx, 0]
        price_ax = axes[row_idx, 1]
        total_demand = [by_node[node][hour]["demand"] for hour in hours]
        served_demand = [by_node[node][hour]["served"] for hour in hours]
        prices = [by_node[node][hour]["price"] for hour in hours]

        demand_ax.bar(hours, total_demand, width=0.82, color="#d7dce2", label="Demand")
        demand_ax.bar(hours, served_demand, width=0.54, color="#2878b5", label="Served demand")
        demand_ax.set_ylabel(f"{node}\nMW")
        demand_ax.grid(axis="y", alpha=0.25)
        if row_idx == 0:
            demand_ax.set_title("Demand and Served Demand")
            demand_ax.legend(loc="upper right", frameon=False)

        price_ax.plot(hours, prices, color="#b33f62", marker="o", linewidth=1.6, markersize=3.0)
        price_ax.set_ylabel("EUR/MWh")
        price_ax.grid(axis="y", alpha=0.25)
        if row_idx == 0:
            price_ax.set_title("Nodal Price")

    grid = axes[-1, 0].get_gridspec()
    axes[-1, 0].remove()
    axes[-1, 1].remove()
    gen_ax = fig.add_subplot(grid[-1, :])
    unit_names = sorted(generation_by_unit)
    gen_series = [[generation_by_unit[unit][hour] for hour in hours] for unit in unit_names]
    gen_ax.stackplot(hours, gen_series, labels=unit_names, alpha=0.85)
    gen_ax.set_title("System-Wide Generation Dispatch")
    gen_ax.set_xlabel("Hour")
    gen_ax.set_ylabel("MW")
    gen_ax.grid(axis="y", alpha=0.25)
    gen_ax.legend(loc="upper left", ncol=min(4, max(1, len(unit_names))), frameon=False)
    gen_ax.set_xlim(min(hours) - 0.6, max(hours) + 0.6)

    summary_path = results_dir / "summary.json"
    subtitle = ""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        subtitle = (
            f"profit={summary.get('profit_eur_per_day', 0.0):,.0f} EUR/day, "
            f"storage={summary.get('total_power_mw', 0.0):,.1f} MW / "
            f"{summary.get('total_energy_mwh', 0.0):,.1f} MWh"
        )
    fig.suptitle(f"Node Market Overview\n{subtitle}".strip(), fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a nodal market overview plot from MPEC CSV exports.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_plot(args.results_dir, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
