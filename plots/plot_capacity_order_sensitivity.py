"""Plot capacity-only EPEC order-sensitivity outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = (
    ROOT / "model" / "output" / "order_sensitivity_2026-07-30_30iters" / "capacity_only"
)
STRATEGIC_RESULTS_DIR = (
    ROOT / "model" / "output" / "order_sensitivity_2026-07-30_30iters" / "strategic_operation"
)
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "output"
INVESTORS = ("I1", "I2", "I3", "I4")
ACTIVE_NODES = ("N3", "N8", "N9")
TAIL_WARNING_ORDERS = {
    "I1-I2-I3-I4",
    "I1-I4-I2-I3",
    "I2-I1-I3-I4",
}
COLORS = dict(zip(INVESTORS, plt.get_cmap("tab10").colors[:4], strict=True))
MARKERS = dict(zip(INVESTORS, ("o", "s", "^", "D"), strict=True))


@dataclass(frozen=True)
class Run:
    order: str
    capacities: np.ndarray
    tail_warning: bool
    converged: bool

    @property
    def first(self) -> str:
        return self.order.split("-")[0]

    @property
    def last(self) -> str:
        return self.order.split("-")[-1]


def load_runs(results_dir: Path, *, require_converged: bool = True) -> list[Run]:
    runs: list[Run] = []
    for run_dir in sorted(results_dir.glob("order_*")):
        capacity_path = run_dir / "final_capacities.csv"
        summary_path = run_dir / "summary.json"
        if not capacity_path.exists() or not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if require_converged and not summary.get("converged", False):
            continue

        values: dict[tuple[str, str], float] = {}
        with capacity_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                values[(row["investor"], row["node"])] = float(row["x_power_mw"])

        order = run_dir.name.removeprefix("order_")
        matrix = np.array(
            [[values.get((investor, node), 0.0) for node in ACTIVE_NODES] for investor in INVESTORS]
        )
        runs.append(Run(order, matrix, order in TAIL_WARNING_ORDERS, bool(summary.get("converged", False))))

    if not runs:
        raise ValueError(f"No completed capacity-only runs found below {results_dir}")
    return runs


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)


def plot_distributions(runs: list[Run], output_dir: Path) -> Path:
    fig, axes = plt.subplots(4, 3, figsize=(11, 10), sharey=True)
    rng = np.random.default_rng(20260731)

    for investor_index, investor in enumerate(INVESTORS):
        for node_index, node in enumerate(ACTIVE_NODES):
            ax = axes[investor_index, node_index]
            values = np.array([run.capacities[investor_index, node_index] for run in runs])
            jitter = rng.uniform(-0.16, 0.16, len(runs))

            for x, value, run in zip(jitter, values, runs, strict=True):
                ax.scatter(
                    x,
                    value,
                    s=34,
                    marker="o",
                    facecolor="none" if run.tail_warning else COLORS[investor],
                    edgecolor=COLORS[investor],
                    linewidth=1.3 if run.tail_warning else 0.6,
                    alpha=0.85,
                )

            median = float(np.median(values))
            q1, q3 = np.quantile(values, [0.25, 0.75])
            ax.vlines(0.28, q1, q3, color="0.25", linewidth=2.0)
            ax.hlines(median, 0.20, 0.36, color="0.10", linewidth=2.6)
            ax.set_xlim(-0.22, 0.42)
            ax.set_ylim(-2, 65)
            ax.set_xticks([])
            style_axis(ax)

            if investor_index == 0:
                ax.set_title(node)
            if node_index == 0:
                ax.set_ylabel(f"{investor}\nPower capacity (MW)")

    legend = [
        Line2D([0], [0], marker="o", linestyle="", color="0.25", label="completed run"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            linestyle="",
            color="0.25",
            label="tail-drift warning",
        ),
        Line2D([0], [0], color="0.10", linewidth=2.6, label="median; vertical bar = IQR"),
    ]
    fig.subplots_adjust(left=0.09, right=0.985, top=0.89, bottom=0.075, wspace=0.04, hspace=0.06)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    fig.suptitle("Capacity-only order sensitivity: investor-node distributions", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.018,
        "Each point is one of 22 completed EPECs. N1, N4, N5, and N6 are zero; "
        "N2 and N7 are below 0.011 MW.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    return save_figure(fig, output_dir, "capacity_distributions.png")


def plot_boxplots_by_node(
    runs: list[Run],
    output_dir: Path,
    *,
    title: str,
    filename: str,
    subtitle: str | None = None,
    show_points: bool = False,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.8), sharey=True)
    rng = np.random.default_rng(20260731)
    maximum_value = max(float(run.capacities.max()) for run in runs)
    y_upper = max(67.0, float(np.ceil(maximum_value / 10.0) * 10.0 + 2.0))

    for node_index, (node, ax) in enumerate(zip(ACTIVE_NODES, axes, strict=True)):
        distributions = [
            np.array([run.capacities[investor_index, node_index] for run in runs])
            for investor_index in range(len(INVESTORS))
        ]
        positions = np.arange(len(INVESTORS), dtype=float)
        boxes = ax.boxplot(
            distributions,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.10", "linewidth": 2.0},
            whiskerprops={"color": "0.35", "linewidth": 1.2},
            capprops={"color": "0.35", "linewidth": 1.2},
        )
        for investor_index, box in enumerate(boxes["boxes"]):
            box.set(
                facecolor=COLORS[INVESTORS[investor_index]],
                edgecolor=COLORS[INVESTORS[investor_index]],
                alpha=0.28,
                linewidth=1.4,
            )

        if show_points:
            for investor_index, values in enumerate(distributions):
                investor = INVESTORS[investor_index]
                point_x = positions[investor_index] + rng.uniform(-0.16, 0.16, len(runs))
                for x, value, run in zip(point_x, values, runs, strict=True):
                    ax.scatter(
                        x,
                        value,
                        s=22,
                        marker="o",
                        facecolor=COLORS[investor] if run.converged else "none",
                        edgecolor=COLORS[investor],
                        linewidth=1.0,
                        alpha=0.62,
                        zorder=3,
                    )

        ax.set_xticks(positions, INVESTORS)
        ax.set_xlim(-0.55, len(INVESTORS) - 0.45)
        ax.set_ylim(-2, y_upper)
        ax.set_xlabel("Investor")
        ax.set_title(node)
        style_axis(ax)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Nodal power capacity (MW)")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.12, wspace=0.08)
    fig.suptitle(title, fontsize=14, y=0.98)
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha="center", fontsize=9, color="0.35")
    return save_figure(fig, output_dir, filename)


def plot_system_tradeoff(runs: list[Run], output_dir: Path) -> Path:
    totals = np.array([run.capacities.sum(axis=0) for run in runs])
    n3, n8, n9 = totals.T
    constant_sum = float(np.median(n3 + n9))

    fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    xline = np.linspace(0, 100, 200)
    ax.plot(xline, constant_sum - xline, color="0.25", linewidth=1.4, label=f"N3 + N9 = {constant_sum:.3f} MW")

    for run, x, y in zip(runs, n3, n9, strict=True):
        ax.scatter(
            x,
            y,
            s=70,
            marker=MARKERS[run.last],
            facecolor="none" if run.tail_warning else COLORS[run.first],
            edgecolor=COLORS[run.first],
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set(xlim=(-3, 103), ylim=(-3, 103), xlabel="Total capacity at N3 (MW)", ylabel="Total capacity at N9 (MW)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.88", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("System siting trade-off across 22 solve orders", fontsize=14)
    ax.text(
        0.03,
        0.05,
        f"corr(N3, N9) = {np.corrcoef(n3, n9)[0, 1]:.3f}\nN8 = {np.median(n8):.0f} MW in every run",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
    )

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS[i], label=f"first: {i}") for i in INVESTORS
    ]
    marker_handles = [
        Line2D([0], [0], marker=MARKERS[i], markerfacecolor="none", linestyle="", color="0.25", label=f"last: {i}")
        for i in INVESTORS
    ]
    line_handle = Line2D([0], [0], color="0.25", label=f"constant-sum line")
    ax.legend(handles=color_handles + marker_handles + [line_handle], ncol=3, frameon=False, loc="upper center")
    return save_figure(fig, output_dir, "system_n3_n9_tradeoff.png")


def plot_clustered_heatmap(runs: list[Run], output_dir: Path) -> Path:
    values = np.array([run.capacities.reshape(-1) for run in runs])
    clustered_order = leaves_list(linkage(values, method="ward", metric="euclidean"))
    ordered_values = values[clustered_order]
    ordered_runs = [runs[index] for index in clustered_order]

    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = ax.imshow(ordered_values, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=65)
    labels = [f"{investor}\n{node}" for investor in INVESTORS for node in ACTIVE_NODES]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(
        range(len(ordered_runs)),
        [f"{'† ' if run.tail_warning else ''}{run.order}" for run in ordered_runs],
        fontsize=8.5,
    )
    for tick, run in zip(ax.get_yticklabels(), ordered_runs, strict=True):
        tick.set_color(COLORS[run.first])
    for boundary in (2.5, 5.5, 8.5):
        ax.axvline(boundary, color="white", linewidth=1.2)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Power capacity (MW)")
    ax.set_title("Ward-clustered ownership and siting fingerprints", fontsize=14)
    ax.set_xlabel("Investor-node combination")
    ax.set_ylabel("Solve order (label colour = first investor; † = tail warning)")
    return save_figure(fig, output_dir, "capacity_clustered_heatmap.png")


def centered_pca(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    for index in range(components.shape[0]):
        largest = int(np.argmax(np.abs(components[index])))
        if components[index, largest] < 0:
            components[index] *= -1
    scores = centered @ components[:2].T
    variances = singular_values**2 / (len(values) - 1)
    ratios = variances / variances.sum()
    return scores, components[:2], ratios[:2]


def loading_summary(components: np.ndarray) -> list[str]:
    labels = [f"{investor}:{node}" for investor in INVESTORS for node in ACTIVE_NODES]
    summaries: list[str] = []
    for component in components:
        strongest = np.argsort(np.abs(component))[::-1][:2]
        summaries.append("; ".join(f"{component[index]:+.2f} {labels[index]}" for index in strongest))
    return summaries


def plot_pca(runs: list[Run], output_dir: Path) -> Path:
    values = np.array([run.capacities.reshape(-1) for run in runs])
    scores, components, ratios = centered_pca(values)
    summaries = loading_summary(components)

    fig, ax = plt.subplots(figsize=(9.5, 7), constrained_layout=True)
    for run, (pc1, pc2) in zip(runs, scores, strict=True):
        ax.scatter(
            pc1,
            pc2,
            s=72,
            marker=MARKERS[run.last],
            facecolor="none" if run.tail_warning else COLORS[run.first],
            edgecolor=COLORS[run.first],
            linewidth=1.5,
            alpha=0.9,
        )

    extreme_indices = np.argsort(np.linalg.norm(scores, axis=1))[::-1][:5]
    for index in extreme_indices:
        if scores[index, 1] > 50:
            top_group = [candidate for candidate in extreme_indices if scores[candidate, 1] > 50]
            leftmost = min(top_group, key=lambda candidate: scores[candidate, 0])
            offset = (-8, 8) if index == leftmost else (8, 8)
            alignment = "right" if index == leftmost else "left"
        else:
            offset = (5, 5)
            alignment = "left"
        ax.annotate(
            runs[index].order,
            scores[index],
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha=alignment,
        )

    ax.axhline(0, color="0.82", linewidth=0.8)
    ax.axvline(0, color="0.82", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.91", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}%): {summaries[0]}")
    ax.set_ylabel(f"PC2 ({ratios[1] * 100:.1f}%): {summaries[1]}")
    ax.set_title(f"PCA of 12 investor-node capacities ({ratios.sum() * 100:.1f}% shown)", fontsize=14)

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS[i], label=f"first: {i}") for i in INVESTORS
    ]
    marker_handles = [
        Line2D([0], [0], marker=MARKERS[i], markerfacecolor="none", linestyle="", color="0.25", label=f"last: {i}")
        for i in INVESTORS
    ]
    ax.legend(
        handles=color_handles + marker_handles,
        ncol=2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )
    return save_figure(fig, output_dir, "capacity_pca.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = load_runs(args.results_dir)
    outputs = [
        plot_distributions(runs, args.output_dir),
        plot_boxplots_by_node(
            runs,
            args.output_dir,
            title="Capacity-only order sensitivity: investor boxplots by node",
            filename="capacity_boxplots_by_node.png",
        ),
        plot_system_tradeoff(runs, args.output_dir),
        plot_clustered_heatmap(runs, args.output_dir),
        plot_pca(runs, args.output_dir),
    ]
    if STRATEGIC_RESULTS_DIR.exists():
        strategic_runs = load_runs(STRATEGIC_RESULTS_DIR, require_converged=False)
        converged_count = sum(
            json.loads(
                (STRATEGIC_RESULTS_DIR / f"order_{run.order}" / "summary.json").read_text(encoding="utf-8")
            ).get("converged", False)
            for run in strategic_runs
        )
        outputs.append(
            plot_boxplots_by_node(
                strategic_runs,
                args.output_dir,
                title="Strategic-operation order sensitivity: investor boxplots by node",
                filename="strategic_operation_capacity_boxplots_by_node.png",
                subtitle=(
                    f"{len(strategic_runs)} completed runs: {converged_count} converged; "
                    f"{len(strategic_runs) - converged_count} reached the iteration limit · "
                    "filled = converged; hollow = iteration limit"
                ),
                show_points=True,
            )
        )
    print(f"Loaded {len(runs)} completed capacity-only runs.")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
