"""Figures for the regret-region investigation (static PNG + SVG).

Every figure is drawn from the CSV tables in output/, which double as the
table view.  One quantity per axis; identity by fixed categorical slots
(I1/I2/I3 or rho = 100/50/25 always keep the same slot), legends for >= 2
series, recessive hairline grids.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import regret_common as rc  # noqa: E402

OUT = rc.OUTPUT_ROOT / "figures"
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
INV_COLOR = dict(zip(("I1", "I2", "I3"), SLOTS))
RHO_COLOR = {100.0: SLOTS[0], 50.0: SLOTS[1], 25.0: SLOTS[2]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5, "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False,
    "axes.axisbelow": True,
})


def read(path: Path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


PATH_ORDER = ["I1_incumbent_start", "I1_selected_replace_N6_10mw_4h", "I2_incumbent_start",
              "I2_selected_replace_N3_10mw_2h", "I2_bridge_between_responses", "I3_incumbent_start",
              "I3_selected_duration_8h"]
PATH_TITLE = {
    "I1_incumbent_start": "I1: incumbent -> incumbent-start response",
    "I1_selected_replace_N6_10mw_4h": "I1: incumbent -> selected response",
    "I2_incumbent_start": "I2: incumbent -> incumbent-start response",
    "I2_selected_replace_N3_10mw_2h": "I2: incumbent -> selected response",
    "I2_bridge_between_responses": "I2: bridge between its two responses",
    "I3_incumbent_start": "I3: incumbent -> incumbent-start response",
    "I3_selected_duration_8h": "I3: incumbent -> selected duration_8h response",
}


def path_figures():
    samples = read(rc.OUTPUT_ROOT / "paths" / "path_samples.csv")
    trans = read(rc.OUTPUT_ROOT / "paths" / "active_set_transitions.csv") or []
    if not samples:
        return
    by = defaultdict(list)
    for r in samples:
        if r["status"] == "optimal":
            by[(r["path"], float(r["rho"]))].append((f(r["s"]), f(r["gain_vs_incumbent"]), f(r["relative_gain"])))
    changes = defaultdict(list)
    for t in trans:
        if t["lines_priced_entered"] or t["lines_priced_left"]:
            changes[(t["path"], float(t["rho"]))].append(0.5 * (f(t["s_from"]) + f(t["s_to"])))
    paths = [p for p in PATH_ORDER if (p, 100.0) in by]

    # Figure 1: gain along each path at rho = 100, price-forming active-set changes marked
    fig, axes = plt.subplots(len(paths), 2, figsize=(10.5, 2.1 * len(paths)), squeeze=False)
    for k, name in enumerate(paths):
        pts = sorted(by[(name, 100.0)])
        s = np.array([p[0] for p in pts])
        g = np.array([p[1] for p in pts])
        for col, xscale in enumerate(("linear", "log")):
            ax = axes[k, col]
            mask = s > 0 if xscale == "log" else np.ones_like(s, dtype=bool)
            for c in changes[(name, 100.0)]:
                if xscale == "linear" or c > 0:
                    ax.axvline(c, color=MUTED, linewidth=0.8, alpha=0.55, zorder=1)
            ax.plot(s[mask], g[mask], color=SLOTS[0], zorder=3)
            ax.plot(s[mask], g[mask], "o", color=SLOTS[0], markersize=3.2, markeredgecolor=SURFACE,
                    markeredgewidth=0.8, zorder=4)
            ax.axhline(0, color=BASE, linewidth=1.0, zorder=2)
            ax.set_xscale(xscale)
            if xscale == "linear":
                ax.set_title(PATH_TITLE.get(name, name), loc="left")
                ax.annotate(f"{g[-1]:,.0f} EUR/day at s=1", xy=(s[-1], g[-1]), xytext=(-4, 6),
                            textcoords="offset points", ha="right", color=INK2, fontsize=8)
            else:
                ax.set_title("same path, log-scaled s", loc="left", color=INK2)
            ax.set_ylabel("unilateral gain (EUR/day)")
    for ax in axes[-1]:
        ax.set_xlabel("path parameter s (0 = incumbent, 1 = audited response)")
    fig.suptitle("Exact-reclear unilateral gain along audited deviation paths, rho = 100\n"
                 "grey hairlines: a congestion multiplier switches on or off between neighbouring samples",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig1_path_gains_rho100")

    # Figure 2: rho comparison
    fig, axes = plt.subplots(math.ceil(len(paths) / 2), 2, figsize=(10.5, 2.3 * math.ceil(len(paths) / 2)), squeeze=False)
    for k, name in enumerate(paths):
        ax = axes[k // 2, k % 2]
        for rho in (100.0, 50.0, 25.0):
            pts = sorted(by.get((name, rho), []))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=RHO_COLOR[rho], label=f"rho = {rho:g}")
        ax.axhline(0, color=BASE, linewidth=1.0)
        ax.set_title(PATH_TITLE.get(name, name), loc="left")
        ax.set_ylabel("gain vs incumbent (EUR/day)")
        ax.set_xlabel("s")
        if k == 0:
            ax.legend(loc="best")
    for j in range(len(paths), axes.size):
        axes.flat[j].set_visible(False)
    fig.suptitle("Same capacity paths re-cleared under three demand-response penalties", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig2_path_gains_by_rho")


def bracket_figure():
    rows = read(rc.OUTPUT_ROOT / "paths" / "brackets.csv")
    cls = read(rc.OUTPUT_ROOT / "paths" / "bracket_classification.csv") or []
    if not rows:
        return
    label = {(c["path"], float(c["rho"]), c["bracket"]): c["classification"] for c in cls}
    by = defaultdict(list)
    for r in rows:
        by[(r["path"], float(r["rho"]), r["bracket"], r["kind"])].append((f(r["width"]), abs(f(r["dpi"]))))
    samples = read(rc.OUTPUT_ROOT / "paths" / "path_samples.csv") or []
    verify = [f(r.get("verify_max_abs_profit_diff")) for r in samples if float(r["rho"]) == 100.0]
    noise_floor = max([v for v in verify if math.isfinite(v)] or [0.0])
    marker = {"change_resolves_to_zero": "o", "steep_continuous": "o", "unresolved": "s",
              "discontinuity_like": "^", "numerically_unreliable_reclear": "X", "not_traced": "."}
    marker_label = {"o": "change shrinks with width (continuous)", "s": "residual below reclear resolution",
                    "^": "residual above resolution (jump-like)", "X": "reclear verification error too large"}
    paths = [p for p in PATH_ORDER if any(k[0] == p for k in by)]
    fig, axes = plt.subplots(math.ceil(len(paths) / 2), 2, figsize=(10.5, 2.6 * math.ceil(len(paths) / 2) + 0.6), squeeze=False)
    for k, name in enumerate(paths):
        ax = axes[k // 2, k % 2]
        for (p, rho, bid, kind), pts in by.items():
            if p != name or rho != 100.0:
                continue
            w = np.array([x[0] for x in pts])
            d = np.maximum(np.array([x[1] for x in pts]), 1e-9)
            color = SLOTS[0] if kind == "cliff" else SLOTS[1]
            ax.plot(w, d, color=color, linewidth=1.4, alpha=0.9)
            ax.plot(w[-1], d[-1], marker.get(label.get((p, rho, bid), ""), "o"), color=INK2, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=5)
        xs = np.logspace(-10, -1, 10)
        ax.plot(xs, 1e3 * xs / 1e-2, color=MUTED, linewidth=0.9)
        ax.axhline(noise_floor, color=MUTED, linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e5)
        ax.set_title(PATH_TITLE.get(name, name), loc="left")
        ax.set_xlabel("bracket width in s")
        ax.set_ylabel("|profit change| (EUR/day)")
    for j in range(len(paths), axes.size):
        axes.flat[j].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=SLOTS[0], linewidth=2, label="cliff bracket (largest profit change)"),
               Line2D([], [], color=SLOTS[1], linewidth=2, label="regime bracket (congestion set changes)"),
               Line2D([], [], color=MUTED, linewidth=1, label=f"slope-1 reference / reclear resolution {noise_floor:.2f} EUR/day")]
    handles += [Line2D([], [], color=INK2, marker=m, linestyle="none", markersize=6, label=t) for m, t in marker_label.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Shrinking-bracket traces at rho = 100: a continuous transition falls with slope 1, a jump stays flat",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig3_bracket_scaling_rho100")


def mpec_figure():
    rows = read(rc.OUTPUT_ROOT / "mpec_consistency" / "mpec_variants.csv")
    if not rows:
        return
    order = ["run_eps1e-3_tol1e-4", "eps1e-3_tol1e-6", "eps1e-3_tol1e-8", "eps1e-4_tol1e-4", "eps1e-5_tol1e-6",
             "eps1e-6_tol1e-6", "eps1e-6_tol1e-8", "eps1e-8_tol1e-8", "strong_duality_tol1e-4",
             "strong_duality_tol1e-6", "strong_duality_tol1e-8", "chain0_eps1e-3_tol1e-4", "chain1_eps1e-4_tol1e-6",
             "chain2_eps1e-5_tol1e-7", "chain3_eps1e-6_tol1e-8", "chain4_strong_duality_tol1e-8"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    selected_start = {"I1": "replace_N6_10mw_4h", "I2": "replace_N3_10mw_2h", "I3": "duration_8h"}
    for col, inv in enumerate(("I1", "I2", "I3")):
        for row_k, (field, ylabel) in enumerate((("embedded_minus_reclear_profit", "embedded - reclear profit (EUR/day)"),
                                                ("unilateral_gain", "exact-reclear gain (EUR/day)"))):
            ax = axes[row_k, col]
            # Fixed slots: incumbent start = slot 1, audited selected start = slot 2, in every panel.
            for si, start in enumerate(("incumbent", selected_start[inv])):
                xs, ys = [], []
                for r in rows:
                    if r["investor"] != inv or r["start"] != start or float(r["rho"]) != 100.0 or r["variant"] not in order:
                        continue
                    y = f(r.get(field))
                    if math.isfinite(y) and r.get("termination") in ("optimal", "locallyOptimal", "feasible", "maxIterations", "maxTimeLimit"):
                        xs.append(order.index(r["variant"]) + (si - 0.5) * 0.25)
                        ys.append(y)
                ax.plot(xs, ys, "o", color=SLOTS[si], markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                        label=("start: incumbent" if si == 0 else "start: audited selected start"), linestyle="none")
            if row_k == 0:
                ax.axhspan(-10, 10, color=GRID, alpha=0.6, zorder=0)
                ax.set_yscale("symlog", linthresh=1.0)
                ax.set_title(f"{inv}  (selected start: {selected_start[inv]})", loc="left")
            ax.axhline(0, color=BASE, linewidth=1.0)
            ax.set_ylabel(ylabel)
            if col == 0 and row_k == 0:
                ax.legend(loc="upper right", fontsize=7)
            if row_k == 1:
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order, rotation=70, ha="right", fontsize=6.5)
    fig.suptitle("Best-response MPEC under tighter relaxation, tolerance and strong duality (rho = 100)\n"
                 "shaded band: the audit's +/-10 EUR/day consistency tolerance; failed solves omitted",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig4_mpec_consistency")


def regret_figure():
    rows = read(rc.OUTPUT_ROOT / "region" / "candidate_regret_summary.csv")
    if not rows:
        return
    cands = list(dict.fromkeys(r["candidate"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(cands) * 3 + 1.2), sharey=True)
    height = 0.26
    for k, (field, xlabel, scale) in enumerate((("regret_lower_bound", "best found unilateral gain (EUR/day)", "log"),
                                                ("relative_regret_lower_bound", "gain / own current profit", "log"))):
        ax = axes[k]
        for j, inv in enumerate(("I1", "I2", "I3")):
            ys, xs = [], []
            for c_i, c in enumerate(cands):
                r = next((x for x in rows if x["candidate"] == c and x["investor"] == inv), None)
                if r is None:
                    continue
                ys.append(c_i + (j - 1) * height)
                xs.append(max(f(r[field]), 1e-3))
            ax.barh(ys, xs, height=height * 0.85, color=INV_COLOR[inv], label=inv)
            if inv == "I1":
                for c_i, c in enumerate(cands):
                    r = next((x for x in rows if x["candidate"] == c and x["investor"] == "I1"), None)
                    ub = f(r.get("regret_upper_bound" if k == 0 else "relative_regret_upper_bound")) if r else math.nan
                    if math.isfinite(ub):
                        ax.plot([ub, ub], [c_i - 1.45 * height, c_i - 0.55 * height], color=INK, linewidth=2)
        ax.set_xscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_yticks(range(len(cands)))
        ax.set_yticklabels(cands)
        ax.invert_yaxis()
        if k == 0:
            ax.legend(loc="lower right")
    axes[1].axvline(1e-3, color=MUTED, linewidth=0.9)
    axes[1].annotate("run tolerance 0.1 %", xy=(1e-3, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -10),
                     textcoords="offset points", color=INK2, fontsize=7)
    fig.suptitle("Regret at candidate profiles: finite-search lower bounds (bars) and I1's certified upper bound (black tick)\n"
                 "relative panel divides by max(1, |own current profit|); I1 makes a loss in the decongested candidates",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig5_candidate_regret")


def history_figure():
    rows = read(rc.RUN_DIR / "history.csv")
    if not rows:
        return
    sweep = np.array([int(r["sweep"]) for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True)
    for inv in ("I1", "I2", "I3"):
        axes[0].plot(sweep, [f(r[f"regret_{inv}_eur_per_day"]) for r in rows], color=INV_COLOR[inv], label=inv)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("single-start regret\n(EUR/day, log)")
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].plot(sweep, [f(r["max_raw_power_deviation_mw"]) for r in rows], color=SLOTS[0])
    axes[1].axhline(0.5, color=MUTED, linewidth=0.9)
    axes[1].annotate("capacity tolerance 0.5 MW", xy=(sweep[-1], 0.5), xytext=(-2, 4), textcoords="offset points",
                     ha="right", color=INK2, fontsize=7)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("max raw response\nresidual (MW, log)")
    axes[2].plot(sweep, [f(r["total_power_mw"]) for r in rows], color=SLOTS[0])
    axes[2].set_ylabel("total fleet (MW)")
    axes[2].set_xlabel("Jacobi sweep")
    fig.suptitle("init5mw15mwh_jacobi_rho100_d025_s100: small residuals and a flat fleet coexist with large regret",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig6_history_regret_vs_residual")


def flatness_figure():
    rows = read(rc.OUTPUT_ROOT / "region" / "joint_perturbations_final.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    radii = sorted({f(r["radius_mw"]) for r in rows})
    rng = np.random.default_rng(1)
    for k, inv in enumerate(("I1", "I2", "I3")):
        ax = axes[k]
        for i, radius in enumerate(radii):
            vals = [abs(f(r[f"dprofit_{inv}"])) for r in rows if f(r["radius_mw"]) == radius and r["status"] == "optimal"]
            x = i + rng.uniform(-0.12, 0.12, size=len(vals))
            ax.plot(x, np.maximum(vals, 1e-4), "o", color=INV_COLOR[inv], markersize=4.5, markeredgecolor=SURFACE,
                    markeredgewidth=0.8, alpha=0.9)
        ax.set_yscale("log")
        ax.set_xticks(range(len(radii)))
        ax.set_xticklabels([f"{r:g} MW" for r in radii])
        ax.set_title(inv, loc="left")
        ax.set_xlabel("joint perturbation radius (all investors move)")
        if k == 0:
            ax.set_ylabel("|own profit change| (EUR/day, log)")
    fig.suptitle("Payoff sensitivity around the final profile: 16 random feasible joint perturbations per radius",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig7_joint_perturbation_flatness")


def relaxed_sd_figure():
    rows = read(rc.OUTPUT_ROOT / "relaxed_strong_duality" / "relaxed_sd_variants.csv")
    ref = read(rc.OUTPUT_ROOT / "mpec_consistency" / "mpec_variants.csv") or []
    if not rows:
        return
    selected_start = {"I1": "replace_N6_10mw_4h", "I2": "replace_N3_10mw_2h", "I3": "duration_8h"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    deltas = np.logspace(-3.3, 1.3, 20)
    for col, inv in enumerate(("I1", "I2", "I3")):
        ax = axes[col]
        ax.plot(deltas, deltas, color=MUTED, linewidth=0.9)
        ax.annotate("gap = delta", xy=(deltas[-4], deltas[-4]), xytext=(2, -12), textcoords="offset points",
                    color=INK2, fontsize=7)
        for si, start in enumerate(("incumbent", selected_start[inv])):
            pts = sorted((f(r["epsilon"]), abs(f(r["embedded_minus_reclear_profit"])))
                         for r in rows if r["investor"] == inv and r["start"] == start
                         and math.isfinite(f(r.get("embedded_minus_reclear_profit"))))
            ax.plot([x for x, _ in pts], [max(y, 1e-3) for _, y in pts], color=SLOTS[si], linewidth=2,
                    marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=("start: incumbent" if si == 0 else "start: audited selected start"))
        for variant, text, yoff in (("run_eps1e-3_tol1e-4", "relaxed KKT eps 1e-3 (run)", 4),
                                    ("strong_duality_tol1e-6", "strong duality", 4)):
            r = next((x for x in ref if x["investor"] == inv and x["start"] == "incumbent"
                      and x["variant"] == variant and f(x["rho"]) == 100.0), None)
            if r and math.isfinite(f(r.get("embedded_minus_reclear_profit"))):
                y = max(abs(f(r["embedded_minus_reclear_profit"])), 1e-3)
                ax.axhline(y, color=BASE, linewidth=1.0)
                ax.annotate(text, xy=(10 ** -3.25, y), xytext=(0, yoff), textcoords="offset points", color=INK2, fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{inv}", loc="left")
        ax.set_xlabel("allowed primal - dual gap delta (EUR/day)")
        if col == 0:
            ax.set_ylabel("|embedded - exact-reclear profit| (EUR/day)")
            ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Relaxed strong duality at the final profile: the optimiser spends the whole gap allowance on favourable prices\n"
                 "reference lines: incumbent-start relaxed KKT (eps 1e-3) and exact strong duality",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig8_relaxed_strong_duality")


def main():
    path_figures()
    bracket_figure()
    mpec_figure()
    regret_figure()
    history_figure()
    flatness_figure()
    relaxed_sd_figure()


if __name__ == "__main__":
    main()
