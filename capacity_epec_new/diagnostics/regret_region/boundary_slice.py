"""Two-dimensional exact-reclear slice across the midday congestion boundary.

Coordinates (both feasible by convexity of each investor's segment; I2 frozen):
  a  in [0, 0.1]  I3 along its incumbent -> selected duration_8h response
                  (the decongesting direction: L46/L78 midday multipliers vanish)
  b  in [0, 1]    I1 along its incumbent -> incumbent-start response
                  (the congesting direction: L46 hour 14 becomes priced)

For every grid point all three profits, the priced congestion set and midday
LMPs are recorded.  Within the slice, each investor's best response to the
other's coordinate is extracted and mutual best responses are reported as
slice-restricted equilibrium candidates.  These are equilibria of the
restricted two-coordinate game only; full-dimensional regret must be checked
separately.

Outputs: output/slice/boundary_slice.csv, output/slice/slice_best_responses.csv
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

import regret_common as rc

OUT = rc.OUTPUT_ROOT / "slice"
NODES = [f"N{k}" for k in range(1, 10)]
A = sorted({round(float(x), 6) for x in np.concatenate([np.linspace(0.0, 0.03, 13), [0.04, 0.05, 0.075, 0.1]])})
B = [round(float(x), 6) for x in np.linspace(0.0, 1.0, 11)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = rc.market_data(100.0)
    config = rc.game_config(data)
    static = rc.static_arrays(data)
    midday = [static["times"].index(t) for t in static["times"] if 11 <= t <= 16]
    power, energy = rc.incumbent()
    i3_0 = rc.investor_vector(power, energy, "I3", NODES)
    i3_1 = rc.selected_record("I3")
    i1_0 = rc.investor_vector(power, energy, "I1", NODES)
    i1_1 = rc.audit_record("I1", "incumbent", "refine")
    tasks = []
    for a in A:
        for b in B:
            p3 = {n: i3_0[0][n] + a * (i3_1["power"][n] - i3_0[0][n]) for n in NODES}
            e3 = {n: i3_0[1][n] + a * (i3_1["energy"][n] - i3_0[1][n]) for n in NODES}
            p1 = {n: i1_0[0][n] + b * (i1_1["power"][n] - i1_0[0][n]) for n in NODES}
            e1 = {n: i1_0[1][n] + b * (i1_1["energy"][n] - i1_0[1][n]) for n in NODES}
            fp, fe = rc.with_candidate(power, energy, "I3", p3, e3)
            fp, fe = rc.with_candidate(fp, fe, "I1", p1, e1)
            rc.check_feasible(data, config, "I3", p3, e3, fp)
            rc.check_feasible(data, config, "I1", p1, e1, fp)
            tasks.append(((a, b), 100.0, fp, fe))
    with rc.make_pool(args.workers) as pool:
        results = rc.evaluate_many(pool, tasks)
    rows = []
    grid = {}
    for (a, b), res in sorted(results.items()):
        sets = rc.active_sets(res, static)
        row = dict(a_I3=a, b_I1=b, status=res["status"],
                   **{f"profit_{i}": res["settle"][i]["profit"] for i in rc.INVESTORS},
                   lines_priced=";".join(sorted(sets["lines_priced"])), n_lines_priced=len(sets["lines_priced"]),
                   L46_hour14_priced="L46-@14" in sets["lines_priced"],
                   midday_mean_lmp_N3=float(np.mean(res["lmp"][NODES.index("N3"), midday])),
                   midday_mean_lmp_N6=float(np.mean(res["lmp"][NODES.index("N6"), midday])),
                   midday_mean_lmp_N8=float(np.mean(res["lmp"][NODES.index("N8"), midday])),
                   pd_gap=res["pd_gap"], verify_max_abs_profit_diff=res.get("verify_max_abs_profit_diff", 0.0))
        rows.append(row)
        grid[a, b] = row
    with (OUT / "boundary_slice.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    br_rows = []
    best_a = {b: max(A, key=lambda a: grid[a, b]["profit_I3"]) for b in B}
    best_b = {a: max(B, key=lambda b: grid[a, b]["profit_I1"]) for a in A}
    for b in B:
        a = best_a[b]
        br_rows.append(dict(kind="I3 best a given b", fixed=b, best=a, profit=grid[a, b]["profit_I3"],
                            gain_vs_a0=grid[a, b]["profit_I3"] - grid[0.0, b]["profit_I3"]))
    for a in A:
        b = best_b[a]
        br_rows.append(dict(kind="I1 best b given a", fixed=a, best=b, profit=grid[a, b]["profit_I1"],
                            gain_vs_b0=grid[a, b]["profit_I1"] - grid[a, 0.0]["profit_I1"]))
    mutual = [(a, b) for a in A for b in B if best_a[b] == a and best_b[a] == b]
    for a, b in mutual:
        br_rows.append(dict(kind="mutual slice best response", fixed="", best=f"a={a}, b={b}",
                            profit=f"I1={grid[a, b]['profit_I1']:.2f}; I3={grid[a, b]['profit_I3']:.2f}"))
    with (OUT / "slice_best_responses.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(k for r in br_rows for k in r)))
        w.writeheader()
        w.writerows(br_rows)
    for r in br_rows:
        print(r)


if __name__ == "__main__":
    main()
