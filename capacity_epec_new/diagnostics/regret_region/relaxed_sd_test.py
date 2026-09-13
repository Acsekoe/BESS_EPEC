"""Relaxed strong duality at the final profile: primal - dual <= delta (EUR/day).

Same investors, starts and exact-reclear decomposition as mpec_consistency.py,
with lower_level = "relaxed-strong-duality" and delta in {10, 1, 0.1, 0.01,
0.001} EUR/day.  The earlier relaxed-KKT (eps 1e-3) and exact strong-duality
rows from output/mpec_consistency/mpec_variants.csv are printed alongside for
comparison; they are not recomputed.

Output: output/relaxed_strong_duality/relaxed_sd_variants.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import regret_common as rc
import mpec_consistency as mc

OUT = rc.OUTPUT_ROOT / "relaxed_strong_duality"
DELTAS = (10.0, 1.0, 0.1, 0.01, 0.001)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for inv in rc.INVESTORS:
        for label in ("incumbent", mc.SELECTED_START[inv]):
            sp, se = mc.start_capacities(inv, label)
            for delta in DELTAS:
                spec = dict(lower_level="relaxed-strong-duality", epsilon=delta, tolerance=args.tolerance)
                tasks.append((100.0, inv, label, f"relaxed_sd_delta{delta:g}_tol{args.tolerance:g}", spec, sp, se))
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(mc._task, t) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            cur = mc.CURRENT_RHO100[row["investor"]]
            if "recleared_profit" in row:
                row["current_profit_at_rho"] = cur
                row["unilateral_gain"] = row["recleared_profit"] - cur
                row["relative_gain"] = row["unilateral_gain"] / max(1.0, abs(cur))
            rows.append(row)
            print(f"[{time.perf_counter()-t0:5.0f}s] {row['investor']} {row['start']:<20} {row['variant']:<30}"
                  f" term={row.get('termination')} emb={f(row.get('embedded_profit')):.2f}"
                  f" rec={f(row.get('recleared_profit')):.2f} gap={f(row.get('embedded_minus_reclear_profit')):+.2f}"
                  f" pd={f(row.get('embedded_primal_dual_gap')):.3g} prod={f(row.get('complementarity_max_product')):.1e}"
                  f" t={f(row.get('solve_seconds')):.0f}s", flush=True)
    rows.sort(key=lambda r: (r["investor"], r["start"], -r["epsilon"]))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "relaxed_sd_variants.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    reference = list(csv.DictReader((rc.OUTPUT_ROOT / "mpec_consistency" / "mpec_variants.csv").open(newline="", encoding="utf-8")))
    keep = ("run_eps1e-3_tol1e-4", "eps1e-4_tol1e-4", "strong_duality_tol1e-6")
    print()
    print(f"{'inv':<3} {'start':<20} {'formulation':<32} {'term':<14} {'embedded':>11} {'reclear':>11} {'gap':>8} {'pd gap':>9} {'gain':>9} {'sec':>5}")
    for inv in rc.INVESTORS:
        for label in ("incumbent", mc.SELECTED_START[inv]):
            for r in reference:
                if r["investor"] == inv and r["start"] == label and f(r["rho"]) == 100.0 and r["variant"] in keep:
                    print(f"{inv:<3} {label:<20} {r['variant']:<32} {r.get('termination','')[:14]:<14} {f(r.get('embedded_profit')):>11.2f}"
                          f" {f(r.get('recleared_profit')):>11.2f} {f(r.get('embedded_minus_reclear_profit')):>8.2f}"
                          f" {f(r.get('embedded_primal_dual_gap')):>9.3g} {f(r.get('unilateral_gain')):>9.1f} {f(r.get('solve_seconds')):>5.0f}")
            for r in rows:
                if r["investor"] == inv and r["start"] == label:
                    print(f"{inv:<3} {label:<20} {r['variant']:<32} {str(r.get('termination',''))[:14]:<14} {f(r.get('embedded_profit')):>11.2f}"
                          f" {f(r.get('recleared_profit')):>11.2f} {f(r.get('embedded_minus_reclear_profit')):>8.2f}"
                          f" {f(r.get('embedded_primal_dual_gap')):>9.3g} {f(r.get('unilateral_gain')):>9.1f} {f(r.get('solve_seconds')):>5.0f}")
    print(f"done in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
