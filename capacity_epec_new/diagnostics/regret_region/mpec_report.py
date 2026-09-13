"""Console digest of output/mpec_consistency/mpec_variants.csv (no solves)."""

from __future__ import annotations

import csv
import math

import regret_common as rc


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def main() -> None:
    rows = list(csv.DictReader((rc.OUTPUT_ROOT / "mpec_consistency" / "mpec_variants.csv").open(newline="", encoding="utf-8")))
    print(f"{'rho':>4} {'inv':<3} {'start':<20} {'variant':<32} {'term':<14} {'embedded':>11} {'reclear':>11} {'gap':>8}"
          f" {'gain':>9} {'rel':>7} {'stor_price':>10} {'stor_qty':>9} {'gen_price':>9} {'gen_qty':>8} {'dLMP':>7} {'dMW':>6}")
    for r in rows:
        print(f"{f(r['rho']):>4g} {r['investor']:<3} {r['start']:<20} {r['variant']:<32} {r.get('termination','')[:14]:<14}"
              f" {f(r.get('embedded_profit')):>11.2f} {f(r.get('recleared_profit')):>11.2f}"
              f" {f(r.get('embedded_minus_reclear_profit')):>8.2f} {f(r.get('unilateral_gain')):>9.2f} {f(r.get('relative_gain')):>7.3f}"
              f" {f(r.get('gap_storage_price_effect')):>10.2f} {f(r.get('gap_storage_quantity_effect')):>9.2f}"
              f" {f(r.get('gap_generation_price_effect')):>9.2f} {f(r.get('gap_generation_quantity_effect')):>8.2f}"
              f" {f(r.get('max_abs_lmp_embedded_minus_reclear')):>7.3f} {f(r.get('max_nodal_power_change_mw')):>6.2f}")


if __name__ == "__main__":
    main()
