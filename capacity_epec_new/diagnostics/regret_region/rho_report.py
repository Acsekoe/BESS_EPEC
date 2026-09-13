"""Demand-response sensitivity digest from the path-scan checkpoint (no solves).

For each rho at the final profile (s = 0 of any path):
  * physical demand adjustment: total |DA| (MWh/day), share of daily load,
    largest node-hour |DA| and its share of that node-hour's demand;
  * midday LMPs at the solar nodes N6 and N8;
For each path and rho:
  * steepest sampled secant slope |dprofit/ds| converted to EUR/day per MW
    of the path's largest nodal power change, on the coarse grid and over all
    samples (including bracket traces);
  * the finest traced bracket's slope and residual |dprofit|.
Writes output/paths/rho_sensitivity.csv.
"""

from __future__ import annotations

import csv
import pickle

import numpy as np

import regret_common as rc

RAW = rc.OUTPUT_ROOT / "paths" / "raw" / "scan_state.pkl"
OUT = rc.OUTPUT_ROOT / "paths" / "rho_sensitivity.csv"
NOISE_FLOOR_EUR = 1.0


def main() -> None:
    with RAW.open("rb") as h:
        state = pickle.load(h)
    store, brackets, paths = state["store"], state["brackets"], state["paths"]
    data = rc.market_data(100.0)
    N, T = list(data.nodes), list(data.times)
    demand = rc.static_arrays(data)["demand"]
    load = float(demand.sum())
    rows = []
    seen_rho = set()
    for (name, rho), pts in sorted(store.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        base = pts[0.0]["res"]
        if rho not in seen_rho:
            seen_rho.add(rho)
            da = np.abs(base["da"])
            k = np.unravel_index(np.argmax(da), da.shape)
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(demand > 0, da / demand, 0.0)
            ks = np.unravel_index(np.argmax(share), share.shape)
            midday = [t for t in T if 11 <= t <= 16]
            print(f"rho={rho:g}: total |DA|={da.sum():.3f} MWh/day ({da.sum()/load:.4%} of load);"
                  f" max node-hour |DA|={da[k]:.3f} MW at {N[k[0]]}@{T[k[1]]};"
                  f" max node-hour share={share[ks]:.3%} at {N[ks[0]]}@{T[ks[1]]};"
                  f" mean midday LMP N6={np.mean([base['lmp'][N.index('N6'), T.index(t)] for t in midday]):.2f}"
                  f" N8={np.mean([base['lmp'][N.index('N8'), T.index(t)] for t in midday]):.2f}"
                  f" profits={ {i: round(base['settle'][i]['profit'], 1) for i in rc.INVESTORS} }")
        path = paths[name]
        inv = path["investor"]
        p0, _ = path["start"]
        p1, _ = path["end"]
        dmw = max(abs(p1[n] - p0[n]) for n in N)
        ss = sorted(s for s in pts if pts[s]["res"].get("status") == "optimal")
        prof = {s: pts[s]["res"]["settle"][inv]["profit"] for s in ss}
        coarse = [s for s in ss if pts[s]["phase"] == "coarse"]

        def steepest(grid):
            # Only resolved changes: sub-EUR differences over nano-widths are
            # reclear noise (verification error up to ~0.8 EUR/day), not slopes.
            best = (0.0, None, None)
            for a, b in zip(grid[:-1], grid[1:]):
                if abs(prof[b] - prof[a]) < NOISE_FLOOR_EUR:
                    continue
                slope = abs(prof[b] - prof[a]) / (b - a) / dmw
                if slope > best[0]:
                    best = (slope, a, b)
            return best

        sc, sa = steepest(coarse), steepest(ss)
        finest = []
        for br in brackets.get((name, rho), []):
            resolved = [t for t in br["trace"] if abs(t["dpi"]) >= NOISE_FLOOR_EUR and t["width"] > 0]
            if resolved:
                t = resolved[-1]
                finest.append((abs(t["dpi"]) / t["width"] / dmw, abs(t["dpi"]), t["width"], br["id"]))
        top = max(finest) if finest else (float("nan"), float("nan"), float("nan"), "")
        da = np.abs(base["da"])
        rows.append(dict(path=name, rho=rho, investor=inv, path_max_nodal_power_change_mw=dmw,
                         incumbent_profit=prof[0.0], endpoint_gain=prof[ss[-1]] - prof[0.0],
                         max_gain_on_path=max(prof.values()) - prof[0.0],
                         steepest_coarse_slope_eur_per_day_per_mw=sc[0], steepest_coarse_interval=f"[{sc[1]},{sc[2]}]",
                         steepest_all_samples_slope_eur_per_day_per_mw=sa[0], steepest_all_interval=f"[{sa[1]},{sa[2]}]",
                         finest_bracket_slope_eur_per_day_per_mw=top[0], finest_bracket_abs_dprofit=top[1],
                         finest_bracket_width_s=top[2], finest_bracket=top[3],
                         incumbent_total_abs_da_mwh=float(da.sum()), incumbent_da_share_of_load=float(da.sum() / load),
                         incumbent_max_abs_da_mw=float(da.max())))
    with OUT.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print()
    for r in rows:
        print(f"{r['path']:<34} rho={r['rho']:<4g} endgain={r['endpoint_gain']:>9.1f} maxgain={r['max_gain_on_path']:>9.1f}"
              f" coarse_slope={r['steepest_coarse_slope_eur_per_day_per_mw']:>11.1f} {r['steepest_coarse_interval']:<24}"
              f" all_slope={r['steepest_all_samples_slope_eur_per_day_per_mw']:>11.1f}"
              f" finest={r['finest_bracket_slope_eur_per_day_per_mw']:>11.1f} (|dpi|={r['finest_bracket_abs_dprofit']:.3g}, w={r['finest_bracket_width_s']:.2g})")


if __name__ == "__main__":
    main()
