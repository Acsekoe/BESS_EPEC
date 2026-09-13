"""Regime segments, externalities and gain decomposition along the paths (no solves).

Reads the path-scan checkpoint and writes
  output/paths/regime_segments.csv     consecutive s-ranges with one price-forming congestion set
  output/paths/path_decomposition.csv  gain components and midday prices at selected s
  output/paths/path_externalities.csv  rival profit changes along each path
"""

from __future__ import annotations

import csv
import pickle

import numpy as np

import regret_common as rc

RAW = rc.OUTPUT_ROOT / "paths" / "raw" / "scan_state.pkl"
OUT = rc.OUTPUT_ROOT / "paths"
PROBE_S = (1e-6, 1e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.25, 0.5, 1.0)


def write(name, rows):
    if rows:
        with (OUT / name).open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    with RAW.open("rb") as h:
        state = pickle.load(h)
    store, paths = state["store"], state["paths"]
    data = rc.market_data(100.0)
    N, T = list(data.nodes), list(data.times)
    midday = [T.index(t) for t in T if 11 <= t <= 16]
    seg_rows, dec_rows, ext_rows = [], [], []
    for (name, rho), pts in sorted(store.items()):
        inv = paths[name]["investor"]
        ss = sorted(s for s in pts if "sets" in pts[s])
        # regime segments
        start = ss[0]
        current = pts[start]["sets"]["lines_priced"]
        for prev, s in zip(ss, ss[1:] + [None]):
            nxt = pts[s]["sets"]["lines_priced"] if s is not None else None
            if s is None or nxt != current:
                seg_rows.append(dict(path=name, rho=rho, investor=inv, s_start=repr(start), s_end=repr(prev),
                                     lines_priced=";".join(sorted(current)), n_priced=len(current),
                                     profit_start=pts[start]["res"]["settle"][inv]["profit"],
                                     profit_end=pts[prev]["res"]["settle"][inv]["profit"]))
                if s is not None:
                    start, current = s, nxt
        base = pts[0.0]["res"]
        # decomposition at selected s (nearest sampled point)
        for target in PROBE_S:
            s = min(ss, key=lambda x: abs(x - target))
            res = pts[s]["res"]
            st, b = res["settle"][inv], base["settle"][inv]
            dec_rows.append(dict(
                path=name, rho=rho, investor=inv, target_s=target, sampled_s=repr(s),
                gain=st["profit"] - b["profit"],
                d_storage_settlement=st["storage_settlement"] - b["storage_settlement"],
                d_degradation=-(st["degradation"] - b["degradation"]),
                d_owned_generation_rent=st["owned_generation_rent"] - b["owned_generation_rent"],
                d_capex=-(st["capex"] - b["capex"]),
                midday_mean_lmp_N6=float(np.mean(res["lmp"][N.index("N6"), midday])),
                midday_mean_lmp_N8=float(np.mean(res["lmp"][N.index("N8"), midday])),
                midday_mean_lmp_N3=float(np.mean(res["lmp"][N.index("N3"), midday])),
                evening_mean_lmp_N5=float(np.mean(res["lmp"][N.index("N5"), [T.index(t) for t in T if 18 <= t <= 22]])),
                total_abs_da_mwh=res["total_abs_da_mwh"],
                lines_priced=";".join(sorted(pts[s]["sets"]["lines_priced"])),
                **{f"d_profit_{o}": res["settle"][o]["profit"] - base["settle"][o]["profit"] for o in rc.INVESTORS}))
        # externalities over the whole path
        row = dict(path=name, rho=rho, investor=inv)
        for o in rc.INVESTORS:
            vals = np.array([pts[s]["res"]["settle"][o]["profit"] - base["settle"][o]["profit"] for s in ss])
            row[f"{o}_profit_change_at_s1"] = float(vals[-1])
            row[f"{o}_max_abs_profit_change"] = float(np.max(np.abs(vals)))
        ext_rows.append(row)
    write("regime_segments.csv", seg_rows)
    write("path_decomposition.csv", dec_rows)
    write("path_externalities.csv", ext_rows)
    for r in ext_rows:
        print(f"{r['path']:<34} rho={r['rho']:<4g} " + " ".join(
            f"{o}: end={r[f'{o}_profit_change_at_s1']:+9.1f} max|.|={r[f'{o}_max_abs_profit_change']:8.1f}" for o in rc.INVESTORS))
    print()
    for r in dec_rows:
        if r["rho"] == 100.0 and r["path"] in ("I3_selected_duration_8h", "I3_incumbent_start", "I1_incumbent_start", "I2_incumbent_start"):
            print(f"{r['path']:<26} s~{r['target_s']:<7g} gain={r['gain']:>9.1f} stor={r['d_storage_settlement']:>8.1f}"
                  f" deg={r['d_degradation']:>7.1f} rent={r['d_owned_generation_rent']:>9.1f} capex={r['d_capex']:>7.1f}"
                  f" N6={r['midday_mean_lmp_N6']:6.2f} N8={r['midday_mean_lmp_N8']:6.2f} N3={r['midday_mean_lmp_N3']:6.2f}"
                  f" N5eve={r['evening_mean_lmp_N5']:6.2f} DA={r['total_abs_da_mwh']:6.2f} priced={r['lines_priced']}")
    print()
    for r in seg_rows:
        if r["rho"] == 100.0:
            print(f"{r['path']:<34} [{float(r['s_start']):.4g},{float(r['s_end']):.4g}] pi {r['profit_start']:.1f}->{r['profit_end']:.1f} priced={r['lines_priced']}")


if __name__ == "__main__":
    main()
