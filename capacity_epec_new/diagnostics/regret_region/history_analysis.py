"""What keeps moving in the Jacobi run?

Reconstructs, for every sweep of init5mw15mwh_jacobi_rho100_d025_s100, the
profile the investors answered (state after sweep k-1) and the single-start
responses they returned (starts_sweep_k.json), and reports per coordinate
  * the state range and standard deviation over the late window,
  * the raw response pressure (response - state), its sign-change rate and
    lag-1 autocorrelation (a period-2 cycle shows as strongly negative),
  * the dominant autocorrelation lag of the state series.

With --clear, every late-window state and every investor's response are
re-cleared to record the price-forming congestion set and the profit
decomposition, showing whether the market regime flips along the iteration.

Outputs: output/history/*.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict

import numpy as np

import regret_common as rc

OUT = rc.OUTPUT_ROOT / "history"
NODES = [f"N{k}" for k in range(1, 10)]
WINDOW = range(51, 101)


def load_states():
    states = defaultdict(dict)
    with (rc.RUN_DIR / "capacity_by_investor_node_by_sweep.csv").open(newline="", encoding="utf-8") as h:
        for row in csv.DictReader(h):
            states[int(row["sweep"])][row["investor"], row["node"]] = (float(row["power_mw"]), float(row["energy_mwh"]))
    return states


def load_responses():
    responses = {}
    for k in range(1, 101):
        payload = json.loads((rc.RUN_DIR / f"starts_sweep_{k:03d}.json").read_text(encoding="utf-8"))
        responses[k] = {}
        for inv, records in payload.items():
            sel = next(r for r in records if r.get("selected"))
            responses[k][inv] = sel
    return responses


def autocorr(x, lag):
    x = np.asarray(x, dtype=float)
    if len(x) <= lag or np.std(x) == 0:
        return math.nan
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    states = load_states()
    responses = load_responses()
    history = {int(r["sweep"]): r for r in csv.DictReader((rc.RUN_DIR / "history.csv").open(newline="", encoding="utf-8"))}

    pressure_rows, coord_rows = [], []
    for k in range(1, 101):
        for inv in rc.INVESTORS:
            sel = responses[k][inv]
            for n in NODES:
                p0, e0 = states[k - 1][inv, n]
                pressure_rows.append(dict(sweep=k, investor=inv, node=n, state_power=p0, state_energy=e0,
                                          response_power=sel["power"][n], response_energy=sel["energy"][n],
                                          power_pressure=sel["power"][n] - p0, energy_pressure=sel["energy"][n] - e0,
                                          response_start=sel["start"], response_phase=sel["phase"],
                                          response_exact_profit=sel["exact_profit"]))
    for inv in rc.INVESTORS:
        for n in NODES:
            for kind, idx in (("power", 0), ("energy", 1)):
                series = [states[k][inv, n][idx] for k in WINDOW]
                press = [r[f"{kind}_pressure"] for r in pressure_rows if r["investor"] == inv and r["node"] == n and r["sweep"] in WINDOW]
                signs = np.sign(press)
                changes = float(np.mean(signs[1:] != signs[:-1])) if len(signs) > 1 else math.nan
                lags = {lag: autocorr(series, lag) for lag in range(1, 16)}
                finite_lags = {lag: v for lag, v in lags.items() if math.isfinite(v) and lag >= 2}
                coord_rows.append(dict(
                    investor=inv, node=n, kind=kind,
                    state_mean=float(np.mean(series)), state_range=float(np.ptp(series)), state_std=float(np.std(series)),
                    pressure_mean=float(np.mean(press)), pressure_abs_mean=float(np.mean(np.abs(press))),
                    pressure_sign_change_rate=changes, pressure_lag1_autocorr=autocorr(press, 1),
                    state_lag1_autocorr=lags[1],
                    dominant_state_lag=max(finite_lags, key=finite_lags.get) if finite_lags else "",
                    dominant_state_lag_autocorr=max(finite_lags.values()) if finite_lags else math.nan))
    regret_rows = []
    for inv in rc.INVESTORS:
        reg = [float(history[k][f"regret_{inv}_eur_per_day"]) for k in WINDOW]
        prof = [float(history[k][f"incumbent_profit_{inv}_eur_per_day"]) for k in WINDOW]
        regret_rows.append(dict(investor=inv, regret_mean=float(np.mean(reg)), regret_min=float(np.min(reg)),
                                regret_max=float(np.max(reg)), regret_lag1_autocorr=autocorr(reg, 1),
                                regret_lag2_autocorr=autocorr(reg, 2),
                                relative_regret_mean=float(np.mean(np.array(reg) / np.maximum(1.0, np.abs(prof)))),
                                incumbent_profit_range=float(np.ptp(prof))))
    for name, rows in (("response_pressure_by_sweep.csv", pressure_rows), ("coordinate_oscillation_sweeps51_100.csv", coord_rows),
                       ("regret_statistics_sweeps51_100.csv", regret_rows)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    top = sorted([r for r in coord_rows if r["state_mean"] > 1e-3], key=lambda r: -r["state_range"])[:12]
    for r in top:
        print(f"{r['investor']} {r['node']} {r['kind']:<6} range={r['state_range']:.3f} mean={r['state_mean']:.2f}"
              f" |press|={r['pressure_abs_mean']:.3f} signchg={r['pressure_sign_change_rate']:.2f}"
              f" press_ac1={r['pressure_lag1_autocorr']:.2f} lag*={r['dominant_state_lag']} ac={r['dominant_state_lag_autocorr']:.2f}")
    for r in regret_rows:
        print(r)

    # Ownership exchange: do node totals stay fixed while investors trade capacity?
    exchange_rows = []
    for n in NODES:
        for kind, idx in (("power", 0), ("energy", 1)):
            totals = [sum(states[k][inv, n][idx] for inv in rc.INVESTORS) for k in WINDOW]
            indiv = {inv: [states[k][inv, n][idx] for k in WINDOW] for inv in rc.INVESTORS}
            row = dict(node=n, kind=kind, total_mean=float(np.mean(totals)), total_range=float(np.ptp(totals)),
                       sum_of_investor_ranges=float(sum(np.ptp(v) for v in indiv.values())))
            row["exchange_ratio"] = (row["total_range"] / row["sum_of_investor_ranges"]
                                     if row["sum_of_investor_ranges"] > 0 else math.nan)
            diffs = {inv: np.diff(v) for inv, v in indiv.items()}
            for a, b in (("I1", "I2"), ("I1", "I3"), ("I2", "I3")):
                if np.std(diffs[a]) > 0 and np.std(diffs[b]) > 0:
                    row[f"corr_diff_{a}_{b}"] = float(np.corrcoef(diffs[a], diffs[b])[0, 1])
                else:
                    row[f"corr_diff_{a}_{b}"] = math.nan
            for inv in rc.INVESTORS:
                row[f"diff_lag1_autocorr_{inv}"] = autocorr(diffs[inv], 1)
            exchange_rows.append(row)
    with (OUT / "node_ownership_exchange_sweeps51_100.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(exchange_rows[0]))
        w.writeheader()
        w.writerows(exchange_rows)
    for r in exchange_rows:
        if r["sum_of_investor_ranges"] > 1.0:
            print(f"{r['node']} {r['kind']:<6} total_range={r['total_range']:.2f} sum_indiv_ranges={r['sum_of_investor_ranges']:.2f}"
                  f" ratio={r['exchange_ratio']:.2f} corr(dI1,dI3)={r['corr_diff_I1_I3']:.2f}"
                  f" corr(dI1,dI2)={r['corr_diff_I1_I2']:.2f} corr(dI2,dI3)={r['corr_diff_I2_I3']:.2f}"
                  f" dAC1 I1/I2/I3={r['diff_lag1_autocorr_I1']:.2f}/{r['diff_lag1_autocorr_I2']:.2f}/{r['diff_lag1_autocorr_I3']:.2f}")

    if not args.clear:
        return
    data = rc.market_data(100.0)
    static = rc.static_arrays(data)
    tasks = []
    for k in WINDOW:
        state_p = {key: v[0] for key, v in states[k - 1].items()}
        state_e = {key: v[1] for key, v in states[k - 1].items()}
        tasks.append(((k, "state"), 100.0, state_p, state_e))
        for inv in rc.INVESTORS:
            sel = responses[k][inv]
            p, e = rc.with_candidate(state_p, state_e, inv, sel["power"], sel["energy"])
            tasks.append(((k, inv), 100.0, p, e))
    with rc.make_pool(args.workers) as pool:
        results = rc.evaluate_many(pool, tasks)
    rows = []
    for k in WINDOW:
        base = results[(k, "state")]
        sets = rc.active_sets(base, static)
        row = dict(sweep=k, state_lines_priced=";".join(sorted(sets["lines_priced"])),
                   state_lines_priced_hash=rc.signature_hash(sets["lines_priced"]))
        for inv in rc.INVESTORS:
            row[f"state_profit_{inv}"] = base["settle"][inv]["profit"]
            res = results[(k, inv)]
            rs = rc.active_sets(res, static)
            st, b = res["settle"][inv], base["settle"][inv]
            row[f"{inv}_response_gain"] = st["profit"] - b["profit"]
            row[f"{inv}_gain_storage_surplus"] = st["storage_operating_surplus"] - b["storage_operating_surplus"]
            row[f"{inv}_gain_generation_rent"] = st["owned_generation_rent"] - b["owned_generation_rent"]
            row[f"{inv}_gain_capex"] = -(st["capex"] - b["capex"])
            row[f"{inv}_response_lines_priced_entered"] = ";".join(sorted(rs["lines_priced"] - sets["lines_priced"]))
            row[f"{inv}_response_lines_priced_left"] = ";".join(sorted(sets["lines_priced"] - rs["lines_priced"]))
            row[f"{inv}_response_max_abs_lmp_change"] = float(np.max(np.abs(res["lmp"] - base["lmp"])))
        rows.append(row)
    with (OUT / "late_window_regimes.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("regime hashes over late window:", len({r["state_lines_priced_hash"] for r in rows}))


if __name__ == "__main__":
    main()
