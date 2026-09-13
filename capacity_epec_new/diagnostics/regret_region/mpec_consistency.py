"""Do the best-response NLP and the final settlement represent the same game?

At the final profile of init5mw15mwh_jacobi_rho100_d025_s100, every investor's
MPEC is re-solved from (a) the incumbent and (b) the start whose refinement
the audit selected, under a documented set of lower-level settings:

  relaxed KKT with epsilon in {1e-3 (run), 1e-4, 1e-5, 1e-6, 1e-8} and IPOPT
  tolerance in {1e-4 (run), 1e-6, 1e-8}; the maintained strong-duality
  formulation; and an epsilon-continuation chain that warm-starts each stage
  from the previous stage's capacities.  At rho = 25 and 50 the run settings
  are repeated from the same starts.

Nothing is loosened: the audit tolerance of 10 EUR/day is only reported.

For every returned capacity the maintained reclear is evaluated and the
embedded-minus-reclear profit is decomposed exactly (symmetric price/quantity
split) into storage-settlement, generation-rent, degradation and capex parts.

Outputs: output/mpec_consistency/mpec_variants.csv, mpec_variants.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyomo.environ as pyo

import regret_common as rc
import capacity_game
import mpec
import solvers

OUT = rc.OUTPUT_ROOT / "mpec_consistency"
AUDIT_TOLERANCE = 10.0
CURRENT_RHO100 = {"I1": 1920.9181763696738, "I2": 73564.62737151793, "I3": 90099.78858565052}

VARIANTS = {
    "run_eps1e-3_tol1e-4": dict(lower_level="relaxed-kkt", epsilon=1e-3, tolerance=1e-4),
    "eps1e-4_tol1e-4": dict(lower_level="relaxed-kkt", epsilon=1e-4, tolerance=1e-4),
    "eps1e-3_tol1e-6": dict(lower_level="relaxed-kkt", epsilon=1e-3, tolerance=1e-6),
    "eps1e-3_tol1e-8": dict(lower_level="relaxed-kkt", epsilon=1e-3, tolerance=1e-8),
    "eps1e-5_tol1e-6": dict(lower_level="relaxed-kkt", epsilon=1e-5, tolerance=1e-6),
    "eps1e-6_tol1e-6": dict(lower_level="relaxed-kkt", epsilon=1e-6, tolerance=1e-6),
    "eps1e-6_tol1e-8": dict(lower_level="relaxed-kkt", epsilon=1e-6, tolerance=1e-8),
    "eps1e-8_tol1e-8": dict(lower_level="relaxed-kkt", epsilon=1e-8, tolerance=1e-8),
    "strong_duality_tol1e-4": dict(lower_level="strong-duality", epsilon=0.0, tolerance=1e-4),
    "strong_duality_tol1e-6": dict(lower_level="strong-duality", epsilon=0.0, tolerance=1e-6),
    "strong_duality_tol1e-8": dict(lower_level="strong-duality", epsilon=0.0, tolerance=1e-8),
}
CHAIN = [
    ("chain0_eps1e-3_tol1e-4", dict(lower_level="relaxed-kkt", epsilon=1e-3, tolerance=1e-4)),
    ("chain1_eps1e-4_tol1e-6", dict(lower_level="relaxed-kkt", epsilon=1e-4, tolerance=1e-6)),
    ("chain2_eps1e-5_tol1e-7", dict(lower_level="relaxed-kkt", epsilon=1e-5, tolerance=1e-7)),
    ("chain3_eps1e-6_tol1e-8", dict(lower_level="relaxed-kkt", epsilon=1e-6, tolerance=1e-8)),
    ("chain4_strong_duality_tol1e-8", dict(lower_level="strong-duality", epsilon=0.0, tolerance=1e-8)),
]
SELECTED_START = {"I1": "replace_N6_10mw_4h", "I2": "replace_N3_10mw_2h", "I3": "duration_8h"}


def start_capacities(investor: str, label: str):
    rec = rc.audit_record(investor, label, "screen")
    return rec["power"], rec["energy"]


def solve_and_decompose(rho: float, investor_id: str, start_label: str, variant: str, spec: dict,
                        start_power: dict, start_energy: dict) -> dict:
    data = rc.market_data(rho)
    config = rc.game_config(data, epsilon=spec["epsilon"], tolerance=spec["tolerance"],
                            lower_level=spec["lower_level"], max_seconds=spec.get("max_seconds", 300.0))
    reference = rc.game_config(data)  # procedural reclear always uses the run's clearing settings
    investor = next(i for i in config.investors if i.investor_id == investor_id)
    power, energy = rc.incumbent()
    N, T = list(data.nodes), list(data.times)
    row = dict(rho=rho, investor=investor_id, start=start_label, variant=variant,
               lower_level=spec["lower_level"], epsilon=spec["epsilon"], tolerance=spec["tolerance"])
    tick = time.perf_counter()
    try:
        model = capacity_game.build_best_response_model(data, config, investor, power, energy,
                                                        start_power, start_energy)
    except Exception as exc:
        row.update(termination=f"build_error: {exc}", seconds=time.perf_counter() - tick)
        return row
    outcome = solvers.solve_mpec(model, config.solver)
    row.update(termination=outcome.termination, optimal=outcome.optimal, has_solution=outcome.has_solution,
               solve_seconds=outcome.seconds)
    try:
        violation = solvers.maximum_bound_violation(model)
        diag = mpec.complementarity_diagnostics(model)
        bounds = mpec.artificial_bound_diagnostics(model)
    except Exception as exc:
        row.update(diagnostic_error=str(exc))
        return row
    row.update(max_bound_violation=violation, complementarity_max_product=diag["maximum_product"],
               complementarity_min_product=diag["minimum_product"],
               complementarity_excess_over_epsilon=diag["maximum_upper_bound_violation"],
               embedded_primal_dual_gap=diag["primal_dual_gap_eur_per_day"],
               artificial_bound_utilization=bounds["maximum_artificial_bound_utilization"])
    cand_p = {n: max(0.0, float(pyo.value(model.X_power[n]))) for n in N}
    cand_e = {n: max(0.0, float(pyo.value(model.X_energy[n]))) for n in N}
    for n in N:
        row[f"P_{n}"] = cand_p[n]
    for n in N:
        row[f"E_{n}"] = cand_e[n]
    row["max_nodal_power_change_mw"] = max(abs(cand_p[n] - power[investor_id, n]) for n in N)
    row["max_nodal_energy_change_mwh"] = max(abs(cand_e[n] - energy[investor_id, n]) for n in N)
    val = lambda x: float(pyo.value(x))
    emb = dict(storage_settlement=val(model.spot_revenue), generation_rent=val(model.generation_rent),
               degradation=val(model.active_degradation), capex=val(model.daily_capex),
               profit=val(model.unregularized_profit), primal_objective=val(model.primal_objective),
               dual_objective=val(model.dual_objective))
    for k, v in emb.items():
        row[f"embedded_{k}"] = v
    lam_e = np.array([[val(model.lam[n, t]) for t in T] for n in N])
    q_e = np.array([[val(model.P_discharge[investor_id, n, t]) - val(model.P_charge[investor_id, n, t])
                     for t in T] for n in N])
    ch_e = np.array([[val(model.P_charge[investor_id, n, t]) for t in T] for n in N])
    dis_e = np.array([[val(model.P_discharge[investor_id, n, t]) for t in T] for n in N])

    full_p, full_e = rc.with_candidate(power, energy, investor_id, cand_p, cand_e)
    try:
        rc.check_feasible(data, reference, investor_id, cand_p, cand_e, full_p)
        row["feasible"] = True
    except ValueError as exc:
        row["feasible"] = f"no: {exc}"
    res = rc.evaluate_profile(data, reference, full_p, full_e)
    row["reclear_status"] = res.get("status")
    if res.get("status") != "optimal":
        return row
    st = res["settle"][investor_id]
    u = rc.INVESTORS.index(investor_id)
    lam_r = res["lmp"]
    q_r = res["discharge"][u] - res["charge"][u]
    row.update(recleared_profit=st["profit"], reclear_storage_settlement=st["storage_settlement"],
               reclear_generation_rent=st["owned_generation_rent"], reclear_degradation=st["degradation"],
               reclear_capex=st["capex"], reclear_objective=res["objective"], reclear_pd_gap=res["pd_gap"],
               reclear_max_complementarity=res["max_complementarity"])
    row["embedded_minus_reclear_profit"] = emb["profit"] - st["profit"]
    row["abs_gap_exceeds_audit_tolerance"] = abs(emb["profit"] - st["profit"]) > AUDIT_TOLERANCE
    # exact symmetric decomposition
    row["gap_storage_price_effect"] = float(np.sum((lam_e - lam_r) * 0.5 * (q_e + q_r)))
    row["gap_storage_quantity_effect"] = float(np.sum((q_e - q_r) * 0.5 * (lam_e + lam_r)))
    G = list(data.generators)
    gen_node = data.nodes_by_generator()
    price_eff = qty_eff = 0.0
    for g, share in investor.owned_generation_shares.items():
        a = N.index(gen_node[g][0])
        b = G.index(g)
        for k, t in enumerate(T):
            pe = val(model.P_gen[g, t]) if (g, t) in model.GT else 0.0
            pr = res["gen"][b, k]
            cost = data.generation_cost[g]
            price_eff += share * (lam_e[a, k] - lam_r[a, k]) * 0.5 * (pe + pr)
            qty_eff += share * (pe - pr) * (0.5 * (lam_e[a, k] + lam_r[a, k]) - cost)
    row["gap_generation_price_effect"] = price_eff
    row["gap_generation_quantity_effect"] = qty_eff
    row["gap_degradation"] = -(emb["degradation"] - st["degradation"])
    row["gap_capex"] = -(emb["capex"] - st["capex"])
    row["gap_decomposition_residual"] = row["embedded_minus_reclear_profit"] - (
        row["gap_storage_price_effect"] + row["gap_storage_quantity_effect"] + price_eff + qty_eff
        + row["gap_degradation"] + row["gap_capex"])
    row["max_abs_lmp_embedded_minus_reclear"] = float(np.max(np.abs(lam_e - lam_r)))
    owned_nodes = sorted({N.index(gen_node[g][0]) for g in investor.owned_generation_shares})
    row["max_abs_lmp_diff_at_owned_generation_nodes"] = (
        float(np.max(np.abs(lam_e[owned_nodes] - lam_r[owned_nodes]))) if owned_nodes else math.nan)
    row["max_abs_active_dispatch_diff_mw"] = float(max(np.max(np.abs(ch_e - res["charge"][u])),
                                                       np.max(np.abs(dis_e - res["discharge"][u]))))
    row["embedded_primal_cost_minus_reclear_optimum"] = emb["primal_objective"] - res["objective"]
    row["reclear_optimum_minus_embedded_dual_objective"] = res["objective"] - emb["dual_objective"]
    row["seconds"] = time.perf_counter() - tick
    return row


def _task(args):
    return solve_and_decompose(*args)


def _chain_task(args):
    rho, investor_id, start_label, start_power, start_energy = args
    rows = []
    p, e = start_power, start_energy
    for name, spec in CHAIN:
        row = solve_and_decompose(rho, investor_id, start_label, name, spec, p, e)
        rows.append(row)
        if "P_N1" in row:
            p = {n: row[f"P_{n}"] for n in (f"N{k}" for k in range(1, 10))}
            e = {n: row[f"E_{n}"] for n in (f"N{k}" for k in range(1, 10))}
    return rows


def current_profits(rho: float) -> dict:
    data = rc.market_data(rho)
    config = rc.game_config(data)
    power, energy = rc.incumbent()
    res = rc.evaluate_profile(data, config, power, energy)
    return {i: res["settle"][i]["profit"] for i in rc.INVESTORS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, chains = [], []
    for inv in rc.INVESTORS:
        for label in ("incumbent", SELECTED_START[inv]):
            sp, se = start_capacities(inv, label)
            for variant, spec in VARIANTS.items():
                tasks.append((100.0, inv, label, variant, spec, sp, se))
            for rho in (50.0, 25.0):
                tasks.append((rho, inv, label, "run_eps1e-3_tol1e-4", VARIANTS["run_eps1e-3_tol1e-4"], sp, se))
            chains.append((100.0, inv, label, sp, se))
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, t) for t in tasks] + [pool.submit(_chain_task, c) for c in chains]
        for fut in as_completed(futures):
            out = fut.result()
            for row in (out if isinstance(out, list) else [out]):
                rows.append(row)
                print(f"[{time.perf_counter()-t0:6.0f}s] rho={row['rho']} {row['investor']} {row['start']:<20} {row['variant']:<30}"
                      f" term={row.get('termination')} emb={row.get('embedded_profit', float('nan')):.2f}"
                      f" rec={row.get('recleared_profit', float('nan')):.2f}"
                      f" gap={row.get('embedded_minus_reclear_profit', float('nan')):+.2f}"
                      f" prod={row.get('complementarity_max_product', float('nan')):.1e}", flush=True)
    currents = {rho: current_profits(rho) for rho in (100.0, 50.0, 25.0)}
    for row in rows:
        cur = currents[row["rho"]][row["investor"]]
        row["current_profit_at_rho"] = cur
        if "recleared_profit" in row:
            row["unilateral_gain"] = row["recleared_profit"] - cur
            row["relative_gain"] = row["unilateral_gain"] / max(1.0, abs(cur))
    rows.sort(key=lambda r: (r["rho"], r["investor"], r["start"], r["variant"]))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "mpec_variants.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / "current_profits_by_rho.json").write_text(json.dumps(currents, indent=2), encoding="utf-8")
    print("done", len(rows), "rows", f"{time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
