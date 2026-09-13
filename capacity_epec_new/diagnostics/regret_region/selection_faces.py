"""How much of an investor's settlement is decided by price/dispatch selection?

For a fixed capacity profile the ISO QP has a unique optimal cost and a unique
demand adjustment (the only strictly convex term), hence unique LMPs at loaded
nodes.  Other prices, and the split of dispatch among co-located batteries, can
be non-unique.  Two LPs bound the selection freedom around an EXACT optimum
(re-solved with MA27 so that the reference solve's ghost-capacity residuals do
not create spurious slack):

  dual face    primal dispatch fixed at the exact optimum; ISO duals range over
               dual feasibility (the maintained MPEC stationarity constraints)
               with dual objective >= optimal cost - delta.  Objective:
               max/min of the investor's storage settlement + owned-generation
               rent, which is linear in the prices once dispatch is fixed.

  primal face  LMPs fixed at the exact optimum; dispatch ranges over primal
               feasibility with ISO cost <= optimal cost + delta and demand
               adjustment fixed at its unique value.  Objective: max/min of
               the investor's settlement - degradation + owned-generation rent.

Both are one-sided (the joint primal-dual face is bilinear), so they give lower
bounds on the true selection range.  The MPEC's own artificial price and dual
bounds are kept in the dual face LP.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyomo.environ as pyo

import regret_common as rc
import iso_market
import mpec

OUT = rc.OUTPUT_ROOT / "selection_faces"
DELTAS = (1e-2, 1e-5)


def _solve_lp(model, sense_obj):
    from pyomo.contrib.appsi.base import TerminationCondition
    from pyomo.contrib.appsi.solvers import Highs

    solver = Highs()
    solver.config.load_solution = False
    result = solver.solve(model)
    status = str(result.termination_condition)
    if result.termination_condition != TerminationCondition.optimal:
        return status, math.nan
    result.solution_loader.load_vars()
    return status, float(pyo.value(sense_obj))


def exact_state(data, config, power, energy):
    market = rc.resolve_market(data, config, power, energy, "ma27")
    res = rc.extract_market(market, data, config, power, energy)
    return res


def dual_face(data, config, investor_id, power, energy, res, delta):
    N, T, S, G, L = list(data.nodes), list(data.times), list(data.soc_times), list(data.generators), list(data.lines)
    units = list(config.investor_ids)
    investor = next(i for i in config.investors if i.investor_id == investor_id)
    rivals = [i for i in config.investors if i.investor_id != investor_id]
    m = mpec.build_capacity_mpec(
        data,
        investor=investor,
        rival_power={r.investor_id: {n: power[r.investor_id, n] for n in N} for r in rivals},
        rival_energy={r.investor_id: {n: energy[r.investor_id, n] for n in N} for r in rivals},
        rival_degradation={r.investor_id: r.degradation_eur_per_mwh for r in rivals},
        node_connection_limit=config.node_limits(data),
        lower_level="strong-duality",
        complementarity_epsilon=0.0,
        price_bound=config.price_bound,
        dual_bound=config.dual_bound,
        sparse_capacity_tol=config.sparse_capacity_tol,
    )
    for n in N:
        m.X_power[n].fix(power[investor_id, n])
        m.X_energy[n].fix(energy[investor_id, n])
    ui = {u: k for k, u in enumerate(units)}
    for g, t in m.GT:
        m.P_gen[g, t].fix(res["gen"][G.index(g), T.index(t)])
    for i, n in m.IN:
        a, k = ui[i], N.index(n)
        for b, t in enumerate(T):
            m.P_charge[i, n, t].fix(res["charge"][a, k, b])
            m.P_discharge[i, n, t].fix(res["discharge"][a, k, b])
        for b, s in enumerate(S):
            m.SOC[i, n, s].fix(res["soc"][a, k, b])
    for k, n in enumerate(N):
        for b, t in enumerate(T):
            m.NetInjection[n, t].fix(res["ni"][k, b])
            m.DemandAdjustment[n, t].fix(res["da"][k, b] if data.demand_is_adjustable(n, t) else 0.0)
    for con in m.component_objects(pyo.Constraint, descend_into=True):
        con.deactivate()
    for name in ("gen_stationarity", "charge_stationarity", "discharge_stationarity",
                 "netinjection_stationarity", "demand_adjustment_stationarity", "soc_stationarity"):
        getattr(m, name).activate()
    m.objective.deactivate()
    rho = data.demand_adjustment_penalty_eur_per_mw2
    demand = {(n, t): data.demand_el[n, t] for n in N for t in T}
    cap = {(g, t): data.generation_capacity[g, t] for g, t in m.GT}
    pw = {(i, n): power[i, n] for i, n in m.IN}
    en = {(i, n): energy[i, n] for i, n in m.IN}
    da_const = 0.5 * rho * float(np.sum(res["da"] ** 2))
    dual_linear = (
        sum(demand[n, t] * m.lam[n, t] for n in N for t in T)
        + sum(cap[g, t] * m.nu_gen[g, t] for g, t in m.GT)
        + sum(data.line_limit[l] * (m.mu_up[l, t] - m.mu_dn[l, t]) for l in L for t in T)
        + sum(pw[i, n] * (m.rho_ch[i, n, t] + m.sig_dis[i, n, t]) for i, n in m.IN for t in T)
        + sum(en[i, n] * m.del_soc[i, n, s] for i, n in m.IN for s in S)
    )
    optimal_cost = res["objective"]
    m.face_cut = pyo.Constraint(expr=dual_linear >= optimal_cost + da_const - delta)
    u = ui[investor_id]
    q = res["discharge"][u] - res["charge"][u]
    gen_node = data.nodes_by_generator()
    revenue = sum(float(q[k, b]) * m.lam[n, t] for k, n in enumerate(N) for b, t in enumerate(T))
    rent = 0
    for g, share in investor.owned_generation_shares.items():
        node = gen_node[g][0]
        for b, t in enumerate(T):
            p = float(res["gen"][G.index(g), b])
            if p:
                rent = rent + share * p * m.lam[node, t]
    rent_const = -sum(share * data.generation_cost[g] * float(res["gen"][G.index(g), b])
                      for g, share in investor.owned_generation_shares.items() for b in range(len(T)))
    out = {}
    for sense, label in ((pyo.maximize, "max"), (pyo.minimize, "min")):
        if hasattr(m, "face_obj"):
            m.del_component(m.face_obj)
        m.face_obj = pyo.Objective(expr=revenue + rent, sense=sense)
        status, value = _solve_lp(m, m.face_obj.expr)
        out[f"dual_face_{label}_settlement_plus_rent"] = value + rent_const if math.isfinite(value) else math.nan
        out[f"dual_face_{label}_status"] = status
        if math.isfinite(value):
            out[f"dual_face_{label}_price_bound_utilization"] = max(abs(pyo.value(v)) for v in m.lam.values()) / config.price_bound
    ref_value = float(np.sum(q * res["lmp"])) + sum(
        share * (res["lmp"][N.index(gen_node[g][0]), b] - data.generation_cost[g]) * res["gen"][G.index(g), b]
        for g, share in investor.owned_generation_shares.items() for b in range(len(T)))
    out["exact_settlement_plus_rent"] = ref_value
    out["dual_face_range"] = out["dual_face_max_settlement_plus_rent"] - out["dual_face_min_settlement_plus_rent"]
    return out


def primal_face(data, config, investor_id, power, energy, res, delta):
    N, T, G = list(data.nodes), list(data.times), list(data.generators)
    units = list(config.investor_ids)
    investor = next(i for i in config.investors if i.investor_id == investor_id)
    m = iso_market.build_market(data, power, energy, config.degradation())
    for k, n in enumerate(N):
        for b, t in enumerate(T):
            if data.demand_is_adjustable(n, t):
                m.DemandAdjustment[n, t].fix(float(res["da"][k, b]))
    m.objective.deactivate()
    deg = config.degradation()
    rho = data.demand_adjustment_penalty_eur_per_mw2
    da_const = 0.5 * rho * float(np.sum(res["da"] ** 2))
    cost = (sum(data.offer(g) * m.P_gen[g, t] for g in G for t in T)
            + sum(0.5 * deg[i] * (m.P_charge[i, n, t] + m.P_discharge[i, n, t]) for i in units for n in N for t in T))
    m.face_cut = pyo.Constraint(expr=cost + da_const <= res["objective"] + delta)
    lam = res["lmp"]
    gen_node = data.nodes_by_generator()
    unit = investor_id
    settlement = sum(float(lam[k, b]) * (m.P_discharge[unit, n, t] - m.P_charge[unit, n, t])
                     for k, n in enumerate(N) for b, t in enumerate(T))
    degradation = 0.5 * deg[unit] * sum(m.P_charge[unit, n, t] + m.P_discharge[unit, n, t] for n in N for t in T)
    rent = sum(share * (float(lam[N.index(gen_node[g][0]), b]) - data.generation_cost[g]) * m.P_gen[g, t]
               for g, share in investor.owned_generation_shares.items() for b, t in enumerate(T))
    out = {}
    for sense, label in ((pyo.maximize, "max"), (pyo.minimize, "min")):
        if hasattr(m, "face_obj"):
            m.del_component(m.face_obj)
        m.face_obj = pyo.Objective(expr=settlement - degradation + rent, sense=sense)
        status, value = _solve_lp(m, m.face_obj.expr)
        out[f"primal_face_{label}_surplus_plus_rent"] = value
        out[f"primal_face_{label}_status"] = status
    u = units.index(unit)
    exact = (float(np.sum((res["discharge"][u] - res["charge"][u]) * lam))
             - 0.5 * deg[unit] * float(np.sum(res["charge"][u] + res["discharge"][u]))
             + sum(share * (lam[N.index(gen_node[g][0]), b] - data.generation_cost[g]) * res["gen"][G.index(g), b]
                   for g, share in investor.owned_generation_shares.items() for b in range(len(T))))
    out["exact_surplus_plus_rent"] = exact
    out["primal_face_range"] = out["primal_face_max_surplus_plus_rent"] - out["primal_face_min_surplus_plus_rent"]
    return out


def face_task(args):
    label, rho, investor_id, power, energy = args
    data = rc.market_data(rho)
    config = rc.game_config(data)
    tick = time.perf_counter()
    row = dict(profile=label, rho=rho, investor=investor_id)
    try:
        res = exact_state(data, config, power, energy)
    except Exception as exc:
        row["error"] = f"exact clear: {exc}"
        return row
    row["exact_pd_gap"] = res["pd_gap"]
    for delta in DELTAS:
        try:
            for k, v in dual_face(data, config, investor_id, power, energy, res, delta).items():
                row[f"d{delta:g}_{k}"] = v
        except Exception as exc:
            row[f"d{delta:g}_dual_face_error"] = str(exc)[:300]
        try:
            for k, v in primal_face(data, config, investor_id, power, energy, res, delta).items():
                row[f"d{delta:g}_{k}"] = v
        except Exception as exc:
            row[f"d{delta:g}_primal_face_error"] = str(exc)[:300]
    row["seconds"] = time.perf_counter() - tick
    return row


def run_tasks(tasks, workers, out_name):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(face_task, t) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(row["profile"], row["rho"], row["investor"],
                  {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                   if k.endswith("_range") or k.endswith("error")}, flush=True)
    rows.sort(key=lambda r: (r["profile"], r["rho"], r["investor"]))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / out_name).open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    power, energy = rc.incumbent()
    tasks = []
    for rho in (100.0, 50.0, 25.0):
        for inv in rc.INVESTORS:
            tasks.append(("incumbent", rho, inv, power, energy))
    for inv in rc.INVESTORS:
        for rec in rc.audit_records()[inv]:
            if rec["phase"] == "refine":
                p, e = rc.with_candidate(power, energy, inv, rec["power"], rec["energy"])
                tasks.append((f"{inv}:{rec['start']}", 100.0, inv, p, e))
    run_tasks(tasks, args.workers, "faces_incumbent_and_audit.csv")


if __name__ == "__main__":
    main()
