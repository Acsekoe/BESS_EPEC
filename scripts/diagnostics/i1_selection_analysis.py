"""I1 all-nodes: MPEC vs stand-alone LLP duals, optimistic price selection, and the planner benchmark."""
import copy
import csv
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
RUN = ROOT / "model" / "output" / "all_i1_bilinear_gurobi"
sys.path.insert(0, str(ROOT / "model"))
import pyomo.environ as pyo

import mpec
from dual_llp import dual_profit
from prepare_input import load_case, prepare_input
from primal_llp import add_primal, build_primal
from solution_space import extrema, face_models
from solve import require_optimal

INV = "I1"
data, base_profile = load_case()
chosen = json.loads((RUN / "capacities.json").read_text())
summary = json.loads((RUN / "summary.json").read_text())


def settlement(m, prices, investor):
    """Direct nodal settlement of the investor's storage and owned generation, minus capex."""
    storage = sum(prices[n, t] * (pyo.value(m.discharge[i, n, t]) - pyo.value(m.charge[i, n, t]))
                  - 0.5 * m.degradation[i] * pyo.value(m.charge[i, n, t] + m.discharge[i, n, t])
                  for i, n in m.storage_pairs if i == investor for t in m.timesteps)
    generation = sum(m.owned_share[investor, g] * (prices[m.generator_node[g], t] - m.generation_cost[g])
                     * pyo.value(m.generation[g, t]) for g in m.generators for t in m.timesteps)
    capex = sum(m.cost_power_daily[investor] * m.installed_power[investor, n]
                + m.cost_energy_daily[investor] * m.installed_energy[investor, n] for n in m.nodes)
    return storage + generation - capex


print("=== 1. Stand-alone LLP at the MPEC capacity")
embedded = {(r["node"], int(r["hour"])): float(r["price_eur_per_mwh"])
            for r in csv.DictReader((RUN / "embedded_prices.csv").open())}
llp = build_primal(data, chosen)
require_optimal(llp)
llp_price = {(n, t): llp.dual[llp.nodal_balance[n, t]] for n in data["nodes"] for t in data["times"]}
diff = sorted(((abs(llp_price[k] - embedded[k]), k) for k in embedded), reverse=True)
print(f"market cost LLP {pyo.value(llp.market_cost):.4f} vs MPEC {summary['embedded_market_cost_eur']:.4f}")
print(f"node-hours with |price difference| > 1e-3: {sum(d > 1e-3 for d, _ in diff)} of {len(diff)}")
for d, (n, t) in diff[:10]:
    if d > 1e-3:
        print(f"    {n} h{t:>2}: LLP {llp_price[n, t]:9.3f}  MPEC {embedded[n, t]:9.3f}")
print(f"I1 profit with LLP prices and LLP dispatch: {settlement(llp, llp_price, INV):.3f}"
      f"   (MPEC: {summary['profit_eur_per_day']:.3f})")
emb_dispatch = {r["coordinate"]: float(r["value"]) for r in csv.DictReader((RUN / "embedded_dispatch.csv").open())}
gaps = []
for i, n, t in llp.charge:
    if i == INV:
        gaps.append(abs(pyo.value(llp.charge[i, n, t]) - emb_dispatch[f"charge|{i}|{n}|{t}"]))
        gaps.append(abs(pyo.value(llp.discharge[i, n, t]) - emb_dispatch[f"discharge|{i}|{n}|{t}"]))
print(f"largest I1 charge/discharge difference LLP vs MPEC: {max(gaps):.4f} MW")

print("\n=== 2. I1 profit range over all optimal market prices, at scaled MPEC capacity")
for scale in (0.99, 0.999, 1.0, 1.001, 1.01):
    profile = copy.deepcopy(chosen)
    for field in ("power_mw", "energy_mwh"):
        profile[INV][field] = {n: v * scale for n, v in profile[INV][field].items()}
    _, dual, _ = face_models(data, profile)
    payoff = extrema(dual, dual_profit(dual, INV))
    print(f"    scale {scale:6.3f}: min {payoff['minimum']:10.2f}  max {payoff['maximum']:10.2f}"
          f"  width {payoff['width']:9.2f}")

print("\n=== 3. Planner benchmark: min market cost + I1 capex, I1 capacity variable (one LP)")
m = prepare_input(data, base_profile, active_investor=INV)
m.X_power = pyo.Var(m.investment_nodes, domain=pyo.NonNegativeReals)
m.X_energy = pyo.Var(m.investment_nodes, domain=pyo.NonNegativeReals)
m.investment_power_limit = pyo.Constraint(m.investment_nodes, rule=mpec.investment_power_limit_rule)
m.investment_energy_limit = pyo.Constraint(m.investment_nodes, rule=mpec.investment_energy_limit_rule)
m.duration_min = pyo.Constraint(m.investment_nodes, rule=mpec.duration_min_rule)
m.duration_max = pyo.Constraint(m.investment_nodes, rule=mpec.duration_max_rule)
m.power_capacity = pyo.Expression(m.storage_pairs, rule=mpec.power_capacity_rule)
m.energy_capacity = pyo.Expression(m.storage_pairs, rule=mpec.energy_capacity_rule)
add_primal(m)
m.capex = pyo.Expression(rule=mpec.investment_cost_rule)
m.objective = pyo.Objective(expr=m.market_cost + m.capex, sense=pyo.minimize)
require_optimal(m)
planner = copy.deepcopy(base_profile)
planner[INV]["power_mw"] = {n: pyo.value(m.X_power[n]) for n in m.investment_nodes}
planner[INV]["energy_mwh"] = {n: pyo.value(m.X_energy[n]) for n in m.investment_nodes}
mpec_cap = chosen[INV]
print(f"{'node':5s} {'planner MW':>11s} {'MWh':>9s} {'MPEC MW':>9s} {'MWh':>9s}")
for n in data["nodes"]:
    p, e = planner[INV]["power_mw"][n], planner[INV]["energy_mwh"][n]
    q, f = mpec_cap["power_mw"].get(n, 0.0), mpec_cap["energy_mwh"].get(n, 0.0)
    if max(p, e, q, f) > 1e-6:
        print(f"{n:5s} {p:11.2f} {e:9.2f} {q:9.2f} {f:9.2f}")
print(f"total {sum(planner[INV]['power_mw'].values()):11.2f} {sum(planner[INV]['energy_mwh'].values()):9.2f}"
      f" {sum(mpec_cap['power_mw'].values()):9.2f} {sum(mpec_cap['energy_mwh'].values()):9.2f}")
mpec_capex = sum(m.cost_power_daily[INV] * mpec_cap["power_mw"].get(n, 0) + m.cost_energy_daily[INV]
                 * mpec_cap["energy_mwh"].get(n, 0) for n in data["nodes"])
print(f"system cost (market + I1 capex): planner {pyo.value(m.objective):.2f},"
      f" at MPEC capacity {summary['embedded_market_cost_eur'] + mpec_capex:.2f}")
_, dual, _ = face_models(data, planner)
payoff = extrema(dual, dual_profit(dual, INV))
print(f"I1 profit at planner capacity over optimal prices: min {payoff['minimum']:.2f} max {payoff['maximum']:.2f}")
