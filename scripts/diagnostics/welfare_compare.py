"""Compare capacity allocations by social cost, against a social planner.

Social cost = market cost (generation, curtailment, storage dispatch and
balancing cost, less the flexible demand's benefit) + every investor's daily
capex. Capacity and access payments are transfers and excluded. Each profile
is re-cleared with unique prices (balancing eps, Ipopt).

The planner chooses storage MW and MWh at every node to minimize the same
social cost, with one investor's cost parameters (--planner-costs, default I1,
WACC 8 %) and the hosting limits --hosting: a convex QP (Gurobi, proven). The
dual of each hosting limit is the social value of one more MW of hosting
capacity at that node [EUR/MW/day] (power limit plus ratio_max times the
energy limit it implies).

python scripts/diagnostics/welfare_compare.py . --hosting N5=100 --profiles "queue, owners first=model/output/scarce_N5/fcfs_owners_first/capacities.json"
"""
import argparse
import copy
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--data", type=Path, default=Path("model/input/five_bus_market_data.json"))
parser.add_argument("--profile", type=Path, default=Path("model/input/five_bus_capacities.json"))
parser.add_argument("--profiles", nargs="*", default=[], metavar="LABEL=PATH")
parser.add_argument("--hosting", nargs="*", default=[], metavar="NODE=MW")
parser.add_argument("--planner-costs", default="I1")
parser.add_argument("--balancing-eps", type=float, default=1e-4)
parser.add_argument("--output", type=Path)
args = parser.parse_args()

sys.path.insert(0, str(args.root / "model"))
import pyomo.environ as pyo  # noqa: E402
from capacity_clearing import exact_profits  # noqa: E402
from mpec import add_investment, chosen_profile  # noqa: E402
from prepare_input import load_case, prepare_input  # noqa: E402
from primal_llp import add_primal  # noqa: E402
from solution_space import unique_price_reclear  # noqa: E402
from solve import require_optimal  # noqa: E402

data, base = load_case(args.root / args.data, args.root / args.profile)
hosting = {k: float(v) for k, v in (item.split("=") for item in args.hosting)}
limits = {n: hosting.get(n, 1000.0) for n in data["nodes"]}


def evaluate(profile):
    market, price = unique_price_reclear(data, profile, args.balancing_eps)
    capex = sum(pyo.value(market.cost_power_daily[i] * market.installed_power[i, n]
                          + market.cost_energy_daily[i] * market.installed_energy[i, n])
                for i in market.investors for n in market.nodes)
    curtailed = {g: sum(pyo.value(market.curtailment[g, t]) for t in market.timesteps)
                 for g in market.generators if g.startswith("RES")}
    return dict(social_cost_eur=pyo.value(market.market_cost) + capex, market_cost_eur=pyo.value(market.market_cost),
                capex_eur=capex, curtailed_mwh=curtailed,
                flexible_demand_mwh=sum(pyo.value(v) for v in market.flexible_demand.values()),
                storage_mw={n: round(sum(c["power_mw"].get(n, 0.0) for c in profile.values()), 2) for n in data["nodes"]},
                storage_mwh={n: round(sum(c["energy_mwh"].get(n, 0.0) for c in profile.values()), 1) for n in data["nodes"]},
                market_profit={i: v["market_profit"] for i, v in exact_profits(data, profile, None, args.balancing_eps).items()})


# Planner: the cheapest investor's costs, every other investor at zero capacity.
empty = copy.deepcopy(base)
for c in empty.values():
    c["power_mw"], c["energy_mwh"] = {}, {}
m = prepare_input(data, empty, active_investor=args.planner_costs, node_limit=limits,
                  balancing_eps=args.balancing_eps, capacity_price={})
add_investment(m)
add_primal(m)
i = args.planner_costs
m.planner_capex = pyo.Expression(expr=sum(m.cost_power_daily[i] * m.X_power[n] + m.cost_energy_daily[i] * m.X_energy[n]
                                          for n in m.investment_nodes))
m.objective = pyo.Objective(expr=m.market_cost + m.planner_capex, sense=pyo.minimize)
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
require_optimal(m, seconds=600)
results = {"social planner": evaluate(chosen_profile(m))}
results["social planner"]["planner_objective_eur"] = pyo.value(m.objective)
# Minimization with X <= H: the dual is <= 0, so relaxing H by 1 MW saves -dual.
results["social planner"]["hosting_value_eur_per_mw_day"] = {
    n: -(m.dual[m.investment_power_limit[n]] + m.ratio_max[i] * m.dual[m.investment_energy_limit[n]])
    for n in hosting}
results["no storage"] = evaluate(empty)
for item in args.profiles:
    label, path = item.rsplit("=", 1)
    results[label] = evaluate(json.loads((args.root / path).read_text(encoding="utf-8")))

planner = results["social planner"]["social_cost_eur"]
print(f"{'allocation':32s} {'social cost':>12s} {'vs planner':>11s} {'N2 MW':>7s} {'N5 MW':>7s} "
      f"{'curt. N2':>9s} {'curt. N5':>9s}  market profits")
for label, r in results.items():
    profits = " ".join(f"{i} {v:.0f}" for i, v in r["market_profit"].items())
    print(f"{label:32s} {r['social_cost_eur']:12.0f} {r['social_cost_eur'] - planner:+11.0f} {r['storage_mw']['N2']:7.1f} "
          f"{r['storage_mw']['N5']:7.1f} {r['curtailed_mwh'].get('RES_Wind_N2', 0):9.1f} "
          f"{r['curtailed_mwh'].get('RES_PV_N5', 0):9.1f}  {profits}")
if args.output:
    args.output.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
