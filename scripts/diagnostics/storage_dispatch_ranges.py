"""Is an investor's storage dispatch unique at fixed capacities?

Minimizes and maximizes each charge, discharge and SOC value of --investor over
all market solutions within --tolerance EUR of the optimal cost (primal QP with
balancing eps). `run.py ranges` needs the dual QP too, which at eps = 1e-4
fails its 1e-5 EUR primal-dual check; dispatch ranges need only the primal.
With a dispatch cost slope the near-optimal set is an ellipsoid: widths of
about 2 * sqrt(2 * tolerance / slope) remain even though the optimum is unique.

--degradation and --dispatch-slope override every investor's profile value.
Also prints the investor's unique-price profit (Ipopt reclear) and dispatch cost.

python scripts/diagnostics/storage_dispatch_ranges.py .
    --profile model/output/capex120_curtail_I1_pocket_gurobi_sos1_gap10/capacities.json
"""
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--data", type=Path, default=Path("model/input/market_data.json"))
parser.add_argument("--profile", type=Path, required=True)
parser.add_argument("--investor", default="I1")
parser.add_argument("--eps", type=float, default=1e-4)
parser.add_argument("--tolerance", type=float, default=1e-3, help="Allowed market cost above the optimum [EUR]")
parser.add_argument("--degradation", type=float, help="Override degradation_eur_per_mwh for every investor")
parser.add_argument("--dispatch-slope", type=float, help="Override dispatch_cost_slope_eur_per_mwh_per_mw")
args = parser.parse_args()
sys.path.insert(0, str(args.root / "model"))
import pyomo.environ as pyo
from prepare_input import load_case
from primal_llp import build_primal
from solve import require_optimal
from solution_space import extrema, settlement_profit, unique_price_reclear

data, profile = load_case(args.root / args.data, args.root / args.profile)
for config in profile.values():
    if args.degradation is not None:
        config["degradation_eur_per_mwh"] = args.degradation
    if args.dispatch_slope is not None:
        config["dispatch_cost_slope_eur_per_mwh_per_mw"] = args.dispatch_slope

reclear, price = unique_price_reclear(data, profile, args.eps)
i = args.investor
dispatch_cost = sum(0.5 * pyo.value(reclear.dispatch_slope[i]) * (pyo.value(reclear.charge[i, n, t]) ** 2
                                                                  + pyo.value(reclear.discharge[i, n, t]) ** 2)
                    for j, n in reclear.storage_pairs if j == i for t in reclear.timesteps)
print(f"{i} unique-price profit {settlement_profit(reclear, i, price):.2f} EUR/day, "
      f"dispatch cost {dispatch_cost:.2f} EUR/day")

primal = build_primal(data, profile, args.eps)
require_optimal(primal)
optimum = pyo.value(primal.market_cost)
for suffix in list(primal.component_objects(pyo.Suffix)):
    primal.del_component(suffix)  # the face constraint is quadratic: Gurobi gives no duals
primal.optimal_face = pyo.Constraint(expr=primal.market_cost <= optimum + args.tolerance)
rows = []
for name, variable in (("charge", primal.charge), ("discharge", primal.discharge), ("soc", primal.soc)):
    for index, v in variable.items():
        if index[0] == args.investor:
            r = extrema(primal, v)
            rows.append((r["width"], f"{name}{index}", r["minimum"], r["maximum"]))
wide = sorted((r for r in rows if r[0] is not None and r[0] > 0.1), reverse=True)
print(f"market cost {optimum:.4f}; {args.investor} storage values: {len(rows)}, width > 0.1: {len(wide)}, "
      f"unresolved: {sum(r[0] is None for r in rows)}, largest width: "
      f"{max((r[0] for r in rows if r[0] is not None), default=0):.3f}")
for width, name, low, high in wide[:25]:
    print(f"  {width:8.2f}  {name:28s} [{low:8.2f}, {high:8.2f}]")
