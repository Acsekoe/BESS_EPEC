"""Warm start for the SOS1 MPEC (prototype; not used by model/run.py).

For each candidate capacity (MW at --node, duration --ratio): clear the market
exactly (unique prices, Ipopt), copy its dispatch into the MPEC, fix the zero
member of every SOS1 set from the market's active set, and solve the remaining
QP with capacities fixed. That gives a full KKT point in about 0.3 s. The best
candidate by exact profit is passed to Gurobi as a MIP start for the free MPEC.
Without it, Gurobi found no feasible point for I3/I4 (5-bus) in 600 s.

python scripts/diagnostics/active_set_start.py . --investor I3 --seconds 120 --candidates 0 60 90 120
"""
import argparse
import copy
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--data", type=Path, default=Path("model/input/ieee5_market_data.json"))
parser.add_argument("--profile", type=Path, default=Path("model/input/ieee5_capacities.json"))
parser.add_argument("--investor", default="I1")
parser.add_argument("--nodes", nargs="+", default=["N1", "N2", "N3", "N4", "N5"])
parser.add_argument("--node", default="N5", help="Candidate start capacities are placed here")
parser.add_argument("--candidates", type=float, nargs="+", default=[0.0], help="Start power candidates [MW]")
parser.add_argument("--ratio", type=float, default=2.8, help="Start duration [MWh per MW]")
parser.add_argument("--balancing-eps", type=float, default=1e-4)
parser.add_argument("--seconds", type=float, default=120)
args = parser.parse_args()
sys.path.insert(0, str(args.root / "model"))
import pyomo.environ as pyo
from prepare_input import load_case
from mpec import build_mpec
from solve import make_solver, objective_bounds, solve
from solution_space import settlement_profit, unique_price_reclear

data, profile = load_case(args.root / args.data, args.root / args.profile)
i = args.investor
m = build_mpec(data, profile, i, nodes=args.nodes, balancing_eps=args.balancing_eps)


def kkt_point(power):
    """Fix capacity, take the active set from a cleared market, solve the rest as a QP."""
    trial = copy.deepcopy(profile)
    trial[i]["power_mw"] = {args.node: power} if power > 0 else {}
    trial[i]["energy_mwh"] = {args.node: args.ratio * power} if power > 0 else {}
    market, price = unique_price_reclear(data, trial, args.balancing_eps)
    exact = settlement_profit(market, i, price)
    for n in args.nodes:
        m.X_power[n].fix(power if n == args.node else 0.0)
        m.X_energy[n].fix(args.ratio * power if n == args.node else 0.0)
    for variable in (m.charge, m.discharge, m.soc):  # storage absent from the market at zero capacity
        for v in variable.values():
            v.set_value(0.0)
    for v in market.component_data_objects(pyo.Var):  # dispatch has the same names and indices
        target = m.find_component(v.name)
        if target is not None:
            target.set_value(v.value, skip_validation=True)
    for (g, t), s in m.slack_gen_upper.items():
        s.set_value(pyo.value(m.generation_capacity[g, t] - m.generation[g, t]), skip_validation=True)
    for (n, t), s in m.slack_shed_upper.items():
        s.set_value(pyo.value(m.demand[n, t] - m.load_shed[n, t]), skip_validation=True)
    for (line, t), s in m.slack_line_upper.items():
        s.set_value(pyo.value(m.line_limit[line] - m.line_flow[line, t]), skip_validation=True)
        m.slack_line_lower[line, t].set_value(pyo.value(m.line_limit[line] + m.line_flow[line, t]), skip_validation=True)
    for (j, n, t), s in m.slack_power.items():
        s.set_value(pyo.value(m.power_capacity[j, n] - m.charge[j, n, t] - m.discharge[j, n, t]), skip_validation=True)
    for (j, n, t), s in m.slack_soc_upper.items():
        s.set_value(pyo.value(m.energy_capacity[j, n] - m.soc[j, n, t]), skip_validation=True)
    # Fix the zero member of every SOS1 set exactly (IntFeasTol 1e-9 rejects near-zeros).
    fixed = []
    for sos in m.component_data_objects(pyo.SOSConstraint):
        member, mu = [v for v, _ in sos.get_items()]
        zero = mu if (member.value or 0.0) > 1e-5 else member
        zero.fix(0.0)
        fixed.append(zero)
    for c in m.component_objects(pyo.SOSConstraint):
        c.deactivate()
    result = solve(m, seconds=60)
    ok = result.solver.termination_condition == pyo.TerminationCondition.optimal
    values = {v.name: v.value for v in m.component_data_objects(pyo.Var)} if ok else None
    linear = pyo.value(m.profit_linear) if ok else None
    for v in fixed:
        v.unfix()
    for c in m.component_objects(pyo.SOSConstraint):
        c.activate()
    return ok, exact, linear, values


best = None
for power in args.candidates:
    start = time.time()
    ok, exact, linear, values = kkt_point(power)
    print(f"{args.node} {power:6.1f} MW: exact profit {exact:.2f}, KKT point {'found' if ok else 'not found'}"
          f" (profit_linear {linear}), {time.time() - start:.1f} s", flush=True)
    if ok and (best is None or exact > best[1]):
        best = (power, exact, values)

for n in args.nodes:
    m.X_power[n].unfix()
    m.X_energy[n].unfix()
for v in m.component_data_objects(pyo.Var):
    v.set_value(best[2][v.name], skip_validation=True)
print(f"start: {args.node} {best[0]} MW ({best[1]:.2f})", flush=True)
solver = make_solver()
solver.options.update({"TimeLimit": args.seconds, "MIPGap": 1e-6, "NonConvex": 2, "PreSOS1BigM": 0, "IntFeasTol": 1e-9})
start = time.time()
result = solver.solve(m, warmstart=True, load_solutions=False)
incumbent, bound = objective_bounds(solver)
print(f"free MPEC with warm start: {result.solver.termination_condition}, incumbent {incumbent}, bound {bound}, "
      f"{time.time() - start:.1f} s", flush=True)
if solver._solver_model.SolCount > 0:
    x = solver._pyomo_var_to_solver_var_map
    print("best solution capacity:", {n: (round(x[m.X_power[n]].X, 2), round(x[m.X_energy[n]].X, 2))
                                      for n in args.nodes if x[m.X_power[n]].X > 1e-6})
