"""Warm start for the SOS1 MPEC (prototype; not used by model/run.py).

For each candidate capacity: clear the market exactly (unique prices, Ipopt),
copy its dispatch into the MPEC, fix the zero member of every SOS1 set from the
market's active set, and solve the remaining QP with capacities fixed. That
gives a full KKT point in about 0.3 s. The best candidate by exact profit is
passed to Gurobi as a MIP start for the free MPEC. Without it, Gurobi found no
feasible point for I3/I4 (5-bus) in 600 s, or only the zero-storage point.

Candidates: --candidates MW at --node with duration --ratio, and/or the
capacities of --start-investor in each --start-profile (for example another
investor's exported MPEC optimum).

With --output DIR the best incumbent is checked like a run.py result (exact
unique-price reclear, settlement vs strong-duality profit, complementarity)
and exported as DIR/capacities.json and DIR/summary.json. A time-limited
incumbent is not a proven best response.

python scripts/diagnostics/active_set_start.py . --investor I3 --seconds 120 --candidates 0 60 90 120
python scripts/diagnostics/active_set_start.py . --data model/input/ieee5_v6_market_data.json --profile model/input/ieee5_v6_capacities.json --investor I3 --start-profile model/output/ieee5_v6_I1_all/capacities.json --start-investor I1 --seconds 600 --output model/output/<new_dir>
"""
import argparse
import copy
import json
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
parser.add_argument("--candidates", type=float, nargs="*", default=[0.0], help="Start power candidates [MW]")
parser.add_argument("--ratio", type=float, default=2.8, help="Start duration [MWh per MW]")
parser.add_argument("--start-profile", type=Path, nargs="*", default=[], help="capacities.json files with start candidates")
parser.add_argument("--start-investor", help="Investor whose capacities in --start-profile are used (default: --investor)")
parser.add_argument("--balancing-eps", type=float, default=1e-4)
parser.add_argument("--seconds", type=float, default=120)
parser.add_argument("--output", type=Path, help="Empty directory for the checked incumbent")
args = parser.parse_args()
sys.path.insert(0, str(args.root / "model"))
import pyomo.environ as pyo
from prepare_input import load_case
from mpec import build_mpec, chosen_profile, complementarity_pairs
from solve import make_solver, objective_bounds, residual, solve, solver_version
from solution_space import settlement_profit, unique_price_reclear

data, profile = load_case(args.root / args.data, args.root / args.profile)
i = args.investor
m = build_mpec(data, profile, i, nodes=args.nodes, balancing_eps=args.balancing_eps)

candidates = [(f"{args.node} {power:g} MW x {args.ratio:g} h", {args.node: (power, args.ratio * power)} if power > 0 else {})
              for power in args.candidates]
for path in args.start_profile:
    start = json.loads((args.root / path).read_text(encoding="utf-8"))[args.start_investor or i]
    candidates.append((f"{path} ({args.start_investor or i})",
                       {n: (start["power_mw"][n], start["energy_mwh"][n]) for n in start["power_mw"]
                        if n in args.nodes and start["power_mw"][n] > 0}))


def kkt_point(capacity):
    """Fix capacity {node: (MW, MWh)}, take the active set from a cleared market, solve the rest as a QP."""
    trial = copy.deepcopy(profile)
    trial[i]["power_mw"] = {n: c[0] for n, c in capacity.items()}
    trial[i]["energy_mwh"] = {n: c[1] for n, c in capacity.items()}
    market, price = unique_price_reclear(data, trial, args.balancing_eps)
    exact = settlement_profit(market, i, price)
    for n in args.nodes:
        m.X_power[n].fix(capacity.get(n, (0.0, 0.0))[0])
        m.X_energy[n].fix(capacity.get(n, (0.0, 0.0))[1])
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
    # Ipopt (tol 1e-12) leaves true zeros far below 1e-7; a coarser threshold (1e-5)
    # fixed small but nonzero quantities at a knife-edge capacity, and the QP became infeasible.
    fixed = []
    for sos in m.component_data_objects(pyo.SOSConstraint):
        member, mu = [v for v, _ in sos.get_items()]
        zero = mu if (member.value or 0.0) > 1e-7 else member
        zero.fix(0.0)
        fixed.append(zero)
    for c in m.component_objects(pyo.SOSConstraint):
        c.deactivate()
    # Dual simplex, not barrier: Gurobi rejects a MIP start that violates a row by more
    # than FeasibilityTol 1e-6, and barrier returned a point with 2e-6 violations for an
    # infeasible active set instead of reporting it (V6, I3 at I1's optimum).
    exact_qp = make_solver()
    exact_qp.options.update({"Method": 1, "FeasibilityTol": 1e-9, "OptimalityTol": 1e-9})
    result = solve(m, seconds=60, solver=exact_qp)
    ok = result.solver.termination_condition == pyo.TerminationCondition.optimal
    values = {v.name: v.value for v in m.component_data_objects(pyo.Var)} if ok else None
    linear = pyo.value(m.profit_linear) if ok else None
    for v in fixed:
        v.unfix()
    for c in m.component_objects(pyo.SOSConstraint):
        c.activate()
    return ok, exact, linear, values


best = None
for label, capacity in candidates:
    start = time.time()
    ok, exact, linear, values = kkt_point(capacity)
    print(f"{label}: exact profit {exact:.2f}, KKT point {'found' if ok else 'not found'}"
          f" (profit_linear {linear}), {time.time() - start:.1f} s", flush=True)
    if ok and (best is None or exact > best[1]):
        best = (label, exact, values)

for n in args.nodes:
    m.X_power[n].unfix()
    m.X_energy[n].unfix()
for v in m.component_data_objects(pyo.Var):
    v.set_value(best[2][v.name], skip_validation=True)
print(f"start: {best[0]} ({best[1]:.2f})", flush=True)
solver = make_solver()
solver.options.update({"TimeLimit": args.seconds, "MIPGap": 1e-6, "NonConvex": 2, "PreSOS1BigM": 0, "IntFeasTol": 1e-9})
if args.output:
    args.output.mkdir(parents=True, exist_ok=False)
    solver.options["LogFile"] = str(args.output / "gurobi.log")
start = time.time()
result = solver.solve(m, warmstart=True, load_solutions=False)
incumbent, bound = objective_bounds(solver)
status = str(result.solver.termination_condition)
print(f"free MPEC with warm start: {status}, incumbent {incumbent}, bound {bound}, {time.time() - start:.1f} s", flush=True)
if solver._solver_model.SolCount > 0:
    for v, gurobi_var in solver._pyomo_var_to_solver_var_map.items():  # load the incumbent, also when time-limited
        v.set_value(gurobi_var.X, skip_validation=True)
    selected = chosen_profile(m)
    print("best solution capacity:", {n: (round(selected[i]["power_mw"][n], 2), round(selected[i]["energy_mwh"][n], 2))
                                      for n in args.nodes if selected[i]["power_mw"][n] > 1e-6})
    if args.output:
        market, price = unique_price_reclear(data, selected, args.balancing_eps)
        pairs = list(complementarity_pairs(m))
        summary = dict(status=status, investor=i, solver="gurobi", solver_version=solver_version(),
                       data=str(args.data), profile=str(args.profile), balancing_eps=args.balancing_eps,
                       seconds=args.seconds, start=best[0], start_exact_profit_eur_per_day=best[1],
                       solver_lower_bound=incumbent, solver_upper_bound=bound,
                       profit_eur_per_day=pyo.value(m.profit), linear_profit_eur_per_day=pyo.value(m.profit_linear),
                       exact_reclear_profit_eur_per_day=settlement_profit(market, i, price),
                       maximum_constraint_violation=residual(m),
                       maximum_complementarity_product=max(abs(pyo.value(s * mu)) for s, mu in pairs),
                       interpretation="Warm-started MPEC incumbent; a best response only if status is optimal.")
        (args.output / "capacities.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
