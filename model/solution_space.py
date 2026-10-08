"""Coordinate ranges on primal and dual optimal faces at fixed capacities."""
import pyomo.environ as pyo

from primal_llp import build_primal, dispatch_variables
from dual_llp import build_dual, dual_profit
from solve import make_solver, require_optimal, residual, solve


def face_models(data, profile, tolerance=1e-6):
    if tolerance < 0:
        raise ValueError("The objective tolerance must be nonnegative.")
    primal = build_primal(data, profile)
    dual = build_dual(data, profile)
    require_optimal(primal)
    require_optimal(dual)
    z = pyo.value(primal.market_cost)
    gap = z - pyo.value(dual.dual_value)
    if abs(gap) > 1e-5 or max(residual(primal), residual(dual)) > 1e-5:
        raise RuntimeError(f"Primal/dual market verification failed: gap={gap}.")
    primal.optimal_face = pyo.Constraint(expr=primal.market_cost <= z + tolerance)
    dual.optimal_face = pyo.Constraint(expr=dual.dual_value >= z - tolerance)
    return primal, dual, dict(market_cost_eur=z, primal_dual_gap_eur=gap,
                                   objective_tolerance_eur=tolerance)


def extrema(model, expression, solver=None):
    solver = solver or make_solver()
    # The faces are feasible, so report "unbounded" rather than "infeasibleOrUnbounded".
    solver.options["DualReductions"] = 0
    values, statuses = [], []
    for sense in (pyo.minimize, pyo.maximize):
        model.objective.set_value(expression)
        model.objective.sense = sense
        result = solve(model, solver=solver)
        status = str(result.solver.termination_condition)
        statuses.append(status)
        values.append(float(pyo.value(expression)) if status == "optimal" else None)
    return dict(minimum=values[0], maximum=values[1], minimum_status=statuses[0], maximum_status=statuses[1],
                width=values[1] - values[0] if None not in values else None)


def analyze(data, profile, *, tolerance=1e-6, dispatch="storage", progress=None):
    primal, dual, summary = face_models(data, profile, tolerance)
    output = []
    targets = [("price", f"{n}|{t}", dual.price[n, t], dual)
               for n in data["nodes"] for t in data["times"]]
    targets += [("profit", i, dual_profit(dual, i), dual) for i in profile]
    if dispatch != "none":
        targets += [("dispatch", name, variable, primal) for name, variable in dispatch_variables(primal)
                    if dispatch == "all" or name.startswith(("charge|", "discharge|", "soc|"))]
    for j, (kind, name, expression, model) in enumerate(targets):
        output.append(dict(kind=kind, coordinate=name, **extrema(model, expression)))
        if progress and (j + 1) % 100 == 0:
            progress(f"Completed {j + 1}/{len(targets)} coordinate ranges", flush=True)
    summary["range_count"] = len(output)
    summary["unresolved_or_unbounded_count"] = sum(r["width"] is None for r in output)
    summary["nonunique_price_count_at_1e-3"] = sum(r["kind"] == "price" and r["width"] is not None and r["width"] > 1e-3 for r in output)
    summary["interpretation"] = "Marginal ranges at one fixed capacity profile; endpoints need not be jointly attainable. Positive tolerance includes near-optimal solutions. This is not an equilibrium enumeration."
    return output, summary
