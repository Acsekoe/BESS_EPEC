"""Coordinate ranges on primal and dual optimal faces at fixed capacities, and
the unique-price reclear used with nodal balancing (eps > 0)."""
import pyomo.environ as pyo

from primal_llp import build_primal, dispatch_variables
from dual_llp import build_dual, dual_profit
from solve import make_nlp_solver, make_solver, require_optimal, residual, solve, solve_nlp


def unique_price_reclear(data, profile, balancing_eps, seconds=600):
    """With balancing, prices are unique: recover them from the primal QP as
    price = b/eps. The dual QP is not used: its curvature in price is only eps,
    so ordinary tolerances leave prices far from the unique ones (900 EUR/MWh
    off on IEEE-9 at a supply step). The QP is convex, so Ipopt's local
    optimum is global; tight tolerances pin b to about 1e-12 MW."""
    if balancing_eps <= 0:
        raise ValueError("The unique-price reclear needs balancing_eps > 0.")
    primal = build_primal(data, profile, balancing_eps)
    solver = make_nlp_solver()
    solver.options.update({"tol": 1e-12, "constr_viol_tol": 1e-12, "dual_inf_tol": 1e-12,
                           "compl_inf_tol": 1e-12})
    result = solve_nlp(primal, seconds=seconds, solver=solver)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Unique-price reclear not solved: {result.solver.termination_condition}")
    price = {(n, t): pyo.value(primal.balancing[n, t]) / balancing_eps
             for n in primal.nodes for t in primal.timesteps}
    return primal, price


def settlement_profit(m, investor, price):
    """Investor's nodal settlement at the given prices and m's dispatch: storage
    arbitrage less degradation, owned generation margin, less capex. At unique
    prices every optimal dispatch gives the same value (each asset is then a
    price taker), so this is the investor's unique profit."""
    storage = sum(price[n, t] * (pyo.value(m.discharge[i, n, t]) - pyo.value(m.charge[i, n, t]))
                  - 0.5 * pyo.value(m.degradation[i]) * (pyo.value(m.charge[i, n, t]) + pyo.value(m.discharge[i, n, t]))
                  for i, n in m.storage_pairs if i == investor for t in m.timesteps)
    generation = sum(pyo.value(m.owned_share[investor, g])
                     * (price[pyo.value(m.generator_node[g]), t] - pyo.value(m.generation_cost[g]))
                     * pyo.value(m.generation[g, t])
                     for g in m.generators for t in m.timesteps)
    capex = sum(pyo.value(m.cost_power_daily[investor] * m.installed_power[investor, n]
                          + m.cost_energy_daily[investor] * m.installed_energy[investor, n]) for n in m.nodes)
    return storage + generation - capex


def face_models(data, profile, tolerance=1e-6, balancing_eps=0.0):
    if tolerance < 0:
        raise ValueError("The objective tolerance must be nonnegative.")
    primal = build_primal(data, profile, balancing_eps)
    dual = build_dual(data, profile, balancing_eps)
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


def analyze(data, profile, *, tolerance=1e-6, dispatch="storage", progress=None, balancing_eps=0.0):
    primal, dual, summary = face_models(data, profile, tolerance, balancing_eps)
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
