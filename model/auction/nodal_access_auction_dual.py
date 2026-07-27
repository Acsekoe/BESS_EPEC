"""Small dual LP for the pay-as-bid nodal access auction."""

from __future__ import annotations

from typing import Iterable, Mapping

import pyomo.environ as pyo

from nodal_access_auction_primal import Bid, demo_bids, priority_offsets, validate_inputs
from solver_utils import get_ipopt_solver


def build_dual(
    bids: Iterable[Bid],
    limits: Mapping[str, float],
    tie_break_epsilon_eur_per_mw_day: float = 0.0,
    optimal_bid_value_by_node: Mapping[str, float] | None = None,
) -> pyo.ConcreteModel:
    """Build the auction dual, optionally selecting minimum supporting prices.

    Without ``optimal_bid_value_by_node`` this is the ordinary auction dual.
    When optimal primal values are supplied, the dual is restricted to that
    optimal face and minimizes the sum of nodal capacity duals. This implements
    the right-hand/minimum-shadow-price convention at degenerate capacity
    breakpoints.
    """
    data = validate_inputs(bids, limits)
    model = pyo.ConcreteModel(name="nodal_access_auction_dual")
    model.N = pyo.Set(initialize=list(data.limits), ordered=True)
    model.K = pyo.RangeSet(0, len(data.bids) - 1)
    model.bid = {k: bid for k, bid in enumerate(data.bids)}
    model.limit = data.limits
    model.priority_offset = priority_offsets(
        (bid.investor for bid in data.bids), tie_break_epsilon_eur_per_mw_day
    )
    model.capacity_dual = pyo.Var(model.N, domain=pyo.NonNegativeReals, initialize=0.0)
    model.quantity_dual = pyo.Var(model.K, domain=pyo.NonNegativeReals, initialize=0.0)
    model.dual_feasibility = pyo.Constraint(
        model.K,
        rule=lambda m, k: m.capacity_dual[m.bid[k].node] + m.quantity_dual[k]
        >= m.bid[k].price_eur_per_mw + m.priority_offset[m.bid[k].investor],
    )
    model.dual_cost_by_node = pyo.Expression(
        model.N,
        rule=lambda m, n: m.limit[n] * m.capacity_dual[n]
        + sum(
            m.bid[k].quantity_mw * m.quantity_dual[k]
            for k in m.K
            if m.bid[k].node == n
        ),
    )
    model.dual_cost = pyo.Expression(expr=sum(model.dual_cost_by_node[n] for n in model.N))
    if optimal_bid_value_by_node is None:
        model.objective = pyo.Objective(expr=model.dual_cost, sense=pyo.minimize)
    else:
        missing = set(data.limits) - set(optimal_bid_value_by_node)
        if missing:
            raise ValueError(f"Missing optimal auction values for nodes {sorted(missing)}")
        optimal_value = {n: float(optimal_bid_value_by_node[n]) for n in data.limits}
        optimal_face_tolerance = {
            n: 1.0e-7 * max(1.0, abs(optimal_value[n])) for n in data.limits
        }
        model.optimal_dual_face = pyo.Constraint(
            model.N,
            rule=lambda m, n: m.dual_cost_by_node[n]
            <= optimal_value[n] + optimal_face_tolerance[n],
        )
        model.objective = pyo.Objective(
            expr=sum(model.capacity_dual[n] for n in model.N),
            sense=pyo.minimize,
        )
    return model


def solve(model: pyo.ConcreteModel) -> str:
    result = get_ipopt_solver({"max_cpu_time": 60.0}).solve(model, tee=False)
    return str(result.solver.termination_condition)


def main() -> int:
    bids, limits = demo_bids()
    model = build_dual(bids, limits)
    termination = solve(model)
    print(f"Dual termination: {termination}")
    print(f"Dual objective: {pyo.value(model.dual_cost):,.2f} EUR/day")
    for node in model.N:
        print(f"  {node}: {pyo.value(model.capacity_dual[node]):.3f} EUR/MW/day")
    return 0 if termination == "optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
