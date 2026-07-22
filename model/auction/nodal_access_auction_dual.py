"""Small dual LP for the pay-as-bid nodal access auction."""

from __future__ import annotations

from typing import Iterable, Mapping

import pyomo.environ as pyo

from nodal_access_auction_primal import Bid, demo_bids, validate_inputs
from solver_utils import get_ipopt_solver


def build_dual(bids: Iterable[Bid], limits: Mapping[str, float]) -> pyo.ConcreteModel:
    """Minimize nodal-capacity and bid-quantity opportunity cost."""
    data = validate_inputs(bids, limits)
    model = pyo.ConcreteModel(name="nodal_access_auction_dual")
    model.N = pyo.Set(initialize=list(data.limits), ordered=True)
    model.K = pyo.RangeSet(0, len(data.bids) - 1)
    model.bid = {k: bid for k, bid in enumerate(data.bids)}
    model.limit = data.limits
    model.capacity_dual = pyo.Var(model.N, domain=pyo.NonNegativeReals, initialize=0.0)
    model.quantity_dual = pyo.Var(model.K, domain=pyo.NonNegativeReals, initialize=0.0)
    model.dual_feasibility = pyo.Constraint(
        model.K,
        rule=lambda m, k: m.capacity_dual[m.bid[k].node] + m.quantity_dual[k]
        >= m.bid[k].price_eur_per_mw,
    )
    model.dual_cost = pyo.Expression(
        expr=sum(model.limit[n] * model.capacity_dual[n] for n in model.N)
        + sum(model.bid[k].quantity_mw * model.quantity_dual[k] for k in model.K)
    )
    model.objective = pyo.Objective(expr=model.dual_cost, sense=pyo.minimize)
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
