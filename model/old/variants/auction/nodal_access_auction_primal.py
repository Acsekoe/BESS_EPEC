"""Primal problem for continuous nodal access allocation.

With ``quadratic_epsilon_eur_per_mw2_day = 0`` this is the legacy pay-as-bid
allocation LP. With a positive epsilon the objective becomes strictly concave,
the allocation is unique and continuous in the bids, and exact price ties are
split symmetrically among uncapped bidders subject to their quantity limits.
The epsilon is an allocation regularizer only; settlement uses either the raw
bid (pay-as-bid) or the nodal clearing price (uniform rule).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pyomo.environ as pyo

_MODEL_DIR = Path(__file__).resolve().parent.parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.append(str(_MODEL_DIR))

from solver_utils import get_ipopt_solver


@dataclass(frozen=True)
class Bid:
    investor: str
    node: str
    quantity_mw: float
    price_eur_per_mw: float


@dataclass(frozen=True)
class AuctionInput:
    bids: tuple[Bid, ...]
    limits: dict[str, float]


def validate_inputs(bids: Iterable[Bid], limits: Mapping[str, float]) -> AuctionInput:
    data = AuctionInput(tuple(bids), {str(n): float(v) for n, v in limits.items()})
    if not data.bids or not data.limits:
        raise ValueError("At least one bid and one node limit are required.")
    if any(limit < 0.0 for limit in data.limits.values()):
        raise ValueError("Node limits must be nonnegative.")
    keys: set[tuple[str, str]] = set()
    for bid in data.bids:
        key = (bid.investor, bid.node)
        if key in keys:
            raise ValueError(f"Duplicate investor-node bid: {key}.")
        keys.add(key)
        if bid.node not in data.limits:
            raise ValueError(f"Unknown bid node: {bid.node}.")
        if bid.quantity_mw < 0.0 or bid.price_eur_per_mw < 0.0:
            raise ValueError("Bid quantities and prices must be nonnegative.")
    return data


def priority_offsets(investors: Iterable[str], epsilon: float) -> dict[str, float]:
    """Return deterministic sub-tick merit adders for unique auction ranking."""
    if epsilon < 0.0:
        raise ValueError("Tie-break epsilon must be nonnegative.")
    ordered = sorted(set(investors))
    return {
        investor: epsilon * (len(ordered) - rank)
        for rank, investor in enumerate(ordered)
    }


def build_primal(
    bids: Iterable[Bid],
    limits: Mapping[str, float],
    tie_break_epsilon_eur_per_mw_day: float = 0.0,
    quadratic_epsilon_eur_per_mw2_day: float = 0.0,
) -> pyo.ConcreteModel:
    """Maximize accepted bid value subject to bid and nodal quantity limits."""
    data = validate_inputs(bids, limits)
    if quadratic_epsilon_eur_per_mw2_day < 0.0:
        raise ValueError("Quadratic auction epsilon must be nonnegative.")
    model = pyo.ConcreteModel(name="nodal_access_auction_primal")
    model.N = pyo.Set(initialize=list(data.limits), ordered=True)
    model.K = pyo.RangeSet(0, len(data.bids) - 1)
    model.bid = {k: bid for k, bid in enumerate(data.bids)}
    model.limit = data.limits
    model.quadratic_epsilon = float(quadratic_epsilon_eur_per_mw2_day)
    model.priority_offset = priority_offsets(
        (bid.investor for bid in data.bids), tie_break_epsilon_eur_per_mw_day
    )
    model.award = pyo.Var(model.K, domain=pyo.NonNegativeReals, initialize=0.0)
    def node_limit_rule(m: pyo.ConcreteModel, node: str):
        node_bids = [k for k in m.K if m.bid[k].node == node]
        if not node_bids:
            return pyo.Constraint.Feasible
        return sum(m.award[k] for k in node_bids) <= m.limit[node]

    model.node_limit = pyo.Constraint(model.N, rule=node_limit_rule)
    model.bid_limit = pyo.Constraint(
        model.K,
        rule=lambda m, k: m.award[k] <= m.bid[k].quantity_mw,
    )
    model.bid_value_by_node = pyo.Expression(
        model.N,
        rule=lambda m, n: sum(
            (m.bid[k].price_eur_per_mw + m.priority_offset[m.bid[k].investor])
            * m.award[k]
            for k in m.K
            if m.bid[k].node == n
        )
    )
    model.bid_value = pyo.Expression(expr=sum(model.bid_value_by_node[n] for n in model.N))
    model.regularization = pyo.Expression(
        expr=0.5 * model.quadratic_epsilon * sum(model.award[k] ** 2 for k in model.K)
    )
    model.objective = pyo.Objective(
        expr=model.bid_value - model.regularization, sense=pyo.maximize
    )
    return model


def uniform_clearing_prices(
    model: pyo.ConcreteModel,
    award_tol_mw: float = 1.0e-5,
) -> dict[str, float]:
    """Deterministic nodal clearing price from a solved auction primal.

    The price is the KKT multiplier of the nodal capacity limit. It is pinned
    to ``effective_bid - epsilon * award`` whenever some award is strictly
    between zero and its bid quantity. On a degenerate face without an interior
    award, the highest-rejected-bid convention selects the lower end of the
    feasible multiplier interval, clipped so full awards stay consistent.
    """
    epsilon = float(getattr(model, "quadratic_epsilon", 0.0))
    prices: dict[str, float] = {}
    for node in model.N:
        node_bids = [k for k in model.K if model.bid[k].node == node]
        awards = {k: max(0.0, pyo.value(model.award[k])) for k in node_bids}
        if sum(awards.values()) < model.limit[node] - max(award_tol_mw, 1.0e-6):
            prices[node] = 0.0
            continue
        effective = {
            k: model.bid[k].price_eur_per_mw
            + model.priority_offset[model.bid[k].investor]
            for k in node_bids
        }
        interior = [
            k
            for k in node_bids
            if award_tol_mw < awards[k] < model.bid[k].quantity_mw - award_tol_mw
        ]
        if interior:
            price = sum(effective[k] - epsilon * awards[k] for k in interior) / len(interior)
        else:
            price = max(
                (
                    effective[k]
                    for k in node_bids
                    if awards[k] <= award_tol_mw and model.bid[k].quantity_mw > award_tol_mw
                ),
                default=0.0,
            )
        full_upper = min(
            (
                effective[k] - epsilon * awards[k]
                for k in node_bids
                if model.bid[k].quantity_mw > award_tol_mw
                and awards[k] >= model.bid[k].quantity_mw - award_tol_mw
            ),
            default=price,
        )
        prices[node] = max(0.0, min(price, full_upper))
    return prices


def solve(model: pyo.ConcreteModel) -> str:
    result = get_ipopt_solver({"max_cpu_time": 60.0}).solve(model, tee=False)
    return str(result.solver.termination_condition)


def awarded_mw(model: pyo.ConcreteModel) -> dict[tuple[str, str], float]:
    return {
        (model.bid[k].investor, model.bid[k].node): max(0.0, pyo.value(model.award[k]))
        for k in model.K
    }


def demo_bids() -> tuple[list[Bid], dict[str, float]]:
    return [
        Bid("I1", "N8", 70.0, 30.0),
        Bid("I2", "N8", 60.0, 20.0),
        Bid("I3", "N8", 50.0, 10.0),
        Bid("I1", "N3", 40.0, 5.0),
        Bid("I2", "N3", 30.0, 15.0),
        Bid("I3", "N3", 50.0, 8.0),
        Bid("I1", "N1", 20.0, 4.0),
        Bid("I2", "N1", 30.0, 3.0),
    ], {"N1": 100.0, "N3": 100.0, "N8": 100.0}


def main() -> int:
    bids, limits = demo_bids()
    model = build_primal(bids, limits)
    termination = solve(model)
    print(f"Primal termination: {termination}")
    print(f"Accepted bid value: {pyo.value(model.bid_value):,.2f} EUR/day")
    for (investor, node), award in sorted(awarded_mw(model).items()):
        if award > 1e-6:
            print(f"  {node} {investor}: {award:.3f} MW")
    return 0 if termination == "optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
