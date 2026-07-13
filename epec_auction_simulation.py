"""Prototype auction-mediated two-investor BESS capacity game.

The driver supports two bid-generation modes:

* ``heuristic``: cheap rule-based bids for testing the full iterative chain.
* ``mpec``: one-shot investor bid models using primal feasibility, dual
  feasibility, and strong duality for the lower-level spot LP.

In both modes the realized chain is:

previous state -> sealed bids -> nodal auction -> spot LP -> profits.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import pyomo.environ as pyo

from auction import Bid, clear_pay_as_bid_auction
from investor_value_model import InvestorBidModelConfig, solve_investor_bid_model
from primal_market_clearing_model import (
    DEFAULT_DATA_PATH,
    MarketData,
    build_primal_market_clearing_model,
    get_solver,
    load_market_data,
    value,
)


DEFAULT_NODE_LIMIT_MW = 100.0
DEFAULT_EP_RATIO_HOURS = 2.0
DEFAULT_BESS_COST_POWER_EUR_PER_MW = 6_600.0
DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH = 18_800.0
DEFAULT_DEGRADATION_EUR_PER_MWH = 15.0
DEFAULT_LIFETIME_YEARS = 15


@dataclass(frozen=True)
class InvestorConfig:
    investor_id: str
    wacc: float
    max_request_mw_per_node: float = 70.0
    bid_fraction_of_estimated_value: float = 0.35
    max_bid_price_eur_per_mw: float = 50.0


@dataclass(frozen=True)
class InvestorProfit:
    spot_revenue_eur: float
    degradation_cost_eur: float
    capex_daily_eur: float
    access_payment_eur: float

    @property
    def profit_eur(self) -> float:
        return (
            self.spot_revenue_eur
            - self.degradation_cost_eur
            - self.capex_daily_eur
            - self.access_payment_eur
        )


@dataclass(frozen=True)
class MarketState:
    lmps: Mapping[tuple[str, int], float]
    objective_value: float


def capital_recovery_factor(wacc: float, lifetime_years: int = DEFAULT_LIFETIME_YEARS) -> float:
    return wacc * (1.0 + wacc) ** lifetime_years / ((1.0 + wacc) ** lifetime_years - 1.0)


def with_storage_capacities(
    data: MarketData,
    investor_ids: list[str],
    x_power: Mapping[tuple[str, str], float],
    ep_ratio_hours: float,
) -> MarketData:
    x_power_full = {
        (investor_id, node): float(x_power.get((investor_id, node), 0.0))
        for investor_id in investor_ids
        for node in data.nodes
    }
    x_energy_full = {
        (investor_id, node): ep_ratio_hours * power_mw
        for (investor_id, node), power_mw in x_power_full.items()
    }
    return replace(
        data,
        storage_units=list(investor_ids),
        x_power=x_power_full,
        x_energy=x_energy_full,
    )


def solve_market(data: MarketData, solver_name: str | None) -> tuple[pyo.ConcreteModel, MarketState]:
    model = build_primal_market_clearing_model(data)
    actual_solver_name, solver = get_solver(solver_name)
    print(f"Solving market LP with {actual_solver_name}...")
    results = solver.solve(model, tee=False)
    termination = results.solver.termination_condition
    if termination not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}:
        raise RuntimeError(f"Market solve failed with termination={termination}.")

    lmps: dict[tuple[str, int], float] = {}
    for node in model.N:
        for time in model.T:
            dual = model.dual.get(model.nodal_balance[node, time], None)
            if dual is None:
                raise RuntimeError("Solver did not return nodal-balance duals.")
            lmps[(str(node), int(time))] = float(dual)

    return model, MarketState(lmps=lmps, objective_value=value(model.objective))


def node_price_spread_value(
    state: MarketState,
    node: str,
    times: list[int],
    ep_ratio_hours: float,
) -> float:
    prices = [state.lmps[(node, time)] for time in times]
    return max(0.0, ep_ratio_hours * (max(prices) - min(prices)))


def capex_daily_per_mw(config: InvestorConfig, ep_ratio_hours: float) -> float:
    raw_cost_per_mw = (
        DEFAULT_BESS_COST_POWER_EUR_PER_MW
        + ep_ratio_hours * DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH
    )
    return capital_recovery_factor(config.wacc) * raw_cost_per_mw / 365.25


def create_bids(
    config: InvestorConfig,
    state: MarketState,
    nodes: list[str],
    times: list[int],
    ep_ratio_hours: float,
) -> list[Bid]:
    bids: list[Bid] = []
    daily_capex_per_mw = capex_daily_per_mw(config, ep_ratio_hours)

    for node in nodes:
        gross_value = node_price_spread_value(state, node, times, ep_ratio_hours)
        estimated_net_value = max(0.0, gross_value - daily_capex_per_mw)
        bid_price = min(
            config.max_bid_price_eur_per_mw,
            config.bid_fraction_of_estimated_value * estimated_net_value,
        )
        quantity = config.max_request_mw_per_node if bid_price > 1e-9 else 0.0
        bids.append(
            Bid(
                investor=config.investor_id,
                node=node,
                quantity_mw=quantity,
                price_eur_per_mw=bid_price,
            )
        )

    return bids


def compute_profit(
    model: pyo.ConcreteModel,
    config: InvestorConfig,
    access_payment_eur: float,
    ep_ratio_hours: float,
) -> InvestorProfit:
    data: MarketData = model._market_data
    investor_id = config.investor_id

    spot_revenue = 0.0
    degradation_cost = 0.0
    for node in model.N:
        for time in model.T:
            lmp = model.dual[model.nodal_balance[node, time]]
            charge = value(model.P_charge[investor_id, node, time])
            discharge = value(model.P_discharge[investor_id, node, time])
            spot_revenue += float(lmp) * (discharge - charge)
            degradation_cost += 0.5 * DEFAULT_DEGRADATION_EUR_PER_MWH * (charge + discharge)

    installed_cost = sum(
        DEFAULT_BESS_COST_POWER_EUR_PER_MW * data.x_power[investor_id, node]
        + DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH * data.x_energy[investor_id, node]
        for node in model.N
    )
    capex_daily = capital_recovery_factor(config.wacc) * installed_cost / 365.25

    return InvestorProfit(
        spot_revenue_eur=spot_revenue,
        degradation_cost_eur=degradation_cost,
        capex_daily_eur=capex_daily,
        access_payment_eur=access_payment_eur,
    )


def max_allocation_delta(
    old_allocations: Mapping[tuple[str, str], float],
    new_allocations: Mapping[tuple[str, str], float],
    investor_ids: list[str],
    nodes: list[str],
) -> float:
    return max(
        abs(new_allocations.get((investor_id, node), 0.0) - old_allocations.get((investor_id, node), 0.0))
        for investor_id in investor_ids
        for node in nodes
    )


def write_history(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "investor",
        "node",
        "bid_quantity_mw",
        "bid_price_eur_per_mw",
        "accepted_mw",
        "payment_eur",
        "spot_revenue_eur",
        "degradation_cost_eur",
        "capex_daily_eur",
        "access_payment_eur",
        "profit_eur",
        "bid_model_objective_eur",
        "market_objective_eur",
        "max_allocation_delta_mw",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_simulation(args: argparse.Namespace) -> int:
    base_data = load_market_data(args.data)
    investors = [
        InvestorConfig("I1", wacc=0.08, max_bid_price_eur_per_mw=args.max_bid_price_eur_per_mw),
        InvestorConfig("I2", wacc=0.12, max_bid_price_eur_per_mw=args.max_bid_price_eur_per_mw),
    ]
    investor_ids = [investor.investor_id for investor in investors]
    nodes = [str(node) for node in base_data.nodes]
    times = [int(time) for time in base_data.times]
    node_limits = {node: args.node_limit_mw for node in nodes}
    bid_model_configs = [
        InvestorBidModelConfig(
            investor_id=investor.investor_id,
            wacc=investor.wacc,
            max_request_mw_per_node=investor.max_request_mw_per_node,
            ep_ratio_hours=args.ep_ratio_hours,
            max_bid_price_eur_per_mw=args.max_bid_price_eur_per_mw,
        )
        for investor in investors
    ]

    allocations: dict[tuple[str, str], float] = {
        (investor_id, node): 0.0
        for investor_id in investor_ids
        for node in nodes
    }
    market_data = with_storage_capacities(base_data, investor_ids, allocations, args.ep_ratio_hours)
    _, state = solve_market(market_data, args.solver)

    history_rows: list[dict[str, float | int | str]] = []

    for iteration in range(1, args.iterations + 1):
        bid_model_objectives: dict[str, float] = {}
        if args.bidder == "heuristic":
            bids = [
                bid
                for investor in investors
                for bid in create_bids(investor, state, nodes, times, args.ep_ratio_hours)
            ]
        else:
            bid_results = [
                solve_investor_bid_model(
                    data=base_data,
                    config=config,
                    all_investor_ids=investor_ids,
                    fixed_competitor_x_power=allocations,
                    solver_name=args.investor_solver,
                )
                for config in bid_model_configs
            ]
            bids = [bid for result in bid_results for bid in result.bids]
            bid_model_objectives = {
                result.investor_id: result.objective_value
                for result in bid_results
            }
        auction_result = clear_pay_as_bid_auction(bids, node_limits)

        new_allocations = {
            (investor_id, node): auction_result.allocations_mw.get((investor_id, node), 0.0)
            for investor_id in investor_ids
            for node in nodes
        }
        delta = max_allocation_delta(allocations, new_allocations, investor_ids, nodes)

        market_data = with_storage_capacities(base_data, investor_ids, new_allocations, args.ep_ratio_hours)
        model, state = solve_market(market_data, args.solver)

        profits = {
            investor.investor_id: compute_profit(
                model=model,
                config=investor,
                access_payment_eur=auction_result.payments_eur.get(investor.investor_id, 0.0),
                ep_ratio_hours=args.ep_ratio_hours,
            )
            for investor in investors
        }
        bids_by_key = {(bid.investor, bid.node): bid for bid in bids}

        print(f"\nIteration {iteration}: max allocation delta = {delta:.3f} MW")
        for investor in investors:
            profit = profits[investor.investor_id]
            accepted_total = sum(new_allocations[investor.investor_id, node] for node in nodes)
            print(
                f"  {investor.investor_id}: accepted={accepted_total:7.2f} MW, "
                f"payment={profit.access_payment_eur:9.2f}, "
                f"spot={profit.spot_revenue_eur:9.2f}, "
                f"profit={profit.profit_eur:9.2f}"
            )
            if investor.investor_id in bid_model_objectives:
                print(f"      bid-model objective={bid_model_objectives[investor.investor_id]:9.2f}")

        for investor in investors:
            profit = profits[investor.investor_id]
            for node in nodes:
                bid = bids_by_key[(investor.investor_id, node)]
                accepted_mw = new_allocations[investor.investor_id, node]
                history_rows.append(
                    {
                        "iteration": iteration,
                        "investor": investor.investor_id,
                        "node": node,
                        "bid_quantity_mw": bid.quantity_mw,
                        "bid_price_eur_per_mw": bid.price_eur_per_mw,
                        "accepted_mw": accepted_mw,
                        "payment_eur": accepted_mw * bid.price_eur_per_mw,
                        "spot_revenue_eur": profit.spot_revenue_eur,
                        "degradation_cost_eur": profit.degradation_cost_eur,
                        "capex_daily_eur": profit.capex_daily_eur,
                        "access_payment_eur": profit.access_payment_eur,
                        "profit_eur": profit.profit_eur,
                        "bid_model_objective_eur": bid_model_objectives.get(investor.investor_id, 0.0),
                        "market_objective_eur": state.objective_value,
                        "max_allocation_delta_mw": delta,
                    }
                )

        allocations = new_allocations
        if delta <= args.tolerance_mw:
            print(f"\nConverged after {iteration} iterations.")
            break

    write_history(args.output, history_rows)
    print(f"\nWrote auction simulation history to {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-investor pay-as-bid BESS auction prototype.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--solver", default="appsi_highs")
    parser.add_argument("--bidder", choices=["heuristic", "mpec"], default="mpec")
    parser.add_argument("--investor-solver", default="ipopt", help="Solver for the nonlinear investor bid models.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="MPEC bidding is currently intended as a one-shot value-model test; heuristic bidding can be iterated.",
    )
    parser.add_argument("--tolerance-mw", type=float, default=0.1)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--ep-ratio-hours", type=float, default=DEFAULT_EP_RATIO_HOURS)
    parser.add_argument("--max-bid-price-eur-per-mw", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=Path("output/auction_simulation_history.csv"))
    return parser.parse_args()


def main() -> int:
    return run_simulation(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
