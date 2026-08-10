"""Common auction and spot settlement for the multi-investor auction EPEC."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pyomo.environ as pyo

_AUCTION_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _AUCTION_DIR.parent
_PRIMAL_DUAL_DIR = _MODEL_DIR / "Primal and dual problems"
for module_dir in (_MODEL_DIR, _PRIMAL_DUAL_DIR):
    if str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))

from config import GaussSeidelConfig
from nodal_access_auction_primal import (
    Bid,
    awarded_mw,
    build_primal as build_auction_primal,
    uniform_clearing_prices,
)
from primal_market_clearing_model import MarketData, value
from single_investor_mpec import (
    build_fixed_demand_primal_model,
    capital_recovery_factor,
    fixed_demand_reference_lambda,
)
from solver_utils import get_ipopt_solver


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _last_optimal_profit(history: list[dict[str, Any]], investor: str) -> float | None:
    for row in reversed(history):
        if row["investor"] == investor and row["termination"] == "optimal":
            return float(row["profit_eur_per_day"])
    return None


def _positive_raw_price_groups(
    state: Any,
    investors: tuple[str, ...],
    node: str,
    tolerance_mw: float,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for investor in investors:
        if state.quantity_mw[investor][node] <= tolerance_mw:
            continue
        price = state.price_eur_per_mw_day[investor][node]
        groups.setdefault(f"{price:.12g}", []).append(investor)
    return groups


def compute_joint_auction_settlement(
    data: MarketData,
    cfg: GaussSeidelConfig,
    state: Any,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Re-clear the final bids and settle every investor in one common spot LP."""

    from epec_diagonalization import four_investor_portfolio_profiles

    profiles_by_id = {
        profile.investor_id: profile for profile in four_investor_portfolio_profiles(data)
    }
    profiles = [profiles_by_id[investor] for investor in cfg.investor_order]
    limits = {node: cfg.node_limit_mw for node in data.nodes}
    bids = [
        Bid(
            investor=investor,
            node=node,
            quantity_mw=max(0.0, state.quantity_mw[investor][node]),
            price_eur_per_mw=max(0.0, state.price_eur_per_mw_day[investor][node]),
        )
        for investor in cfg.investor_order
        for node in data.nodes
        if state.quantity_mw[investor][node] > cfg.capacity_cleanup_tol_mw
    ]
    if not bids:
        bids = [Bid(cfg.investor_order[0], data.nodes[0], 0.0, 0.0)]

    priority_epsilon = (
        cfg.tie_break_epsilon_eur_per_mw_day
        if cfg.bid_price_tick_eur_per_mw_day > 0.0
        else 0.0
    )
    quadratic_epsilon = (
        cfg.auction_quadratic_epsilon_eur_per_mw2_day
        if cfg.payment_rule == "uniform"
        else 0.0
    )
    auction = build_auction_primal(bids, limits, priority_epsilon, quadratic_epsilon)
    auction_result = get_ipopt_solver(
        {
            "max_cpu_time": min(cfg.max_cpu_time, 60.0),
            "tol": min(cfg.solver_tol, 1.0e-9),
            "acceptable_tol": min(cfg.solver_tol, 1.0e-8),
        }
    ).solve(auction, tee=False)
    auction_termination = str(auction_result.solver.termination_condition)
    if auction_termination != "optimal":
        raise RuntimeError(
            f"Final independent auction did not solve optimally ({auction_termination})."
        )
    flat_awards = awarded_mw(auction)
    clearing_prices = (
        uniform_clearing_prices(auction)
        if cfg.payment_rule == "uniform"
        else {node: 0.0 for node in data.nodes}
    )
    awards = {
        investor: {
            node: max(0.0, flat_awards.get((investor, node), 0.0))
            for node in data.nodes
        }
        for investor in cfg.investor_order
    }
    energy = {
        investor: {
            node: awards[investor][node] * state.duration_hours[investor][node]
            for node in data.nodes
        }
        for investor in cfg.investor_order
    }
    max_award_reclear_difference = max(
        abs(awards[investor][node] - state.awards_mw[investor][node])
        for investor in cfg.investor_order
        for node in data.nodes
    )

    joint_data = replace(
        data,
        storage_units=list(cfg.investor_order),
        x_power={
            (investor, node): awards[investor][node]
            for investor in cfg.investor_order
            for node in data.nodes
        },
        x_energy={
            (investor, node): energy[investor][node]
            for investor in cfg.investor_order
            for node in data.nodes
        },
    )
    degradation_by_investor = {
        profile.investor_id: profile.degradation_eur_per_mwh for profile in profiles
    }
    spot = build_fixed_demand_primal_model(
        joint_data,
        storage_degradation_eur_per_mwh=degradation_by_investor,
        dispatch_regularization_eur_per_mw2h=0.0,
    )
    spot_result = get_ipopt_solver(
        {
            "max_cpu_time": cfg.max_cpu_time,
            "tol": cfg.solver_tol,
            "acceptable_tol": cfg.solver_tol,
        }
    ).solve(spot, tee=False)
    spot_termination = str(spot_result.solver.termination_condition)
    if spot_termination != "optimal":
        raise RuntimeError(f"Final joint spot market did not solve optimally ({spot_termination}).")
    lam = fixed_demand_reference_lambda(spot)
    generator_node = {
        generator: node
        for node in data.nodes
        for generator in data.generators_at_node.get(node, [])
    }

    investors_out: dict[str, dict[str, object]] = {}
    for profile in profiles:
        investor = profile.investor_id
        charge = sum(
            value(spot.P_charge[investor, node, time])
            for node in spot.N
            for time in spot.T
        )
        discharge = sum(
            value(spot.P_discharge[investor, node, time])
            for node in spot.N
            for time in spot.T
        )
        spot_revenue = sum(
            lam[node, time]
            * (
                value(spot.P_discharge[investor, node, time])
                - value(spot.P_charge[investor, node, time])
            )
            for node in spot.N
            for time in spot.T
        )
        generation_rent = 0.0
        for generator, share in profile.owned_generation_shares.items():
            node = generator_node.get(generator)
            if node is None or share == 0.0:
                continue
            generation_rent += share * sum(
                (lam[node, time] - data.generation_cost[generator])
                * value(spot.P_gen[generator, time])
                for time in spot.T
            )
        degradation_cost = 0.5 * profile.degradation_eur_per_mwh * (charge + discharge)
        daily_crf = capital_recovery_factor(profile.wacc, profile.lifetime_years) / 365.25
        capex = daily_crf * sum(
            profile.cost_power_eur_per_mw * awards[investor][node]
            + profile.cost_energy_eur_per_mwh * energy[investor][node]
            for node in data.nodes
        )
        if cfg.payment_rule == "uniform":
            access_payment = sum(
                clearing_prices[node] * awards[investor][node] for node in data.nodes
            )
        else:
            access_payment = sum(
                state.price_eur_per_mw_day[investor][node] * awards[investor][node]
                for node in data.nodes
            )
        settled_profit = (
            spot_revenue
            + generation_rent
            - degradation_cost
            - capex
            - access_payment
        )

        proportional_revenue = 0.0
        proportional_throughput = 0.0
        for node in spot.N:
            total_award = sum(awards[other][node] for other in cfg.investor_order)
            share = awards[investor][node] / total_award if total_award > 1.0e-9 else 0.0
            for time in spot.T:
                aggregate_net = sum(
                    value(spot.P_discharge[other, node, time])
                    - value(spot.P_charge[other, node, time])
                    for other in cfg.investor_order
                )
                aggregate_throughput = sum(
                    value(spot.P_discharge[other, node, time])
                    + value(spot.P_charge[other, node, time])
                    for other in cfg.investor_order
                )
                proportional_revenue += share * lam[node, time] * aggregate_net
                proportional_throughput += share * aggregate_throughput
        proportional_profit = (
            proportional_revenue
            + generation_rent
            - 0.5 * profile.degradation_eur_per_mwh * proportional_throughput
            - capex
            - access_payment
        )
        last_profit = _last_optimal_profit(history, investor)
        investors_out[investor] = {
            "wacc": profile.wacc,
            "owned_generation_shares": dict(profile.owned_generation_shares),
            "total_bid_quantity_mw": sum(state.quantity_mw[investor].values()),
            "total_awarded_power_mw": sum(awards[investor].values()),
            "total_energy_mwh": sum(energy[investor].values()),
            "settled_spot_revenue_eur_per_day": spot_revenue,
            "settled_generation_rent_eur_per_day": generation_rent,
            "settled_degradation_cost_eur_per_day": degradation_cost,
            "capex_eur_per_day": capex,
            "access_payment_eur_per_day": access_payment,
            "settled_profit_eur_per_day": settled_profit,
            "capacity_proportional_profit_eur_per_day": proportional_profit,
            "dispatch_attribution_band_eur_per_day": abs(
                settled_profit - proportional_profit
            ),
            "last_optimistic_best_response_profit_eur_per_day": last_profit,
            "optimistic_minus_settled_eur_per_day": (
                None if last_profit is None else last_profit - settled_profit
            ),
            "throughput_mwh": charge + discharge,
        }

    node_summary: dict[str, dict[str, object]] = {}
    all_raw_bids_unique = True
    all_effective_bids_unique = True
    all_positive_bids_on_tick = True
    max_node_overload = 0.0
    for node in data.nodes:
        raw_groups = _positive_raw_price_groups(
            state, cfg.investor_order, node, cfg.capacity_cleanup_tol_mw
        )
        duplicates = {price: ids for price, ids in raw_groups.items() if len(ids) > 1}
        all_raw_bids_unique = all_raw_bids_unique and not duplicates
        effective_groups: dict[str, list[str]] = {}
        for investor in cfg.investor_order:
            if state.quantity_mw[investor][node] <= cfg.capacity_cleanup_tol_mw:
                continue
            effective_price = (
                state.price_eur_per_mw_day[investor][node]
                + auction.priority_offset[investor]
            )
            effective_groups.setdefault(f"{effective_price:.12g}", []).append(investor)
        effective_duplicates = {
            price: ids for price, ids in effective_groups.items() if len(ids) > 1
        }
        all_effective_bids_unique = all_effective_bids_unique and not effective_duplicates
        if cfg.bid_price_tick_eur_per_mw_day > 0.0:
            tick = cfg.bid_price_tick_eur_per_mw_day
            all_positive_bids_on_tick = all_positive_bids_on_tick and all(
                abs(state.price_eur_per_mw_day[investor][node] / tick - round(
                    state.price_eur_per_mw_day[investor][node] / tick
                ))
                <= 1.0e-8
                for investor in cfg.investor_order
                if state.quantity_mw[investor][node] > cfg.capacity_cleanup_tol_mw
            )
        total_award = sum(awards[investor][node] for investor in cfg.investor_order)
        max_node_overload = max(max_node_overload, max(0.0, total_award - limits[node]))
        node_summary[node] = {
            "limit_mw": limits[node],
            "clearing_price_eur_per_mw_day": clearing_prices[node],
            "total_bid_quantity_mw": sum(
                state.quantity_mw[investor][node] for investor in cfg.investor_order
            ),
            "total_awarded_mw": total_award,
            "unused_access_mw": max(0.0, limits[node] - total_award),
            "raw_price_duplicate_groups": duplicates,
            "effective_price_duplicate_groups": effective_duplicates,
            "bids": {
                investor: {
                    "quantity_mw": state.quantity_mw[investor][node],
                    "raw_price_eur_per_mw_day": state.price_eur_per_mw_day[investor][node],
                    "award_mw": awards[investor][node],
                    "energy_mwh": energy[investor][node],
                }
                for investor in cfg.investor_order
            },
        }

    return {
        "auction_termination": auction_termination,
        "spot_termination": spot_termination,
        "payment_rule": (
            "uniform_clearing_price"
            if cfg.payment_rule == "uniform"
            else "pay_as_bid_raw_price"
        ),
        "auction_priority_epsilon_eur_per_mw_day": priority_epsilon,
        "auction_quadratic_epsilon_eur_per_mw2_day": quadratic_epsilon,
        "clearing_prices_eur_per_mw_day": dict(clearing_prices),
        "auction_effective_bid_value_eur_per_day": value(auction.bid_value),
        "total_access_payment_eur_per_day": sum(
            row["access_payment_eur_per_day"] for row in investors_out.values()
        ),
        "joint_spot_objective_eur_per_day": value(spot.objective),
        "lambda_min_eur_per_mwh": min(lam.values()),
        "lambda_max_eur_per_mwh": max(lam.values()),
        "max_stored_vs_recleared_award_difference_mw": max_award_reclear_difference,
        "max_node_overload_mw": max_node_overload,
        "auction_limit_feasible_within_solver_tolerance": max_node_overload <= max(
            1.0e-5, cfg.solver_tol
        ),
        "all_raw_positive_bid_prices_unique_by_node": all_raw_bids_unique,
        "all_effective_positive_bid_prices_unique_by_node": all_effective_bids_unique,
        "all_positive_bid_prices_on_tick": all_positive_bids_on_tick,
        "investors": investors_out,
        "nodes": node_summary,
        "awards_mw": awards,
        "energy_mwh": energy,
        "reference_lambda": lam,
        "reference_model": spot,
    }


def export_joint_auction_settlement(
    output_dir: Path,
    data: MarketData,
    cfg: GaussSeidelConfig,
    state: Any,
    history: list[dict[str, Any]],
    stop_reason: str,
    settlement: Mapping[str, Any],
) -> None:
    """Write consolidated four-investor auction and common spot results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    awards = settlement["awards_mw"]
    energy = settlement["energy_mwh"]
    clearing_prices = settlement["clearing_prices_eur_per_mw_day"]
    uniform = cfg.payment_rule == "uniform"
    _write_csv(
        output_dir / "final_bids_and_awards.csv",
        [
            {
                "investor": investor,
                "node": node,
                "bid_quantity_mw": state.quantity_mw[investor][node],
                "raw_bid_price_eur_per_mw_day": state.price_eur_per_mw_day[investor][node],
                "clearing_price_eur_per_mw_day": clearing_prices[node],
                "award_mw": awards[investor][node],
                "duration_hours": state.duration_hours[investor][node],
                "energy_mwh": energy[investor][node],
                "access_payment_eur_per_day": (
                    (
                        clearing_prices[node]
                        if uniform
                        else state.price_eur_per_mw_day[investor][node]
                    )
                    * awards[investor][node]
                ),
            }
            for investor in cfg.investor_order
            for node in data.nodes
        ],
    )
    spot = settlement["reference_model"]
    lam = settlement["reference_lambda"]
    _write_csv(
        output_dir / "joint_node_hour_prices.csv",
        [
            {
                "hour": time,
                "node": node,
                "lambda_joint_eur_per_mwh": lam[node, time],
            }
            for time in spot.T
            for node in spot.N
        ],
    )
    _write_csv(
        output_dir / "joint_storage_hour_operation.csv",
        [
            {
                "investor": investor,
                "hour": time,
                "node": node,
                "p_charge_mw": value(spot.P_charge[investor, node, time]),
                "p_discharge_mw": value(spot.P_discharge[investor, node, time]),
                "net_injection_mw": (
                    value(spot.P_discharge[investor, node, time])
                    - value(spot.P_charge[investor, node, time])
                ),
                "lambda_joint_eur_per_mwh": lam[node, time],
            }
            for investor in cfg.investor_order
            for time in spot.T
            for node in spot.N
        ],
    )
    public_settlement = {
        key: value_to_export
        for key, value_to_export in settlement.items()
        if key not in {"reference_model", "reference_lambda"}
    }
    (output_dir / "joint_settlement.json").write_text(
        json.dumps(public_settlement, indent=2), encoding="utf-8"
    )
    summary = {
        "converged": stop_reason == "converged",
        "stop_reason": stop_reason,
        "completed_iteration": state.completed_iteration,
        "update_rule": cfg.update_rule,
        "investor_order": list(cfg.investor_order),
        "active_nodes": None if cfg.active_nodes is None else list(cfg.active_nodes),
        "bid_price_tick_eur_per_mw_day": cfg.bid_price_tick_eur_per_mw_day,
        "tie_break_epsilon_eur_per_mw_day": cfg.tie_break_epsilon_eur_per_mw_day,
        "payment_rule": cfg.payment_rule,
        "auction_quadratic_epsilon_eur_per_mw2_day": (
            cfg.auction_quadratic_epsilon_eur_per_mw2_day
            if cfg.payment_rule == "uniform"
            else 0.0
        ),
        "clearing_price_tolerance_eur_per_mw_day": (
            cfg.clearing_price_tolerance_eur_per_mw_day
        ),
        "formulation": (
            f"four-investor Gauss-{cfg.update_rule.capitalize()} EPEC; every best response "
            "embeds auction and spot primal/dual/strong-duality systems"
        ),
        "final_common_reclear": public_settlement,
        "investor_response_count": len(history),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def print_joint_auction_summary(stop_reason: str, settlement: Mapping[str, Any]) -> None:
    """Print the all-investor outcome using common auction and spot clearing."""

    print("\nAuction EPEC common settlement")
    print(f"  status: {stop_reason}")
    if stop_reason != "converged":
        print("  WARNING: the final iterate is diagnostic; it is not a converged equilibrium.")
    print(f"  payment rule: {settlement['payment_rule']}")
    positive_clearing = {
        node: price
        for node, price in settlement["clearing_prices_eur_per_mw_day"].items()
        if price > 0.0
    }
    if positive_clearing:
        print(
            "  nodal access clearing prices: "
            + ", ".join(
                f"{node} {price:.2f} EUR/MW/day"
                for node, price in sorted(positive_clearing.items())
            )
        )
    print(
        "  joint lambda range: "
        f"{settlement['lambda_min_eur_per_mwh']:.4f} to "
        f"{settlement['lambda_max_eur_per_mwh']:.4f} EUR/MWh"
    )
    print(
        "  independent auction reclear difference: "
        f"{settlement['max_stored_vs_recleared_award_difference_mw']:.3e} MW"
    )
    for investor, row in settlement["investors"].items():
        gap = row["optimistic_minus_settled_eur_per_day"]
        gap_text = "n/a" if gap is None else f"{gap:+,.2f}"
        print(
            f"  {investor}: award {row['total_awarded_power_mw']:.3f} MW / "
            f"{row['total_energy_mwh']:.3f} MWh, access "
            f"{row['access_payment_eur_per_day']:,.2f} EUR/day, settled profit "
            f"{row['settled_profit_eur_per_day']:,.2f} EUR/day "
            f"(optimistic-settled {gap_text})"
        )
