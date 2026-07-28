"""Gauss-Jacobi or Gauss-Seidel diagonalization of the clean auction EPEC.

In Jacobi mode every investor solves against one frozen bid snapshot, all bid
proposals are applied simultaneously, and one common auction determines the
new awards. Seidel mode applies each response immediately. Continuous mode uses
one solve per investor. Exact tick mode enumerates economically distinct
fixed-price candidates with deterministic sub-tick merit priority. There is no
damping, outside-option re-solve, or continuous-price rounding.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pyomo.environ as pyo

_AUCTION_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _AUCTION_DIR.parent
_PRIMAL_DUAL_DIR = _MODEL_DIR / "Primal and dual problems"
for module_dir in (_MODEL_DIR, _PRIMAL_DUAL_DIR):
    if str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))

from config import GaussSeidelConfig, parse_gauss_seidel_cli
from auction_epec_results import (
    compute_joint_auction_settlement,
    export_joint_auction_settlement,
    print_joint_auction_summary,
)
from nodal_access_auction_primal import (
    Bid,
    awarded_mw,
    build_primal as build_auction_primal,
    uniform_clearing_prices,
)
from primal_market_clearing_model import MarketData, load_market_data, value
from single_investor_auction_mpec import (
    RivalBid,
    build_one_leader_two_follower_mpec,
    initialize_from_independent_followers,
    load_rival_bid_vector,
    pay_as_bid_tick_candidates,
    solve_mpec,
    summarize,
    tick_candidate_diagnostic,
)
from solver_utils import get_ipopt_solver


@dataclass
class StrategyState:
    quantity_mw: dict[str, dict[str, float]]
    price_eur_per_mw_day: dict[str, dict[str, float]]
    duration_hours: dict[str, dict[str, float]]
    awards_mw: dict[str, dict[str, float]]
    completed_iteration: int = 0


@dataclass(frozen=True)
class SolveRecord:
    termination: str
    seconds: float
    model: pyo.ConcreteModel
    summary: dict[str, object]
    selected_tick_price_eur_per_mw_day: float | None = None
    tick_candidate_diagnostics: tuple[dict[str, object], ...] = ()


def _jsonable(value_to_convert: Any) -> Any:
    if isinstance(value_to_convert, Path):
        return str(value_to_convert)
    if isinstance(value_to_convert, tuple):
        return [_jsonable(item) for item in value_to_convert]
    if isinstance(value_to_convert, dict):
        return {str(key): _jsonable(item) for key, item in value_to_convert.items()}
    if isinstance(value_to_convert, list):
        return [_jsonable(item) for item in value_to_convert]
    return value_to_convert


def validate_config(cfg: GaussSeidelConfig) -> None:
    if cfg.update_rule not in {"jacobi", "seidel"}:
        raise ValueError("Update rule must be 'jacobi' or 'seidel'.")
    if len(set(cfg.investor_order)) != len(cfg.investor_order):
        raise ValueError("Investor order contains duplicates.")
    if len(cfg.investor_order) < 2:
        raise ValueError("At least two investors are required for auction competition.")
    if cfg.node_limit_mw <= 0.0 or cfg.max_bid_price_eur_per_mw_day <= 0.0:
        raise ValueError("Nodal limit and maximum bid price must be positive.")
    if cfg.max_iterations <= 0 or cfg.max_cpu_time <= 0.0:
        raise ValueError("Iteration and CPU-time limits must be positive.")
    if cfg.price_bound_eur_per_mwh <= 0.0 or cfg.dual_bound_eur_per_mwh <= 0.0:
        raise ValueError("Price and internal-dual bounds must be positive.")
    if cfg.solver_tol <= 0.0 or cfg.capacity_cleanup_tol_mw < 0.0:
        raise ValueError("Solver tolerance must be positive and cleanup tolerance nonnegative.")
    if not 0.0 < cfg.zero_bid_numerical_quantity_mw <= cfg.node_limit_mw:
        raise ValueError(
            "Zero-bid numerical quantity must be positive and no greater than the nodal limit."
        )
    if not 0.0 < cfg.zero_bid_numerical_price_eur_per_mw_day <= cfg.max_bid_price_eur_per_mw_day:
        raise ValueError(
            "Zero-bid numerical price must be positive and no greater than the bid-price cap."
        )
    if min(
        cfg.quantity_tolerance_mw,
        cfg.award_tolerance_mw,
        cfg.price_tolerance_eur_per_mw_day,
        cfg.duration_tolerance_hours,
    ) < 0.0:
        raise ValueError("Convergence tolerances must be nonnegative.")
    if cfg.max_consecutive_failures <= 0:
        raise ValueError("Maximum consecutive failures must be positive.")
    if cfg.bid_price_tick_eur_per_mw_day < 0.0:
        raise ValueError("Bid-price tick must be nonnegative.")
    if cfg.tie_break_epsilon_eur_per_mw_day < 0.0:
        raise ValueError("Tie-break epsilon must be nonnegative.")
    if cfg.payment_rule not in {"uniform", "pay_as_bid"}:
        raise ValueError("Payment rule must be 'uniform' or 'pay_as_bid'.")
    if cfg.payment_rule == "uniform":
        if cfg.auction_quadratic_epsilon_eur_per_mw2_day <= 0.0:
            raise ValueError("The uniform payment rule requires a positive quadratic epsilon.")
        if cfg.bid_price_tick_eur_per_mw_day > 0.0:
            raise ValueError(
                "Exact tick enumeration is a pay-as-bid device; use --payment-rule pay_as_bid."
            )
    if cfg.clearing_price_tolerance_eur_per_mw_day < 0.0:
        raise ValueError("Clearing-price tolerance must be nonnegative.")
    if cfg.bid_price_tick_eur_per_mw_day > 0.0:
        if cfg.active_nodes is None or len(cfg.active_nodes) != 1:
            raise ValueError("Exact tick enumeration requires exactly one active node.")
        if cfg.tie_break_epsilon_eur_per_mw_day <= 0.0:
            raise ValueError("Tick enumeration requires a positive merit offset.")
        maximum_priority_gap = (
            cfg.tie_break_epsilon_eur_per_mw_day * (len(cfg.investor_order) - 1)
        )
        if maximum_priority_gap >= cfg.bid_price_tick_eur_per_mw_day:
            raise ValueError("The maximum priority gap must be smaller than one bid tick.")
        if cfg.price_tolerance_eur_per_mw_day >= cfg.bid_price_tick_eur_per_mw_day:
            raise ValueError("Grid-mode price tolerance must be smaller than one bid tick.")


def _empty_nested(investors: tuple[str, ...], nodes: Any) -> dict[str, dict[str, float]]:
    return {investor: {node: 0.0 for node in nodes} for investor in investors}


def _cleanup_strategy(state: StrategyState, cfg: GaussSeidelConfig) -> None:
    active_nodes = None if cfg.active_nodes is None else set(cfg.active_nodes)
    for investor in cfg.investor_order:
        for node in state.quantity_mw[investor]:
            outside_scope = active_nodes is not None and node not in active_nodes
            negligible = state.quantity_mw[investor][node] <= cfg.capacity_cleanup_tol_mw
            if outside_scope or negligible:
                state.quantity_mw[investor][node] = 0.0
                state.price_eur_per_mw_day[investor][node] = 0.0
                state.awards_mw[investor][node] = 0.0


def initial_state(
    data: MarketData,
    investor_order: tuple[str, ...],
    initial_bids_path: Path,
    cfg: GaussSeidelConfig,
) -> StrategyState:
    records = load_rival_bid_vector(initial_bids_path)
    indexed = {(bid.investor, bid.node): bid for bid in records}
    expected = {(investor, node) for investor in investor_order for node in data.nodes}
    missing = expected - set(indexed)
    if missing:
        raise ValueError(f"Initial bid file is missing investor-node records: {sorted(missing)}")
    state = StrategyState(
        quantity_mw={
            investor: {node: float(indexed[investor, node].quantity_mw) for node in data.nodes}
            for investor in investor_order
        },
        price_eur_per_mw_day={
            investor: {node: float(indexed[investor, node].price_eur_per_mw_day) for node in data.nodes}
            for investor in investor_order
        },
        duration_hours={
            investor: {node: float(indexed[investor, node].duration_hours) for node in data.nodes}
            for investor in investor_order
        },
        awards_mw=_empty_nested(investor_order, data.nodes),
    )
    _cleanup_strategy(state, cfg)
    state.awards_mw, _ = clear_common_auction(state, data, cfg)
    return state


def state_from_checkpoint(path: Path, data: MarketData, cfg: GaussSeidelConfig) -> StrategyState:
    raw = json.loads(path.read_text(encoding="utf-8"))
    state_raw = raw["state"]
    state = StrategyState(
        quantity_mw={i: {n: float(v) for n, v in row.items()} for i, row in state_raw["quantity_mw"].items()},
        price_eur_per_mw_day={
            i: {n: float(v) for n, v in row.items()}
            for i, row in state_raw["price_eur_per_mw_day"].items()
        },
        duration_hours={
            i: {n: float(v) for n, v in row.items()}
            for i, row in state_raw["duration_hours"].items()
        },
        awards_mw={
            i: {n: float(v) for n, v in row.items()}
            for i, row in state_raw["awards_mw"].items()
        },
        completed_iteration=int(state_raw["completed_iteration"]),
    )
    if set(state.quantity_mw) != set(cfg.investor_order):
        raise ValueError("Checkpoint investors do not match configured investor order.")
    for investor in cfg.investor_order:
        if set(state.quantity_mw[investor]) != set(data.nodes):
            raise ValueError(f"Checkpoint nodes do not match market data for {investor}.")
    _cleanup_strategy(state, cfg)
    return state


def state_bids(state: StrategyState, data: MarketData, cleanup_tol_mw: float) -> list[Bid]:
    bids = [
        Bid(
            investor,
            node,
            max(0.0, state.quantity_mw[investor][node]),
            max(0.0, state.price_eur_per_mw_day[investor][node]),
        )
        for investor in state.quantity_mw
        for node in data.nodes
        if state.quantity_mw[investor][node] > cleanup_tol_mw
    ]
    if not bids:
        first_investor = next(iter(state.quantity_mw))
        bids.append(Bid(first_investor, data.nodes[0], 0.0, 0.0))
    return bids


def effective_quadratic_epsilon(cfg: GaussSeidelConfig) -> float:
    return (
        cfg.auction_quadratic_epsilon_eur_per_mw2_day
        if cfg.payment_rule == "uniform"
        else 0.0
    )


def clear_common_auction(
    state: StrategyState,
    data: MarketData,
    cfg: GaussSeidelConfig,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Clear the complete bid vector; return common awards and clearing prices."""
    limits = {node: cfg.node_limit_mw for node in data.nodes}
    auction = build_auction_primal(
        state_bids(state, data, cfg.capacity_cleanup_tol_mw),
        limits,
        cfg.tie_break_epsilon_eur_per_mw_day
        if cfg.bid_price_tick_eur_per_mw_day > 0.0
        else 0.0,
        effective_quadratic_epsilon(cfg),
    )
    solver = get_ipopt_solver(
        {
            "max_cpu_time": min(cfg.max_cpu_time, 60.0),
            "tol": min(cfg.solver_tol, 1.0e-9),
            "acceptable_tol": min(cfg.solver_tol, 1.0e-8),
        }
    )
    termination = str(solver.solve(auction, tee=False).solver.termination_condition)
    if termination != "optimal":
        raise RuntimeError(f"Common auction clearing failed: {termination}")
    flat = awarded_mw(auction)
    awards = {
        investor: {
            node: (
                max(0.0, flat.get((investor, node), 0.0))
                if flat.get((investor, node), 0.0) > cfg.capacity_cleanup_tol_mw
                else 0.0
            )
            for node in data.nodes
        }
        for investor in state.quantity_mw
    }
    clearing = (
        uniform_clearing_prices(auction)
        if cfg.payment_rule == "uniform"
        else {node: 0.0 for node in data.nodes}
    )
    return awards, clearing


def rivals_from_state(
    state: StrategyState,
    active_investor: str,
    data: MarketData,
    cleanup_tol_mw: float,
) -> list[RivalBid]:
    return [
        RivalBid(
            investor=investor,
            node=node,
            quantity_mw=state.quantity_mw[investor][node],
            price_eur_per_mw_day=state.price_eur_per_mw_day[investor][node],
            duration_hours=state.duration_hours[investor][node],
        )
        for investor in state.quantity_mw
        if investor != active_investor
        for node in data.nodes
        if state.quantity_mw[investor][node] > cleanup_tol_mw
    ]


def _solve_response_at_price(
    data: MarketData,
    cfg: GaussSeidelConfig,
    state: StrategyState,
    investor_profile: Any,
    fixed_bid_price: float | None = None,
) -> SolveRecord:
    investor = investor_profile.investor_id
    numerical_initial_quantity = {
        node: (
            state.quantity_mw[investor][node]
            if state.quantity_mw[investor][node] > cfg.capacity_cleanup_tol_mw
            else cfg.zero_bid_numerical_quantity_mw
        )
        for node in data.nodes
    }
    numerical_initial_price = {
        node: (
            state.price_eur_per_mw_day[investor][node]
            if state.quantity_mw[investor][node] > cfg.capacity_cleanup_tol_mw
            else cfg.zero_bid_numerical_price_eur_per_mw_day
        )
        for node in data.nodes
    }
    model = build_one_leader_two_follower_mpec(
        data,
        rivals_from_state(state, investor, data, cfg.capacity_cleanup_tol_mw),
        active_investor=investor_profile,
        active_nodes=cfg.active_nodes,
        node_limit_mw=cfg.node_limit_mw,
        max_bid_price_eur_per_mw_day=cfg.max_bid_price_eur_per_mw_day,
        initial_bid_quantity_mw=numerical_initial_quantity,
        initial_bid_price_eur_per_mw_day=numerical_initial_price,
        initial_duration_hours=state.duration_hours[investor],
        price_bound_eur_per_mwh=cfg.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=cfg.dual_bound_eur_per_mwh,
        capacity_cleanup_tol_mw=cfg.capacity_cleanup_tol_mw,
        tie_break_epsilon_eur_per_mw_day=(
            cfg.tie_break_epsilon_eur_per_mw_day
            if cfg.bid_price_tick_eur_per_mw_day > 0.0
            else 0.0
        ),
        payment_rule=cfg.payment_rule,
        auction_quadratic_epsilon_eur_per_mw2_day=effective_quadratic_epsilon(cfg),
    )
    if fixed_bid_price is not None:
        model.active_bid_price[cfg.active_nodes[0]].fix(fixed_bid_price)
    initialize_from_independent_followers(model)
    started = time.perf_counter()
    termination = solve_mpec(model, cfg.max_cpu_time, cfg.solver_tol, tee=cfg.tee)
    seconds = time.perf_counter() - started
    summary = summarize(model, termination)
    summary["solve_seconds"] = seconds
    return SolveRecord(termination, seconds, model, summary)


def solve_response_model(
    data: MarketData,
    cfg: GaussSeidelConfig,
    state: StrategyState,
    investor_profile: Any,
) -> SolveRecord:
    """Solve one continuous response or enumerate exact tick-grid prices."""
    if cfg.bid_price_tick_eur_per_mw_day <= 0.0:
        return _solve_response_at_price(data, cfg, state, investor_profile)

    started = time.perf_counter()
    investor = investor_profile.investor_id
    rivals = rivals_from_state(state, investor, data, cfg.capacity_cleanup_tol_mw)
    prices = pay_as_bid_tick_candidates(
        active_investor=investor,
        active_node=cfg.active_nodes[0],
        rivals=rivals,
        max_bid_price_eur_per_mw_day=cfg.max_bid_price_eur_per_mw_day,
        bid_price_tick_eur_per_mw_day=cfg.bid_price_tick_eur_per_mw_day,
        tie_break_epsilon_eur_per_mw_day=cfg.tie_break_epsilon_eur_per_mw_day,
    )
    records: list[tuple[SolveRecord, dict[str, object]]] = []
    for price in prices:
        print(f"    {investor} tick candidate: {price:.12g} EUR/MW/day")
        response = _solve_response_at_price(
            data, cfg, state, investor_profile, fixed_bid_price=price
        )
        diagnostic = tick_candidate_diagnostic(response.summary, response.seconds, price)
        print(
            f"      termination={response.termination}, valid={diagnostic['valid']}, "
            f"profit={response.summary['profit_eur_per_day']:,.2f} EUR/day"
        )
        records.append((response, diagnostic))
    valid_records = [record for record in records if record[1]["valid"]]
    selected_pool = valid_records or records
    selected_response, selected_diagnostic = max(
        selected_pool, key=lambda record: record[0].summary["profit_eur_per_day"]
    )
    termination = selected_response.termination if valid_records else "no_valid_tick_candidate"
    return SolveRecord(
        termination=termination,
        seconds=time.perf_counter() - started,
        model=selected_response.model,
        summary=selected_response.summary,
        selected_tick_price_eur_per_mw_day=(
            selected_diagnostic["fixed_bid_price_eur_per_mw_day"]
            if valid_records
            else None
        ),
        tick_candidate_diagnostics=tuple(record[1] for record in records),
    )


def apply_response(
    state: StrategyState,
    investor: str,
    response: SolveRecord,
    data: MarketData,
    cfg: GaussSeidelConfig,
    *,
    update_awards: bool = True,
    reference_awards: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[str, dict[str, float]]:
    if response.termination != "optimal":
        return "retain_after_solver_failure", {
            "max_quantity_change_mw": 0.0,
            "max_price_change_eur_per_mw_day": 0.0,
            "max_duration_change_hours": 0.0,
            "max_award_change_mw": 0.0,
        }

    old_quantity = dict(state.quantity_mw[investor])
    old_price = dict(state.price_eur_per_mw_day[investor])
    old_duration = dict(state.duration_hours[investor])
    old_awards = (
        {i: dict(row) for i, row in reference_awards.items()}
        if reference_awards is not None
        else {i: dict(row) for i, row in state.awards_mw.items()}
    )
    model = response.model
    for node in data.nodes:
        quantity = max(0.0, value(model.active_bid_quantity[node]))
        price = max(0.0, value(model.active_bid_price[node]))
        award = max(0.0, value(model.award[investor, node]))
        if quantity <= cfg.capacity_cleanup_tol_mw:
            quantity = 0.0
            price = 0.0
            award = 0.0
        state.quantity_mw[investor][node] = quantity
        state.price_eur_per_mw_day[investor][node] = price
        if award > cfg.capacity_cleanup_tol_mw:
            duration = value(model.active_energy[node]) / award
            state.duration_hours[investor][node] = min(
                max(duration, model._active_investor.ratio_min),
                model._active_investor.ratio_max,
            )

    embedded_awards = _empty_nested(tuple(state.quantity_mw), data.nodes)
    for awarded_investor, node in model.A:
        embedded_awards[awarded_investor][node] = max(
            0.0, value(model.award[awarded_investor, node])
        )
    if update_awards:
        state.awards_mw = embedded_awards
    _cleanup_strategy(state, cfg)
    return "candidate", {
        "max_quantity_change_mw": max(
            abs(state.quantity_mw[investor][n] - old_quantity[n]) for n in data.nodes
        ),
        "max_price_change_eur_per_mw_day": max(
            abs(state.price_eur_per_mw_day[investor][n] - old_price[n]) for n in data.nodes
        ),
        "max_duration_change_hours": max(
            abs(state.duration_hours[investor][n] - old_duration[n]) for n in data.nodes
        ),
        "max_award_change_mw": max(
            abs(embedded_awards[i][n] - old_awards[i][n])
            for i in embedded_awards
            for n in data.nodes
        ),
    }


def copy_state(state: StrategyState) -> StrategyState:
    return StrategyState(
        quantity_mw={i: dict(row) for i, row in state.quantity_mw.items()},
        price_eur_per_mw_day={i: dict(row) for i, row in state.price_eur_per_mw_day.items()},
        duration_hours={i: dict(row) for i, row in state.duration_hours.items()},
        awards_mw={i: dict(row) for i, row in state.awards_mw.items()},
        completed_iteration=state.completed_iteration,
    )


def max_award_difference(
    left: Mapping[str, Mapping[str, float]],
    right: Mapping[str, Mapping[str, float]],
    investors: tuple[str, ...],
    nodes: Any,
) -> float:
    return max(
        abs(left[investor][node] - right[investor][node])
        for investor in investors
        for node in nodes
    )


def embedded_clearing_price_deviation(
    summary: Mapping[str, Any],
    state: StrategyState,
    investor: str,
    common_clearing: Mapping[str, float],
    cfg: GaussSeidelConfig,
    data: MarketData,
) -> float:
    """Gap between embedded and common clearing prices at payment-relevant nodes."""
    if cfg.payment_rule != "uniform":
        return 0.0
    deviation = 0.0
    for node in data.nodes:
        row = summary["nodes"][node]
        payment_relevant = (
            row["active_award_mw"] > cfg.capacity_cleanup_tol_mw
            or state.awards_mw[investor][node] > cfg.capacity_cleanup_tol_mw
        )
        if payment_relevant:
            deviation = max(
                deviation,
                abs(row["auction_capacity_dual_eur_per_mw_day"] - common_clearing[node]),
            )
    return deviation


def record_response(
    *,
    iteration: int,
    investor: str,
    response: SolveRecord,
    selection: str,
    changes: Mapping[str, float],
    common_award_change_mw: float,
    clearing_price_deviation_eur_per_mw_day: float,
    state: StrategyState,
    cfg: GaussSeidelConfig,
    history: list[dict[str, Any]],
    history_path: Path,
) -> None:
    row = {
        "iteration": iteration,
        "investor": investor,
        "update_rule": cfg.update_rule,
        "selection": selection,
        "termination": response.termination,
        "solve_seconds": response.seconds,
        "profit_eur_per_day": response.summary["profit_eur_per_day"],
        "selected_tick_price_eur_per_mw_day": response.selected_tick_price_eur_per_mw_day,
        "access_payment_eur_per_day": response.summary["access_payment_eur_per_day"],
        "embedded_vs_common_clearing_price_deviation_eur_per_mw_day": (
            clearing_price_deviation_eur_per_mw_day
        ),
        "auction_strong_duality_gap": response.summary["auction_strong_duality_gap"],
        "spot_strong_duality_gap": response.summary["spot_strong_duality_gap"],
        "max_abs_price_dual_fraction_of_bound": response.summary[
            "max_abs_price_dual_fraction_of_bound"
        ],
        "max_abs_internal_dual_fraction_of_bound": response.summary[
            "max_abs_internal_dual_fraction_of_bound"
        ],
        "embedded_storage_pair_count": response.summary["embedded_storage_pair_count"],
        "variable_count": response.summary["variable_count"],
        "constraint_count": response.summary["constraint_count"],
        "selected_total_bid_quantity_mw": sum(state.quantity_mw[investor].values()),
        "embedded_active_award_mw": sum(
            node_row["active_award_mw"] for node_row in response.summary["nodes"].values()
        ),
        "selected_total_award_mw": sum(state.awards_mw[investor].values()),
        "common_reclear_max_award_change_mw": common_award_change_mw,
        **changes,
    }
    history.append(row)
    response_dir = cfg.output_dir / "iterations" / f"iteration_{iteration:03d}"
    response_dir.mkdir(parents=True, exist_ok=True)
    response_payload = {
        **row,
        "response_summary": response.summary,
        "tick_candidate_diagnostics": response.tick_candidate_diagnostics,
        "selected_strategy": {
            "quantity_mw": state.quantity_mw[investor],
            "price_eur_per_mw_day": state.price_eur_per_mw_day[investor],
            "duration_hours": state.duration_hours[investor],
            "common_awards_mw": state.awards_mw[investor],
        },
    }
    (response_dir / f"{investor}.json").write_text(
        json.dumps(_jsonable(response_payload), indent=2), encoding="utf-8"
    )
    write_history(history_path, history)


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in history for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_checkpoint(
    path: Path,
    cfg: GaussSeidelConfig,
    state: StrategyState,
    stop_reason: str,
    converged: bool,
) -> None:
    payload = {
        "config": _jsonable(asdict(cfg)),
        "state": _jsonable(asdict(state)),
        "stop_reason": stop_reason,
        "converged": converged,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_gauss_seidel(
    data: MarketData,
    cfg: GaussSeidelConfig,
) -> tuple[StrategyState, list[dict[str, Any]], str]:
    validate_config(cfg)
    from epec_diagonalization import four_investor_portfolio_profiles

    profiles = {profile.investor_id: profile for profile in four_investor_portfolio_profiles(data)}
    missing_profiles = set(cfg.investor_order) - set(profiles)
    if missing_profiles:
        raise ValueError(f"No thesis investor profiles for {sorted(missing_profiles)}")
    if cfg.active_nodes is not None:
        unknown_nodes = set(cfg.active_nodes) - set(data.nodes)
        if unknown_nodes:
            raise ValueError(f"Unknown active auction nodes: {sorted(unknown_nodes)}")

    state = (
        state_from_checkpoint(cfg.resume_path, data, cfg)
        if cfg.resume_path is not None
        else initial_state(data, cfg.investor_order, cfg.initial_bids_path, cfg)
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "run_config.json").write_text(
        json.dumps(_jsonable(asdict(cfg)), indent=2), encoding="utf-8"
    )

    history_path = cfg.output_dir / "iteration_history.csv"
    history: list[dict[str, Any]] = []
    if cfg.resume_path is not None and history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
    consecutive_failures = {investor: 0 for investor in cfg.investor_order}
    stop_reason = "max_iterations"
    start_iteration = state.completed_iteration + 1
    for iteration in range(start_iteration, cfg.max_iterations + 1):
        previous_state = copy_state(state)
        previous_quantity = previous_state.quantity_mw
        previous_price = previous_state.price_eur_per_mw_day
        previous_duration = previous_state.duration_hours
        previous_awards = previous_state.awards_mw
        sweep_had_failure = False
        outcomes: list[
            tuple[str, SolveRecord, str, dict[str, float]]
        ] = []
        abort_investor: str | None = None
        print(f"Iteration {iteration} [{cfg.update_rule}]")

        response_state = previous_state if cfg.update_rule == "jacobi" else state
        solved_responses: list[tuple[str, SolveRecord]] = []
        for investor in cfg.investor_order:
            response_mode = (
                "tick-grid best response"
                if cfg.bid_price_tick_eur_per_mw_day > 0.0
                else "continuous best response"
            )
            snapshot_note = " against common snapshot" if cfg.update_rule == "jacobi" else ""
            print(f"  {investor}: {response_mode}{snapshot_note}")
            response = solve_response_model(data, cfg, response_state, profiles[investor])
            solved_responses.append((investor, response))
            if response.termination == "optimal":
                consecutive_failures[investor] = 0
            else:
                consecutive_failures[investor] += 1
                sweep_had_failure = True
            if consecutive_failures[investor] >= cfg.max_consecutive_failures:
                abort_investor = investor
                break
            if cfg.update_rule == "seidel":
                selection, changes = apply_response(state, investor, response, data, cfg)
                outcomes.append((investor, response, selection, changes))

        if abort_investor is not None:
            stop_reason = f"repeated_solver_failure_{abort_investor}"
            state = previous_state
            write_checkpoint(cfg.output_dir / "checkpoint.json", cfg, state, stop_reason, False)
            return state, history, stop_reason

        if cfg.update_rule == "jacobi":
            for investor, response in solved_responses:
                selection, changes = apply_response(
                    state,
                    investor,
                    response,
                    data,
                    cfg,
                    update_awards=False,
                    reference_awards=previous_awards,
                )
                outcomes.append((investor, response, selection, changes))

        # The definitive sweep allocation is always obtained by one common
        # auction over the complete updated bid vector. In Jacobi this is the
        # only market clearing after simultaneous sealed-bid proposals.
        state.awards_mw, common_clearing = clear_common_auction(state, data, cfg)
        _cleanup_strategy(state, cfg)
        common_award_change = max_award_difference(
            state.awards_mw, previous_awards, cfg.investor_order, data.nodes
        )
        max_unilateral_award_change = max(
            (changes["max_award_change_mw"] for _, _, _, changes in outcomes),
            default=0.0,
        )
        clearing_price_deviations = {
            investor: embedded_clearing_price_deviation(
                response.summary, state, investor, common_clearing, cfg, data
            )
            for investor, response, _, _ in outcomes
        }
        max_clearing_price_deviation = max(clearing_price_deviations.values(), default=0.0)

        for investor, response, selection, changes in outcomes:
            record_response(
                iteration=iteration,
                investor=investor,
                response=response,
                selection=selection,
                changes=changes,
                common_award_change_mw=common_award_change,
                clearing_price_deviation_eur_per_mw_day=clearing_price_deviations[investor],
                state=state,
                cfg=cfg,
                history=history,
                history_path=history_path,
            )
            embedded_award = sum(
                row["active_award_mw"] for row in response.summary["nodes"].values()
            )
            print(
                f"    {investor}: termination={response.termination}, "
                f"bid={sum(state.quantity_mw[investor].values()):.3f} MW, "
                f"embedded award={embedded_award:.3f} MW, "
                f"common award={sum(state.awards_mw[investor].values()):.3f} MW"
            )

        state.completed_iteration = iteration
        max_quantity_change = max(
            abs(state.quantity_mw[i][n] - previous_quantity[i][n])
            for i in cfg.investor_order
            for n in data.nodes
        )
        max_price_change = max(
            abs(state.price_eur_per_mw_day[i][n] - previous_price[i][n])
            for i in cfg.investor_order
            for n in data.nodes
        )
        max_duration_change = max(
            abs(state.duration_hours[i][n] - previous_duration[i][n])
            for i in cfg.investor_order
            for n in data.nodes
        )
        converged = (
            not sweep_had_failure
            and max_quantity_change <= cfg.quantity_tolerance_mw
            and common_award_change <= cfg.award_tolerance_mw
            and max_unilateral_award_change <= cfg.award_tolerance_mw
            and max_price_change <= cfg.price_tolerance_eur_per_mw_day
            and max_duration_change <= cfg.duration_tolerance_hours
            and max_clearing_price_deviation <= cfg.clearing_price_tolerance_eur_per_mw_day
        )
        print(
            f"  sweep changes: q={max_quantity_change:.4f} MW, "
            f"common award={common_award_change:.4f} MW, "
            f"max unilateral award={max_unilateral_award_change:.4f} MW, "
            f"p={max_price_change:.4f} EUR/MW/day, h={max_duration_change:.4f} h, "
            f"mu dev={max_clearing_price_deviation:.4f} EUR/MW/day"
        )
        write_checkpoint(
            cfg.output_dir / "checkpoint.json",
            cfg,
            state,
            "converged" if converged else "running",
            converged,
        )
        if converged:
            stop_reason = "converged"
            break

    write_checkpoint(
        cfg.output_dir / "final_state.json",
        cfg,
        state,
        stop_reason,
        stop_reason == "converged",
    )
    return state, history, stop_reason


def main() -> int:
    cfg = parse_gauss_seidel_cli()
    data = load_market_data(cfg.data_path)
    state, history, stop_reason = run_gauss_seidel(data, cfg)
    settlement = compute_joint_auction_settlement(data, cfg, state, history)
    state.awards_mw = settlement["awards_mw"]
    write_checkpoint(
        cfg.output_dir / "final_state.json",
        cfg,
        state,
        stop_reason,
        stop_reason == "converged",
    )
    export_joint_auction_settlement(
        cfg.output_dir,
        data,
        cfg,
        state,
        history,
        stop_reason,
        settlement,
    )
    print_joint_auction_summary(stop_reason, settlement)
    print(
        f"Stopped after iteration {state.completed_iteration}: {stop_reason}; "
        f"{len(history)} investor responses recorded in {cfg.output_dir}"
    )
    return 0 if stop_reason in {"converged", "max_iterations"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
