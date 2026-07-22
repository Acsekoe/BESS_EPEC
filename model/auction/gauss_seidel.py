"""Gauss-Seidel diagonalization for the four-investor access-auction game.

Each investor solves the reusable one-leader/two-follower MPEC against the
latest fixed rival bid vectors. Accepted updates are applied immediately, so
later investors in a sweep observe earlier same-sweep changes. The economic
game remains simultaneous; Gauss-Seidel is the numerical fixed-point method.

Every candidate response is compared with an explicit zero-bid outside option
by default. This prevents a locally solved, positive-award response from being
accepted when the investor can earn more by staying out of the auction.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pyomo.environ as pyo

from config import GaussSeidelConfig, parse_gauss_seidel_cli
from single_investor_auction_mpec import (
    RivalBid,
    build_one_leader_two_follower_mpec,
    initialize_from_independent_followers,
    load_rival_bid_vector,
    solve_mpec,
    summarize,
)

# Importing the MPEC above installs the current ``Primal and dual problems``
# directory as a module fallback before these moved standalone helpers load.
from nodal_access_auction_primal import Bid, awarded_mw, build_primal as build_auction_primal
from primal_market_clearing_model import MarketData, load_market_data, value
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
    profit_eur_per_day: float
    auction_gap: float
    spot_gap: float
    model: pyo.ConcreteModel


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
    if len(set(cfg.investor_order)) != len(cfg.investor_order):
        raise ValueError("Investor order contains duplicates.")
    if len(cfg.investor_order) < 2:
        raise ValueError("At least two investors are required for auction competition.")
    if cfg.node_limit_mw <= 0.0:
        raise ValueError("Nodal auction limit must be positive.")
    if not 0.0 <= cfg.min_bid_price_eur_per_mw_day <= cfg.max_bid_price_eur_per_mw_day:
        raise ValueError("Bid-price bounds must satisfy 0 <= minimum <= maximum.")
    if cfg.max_iterations <= 0 or cfg.max_cpu_time <= 0.0:
        raise ValueError("Iteration and CPU-time limits must be positive.")
    if not 0.0 < cfg.damping <= 1.0:
        raise ValueError("Damping must be in (0, 1].")
    if min(
        cfg.quantity_tolerance_mw,
        cfg.price_tolerance_eur_per_mw_day,
        cfg.duration_tolerance_hours,
        cfg.award_tolerance_mw,
        cfg.active_award_zero_tolerance_mw,
        cfg.outside_option_tolerance_eur_per_day,
    ) < 0.0:
        raise ValueError("Convergence and outside-option tolerances must be nonnegative.")
    if cfg.max_consecutive_failures <= 0:
        raise ValueError("Maximum consecutive failures must be positive.")


def initial_state(
    data: MarketData,
    investor_order: tuple[str, ...],
    initial_bids_path: Path,
) -> StrategyState:
    records = load_rival_bid_vector(initial_bids_path)
    indexed = {(bid.investor, bid.node): bid for bid in records}
    expected = {(investor, node) for investor in investor_order for node in data.nodes}
    missing = expected - set(indexed)
    if missing:
        raise ValueError(f"Initial bid file is missing investor-node records: {sorted(missing)}")
    quantity = {
        investor: {node: float(indexed[investor, node].quantity_mw) for node in data.nodes}
        for investor in investor_order
    }
    price = {
        investor: {node: float(indexed[investor, node].price_eur_per_mw_day) for node in data.nodes}
        for investor in investor_order
    }
    duration = {
        investor: {node: float(indexed[investor, node].duration_hours) for node in data.nodes}
        for investor in investor_order
    }
    return StrategyState(quantity, price, duration, _empty_nested(investor_order, data.nodes))


def _empty_nested(investors: tuple[str, ...], nodes: Any) -> dict[str, dict[str, float]]:
    return {investor: {node: 0.0 for node in nodes} for investor in investors}


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
            i: {n: float(v) for n, v in row.items()} for i, row in state_raw["duration_hours"].items()
        },
        awards_mw={i: {n: float(v) for n, v in row.items()} for i, row in state_raw["awards_mw"].items()},
        completed_iteration=int(state_raw["completed_iteration"]),
    )
    expected_investors = set(cfg.investor_order)
    if set(state.quantity_mw) != expected_investors:
        raise ValueError("Checkpoint investors do not match configured investor order.")
    for investor in cfg.investor_order:
        if set(state.quantity_mw[investor]) != set(data.nodes):
            raise ValueError(f"Checkpoint nodes do not match market data for {investor}.")
    return state


def state_bids(state: StrategyState, data: MarketData) -> list[Bid]:
    return [
        Bid(
            investor,
            node,
            max(0.0, state.quantity_mw[investor][node]),
            max(0.0, state.price_eur_per_mw_day[investor][node]),
        )
        for investor in state.quantity_mw
        for node in data.nodes
    ]


def clear_auction(
    state: StrategyState,
    data: MarketData,
    node_limit_mw: float,
    max_cpu_time: float,
) -> dict[str, dict[str, float]]:
    limits = {node: float(node_limit_mw) for node in data.nodes}
    auction = build_auction_primal(state_bids(state, data), limits)
    result = get_ipopt_solver({"max_cpu_time": min(max_cpu_time, 60.0)}).solve(auction, tee=False)
    termination = str(result.solver.termination_condition)
    if termination != "optimal":
        raise RuntimeError(f"Independent auction clearing failed: {termination}")
    flat = awarded_mw(auction)
    return {
        investor: {node: max(0.0, flat.get((investor, node), 0.0)) for node in data.nodes}
        for investor in state.quantity_mw
    }


def rivals_from_state(
    state: StrategyState,
    active_investor: str,
    data: MarketData,
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
    ]


def solve_response_model(
    data: MarketData,
    cfg: GaussSeidelConfig,
    state: StrategyState,
    investor_profile: Any,
    *,
    force_zero_bid: bool,
) -> SolveRecord:
    investor = investor_profile.investor_id
    model = build_one_leader_two_follower_mpec(
        data,
        rivals_from_state(state, investor, data),
        active_investor=investor_profile,
        node_limit_mw=cfg.node_limit_mw,
        min_bid_price_eur_per_mw_day=cfg.min_bid_price_eur_per_mw_day,
        max_bid_price_eur_per_mw_day=cfg.max_bid_price_eur_per_mw_day,
        initial_bid_quantity_mw=state.quantity_mw[investor],
        initial_bid_price_eur_per_mw_day=state.price_eur_per_mw_day[investor],
        initial_duration_hours=state.duration_hours[investor],
        dual_bound_scale=cfg.dual_bound_scale,
    )
    if force_zero_bid:
        for node in data.nodes:
            model.active_bid_quantity[node].fix(0.0)
            model.active_bid_price[node].fix(cfg.min_bid_price_eur_per_mw_day)
            model.active_energy[node].fix(0.0)
    initialize_from_independent_followers(model)
    started = time.perf_counter()
    termination = solve_mpec(model, cfg.max_cpu_time, tee=cfg.tee)
    seconds = time.perf_counter() - started
    return SolveRecord(
        termination=termination,
        seconds=seconds,
        profit_eur_per_day=value(model.active_profit),
        auction_gap=value(model.auction_primal_value) - value(model.auction_dual_value),
        spot_gap=value(model.spot_primal_value) - value(model.spot_dual_value),
        model=model,
    )


def candidate_target(
    candidate: SolveRecord,
    baseline: SolveRecord | None,
    state: StrategyState,
    active_investor: str,
    data: MarketData,
    cfg: GaussSeidelConfig,
) -> tuple[str, dict[str, float], dict[str, float], dict[str, float]]:
    zero_quantity = {node: 0.0 for node in data.nodes}
    zero_price = {node: 0.0 for node in data.nodes}
    old_duration = dict(state.duration_hours[active_investor])
    if candidate.termination != "optimal":
        if baseline is not None and baseline.termination == "optimal":
            return "zero_after_candidate_failure", zero_quantity, zero_price, old_duration
        return (
            "retain_after_solver_failure",
            dict(state.quantity_mw[active_investor]),
            dict(state.price_eur_per_mw_day[active_investor]),
            old_duration,
        )
    if cfg.use_outside_option and (baseline is None or baseline.termination != "optimal"):
        return (
            "retain_after_baseline_failure",
            dict(state.quantity_mw[active_investor]),
            dict(state.price_eur_per_mw_day[active_investor]),
            old_duration,
        )

    total_award = sum(max(0.0, value(candidate.model.award[active_investor, n])) for n in data.nodes)
    baseline_dominates = baseline is not None and (
        candidate.profit_eur_per_day
        <= baseline.profit_eur_per_day + cfg.outside_option_tolerance_eur_per_day
    )
    if total_award <= cfg.active_award_zero_tolerance_mw or baseline_dominates:
        reason = "zero_negligible_award" if total_award <= cfg.active_award_zero_tolerance_mw else "zero_dominates_candidate"
        return reason, zero_quantity, zero_price, old_duration

    target_quantity: dict[str, float] = {}
    target_price: dict[str, float] = {}
    target_duration: dict[str, float] = {}
    profile = candidate.model._active_investor
    for node in data.nodes:
        award = max(0.0, value(candidate.model.award[active_investor, node]))
        if award <= cfg.active_award_zero_tolerance_mw:
            target_quantity[node] = 0.0
            target_price[node] = 0.0
            target_duration[node] = old_duration[node]
            continue
        target_quantity[node] = max(0.0, value(candidate.model.active_bid_quantity[node]))
        target_price[node] = max(0.0, value(candidate.model.active_bid_price[node]))
        duration = value(candidate.model.active_energy[node]) / award
        target_duration[node] = min(max(duration, profile.ratio_min), profile.ratio_max)
    return "candidate", target_quantity, target_price, target_duration


def apply_target(
    state: StrategyState,
    investor: str,
    target_quantity: Mapping[str, float],
    target_price: Mapping[str, float],
    target_duration: Mapping[str, float],
    cfg: GaussSeidelConfig,
) -> dict[str, float]:
    max_quantity_change = 0.0
    max_price_change = 0.0
    max_duration_change = 0.0
    for node in state.quantity_mw[investor]:
        old_q = state.quantity_mw[investor][node]
        old_p = state.price_eur_per_mw_day[investor][node]
        old_h = state.duration_hours[investor][node]
        new_q = (1.0 - cfg.damping) * old_q + cfg.damping * float(target_quantity[node])
        new_p = (1.0 - cfg.damping) * old_p + cfg.damping * float(target_price[node])
        new_h = (1.0 - cfg.damping) * old_h + cfg.damping * float(target_duration[node])
        if new_q <= cfg.active_award_zero_tolerance_mw:
            new_q = 0.0
            new_p = 0.0
        state.quantity_mw[investor][node] = new_q
        state.price_eur_per_mw_day[investor][node] = new_p
        state.duration_hours[investor][node] = new_h
        max_quantity_change = max(max_quantity_change, abs(new_q - old_q))
        max_price_change = max(max_price_change, abs(new_p - old_p))
        max_duration_change = max(max_duration_change, abs(new_h - old_h))
    return {
        "max_quantity_change_mw": max_quantity_change,
        "max_price_change_eur_per_mw_day": max_price_change,
        "max_duration_change_hours": max_duration_change,
    }


def award_change(
    old_awards: Mapping[str, Mapping[str, float]],
    new_awards: Mapping[str, Mapping[str, float]],
) -> float:
    return max(
        abs(float(new_awards[investor][node]) - float(old_awards[investor][node]))
        for investor in new_awards
        for node in new_awards[investor]
    )


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
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


def run_gauss_seidel(data: MarketData, cfg: GaussSeidelConfig) -> tuple[StrategyState, list[dict[str, Any]], str]:
    validate_config(cfg)
    from epec_diagonalization import four_investor_portfolio_profiles

    profiles = {profile.investor_id: profile for profile in four_investor_portfolio_profiles(data)}
    missing_profiles = set(cfg.investor_order) - set(profiles)
    if missing_profiles:
        raise ValueError(f"No thesis investor profiles for {sorted(missing_profiles)}")

    state = (
        state_from_checkpoint(cfg.resume_path, data, cfg)
        if cfg.resume_path is not None
        else initial_state(data, cfg.investor_order, cfg.initial_bids_path)
    )
    state.awards_mw = clear_auction(state, data, cfg.node_limit_mw, cfg.max_cpu_time)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "run_config.json").write_text(
        json.dumps(_jsonable(asdict(cfg)), indent=2), encoding="utf-8"
    )

    history: list[dict[str, Any]] = []
    consecutive_failures = {investor: 0 for investor in cfg.investor_order}
    stop_reason = "max_iterations"
    start_iteration = state.completed_iteration + 1

    for iteration in range(start_iteration, cfg.max_iterations + 1):
        previous_quantity = {i: dict(row) for i, row in state.quantity_mw.items()}
        previous_price = {i: dict(row) for i, row in state.price_eur_per_mw_day.items()}
        previous_duration = {i: dict(row) for i, row in state.duration_hours.items()}
        previous_awards = {i: dict(row) for i, row in state.awards_mw.items()}
        sweep_had_failure = False
        print(f"Iteration {iteration}")

        for investor in cfg.investor_order:
            print(f"  {investor}: candidate best response")
            candidate = solve_response_model(data, cfg, state, profiles[investor], force_zero_bid=False)
            baseline = None
            if cfg.use_outside_option:
                print(f"  {investor}: zero-bid outside option")
                baseline = solve_response_model(data, cfg, state, profiles[investor], force_zero_bid=True)
            selection, target_q, target_p, target_h = candidate_target(
                candidate, baseline, state, investor, data, cfg
            )
            changes = apply_target(state, investor, target_q, target_p, target_h, cfg)
            old_awards = {i: dict(row) for i, row in state.awards_mw.items()}
            state.awards_mw = clear_auction(
                state, data, cfg.node_limit_mw, cfg.max_cpu_time
            )
            response_award_change = award_change(old_awards, state.awards_mw)

            response_ok = candidate.termination == "optimal" and (
                not cfg.use_outside_option
                or (baseline is not None and baseline.termination == "optimal")
            )
            if response_ok:
                consecutive_failures[investor] = 0
            else:
                consecutive_failures[investor] += 1
                sweep_had_failure = True
            baseline_term = baseline.termination if baseline is not None else "skipped"
            baseline_profit = baseline.profit_eur_per_day if baseline is not None else float("nan")
            active_award = sum(state.awards_mw[investor].values())
            row = {
                "iteration": iteration,
                "investor": investor,
                "selection": selection,
                "candidate_termination": candidate.termination,
                "candidate_seconds": candidate.seconds,
                "candidate_profit_eur_per_day": candidate.profit_eur_per_day,
                "candidate_auction_gap": candidate.auction_gap,
                "candidate_spot_gap": candidate.spot_gap,
                "baseline_termination": baseline_term,
                "baseline_profit_eur_per_day": baseline_profit,
                "selected_total_award_mw": active_award,
                **changes,
                "response_max_award_change_mw": response_award_change,
            }
            history.append(row)
            response_dir = cfg.output_dir / "iterations" / f"iteration_{iteration:03d}"
            response_dir.mkdir(parents=True, exist_ok=True)
            candidate_summary = summarize(candidate.model, candidate.termination)
            response_payload = {
                **row,
                "candidate_summary": candidate_summary,
                "baseline": None
                if baseline is None
                else {
                    "termination": baseline.termination,
                    "seconds": baseline.seconds,
                    "profit_eur_per_day": baseline.profit_eur_per_day,
                    "auction_gap": baseline.auction_gap,
                    "spot_gap": baseline.spot_gap,
                },
                "selected_strategy": {
                    "quantity_mw": state.quantity_mw[investor],
                    "price_eur_per_mw_day": state.price_eur_per_mw_day[investor],
                    "duration_hours": state.duration_hours[investor],
                    "awards_mw": state.awards_mw[investor],
                },
            }
            (response_dir / f"{investor}.json").write_text(
                json.dumps(_jsonable(response_payload), indent=2), encoding="utf-8"
            )
            write_history(cfg.output_dir / "iteration_history.csv", history)
            print(
                f"    candidate={candidate.termination}, baseline={baseline_term}, "
                f"selection={selection}, award={active_award:.3f} MW"
            )
            if consecutive_failures[investor] >= cfg.max_consecutive_failures:
                stop_reason = f"repeated_solver_failure_{investor}"
                state.quantity_mw = previous_quantity
                state.price_eur_per_mw_day = previous_price
                state.duration_hours = previous_duration
                state.awards_mw = previous_awards
                state.completed_iteration = iteration - 1
                write_checkpoint(cfg.output_dir / "checkpoint.json", cfg, state, stop_reason, False)
                return state, history, stop_reason

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
        max_award_change = award_change(previous_awards, state.awards_mw)
        converged = (
            not sweep_had_failure
            and max_quantity_change <= cfg.quantity_tolerance_mw
            and max_price_change <= cfg.price_tolerance_eur_per_mw_day
            and max_duration_change <= cfg.duration_tolerance_hours
            and max_award_change <= cfg.award_tolerance_mw
        )
        print(
            f"  sweep changes: q={max_quantity_change:.4f} MW, "
            f"p={max_price_change:.4f}, h={max_duration_change:.4f} h, "
            f"award={max_award_change:.4f} MW"
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
    print(
        f"Stopped after iteration {state.completed_iteration}: {stop_reason}; "
        f"{len(history)} investor responses recorded in {cfg.output_dir}"
    )
    return 0 if stop_reason in {"converged", "max_iterations"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
