"""Multi-investor strategic-operation EPEC via Jacobi or Gauss-Seidel.

This driver is deliberately separate from ``epec_diagonalization.py``. The
maintained driver keeps storage fully available to the ISO; this experiment
adds hourly charge/discharge quantities and, optionally, charging buy-bid and
discharging sell-offer prices to every investor's strategy.

A fresh run first solves one common-snapshot Gauss-Jacobi best-response sweep
and projects overloaded nodes proportionally. Subsequent sweeps use either
full Gauss-Jacobi or Gauss-Seidel updates. In Jacobi mode every investor sees
the same complete sweep-start strategy; all damped proposals are applied only
after every best response has finished.

An optional proximal penalty can select a nearby capacity/withholding best
response during diagonalization. It is disabled by default and excluded from
the Jacobi initializer. A converged penalized run remains a regularized
candidate that requires an unpenalized best-response check.

An optional direct epsilon-times-square penalty pins strategic bid prices
toward zero without directly penalizing capacity, energy, or quantity offers.
Convergence requires both applied and raw best-response capacity, withholding,
and canonical active-price norms for three consecutive sweeps; the old
maximum-relative changes remain diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import pyomo.environ as pyo

from epec_diagonalization import (
    DEFAULT_CAPACITY_CLEANUP_TOL_MW_MWH,
    DEFAULT_DAMPING,
    DEFAULT_FLOOR_MW,
    DEFAULT_FLOOR_MWH,
    DEFAULT_TOL_REL,
    EpecConfig,
    clean_capacity_pair,
    four_investor_portfolio_profiles,
    order_investors,
    relative_delta,
)
from epec_results import compute_joint_settlement, export_epec_results, print_epec_summary
from ieee9_strategic_operation_mpec import (
    DEFAULT_DATA_PATH,
    apply_generator_calibration,
    build_ieee9_strategic_operation_mpec,
)
from single_investor_mpec import (
    DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    DEFAULT_INITIAL_POWER_MW,
    DEFAULT_INITIAL_RATIO_HOURS,
    DEFAULT_NODE_LIMIT_MW,
    DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    DEFAULT_SOLVER_TOL,
    InvestorConfig,
    QuadraticDemandCurve,
    default_quadratic_demand_curve,
    initialize_from_reference_dispatch,
    investment_headroom_shadow_price,
    load_market_data,
    value,
)
from single_investor_mpec_results import _write_csv
from solver_utils import get_ipopt_solver
from tikhonov_kkt.mpec_strategic_operation_strong_duality import (
    build_strategic_operation_tikhonov_mpec,
    initialize_strategic_mpec_from_soft_market,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "epec_strategic_operation"
OFFER_CANONICAL_SLACK_MW = 1.0e-4


def configured_demand_curve(cfg: EpecConfig) -> QuadraticDemandCurve:
    return QuadraticDemandCurve(
        alpha=cfg.quadratic_demand_alpha_eur_per_mwh,
        beta=cfg.quadratic_demand_beta_eur_per_mwh_per_share,
    )


@dataclass
class StrategicBestResponse:
    investor_id: str
    termination: str
    solve_seconds: float
    proposed_power: dict[str, float]
    proposed_energy: dict[str, float]
    proposed_offer_charge: dict[tuple[str, int], float]
    proposed_offer_discharge: dict[tuple[str, int], float]
    proposed_bid_price_charge: dict[tuple[str, int], float]
    proposed_offer_price_discharge: dict[tuple[str, int], float]
    accepted_charge: dict[tuple[str, int], float]
    accepted_discharge: dict[tuple[str, int], float]
    private_headroom_limit_mw: dict[str, float]
    optimistic_mpec_profit_eur_per_day: float
    proximal_penalty_eur_per_day: float
    epsilon_penalty_eur_per_day: float
    penalized_objective_eur_per_day: float
    access_shadow_price_eur_per_mw_day: dict[str, float]
    strong_duality_gap: float
    selected_price_eur_per_mwh: dict[tuple[str, int], float]
    model: pyo.ConcreteModel | None

    @property
    def ok(self) -> bool:
        return self.termination == "optimal"


@dataclass
class StrategicEpecState:
    x_power: dict[tuple[str, str], float]
    x_energy: dict[tuple[str, str], float]
    offer_charge: dict[tuple[str, str, int], float]
    offer_discharge: dict[tuple[str, str, int], float]
    bid_price_charge: dict[tuple[str, str, int], float]
    offer_price_discharge: dict[tuple[str, str, int], float]
    iteration: int = 0
    converged: bool = False
    stop_reason: str = ""
    history: list[dict] = field(default_factory=list)
    trajectory: list[dict] = field(default_factory=list)
    projection_events: list[dict] = field(default_factory=list)
    final_models: dict[str, pyo.ConcreteModel] = field(default_factory=dict)
    final_selected_prices: dict[str, dict[tuple[str, int], float]] = field(
        default_factory=dict
    )
    final_access_shadow_prices: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    initialization_method: str = "uniform_seed_full_availability"
    initializer_summary: dict = field(default_factory=dict)
    offer_convergence_history: list[dict] = field(default_factory=list)
    proximal_history: list[dict] = field(default_factory=list)
    consecutive_converged_sweeps: int = 0


def separate_rival_strategies(
    state: StrategicEpecState,
    cfg: EpecConfig,
    nodes: list[str],
    times: list[int],
    active_id: str,
):
    rival_power: dict[str, dict[str, float]] = {}
    rival_energy: dict[str, dict[str, float]] = {}
    rival_charge: dict[str, dict[tuple[str, int], float]] = {}
    rival_discharge: dict[str, dict[tuple[str, int], float]] = {}
    rival_charge_price: dict[str, dict[tuple[str, int], float]] = {}
    rival_discharge_price: dict[str, dict[tuple[str, int], float]] = {}
    for investor in cfg.investors:
        rival_id = investor.investor_id
        if rival_id == active_id:
            continue
        rival_power[rival_id] = {
            node: max(0.0, state.x_power[rival_id, node]) for node in nodes
        }
        rival_energy[rival_id] = {
            node: max(0.0, state.x_energy[rival_id, node]) for node in nodes
        }
        rival_charge[rival_id] = {
            (node, time): min(
                max(0.0, state.offer_charge[rival_id, node, time]),
                rival_power[rival_id][node],
            )
            for node in nodes
            for time in times
        }
        rival_discharge[rival_id] = {
            (node, time): min(
                max(0.0, state.offer_discharge[rival_id, node, time]),
                rival_power[rival_id][node],
            )
            for node in nodes
            for time in times
        }
        rival_charge_price[rival_id] = {
            (node, time): state.bid_price_charge[rival_id, node, time]
            for node in nodes
            for time in times
        }
        rival_discharge_price[rival_id] = {
            (node, time): state.offer_price_discharge[rival_id, node, time]
            for node in nodes
            for time in times
        }
    return (
        rival_power,
        rival_energy,
        rival_charge,
        rival_discharge,
        rival_charge_price,
        rival_discharge_price,
    )


def _effective_offer(raw_offer: float, accepted: float, installed_power: float) -> float:
    """Canonicalize economically irrelevant slack offers to full availability."""

    if raw_offer > accepted + OFFER_CANONICAL_SLACK_MW:
        return installed_power
    return min(max(0.0, raw_offer), installed_power)


def _effective_bid_price(
    raw_price: float,
    offered_quantity_mw: float,
    truthful_price: float,
    bound: float,
) -> float:
    """Canonicalize a price attached to zero offered quantity."""

    if offered_quantity_mw <= OFFER_CANONICAL_SLACK_MW:
        return truthful_price
    return min(bound, max(-bound, raw_price))


def proximal_coefficient_for_iteration(cfg: EpecConfig, iteration: int) -> float:
    """Return the fixed or staircase proximal coefficient for one sweep."""

    step = cfg.strategic_proximal_penalty_step_eur_per_mw2_day
    if step <= 0.0:
        return cfg.strategic_proximal_penalty_eur_per_mw2_day
    zero_iterations = cfg.strategic_proximal_penalty_initial_zero_iterations
    if int(iteration) <= zero_iterations:
        return 0.0
    block = 1 + (
        int(iteration) - zero_iterations - 1
    ) // cfg.strategic_proximal_penalty_step_iterations
    return step * block


def proximal_regularization_enabled(cfg: EpecConfig) -> bool:
    return (
        cfg.strategic_proximal_penalty_eur_per_mw2_day > 0.0
        or cfg.strategic_proximal_penalty_step_eur_per_mw2_day > 0.0
    )


def strategy_regularization_enabled(cfg: EpecConfig) -> bool:
    return proximal_regularization_enabled(cfg) or cfg.strategic_epsilon_penalty > 0.0


def solve_best_response(
    data,
    cfg: EpecConfig,
    investor: InvestorConfig,
    rival_power,
    rival_energy,
    rival_charge,
    rival_discharge,
    rival_charge_price,
    rival_discharge_price,
    previous_power: dict[str, float],
    previous_energy: dict[str, float],
    previous_charge: dict[tuple[str, int], float],
    previous_discharge: dict[tuple[str, int], float],
    previous_charge_price: dict[tuple[str, int], float],
    previous_discharge_price: dict[tuple[str, int], float],
    *,
    initial_guess_power: dict[str, float] | None = None,
    initial_guess_energy: dict[str, float] | None = None,
    tee: bool = False,
) -> StrategicBestResponse:
    nodes = list(data.nodes)
    times = [int(time) for time in data.times]
    guess_power = initial_guess_power or previous_power
    guess_energy = initial_guess_energy or previous_energy

    def attempt(shrink: float):
        builder = (
            build_strategic_operation_tikhonov_mpec
            if cfg.lower_level_optimality == "tikhonov-strong-duality"
            else build_ieee9_strategic_operation_mpec
        )
        builder_kwargs = {}
        if cfg.lower_level_optimality == "tikhonov-strong-duality":
            builder_kwargs["dual_tikhonov_gamma"] = cfg.dual_tikhonov_gamma
        model = builder(
            data,
            investor=investor,
            rival_power_mw_by_unit=rival_power,
            rival_energy_mwh_by_unit=rival_energy,
            rival_offer_charge_mw_by_unit=rival_charge,
            rival_offer_discharge_mw_by_unit=rival_discharge,
            strategic_bid_prices=cfg.strategic_bid_prices,
            rival_bid_price_charge_eur_per_mwh_by_unit=rival_charge_price,
            rival_offer_price_discharge_eur_per_mwh_by_unit=rival_discharge_price,
            bid_price_bound_eur_per_mwh=(
                cfg.strategic_bid_price_bound_eur_per_mwh
            ),
            rival_degradation_eur_per_mwh_by_unit={
                other.investor_id: other.degradation_eur_per_mwh
                for other in cfg.investors
                if other.investor_id != investor.investor_id
            },
            node_limit_mw=cfg.node_limit_mw,
            initial_power_mw=cfg.seed_power_mw,
            initial_ratio_hours=cfg.seed_ratio_hours,
            price_bound_eur_per_mwh=cfg.price_bound_eur_per_mwh,
            price_lower_bound_eur_per_mwh=cfg.price_lower_bound_eur_per_mwh,
            price_upper_bound_eur_per_mwh=cfg.price_upper_bound_eur_per_mwh,
            dual_bound_eur_per_mwh=cfg.dual_bound_eur_per_mwh,
            dispatch_regularization_eur_per_mw2h=cfg.dispatch_regularization_eur_per_mw2h,
            system_price_settlement=cfg.system_price_settlement,
            solver_tol=cfg.solver_tol,
            quad_demand=configured_demand_curve(cfg),
            use_demand_curve=cfg.use_demand_curve,
            proximal_penalty_eur_per_mw2_day=(
                cfg.strategic_proximal_penalty_eur_per_mw2_day
            ),
            proximal_energy_scale_hours=cfg.strategic_proximal_energy_scale_hours,
            proximal_price_scale_eur_per_mwh=(
                cfg.strategic_proximal_price_scale_eur_per_mwh
            ),
            proximal_reference_power_mw=previous_power,
            proximal_reference_energy_mwh=previous_energy,
            proximal_reference_offer_charge_mw=previous_charge,
            proximal_reference_offer_discharge_mw=previous_discharge,
            proximal_reference_bid_price_charge_eur_per_mwh=(
                previous_charge_price
            ),
            proximal_reference_offer_price_discharge_eur_per_mwh=(
                previous_discharge_price
            ),
            strategic_epsilon_penalty=cfg.strategic_epsilon_penalty,
            initialize_model=False,
            **builder_kwargs,
        )
        for node in nodes:
            headroom = max(
                0.0,
                cfg.node_limit_mw
                - sum(rival_power[unit][node] for unit in rival_power),
            )
            power = min(max(0.0, shrink * guess_power[node]), headroom)
            energy = min(
                max(investor.ratio_min * power, shrink * guess_energy[node]),
                investor.ratio_max * headroom,
            )
            model.X_power[node].set_value(power)
            model.X_energy[node].set_value(energy)

        # The ordinary formulation uses the full-availability clear as a broad
        # warm start. The Tikhonov formulation is initialized below from the
        # exact fixed-fleet, fixed-bid soft primal/dual pair instead.
        if cfg.lower_level_optimality != "tikhonov-strong-duality":
            initialize_from_reference_dispatch(model, data, cfg.seed_ratio_hours)
        for node in nodes:
            power = value(model.X_power[node])
            for time_ in times:
                model.Q_offer_charge[node, time_].set_value(
                    min(power, max(0.0, shrink * previous_charge[node, time_]))
                )
                model.Q_offer_discharge[node, time_].set_value(
                    min(power, max(0.0, shrink * previous_discharge[node, time_]))
                )
                if cfg.strategic_bid_prices:
                    model.p_bid_charge[node, time_].set_value(
                        previous_charge_price[node, time_]
                    )
                    model.p_offer_discharge[node, time_].set_value(
                        previous_discharge_price[node, time_]
                    )

        if cfg.lower_level_optimality == "tikhonov-strong-duality":
            initialize_strategic_mpec_from_soft_market(
                model,
                solver_tol=cfg.solver_tol,
                max_cpu_time=cfg.max_cpu_time,
                tee=False,
            )

        start = time.perf_counter()
        try:
            results = get_ipopt_solver(
                {
                    "max_cpu_time": cfg.max_cpu_time,
                    "tol": cfg.solver_tol,
                    "acceptable_tol": cfg.solver_tol,
                }
            ).solve(model, tee=tee)
            termination = str(results.solver.termination_condition)
        except (RuntimeError, ValueError) as exc:
            termination = f"solver_exception: {type(exc).__name__}"
        return model, termination, time.perf_counter() - start

    model, termination, seconds = attempt(1.0)
    if termination != "optimal":
        model, termination, retry_seconds = attempt(0.9)
        seconds += retry_seconds

    headroom = {
        node: cfg.node_limit_mw
        - sum(rival_power[unit][node] for unit in rival_power)
        for node in nodes
    }
    if termination != "optimal":
        return StrategicBestResponse(
            investor.investor_id,
            termination,
            seconds,
            dict(previous_power),
            dict(previous_energy),
            dict(previous_charge),
            dict(previous_discharge),
            dict(previous_charge_price),
            dict(previous_discharge_price),
            {(node, time_): 0.0 for node in nodes for time_ in times},
            {(node, time_): 0.0 for node in nodes for time_ in times},
            headroom,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            {node: float("nan") for node in nodes},
            float("nan"),
            {},
            None,
        )

    proposed_power = {node: max(0.0, value(model.X_power[node])) for node in nodes}
    proposed_energy = {node: max(0.0, value(model.X_energy[node])) for node in nodes}
    accepted_charge = {
        (node, time_): max(0.0, value(model.P_charge[investor.investor_id, node, time_]))
        for node in nodes
        for time_ in times
    }
    accepted_discharge = {
        (node, time_): max(0.0, value(model.P_discharge[investor.investor_id, node, time_]))
        for node in nodes
        for time_ in times
    }
    proposed_charge = {
        (node, time_): _effective_offer(
            value(model.Q_offer_charge[node, time_]),
            accepted_charge[node, time_],
            proposed_power[node],
        )
        for node in nodes
        for time_ in times
    }
    proposed_discharge = {
        (node, time_): _effective_offer(
            value(model.Q_offer_discharge[node, time_]),
            accepted_discharge[node, time_],
            proposed_power[node],
        )
        for node in nodes
        for time_ in times
    }
    proposed_charge_price = {
        (node, time_): _effective_bid_price(
            value(model.p_bid_charge[node, time_]),
            proposed_charge[node, time_],
            -0.5 * investor.degradation_eur_per_mwh,
            cfg.strategic_bid_price_bound_eur_per_mwh,
        )
        if cfg.strategic_bid_prices
        else previous_charge_price[node, time_]
        for node in nodes
        for time_ in times
    }
    proposed_discharge_price = {
        (node, time_): _effective_bid_price(
            value(model.p_offer_discharge[node, time_]),
            proposed_discharge[node, time_],
            0.5 * investor.degradation_eur_per_mwh,
            cfg.strategic_bid_price_bound_eur_per_mwh,
        )
        if cfg.strategic_bid_prices
        else previous_discharge_price[node, time_]
        for node in nodes
        for time_ in times
    }
    return StrategicBestResponse(
        investor.investor_id,
        termination,
        seconds,
        proposed_power,
        proposed_energy,
        proposed_charge,
        proposed_discharge,
        proposed_charge_price,
        proposed_discharge_price,
        accepted_charge,
        accepted_discharge,
        headroom,
        value(model.investor_profit_expr),
        value(model.proximal_penalty_expr),
        value(model.strategic_epsilon_penalty_expr),
        value(model.objective.expr),
        {node: investment_headroom_shadow_price(model, node) for node in nodes},
        abs(value(model.primal_objective_expr) - value(model.dual_objective_expr)),
        {
            (node, time_): value(model.lam_sys[time_])
            if cfg.system_price_settlement
            else value(model.lam[node, time_])
            for node in nodes
            for time_ in times
        },
        model,
    )


@dataclass(frozen=True)
class StrategicBestResponseTask:
    investor_id: str
    arguments: tuple
    initial_guess_power: dict[str, float] | None = None
    initial_guess_energy: dict[str, float] | None = None


def best_response_task(
    data,
    cfg: EpecConfig,
    investor: InvestorConfig,
    state: StrategicEpecState,
    nodes: list[str],
    times: list[int],
    *,
    initial_guess_power: dict[str, float] | None = None,
    initial_guess_energy: dict[str, float] | None = None,
) -> StrategicBestResponseTask:
    investor_id = investor.investor_id
    rival = separate_rival_strategies(state, cfg, nodes, times, investor_id)
    arguments = (
        data,
        cfg,
        investor,
        *rival,
        {node: state.x_power[investor_id, node] for node in nodes},
        {node: state.x_energy[investor_id, node] for node in nodes},
        {
            (node, time_): state.offer_charge[investor_id, node, time_]
            for node in nodes
            for time_ in times
        },
        {
            (node, time_): state.offer_discharge[investor_id, node, time_]
            for node in nodes
            for time_ in times
        },
        {
            (node, time_): state.bid_price_charge[investor_id, node, time_]
            for node in nodes
            for time_ in times
        },
        {
            (node, time_): state.offer_price_discharge[investor_id, node, time_]
            for node in nodes
            for time_ in times
        },
    )
    return StrategicBestResponseTask(
        investor_id,
        arguments,
        initial_guess_power,
        initial_guess_energy,
    )


def _solve_best_response_task(
    task: StrategicBestResponseTask, *, retain_model: bool, tee: bool
) -> StrategicBestResponse:
    response = solve_best_response(
        *task.arguments,
        initial_guess_power=task.initial_guess_power,
        initial_guess_energy=task.initial_guess_energy,
        tee=tee,
    )
    if not retain_model:
        response.model = None
    return response


def solve_best_response_tasks(
    tasks: list[StrategicBestResponseTask],
    *,
    parallel_workers: int,
    tee: bool,
) -> list[StrategicBestResponse]:
    """Solve tasks in parallel while returning results in configured ID order."""

    if parallel_workers <= 1:
        return [
            _solve_best_response_task(task, retain_model=True, tee=tee)
            for task in tasks
        ]
    if tee:
        raise ValueError("Parallel strategic solves do not support --tee.")
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            task.investor_id: executor.submit(
                _solve_best_response_task,
                task,
                retain_model=False,
                tee=False,
            )
            for task in tasks
        }
        responses = []
        for task in tasks:
            try:
                responses.append(futures[task.investor_id].result())
            except Exception as exc:
                raise RuntimeError(
                    f"Parallel strategic best response failed for {task.investor_id}: {exc}"
                ) from exc
        return responses


def clean_and_bound_state(
    state: StrategicEpecState, cfg: EpecConfig, nodes, times, eta: float
) -> None:
    for investor in cfg.investors:
        investor_id = investor.investor_id
        for node in nodes:
            key = investor_id, node
            state.x_power[key], state.x_energy[key] = clean_capacity_pair(
                state.x_power[key],
                state.x_energy[key],
                cfg.capacity_cleanup_tol_mw_mwh,
            )
            power = state.x_power[key]
            for time_ in times:
                state.offer_charge[investor_id, node, time_] = min(
                    power, max(0.0, state.offer_charge[investor_id, node, time_])
                )
                state.offer_discharge[investor_id, node, time_] = min(
                    power, max(0.0, state.offer_discharge[investor_id, node, time_])
                )
                price_bound = cfg.strategic_bid_price_bound_eur_per_mwh
                state.bid_price_charge[investor_id, node, time_] = _effective_bid_price(
                    state.bid_price_charge[investor_id, node, time_],
                    state.offer_charge[investor_id, node, time_],
                    -0.5 * investor.degradation_eur_per_mwh,
                    price_bound,
                )
                state.offer_price_discharge[investor_id, node, time_] = _effective_bid_price(
                    state.offer_price_discharge[investor_id, node, time_],
                    state.offer_discharge[investor_id, node, time_],
                    0.5 * investor.degradation_eur_per_mwh,
                    price_bound,
                )
                state.bid_price_charge[investor_id, node, time_] = min(
                    state.bid_price_charge[investor_id, node, time_],
                    (eta**2) * price_bound,
                )
                state.offer_price_discharge[investor_id, node, time_] = max(
                    state.offer_price_discharge[investor_id, node, time_],
                    state.bid_price_charge[investor_id, node, time_] / (eta**2),
                )


def apply_damped_update(state, cfg, nodes, times, response, eta) -> None:
    alpha = cfg.damping
    investor_id = response.investor_id
    for node in nodes:
        state.x_power[investor_id, node] = (
            (1.0 - alpha) * state.x_power[investor_id, node]
            + alpha * response.proposed_power[node]
        )
        state.x_energy[investor_id, node] = (
            (1.0 - alpha) * state.x_energy[investor_id, node]
            + alpha * response.proposed_energy[node]
        )
        for time_ in times:
            state.offer_charge[investor_id, node, time_] = (
                (1.0 - alpha) * state.offer_charge[investor_id, node, time_]
                + alpha * response.proposed_offer_charge[node, time_]
            )
            state.offer_discharge[investor_id, node, time_] = (
                (1.0 - alpha) * state.offer_discharge[investor_id, node, time_]
                + alpha * response.proposed_offer_discharge[node, time_]
            )
            if cfg.strategic_bid_prices:
                state.bid_price_charge[investor_id, node, time_] = (
                    (1.0 - alpha) * state.bid_price_charge[investor_id, node, time_]
                    + alpha * response.proposed_bid_price_charge[node, time_]
                )
                state.offer_price_discharge[investor_id, node, time_] = (
                    (1.0 - alpha)
                    * state.offer_price_discharge[investor_id, node, time_]
                    + alpha * response.proposed_offer_price_discharge[node, time_]
                )
    clean_and_bound_state(state, cfg, nodes, times, eta)


def project_joint_limit(state, cfg, nodes, times, eta) -> None:
    for node in nodes:
        total = sum(state.x_power[investor.investor_id, node] for investor in cfg.investors)
        if total <= cfg.node_limit_mw + 1e-8:
            continue
        scale = cfg.node_limit_mw / total
        for investor in cfg.investors:
            investor_id = investor.investor_id
            state.x_power[investor_id, node] *= scale
            state.x_energy[investor_id, node] *= scale
            for time_ in times:
                state.offer_charge[investor_id, node, time_] *= scale
                state.offer_discharge[investor_id, node, time_] *= scale
        state.projection_events.append(
            {
                "iteration": state.iteration,
                "node": node,
                "total_before_mw": total,
                "scale": scale,
            }
        )
        print(
            f"  [projection] iter {state.iteration}, {node}: "
            f"{total:.3f} MW -> {cfg.node_limit_mw:.1f} MW"
        )
    clean_and_bound_state(state, cfg, nodes, times, eta)


def projected_jacobi_initial_state(data, cfg, *, tee=False) -> StrategicEpecState:
    nodes = list(data.nodes)
    times = [int(time_) for time_ in data.times]
    snapshot_power = cfg.jacobi_initializer_snapshot_power_mw
    snapshot_ratio = cfg.jacobi_initializer_snapshot_ratio_hours
    snapshot = StrategicEpecState(
        x_power={
            (investor.investor_id, node): snapshot_power
            for investor in cfg.investors
            for node in nodes
        },
        x_energy={
            (investor.investor_id, node): snapshot_power * snapshot_ratio
            for investor in cfg.investors
            for node in nodes
        },
        offer_charge={
            (investor.investor_id, node, time_): snapshot_power
            for investor in cfg.investors
            for node in nodes
            for time_ in times
        },
        offer_discharge={
            (investor.investor_id, node, time_): snapshot_power
            for investor in cfg.investors
            for node in nodes
            for time_ in times
        },
        bid_price_charge={
            (investor.investor_id, node, time_): -0.5
            * investor.degradation_eur_per_mwh
            for investor in cfg.investors
            for node in nodes
            for time_ in times
        },
        offer_price_discharge={
            (investor.investor_id, node, time_): 0.5
            * investor.degradation_eur_per_mwh
            for investor in cfg.investors
            for node in nodes
            for time_ in times
        },
        initialization_method="strategic_jacobi_common_snapshot_projected",
    )
    numerical_power = {node: cfg.seed_power_mw for node in nodes}
    numerical_energy = {
        node: cfg.seed_power_mw * cfg.seed_ratio_hours for node in nodes
    }
    initializer_cfg = replace(
        cfg,
        strategic_proximal_penalty_eur_per_mw2_day=0.0,
    )
    print(
        "Strategic Jacobi initializer: common snapshot "
        f"{snapshot_power:g} MW/node; numerical guess {cfg.seed_power_mw:g} MW/node; "
        "proximal penalty off"
    )
    tasks = [
        best_response_task(
            data,
            initializer_cfg,
            investor,
            snapshot,
            nodes,
            times,
            initial_guess_power=numerical_power,
            initial_guess_energy=numerical_energy,
        )
        for investor in cfg.investors
    ]
    responses = solve_best_response_tasks(
        tasks,
        parallel_workers=cfg.strategic_parallel_workers,
        tee=tee,
    )
    for response in responses:
        investor_id = response.investor_id
        if not response.ok:
            raise RuntimeError(
                f"Strategic Jacobi initializer failed for {investor_id}: {response.termination}"
            )
        print(
            f"  {investor_id}: desired {sum(response.proposed_power.values()):.3f} MW / "
            f"{sum(response.proposed_energy.values()):.3f} MWh"
        )

    state = StrategicEpecState({}, {}, {}, {}, {}, {})
    node_summary = {}
    for node in nodes:
        desired_total = sum(response.proposed_power[node] for response in responses)
        scale = min(1.0, cfg.node_limit_mw / desired_total) if desired_total > 0.0 else 1.0
        for response in responses:
            investor_id = response.investor_id
            state.x_power[investor_id, node] = scale * response.proposed_power[node]
            state.x_energy[investor_id, node] = scale * response.proposed_energy[node]
            for time_ in times:
                state.offer_charge[investor_id, node, time_] = (
                    scale * response.proposed_offer_charge[node, time_]
                )
                state.offer_discharge[investor_id, node, time_] = (
                    scale * response.proposed_offer_discharge[node, time_]
                )
                state.bid_price_charge[investor_id, node, time_] = (
                    response.proposed_bid_price_charge[node, time_]
                )
                state.offer_price_discharge[investor_id, node, time_] = (
                    response.proposed_offer_price_discharge[node, time_]
                )
        node_summary[node] = {
            "desired_total_power_mw": desired_total,
            "projection_scale": scale,
            "projected_total_power_mw": sum(
                state.x_power[investor.investor_id, node]
                for investor in cfg.investors
            ),
        }
    state.initialization_method = "strategic_jacobi_common_snapshot_projected"
    state.initializer_summary = {
        "interpretation": "feasible strategic initialization heuristic, not an equilibrium",
        "proximal_penalty_applied": False,
        "snapshot_power_mw_per_investor_node": snapshot_power,
        "snapshot_ratio_hours": snapshot_ratio,
        "nodes": node_summary,
        "responses": {
            response.investor_id: {
                "desired_power_mw": sum(response.proposed_power.values()),
                "desired_energy_mwh": sum(response.proposed_energy.values()),
                "optimistic_mpec_profit_eur_per_day": response.optimistic_mpec_profit_eur_per_day,
                "proximal_penalty_eur_per_day": response.proximal_penalty_eur_per_day,
                "epsilon_penalty_eur_per_day": response.epsilon_penalty_eur_per_day,
                "penalized_objective_eur_per_day": response.penalized_objective_eur_per_day,
                "strong_duality_gap": response.strong_duality_gap,
            }
            for response in responses
        },
    }
    clean_and_bound_state(state, cfg, nodes, times, data.eta)
    return state


def run_epec(
    data,
    cfg: EpecConfig,
    *,
    tee: bool = False,
    checkpoint_callback: Callable[[StrategicEpecState], None] | None = None,
    initial_state: StrategicEpecState | None = None,
) -> StrategicEpecState:
    if cfg.update_rule not in {"jacobi", "seidel"}:
        raise ValueError(f"Unsupported strategic update rule: {cfg.update_rule!r}")
    nodes = list(data.nodes)
    times = [int(time_) for time_ in data.times]
    if initial_state is None and cfg.automatic_jacobi_initializer:
        state = projected_jacobi_initial_state(data, cfg, tee=tee)
        if checkpoint_callback:
            checkpoint_callback(state)
        print(
            "Strategic Jacobi initializer complete; starting "
            f"Gauss-{cfg.update_rule.capitalize()}."
        )
    elif initial_state is None:
        seed = min(
            cfg.jacobi_initializer_snapshot_power_mw,
            cfg.node_limit_mw / len(cfg.investors),
        )
        snapshot_ratio = cfg.jacobi_initializer_snapshot_ratio_hours
        state = StrategicEpecState(
            x_power={(i.investor_id, n): seed for i in cfg.investors for n in nodes},
            x_energy={(i.investor_id, n): seed * snapshot_ratio for i in cfg.investors for n in nodes},
            offer_charge={(i.investor_id, n, t): seed for i in cfg.investors for n in nodes for t in times},
            offer_discharge={(i.investor_id, n, t): seed for i in cfg.investors for n in nodes for t in times},
            bid_price_charge={
                (i.investor_id, n, t): -0.5 * i.degradation_eur_per_mwh
                for i in cfg.investors
                for n in nodes
                for t in times
            },
            offer_price_discharge={
                (i.investor_id, n, t): 0.5 * i.degradation_eur_per_mwh
                for i in cfg.investors
                for n in nodes
                for t in times
            },
            initialization_method="uniform_common_snapshot_direct",
            initializer_summary={
                "economic_power_mw_per_investor_node": seed,
                "economic_duration_hours": snapshot_ratio,
                "mpec_numerical_guess_power_mw_per_node": cfg.seed_power_mw,
                "mpec_numerical_guess_duration_hours": cfg.seed_ratio_hours,
            },
        )
    else:
        state = initial_state
        # A resumed run is a new convergence audit, potentially under changed
        # damping or proximal controls. Preserve the economic strategy and
        # history, but require the continuation to earn its status afresh.
        state.converged = False
        state.stop_reason = "resumed; convergence not yet assessed"
        state.consecutive_converged_sweeps = 0

    clean_and_bound_state(state, cfg, nodes, times, data.eta)
    consecutive_failures = {investor.investor_id: 0 for investor in cfg.investors}
    final_iteration = state.iteration + cfg.max_iters
    responses: list[StrategicBestResponse] = []

    for iteration in range(state.iteration + 1, final_iteration + 1):
        state.iteration = iteration
        iteration_proximal_coefficient = proximal_coefficient_for_iteration(cfg, iteration)
        iteration_cfg = replace(
            cfg,
            strategic_proximal_penalty_eur_per_mw2_day=iteration_proximal_coefficient,
        )
        power_start = dict(state.x_power)
        energy_start = dict(state.x_energy)
        charge_start = dict(state.offer_charge)
        discharge_start = dict(state.offer_discharge)
        charge_price_start = dict(state.bid_price_charge)
        discharge_price_start = dict(state.offer_price_discharge)
        responses = []

        def numerical_guesses(investor: InvestorConfig):
            investor_id = investor.investor_id
            guess_power = {
                node: (
                    state.x_power[investor_id, node]
                    if state.x_power[investor_id, node]
                    > cfg.capacity_cleanup_tol_mw_mwh
                    else cfg.seed_power_mw
                )
                for node in nodes
            }
            guess_energy = {
                node: (
                    state.x_energy[investor_id, node]
                    if state.x_energy[investor_id, node]
                    > cfg.capacity_cleanup_tol_mw_mwh
                    else cfg.seed_power_mw * cfg.seed_ratio_hours
                )
                for node in nodes
            }
            return guess_power, guess_energy

        if cfg.update_rule == "jacobi":
            tasks = []
            for investor in cfg.investors:
                guess_power, guess_energy = numerical_guesses(investor)
                tasks.append(
                    best_response_task(
                        data,
                        iteration_cfg,
                        investor,
                        state,
                        nodes,
                        times,
                        initial_guess_power=guess_power,
                        initial_guess_energy=guess_energy,
                    )
                )
            responses = solve_best_response_tasks(
                tasks,
                parallel_workers=cfg.strategic_parallel_workers,
                tee=tee,
            )
        else:
            for investor in cfg.investors:
                guess_power, guess_energy = numerical_guesses(investor)
                task = best_response_task(
                    data,
                    iteration_cfg,
                    investor,
                    state,
                    nodes,
                    times,
                    initial_guess_power=guess_power,
                    initial_guess_energy=guess_energy,
                )
                response = solve_best_response_tasks(
                    [task], parallel_workers=1, tee=tee
                )[0]
                responses.append(response)
                if response.ok:
                    apply_damped_update(state, cfg, nodes, times, response, data.eta)

        if cfg.update_rule == "jacobi":
            for response in responses:
                if response.ok:
                    apply_damped_update(state, cfg, nodes, times, response, data.eta)

        project_joint_limit(state, cfg, nodes, times, data.eta)
        all_ok = all(response.ok for response in responses)
        max_rel_power = 0.0
        max_rel_energy = 0.0
        max_rel_offer = 0.0
        max_rel_price = 0.0
        max_abs_capacity = 0.0
        max_abs_offer = 0.0
        max_abs_price = 0.0
        max_undamped_capacity = 0.0
        max_undamped_offer = 0.0
        max_undamped_price = 0.0
        for response in responses:
            investor_id = response.investor_id
            investor_cfg = next(
                investor
                for investor in cfg.investors
                if investor.investor_id == investor_id
            )
            consecutive_failures[investor_id] = 0 if response.ok else consecutive_failures[investor_id] + 1
            rel_power = max(
                relative_delta(state.x_power[investor_id, node], power_start[investor_id, node], cfg.floor_mw)
                for node in nodes
            )
            rel_energy = max(
                relative_delta(state.x_energy[investor_id, node], energy_start[investor_id, node], cfg.floor_mwh)
                for node in nodes
            )
            rel_offer = max(
                max(
                    relative_delta(
                        state.offer_charge[investor_id, node, time_],
                        charge_start[investor_id, node, time_],
                        cfg.floor_mw,
                    ),
                    relative_delta(
                        state.offer_discharge[investor_id, node, time_],
                        discharge_start[investor_id, node, time_],
                        cfg.floor_mw,
                    ),
                )
                for node in nodes
                for time_ in times
            )
            rel_price = (
                max(
                    max(
                        relative_delta(
                            state.bid_price_charge[investor_id, node, time_],
                            charge_price_start[investor_id, node, time_],
                            cfg.strategic_price_floor_eur_per_mwh,
                        ),
                        relative_delta(
                            state.offer_price_discharge[investor_id, node, time_],
                            discharge_price_start[investor_id, node, time_],
                            cfg.strategic_price_floor_eur_per_mwh,
                        ),
                    )
                    for node in nodes
                    for time_ in times
                )
                if cfg.strategic_bid_prices
                else 0.0
            )
            abs_capacity = math.sqrt(
                sum(
                    (
                        state.x_power[investor_id, node]
                        - power_start[investor_id, node]
                    )
                    ** 2
                    + (
                        (
                            state.x_energy[investor_id, node]
                            - energy_start[investor_id, node]
                        )
                        / cfg.strategic_proximal_energy_scale_hours
                    )
                    ** 2
                    for node in nodes
                )
            )
            abs_offer = math.sqrt(
                sum(
                    (
                        (
                            state.x_power[investor_id, node]
                            - state.offer_charge[investor_id, node, time_]
                        )
                        - (
                            power_start[investor_id, node]
                            - charge_start[investor_id, node, time_]
                        )
                    )
                    ** 2
                    + (
                        (
                            state.x_power[investor_id, node]
                            - state.offer_discharge[investor_id, node, time_]
                        )
                        - (
                            power_start[investor_id, node]
                            - discharge_start[investor_id, node, time_]
                        )
                    )
                    ** 2
                    for node in nodes
                    for time_ in times
                )
                / (2.0 * len(times))
            )
            abs_price = (
                math.sqrt(
                    sum(
                        (
                            state.bid_price_charge[investor_id, node, time_]
                            - charge_price_start[investor_id, node, time_]
                        )
                        ** 2
                        + (
                            state.offer_price_discharge[investor_id, node, time_]
                            - discharge_price_start[investor_id, node, time_]
                        )
                        ** 2
                        for node in nodes
                        for time_ in times
                    )
                    / (2.0 * len(times))
                )
                if cfg.strategic_bid_prices
                else 0.0
            )
            undamped_capacity = math.sqrt(
                sum(
                    (response.proposed_power[node] - power_start[investor_id, node]) ** 2
                    + (
                        (
                            response.proposed_energy[node]
                            - energy_start[investor_id, node]
                        )
                        / cfg.strategic_proximal_energy_scale_hours
                    )
                    ** 2
                    for node in nodes
                )
            )
            undamped_offer = math.sqrt(
                sum(
                    (
                        (
                            response.proposed_power[node]
                            - response.proposed_offer_charge[node, time_]
                        )
                        - (
                            power_start[investor_id, node]
                            - charge_start[investor_id, node, time_]
                        )
                    )
                    ** 2
                    + (
                        (
                            response.proposed_power[node]
                            - response.proposed_offer_discharge[node, time_]
                        )
                        - (
                            power_start[investor_id, node]
                            - discharge_start[investor_id, node, time_]
                        )
                    )
                    ** 2
                    for node in nodes
                    for time_ in times
                )
                / (2.0 * len(times))
            )
            if cfg.strategic_bid_prices:
                canonical_response_prices = {}
                for node in nodes:
                    for time_ in times:
                        charge_price = _effective_bid_price(
                            response.proposed_bid_price_charge[node, time_],
                            response.proposed_offer_charge[node, time_],
                            -0.5 * investor_cfg.degradation_eur_per_mwh,
                            cfg.strategic_bid_price_bound_eur_per_mwh,
                        )
                        discharge_price = _effective_bid_price(
                            response.proposed_offer_price_discharge[node, time_],
                            response.proposed_offer_discharge[node, time_],
                            0.5 * investor_cfg.degradation_eur_per_mwh,
                            cfg.strategic_bid_price_bound_eur_per_mwh,
                        )
                        charge_price = min(
                            charge_price,
                            (data.eta**2)
                            * cfg.strategic_bid_price_bound_eur_per_mwh,
                        )
                        discharge_price = max(
                            discharge_price,
                            charge_price / (data.eta**2),
                        )
                        canonical_response_prices[node, time_] = (
                            charge_price,
                            discharge_price,
                        )
                undamped_price = math.sqrt(
                    sum(
                        (
                            canonical_response_prices[node, time_][0]
                            - charge_price_start[investor_id, node, time_]
                        )
                        ** 2
                        + (
                            canonical_response_prices[node, time_][1]
                            - discharge_price_start[investor_id, node, time_]
                        )
                        ** 2
                        for node in nodes
                        for time_ in times
                    )
                    / (2.0 * len(times))
                )
            else:
                undamped_price = 0.0
            max_rel_power = max(max_rel_power, rel_power)
            max_rel_energy = max(max_rel_energy, rel_energy)
            max_rel_offer = max(max_rel_offer, rel_offer)
            max_rel_price = max(max_rel_price, rel_price)
            max_abs_capacity = max(max_abs_capacity, abs_capacity)
            max_abs_offer = max(max_abs_offer, abs_offer)
            max_abs_price = max(max_abs_price, abs_price)
            max_undamped_capacity = max(
                max_undamped_capacity, undamped_capacity
            )
            max_undamped_offer = max(max_undamped_offer, undamped_offer)
            max_undamped_price = max(max_undamped_price, undamped_price)
            state.history.append(
                {
                    "iteration": iteration,
                    "investor": investor_id,
                    "termination": response.termination,
                    "solve_seconds": response.solve_seconds,
                    "optimistic_mpec_profit_eur_per_day": response.optimistic_mpec_profit_eur_per_day,
                    "max_access_shadow_price_eur_per_mw_day": max(response.access_shadow_price_eur_per_mw_day.values()),
                    "strong_duality_gap": response.strong_duality_gap,
                    "total_power_mw": sum(state.x_power[investor_id, node] for node in nodes),
                    "total_energy_mwh": sum(state.x_energy[investor_id, node] for node in nodes),
                    "max_rel_delta_power": rel_power,
                    "max_rel_delta_energy": rel_energy,
                    "abs_capacity_step_mw_equivalent": abs_capacity,
                    "abs_offer_step_mw": abs_offer,
                    "abs_price_step_eur_per_mwh": abs_price,
                    "max_undamped_delta_power_mw": max(
                        abs(response.proposed_power[node] - power_start[investor_id, node]) for node in nodes
                    ),
                    "undamped_capacity_residual_mw_equivalent": undamped_capacity,
                    "undamped_offer_residual_mw": undamped_offer,
                    "undamped_price_residual_eur_per_mwh": undamped_price,
                }
            )
            state.offer_convergence_history.append(
                {
                    "iteration": iteration,
                    "investor": investor_id,
                    "max_rel_delta_offer": rel_offer,
                    "max_rel_delta_price": rel_price,
                    "abs_offer_step_mw": abs_offer,
                    "abs_price_step_eur_per_mwh": abs_price,
                    "undamped_offer_residual_mw": undamped_offer,
                    "undamped_price_residual_eur_per_mwh": undamped_price,
                    "charge_offer_capacity_hours_mwh": sum(
                        state.offer_charge[investor_id, node, time_] for node in nodes for time_ in times
                    ),
                    "discharge_offer_capacity_hours_mwh": sum(
                        state.offer_discharge[investor_id, node, time_] for node in nodes for time_ in times
                    ),
                    "mean_charge_bid_price_eur_per_mwh": sum(
                        state.bid_price_charge[investor_id, node, time_]
                        for node in nodes
                        for time_ in times
                    ) / (len(nodes) * len(times)),
                    "mean_discharge_offer_price_eur_per_mwh": sum(
                        state.offer_price_discharge[investor_id, node, time_]
                        for node in nodes
                        for time_ in times
                    ) / (len(nodes) * len(times)),
                }
            )
            state.proximal_history.append(
                {
                    "iteration": iteration,
                    "investor": investor_id,
                    "proximal_coefficient_eur_per_mw2_day": iteration_proximal_coefficient,
                    "unpenalized_profit_eur_per_day": response.optimistic_mpec_profit_eur_per_day,
                    "proximal_penalty_eur_per_day": response.proximal_penalty_eur_per_day,
                    "strategic_epsilon_penalty": cfg.strategic_epsilon_penalty,
                    "epsilon_penalty_eur_per_day": response.epsilon_penalty_eur_per_day,
                    "penalized_objective_eur_per_day": response.penalized_objective_eur_per_day,
                }
            )
            for node in nodes:
                state.trajectory.append(
                    {
                        "iteration": iteration,
                        "investor": investor_id,
                        "node": node,
                        "x_power_mw": state.x_power[investor_id, node],
                        "x_energy_mwh": state.x_energy[investor_id, node],
                        "proposed_x_power_mw": response.proposed_power[node],
                        "private_headroom_limit_mw": response.private_headroom_limit_mw[node],
                        "private_headroom_slack_mw": max(0.0, response.private_headroom_limit_mw[node] - response.proposed_power[node]),
                        "access_shadow_price_eur_per_mw_day": response.access_shadow_price_eur_per_mw_day[node],
                        "headroom_complementarity_residual_eur_per_day": response.access_shadow_price_eur_per_mw_day[node]
                        * max(0.0, response.private_headroom_limit_mw[node] - response.proposed_power[node]),
                        "headroom_mw": cfg.node_limit_mw - sum(state.x_power[other.investor_id, node] for other in cfg.investors),
                    }
                )

        sweep_converged = (
            all_ok
            and max_abs_capacity < cfg.strategic_tol_abs_capacity_mw
            and max_abs_offer < cfg.strategic_tol_abs_offer_mw
            and max_abs_price < cfg.strategic_tol_abs_price_eur_per_mwh
            and max_undamped_capacity < cfg.strategic_tol_abs_capacity_mw
            and max_undamped_offer < cfg.strategic_tol_abs_offer_mw
            and max_undamped_price < cfg.strategic_tol_abs_price_eur_per_mwh
        )
        state.consecutive_converged_sweeps = (
            state.consecutive_converged_sweeps + 1 if sweep_converged else 0
        )
        for row in state.history[-len(responses):]:
            row["converged_sweep_streak"] = state.consecutive_converged_sweeps

        print(
            f"iter {iteration:2d} [strategic {cfg.update_rule}] abs "
            f"dStrategy={max_abs_capacity:.4f} MW-eq "
            f"dOffer={max_abs_offer:.4f} MW "
            f"dPrice={max_abs_price:.4f} EUR/MWh; "
            f"raw=({max_undamped_capacity:.4f} MW-eq, "
            f"{max_undamped_offer:.4f} MW, "
            f"{max_undamped_price:.4f} EUR/MWh); "
            f"streak={state.consecutive_converged_sweeps}/"
            f"{cfg.strategic_consecutive_converged_sweeps}; "
            f"max_rel dP={max_rel_power:.4f} dE={max_rel_energy:.4f} "
            f"dOffer={max_rel_offer:.4f} dPrice={max_rel_price:.4f}; "
            f"rho={iteration_proximal_coefficient:g}; "
            + ", ".join(
                f"{response.investor_id}={response.optimistic_mpec_profit_eur_per_day:,.0f}"
                if response.ok
                else f"{response.investor_id}=FAILED"
                for response in responses
            )
        )

        should_stop = False
        if any(count >= cfg.max_consecutive_failures for count in consecutive_failures.values()):
            state.stop_reason = "aborted: repeated strategic MPEC failures"
            should_stop = True
        elif (
            state.consecutive_converged_sweeps
            >= cfg.strategic_consecutive_converged_sweeps
        ):
            state.converged = True
            state.stop_reason = (
                f"converged in {iteration} iterations after "
                f"{cfg.strategic_consecutive_converged_sweeps} consecutive sweeps"
            )
            should_stop = True
        state.final_models = {
            response.investor_id: response.model
            for response in responses
            if response.model is not None
        }
        state.final_selected_prices = {
            response.investor_id: response.selected_price_eur_per_mwh
            for response in responses
            if response.selected_price_eur_per_mwh
        }
        state.final_access_shadow_prices = {
            response.investor_id: response.access_shadow_price_eur_per_mw_day
            for response in responses
            if response.ok
        }
        if checkpoint_callback:
            checkpoint_callback(state)
        if should_stop:
            break
    else:
        state.stop_reason = f"max iterations ({final_iteration}) reached without convergence"
    print(state.stop_reason)
    return state


def export_checkpoint(output_dir: Path, state: StrategicEpecState, cfg: EpecConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "status": state.stop_reason,
        "iteration": state.iteration,
        "converged": state.converged,
        "investor_solve_order": [
            investor.investor_id for investor in cfg.investors
        ],
        "update_rule": cfg.update_rule,
        "parallel_workers": cfg.strategic_parallel_workers,
        "node_limit_mw": cfg.node_limit_mw,
        "price_lower_bound_eur_per_mwh": cfg.price_lower_bound_eur_per_mwh,
        "price_upper_bound_eur_per_mwh": cfg.price_upper_bound_eur_per_mwh,
        "initialization_method": state.initialization_method,
        "initializer_summary": state.initializer_summary,
        "proximal_penalty_eur_per_mw2_day": (
            cfg.strategic_proximal_penalty_eur_per_mw2_day
        ),
        "proximal_energy_scale_hours": cfg.strategic_proximal_energy_scale_hours,
        "proximal_price_scale_eur_per_mwh": (
            cfg.strategic_proximal_price_scale_eur_per_mwh
        ),
        "proximal_penalty_step_eur_per_mw2_day": (
            cfg.strategic_proximal_penalty_step_eur_per_mw2_day
        ),
        "proximal_penalty_step_iterations": (
            cfg.strategic_proximal_penalty_step_iterations
        ),
        "proximal_penalty_initial_zero_iterations": (
            cfg.strategic_proximal_penalty_initial_zero_iterations
        ),
        "strategic_bid_prices": cfg.strategic_bid_prices,
        "strategic_bid_price_bound_eur_per_mwh": (
            cfg.strategic_bid_price_bound_eur_per_mwh
        ),
        "strategic_epsilon_penalty": cfg.strategic_epsilon_penalty,
        "tol_abs_capacity_mw_equivalent": cfg.strategic_tol_abs_capacity_mw,
        "tol_abs_offer_mw": cfg.strategic_tol_abs_offer_mw,
        "tol_abs_price_eur_per_mwh": cfg.strategic_tol_abs_price_eur_per_mwh,
        "consecutive_converged_sweeps_required": (
            cfg.strategic_consecutive_converged_sweeps
        ),
        "consecutive_converged_sweeps": state.consecutive_converged_sweeps,
        "x_power_mw": {f"{i}|{n}": v for (i, n), v in state.x_power.items()},
        "x_energy_mwh": {f"{i}|{n}": v for (i, n), v in state.x_energy.items()},
        "offer_charge_mw": {f"{i}|{n}|{t}": v for (i, n, t), v in state.offer_charge.items()},
        "offer_discharge_mw": {f"{i}|{n}|{t}": v for (i, n, t), v in state.offer_discharge.items()},
        "bid_price_charge_eur_per_mwh": {
            f"{i}|{n}|{t}": v for (i, n, t), v in state.bid_price_charge.items()
        },
        "offer_price_discharge_eur_per_mwh": {
            f"{i}|{n}|{t}": v
            for (i, n, t), v in state.offer_price_discharge.items()
        },
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def load_checkpoint(
    path: Path,
    data,
    cfg: EpecConfig,
    *,
    allow_update_rule_change: bool = False,
) -> StrategicEpecState:
    checkpoint_path = path / "checkpoint.json" if path.is_dir() else path
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_update_rule = raw.get("update_rule")
    if (
        checkpoint_update_rule is not None
        and checkpoint_update_rule != cfg.update_rule
        and not allow_update_rule_change
    ):
        raise ValueError(
            "checkpoint update rule does not match this run: "
            f"{checkpoint_update_rule!r} != {cfg.update_rule!r}"
        )

    def capacities(field):
        return {
            tuple(key.split("|", 1)): float(value_)
            for key, value_ in raw[field].items()
        }

    def offers(field):
        restored = {}
        for key, value_ in raw[field].items():
            investor, node, time_ = key.split("|")
            restored[investor, node, int(time_)] = float(value_)
        return restored

    charge_prices = (
        offers("bid_price_charge_eur_per_mwh")
        if "bid_price_charge_eur_per_mwh" in raw
        else {
            (investor.investor_id, node, int(time_)): -0.5
            * investor.degradation_eur_per_mwh
            for investor in cfg.investors
            for node in data.nodes
            for time_ in data.times
        }
    )
    discharge_prices = (
        offers("offer_price_discharge_eur_per_mwh")
        if "offer_price_discharge_eur_per_mwh" in raw
        else {
            (investor.investor_id, node, int(time_)): 0.5
            * investor.degradation_eur_per_mwh
            for investor in cfg.investors
            for node in data.nodes
            for time_ in data.times
        }
    )
    state = StrategicEpecState(
        capacities("x_power_mw"),
        capacities("x_energy_mwh"),
        offers("offer_charge_mw"),
        offers("offer_discharge_mw"),
        charge_prices,
        discharge_prices,
        iteration=int(raw["iteration"]),
        initialization_method=str(raw.get("initialization_method", "checkpoint_resume")),
        initializer_summary=dict(raw.get("initializer_summary", {})),
        consecutive_converged_sweeps=int(
            raw.get("consecutive_converged_sweeps", 0)
        ),
    )
    clean_and_bound_state(
        state, cfg, list(data.nodes), [int(t) for t in data.times], data.eta
    )
    return state


def export_final(output_dir, data, state, cfg, settlement, data_path, calibration) -> None:
    export_epec_results(output_dir, data, state, cfg, settlement, data_path)
    export_checkpoint(output_dir, state, cfg)
    units = [investor.investor_id for investor in cfg.investors]
    reference = settlement["reference_model"]
    prices = settlement["reference_lambda"]
    _write_csv(
        output_dir / "strategic_quantity_offers.csv",
        [
            "investor", "hour", "node", "installed_power_mw",
            "charge_offer_mw", "charge_bid_price_eur_per_mwh",
            "accepted_charge_mw", "discharge_offer_mw",
            "discharge_offer_price_eur_per_mwh", "accepted_discharge_mw",
            "joint_lambda_eur_per_mwh",
        ],
        [
            {
                "investor": investor,
                "hour": time_,
                "node": node,
                "installed_power_mw": state.x_power[investor, node],
                "charge_offer_mw": state.offer_charge[investor, node, int(time_)],
                "charge_bid_price_eur_per_mwh": state.bid_price_charge[
                    investor, node, int(time_)
                ],
                "accepted_charge_mw": value(reference.P_charge[investor, node, time_]),
                "discharge_offer_mw": state.offer_discharge[
                    investor, node, int(time_)
                ],
                "discharge_offer_price_eur_per_mwh": state.offer_price_discharge[
                    investor, node, int(time_)
                ],
                "accepted_discharge_mw": value(reference.P_discharge[investor, node, time_]),
                "joint_lambda_eur_per_mwh": prices[node, time_],
            }
            for investor in units
            for time_ in reference.T
            for node in reference.N
        ],
    )
    _write_csv(
        output_dir / "offer_convergence_history.csv",
        [
            "iteration", "investor", "max_rel_delta_offer", "max_rel_delta_price",
            "abs_offer_step_mw", "abs_price_step_eur_per_mwh",
            "undamped_offer_residual_mw", "undamped_price_residual_eur_per_mwh",
            "charge_offer_capacity_hours_mwh", "discharge_offer_capacity_hours_mwh",
            "mean_charge_bid_price_eur_per_mwh",
            "mean_discharge_offer_price_eur_per_mwh",
        ],
        state.offer_convergence_history,
    )
    _write_csv(
        output_dir / "proximal_history.csv",
        [
            "iteration", "investor", "proximal_coefficient_eur_per_mw2_day",
            "unpenalized_profit_eur_per_day",
            "proximal_penalty_eur_per_day", "strategic_epsilon_penalty",
            "epsilon_penalty_eur_per_day", "penalized_objective_eur_per_day",
        ],
        state.proximal_history,
    )
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": (
                "multi_investor_strategic_two_sided_price_quantity_bids"
                if cfg.strategic_bid_prices
                else "multi_investor_strategic_hourly_quantity_offers"
            ),
            "investor_solve_order": [
                investor.investor_id for investor in cfg.investors
            ],
            "update_rule": cfg.update_rule,
            "parallel_workers": cfg.strategic_parallel_workers,
            "generator_calibration": calibration,
            "price_lower_bound_eur_per_mwh": cfg.price_lower_bound_eur_per_mwh,
            "price_upper_bound_eur_per_mwh": cfg.price_upper_bound_eur_per_mwh,
            "offer_convergence_required": True,
            "proximal_penalty_eur_per_mw2_day": (
                cfg.strategic_proximal_penalty_eur_per_mw2_day
            ),
            "proximal_energy_scale_hours": cfg.strategic_proximal_energy_scale_hours,
            "proximal_price_scale_eur_per_mwh": (
                cfg.strategic_proximal_price_scale_eur_per_mwh
            ),
            "proximal_penalty_step_eur_per_mw2_day": (
                cfg.strategic_proximal_penalty_step_eur_per_mw2_day
            ),
            "proximal_penalty_step_iterations": (
                cfg.strategic_proximal_penalty_step_iterations
            ),
            "proximal_penalty_initial_zero_iterations": (
                cfg.strategic_proximal_penalty_initial_zero_iterations
            ),
            "final_proximal_coefficient_eur_per_mw2_day": (
                proximal_coefficient_for_iteration(cfg, state.iteration)
            ),
            "proximal_penalty_target": (
                "MW/MWh capacity, withheld charge/discharge quantity, and normalized "
                "two-sided bid-price changes relative to the previous strategy"
                if cfg.strategic_bid_prices
                else "MW/MWh capacity and withheld charge/discharge quantity changes "
                "relative to the investor's previous strategy"
            ),
            "equilibrium_interpretation": (
                "regularized strategy-selection candidate; requires an "
                "unpenalized best-response check"
                if strategy_regularization_enabled(cfg)
                else "unregularized optimistic EPEC candidate"
            ),
            "strategy_space": (
                "nodal MW/MWh investment plus hourly charging buy-bid and "
                "discharging sell-offer price/quantity pairs"
                if cfg.strategic_bid_prices
                else "nodal MW/MWh investment plus hourly charge/discharge quantity offers"
            ),
            "strategic_bid_prices": cfg.strategic_bid_prices,
            "strategic_bid_price_bound_eur_per_mwh": (
                cfg.strategic_bid_price_bound_eur_per_mwh
            ),
            "strategic_epsilon_penalty": cfg.strategic_epsilon_penalty,
            "strategic_epsilon_penalty_target": (
                "direct squared charging buy-bid and discharging sell-offer prices; "
                "no direct capacity, energy, or quantity-offer term"
            ),
            "convergence_metric": (
                "applied and raw absolute strategy norms; inactive bid prices "
                "are canonicalized; relative maxima are diagnostics"
            ),
            "tol_abs_capacity_mw_equivalent": cfg.strategic_tol_abs_capacity_mw,
            "tol_abs_offer_mw": cfg.strategic_tol_abs_offer_mw,
            "tol_abs_price_eur_per_mwh": cfg.strategic_tol_abs_price_eur_per_mwh,
            "consecutive_converged_sweeps_required": (
                cfg.strategic_consecutive_converged_sweeps
            ),
            "consecutive_converged_sweeps": state.consecutive_converged_sweeps,
            "strategic_bid_price_semantics": (
                "complete submitted prices replace private degradation in ISO clearing; "
                "physical degradation remains in investor profit"
                if cfg.strategic_bid_prices
                else "ISO clearing uses physical degradation costs"
            ),
            "rival_representation": (
                "separate battery per investor with frozen nodal MW/MWh and "
                "hourly charge/discharge price-quantity bids"
                if cfg.strategic_bid_prices
                else "separate battery per investor with frozen nodal MW/MWh and "
                "hourly charge/discharge quantity offers"
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_config_path = output_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config.update(
        {
            "experiment": (
                "multi_investor_strategic_two_sided_price_quantity_bids"
                if cfg.strategic_bid_prices
                else "multi_investor_strategic_hourly_quantity_offers"
            ),
            "investor_solve_order": [
                investor.investor_id for investor in cfg.investors
            ],
            "update_rule": cfg.update_rule,
            "parallel_workers": cfg.strategic_parallel_workers,
            "generator_calibration": calibration,
            "price_lower_bound_eur_per_mwh": cfg.price_lower_bound_eur_per_mwh,
            "price_upper_bound_eur_per_mwh": cfg.price_upper_bound_eur_per_mwh,
            "offer_convergence_required": True,
            "proximal_penalty_eur_per_mw2_day": (
                cfg.strategic_proximal_penalty_eur_per_mw2_day
            ),
            "proximal_energy_scale_hours": cfg.strategic_proximal_energy_scale_hours,
            "proximal_price_scale_eur_per_mwh": (
                cfg.strategic_proximal_price_scale_eur_per_mwh
            ),
            "proximal_penalty_step_eur_per_mw2_day": (
                cfg.strategic_proximal_penalty_step_eur_per_mw2_day
            ),
            "proximal_penalty_step_iterations": (
                cfg.strategic_proximal_penalty_step_iterations
            ),
            "proximal_penalty_initial_zero_iterations": (
                cfg.strategic_proximal_penalty_initial_zero_iterations
            ),
            "strategic_bid_prices": cfg.strategic_bid_prices,
            "strategic_bid_price_bound_eur_per_mwh": (
                cfg.strategic_bid_price_bound_eur_per_mwh
            ),
            "strategic_epsilon_penalty": cfg.strategic_epsilon_penalty,
            "convergence_metric": (
                "applied and raw absolute strategy norms with canonical "
                "inactive bid prices"
            ),
            "tol_abs_capacity_mw_equivalent": cfg.strategic_tol_abs_capacity_mw,
            "tol_abs_offer_mw": cfg.strategic_tol_abs_offer_mw,
            "tol_abs_price_eur_per_mwh": cfg.strategic_tol_abs_price_eur_per_mwh,
            "consecutive_converged_sweeps_required": (
                cfg.strategic_consecutive_converged_sweeps
            ),
        }
    )
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategic-operation EPEC diagonalization")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--investor-set", choices=["portfolio4", "wacc"], default="portfolio4")
    parser.add_argument(
        "--investor-order",
        nargs="+",
        default=None,
        help=(
            "Deterministic solve/result order using configured investor IDs, "
            "for example: --investor-order I3 I1 I4 I2."
        ),
    )
    parser.add_argument(
        "--update-rule",
        choices=["jacobi", "seidel"],
        default="seidel",
        help="Full-sweep strategic update rule.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Independent worker processes for Jacobi best responses.",
    )
    parser.add_argument("--wacc", type=float, nargs="+", default=[0.08, 0.12])
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--max-iters", type=int, default=60)
    parser.add_argument(
        "--tol-rel",
        type=float,
        default=DEFAULT_TOL_REL,
        help="Diagnostic relative tolerance; no longer used to stop this EPEC.",
    )
    parser.add_argument(
        "--tol-abs-capacity-mw",
        type=float,
        default=0.5,
        help="Absolute MW-equivalent norm tolerance for each investor's MW/MWh step.",
    )
    parser.add_argument(
        "--tol-abs-offer-mw",
        type=float,
        default=0.25,
        help="Absolute RMS tolerance for changes in withheld charge/discharge MW.",
    )
    parser.add_argument(
        "--tol-abs-price-eur-per-mwh",
        type=float,
        default=0.5,
        help="Absolute RMS tolerance for strategic bid-price changes.",
    )
    parser.add_argument(
        "--consecutive-converged-sweeps",
        type=int,
        default=3,
        help="Number of consecutive sweeps that must satisfy all absolute tolerances.",
    )
    parser.add_argument("--floor-mw", type=float, default=DEFAULT_FLOOR_MW)
    parser.add_argument("--floor-mwh", type=float, default=DEFAULT_FLOOR_MWH)
    parser.add_argument("--seed-power-mw", type=float, default=DEFAULT_INITIAL_POWER_MW)
    parser.add_argument("--seed-ratio-hours", type=float, default=4.0)
    parser.add_argument("--initializer-snapshot-power-mw", type=float, default=0.0)
    parser.add_argument("--initializer-snapshot-ratio-hours", type=float, default=4.0)
    parser.add_argument("--skip-jacobi-initializer", action="store_true")
    parser.add_argument("--max-cpu-time", type=float, default=180.0)
    parser.add_argument("--solver-tol", type=float, default=DEFAULT_SOLVER_TOL)
    parser.add_argument(
        "--price-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_PRICE_BOUND_EUR_PER_MWH,
        help="Legacy symmetric absolute bound for lam and lam_sys.",
    )
    parser.add_argument(
        "--price-lower-bound-eur-per-mwh",
        type=float,
        default=None,
        help="Optional asymmetric lower bound for lam and lam_sys.",
    )
    parser.add_argument(
        "--price-upper-bound-eur-per-mwh",
        type=float,
        default=None,
        help="Optional asymmetric upper bound for lam and lam_sys.",
    )
    parser.add_argument("--dual-bound-eur-per-mwh", type=float, default=DEFAULT_DUAL_BOUND_EUR_PER_MWH)
    parser.add_argument(
        "--demand-model",
        choices=["fixed", "quadratic"],
        default="fixed",
        help="Lower-level demand representation. The maintained base model uses fixed demand.",
    )
    parser.add_argument(
        "--demand-curve-alpha-eur-per-mwh",
        type=float,
        default=default_quadratic_demand_curve().alpha,
        help="Quadratic mode: marginal willingness to pay at zero curtailment.",
    )
    parser.add_argument(
        "--demand-curve-beta-eur-per-mwh-per-share",
        type=float,
        default=default_quadratic_demand_curve().beta,
        help="Quadratic mode: marginal-WTP increase over one unit of curtailed demand share.",
    )
    parser.add_argument(
        "--strategic-bid-prices",
        action="store_true",
        help=(
            "Enable strategic charging buy-bid and discharging sell-offer prices; "
            "without this flag the quantity-only experiment is reproduced."
        ),
    )
    parser.add_argument(
        "--bid-price-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_PRICE_BOUND_EUR_PER_MWH,
        help="Symmetric bound on strategic charge/discharge bid prices.",
    )
    parser.add_argument(
        "--floor-price-eur-per-mwh",
        type=float,
        default=1.0,
        help="Denominator floor used by the relative bid-price convergence test.",
    )
    parser.add_argument(
        "--strategic-epsilon-penalty",
        type=float,
        default=0.0,
        help=(
            "Coefficient epsilon on the direct sum of squared strategic charging "
            "buy-bid and discharging sell-offer prices; capacity, energy, and "
            "quantity offers are excluded; zero preserves the prior objective."
        ),
    )
    parser.add_argument("--dispatch-regularization", type=float, default=0.0)
    parser.add_argument(
        "--lower-level-optimality",
        choices=["strong-duality", "tikhonov-strong-duality"],
        default="strong-duality",
        help="Embedded ISO optimality formulation.",
    )
    parser.add_argument(
        "--dual-tikhonov-gamma",
        type=float,
        default=0.0,
        help="Positive finite-gamma coefficient required by Tikhonov strong duality.",
    )
    parser.add_argument(
        "--proximal-penalty-eur-per-mw2-day",
        type=float,
        default=0.0,
        help=(
            "Optional coefficient on squared changes from each investor's previous "
            "MW-equivalent strategy; zero preserves the unregularized game."
        ),
    )
    parser.add_argument(
        "--proximal-energy-scale-hours",
        type=float,
        default=4.0,
        help="Hours used to convert MWh changes to MW-equivalent proximal distance.",
    )
    parser.add_argument(
        "--proximal-price-scale-eur-per-mwh",
        type=float,
        default=10.0,
        help="Price change corresponding to one normalized proximal-distance unit.",
    )
    parser.add_argument(
        "--proximal-penalty-step-eur-per-mw2-day",
        type=float,
        default=0.0,
        help=(
            "Optional staircase increment: the first block has zero penalty, "
            "then each block increases the coefficient by this amount."
        ),
    )
    parser.add_argument(
        "--proximal-penalty-step-iters",
        type=int,
        default=5,
        help="Number of iterations per staircase proximal-penalty block.",
    )
    parser.add_argument(
        "--proximal-penalty-zero-iters",
        type=int,
        default=5,
        help="Initial iterations with zero staircase proximal penalty.",
    )
    parser.add_argument("--capacity-cleanup-tol", type=float, default=DEFAULT_CAPACITY_CLEANUP_TOL_MW_MWH)
    parser.add_argument("--conventional-capacity-adder-mw", type=float, default=0.0)
    parser.add_argument("--peaker-node", type=str, default=None)
    parser.add_argument("--peaker-capacity-mw", type=float, default=0.0)
    parser.add_argument("--peaker-cost-eur-per-mwh", type=float, default=95.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "seidel_scarcity95")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--allow-resume-update-rule-change",
        action="store_true",
        help=(
            "Allow a checkpoint strategy to seed a different Jacobi/Seidel "
            "update rule. Intended only for explicit algorithm diagnostics."
        ),
    )
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.damping <= 1.0:
        raise SystemExit("--damping must be in (0, 1].")
    if args.max_iters <= 0:
        raise SystemExit("--max-iters must be positive.")
    if args.parallel_workers <= 0:
        raise SystemExit("--parallel-workers must be positive.")
    if args.update_rule == "seidel" and args.parallel_workers != 1:
        raise SystemExit("--parallel-workers greater than one requires --update-rule jacobi.")
    if args.tee and args.parallel_workers > 1:
        raise SystemExit("--tee is not supported with parallel strategic solves.")
    if (
        args.tol_abs_capacity_mw <= 0.0
        or args.tol_abs_offer_mw <= 0.0
        or args.tol_abs_price_eur_per_mwh <= 0.0
    ):
        raise SystemExit("All absolute convergence tolerances must be positive.")
    if args.consecutive_converged_sweeps <= 0:
        raise SystemExit("--consecutive-converged-sweeps must be positive.")
    if args.solver_tol <= 0.0:
        raise SystemExit("--solver-tol must be positive.")
    if args.price_bound_eur_per_mwh <= 0.0 or args.dual_bound_eur_per_mwh <= 0.0:
        raise SystemExit("Price and dual bounds must be positive.")
    if args.demand_curve_beta_eur_per_mwh_per_share <= 0.0:
        raise SystemExit("--demand-curve-beta-eur-per-mwh-per-share must be positive.")
    effective_price_lower = (
        -args.price_bound_eur_per_mwh
        if args.price_lower_bound_eur_per_mwh is None
        else args.price_lower_bound_eur_per_mwh
    )
    effective_price_upper = (
        args.price_bound_eur_per_mwh
        if args.price_upper_bound_eur_per_mwh is None
        else args.price_upper_bound_eur_per_mwh
    )
    if effective_price_lower >= effective_price_upper:
        raise SystemExit("The lower electricity-price bound must be below the upper bound.")
    if args.bid_price_bound_eur_per_mwh <= 0.0:
        raise SystemExit("--bid-price-bound-eur-per-mwh must be positive.")
    if args.floor_price_eur_per_mwh <= 0.0:
        raise SystemExit("--floor-price-eur-per-mwh must be positive.")
    if args.strategic_epsilon_penalty < 0.0:
        raise SystemExit("--strategic-epsilon-penalty must be non-negative.")
    if args.seed_power_mw < 0.0 or args.initializer_snapshot_power_mw < 0.0:
        raise SystemExit("Seed and initializer power must be non-negative.")
    if args.dispatch_regularization < 0.0 or args.capacity_cleanup_tol < 0.0:
        raise SystemExit("Regularization and cleanup tolerance must be non-negative.")
    if args.dual_tikhonov_gamma < 0.0:
        raise SystemExit("--dual-tikhonov-gamma must be non-negative.")
    if (
        args.lower_level_optimality == "tikhonov-strong-duality"
        and args.dual_tikhonov_gamma <= 0.0
    ):
        raise SystemExit(
            "Tikhonov strong duality requires --dual-tikhonov-gamma > 0."
        )
    if (
        args.lower_level_optimality == "tikhonov-strong-duality"
        and (args.demand_model != "fixed" or args.dispatch_regularization != 0.0)
    ):
        raise SystemExit(
            "The strategic Tikhonov EPEC requires fixed demand and zero dispatch regularization."
        )
    if args.proximal_penalty_eur_per_mw2_day < 0.0:
        raise SystemExit("--proximal-penalty-eur-per-mw2-day must be non-negative.")
    if args.proximal_penalty_step_eur_per_mw2_day < 0.0:
        raise SystemExit("--proximal-penalty-step-eur-per-mw2-day must be non-negative.")
    if (
        args.proximal_penalty_eur_per_mw2_day > 0.0
        and args.proximal_penalty_step_eur_per_mw2_day > 0.0
    ):
        raise SystemExit("Choose either a fixed or staircase proximal penalty, not both.")
    if args.proximal_penalty_step_iters <= 0:
        raise SystemExit("--proximal-penalty-step-iters must be positive.")
    if args.proximal_penalty_zero_iters < 0:
        raise SystemExit("--proximal-penalty-zero-iters must be non-negative.")
    if args.proximal_energy_scale_hours <= 0.0:
        raise SystemExit("--proximal-energy-scale-hours must be positive.")
    if args.proximal_price_scale_eur_per_mwh <= 0.0:
        raise SystemExit("--proximal-price-scale-eur-per-mwh must be positive.")
    base_data = load_market_data(args.data)
    data, calibration = apply_generator_calibration(
        base_data,
        conventional_capacity_adder_mw=args.conventional_capacity_adder_mw,
        peaker_node=args.peaker_node,
        peaker_capacity_mw=args.peaker_capacity_mw,
        peaker_cost_eur_per_mwh=args.peaker_cost_eur_per_mwh,
    )
    investors = (
        four_investor_portfolio_profiles(data)
        if args.investor_set == "portfolio4"
        else tuple(
            InvestorConfig(investor_id=f"I{k + 1}", wacc=wacc)
            for k, wacc in enumerate(args.wacc)
        )
    )
    try:
        investors = order_investors(investors, args.investor_order)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cfg = EpecConfig(
        investors=investors,
        node_limit_mw=args.node_limit_mw,
        update_rule=args.update_rule,
        damping=args.damping,
        max_iters=args.max_iters,
        tol_rel=args.tol_rel,
        floor_mw=args.floor_mw,
        floor_mwh=args.floor_mwh,
        seed_power_mw=args.seed_power_mw,
        seed_ratio_hours=args.seed_ratio_hours,
        max_cpu_time=args.max_cpu_time,
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        price_lower_bound_eur_per_mwh=args.price_lower_bound_eur_per_mwh,
        price_upper_bound_eur_per_mwh=args.price_upper_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
        lower_level_optimality=args.lower_level_optimality,
        dual_tikhonov_gamma=args.dual_tikhonov_gamma,
        use_demand_curve=args.demand_model == "quadratic",
        quadratic_demand_alpha_eur_per_mwh=args.demand_curve_alpha_eur_per_mwh,
        quadratic_demand_beta_eur_per_mwh_per_share=(
            args.demand_curve_beta_eur_per_mwh_per_share
        ),
        dispatch_regularization_eur_per_mw2h=args.dispatch_regularization,
        solver_tol=args.solver_tol,
        capacity_cleanup_tol_mw_mwh=args.capacity_cleanup_tol,
        automatic_jacobi_initializer=not args.skip_jacobi_initializer,
        jacobi_initializer_snapshot_power_mw=args.initializer_snapshot_power_mw,
        jacobi_initializer_snapshot_ratio_hours=args.initializer_snapshot_ratio_hours,
        strategic_proximal_penalty_eur_per_mw2_day=(
            args.proximal_penalty_eur_per_mw2_day
        ),
        strategic_proximal_energy_scale_hours=args.proximal_energy_scale_hours,
        strategic_proximal_price_scale_eur_per_mwh=(
            args.proximal_price_scale_eur_per_mwh
        ),
        strategic_proximal_penalty_step_eur_per_mw2_day=(
            args.proximal_penalty_step_eur_per_mw2_day
        ),
        strategic_proximal_penalty_step_iterations=args.proximal_penalty_step_iters,
        strategic_proximal_penalty_initial_zero_iterations=(
            args.proximal_penalty_zero_iters
        ),
        strategic_bid_prices=args.strategic_bid_prices,
        strategic_bid_price_bound_eur_per_mwh=args.bid_price_bound_eur_per_mwh,
        strategic_price_floor_eur_per_mwh=args.floor_price_eur_per_mwh,
        strategic_epsilon_penalty=args.strategic_epsilon_penalty,
        strategic_tol_abs_capacity_mw=args.tol_abs_capacity_mw,
        strategic_tol_abs_offer_mw=args.tol_abs_offer_mw,
        strategic_tol_abs_price_eur_per_mwh=(
            args.tol_abs_price_eur_per_mwh
        ),
        strategic_consecutive_converged_sweeps=(
            args.consecutive_converged_sweeps
        ),
        strategic_parallel_workers=args.parallel_workers,
    )
    initial_state = None
    if args.resume_from:
        checkpoint_path = (
            args.resume_from / "checkpoint.json"
            if args.resume_from.is_dir()
            else args.resume_from
        )
        try:
            initial_state = load_checkpoint(
                checkpoint_path,
                data,
                cfg,
                allow_update_rule_change=args.allow_resume_update_rule_change,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot resume strategic EPEC run: {exc}") from exc
        cfg = replace(
            cfg,
            starting_iteration=initial_state.iteration,
            resume_from=str(checkpoint_path),
        )
    print(
        f"Strategic-operation EPEC: {len(investors)} investors, "
        f"solve_order={','.join(i.investor_id for i in investors)}, "
        f"Jacobi initializer={'on' if cfg.automatic_jacobi_initializer else 'off'}, "
        f"Gauss-{cfg.update_rule.capitalize()} damping={cfg.damping}, absolute tolerances="
        f"({cfg.strategic_tol_abs_capacity_mw:g} MW-eq, "
        f"{cfg.strategic_tol_abs_offer_mw:g} MW, "
        f"{cfg.strategic_tol_abs_price_eur_per_mwh:g} EUR/MWh) for "
        f"{cfg.strategic_consecutive_converged_sweeps} sweeps, "
        f"parallel_workers={cfg.strategic_parallel_workers}, "
        f"lower_level_optimality={cfg.lower_level_optimality}, "
        f"gamma={cfg.dual_tikhonov_gamma:.3e}, "
        f"demand={args.demand_model}, "
        f"strategic_prices={'on' if cfg.strategic_bid_prices else 'off'}, "
        f"epsilon={cfg.strategic_epsilon_penalty:g}, "
        f"proximal_fixed={cfg.strategic_proximal_penalty_eur_per_mw2_day:g}, "
        f"proximal_step={cfg.strategic_proximal_penalty_step_eur_per_mw2_day:g} "
        f"EUR/MW^2/day after {cfg.strategic_proximal_penalty_initial_zero_iterations} "
        f"zero-penalty iterations, then every "
        f"{cfg.strategic_proximal_penalty_step_iterations} iterations, "
        f"calibration={calibration}"
    )
    if cfg.resume_from:
        print(
            f"Resuming from iteration {initial_state.iteration}; "
            f"--max-iters={cfg.max_iters} means additional {cfg.update_rule} sweeps."
        )
    checkpoint_callback = None
    if not args.no_export:
        checkpoint_callback = lambda state: export_checkpoint(args.output_dir, state, cfg)
        print(f"Strategic checkpoints: {args.output_dir}")
    state = run_epec(
        data,
        cfg,
        tee=args.tee,
        checkpoint_callback=checkpoint_callback,
        initial_state=initial_state,
    )
    quad = configured_demand_curve(cfg)
    settlement = compute_joint_settlement(data, quad, state, cfg)
    print_epec_summary(state, cfg, settlement)
    if not args.no_export:
        export_final(args.output_dir, data, state, cfg, settlement, args.data, calibration)
        print(f"Wrote strategic-operation EPEC outputs to {args.output_dir}")
    return 0 if state.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
