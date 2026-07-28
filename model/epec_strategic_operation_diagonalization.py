"""Multi-investor strategic-operation EPEC via Jacobi-initialized Gauss-Seidel.

This driver is deliberately separate from ``epec_diagonalization.py``. The
maintained driver keeps storage fully available to the ISO; this experiment
adds hourly charge/discharge quantity offers to every investor's strategy.

A fresh run first solves one common-snapshot Gauss-Jacobi best-response sweep,
projects overloaded nodes proportionally, and then starts damped Gauss-Seidel.
Capacities and effective hourly offers are both frozen for rivals and updated
for the active investor.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
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
    default_quadratic_demand_curve,
    initialize_from_reference_dispatch,
    investment_headroom_shadow_price,
    load_market_data,
    value,
)
from single_investor_mpec_results import _write_csv
from solver_utils import get_ipopt_solver


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "epec_strategic_operation"
OFFER_CANONICAL_SLACK_MW = 1.0e-4


@dataclass
class StrategicBestResponse:
    investor_id: str
    termination: str
    solve_seconds: float
    proposed_power: dict[str, float]
    proposed_energy: dict[str, float]
    proposed_offer_charge: dict[tuple[str, int], float]
    proposed_offer_discharge: dict[tuple[str, int], float]
    accepted_charge: dict[tuple[str, int], float]
    accepted_discharge: dict[tuple[str, int], float]
    private_headroom_limit_mw: dict[str, float]
    optimistic_mpec_profit_eur_per_day: float
    access_shadow_price_eur_per_mw_day: dict[str, float]
    strong_duality_gap: float
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
    iteration: int = 0
    converged: bool = False
    stop_reason: str = ""
    history: list[dict] = field(default_factory=list)
    trajectory: list[dict] = field(default_factory=list)
    projection_events: list[dict] = field(default_factory=list)
    final_models: dict[str, pyo.ConcreteModel] = field(default_factory=dict)
    initialization_method: str = "uniform_seed_full_availability"
    initializer_summary: dict = field(default_factory=dict)
    offer_convergence_history: list[dict] = field(default_factory=list)


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
    return rival_power, rival_energy, rival_charge, rival_discharge


def _effective_offer(raw_offer: float, accepted: float, installed_power: float) -> float:
    """Canonicalize economically irrelevant slack offers to full availability."""

    if raw_offer > accepted + OFFER_CANONICAL_SLACK_MW:
        return installed_power
    return min(max(0.0, raw_offer), installed_power)


def solve_best_response(
    data,
    cfg: EpecConfig,
    investor: InvestorConfig,
    rival_power,
    rival_energy,
    rival_charge,
    rival_discharge,
    previous_power: dict[str, float],
    previous_energy: dict[str, float],
    previous_charge: dict[tuple[str, int], float],
    previous_discharge: dict[tuple[str, int], float],
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
        model = build_ieee9_strategic_operation_mpec(
            data,
            investor=investor,
            rival_power_mw_by_unit=rival_power,
            rival_energy_mwh_by_unit=rival_energy,
            rival_offer_charge_mw_by_unit=rival_charge,
            rival_offer_discharge_mw_by_unit=rival_discharge,
            rival_degradation_eur_per_mwh_by_unit={
                other.investor_id: other.degradation_eur_per_mwh
                for other in cfg.investors
                if other.investor_id != investor.investor_id
            },
            node_limit_mw=cfg.node_limit_mw,
            initial_power_mw=cfg.seed_power_mw,
            initial_ratio_hours=cfg.seed_ratio_hours,
            price_bound_eur_per_mwh=cfg.price_bound_eur_per_mwh,
            dual_bound_eur_per_mwh=cfg.dual_bound_eur_per_mwh,
            dispatch_regularization_eur_per_mw2h=cfg.dispatch_regularization_eur_per_mw2h,
            system_price_settlement=cfg.system_price_settlement,
            solver_tol=cfg.solver_tol,
            initialize_model=False,
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

        # Use the existing full-availability market clear as a broad primal/dual
        # warm start, then restore the strategic offer strategy.
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
            {(node, time_): 0.0 for node in nodes for time_ in times},
            {(node, time_): 0.0 for node in nodes for time_ in times},
            headroom,
            float("nan"),
            {node: float("nan") for node in nodes},
            float("nan"),
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
    return StrategicBestResponse(
        investor.investor_id,
        termination,
        seconds,
        proposed_power,
        proposed_energy,
        proposed_charge,
        proposed_discharge,
        accepted_charge,
        accepted_discharge,
        headroom,
        value(model.investor_profit_expr),
        {node: investment_headroom_shadow_price(model, node) for node in nodes},
        abs(value(model.primal_objective_expr) - value(model.dual_objective_expr)),
        model,
    )


def clean_and_bound_state(state: StrategicEpecState, cfg: EpecConfig, nodes, times) -> None:
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


def apply_damped_update(state, cfg, nodes, times, response) -> None:
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
    clean_and_bound_state(state, cfg, nodes, times)


def project_joint_limit(state, cfg, nodes, times) -> None:
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
    clean_and_bound_state(state, cfg, nodes, times)


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
        initialization_method="strategic_jacobi_common_snapshot_projected",
    )
    responses: list[StrategicBestResponse] = []
    numerical_power = {node: cfg.seed_power_mw for node in nodes}
    numerical_energy = {
        node: cfg.seed_power_mw * cfg.seed_ratio_hours for node in nodes
    }
    print(
        "Strategic Jacobi initializer: common snapshot "
        f"{snapshot_power:g} MW/node; numerical guess {cfg.seed_power_mw:g} MW/node"
    )
    for investor in cfg.investors:
        rival = separate_rival_strategies(
            snapshot, cfg, nodes, times, investor.investor_id
        )
        investor_id = investor.investor_id
        response = solve_best_response(
            data,
            cfg,
            investor,
            *rival,
            {node: snapshot.x_power[investor_id, node] for node in nodes},
            {node: snapshot.x_energy[investor_id, node] for node in nodes},
            {
                (node, time_): snapshot.offer_charge[investor_id, node, time_]
                for node in nodes
                for time_ in times
            },
            {
                (node, time_): snapshot.offer_discharge[investor_id, node, time_]
                for node in nodes
                for time_ in times
            },
            initial_guess_power=numerical_power,
            initial_guess_energy=numerical_energy,
            tee=tee,
        )
        if not response.ok:
            raise RuntimeError(
                f"Strategic Jacobi initializer failed for {investor_id}: {response.termination}"
            )
        responses.append(response)
        print(
            f"  {investor_id}: desired {sum(response.proposed_power.values()):.3f} MW / "
            f"{sum(response.proposed_energy.values()):.3f} MWh"
        )

    state = StrategicEpecState({}, {}, {}, {})
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
        "snapshot_power_mw_per_investor_node": snapshot_power,
        "snapshot_ratio_hours": snapshot_ratio,
        "nodes": node_summary,
        "responses": {
            response.investor_id: {
                "desired_power_mw": sum(response.proposed_power.values()),
                "desired_energy_mwh": sum(response.proposed_energy.values()),
                "optimistic_mpec_profit_eur_per_day": response.optimistic_mpec_profit_eur_per_day,
                "strong_duality_gap": response.strong_duality_gap,
            }
            for response in responses
        },
    }
    clean_and_bound_state(state, cfg, nodes, times)
    return state


def run_epec(
    data,
    cfg: EpecConfig,
    *,
    tee: bool = False,
    checkpoint_callback: Callable[[StrategicEpecState], None] | None = None,
    initial_state: StrategicEpecState | None = None,
) -> StrategicEpecState:
    nodes = list(data.nodes)
    times = [int(time_) for time_ in data.times]
    if initial_state is None and cfg.automatic_jacobi_initializer:
        state = projected_jacobi_initial_state(data, cfg, tee=tee)
        if checkpoint_callback:
            checkpoint_callback(state)
        print("Strategic Jacobi initializer complete; starting Gauss-Seidel.")
    elif initial_state is None:
        seed = min(cfg.seed_power_mw, cfg.node_limit_mw / len(cfg.investors))
        state = StrategicEpecState(
            x_power={(i.investor_id, n): seed for i in cfg.investors for n in nodes},
            x_energy={(i.investor_id, n): seed * cfg.seed_ratio_hours for i in cfg.investors for n in nodes},
            offer_charge={(i.investor_id, n, t): seed for i in cfg.investors for n in nodes for t in times},
            offer_discharge={(i.investor_id, n, t): seed for i in cfg.investors for n in nodes for t in times},
        )
    else:
        state = initial_state

    clean_and_bound_state(state, cfg, nodes, times)
    consecutive_failures = {investor.investor_id: 0 for investor in cfg.investors}
    final_iteration = state.iteration + cfg.max_iters
    responses: list[StrategicBestResponse] = []

    for iteration in range(state.iteration + 1, final_iteration + 1):
        state.iteration = iteration
        power_start = dict(state.x_power)
        energy_start = dict(state.x_energy)
        charge_start = dict(state.offer_charge)
        discharge_start = dict(state.offer_discharge)
        responses = []

        for investor in cfg.investors:
            investor_id = investor.investor_id
            rival = separate_rival_strategies(state, cfg, nodes, times, investor_id)
            response = solve_best_response(
                data,
                cfg,
                investor,
                *rival,
                {node: state.x_power[investor_id, node] for node in nodes},
                {node: state.x_energy[investor_id, node] for node in nodes},
                {(node, time_): state.offer_charge[investor_id, node, time_] for node in nodes for time_ in times},
                {(node, time_): state.offer_discharge[investor_id, node, time_] for node in nodes for time_ in times},
                tee=tee,
            )
            responses.append(response)
            if response.ok:
                apply_damped_update(state, cfg, nodes, times, response)

        project_joint_limit(state, cfg, nodes, times)
        all_ok = all(response.ok for response in responses)
        max_rel_power = 0.0
        max_rel_energy = 0.0
        max_rel_offer = 0.0
        for response in responses:
            investor_id = response.investor_id
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
            max_rel_power = max(max_rel_power, rel_power)
            max_rel_energy = max(max_rel_energy, rel_energy)
            max_rel_offer = max(max_rel_offer, rel_offer)
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
                    "max_undamped_delta_power_mw": max(
                        abs(response.proposed_power[node] - power_start[investor_id, node]) for node in nodes
                    ),
                }
            )
            state.offer_convergence_history.append(
                {
                    "iteration": iteration,
                    "investor": investor_id,
                    "max_rel_delta_offer": rel_offer,
                    "charge_offer_capacity_hours_mwh": sum(
                        state.offer_charge[investor_id, node, time_] for node in nodes for time_ in times
                    ),
                    "discharge_offer_capacity_hours_mwh": sum(
                        state.offer_discharge[investor_id, node, time_] for node in nodes for time_ in times
                    ),
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

        print(
            f"iter {iteration:2d} [strategic seidel] max_rel "
            f"dP={max_rel_power:.4f} dE={max_rel_energy:.4f} dOffer={max_rel_offer:.4f}; "
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
            all_ok
            and max_rel_power < cfg.tol_rel
            and max_rel_energy < cfg.tol_rel
            and max_rel_offer < cfg.tol_rel
        ):
            state.converged = True
            state.stop_reason = f"converged in {iteration} iterations"
            should_stop = True
        state.final_models = {
            response.investor_id: response.model
            for response in responses
            if response.model is not None
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
        "node_limit_mw": cfg.node_limit_mw,
        "initialization_method": state.initialization_method,
        "initializer_summary": state.initializer_summary,
        "x_power_mw": {f"{i}|{n}": v for (i, n), v in state.x_power.items()},
        "x_energy_mwh": {f"{i}|{n}": v for (i, n), v in state.x_energy.items()},
        "offer_charge_mw": {f"{i}|{n}|{t}": v for (i, n, t), v in state.offer_charge.items()},
        "offer_discharge_mw": {f"{i}|{n}|{t}": v for (i, n, t), v in state.offer_discharge.items()},
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def load_checkpoint(path: Path, data, cfg: EpecConfig) -> StrategicEpecState:
    checkpoint_path = path / "checkpoint.json" if path.is_dir() else path
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))

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

    state = StrategicEpecState(
        capacities("x_power_mw"),
        capacities("x_energy_mwh"),
        offers("offer_charge_mw"),
        offers("offer_discharge_mw"),
        iteration=int(raw["iteration"]),
        initialization_method=str(raw.get("initialization_method", "checkpoint_resume")),
        initializer_summary=dict(raw.get("initializer_summary", {})),
    )
    clean_and_bound_state(state, cfg, list(data.nodes), [int(t) for t in data.times])
    return state


def export_final(output_dir, data, state, cfg, settlement, data_path, calibration) -> None:
    quad = default_quadratic_demand_curve()
    export_epec_results(output_dir, data, state, cfg, settlement, data_path)
    export_checkpoint(output_dir, state, cfg)
    units = [investor.investor_id for investor in cfg.investors]
    reference = settlement["reference_model"]
    prices = settlement["reference_lambda"]
    _write_csv(
        output_dir / "strategic_quantity_offers.csv",
        [
            "investor", "hour", "node", "installed_power_mw",
            "charge_offer_mw", "accepted_charge_mw",
            "discharge_offer_mw", "accepted_discharge_mw",
            "joint_lambda_eur_per_mwh",
        ],
        [
            {
                "investor": investor,
                "hour": time_,
                "node": node,
                "installed_power_mw": state.x_power[investor, node],
                "charge_offer_mw": state.offer_charge[investor, node, int(time_)],
                "accepted_charge_mw": value(reference.P_charge[investor, node, time_]),
                "discharge_offer_mw": state.offer_discharge[investor, node, int(time_)],
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
            "iteration", "investor", "max_rel_delta_offer",
            "charge_offer_capacity_hours_mwh", "discharge_offer_capacity_hours_mwh",
        ],
        state.offer_convergence_history,
    )
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": "multi_investor_strategic_hourly_quantity_offers",
            "generator_calibration": calibration,
            "offer_convergence_required": True,
            "strategy_space": "nodal MW/MWh investment plus hourly charge/discharge quantity offers",
            "rival_representation": (
                "separate battery per investor with frozen nodal MW/MWh and "
                "hourly charge/discharge offers"
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_config_path = output_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config.update(
        {
            "experiment": "multi_investor_strategic_hourly_quantity_offers",
            "generator_calibration": calibration,
            "offer_convergence_required": True,
        }
    )
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategic-operation EPEC diagonalization")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--investor-set", choices=["portfolio4", "wacc"], default="portfolio4")
    parser.add_argument("--wacc", type=float, nargs="+", default=[0.08, 0.12])
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--max-iters", type=int, default=60)
    parser.add_argument("--tol-rel", type=float, default=DEFAULT_TOL_REL)
    parser.add_argument("--floor-mw", type=float, default=DEFAULT_FLOOR_MW)
    parser.add_argument("--floor-mwh", type=float, default=DEFAULT_FLOOR_MWH)
    parser.add_argument("--seed-power-mw", type=float, default=DEFAULT_INITIAL_POWER_MW)
    parser.add_argument("--seed-ratio-hours", type=float, default=4.0)
    parser.add_argument("--initializer-snapshot-power-mw", type=float, default=0.0)
    parser.add_argument("--initializer-snapshot-ratio-hours", type=float, default=4.0)
    parser.add_argument("--skip-jacobi-initializer", action="store_true")
    parser.add_argument("--max-cpu-time", type=float, default=180.0)
    parser.add_argument("--solver-tol", type=float, default=DEFAULT_SOLVER_TOL)
    parser.add_argument("--price-bound-eur-per-mwh", type=float, default=DEFAULT_PRICE_BOUND_EUR_PER_MWH)
    parser.add_argument("--dual-bound-eur-per-mwh", type=float, default=DEFAULT_DUAL_BOUND_EUR_PER_MWH)
    parser.add_argument("--dispatch-regularization", type=float, default=0.0)
    parser.add_argument("--capacity-cleanup-tol", type=float, default=DEFAULT_CAPACITY_CLEANUP_TOL_MW_MWH)
    parser.add_argument("--conventional-capacity-adder-mw", type=float, default=20.0)
    parser.add_argument("--peaker-node", type=str, default="N5")
    parser.add_argument("--peaker-capacity-mw", type=float, default=200.0)
    parser.add_argument("--peaker-cost-eur-per-mwh", type=float, default=95.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "seidel_scarcity95")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.damping <= 1.0:
        raise SystemExit("--damping must be in (0, 1].")
    if args.max_iters <= 0:
        raise SystemExit("--max-iters must be positive.")
    if args.solver_tol <= 0.0:
        raise SystemExit("--solver-tol must be positive.")
    if args.price_bound_eur_per_mwh <= 0.0 or args.dual_bound_eur_per_mwh <= 0.0:
        raise SystemExit("Price and dual bounds must be positive.")
    if args.seed_power_mw < 0.0 or args.initializer_snapshot_power_mw < 0.0:
        raise SystemExit("Seed and initializer power must be non-negative.")
    if args.dispatch_regularization < 0.0 or args.capacity_cleanup_tol < 0.0:
        raise SystemExit("Regularization and cleanup tolerance must be non-negative.")
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
    cfg = EpecConfig(
        investors=investors,
        node_limit_mw=args.node_limit_mw,
        update_rule="seidel",
        damping=args.damping,
        max_iters=args.max_iters,
        tol_rel=args.tol_rel,
        floor_mw=args.floor_mw,
        floor_mwh=args.floor_mwh,
        seed_power_mw=args.seed_power_mw,
        seed_ratio_hours=args.seed_ratio_hours,
        max_cpu_time=args.max_cpu_time,
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
        use_demand_curve=False,
        dispatch_regularization_eur_per_mw2h=args.dispatch_regularization,
        solver_tol=args.solver_tol,
        capacity_cleanup_tol_mw_mwh=args.capacity_cleanup_tol,
        automatic_jacobi_initializer=not args.skip_jacobi_initializer,
        jacobi_initializer_snapshot_power_mw=args.initializer_snapshot_power_mw,
        jacobi_initializer_snapshot_ratio_hours=args.initializer_snapshot_ratio_hours,
    )
    initial_state = None
    if args.resume_from:
        checkpoint_path = (
            args.resume_from / "checkpoint.json"
            if args.resume_from.is_dir()
            else args.resume_from
        )
        try:
            initial_state = load_checkpoint(checkpoint_path, data, cfg)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot resume strategic EPEC run: {exc}") from exc
        cfg = replace(
            cfg,
            starting_iteration=initial_state.iteration,
            resume_from=str(checkpoint_path),
        )
    print(
        f"Strategic-operation EPEC: {len(investors)} investors, "
        f"Jacobi initializer={'on' if cfg.automatic_jacobi_initializer else 'off'}, "
        f"Gauss-Seidel damping={cfg.damping}, tol_rel={cfg.tol_rel}, "
        f"calibration={calibration}"
    )
    if cfg.resume_from:
        print(
            f"Resuming from iteration {initial_state.iteration}; "
            f"--max-iters={cfg.max_iters} means additional Seidel sweeps."
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
    quad = default_quadratic_demand_curve()
    settlement = compute_joint_settlement(data, quad, state, cfg)
    print_epec_summary(state, cfg, settlement)
    if not args.no_export:
        export_final(args.output_dir, data, state, cfg, settlement, args.data, calibration)
        print(f"Wrote strategic-operation EPEC outputs to {args.output_dir}")
    return 0 if state.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
