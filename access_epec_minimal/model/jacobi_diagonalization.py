"""Four-investor capacity-only EPEC solved by damped Jacobi updates.

The only strategic variables are nodal BESS power and energy capacity. Market
dispatch and LMPs remain lower-level ISO variables embedded through either
relaxed KKT or exact strong-duality conditions. There are no access requests,
awards, bids, or operational offer variables in this module.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import pyomo.environ as pyo

import mpec_relaxed_kkt
import mpec_strong_duality
from investors import InvestorConfig
from primal_market_clearing_model import (
    MarketData,
    build_primal_market_clearing_model,
    effective_generation_offer,
)


@dataclass(frozen=True)
class JacobiConfig:
    investors: tuple[InvestorConfig, ...]
    formulation: str = "relaxed-kkt"
    node_limit_mw: float = 1_000.0
    max_sweeps: int = 60
    damping: float = 0.25
    tolerance_mw: float = 0.5
    tolerance_mwh: float = 1.0
    consecutive_sweeps: int = 2
    initial_power_mw: float = 5.0
    initial_ratio_hours: float = 3.0
    cleanup_tolerance: float = 1.0e-6
    complementarity_epsilon: float = 1.0e-3
    price_bound: float = 500.0
    dual_bound: float = 10_000.0
    sparse_capacity_tol: float = 1.0e-8
    proximal_penalty: float = 0.01
    proximal_energy_scale: float = 2.0
    parallel_workers: int = 1
    ipopt_linear_solver: str = "ma57"
    max_solver_iterations: int = 3_000
    max_solve_seconds: float = 600.0
    solver_tolerance: float = 1.0e-6
    tee: bool = False
    ipopt_executable: str | None = None


@dataclass(frozen=True)
class SolveOutcome:
    termination: str
    has_solution: bool
    optimal: bool
    seconds: float


@dataclass(frozen=True)
class BestResponseResult:
    investor_id: str
    outcome: SolveOutcome
    proposed_power: dict[str, float]
    proposed_energy: dict[str, float]
    profit_eur_per_day: float
    complementarity_max_product: float
    complementarity_max_violation: float
    primal_dual_gap_eur_per_day: float


@dataclass
class JacobiResult:
    power: dict[tuple[str, str], float]
    energy: dict[tuple[str, str], float]
    history: list[dict[str, object]] = field(default_factory=list)
    sweep: int = 0
    converged: bool = False
    stable_sweeps: int = 0
    stop_reason: str = ""
    final_responses: dict[str, BestResponseResult] = field(default_factory=dict)


SweepCallback = Callable[[JacobiResult], None]


def four_investors(data: MarketData) -> tuple[InvestorConfig, ...]:
    """Return the maintained merchant/wind-heavy/solar-heavy population."""

    wind = [generator for generator in data.generators if "Wind" in generator]
    solar = [generator for generator in data.generators if "PV" in generator]
    if not wind or not solar:
        raise ValueError("The four-investor profile requires wind and PV generation.")
    wind_heavy = {
        **{generator: 0.8 for generator in wind},
        **{generator: 0.2 for generator in solar},
    }
    solar_heavy = {
        **{generator: 0.2 for generator in wind},
        **{generator: 0.8 for generator in solar},
    }
    return (
        InvestorConfig("I1", wacc=0.08),
        InvestorConfig("I2", wacc=0.12),
        InvestorConfig("I3", wacc=0.08, owned_generation_shares=wind_heavy),
        InvestorConfig("I4", wacc=0.08, owned_generation_shares=solar_heavy),
    )


def _validate(data: MarketData, config: JacobiConfig) -> None:
    if not config.investors:
        raise ValueError("At least one investor is required.")
    if config.formulation not in {"relaxed-kkt", "strong-duality"}:
        raise ValueError(f"Unknown MPEC formulation: {config.formulation}")
    ids = [investor.investor_id for investor in config.investors]
    if len(ids) != len(set(ids)):
        raise ValueError("Investor identifiers must be unique.")
    if config.node_limit_mw <= 0.0:
        raise ValueError("The nodal capacity limit must be positive.")
    if not 0.0 < config.damping <= 1.0:
        raise ValueError("Damping must lie in (0, 1].")
    if min(config.tolerance_mw, config.tolerance_mwh) < 0.0:
        raise ValueError("Convergence tolerances cannot be negative.")
    if config.max_sweeps <= 0 or config.consecutive_sweeps <= 0:
        raise ValueError("Sweep counts must be positive.")
    if config.complementarity_epsilon < 0.0:
        raise ValueError("Complementarity epsilon cannot be negative.")
    if config.parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive.")
    if config.initial_power_mw * len(config.investors) > config.node_limit_mw:
        raise ValueError("Initial aggregate power exceeds the nodal capacity limit.")
    unknown = {
        generator
        for investor in config.investors
        for generator in investor.owned_generation_shares
        if generator not in data.generators
    }
    if unknown:
        raise ValueError(f"Unknown owned generators: {sorted(unknown)}")


def initial_state(data: MarketData, config: JacobiConfig) -> JacobiResult:
    _validate(data, config)
    power = {
        (investor.investor_id, node): config.initial_power_mw
        for investor in config.investors
        for node in data.nodes
    }
    energy = {
        (investor.investor_id, node): config.initial_power_mw
        * min(
            investor.ratio_max,
            max(investor.ratio_min, config.initial_ratio_hours),
        )
        for investor in config.investors
        for node in data.nodes
    }
    return JacobiResult(power=power, energy=energy)


def _fixed_storage_data(model: pyo.ConcreteModel, data: MarketData) -> MarketData:
    active = model._active_id
    units = [active, *model._rival_ids]
    power = {}
    energy = {}
    for investor_id in units:
        for node in data.nodes:
            if investor_id == active:
                power[investor_id, node] = float(pyo.value(model.X_power[node]))
                energy[investor_id, node] = float(pyo.value(model.X_energy[node]))
            else:
                power[investor_id, node] = model._rival_power[investor_id][node]
                energy[investor_id, node] = model._rival_energy[investor_id][node]
    return replace(data, storage_units=units, x_power=power, x_energy=energy)


def initialise_lower_level(
    model: pyo.ConcreteModel, data: MarketData, config: JacobiConfig
) -> None:
    """Warm-start the embedded KKT system from an exact fixed-capacity LP."""

    fixed_data = _fixed_storage_data(model, data)
    lower = build_primal_market_clearing_model(fixed_data, include_load_shed=False)
    lower.objective.deactivate()
    lower.objective_with_degradation = pyo.Objective(
        expr=sum(
            effective_generation_offer(data, generator)
            * lower.P_gen[generator, time_]
            for generator in lower.G
            for time_ in lower.T
        )
        + sum(
            0.5
            * model._unit_degradation[investor_id]
            * (
                lower.P_charge[investor_id, node, time_]
                + lower.P_discharge[investor_id, node, time_]
            )
            for investor_id in lower.I
            for node in lower.N
            for time_ in lower.T
        )
        + 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(
            lower.DemandAdjustment[node, time_] ** 2
            for node in lower.N
            for time_ in lower.T
        ),
        sense=pyo.minimize,
    )
    result = solve_exact_market_model(lower, config)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return

    def seed(variable: pyo.VarData, value: float) -> None:
        clipped = float(value)
        if variable.lb is not None:
            clipped = max(clipped, float(variable.lb))
        if variable.ub is not None:
            clipped = min(clipped, float(variable.ub))
        variable.set_value(clipped)

    for generator, time_ in model.GT:
        seed(model.P_gen[generator, time_], pyo.value(lower.P_gen[generator, time_]))
        seed(
            model.nu_gen[generator, time_],
            lower.dual[lower.generation_capacity_bound[generator, time_]]
        )
    for node in model.N:
        for time_ in model.T:
            seed(model.NetInjection[node, time_],
                pyo.value(lower.NetInjection[node, time_])
            )
            seed(model.DemandAdjustment[node, time_],
                pyo.value(lower.DemandAdjustment[node, time_])
            )
            seed(model.lam[node, time_], lower.dual[lower.nodal_balance[node, time_]])
    for time_ in model.T:
        seed(model.lam_sys[time_], lower.dual[lower.system_balance[time_]])
    for line in model.L:
        for time_ in model.T:
            seed(model.mu_up[line, time_],
                lower.dual[lower.line_upper_bound[line, time_]]
            )
            seed(model.mu_dn[line, time_],
                lower.dual[lower.line_lower_bound[line, time_]]
            )
    for investor_id, node in model.IN:
        for time_ in model.T:
            seed(model.P_charge[investor_id, node, time_],
                pyo.value(lower.P_charge[investor_id, node, time_])
            )
            seed(model.P_discharge[investor_id, node, time_],
                pyo.value(lower.P_discharge[investor_id, node, time_])
            )
            seed(model.rho_ch[investor_id, node, time_],
                lower.dual[lower.charge_power_bound[investor_id, node, time_]]
            )
            seed(model.sig_dis[investor_id, node, time_],
                lower.dual[lower.discharge_power_bound[investor_id, node, time_]]
            )
            seed(model.gam[investor_id, node, time_],
                lower.dual[lower.soc_transition[investor_id, node, time_]]
            )
        for soc_time in model.T_SOC:
            seed(model.SOC[investor_id, node, soc_time],
                pyo.value(lower.SOC[investor_id, node, soc_time])
            )
            seed(model.del_soc[investor_id, node, soc_time],
                lower.dual[lower.soc_capacity_bound[investor_id, node, soc_time]]
            )
        seed(model.rho_per[investor_id, node],
            lower.dual[lower.soc_periodicity[investor_id, node]]
        )


def build_best_response(
    data: MarketData,
    config: JacobiConfig,
    investor: InvestorConfig,
    snapshot_power: dict[tuple[str, str], float],
    snapshot_energy: dict[tuple[str, str], float],
) -> pyo.ConcreteModel:
    active = investor.investor_id
    rivals = [item for item in config.investors if item.investor_id != active]
    builder = (
        mpec_relaxed_kkt.build_model
        if config.formulation == "relaxed-kkt"
        else mpec_strong_duality.build_model
    )
    formulation_options = (
        {"complementarity_epsilon": config.complementarity_epsilon}
        if config.formulation == "relaxed-kkt"
        else {}
    )
    model = builder(
        data,
        investor=investor,
        rival_power={
            rival.investor_id: {
                node: snapshot_power[rival.investor_id, node] for node in data.nodes
            }
            for rival in rivals
        },
        rival_energy={
            rival.investor_id: {
                node: snapshot_energy[rival.investor_id, node] for node in data.nodes
            }
            for rival in rivals
        },
        rival_degradation_eur_per_mwh=15.0,
        node_limit_mw=config.node_limit_mw,
        initial_power_mw=0.0,
        initial_ratio_hours=config.initial_ratio_hours,
        price_bound=config.price_bound,
        dual_bound=config.dual_bound,
        sparse_capacity_tol=config.sparse_capacity_tol,
        proximal_power={node: snapshot_power[active, node] for node in data.nodes},
        proximal_energy={node: snapshot_energy[active, node] for node in data.nodes},
        proximal_penalty=config.proximal_penalty,
        proximal_energy_scale=config.proximal_energy_scale,
        **formulation_options,
    )
    for node in data.nodes:
        model.X_power[node].set_value(snapshot_power[active, node])
        model.X_energy[node].set_value(snapshot_energy[active, node])
    initialise_lower_level(model, data, config)
    return model


def _ipopt_path(config: JacobiConfig) -> Path | None:
    candidates = []
    if config.ipopt_executable:
        candidates.append(Path(config.ipopt_executable))
    if os.environ.get("IPOPT_EXECUTABLE"):
        candidates.append(Path(os.environ["IPOPT_EXECUTABLE"]))
    if shutil.which("ipopt"):
        candidates.append(Path(shutil.which("ipopt") or ""))
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "idaes" / "bin" / "ipopt.exe")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def solve_exact_market_model(
    model: pyo.ConcreteModel, config: JacobiConfig
) -> pyo.opt.SolverResults:
    """Solve a fixed-capacity convex market and import its unique duals."""

    executable = _ipopt_path(config)
    kwargs = {"solver_io": "nl"}
    if executable is not None:
        kwargs["executable"] = str(executable)
    solver = pyo.SolverFactory("ipopt", **kwargs)
    if not solver.available(exception_flag=False):
        raise RuntimeError("IPOPT is unavailable for the exact market solve.")
    solver.options.update(
        {
            "linear_solver": config.ipopt_linear_solver,
            "max_iter": config.max_solver_iterations,
            "max_cpu_time": config.max_solve_seconds,
            "tol": min(config.solver_tolerance, 1.0e-8),
            "acceptable_tol": 1.0e-7,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "print_level": 0,
        }
    )
    return solver.solve(model, tee=False)


def solve_best_response(
    data: MarketData,
    config: JacobiConfig,
    investor: InvestorConfig,
    snapshot_power: dict[tuple[str, str], float],
    snapshot_energy: dict[tuple[str, str], float],
) -> BestResponseResult:
    model = build_best_response(data, config, investor, snapshot_power, snapshot_energy)
    executable = _ipopt_path(config)
    kwargs = {"solver_io": "nl"}
    if executable is not None:
        kwargs["executable"] = str(executable)
    solver = pyo.SolverFactory("ipopt", **kwargs)
    if not solver.available(exception_flag=False):
        raise RuntimeError("IPOPT is unavailable.")
    solver.options.update(
        {
            "linear_solver": config.ipopt_linear_solver,
            "max_iter": config.max_solver_iterations,
            "max_cpu_time": config.max_solve_seconds,
            "tol": config.solver_tolerance,
            "acceptable_tol": max(config.solver_tolerance, 1.0e-4),
            "constr_viol_tol": config.solver_tolerance,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "print_level": 5 if config.tee else 0,
        }
    )
    started = time.perf_counter()
    try:
        result = solver.solve(model, tee=config.tee)
        termination = result.solver.termination_condition
        optimal = termination in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
        }
        has_solution = optimal or termination == pyo.TerminationCondition.feasible
        outcome = SolveOutcome(
            str(termination), has_solution, optimal, time.perf_counter() - started
        )
    except Exception as exc:
        outcome = SolveOutcome(
            f"error: {exc}", False, False, time.perf_counter() - started
        )

    if not outcome.has_solution:
        return BestResponseResult(
            investor.investor_id,
            outcome,
            {node: snapshot_power[investor.investor_id, node] for node in data.nodes},
            {node: snapshot_energy[investor.investor_id, node] for node in data.nodes},
            math.nan,
            math.nan,
            math.nan,
            math.nan,
        )
    diagnostics = (
        mpec_relaxed_kkt.diagnostics(model)
        if config.formulation == "relaxed-kkt"
        else mpec_strong_duality.diagnostics(model)
    )
    return BestResponseResult(
        investor.investor_id,
        outcome,
        {node: max(0.0, float(pyo.value(model.X_power[node]))) for node in data.nodes},
        {node: max(0.0, float(pyo.value(model.X_energy[node]))) for node in data.nodes},
        float(pyo.value(model.unregularized_profit)),
        float(diagnostics["maximum_product"]),
        max(
            float(diagnostics["maximum_upper_bound_violation"]),
            float(diagnostics["maximum_nonnegativity_violation"]),
        ),
        float(diagnostics["primal_dual_gap_eur_per_day"]),
    )


def _solve_all(
    data: MarketData,
    config: JacobiConfig,
    snapshot_power: dict[tuple[str, str], float],
    snapshot_energy: dict[tuple[str, str], float],
) -> dict[str, BestResponseResult]:
    if config.parallel_workers == 1:
        return {
            investor.investor_id: solve_best_response(
                data, config, investor, snapshot_power, snapshot_energy
            )
            for investor in config.investors
        }
    responses = {}
    with ProcessPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = {
            executor.submit(
                solve_best_response,
                data,
                config,
                investor,
                snapshot_power,
                snapshot_energy,
            ): investor.investor_id
            for investor in config.investors
        }
        for future in as_completed(futures):
            investor_id = futures[future]
            try:
                responses[investor_id] = future.result()
            except Exception as exc:
                responses[investor_id] = BestResponseResult(
                    investor_id,
                    SolveOutcome(f"error: {exc}", False, False, 0.0),
                    {node: snapshot_power[investor_id, node] for node in data.nodes},
                    {node: snapshot_energy[investor_id, node] for node in data.nodes},
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                )
    return responses


def audit_state(
    data: MarketData, config: JacobiConfig, state: JacobiResult
) -> dict[str, BestResponseResult]:
    return _solve_all(data, config, dict(state.power), dict(state.energy))


def run_jacobi(
    data: MarketData,
    config: JacobiConfig,
    *,
    initial: JacobiResult | None = None,
    on_sweep: SweepCallback | None = None,
) -> JacobiResult:
    _validate(data, config)
    state = initial if initial is not None else initial_state(data, config)

    for sweep in range(state.sweep + 1, config.max_sweeps + 1):
        old_power = dict(state.power)
        old_energy = dict(state.energy)
        responses = _solve_all(data, config, old_power, old_energy)
        all_optimal = all(response.outcome.optimal for response in responses.values())
        maximum_power_deviation = 0.0
        maximum_energy_deviation = 0.0
        maximum_product = 0.0
        maximum_violation = 0.0
        maximum_gap = 0.0

        for investor in config.investors:
            investor_id = investor.investor_id
            response = responses[investor_id]
            for node in data.nodes:
                key = investor_id, node
                proposed_power = (
                    response.proposed_power[node] if response.outcome.optimal else old_power[key]
                )
                proposed_energy = (
                    response.proposed_energy[node] if response.outcome.optimal else old_energy[key]
                )
                maximum_power_deviation = max(
                    maximum_power_deviation, abs(proposed_power - old_power[key])
                )
                maximum_energy_deviation = max(
                    maximum_energy_deviation, abs(proposed_energy - old_energy[key])
                )
                state.power[key] = (
                    (1.0 - config.damping) * old_power[key]
                    + config.damping * proposed_power
                )
                state.energy[key] = (
                    (1.0 - config.damping) * old_energy[key]
                    + config.damping * proposed_energy
                )
                if state.power[key] < config.cleanup_tolerance:
                    state.power[key] = 0.0
                    state.energy[key] = 0.0
            if response.outcome.has_solution:
                maximum_product = max(maximum_product, response.complementarity_max_product)
                maximum_violation = max(
                    maximum_violation, response.complementarity_max_violation
                )
                maximum_gap = max(maximum_gap, abs(response.primal_dual_gap_eur_per_day))

        for node in data.nodes:
            total = sum(state.power[investor.investor_id, node] for investor in config.investors)
            if total > config.node_limit_mw:
                scale = config.node_limit_mw / total
                for investor in config.investors:
                    key = investor.investor_id, node
                    state.power[key] *= scale
                    state.energy[key] *= scale

        stable = (
            all_optimal
            and maximum_power_deviation <= config.tolerance_mw
            and maximum_energy_deviation <= config.tolerance_mwh
        )
        state.stable_sweeps = state.stable_sweeps + 1 if stable else 0
        state.sweep = sweep
        row = {
            "sweep": sweep,
            "all_best_responses_optimal": all_optimal,
            "max_raw_power_deviation_mw": maximum_power_deviation,
            "max_raw_energy_deviation_mwh": maximum_energy_deviation,
            "max_complementarity_product": maximum_product,
            "max_complementarity_violation": maximum_violation,
            "max_absolute_primal_dual_gap_eur_per_day": maximum_gap,
            "stable_sweeps": state.stable_sweeps,
            "total_power_mw": sum(state.power.values()),
            "total_energy_mwh": sum(state.energy.values()),
            "solve_seconds": sum(response.outcome.seconds for response in responses.values()),
        }
        for investor in config.investors:
            investor_id = investor.investor_id
            row[f"termination_{investor_id}"] = responses[investor_id].outcome.termination
            row[f"profit_{investor_id}_eur_per_day"] = responses[
                investor_id
            ].profit_eur_per_day
        state.history.append(row)
        state.final_responses = responses
        if on_sweep is not None:
            on_sweep(state)
        if state.stable_sweeps >= config.consecutive_sweeps:
            state.converged = True
            state.stop_reason = (
                "raw capacity best-response residuals passed for "
                f"{state.stable_sweeps} consecutive sweeps"
            )
            break

    if not state.converged:
        state.stop_reason = "maximum sweeps reached"
    return state
