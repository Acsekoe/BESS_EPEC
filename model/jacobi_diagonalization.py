"""Multi-investor capacity game solved by simultaneous Gauss-Jacobi updates."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import pyomo.environ as pyo

import mpec_kkt_bigm
import mpec_strategic_operation
import mpec_strong_duality
from mpec_strong_duality import InvestorConfig
from primal_market_clearing_model import MarketData, build_primal_market_clearing_model


@dataclass(frozen=True)
class JacobiConfig:
    investors: tuple[InvestorConfig, ...]
    formulation: str = "strong-duality"
    node_limit_mw: float = 100.0
    max_sweeps: int = 60
    damping: float = 0.7
    tolerance_mw: float = 0.5
    tolerance_mwh: float = 1.0
    consecutive_sweeps: int = 2
    initial_power_mw: float = 0.0
    initial_ratio_hours: float = 2.0
    numerical_initial_power_mw: float = 10.0
    cleanup_tolerance: float = 1e-6
    proximal_penalty: float = 0.0
    proximal_energy_scale: float = 2.0
    price_bound: float = 500.0
    dual_bound: float = 10_000.0
    big_m_dual: float = 800.0
    sparse_capacity_tol: float = 1e-8
    warm_start_lower_level: bool = True
    bid_price_bound: float = 500.0
    tolerance_bid_eur_per_mwh: float = 0.5


@dataclass(frozen=True)
class SolveOutcome:
    termination: str
    has_solution: bool
    optimal: bool
    seconds: float
    best_feasible_objective: float | None = None
    best_objective_bound: float | None = None


@dataclass(frozen=True)
class BestResponseResult:
    investor_id: str
    outcome: SolveOutcome
    proposed_power: dict[str, float]
    proposed_energy: dict[str, float]
    profit_eur_per_day: float
    proposed_bid_charge: dict[tuple[str, int], float] = field(default_factory=dict)
    proposed_offer_discharge: dict[tuple[str, int], float] = field(default_factory=dict)


@dataclass
class JacobiResult:
    power: dict[tuple[str, str], float]
    energy: dict[tuple[str, str], float]
    history: list[dict[str, object]] = field(default_factory=list)
    sweep: int = 0
    converged: bool = False
    stop_reason: str = ""
    projection_count: int = 0
    stable_sweeps: int = 0
    bid_charge: dict[tuple[str, str, int], float] = field(default_factory=dict)
    offer_discharge: dict[tuple[str, str, int], float] = field(default_factory=dict)


ModelSolver = Callable[[pyo.ConcreteModel], SolveOutcome]
BatchSolver = Callable[
    [
        MarketData,
        JacobiConfig,
        dict[tuple[str, str], float],
        dict[tuple[str, str], float],
        dict[tuple[str, str, int], float],
        dict[tuple[str, str, int], float],
    ],
    dict[str, BestResponseResult],
]
SweepCallback = Callable[[JacobiResult], None]


def four_investors(data: MarketData) -> tuple[InvestorConfig, ...]:
    """Return the maintained merchant/portfolio investor population."""

    wind = [g for g in data.generators if "Wind" in g]
    solar = [g for g in data.generators if "PV" in g]
    if not wind or not solar:
        raise ValueError("The four-investor profile requires both wind and PV generators.")
    wind_heavy = {**{g: 0.8 for g in wind}, **{g: 0.2 for g in solar}}
    solar_heavy = {**{g: 0.2 for g in wind}, **{g: 0.8 for g in solar}}
    return (
        InvestorConfig("I1", wacc=0.08),
        InvestorConfig("I2", wacc=0.12),
        InvestorConfig("I3", wacc=0.08, owned_generation_shares=wind_heavy),
        InvestorConfig("I4", wacc=0.08, owned_generation_shares=solar_heavy),
    )


def _validate(config: JacobiConfig) -> None:
    if config.formulation not in {
        "strong-duality",
        "kkt-bigm",
        "strategic-operation",
    }:
        raise ValueError(f"Unknown formulation: {config.formulation}")
    if not 0.0 < config.damping <= 1.0:
        raise ValueError("damping must be in (0, 1].")
    if config.max_sweeps <= 0 or config.consecutive_sweeps <= 0:
        raise ValueError("Sweep counts must be positive.")
    if config.tolerance_mw < 0.0 or config.tolerance_mwh < 0.0:
        raise ValueError("Convergence tolerances cannot be negative.")
    if config.node_limit_mw <= 0.0:
        raise ValueError("node_limit_mw must be positive.")
    if min(
        config.initial_power_mw,
        config.numerical_initial_power_mw,
        config.cleanup_tolerance,
        config.proximal_penalty,
        config.sparse_capacity_tol,
    ) < 0.0:
        raise ValueError("Capacity seeds, tolerances, and penalties cannot be negative.")
    if config.price_bound <= 0.0 or config.dual_bound <= 0.0:
        raise ValueError("Price and dual bounds must be positive.")
    if config.bid_price_bound <= 0.0 or config.tolerance_bid_eur_per_mwh < 0.0:
        raise ValueError("The strategic bid bound must be positive and tolerance non-negative.")
    if config.formulation == "kkt-bigm" and config.big_m_dual < config.price_bound:
        raise ValueError("The KKT dual Big-M must be at least the price bound.")
    ids = [investor.investor_id for investor in config.investors]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Jacobi requires at least one investor and unique investor IDs.")


def _initial_state(data: MarketData, config: JacobiConfig) -> JacobiResult:
    seed = min(config.initial_power_mw, config.node_limit_mw / len(config.investors))
    ratio = max(
        max(investor.ratio_min for investor in config.investors),
        config.initial_ratio_hours,
    )
    power = {
        (investor.investor_id, n): seed
        for investor in config.investors
        for n in data.nodes
    }
    energy = {
        (investor.investor_id, n): seed * min(ratio, investor.ratio_max)
        for investor in config.investors
        for n in data.nodes
    }
    if config.formulation == "strategic-operation":
        bid_charge = {
            (investor.investor_id, n, int(t)): -min(
                config.bid_price_bound,
                0.5 * investor.degradation_eur_per_mwh,
            )
            for investor in config.investors
            for n in data.nodes
            for t in data.times
        }
        offer_discharge = {
            (investor.investor_id, n, int(t)): min(
                config.bid_price_bound,
                0.5 * investor.degradation_eur_per_mwh,
            )
            for investor in config.investors
            for n in data.nodes
            for t in data.times
        }
    else:
        bid_charge = {}
        offer_discharge = {}
    return JacobiResult(
        power=power,
        energy=energy,
        bid_charge=bid_charge,
        offer_discharge=offer_discharge,
    )


def _fixed_storage_data(model: pyo.ConcreteModel, data: MarketData) -> MarketData:
    active = model._active_id
    units = [active, *model._rival_ids]
    x_power: dict[tuple[str, str], float] = {}
    x_energy: dict[tuple[str, str], float] = {}
    for i in units:
        for n in data.nodes:
            if i == active:
                x_power[i, n] = pyo.value(model.X_power[n])
                x_energy[i, n] = pyo.value(model.X_energy[n])
            else:
                x_power[i, n] = model._rival_power[i][n]
                x_energy[i, n] = model._rival_energy[i][n]
    return replace(data, storage_units=units, x_power=x_power, x_energy=x_energy)


def initialise_lower_level(model: pyo.ConcreteModel, data: MarketData) -> None:
    """Seed the MPEC with one optimal fixed-capacity lower-level solution."""

    fixed_data = _fixed_storage_data(model, data)
    lower = build_primal_market_clearing_model(fixed_data, include_load_shed=False)
    lower.objective.deactivate()

    if getattr(model, "_strategic_operation", False):
        active = model._active_id

        def charge_bid(unit: str, node: str, time: int) -> float:
            return (
                pyo.value(model.BidCharge[node, time])
                if unit == active
                else model._rival_bid_charge.get(
                    (unit, node, int(time)),
                    -min(
                        model._bid_price_bound,
                        0.5 * model._unit_degradation[unit],
                    ),
                )
            )

        def discharge_offer(unit: str, node: str, time: int) -> float:
            return (
                pyo.value(model.OfferDischarge[node, time])
                if unit == active
                else model._rival_offer_discharge.get(
                    (unit, node, int(time)),
                    min(
                        model._bid_price_bound,
                        0.5 * model._unit_degradation[unit],
                    ),
                )
            )

        storage_cost = sum(
            discharge_offer(i, n, int(t)) * lower.P_discharge[i, n, t]
            - charge_bid(i, n, int(t)) * lower.P_charge[i, n, t]
            for i in lower.I
            for n in lower.N
            for t in lower.T
        )
    else:
        storage_cost = sum(
            0.5
            * model._unit_degradation[i]
            * (lower.P_charge[i, n, t] + lower.P_discharge[i, n, t])
            for i in lower.I
            for n in lower.N
            for t in lower.T
        )

    lower.objective_with_degradation = pyo.Objective(
        expr=sum(
            data.generation_cost[g] * lower.P_gen[g, t]
            for g in lower.G
            for t in lower.T
        )
        + storage_cost,
        sense=pyo.minimize,
    )
    result = pyo.SolverFactory("highs").solve(lower, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return

    for g, t in model.GT:
        model.P_gen[g, t].set_value(pyo.value(lower.P_gen[g, t]))
        model.nu_gen[g, t].set_value(lower.dual[lower.generation_capacity_bound[g, t]])
    for n in model.N:
        for t in model.T:
            model.NetInjection[n, t].set_value(pyo.value(lower.NetInjection[n, t]))
            model.lam[n, t].set_value(lower.dual[lower.nodal_balance[n, t]])
    for t in model.T:
        model.lam_sys[t].set_value(lower.dual[lower.system_balance[t]])
    for l in model.L:
        for t in model.T:
            model.mu_up[l, t].set_value(lower.dual[lower.line_upper_bound[l, t]])
            model.mu_dn[l, t].set_value(lower.dual[lower.line_lower_bound[l, t]])
    for i, n in model.IN:
        for t in model.T:
            model.P_charge[i, n, t].set_value(pyo.value(lower.P_charge[i, n, t]))
            model.P_discharge[i, n, t].set_value(pyo.value(lower.P_discharge[i, n, t]))
            model.rho_ch[i, n, t].set_value(lower.dual[lower.charge_power_bound[i, n, t]])
            model.sig_dis[i, n, t].set_value(lower.dual[lower.discharge_power_bound[i, n, t]])
            model.gam[i, n, t].set_value(lower.dual[lower.soc_transition[i, n, t]])
        for tau in model.T_SOC:
            model.SOC[i, n, tau].set_value(pyo.value(lower.SOC[i, n, tau]))
            model.del_soc[i, n, tau].set_value(lower.dual[lower.soc_capacity_bound[i, n, tau]])
        model.rho_per[i, n].set_value(lower.dual[lower.soc_periodicity[i, n]])


def initialise_bigm_binaries(model: pyo.ConcreteModel, data: MarketData) -> None:
    """Choose a feasible Big-M pattern from the seeded lower-level optimum."""

    active = model._active_id
    tolerance = 1e-6

    def set_binary(name: str, slack: Callable[[tuple], float]) -> None:
        binary = getattr(model, f"{name}_binary")
        for key in binary:
            binary[key].set_value(1 if slack(key) > tolerance else 0)

    def flow(line: str, time_: int) -> float:
        return sum(
            data.ptdf[line, n] * pyo.value(model.NetInjection[n, time_])
            for n in model.N
        )

    set_binary("gen_variable", lambda key: pyo.value(model.P_gen[key]))
    set_binary("charge_variable", lambda key: pyo.value(model.P_charge[key]))
    set_binary("discharge_variable", lambda key: pyo.value(model.P_discharge[key]))
    set_binary("soc_variable", lambda key: pyo.value(model.SOC[key]))
    set_binary(
        "generation_capacity",
        lambda key: data.generation_capacity[key] - pyo.value(model.P_gen[key]),
    )
    set_binary(
        "line_upper", lambda key: data.line_limit[key[0]] - flow(*key)
    )
    set_binary(
        "line_lower", lambda key: flow(*key) + data.line_limit[key[0]]
    )

    def power_capacity(key: tuple) -> float:
        i, n, _ = key
        return (
            pyo.value(model.X_power[n]) if i == active else model._rival_power[i][n]
        )

    def energy_capacity(key: tuple) -> float:
        i, n, _ = key
        return (
            pyo.value(model.X_energy[n]) if i == active else model._rival_energy[i][n]
        )

    set_binary(
        "charge_capacity",
        lambda key: power_capacity(key) - pyo.value(model.P_charge[key]),
    )
    set_binary(
        "discharge_capacity",
        lambda key: power_capacity(key) - pyo.value(model.P_discharge[key]),
    )
    set_binary(
        "soc_capacity",
        lambda key: energy_capacity(key) - pyo.value(model.SOC[key]),
    )


def build_best_response(
    data: MarketData,
    config: JacobiConfig,
    investor: InvestorConfig,
    snapshot_power: dict[tuple[str, str], float],
    snapshot_energy: dict[tuple[str, str], float],
    snapshot_bid_charge: dict[tuple[str, str, int], float] | None = None,
    snapshot_offer_discharge: dict[tuple[str, str, int], float] | None = None,
) -> pyo.ConcreteModel:
    active = investor.investor_id
    rivals = [other for other in config.investors if other.investor_id != active]
    rival_power = {
        other.investor_id: {
            n: snapshot_power[other.investor_id, n] for n in data.nodes
        }
        for other in rivals
    }
    rival_energy = {
        other.investor_id: {
            n: snapshot_energy[other.investor_id, n] for n in data.nodes
        }
        for other in rivals
    }
    common_kwargs = dict(
        data=data,
        investor=investor,
        rival_power=rival_power,
        rival_energy=rival_energy,
        node_limit_mw=config.node_limit_mw,
        initial_power_mw=config.numerical_initial_power_mw,
        initial_ratio_hours=config.initial_ratio_hours,
        price_bound=config.price_bound,
        dual_bound=config.dual_bound,
        big_m_dual=config.big_m_dual,
        sparse_capacity_tol=config.sparse_capacity_tol,
        proximal_power={n: snapshot_power[active, n] for n in data.nodes},
        proximal_energy={n: snapshot_energy[active, n] for n in data.nodes},
        proximal_penalty=config.proximal_penalty,
        proximal_energy_scale=config.proximal_energy_scale,
    )
    if config.formulation == "strong-duality":
        model = mpec_strong_duality.build_model(**common_kwargs)
    elif config.formulation == "kkt-bigm":
        model = mpec_kkt_bigm.build_model(**common_kwargs)
    else:
        if snapshot_bid_charge is None or snapshot_offer_discharge is None:
            raise ValueError("Strategic-operation best responses require price snapshots.")
        rival_bid_charge = {
            other.investor_id: {
                (n, int(t)): snapshot_bid_charge[other.investor_id, n, int(t)]
                for n in data.nodes
                for t in data.times
            }
            for other in rivals
        }
        rival_offer_discharge = {
            other.investor_id: {
                (n, int(t)): snapshot_offer_discharge[
                    other.investor_id, n, int(t)
                ]
                for n in data.nodes
                for t in data.times
            }
            for other in rivals
        }
        model = mpec_strategic_operation.build_model(
            **common_kwargs,
            rival_bid_charge=rival_bid_charge,
            rival_offer_discharge=rival_offer_discharge,
            initial_bid_charge={
                (n, int(t)): snapshot_bid_charge[active, n, int(t)]
                for n in data.nodes
                for t in data.times
            },
            initial_offer_discharge={
                (n, int(t)): snapshot_offer_discharge[active, n, int(t)]
                for n in data.nodes
                for t in data.times
            },
            bid_price_bound=config.bid_price_bound,
        )
    if config.warm_start_lower_level:
        initialise_lower_level(model, data)
        if config.formulation == "kkt-bigm":
            initialise_bigm_binaries(model, data)
    return model


def collect_best_response(
    model: pyo.ConcreteModel,
    outcome: SolveOutcome,
    data: MarketData,
) -> BestResponseResult:
    """Extract the small serialisable result needed by the Jacobi update."""

    investor_id = model._active_id
    proposed_power = {
        n: max(0.0, pyo.value(model.X_power[n])) for n in data.nodes
    }
    proposed_energy = {
        n: max(0.0, pyo.value(model.X_energy[n])) for n in data.nodes
    }
    proposed_bid_charge = (
        {
            (n, int(t)): pyo.value(model.BidCharge[n, t])
            for n in data.nodes
            for t in data.times
        }
        if hasattr(model, "BidCharge") and outcome.has_solution
        else {}
    )
    proposed_offer_discharge = (
        {
            (n, int(t)): pyo.value(model.OfferDischarge[n, t])
            for n in data.nodes
            for t in data.times
        }
        if hasattr(model, "OfferDischarge") and outcome.has_solution
        else {}
    )
    profit = (
        pyo.value(
            model.kkt_unregularized_profit
            if hasattr(model, "kkt_unregularized_profit")
            else model.unregularized_profit
        )
        if outcome.has_solution
        else math.nan
    )
    return BestResponseResult(
        investor_id,
        outcome,
        proposed_power,
        proposed_energy,
        profit,
        proposed_bid_charge,
        proposed_offer_discharge,
    )


def run_jacobi(
    data: MarketData,
    config: JacobiConfig,
    solve: ModelSolver | None,
    on_sweep: SweepCallback | None = None,
    solve_batch: BatchSolver | None = None,
    initial_state: JacobiResult | None = None,
) -> JacobiResult:
    """Solve all investor best responses against one frozen snapshot per sweep."""

    _validate(config)
    if solve is None and solve_batch is None:
        raise ValueError("Either a sequential solver or a batch solver is required.")
    state = initial_state if initial_state is not None else _initial_state(data, config)
    stable_sweeps = state.stable_sweeps
    if state.converged:
        return state

    for sweep in range(state.sweep + 1, config.max_sweeps + 1):
        old_power = dict(state.power)
        old_energy = dict(state.energy)
        old_bid_charge = dict(state.bid_charge)
        old_offer_discharge = dict(state.offer_discharge)
        proposals_power: dict[tuple[str, str], float] = {}
        proposals_energy: dict[tuple[str, str], float] = {}
        proposals_bid_charge: dict[tuple[str, str, int], float] = {}
        proposals_offer_discharge: dict[tuple[str, str, int], float] = {}

        if solve_batch is not None:
            responses = solve_batch(
                data,
                config,
                old_power,
                old_energy,
                old_bid_charge,
                old_offer_discharge,
            )
        else:
            models = {
                investor.investor_id: build_best_response(
                    data,
                    config,
                    investor,
                    old_power,
                    old_energy,
                    old_bid_charge,
                    old_offer_discharge,
                )
                for investor in config.investors
            }
            responses: dict[str, BestResponseResult] = {}
            for investor in config.investors:
                investor_id = investor.investor_id
                model = models[investor_id]
                started = time.perf_counter()
                try:
                    assert solve is not None
                    outcome = solve(model)
                except Exception as exc:  # retain the current iterate on a failed best response
                    outcome = SolveOutcome(
                        f"error: {exc}", False, False, time.perf_counter() - started
                    )
                responses[investor_id] = collect_best_response(model, outcome, data)

        for investor in config.investors:
            investor_id = investor.investor_id
            response = responses[investor_id]
            outcome = response.outcome
            for n in data.nodes:
                proposals_power[investor_id, n] = (
                    response.proposed_power[n]
                    if outcome.optimal
                    else old_power[investor_id, n]
                )
                proposals_energy[investor_id, n] = (
                    response.proposed_energy[n]
                    if outcome.optimal
                    else old_energy[investor_id, n]
                )
                if config.formulation == "strategic-operation":
                    for t in data.times:
                        price_key = investor_id, n, int(t)
                        proposals_bid_charge[price_key] = (
                            response.proposed_bid_charge[n, int(t)]
                            if outcome.optimal
                            else old_bid_charge[price_key]
                        )
                        proposals_offer_discharge[price_key] = (
                            response.proposed_offer_discharge[n, int(t)]
                            if outcome.optimal
                            else old_offer_discharge[price_key]
                        )

        for key in state.power:
            state.power[key] = (1.0 - config.damping) * old_power[key] + config.damping * proposals_power[key]
            state.energy[key] = (1.0 - config.damping) * old_energy[key] + config.damping * proposals_energy[key]
            if state.power[key] < config.cleanup_tolerance:
                state.power[key] = 0.0
                state.energy[key] = 0.0

        if config.formulation == "strategic-operation":
            investors_by_id = {
                investor.investor_id: investor for investor in config.investors
            }
            for key in state.bid_charge:
                investor_id, node, _ = key
                state.bid_charge[key] = (
                    (1.0 - config.damping) * old_bid_charge[key]
                    + config.damping * proposals_bid_charge[key]
                )
                state.offer_discharge[key] = (
                    (1.0 - config.damping) * old_offer_discharge[key]
                    + config.damping * proposals_offer_discharge[key]
                )
                if state.power[investor_id, node] < config.cleanup_tolerance:
                    half_cost = min(
                        config.bid_price_bound,
                        0.5
                        * investors_by_id[investor_id].degradation_eur_per_mwh,
                    )
                    state.bid_charge[key] = -half_cost
                    state.offer_discharge[key] = half_cost

        for n in data.nodes:
            total = sum(state.power[investor.investor_id, n] for investor in config.investors)
            if total > config.node_limit_mw + 1e-9:
                scale = config.node_limit_mw / total
                state.projection_count += 1
                for investor in config.investors:
                    key = investor.investor_id, n
                    state.power[key] *= scale
                    state.energy[key] *= scale

        max_raw_power = max(abs(proposals_power[key] - old_power[key]) for key in state.power)
        max_raw_energy = max(abs(proposals_energy[key] - old_energy[key]) for key in state.energy)
        max_iterate_power = max(abs(state.power[key] - old_power[key]) for key in state.power)
        max_iterate_energy = max(abs(state.energy[key] - old_energy[key]) for key in state.energy)
        active_price_keys = [
            key
            for key in proposals_bid_charge
            if max(
                old_power[key[0], key[1]],
                proposals_power[key[0], key[1]],
            )
            > config.sparse_capacity_tol
        ]
        max_raw_bid = max(
            (
                max(
                    abs(proposals_bid_charge[key] - old_bid_charge[key]),
                    abs(
                        proposals_offer_discharge[key]
                        - old_offer_discharge[key]
                    ),
                )
                for key in active_price_keys
            ),
            default=0.0,
        )
        max_iterate_bid = max(
            (
                max(
                    abs(state.bid_charge[key] - old_bid_charge[key]),
                    abs(state.offer_discharge[key] - old_offer_discharge[key]),
                )
                for key in active_price_keys
            ),
            default=0.0,
        )
        all_optimal = all(response.outcome.optimal for response in responses.values())
        is_stable = (
            all_optimal
            and max_raw_power <= config.tolerance_mw
            and max_raw_energy <= config.tolerance_mwh
            and max_raw_bid <= config.tolerance_bid_eur_per_mwh
        )
        stable_sweeps = stable_sweeps + 1 if is_stable else 0
        state.stable_sweeps = stable_sweeps

        for investor in config.investors:
            investor_id = investor.investor_id
            response = responses[investor_id]
            outcome = response.outcome
            state.history.append(
                {
                    "sweep": sweep,
                    "investor": investor_id,
                    "termination": outcome.termination,
                    "optimal": outcome.optimal,
                    "solve_seconds": outcome.seconds,
                    "best_feasible_objective": outcome.best_feasible_objective,
                    "best_objective_bound": outcome.best_objective_bound,
                    "profit_eur_per_day": response.profit_eur_per_day,
                    "old_power_mw": sum(old_power[investor_id, n] for n in data.nodes),
                    "best_response_power_mw": sum(proposals_power[investor_id, n] for n in data.nodes),
                    "new_power_mw": sum(state.power[investor_id, n] for n in data.nodes),
                    "old_energy_mwh": sum(old_energy[investor_id, n] for n in data.nodes),
                    "best_response_energy_mwh": sum(proposals_energy[investor_id, n] for n in data.nodes),
                    "new_energy_mwh": sum(state.energy[investor_id, n] for n in data.nodes),
                    "max_raw_deviation_mw": max_raw_power,
                    "max_raw_deviation_mwh": max_raw_energy,
                    "max_iterate_change_mw": max_iterate_power,
                    "max_iterate_change_mwh": max_iterate_energy,
                    "max_raw_bid_deviation_eur_per_mwh": max_raw_bid,
                    "max_iterate_bid_change_eur_per_mwh": max_iterate_bid,
                }
            )

        state.sweep = sweep
        if stable_sweeps >= config.consecutive_sweeps:
            state.converged = True
            state.stop_reason = f"converged for {stable_sweeps} consecutive sweeps"
        elif not all_optimal:
            state.stop_reason = "one or more best responses were not solved to optimality"
        else:
            state.stop_reason = "maximum sweeps reached"
        if on_sweep is not None:
            on_sweep(state)
        if state.converged:
            break

    return state
