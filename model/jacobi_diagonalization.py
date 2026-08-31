"""Multi-investor capacity game solved by simultaneous Gauss-Jacobi updates."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import pyomo.environ as pyo

import mpec_kkt_bigm
import mpec_relaxed_kkt
import mpec_strategic_access_relaxed_kkt
import mpec_strategic_operation
import mpec_strategic_price_relaxed_kkt
import mpec_strategic_quantity_relaxed_kkt
import mpec_strong_duality
from mpec_strong_duality import InvestorConfig
from primal_market_clearing_model import MarketData, build_primal_market_clearing_model


@dataclass(frozen=True)
class JacobiConfig:
    investors: tuple[InvestorConfig, ...]
    formulation: str = "strong-duality"
    node_limit_mw: float = 100.0
    max_sweeps: int = 60
    damping: float = 0.25
    tolerance_mw: float = 0.5
    tolerance_mwh: float = 1.0
    consecutive_sweeps: int = 2
    stop_at_convergence: bool = True
    initial_power_mw: float = 0.0
    initial_ratio_hours: float = 2.0
    numerical_initial_power_mw: float = 10.0
    cleanup_tolerance: float = 1e-6
    proximal_penalty: float = 0.0
    proximal_energy_scale: float = 2.0
    proximal_price_scale: float = 10.0
    price_bound: float = 500.0
    dual_bound: float = 10_000.0
    big_m_dual: float = 800.0
    complementarity_epsilon: float = 1.0e-3
    sparse_capacity_tol: float = 1e-8
    warm_start_lower_level: bool = True
    bid_price_bound: float = 500.0
    initial_bid_charge_eur_per_mwh: float = 0.0
    initial_offer_discharge_eur_per_mwh: float = 0.0
    tolerance_bid_eur_per_mwh: float = 0.5
    initial_charge_bid_mw: float = 0.0
    initial_discharge_bid_mw: float = 0.0
    tolerance_quantity_bid_mw: float = 0.5
    access_request_limit_mw: float = 200.0
    access_bid_bound: float = 500.0
    initial_access_bid_eur_per_mw_day: float = 1.0
    access_undamped_sweeps: int = 10
    tolerance_access_bid_eur_per_mw_day: float = 0.5


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
    complementarity_max_product: float | None = None
    complementarity_max_violation: float | None = None
    primal_dual_gap_eur_per_day: float | None = None
    proposed_bid_charge_price: dict[tuple[str, int], float] = field(
        default_factory=dict
    )
    proposed_offer_discharge_price: dict[tuple[str, int], float] = field(
        default_factory=dict
    )
    proposed_access_quantity: dict[str, float] = field(default_factory=dict)
    proposed_access_bid: dict[str, float] = field(default_factory=dict)


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
    bid_charge_price: dict[tuple[str, str, int], float] = field(
        default_factory=dict
    )
    offer_discharge_price: dict[tuple[str, str, int], float] = field(
        default_factory=dict
    )
    access_quantity: dict[tuple[str, str], float] = field(default_factory=dict)
    access_bid: dict[tuple[str, str], float] = field(default_factory=dict)


ModelSolver = Callable[[pyo.ConcreteModel], SolveOutcome]
BatchSolver = Callable[
    [
        MarketData,
        JacobiConfig,
        dict[tuple[str, str], float],
        dict[tuple[str, str], float],
        dict[tuple[str, str, int], float],
        dict[tuple[str, str, int], float],
        dict[tuple[str, str, int], float],
        dict[tuple[str, str, int], float],
        dict[tuple[str, str], float],
        dict[tuple[str, str], float],
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
        "relaxed-kkt",
        "kkt-bigm",
        "strategic-operation",
        "strategic-price-relaxed-kkt",
        "strategic-quantity",
        "strategic-price-quantity",
        "strategic-access",
    }:
        raise ValueError(f"Unknown formulation: {config.formulation}")
    if not 0.0 < config.damping <= 1.0:
        raise ValueError("damping must be in (0, 1].")
    if (
        config.max_sweeps <= 0
        or config.consecutive_sweeps <= 0
        or config.access_undamped_sweeps < 0
    ):
        raise ValueError(
            "Maximum/convergence sweep counts must be positive and the "
            "undamped access-sweep count cannot be negative."
        )
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
    if (
        config.price_bound <= 0.0
        or config.dual_bound <= 0.0
        or config.proximal_energy_scale <= 0.0
    ):
        raise ValueError("Price/dual bounds and the energy scale must be positive.")
    if (
        config.formulation
        in {
            "strategic-price-quantity",
            "strategic-price-relaxed-kkt",
            "strategic-access",
        }
        and config.proximal_price_scale <= 0.0
    ):
        raise ValueError("The proximal price scale must be positive.")
    if config.complementarity_epsilon < 0.0:
        raise ValueError("complementarity_epsilon cannot be negative.")
    if min(
        config.initial_charge_bid_mw,
        config.initial_discharge_bid_mw,
        config.tolerance_quantity_bid_mw,
    ) < 0.0:
        raise ValueError("Strategic quantity seeds and tolerance cannot be negative.")
    if config.bid_price_bound <= 0.0 or config.tolerance_bid_eur_per_mwh < 0.0:
        raise ValueError("The strategic bid bound must be positive and tolerance non-negative.")
    if (
        config.access_request_limit_mw <= 0.0
        or config.access_bid_bound <= 0.0
        or config.initial_access_bid_eur_per_mw_day < 0.0
        or config.initial_access_bid_eur_per_mw_day > config.access_bid_bound
        or config.tolerance_access_bid_eur_per_mw_day < 0.0
    ):
        raise ValueError("Invalid strategic-access request or bid settings.")
    if max(
        abs(config.initial_bid_charge_eur_per_mwh),
        abs(config.initial_offer_discharge_eur_per_mwh),
    ) > config.bid_price_bound:
        raise ValueError("Initial strategic prices must lie within the bid-price bound.")
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
    if config.formulation in {
        "strategic-operation",
        "strategic-price-relaxed-kkt",
    }:
        if (
            config.initial_offer_discharge_eur_per_mwh
            < config.initial_bid_charge_eur_per_mwh / (data.eta**2)
        ):
            raise ValueError(
                "Initial strategic prices permit a negative-cost same-hour storage loop."
            )
        bid_charge = {
            (investor.investor_id, n, int(t)): config.initial_bid_charge_eur_per_mwh
            for investor in config.investors
            for n in data.nodes
            for t in data.times
        }
        offer_discharge = {
            (investor.investor_id, n, int(t)): config.initial_offer_discharge_eur_per_mwh
            for investor in config.investors
            for n in data.nodes
            for t in data.times
        }
    elif config.formulation in {
        "strategic-quantity",
        "strategic-price-quantity",
    }:
        if max(
            config.initial_charge_bid_mw,
            config.initial_discharge_bid_mw,
        ) > seed + 1.0e-9:
            raise ValueError(
                "Initial strategic quantity bids cannot exceed initial installed MW."
            )
        bid_charge = {
            (investor.investor_id, n, int(t)): config.initial_charge_bid_mw
            for investor in config.investors
            for n in data.nodes
            for t in data.times
        }
        offer_discharge = {
            (investor.investor_id, n, int(t)): config.initial_discharge_bid_mw
            for investor in config.investors
            for n in data.nodes
            for t in data.times
        }
    else:
        bid_charge = {}
        offer_discharge = {}
    if config.formulation == "strategic-price-quantity":
        if (
            config.initial_offer_discharge_eur_per_mwh
            < config.initial_bid_charge_eur_per_mwh / (data.eta**2)
        ):
            raise ValueError(
                "Initial strategic prices permit a negative-cost storage loop."
            )
        bid_charge_price = {
            (investor.investor_id, n, int(t)):
            config.initial_bid_charge_eur_per_mwh
            for investor in config.investors
            for n in data.nodes
            for t in data.times
        }
        offer_discharge_price = {
            (investor.investor_id, n, int(t)):
            config.initial_offer_discharge_eur_per_mwh
            for investor in config.investors
            for n in data.nodes
            for t in data.times
        }
    else:
        bid_charge_price = {}
        offer_discharge_price = {}
    if config.formulation == "strategic-access":
        access_quantity = {
            (investor.investor_id, node): min(
                max(0.0, config.initial_power_mw),
                config.node_limit_mw,
                config.access_request_limit_mw,
            )
            for investor in config.investors
            for node in data.nodes
        }
        for investor in config.investors:
            total = sum(
                access_quantity[investor.investor_id, node]
                for node in data.nodes
            )
            if total > config.access_request_limit_mw:
                scale = config.access_request_limit_mw / total
                for node in data.nodes:
                    access_quantity[investor.investor_id, node] *= scale
        access_bid = {
            (investor.investor_id, node): config.initial_access_bid_eur_per_mw_day
            for investor in config.investors
            for node in data.nodes
        }
    else:
        access_quantity = {}
        access_bid = {}
    state = JacobiResult(
        power=power,
        energy=energy,
        bid_charge=bid_charge,
        offer_discharge=offer_discharge,
        bid_charge_price=bid_charge_price,
        offer_discharge_price=offer_discharge_price,
        access_quantity=access_quantity,
        access_bid=access_bid,
    )
    if config.formulation == "strategic-access":
        _clear_access_allocation(data, config, state)
    return state


def _clear_access_allocation(
    data: MarketData,
    config: JacobiConfig,
    state: JacobiResult,
) -> pyo.ConcreteModel:
    """Clear one common exact access/dispatch LP and update derived capacity."""

    quantity = {
        investor.investor_id: {
            node: state.access_quantity[investor.investor_id, node]
            for node in data.nodes
        }
        for investor in config.investors
    }
    bid = {
        investor.investor_id: {
            node: state.access_bid[investor.investor_id, node]
            for node in data.nodes
        }
        for investor in config.investors
    }
    energy = {
        investor.investor_id: {
            node: state.energy[investor.investor_id, node]
            for node in data.nodes
        }
        for investor in config.investors
    }
    lower = mpec_strategic_access_relaxed_kkt.solve_fixed_access_market(
        data,
        access_quantity=quantity,
        access_bid=bid,
        energy_capacity=energy,
        degradation={
            investor.investor_id: investor.degradation_eur_per_mwh
            for investor in config.investors
        },
        node_limit_mw=config.node_limit_mw,
    )
    for investor in config.investors:
        investor_id = investor.investor_id
        for node in data.nodes:
            key = investor_id, node
            awarded = max(0.0, float(pyo.value(lower.X_awarded[key])))
            if awarded < config.cleanup_tolerance:
                awarded = 0.0
            state.power[key] = awarded
    return lower


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
    snapshot_bid_charge_price: dict[tuple[str, str, int], float] | None = None,
    snapshot_offer_discharge_price: dict[tuple[str, str, int], float] | None = None,
    snapshot_access_quantity: dict[tuple[str, str], float] | None = None,
    snapshot_access_bid: dict[tuple[str, str], float] | None = None,
) -> pyo.ConcreteModel:
    active = investor.investor_id
    rivals = [other for other in config.investors if other.investor_id != active]
    if config.formulation == "strategic-access":
        if (
            snapshot_access_quantity is None
            or snapshot_access_bid is None
        ):
            raise ValueError("Strategic-access best responses require access snapshots.")
        model = mpec_strategic_access_relaxed_kkt.build_model(
            data,
            investor=investor,
            rival_access_quantity={
                other.investor_id: {
                    node: snapshot_access_quantity[other.investor_id, node]
                    for node in data.nodes
                }
                for other in rivals
            },
            rival_access_bid={
                other.investor_id: {
                    node: snapshot_access_bid[other.investor_id, node]
                    for node in data.nodes
                }
                for other in rivals
            },
            rival_energy_capacity={
                other.investor_id: {
                    node: snapshot_energy[other.investor_id, node]
                    for node in data.nodes
                }
                for other in rivals
            },
            rival_degradation={
                other.investor_id: other.degradation_eur_per_mwh
                for other in rivals
            },
            node_limit_mw=config.node_limit_mw,
            investor_request_limit_mw=config.access_request_limit_mw,
            access_bid_bound=config.access_bid_bound,
            initial_access_quantity={
                node: snapshot_access_quantity[active, node]
                for node in data.nodes
            },
            initial_access_bid={
                node: snapshot_access_bid[active, node] for node in data.nodes
            },
            initial_energy_capacity={
                node: snapshot_energy[active, node] for node in data.nodes
            },
            price_bound=config.price_bound,
            dual_bound=config.dual_bound,
            complementarity_epsilon=config.complementarity_epsilon,
            proximal_access_quantity={
                node: snapshot_access_quantity[active, node]
                for node in data.nodes
            },
            proximal_access_bid={
                node: snapshot_access_bid[active, node] for node in data.nodes
            },
            proximal_energy_capacity={
                node: snapshot_energy[active, node] for node in data.nodes
            },
            proximal_penalty=config.proximal_penalty,
            proximal_price_scale=config.proximal_price_scale,
            proximal_energy_scale=config.proximal_energy_scale,
        )
        if config.warm_start_lower_level:
            mpec_strategic_access_relaxed_kkt.initialise_lower_level(model, data)
        return model
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
    elif config.formulation == "relaxed-kkt":
        model = mpec_relaxed_kkt.build_model(
            **common_kwargs,
            complementarity_epsilon=config.complementarity_epsilon,
        )
    elif config.formulation == "kkt-bigm":
        model = mpec_kkt_bigm.build_model(**common_kwargs)
    elif config.formulation in {
        "strategic-quantity",
        "strategic-price-quantity",
    }:
        if snapshot_bid_charge is None or snapshot_offer_discharge is None:
            raise ValueError(
                "Strategic-quantity best responses require quantity snapshots."
            )
        combined_prices = config.formulation == "strategic-price-quantity"
        if combined_prices and (
            snapshot_bid_charge_price is None
            or snapshot_offer_discharge_price is None
        ):
            raise ValueError(
                "Strategic price-quantity best responses require price snapshots."
            )
        rival_charge_bid_mw = {
            other.investor_id: {
                (n, int(t)): snapshot_bid_charge[other.investor_id, n, int(t)]
                for n in data.nodes
                for t in data.times
            }
            for other in rivals
        }
        rival_discharge_bid_mw = {
            other.investor_id: {
                (n, int(t)): snapshot_offer_discharge[
                    other.investor_id, n, int(t)
                ]
                for n in data.nodes
                for t in data.times
            }
            for other in rivals
        }
        model = mpec_strategic_quantity_relaxed_kkt.build_model(
            **common_kwargs,
            rival_charge_bid_mw=rival_charge_bid_mw,
            rival_discharge_bid_mw=rival_discharge_bid_mw,
            initial_charge_bid_mw={
                (n, int(t)): snapshot_bid_charge[active, n, int(t)]
                for n in data.nodes
                for t in data.times
            },
            initial_discharge_bid_mw={
                (n, int(t)): snapshot_offer_discharge[active, n, int(t)]
                for n in data.nodes
                for t in data.times
            },
            strategic_prices=combined_prices,
            rival_charge_bid_eur_per_mwh=(
                {
                    other.investor_id: {
                        (n, int(t)): snapshot_bid_charge_price[
                            other.investor_id, n, int(t)
                        ]
                        for n in data.nodes
                        for t in data.times
                    }
                    for other in rivals
                }
                if combined_prices
                else None
            ),
            rival_discharge_offer_eur_per_mwh=(
                {
                    other.investor_id: {
                        (n, int(t)): snapshot_offer_discharge_price[
                            other.investor_id, n, int(t)
                        ]
                        for n in data.nodes
                        for t in data.times
                    }
                    for other in rivals
                }
                if combined_prices
                else None
            ),
            initial_charge_bid_eur_per_mwh=(
                {
                    (n, int(t)): snapshot_bid_charge_price[active, n, int(t)]
                    for n in data.nodes
                    for t in data.times
                }
                if combined_prices
                else None
            ),
            initial_discharge_offer_eur_per_mwh=(
                {
                    (n, int(t)): snapshot_offer_discharge_price[
                        active, n, int(t)
                    ]
                    for n in data.nodes
                    for t in data.times
                }
                if combined_prices
                else None
            ),
            bid_price_bound=config.bid_price_bound,
            proximal_price_scale=config.proximal_price_scale,
            proximal_charge_bid_mw=(
                {
                    (n, int(t)): snapshot_bid_charge[active, n, int(t)]
                    for n in data.nodes
                    for t in data.times
                }
                if combined_prices
                else None
            ),
            proximal_discharge_bid_mw=(
                {
                    (n, int(t)): snapshot_offer_discharge[active, n, int(t)]
                    for n in data.nodes
                    for t in data.times
                }
                if combined_prices
                else None
            ),
            proximal_charge_bid_eur_per_mwh=(
                {
                    (n, int(t)): snapshot_bid_charge_price[active, n, int(t)]
                    for n in data.nodes
                    for t in data.times
                }
                if combined_prices
                else None
            ),
            proximal_discharge_offer_eur_per_mwh=(
                {
                    (n, int(t)): snapshot_offer_discharge_price[
                        active, n, int(t)
                    ]
                    for n in data.nodes
                    for t in data.times
                }
                if combined_prices
                else None
            ),
            complementarity_epsilon=config.complementarity_epsilon,
        )
        if any(
            snapshot_power[active, n] > config.cleanup_tolerance
            for n in data.nodes
        ):
            for n in data.nodes:
                model.X_power[n].set_value(snapshot_power[active, n])
                model.X_energy[n].set_value(snapshot_energy[active, n])
    elif config.formulation in {
        "strategic-operation",
        "strategic-price-relaxed-kkt",
    }:
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
        price_builder = (
            mpec_strategic_price_relaxed_kkt.build_model
            if config.formulation == "strategic-price-relaxed-kkt"
            else mpec_strategic_operation.build_model
        )
        model = price_builder(
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
            **(
                {
                    "proximal_bid_charge": {
                        (n, int(t)): snapshot_bid_charge[active, n, int(t)]
                        for n in data.nodes
                        for t in data.times
                    },
                    "proximal_offer_discharge": {
                        (n, int(t)): snapshot_offer_discharge[
                            active, n, int(t)
                        ]
                        for n in data.nodes
                        for t in data.times
                    },
                    "proximal_price_scale": config.proximal_price_scale,
                    "complementarity_epsilon": config.complementarity_epsilon,
                }
                if config.formulation == "strategic-price-relaxed-kkt"
                else {}
            ),
        )
        if (
            config.formulation == "strategic-price-relaxed-kkt"
            and any(
                snapshot_power[active, n] > config.cleanup_tolerance
                for n in data.nodes
            )
        ):
            for n in data.nodes:
                model.X_power[n].set_value(snapshot_power[active, n])
                model.X_energy[n].set_value(snapshot_energy[active, n])
    else:
        raise ValueError(f"Unknown formulation: {config.formulation}")
    if config.warm_start_lower_level:
        if config.formulation in {
            "strategic-quantity",
            "strategic-price-quantity",
        }:
            mpec_strategic_quantity_relaxed_kkt.initialise_lower_level(
                model, data
            )
        elif config.formulation == "strategic-price-relaxed-kkt":
            mpec_strategic_price_relaxed_kkt.initialise_lower_level(
                model, data
            )
        else:
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
    if getattr(model, "_strategic_access", False):
        proposed_power = {
            n: max(0.0, pyo.value(model.X_awarded[investor_id, n]))
            for n in data.nodes
        }
        proposed_energy = {
            n: max(0.0, pyo.value(model.EnergyCapacity[n]))
            for n in data.nodes
        }
        proposed_access_quantity = {
            n: max(0.0, pyo.value(model.AccessQuantity[n])) for n in data.nodes
        }
        proposed_access_bid = {
            n: max(0.0, pyo.value(model.AccessBid[n])) for n in data.nodes
        }
    else:
        proposed_power = {
            n: max(0.0, pyo.value(model.X_power[n])) for n in data.nodes
        }
        proposed_energy = {
            n: max(0.0, pyo.value(model.X_energy[n])) for n in data.nodes
        }
        proposed_access_quantity = {}
        proposed_access_bid = {}
    if hasattr(model, "BidCharge") and outcome.has_solution:
        proposed_bid_charge = {
            (n, int(t)): pyo.value(model.BidCharge[n, t])
            for n in data.nodes
            for t in data.times
        }
    elif hasattr(model, "ChargeBidMW") and outcome.has_solution:
        proposed_bid_charge = {
            (n, int(t)): pyo.value(model.ChargeBidMW[n, t])
            for n in data.nodes
            for t in data.times
        }
    else:
        proposed_bid_charge = {}
    if hasattr(model, "OfferDischarge") and outcome.has_solution:
        proposed_offer_discharge = {
            (n, int(t)): pyo.value(model.OfferDischarge[n, t])
            for n in data.nodes
            for t in data.times
        }
    elif hasattr(model, "DischargeBidMW") and outcome.has_solution:
        proposed_offer_discharge = {
            (n, int(t)): pyo.value(model.DischargeBidMW[n, t])
            for n in data.nodes
            for t in data.times
        }
    else:
        proposed_offer_discharge = {}
    if hasattr(model, "BidChargeEURPerMWh") and outcome.has_solution:
        proposed_bid_charge_price = {
            (n, int(t)): pyo.value(model.BidChargeEURPerMWh[n, t])
            for n in data.nodes
            for t in data.times
        }
        proposed_offer_discharge_price = {
            (n, int(t)): pyo.value(model.OfferDischargeEURPerMWh[n, t])
            for n in data.nodes
            for t in data.times
        }
    else:
        proposed_bid_charge_price = {}
        proposed_offer_discharge_price = {}
    profit = (
        pyo.value(
            model.kkt_unregularized_profit
            if hasattr(model, "kkt_unregularized_profit")
            else model.unregularized_profit
        )
        if outcome.has_solution
        else math.nan
    )
    relaxed_kkt = (
        mpec_relaxed_kkt.diagnostics(model)
        if outcome.has_solution
        and getattr(model, "_lower_level_optimality", None) == "relaxed-kkt"
        else None
    )
    return BestResponseResult(
        investor_id,
        outcome,
        proposed_power,
        proposed_energy,
        profit,
        proposed_bid_charge,
        proposed_offer_discharge,
        (
            float(relaxed_kkt["maximum_product"])
            if relaxed_kkt is not None
            else None
        ),
        (
            max(
                float(relaxed_kkt["maximum_upper_bound_violation"]),
                float(relaxed_kkt["maximum_nonnegativity_violation"]),
            )
            if relaxed_kkt is not None
            else None
        ),
        (
            float(relaxed_kkt["primal_dual_gap_eur_per_day"])
            if relaxed_kkt is not None
            else None
        ),
        proposed_bid_charge_price,
        proposed_offer_discharge_price,
        proposed_access_quantity,
        proposed_access_bid,
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
    combined_strategic = config.formulation == "strategic-price-quantity"
    price_strategic = config.formulation in {
        "strategic-operation",
        "strategic-price-relaxed-kkt",
    }
    quantity_strategic = config.formulation in {
        "strategic-quantity",
        "strategic-price-quantity",
    }
    access_strategic = config.formulation == "strategic-access"
    strategic_bids = price_strategic or quantity_strategic
    stable_sweeps = state.stable_sweeps
    if state.converged and config.stop_at_convergence:
        return state

    for sweep in range(state.sweep + 1, config.max_sweeps + 1):
        effective_damping = (
            1.0
            if access_strategic and sweep <= config.access_undamped_sweeps
            else config.damping
        )
        old_power = dict(state.power)
        old_energy = dict(state.energy)
        old_bid_charge = dict(state.bid_charge)
        old_offer_discharge = dict(state.offer_discharge)
        old_bid_charge_price = dict(state.bid_charge_price)
        old_offer_discharge_price = dict(state.offer_discharge_price)
        old_access_quantity = dict(state.access_quantity)
        old_access_bid = dict(state.access_bid)
        proposals_power: dict[tuple[str, str], float] = {}
        proposals_energy: dict[tuple[str, str], float] = {}
        proposals_bid_charge: dict[tuple[str, str, int], float] = {}
        proposals_offer_discharge: dict[tuple[str, str, int], float] = {}
        proposals_bid_charge_price: dict[tuple[str, str, int], float] = {}
        proposals_offer_discharge_price: dict[tuple[str, str, int], float] = {}
        proposals_access_quantity: dict[tuple[str, str], float] = {}
        proposals_access_bid: dict[tuple[str, str], float] = {}

        if solve_batch is not None:
            responses = solve_batch(
                data,
                config,
                old_power,
                old_energy,
                old_bid_charge,
                old_offer_discharge,
                old_bid_charge_price,
                old_offer_discharge_price,
                old_access_quantity,
                old_access_bid,
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
                    old_bid_charge_price,
                    old_offer_discharge_price,
                    old_access_quantity,
                    old_access_bid,
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
                if access_strategic:
                    proposals_access_quantity[investor_id, n] = (
                        response.proposed_access_quantity[n]
                        if outcome.optimal
                        else old_access_quantity[investor_id, n]
                    )
                    proposals_access_bid[investor_id, n] = (
                        response.proposed_access_bid[n]
                        if outcome.optimal
                        else old_access_bid[investor_id, n]
                    )
                if strategic_bids:
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
                        if combined_strategic:
                            proposals_bid_charge_price[price_key] = (
                                response.proposed_bid_charge_price[n, int(t)]
                                if outcome.optimal
                                else old_bid_charge_price[price_key]
                            )
                            proposals_offer_discharge_price[price_key] = (
                                response.proposed_offer_discharge_price[n, int(t)]
                                if outcome.optimal
                                else old_offer_discharge_price[price_key]
                            )

        if access_strategic:
            for key in state.access_quantity:
                state.access_quantity[key] = (
                    (1.0 - effective_damping) * old_access_quantity[key]
                    + effective_damping * proposals_access_quantity[key]
                )
                state.access_bid[key] = (
                    (1.0 - effective_damping) * old_access_bid[key]
                    + effective_damping * proposals_access_bid[key]
                )
                state.energy[key] = (
                    (1.0 - effective_damping) * old_energy[key]
                    + effective_damping * proposals_energy[key]
                )
                if state.access_quantity[key] < config.cleanup_tolerance:
                    state.access_quantity[key] = 0.0
            _clear_access_allocation(data, config, state)
        else:
            for key in state.power:
                state.power[key] = (
                    (1.0 - config.damping) * old_power[key]
                    + config.damping * proposals_power[key]
                )
                state.energy[key] = (
                    (1.0 - config.damping) * old_energy[key]
                    + config.damping * proposals_energy[key]
                )
                if state.power[key] < config.cleanup_tolerance:
                    state.power[key] = 0.0
                    state.energy[key] = 0.0

        if price_strategic:
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
        elif quantity_strategic:
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
                    state.bid_charge[key] = 0.0
                    state.offer_discharge[key] = 0.0
            if combined_strategic:
                investors_by_id = {
                    investor.investor_id: investor
                    for investor in config.investors
                }
                for key in state.bid_charge_price:
                    investor_id, node, _ = key
                    state.bid_charge_price[key] = (
                        (1.0 - config.damping) * old_bid_charge_price[key]
                        + config.damping * proposals_bid_charge_price[key]
                    )
                    state.offer_discharge_price[key] = (
                        (1.0 - config.damping)
                        * old_offer_discharge_price[key]
                        + config.damping
                        * proposals_offer_discharge_price[key]
                    )
                    if state.power[investor_id, node] < config.cleanup_tolerance:
                        half_cost = min(
                            config.bid_price_bound,
                            0.5
                            * investors_by_id[
                                investor_id
                            ].degradation_eur_per_mwh,
                        )
                        state.bid_charge_price[key] = -half_cost
                        state.offer_discharge_price[key] = half_cost

        for n in (data.nodes if not access_strategic else []):
            total = sum(state.power[investor.investor_id, n] for investor in config.investors)
            if total > config.node_limit_mw + 1e-9:
                scale = config.node_limit_mw / total
                state.projection_count += 1
                for investor in config.investors:
                    key = investor.investor_id, n
                    state.power[key] *= scale
                    state.energy[key] *= scale
                    if quantity_strategic:
                        for t in data.times:
                            bid_key = investor.investor_id, n, int(t)
                            state.bid_charge[bid_key] *= scale
                            state.offer_discharge[bid_key] *= scale

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
        max_raw_primary_bid = max(
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
        ) if strategic_bids else 0.0
        max_iterate_primary_bid = max(
            (
                max(
                    abs(state.bid_charge[key] - old_bid_charge[key]),
                    abs(state.offer_discharge[key] - old_offer_discharge[key]),
                )
                for key in active_price_keys
            ),
            default=0.0,
        ) if strategic_bids else 0.0
        if price_strategic:
            max_raw_price_bid = max_raw_primary_bid
            max_iterate_price_bid = max_iterate_primary_bid
            max_raw_quantity_bid = 0.0
            max_iterate_quantity_bid = 0.0
        else:
            max_raw_quantity_bid = (
                max_raw_primary_bid if quantity_strategic else 0.0
            )
            max_iterate_quantity_bid = (
                max_iterate_primary_bid if quantity_strategic else 0.0
            )
        if combined_strategic:
            active_combined_price_keys = [
                key
                for key in proposals_bid_charge_price
                if max(
                    old_power[key[0], key[1]],
                    proposals_power[key[0], key[1]],
                )
                > config.sparse_capacity_tol
            ]
            max_raw_price_bid = max(
                (
                    max(
                        abs(
                            proposals_bid_charge_price[key]
                            - old_bid_charge_price[key]
                        ),
                        abs(
                            proposals_offer_discharge_price[key]
                            - old_offer_discharge_price[key]
                        ),
                    )
                    for key in active_combined_price_keys
                ),
                default=0.0,
            )
            max_iterate_price_bid = max(
                (
                    max(
                        abs(
                            state.bid_charge_price[key]
                            - old_bid_charge_price[key]
                        ),
                        abs(
                            state.offer_discharge_price[key]
                            - old_offer_discharge_price[key]
                        ),
                    )
                    for key in active_combined_price_keys
                ),
                default=0.0,
            )
        elif not price_strategic:
            max_raw_price_bid = 0.0
            max_iterate_price_bid = 0.0
        if access_strategic:
            max_raw_access_quantity = max(
                abs(proposals_access_quantity[key] - old_access_quantity[key])
                for key in state.access_quantity
            )
            max_iterate_access_quantity = max(
                abs(state.access_quantity[key] - old_access_quantity[key])
                for key in state.access_quantity
            )
            active_access_keys = [
                key
                for key in state.access_quantity
                if max(
                    old_access_quantity[key],
                    proposals_access_quantity[key],
                )
                > config.sparse_capacity_tol
            ]
            max_raw_access_bid = max(
                (
                    abs(proposals_access_bid[key] - old_access_bid[key])
                    for key in active_access_keys
                ),
                default=0.0,
            )
            max_iterate_access_bid = max(
                (
                    abs(state.access_bid[key] - old_access_bid[key])
                    for key in active_access_keys
                ),
                default=0.0,
            )
        else:
            max_raw_access_quantity = 0.0
            max_iterate_access_quantity = 0.0
            max_raw_access_bid = 0.0
            max_iterate_access_bid = 0.0
        all_optimal = all(response.outcome.optimal for response in responses.values())
        is_stable = (
            all_optimal
            and max_raw_power <= config.tolerance_mw
            and max_raw_energy <= config.tolerance_mwh
            and max_raw_quantity_bid <= config.tolerance_quantity_bid_mw
            and max_raw_price_bid <= config.tolerance_bid_eur_per_mwh
            and max_raw_access_quantity <= config.tolerance_mw
            and max_raw_access_bid
            <= config.tolerance_access_bid_eur_per_mw_day
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
                    "proximal_penalty": config.proximal_penalty,
                    "effective_damping": effective_damping,
                    "complementarity_epsilon": (
                        config.complementarity_epsilon
                        if config.formulation
                        in {
                            "relaxed-kkt",
                            "strategic-price-relaxed-kkt",
                            "strategic-quantity",
                            "strategic-price-quantity",
                            "strategic-access",
                        }
                        else None
                    ),
                    "complementarity_max_product": response.complementarity_max_product,
                    "complementarity_max_violation": response.complementarity_max_violation,
                    "primal_dual_gap_eur_per_day": response.primal_dual_gap_eur_per_day,
                    "profit_eur_per_day": response.profit_eur_per_day,
                    "old_power_mw": sum(old_power[investor_id, n] for n in data.nodes),
                    "best_response_power_mw": sum(proposals_power[investor_id, n] for n in data.nodes),
                    "new_power_mw": sum(state.power[investor_id, n] for n in data.nodes),
                    "old_energy_mwh": sum(old_energy[investor_id, n] for n in data.nodes),
                    "best_response_energy_mwh": sum(proposals_energy[investor_id, n] for n in data.nodes),
                    "new_energy_mwh": sum(state.energy[investor_id, n] for n in data.nodes),
                    "old_access_request_mw": (
                        sum(
                            old_access_quantity[investor_id, n]
                            for n in data.nodes
                        )
                        if access_strategic
                        else None
                    ),
                    "best_response_access_request_mw": (
                        sum(
                            proposals_access_quantity[investor_id, n]
                            for n in data.nodes
                        )
                        if access_strategic
                        else None
                    ),
                    "new_access_request_mw": (
                        sum(
                            state.access_quantity[investor_id, n]
                            for n in data.nodes
                        )
                        if access_strategic
                        else None
                    ),
                    "max_raw_deviation_mw": max_raw_power,
                    "max_raw_deviation_mwh": max_raw_energy,
                    "max_iterate_change_mw": max_iterate_power,
                    "max_iterate_change_mwh": max_iterate_energy,
                    "max_raw_bid_deviation_eur_per_mwh": (
                        max_raw_price_bid
                    ),
                    "max_iterate_bid_change_eur_per_mwh": (
                        max_iterate_price_bid
                    ),
                    "max_raw_quantity_bid_deviation_mw": (
                        max_raw_quantity_bid
                    ),
                    "max_iterate_quantity_bid_change_mw": (
                        max_iterate_quantity_bid
                    ),
                    "max_raw_access_quantity_deviation_mw": (
                        max_raw_access_quantity
                    ),
                    "max_iterate_access_quantity_change_mw": (
                        max_iterate_access_quantity
                    ),
                    "max_raw_access_bid_deviation_eur_per_mw_day": (
                        max_raw_access_bid
                    ),
                    "max_iterate_access_bid_change_eur_per_mw_day": (
                        max_iterate_access_bid
                    ),
                }
            )

        state.sweep = sweep
        if stable_sweeps >= config.consecutive_sweeps:
            state.converged = True
            state.stop_reason = f"converged for {stable_sweeps} consecutive sweeps"
        elif not all_optimal:
            state.converged = False
            state.stop_reason = "one or more best responses were not solved to optimality"
        else:
            state.converged = False
            state.stop_reason = "maximum sweeps reached"
        if on_sweep is not None:
            on_sweep(state)
        if state.converged and config.stop_at_convergence:
            break

    return state
