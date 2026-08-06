"""IEEE-9 single-investor MPEC with strategic hourly quantity offers.

The investor chooses nodal BESS MW/MWh and the charge/discharge power made
available in every node-hour. Optionally it also chooses a charging buy-bid
price and a discharging sell-offer price. The ISO retains control of accepted
operation and clears the full PTDF market with battery SOC dynamics. The convex
lower level for fixed bids is embedded through primal feasibility, dual
feasibility/stationarity, and strong duality.

Quantity-only behavior remains available as a reproducible baseline. In the
two-sided mode, submitted prices replace private degradation in the ISO
objective; physical degradation remains in the investor's realized profit.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pyomo.environ as pyo

from single_investor_mpec import (
    DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
    DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    DEFAULT_INITIAL_POWER_MW,
    DEFAULT_INITIAL_RATIO_HOURS,
    DEFAULT_NODE_LIMIT_MW,
    DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    DEFAULT_SOLVER_TOL,
    InvestorConfig,
    QuadraticDemandCurve,
    build_fixed_demand_primal_model,
    build_single_investor_mpec,
    compute_reference_settlement,
    default_quadratic_demand_curve,
    fixed_demand_reference_lambda,
    fixed_storage_data_from_solution,
    initialize_from_reference_dispatch,
    load_market_data,
)
from single_investor_mpec_results import _write_csv, export_solution
from solver_utils import get_ipopt_solver


MODEL_NAME = "IEEE-9 Strategic Storage Quantity-Offer MPEC"
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (
    MODEL_DIR
    / "data"
    / "processed"
    / "market_data_IEEE_9Bus_distributed_congestion.json"
)
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / "strategic_operation_mpec" / "ieee9_single_investor"


def apply_generator_calibration(
    data,
    *,
    conventional_capacity_adder_mw: float = 0.0,
    pv_availability_scale: float = 1.0,
    peaker_node: str | None = None,
    peaker_capacity_mw: float = 0.0,
    peaker_cost_eur_per_mwh: float = 95.0,
):
    """Return a sensitivity copy with optional conventional and peaker capacity.

    A sufficiently large peaker gives scarcity an economic marginal cost. Merely
    enlarging the existing linear generators cannot pin the price when every
    available resource and strategic offer is binding.
    """

    if conventional_capacity_adder_mw < 0.0:
        raise ValueError("Conventional capacity adder must be non-negative.")
    if pv_availability_scale < 0.0:
        raise ValueError("PV availability scale must be non-negative.")
    if peaker_capacity_mw < 0.0 or peaker_cost_eur_per_mwh < 0.0:
        raise ValueError("Peaker capacity and marginal cost must be non-negative.")
    if peaker_capacity_mw > 0.0 and peaker_node not in data.nodes:
        raise ValueError(f"Unknown peaker node: {peaker_node}")

    conventional = [g for g in data.generators if str(g).startswith("G_IEEE")]
    generation_capacity = dict(data.generation_capacity)
    for generator in conventional:
        for time in data.times:
            generation_capacity[generator, time] += conventional_capacity_adder_mw
    pv_generators = [g for g in data.generators if "PV" in str(g).upper()]
    for generator in pv_generators:
        for time in data.times:
            generation_capacity[generator, time] *= pv_availability_scale

    generators = list(data.generators)
    generators_at_node = {
        node: list(node_generators)
        for node, node_generators in data.generators_at_node.items()
    }
    generation_cost = dict(data.generation_cost)
    peaker_id = None
    if peaker_capacity_mw > 0.0:
        peaker_id = f"G_PEAKER_{peaker_node}"
        if peaker_id in generators:
            raise ValueError(f"Peaker {peaker_id} already exists in the dataset.")
        generators.append(peaker_id)
        generators_at_node[peaker_node].append(peaker_id)
        generation_cost[peaker_id] = peaker_cost_eur_per_mwh
        for time in data.times:
            generation_capacity[peaker_id, time] = peaker_capacity_mw

    calibrated = replace(
        data,
        generators=generators,
        generators_at_node=generators_at_node,
        generation_cost=generation_cost,
        generation_capacity=generation_capacity,
    )
    config = {
        "conventional_generators": conventional,
        "conventional_capacity_adder_mw_each": conventional_capacity_adder_mw,
        "pv_generators": pv_generators,
        "pv_availability_scale": pv_availability_scale,
        "peaker_id": peaker_id,
        "peaker_node": peaker_node if peaker_id is not None else None,
        "peaker_capacity_mw": peaker_capacity_mw if peaker_id is not None else 0.0,
        "peaker_cost_eur_per_mwh": peaker_cost_eur_per_mwh if peaker_id is not None else None,
    }
    return calibrated, config


def add_strategic_quantity_offers(
    model: pyo.ConcreteModel,
    *,
    rival_offer_charge_mw_by_unit: Mapping[str, Mapping[tuple[str, int], float]] | None = None,
    rival_offer_discharge_mw_by_unit: Mapping[str, Mapping[tuple[str, int], float]] | None = None,
) -> None:
    """Replace installed-power dispatch bounds with strategic hourly offers.

    The original model's lower-level dual objective contains
    ``X_power * (rho_charge + sigma_discharge)`` because installed power is the
    RHS of both dispatch bounds. Once the investor can withhold quantity, those
    strong-duality terms must use the corresponding hourly offers instead.
    """

    investor = model._investor_id
    rival_charge_input = rival_offer_charge_mw_by_unit or {}
    rival_discharge_input = rival_offer_discharge_mw_by_unit or {}
    rival_ids = set(model._rival_ids)
    if set(rival_charge_input) - rival_ids or set(rival_discharge_input) - rival_ids:
        raise ValueError("Strategic offer mappings contain an unknown rival investor.")

    fixed_rival_charge: dict[tuple[str, str, int], float] = {}
    fixed_rival_discharge: dict[tuple[str, str, int], float] = {}
    for unit, node in model.IN:
        if unit == investor:
            continue
        capacity = model._rival_power_mw_by_unit[unit][node]
        for time in model.T:
            charge = float(rival_charge_input.get(unit, {}).get((node, int(time)), capacity))
            discharge = float(rival_discharge_input.get(unit, {}).get((node, int(time)), capacity))
            if not -1e-9 <= charge <= capacity + 1e-7:
                raise ValueError(f"Rival charge offer exceeds capacity for {unit}, {node}, {time}.")
            if not -1e-9 <= discharge <= capacity + 1e-7:
                raise ValueError(f"Rival discharge offer exceeds capacity for {unit}, {node}, {time}.")
            fixed_rival_charge[unit, node, int(time)] = max(0.0, min(charge, capacity))
            fixed_rival_discharge[unit, node, int(time)] = max(0.0, min(discharge, capacity))

    original_power_bound_dual_term = sum(
        (
            model.X_power[n]
            if unit == investor
            else model._rival_power_mw_by_unit[unit][n]
        )
        * (model.rho_ch[unit, n, t] + model.sig_dis[unit, n, t])
        for unit, n in model.IN
        for t in model.T
    )

    model.Q_offer_charge = pyo.Var(
        model.N,
        model.T,
        domain=pyo.NonNegativeReals,
        initialize=lambda m, n, t: pyo.value(m.X_power[n]),
    )
    model.Q_offer_discharge = pyo.Var(
        model.N,
        model.T,
        domain=pyo.NonNegativeReals,
        initialize=lambda m, n, t: pyo.value(m.X_power[n]),
    )

    # Upper-level feasibility: an investor cannot offer more than it installs.
    model.offer_charge_capacity = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.Q_offer_charge[n, t] <= m.X_power[n],
    )
    model.offer_discharge_capacity = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.Q_offer_discharge[n, t] <= m.X_power[n],
    )

    # Lower-level feasibility: the ISO selects accepted dispatch inside the
    # submitted offer. All network and SOC constraints remain unchanged.
    model.del_component(model.charge_power_bound)
    model.del_component(model.discharge_power_bound)
    model.charge_power_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, unit, n, t: m.P_charge[unit, n, t]
        <= (
            m.Q_offer_charge[n, t]
            if unit == investor
            else fixed_rival_charge[unit, n, int(t)]
        ),
    )
    model.discharge_power_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, unit, n, t: m.P_discharge[unit, n, t]
        <= (
            m.Q_offer_discharge[n, t]
            if unit == investor
            else fixed_rival_discharge[unit, n, int(t)]
        ),
    )

    strategic_offer_dual_term = sum(
        (
            model.Q_offer_charge[n, t]
            if unit == investor
            else fixed_rival_charge[unit, n, int(t)]
        )
        * model.rho_ch[unit, n, t]
        + (
            model.Q_offer_discharge[n, t]
            if unit == investor
            else fixed_rival_discharge[unit, n, int(t)]
        )
        * model.sig_dis[unit, n, t]
        for unit, n in model.IN
        for t in model.T
    )
    model.dual_objective_expr.set_value(
        model.dual_objective_expr.expr
        - original_power_bound_dual_term
        + strategic_offer_dual_term
    )
    model._strategic_quantity_offers = True
    model._rival_offer_charge_mw_by_unit = fixed_rival_charge
    model._rival_offer_discharge_mw_by_unit = fixed_rival_discharge


def add_strategic_price_bids(
    model: pyo.ConcreteModel,
    *,
    rival_bid_price_charge_eur_per_mwh_by_unit: Mapping[
        str, Mapping[tuple[str, int], float]
    ]
    | None = None,
    rival_offer_price_discharge_eur_per_mwh_by_unit: Mapping[
        str, Mapping[tuple[str, int], float]
    ]
    | None = None,
    bid_price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH,
) -> None:
    """Replace ISO-known degradation with strategic two-sided energy bids.

    ``p_bid_charge`` is the maximum willingness to pay for charging and enters
    the lower-level objective with a negative sign. ``p_offer_discharge`` is
    the minimum sell price and enters with a positive sign. Rival prices are
    frozen parameters during the active investor's best response.
    """

    bound = float(bid_price_bound_eur_per_mwh)
    if bound <= 0.0:
        raise ValueError("The strategic bid-price bound must be positive.")

    investor = model._investor_id
    rival_ids = set(model._rival_ids)
    rival_charge_input = rival_bid_price_charge_eur_per_mwh_by_unit or {}
    rival_discharge_input = rival_offer_price_discharge_eur_per_mwh_by_unit or {}
    if set(rival_charge_input) - rival_ids or set(rival_discharge_input) - rival_ids:
        raise ValueError("Strategic price mappings contain an unknown rival investor.")

    fixed_rival_charge: dict[tuple[str, str, int], float] = {}
    fixed_rival_discharge: dict[tuple[str, str, int], float] = {}
    for unit, node in model.IN:
        if unit == investor:
            continue
        truthful_half_cost = 0.5 * model._storage_degradation_eur_per_mwh[unit]
        for time in model.T:
            charge = float(
                rival_charge_input.get(unit, {}).get(
                    (node, int(time)), -truthful_half_cost
                )
            )
            discharge = float(
                rival_discharge_input.get(unit, {}).get(
                    (node, int(time)), truthful_half_cost
                )
            )
            if not -bound - 1e-9 <= charge <= bound + 1e-9:
                raise ValueError(
                    f"Rival charging bid price is outside bounds for {unit}, {node}, {time}."
                )
            if not -bound - 1e-9 <= discharge <= bound + 1e-9:
                raise ValueError(
                    f"Rival discharge offer price is outside bounds for {unit}, {node}, {time}."
                )
            fixed_rival_charge[unit, node, int(time)] = min(bound, max(-bound, charge))
            fixed_rival_discharge[unit, node, int(time)] = min(
                bound, max(-bound, discharge)
            )

    active_half_cost = 0.5 * model._degradation_eur_per_mwh
    model.p_bid_charge = pyo.Var(
        model.N,
        model.T,
        bounds=(-bound, bound),
        initialize=-active_half_cost,
    )
    model.p_offer_discharge = pyo.Var(
        model.N,
        model.T,
        bounds=(-bound, bound),
        initialize=active_half_cost,
    )
    eta = model._market_data.eta
    model.bid_price_cycle_consistency = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.p_offer_discharge[n, t]
        >= m.p_bid_charge[n, t] / (eta**2),
    )

    for unit, node in model.IN:
        if unit == investor:
            continue
        for time in model.T:
            if (
                fixed_rival_discharge[unit, node, int(time)]
                + 1e-8
                < fixed_rival_charge[unit, node, int(time)] / (eta**2)
            ):
                raise ValueError(
                    "Rival bid prices permit a same-hour charge/discharge loop for "
                    f"{unit}, {node}, {time}."
                )

    def charge_price(unit: str, node: str, time: int):
        return (
            model.p_bid_charge[node, time]
            if unit == investor
            else fixed_rival_charge[unit, node, int(time)]
        )

    def discharge_price(unit: str, node: str, time: int):
        return (
            model.p_offer_discharge[node, time]
            if unit == investor
            else fixed_rival_discharge[unit, node, int(time)]
        )

    dispatch_reg = model._dispatch_regularization_eur_per_mw2h
    model.del_component(model.charge_stationarity)
    model.del_component(model.discharge_stationarity)
    model.charge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: -m.lam[n, t]
        + m.rho_ch[i, n, t]
        - eta * m.gam[i, n, t]
        <= -charge_price(i, n, t) + dispatch_reg * m.P_charge[i, n, t],
    )
    model.discharge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.lam[n, t]
        + m.sig_dis[i, n, t]
        + m.gam[i, n, t] / eta
        <= discharge_price(i, n, t) + dispatch_reg * m.P_discharge[i, n, t],
    )

    bid_cost = sum(
        discharge_price(unit, node, time) * model.P_discharge[unit, node, time]
        - charge_price(unit, node, time) * model.P_charge[unit, node, time]
        for unit, node in model.IN
        for time in model.T
    )
    model.lower_level_strategic_bid_cost_expr = pyo.Expression(expr=bid_cost)
    model.primal_objective_expr.set_value(
        model.primal_objective_expr.expr
        - model.lower_level_storage_degradation_expr
        + model.lower_level_strategic_bid_cost_expr
    )
    model._strategic_price_bids = True
    model._strategic_bid_price_bound_eur_per_mwh = bound
    model._rival_bid_price_charge_eur_per_mwh_by_unit = fixed_rival_charge
    model._rival_offer_price_discharge_eur_per_mwh_by_unit = fixed_rival_discharge


def apply_fixed_two_sided_bids_to_primal(
    model: pyo.ConcreteModel,
    *,
    bid_price_charge_eur_per_mwh: Mapping[tuple[str, str, int], float],
    offer_price_discharge_eur_per_mwh: Mapping[tuple[str, str, int], float],
) -> None:
    """Apply fixed two-sided prices to a standalone market re-clear."""

    bid_cost = sum(
        float(offer_price_discharge_eur_per_mwh[i, n, int(t)])
        * model.P_discharge[i, n, t]
        - float(bid_price_charge_eur_per_mwh[i, n, int(t)])
        * model.P_charge[i, n, t]
        for i in model.I
        for n in model.N
        for t in model.T
    )
    model.strategic_bid_objective_expr = pyo.Expression(expr=bid_cost)
    objective = model.quad_objective if hasattr(model, "quad_objective") else model.objective
    objective.set_value(
        objective.expr
        - model.storage_degradation_objective_expr
        + model.strategic_bid_objective_expr
    )
    model._strategic_price_bids = True


def add_proximal_strategy_penalty(
    model: pyo.ConcreteModel,
    *,
    coefficient_eur_per_mw2_day: float = 0.0,
    energy_scale_hours: float = DEFAULT_INITIAL_RATIO_HOURS,
    price_scale_eur_per_mwh: float = 10.0,
    reference_power_mw: Mapping[str, float] | None = None,
    reference_energy_mwh: Mapping[str, float] | None = None,
    reference_offer_charge_mw: Mapping[tuple[str, int], float] | None = None,
    reference_offer_discharge_mw: Mapping[tuple[str, int], float] | None = None,
    reference_bid_price_charge_eur_per_mwh: Mapping[tuple[str, int], float] | None = None,
    reference_offer_price_discharge_eur_per_mwh: Mapping[tuple[str, int], float] | None = None,
) -> None:
    """Add an optional proximal best-response regularizer.

    Capacity changes are measured directly in MW-equivalent squared distance.
    Energy changes are converted to MW with ``energy_scale_hours``. Offer
    changes are expressed as changes in withheld quantity ``X_power - Q_offer``
    and averaged across hours and charge/discharge directions. Consequently,
    the term stabilizes strategic withholding without rewarding full
    availability or double-penalizing offers when installed power moves.
    """

    coefficient = float(coefficient_eur_per_mw2_day)
    scale_hours = float(energy_scale_hours)
    price_scale = float(price_scale_eur_per_mwh)
    if coefficient < 0.0:
        raise ValueError("The proximal penalty coefficient must be non-negative.")
    if scale_hours <= 0.0:
        raise ValueError("The proximal energy scale must be positive.")
    if price_scale <= 0.0:
        raise ValueError("The proximal price scale must be positive.")

    if coefficient == 0.0:
        model.proximal_distance_mw2_expr = pyo.Expression(expr=0.0)
        model.proximal_penalty_expr = pyo.Expression(expr=0.0)
        model._proximal_penalty_eur_per_mw2_day = 0.0
        model._proximal_energy_scale_hours = scale_hours
        model._proximal_price_scale_eur_per_mwh = price_scale
        model._proximal_reference_power_mw = {}
        model._proximal_reference_energy_mwh = {}
        model._proximal_reference_offer_charge_mw = {}
        model._proximal_reference_offer_discharge_mw = {}
        model._proximal_reference_bid_price_charge_eur_per_mwh = {}
        model._proximal_reference_offer_price_discharge_eur_per_mwh = {}
        return

    nodes = list(model.N)
    times = [int(time) for time in model.T]
    mappings = {
        "power": reference_power_mw,
        "energy": reference_energy_mwh,
        "charge offer": reference_offer_charge_mw,
        "discharge offer": reference_offer_discharge_mw,
    }
    if getattr(model, "_strategic_price_bids", False):
        mappings.update(
            {
                "charging bid price": reference_bid_price_charge_eur_per_mwh,
                "discharge offer price": reference_offer_price_discharge_eur_per_mwh,
            }
        )
    missing_mapping = next((name for name, values in mappings.items() if values is None), None)
    if missing_mapping is not None:
        raise ValueError(
            f"A positive proximal coefficient requires a complete {missing_mapping} reference."
        )

    power_reference = {
        node: float((reference_power_mw or {}).get(node, pyo.value(model.X_power[node])))
        for node in nodes
    }
    energy_reference = {
        node: float((reference_energy_mwh or {}).get(node, pyo.value(model.X_energy[node])))
        for node in nodes
    }
    charge_reference = {
        (node, time): float(
            (reference_offer_charge_mw or {}).get((node, time), power_reference[node])
        )
        for node in nodes
        for time in times
    }
    discharge_reference = {
        (node, time): float(
            (reference_offer_discharge_mw or {}).get((node, time), power_reference[node])
        )
        for node in nodes
        for time in times
    }
    charge_price_reference = {
        (node, time): float(
            (reference_bid_price_charge_eur_per_mwh or {}).get(
                (node, time), pyo.value(model.p_bid_charge[node, time])
            )
        )
        for node in nodes
        for time in times
    } if getattr(model, "_strategic_price_bids", False) else {}
    discharge_price_reference = {
        (node, time): float(
            (reference_offer_price_discharge_eur_per_mwh or {}).get(
                (node, time), pyo.value(model.p_offer_discharge[node, time])
            )
        )
        for node in nodes
        for time in times
    } if getattr(model, "_strategic_price_bids", False) else {}

    for node in nodes:
        if power_reference[node] < -1e-9 or energy_reference[node] < -1e-9:
            raise ValueError(f"Negative proximal capacity reference at {node}.")
        for time in times:
            for label, offer in (
                ("charge", charge_reference[node, time]),
                ("discharge", discharge_reference[node, time]),
            ):
                if not -1e-9 <= offer <= power_reference[node] + 1e-7:
                    raise ValueError(
                        f"Invalid proximal {label} offer reference at {node}, {time}."
                    )

    time_count = max(1, len(times))
    power_distance = sum(
        (model.X_power[node] - power_reference[node]) ** 2 for node in nodes
    )
    energy_distance = sum(
        ((model.X_energy[node] - energy_reference[node]) / scale_hours) ** 2
        for node in nodes
    )
    withholding_distance = sum(
        (
            (model.X_power[node] - model.Q_offer_charge[node, time])
            - (power_reference[node] - charge_reference[node, time])
        )
        ** 2
        + (
            (model.X_power[node] - model.Q_offer_discharge[node, time])
            - (power_reference[node] - discharge_reference[node, time])
        )
        ** 2
        for node in nodes
        for time in times
    ) / (2.0 * time_count)
    price_distance = (
        sum(
            (
                (
                    model.p_bid_charge[node, time]
                    - charge_price_reference[node, time]
                )
                / price_scale
            )
            ** 2
            + (
                (
                    model.p_offer_discharge[node, time]
                    - discharge_price_reference[node, time]
                )
                / price_scale
            )
            ** 2
            for node in nodes
            for time in times
        )
        / (2.0 * time_count)
        if getattr(model, "_strategic_price_bids", False)
        else 0.0
    )

    model.proximal_distance_mw2_expr = pyo.Expression(
        expr=power_distance + energy_distance + withholding_distance + price_distance
    )
    model.proximal_penalty_expr = pyo.Expression(
        expr=0.5 * coefficient * model.proximal_distance_mw2_expr
    )
    model.objective.set_value(model.investor_profit_expr - model.proximal_penalty_expr)
    model._proximal_penalty_eur_per_mw2_day = coefficient
    model._proximal_energy_scale_hours = scale_hours
    model._proximal_price_scale_eur_per_mwh = price_scale
    model._proximal_reference_power_mw = power_reference
    model._proximal_reference_energy_mwh = energy_reference
    model._proximal_reference_offer_charge_mw = charge_reference
    model._proximal_reference_offer_discharge_mw = discharge_reference
    model._proximal_reference_bid_price_charge_eur_per_mwh = charge_price_reference
    model._proximal_reference_offer_price_discharge_eur_per_mwh = discharge_price_reference


def add_strategic_epsilon_penalty(
    model: pyo.ConcreteModel,
    *,
    coefficient: float = 0.0,
) -> None:
    """Pin weak two-sided bid prices with a direct epsilon-times-square penalty.

    The selector deliberately excludes installed MW/MWh and quantity offers so
    it cannot directly change their first-order conditions. Quantity strategies
    remain governed by investor profit and, when enabled, the proximal penalty.
    """

    epsilon = float(coefficient)
    if epsilon < 0.0:
        raise ValueError("The strategic epsilon penalty must be non-negative.")

    squared_prices = 0.0
    if getattr(model, "_strategic_price_bids", False):
        squared_prices = sum(
            model.p_bid_charge[node, time] ** 2
            + model.p_offer_discharge[node, time] ** 2
            for node in model.N
            for time in model.T
        )

    model.strategic_epsilon_penalty_expr = pyo.Expression(
        expr=epsilon * squared_prices
    )
    model.objective.set_value(
        model.investor_profit_expr
        - model.proximal_penalty_expr
        - model.strategic_epsilon_penalty_expr
    )
    model._strategic_epsilon_penalty = epsilon


def build_ieee9_strategic_operation_mpec(
    data,
    *,
    initial_power_mw: float = DEFAULT_INITIAL_POWER_MW,
    initial_ratio_hours: float = DEFAULT_INITIAL_RATIO_HOURS,
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW,
    investor: InvestorConfig | None = None,
    rival_power_mw_by_unit: Mapping[str, Mapping[str, float]] | None = None,
    rival_energy_mwh_by_unit: Mapping[str, Mapping[str, float]] | None = None,
    rival_offer_charge_mw_by_unit: Mapping[str, Mapping[tuple[str, int], float]] | None = None,
    rival_offer_discharge_mw_by_unit: Mapping[str, Mapping[tuple[str, int], float]] | None = None,
    strategic_bid_prices: bool = False,
    rival_bid_price_charge_eur_per_mwh_by_unit: Mapping[
        str, Mapping[tuple[str, int], float]
    ]
    | None = None,
    rival_offer_price_discharge_eur_per_mwh_by_unit: Mapping[
        str, Mapping[tuple[str, int], float]
    ]
    | None = None,
    bid_price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    rival_degradation_eur_per_mwh_by_unit: Mapping[str, float] | None = None,
    price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    price_lower_bound_eur_per_mwh: float | None = None,
    price_upper_bound_eur_per_mwh: float | None = None,
    dual_bound_eur_per_mwh: float = DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    dispatch_regularization_eur_per_mw2h: float = DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
    system_price_settlement: bool = False,
    solver_tol: float = DEFAULT_SOLVER_TOL,
    quad_demand: QuadraticDemandCurve | None = None,
    use_demand_curve: bool = False,
    proximal_penalty_eur_per_mw2_day: float = 0.0,
    proximal_energy_scale_hours: float = DEFAULT_INITIAL_RATIO_HOURS,
    proximal_price_scale_eur_per_mwh: float = 10.0,
    proximal_reference_power_mw: Mapping[str, float] | None = None,
    proximal_reference_energy_mwh: Mapping[str, float] | None = None,
    proximal_reference_offer_charge_mw: Mapping[tuple[str, int], float] | None = None,
    proximal_reference_offer_discharge_mw: Mapping[tuple[str, int], float] | None = None,
    proximal_reference_bid_price_charge_eur_per_mwh: Mapping[
        tuple[str, int], float
    ]
    | None = None,
    proximal_reference_offer_price_discharge_eur_per_mwh: Mapping[
        tuple[str, int], float
    ]
    | None = None,
    strategic_epsilon_penalty: float = 0.0,
    initialize_model: bool = True,
) -> pyo.ConcreteModel:
    """Build the strategic-operation MPEC on a supplied market dataset."""

    model = build_single_investor_mpec(
        data,
        initial_power_mw=initial_power_mw,
        initial_ratio_hours=initial_ratio_hours,
        node_limit_mw=node_limit_mw,
        price_bound_eur_per_mwh=price_bound_eur_per_mwh,
        price_lower_bound_eur_per_mwh=price_lower_bound_eur_per_mwh,
        price_upper_bound_eur_per_mwh=price_upper_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=dual_bound_eur_per_mwh,
        quad_demand=quad_demand or default_quadratic_demand_curve(),
        use_demand_curve=use_demand_curve,
        investor=investor or InvestorConfig(),
        rival_power_mw_by_unit=rival_power_mw_by_unit,
        rival_energy_mwh_by_unit=rival_energy_mwh_by_unit,
        rival_degradation_eur_per_mwh_by_unit=rival_degradation_eur_per_mwh_by_unit,
        dispatch_regularization_eur_per_mw2h=dispatch_regularization_eur_per_mw2h,
        system_price_settlement=system_price_settlement,
        solver_tol=solver_tol,
    )
    model.name = MODEL_NAME
    add_strategic_quantity_offers(
        model,
        rival_offer_charge_mw_by_unit=rival_offer_charge_mw_by_unit,
        rival_offer_discharge_mw_by_unit=rival_offer_discharge_mw_by_unit,
    )
    if strategic_bid_prices:
        add_strategic_price_bids(
            model,
            rival_bid_price_charge_eur_per_mwh_by_unit=(
                rival_bid_price_charge_eur_per_mwh_by_unit
            ),
            rival_offer_price_discharge_eur_per_mwh_by_unit=(
                rival_offer_price_discharge_eur_per_mwh_by_unit
            ),
            bid_price_bound_eur_per_mwh=bid_price_bound_eur_per_mwh,
        )
    else:
        model._strategic_price_bids = False
    add_proximal_strategy_penalty(
        model,
        coefficient_eur_per_mw2_day=proximal_penalty_eur_per_mw2_day,
        energy_scale_hours=proximal_energy_scale_hours,
        price_scale_eur_per_mwh=proximal_price_scale_eur_per_mwh,
        reference_power_mw=proximal_reference_power_mw,
        reference_energy_mwh=proximal_reference_energy_mwh,
        reference_offer_charge_mw=proximal_reference_offer_charge_mw,
        reference_offer_discharge_mw=proximal_reference_offer_discharge_mw,
        reference_bid_price_charge_eur_per_mwh=(
            proximal_reference_bid_price_charge_eur_per_mwh
        ),
        reference_offer_price_discharge_eur_per_mwh=(
            proximal_reference_offer_price_discharge_eur_per_mwh
        ),
    )
    add_strategic_epsilon_penalty(
        model,
        coefficient=strategic_epsilon_penalty,
    )
    if initialize_model:
        initialize_from_reference_dispatch(model, data, initial_ratio_hours)

        # The reference initialization uses the full installed power, which is
        # also the starting offer. Keep the relationship explicit afterwards.
        for n in model.N:
            for t in model.T:
                model.Q_offer_charge[n, t].set_value(pyo.value(model.X_power[n]))
                model.Q_offer_discharge[n, t].set_value(pyo.value(model.X_power[n]))
    model._initial_power_mw = initial_power_mw
    model._initial_ratio_hours = initial_ratio_hours
    return model


def solve_offer_reclear(model: pyo.ConcreteModel) -> dict[str, object]:
    """Independently re-clear the ISO problem with investment and offers fixed."""

    fixed_data = fixed_storage_data_from_solution(model)
    reference = build_fixed_demand_primal_model(
        fixed_data,
        storage_degradation_eur_per_mwh=model._storage_degradation_eur_per_mwh,
        dispatch_regularization_eur_per_mw2h=model._dispatch_regularization_eur_per_mw2h,
    )
    if getattr(model, "_strategic_price_bids", False):
        charge_prices = {}
        discharge_prices = {}
        for unit in reference.I:
            half_cost = 0.5 * model._storage_degradation_eur_per_mwh[str(unit)]
            for node in reference.N:
                for time in reference.T:
                    key = str(unit), str(node), int(time)
                    if str(unit) == model._investor_id:
                        charge_prices[key] = pyo.value(model.p_bid_charge[node, time])
                        discharge_prices[key] = pyo.value(
                            model.p_offer_discharge[node, time]
                        )
                    else:
                        charge_prices[key] = model._rival_bid_price_charge_eur_per_mwh_by_unit.get(
                            key, -half_cost
                        )
                        discharge_prices[key] = model._rival_offer_price_discharge_eur_per_mwh_by_unit.get(
                            key, half_cost
                        )
        apply_fixed_two_sided_bids_to_primal(
            reference,
            bid_price_charge_eur_per_mwh=charge_prices,
            offer_price_discharge_eur_per_mwh=discharge_prices,
        )
    investor = model._investor_id
    reference.offer_charge_bound = pyo.Constraint(
        reference.N,
        reference.T,
        rule=lambda m, n, t: m.P_charge[investor, n, t]
        <= pyo.value(model.Q_offer_charge[n, t]),
    )
    reference.offer_discharge_bound = pyo.Constraint(
        reference.N,
        reference.T,
        rule=lambda m, n, t: m.P_discharge[investor, n, t]
        <= pyo.value(model.Q_offer_discharge[n, t]),
    )
    results = get_ipopt_solver(
        {"tol": model._solver_tol, "acceptable_tol": model._solver_tol}
    ).solve(reference, tee=False)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            "Offer-constrained ISO re-clear failed: "
            f"{results.solver.termination_condition}"
        )

    prices = fixed_demand_reference_lambda(reference)
    reference_objective = pyo.value(reference.objective)
    embedded_objective = pyo.value(model.primal_objective_expr)
    max_dispatch_difference = max(
        max(
            abs(
                pyo.value(model.P_charge[investor, n, t])
                - pyo.value(reference.P_charge[investor, n, t])
            ),
            abs(
                pyo.value(model.P_discharge[investor, n, t])
                - pyo.value(reference.P_discharge[investor, n, t])
            ),
        )
        for n in model.N
        for t in model.T
    )
    max_price_difference = max(
        abs(pyo.value(model.lam[n, t]) - prices[n, t])
        for n in model.N
        for t in model.T
    )
    revenue = sum(
        prices[n, t]
        * (
            pyo.value(reference.P_discharge[investor, n, t])
            - pyo.value(reference.P_charge[investor, n, t])
        )
        for n in reference.N
        for t in reference.T
    )
    degradation = 0.5 * model._degradation_eur_per_mwh * sum(
        pyo.value(reference.P_charge[investor, n, t])
        + pyo.value(reference.P_discharge[investor, n, t])
        for n in reference.N
        for t in reference.T
    )
    profit = revenue - degradation - pyo.value(model.capex_daily_expr)
    return {
        "reference_model": reference,
        "reference_lambda": prices,
        "solver_status": str(results.solver.status),
        "termination": str(results.solver.termination_condition),
        "lower_level_objective_eur_per_day": reference_objective,
        "embedded_primal_objective_eur_per_day": embedded_objective,
        "objective_difference_eur_per_day": embedded_objective - reference_objective,
        "max_dispatch_difference_mw": max_dispatch_difference,
        "max_price_difference_eur_per_mwh": max_price_difference,
        "spot_revenue_eur_per_day": revenue,
        "degradation_cost_eur_per_day": degradation,
        "profit_eur_per_day": profit,
    }


def offer_metrics(model: pyo.ConcreteModel) -> dict[str, float | int]:
    investor = model._investor_id
    total_power = sum(pyo.value(model.X_power[n]) for n in model.N)
    denominator = len(list(model.T)) * total_power
    charge_offer = sum(
        pyo.value(model.Q_offer_charge[n, t]) for n in model.N for t in model.T
    )
    discharge_offer = sum(
        pyo.value(model.Q_offer_discharge[n, t]) for n in model.N for t in model.T
    )
    charge_accepted = sum(
        pyo.value(model.P_charge[investor, n, t]) for n in model.N for t in model.T
    )
    discharge_accepted = sum(
        pyo.value(model.P_discharge[investor, n, t]) for n in model.N for t in model.T
    )
    metrics = {
        "charge_offer_capacity_hours_mwh": charge_offer,
        "discharge_offer_capacity_hours_mwh": discharge_offer,
        "charge_offer_fraction_of_installed_capacity_hours": (
            charge_offer / denominator if denominator > 0.0 else 0.0
        ),
        "discharge_offer_fraction_of_installed_capacity_hours": (
            discharge_offer / denominator if denominator > 0.0 else 0.0
        ),
        "accepted_charge_mwh": charge_accepted,
        "accepted_discharge_mwh": discharge_accepted,
        "binding_charge_offer_node_hours": sum(
            pyo.value(model.X_power[n]) > 1e-4
            and abs(
                pyo.value(model.Q_offer_charge[n, t])
                - pyo.value(model.P_charge[investor, n, t])
            )
            <= 1e-4
            for n in model.N
            for t in model.T
        ),
        "binding_discharge_offer_node_hours": sum(
            pyo.value(model.X_power[n]) > 1e-4
            and abs(
                pyo.value(model.Q_offer_discharge[n, t])
                - pyo.value(model.P_discharge[investor, n, t])
            )
            <= 1e-4
            for n in model.N
            for t in model.T
        ),
    }
    if getattr(model, "_strategic_price_bids", False):
        count = max(1, len(list(model.N)) * len(list(model.T)))
        metrics.update(
            {
                "mean_charge_bid_price_eur_per_mwh": sum(
                    pyo.value(model.p_bid_charge[n, t]) for n in model.N for t in model.T
                ) / count,
                "mean_discharge_offer_price_eur_per_mwh": sum(
                    pyo.value(model.p_offer_discharge[n, t])
                    for n in model.N
                    for t in model.T
                ) / count,
                "min_charge_bid_price_eur_per_mwh": min(
                    pyo.value(model.p_bid_charge[n, t]) for n in model.N for t in model.T
                ),
                "max_charge_bid_price_eur_per_mwh": max(
                    pyo.value(model.p_bid_charge[n, t]) for n in model.N for t in model.T
                ),
                "min_discharge_offer_price_eur_per_mwh": min(
                    pyo.value(model.p_offer_discharge[n, t])
                    for n in model.N
                    for t in model.T
                ),
                "max_discharge_offer_price_eur_per_mwh": max(
                    pyo.value(model.p_offer_discharge[n, t])
                    for n in model.N
                    for t in model.T
                ),
            }
        )
    return metrics


def export_strategic_solution(
    model: pyo.ConcreteModel,
    output_dir: Path,
    solver_status: str,
    termination: str,
    offer_reclear: dict[str, object],
    full_availability: dict[str, object],
) -> None:
    """Export standard MPEC results plus strategic offers and diagnostics."""

    export_solution(
        model,
        output_dir,
        solver_status,
        termination,
        full_availability,
    )
    investor = model._investor_id
    _write_csv(
        output_dir / "strategic_quantity_offers.csv",
        [
            "hour",
            "node",
            "installed_power_mw",
            "charge_offer_mw",
            "charge_bid_price_eur_per_mwh",
            "accepted_charge_mw",
            "discharge_offer_mw",
            "discharge_offer_price_eur_per_mwh",
            "accepted_discharge_mw",
            "embedded_lambda_eur_per_mwh",
        ],
        [
            {
                "hour": t,
                "node": n,
                "installed_power_mw": pyo.value(model.X_power[n]),
                "charge_offer_mw": pyo.value(model.Q_offer_charge[n, t]),
                "charge_bid_price_eur_per_mwh": (
                    pyo.value(model.p_bid_charge[n, t])
                    if getattr(model, "_strategic_price_bids", False)
                    else None
                ),
                "accepted_charge_mw": pyo.value(model.P_charge[investor, n, t]),
                "discharge_offer_mw": pyo.value(model.Q_offer_discharge[n, t]),
                "discharge_offer_price_eur_per_mwh": (
                    pyo.value(model.p_offer_discharge[n, t])
                    if getattr(model, "_strategic_price_bids", False)
                    else None
                ),
                "accepted_discharge_mw": pyo.value(model.P_discharge[investor, n, t]),
                "embedded_lambda_eur_per_mwh": pyo.value(model.lam[n, t]),
            }
            for t in model.T
            for n in model.N
        ],
    )
    _write_csv(
        output_dir / "offer_reclear_prices.csv",
        ["hour", "node", "lambda_offer_reclear_eur_per_mwh"],
        [
            {
                "hour": t,
                "node": n,
                "lambda_offer_reclear_eur_per_mwh": offer_reclear["reference_lambda"][n, t],
            }
            for t in model.T
            for n in model.N
        ],
    )
    full_availability_prices = full_availability["reference_lambda"]
    _write_csv(
        output_dir / "lmp_withholding_comparison.csv",
        [
            "hour",
            "node",
            "lambda_strategic_offer_eur_per_mwh",
            "lambda_full_availability_eur_per_mwh",
            "strategic_minus_full_availability_eur_per_mwh",
        ],
        [
            {
                "hour": t,
                "node": n,
                "lambda_strategic_offer_eur_per_mwh": pyo.value(model.lam[n, t]),
                "lambda_full_availability_eur_per_mwh": full_availability_prices[n, t],
                "strategic_minus_full_availability_eur_per_mwh": (
                    pyo.value(model.lam[n, t]) - full_availability_prices[n, t]
                ),
            }
            for t in model.T
            for n in model.N
        ],
    )
    plot_lmp_withholding_comparison(model, full_availability_prices, output_dir)

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": (
                "strategic_two_sided_price_quantity_bids_with_iso_dispatch"
                if getattr(model, "_strategic_price_bids", False)
                else "strategic_hourly_quantity_offers_with_iso_dispatch"
            ),
            "data_path": getattr(model, "_data_path", None),
            "initial_power_mw_per_node": model._initial_power_mw,
            "initial_ratio_hours": model._initial_ratio_hours,
            "generator_calibration": getattr(model, "_generator_calibration", None),
            "strategic_bid_prices": getattr(model, "_strategic_price_bids", False),
            "strategic_epsilon_penalty": getattr(
                model, "_strategic_epsilon_penalty", 0.0
            ),
            "epsilon_penalty_eur_per_day": pyo.value(
                model.strategic_epsilon_penalty_expr
            ),
            "offer_metrics": offer_metrics(model),
            "offer_reclear_has_material_primal_or_dual_nonuniqueness": (
                offer_reclear["max_dispatch_difference_mw"] > 1e-3
                or offer_reclear["max_price_difference_eur_per_mwh"] > 1.0
            ),
            "offer_reclear": {
                key: value
                for key, value in offer_reclear.items()
                if key not in {"reference_model", "reference_lambda"}
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_lmp_withholding_comparison(
    model: pyo.ConcreteModel,
    full_availability_prices,
    output_dir: Path,
) -> None:
    """Plot hourly LMP evolution with strategic offers and full availability."""

    import matplotlib.pyplot as plt

    hours = list(model.T)
    strategic = {
        node: [pyo.value(model.lam[node, time]) for time in hours]
        for node in model.N
    }
    full = {
        node: [full_availability_prices[node, time] for time in hours]
        for node in model.N
    }
    strategic_min = [min(strategic[node][k] for node in model.N) for k in range(len(hours))]
    strategic_max = [max(strategic[node][k] for node in model.N) for k in range(len(hours))]
    full_min = [min(full[node][k] for node in model.N) for k in range(len(hours))]
    full_max = [max(full[node][k] for node in model.N) for k in range(len(hours))]

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True)
    axes[0].fill_between(hours, strategic_min, strategic_max, color="#d95f02", alpha=0.22)
    axes[0].plot(hours, strategic_max, color="#d95f02", linewidth=1.8, label="Strategic nodal maximum")
    axes[0].plot(hours, strategic_min, color="#d95f02", linewidth=1.0, linestyle=":", label="Strategic nodal minimum")
    axes[0].fill_between(hours, full_min, full_max, color="#1f77b4", alpha=0.18)
    axes[0].plot(hours, full_max, color="#1f77b4", linewidth=1.8, label="Full-availability nodal maximum")
    axes[0].plot(hours, full_min, color="#1f77b4", linewidth=1.0, linestyle=":", label="Full-availability nodal minimum")
    axes[0].set_ylabel("LMP [EUR/MWh]")
    axes[0].set_title("System-wide nodal LMP envelope")
    axes[0].legend(ncol=2, fontsize=8)

    for node, color in (("N5", "#b2182b"), ("N8", "#2166ac")):
        axes[1].plot(hours, strategic[node], color=color, linewidth=1.8, label=f"{node} strategic")
        axes[1].plot(hours, full[node], color=color, linewidth=1.5, linestyle="--", label=f"{node} full availability")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("LMP [EUR/MWh]")
    axes[1].set_title("Congested load node N5 and storage node N8")
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].set_xticks(hours)

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(min(hours), max(hours))
    fig.tight_layout()
    fig.savefig(output_dir / "lmp_with_vs_without_strategic_withholding.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "lmp_with_vs_without_strategic_withholding.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--initial-power-mw", type=float, default=DEFAULT_INITIAL_POWER_MW)
    parser.add_argument("--initial-ratio-hours", type=float, default=4.0)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--conventional-capacity-adder-mw", type=float, default=0.0)
    parser.add_argument("--peaker-node", choices=[f"N{k}" for k in range(1, 10)], default=None)
    parser.add_argument("--peaker-capacity-mw", type=float, default=0.0)
    parser.add_argument("--peaker-cost-eur-per-mwh", type=float, default=95.0)
    parser.add_argument("--solver-tol", type=float, default=DEFAULT_SOLVER_TOL)
    parser.add_argument(
        "--dispatch-regularization",
        type=float,
        default=DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
        help="Optional neutral lower-level quadratic tie-break in EUR/(MW^2 h).",
    )
    parser.add_argument("--max-cpu-time", type=float, default=180.0)
    parser.add_argument("--price-bound-eur-per-mwh", type=float, default=DEFAULT_PRICE_BOUND_EUR_PER_MWH)
    parser.add_argument("--dual-bound-eur-per-mwh", type=float, default=DEFAULT_DUAL_BOUND_EUR_PER_MWH)
    parser.add_argument("--strategic-bid-prices", action="store_true")
    parser.add_argument(
        "--bid-price-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dispatch_regularization < 0.0:
        raise SystemExit("--dispatch-regularization must be non-negative.")
    if args.solver_tol <= 0.0:
        raise SystemExit("--solver-tol must be positive.")
    if args.bid_price_bound_eur_per_mwh <= 0.0:
        raise SystemExit("--bid-price-bound-eur-per-mwh must be positive.")
    base_data = load_market_data(args.data)
    data, generator_calibration = apply_generator_calibration(
        base_data,
        conventional_capacity_adder_mw=args.conventional_capacity_adder_mw,
        peaker_node=args.peaker_node,
        peaker_capacity_mw=args.peaker_capacity_mw,
        peaker_cost_eur_per_mwh=args.peaker_cost_eur_per_mwh,
    )
    model = build_ieee9_strategic_operation_mpec(
        data,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        node_limit_mw=args.node_limit_mw,
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
        strategic_bid_prices=args.strategic_bid_prices,
        bid_price_bound_eur_per_mwh=args.bid_price_bound_eur_per_mwh,
        dispatch_regularization_eur_per_mw2h=args.dispatch_regularization,
        solver_tol=args.solver_tol,
    )
    model._data_path = str(args.data)
    model._generator_calibration = generator_calibration
    solver = get_ipopt_solver(
        {
            "max_cpu_time": args.max_cpu_time,
            "tol": args.solver_tol,
            "acceptable_tol": args.solver_tol,
        }
    )
    results = solver.solve(model, tee=args.tee)
    termination = results.solver.termination_condition
    print(f"Solver status: {results.solver.status}")
    print(f"Termination: {termination}")
    if termination != pyo.TerminationCondition.optimal:
        return 1

    offer_reclear = solve_offer_reclear(model)
    full_availability = compute_reference_settlement(model)
    strong_duality_gap = abs(
        pyo.value(model.primal_objective_expr) - pyo.value(model.dual_objective_expr)
    )

    print("\nStrategic-operation MPEC")
    print(f"  profit: {pyo.value(model.investor_profit_expr):,.3f} EUR/day")
    print(
        "  investment: "
        f"{sum(pyo.value(model.X_power[n]) for n in model.N):,.3f} MW / "
        f"{sum(pyo.value(model.X_energy[n]) for n in model.N):,.3f} MWh"
    )
    for n in model.N:
        if pyo.value(model.X_power[n]) > 1e-3:
            print(
                f"  {n}: {pyo.value(model.X_power[n]):,.3f} MW / "
                f"{pyo.value(model.X_energy[n]):,.3f} MWh"
            )
    print(f"  strong-duality gap: {strong_duality_gap:.3e}")
    print(
        "  offer re-clear lower-level objective difference: "
        f"{offer_reclear['objective_difference_eur_per_day']:.3e} EUR/day"
    )
    if (
        offer_reclear["max_dispatch_difference_mw"] > 1e-3
        or offer_reclear["max_price_difference_eur_per_mwh"] > 1.0
    ):
        print(
            "  nonunique re-clear diagnostic (dispatch/price): "
            f"{offer_reclear['max_dispatch_difference_mw']:.3f} MW / "
            f"{offer_reclear['max_price_difference_eur_per_mwh']:.3f} EUR/MWh"
        )
    metrics = offer_metrics(model)
    print(
        "  offered charge/discharge capacity-hours: "
        f"{metrics['charge_offer_fraction_of_installed_capacity_hours']:.1%} / "
        f"{metrics['discharge_offer_fraction_of_installed_capacity_hours']:.1%}"
    )
    print(
        "  full-availability counterfactual profit at reference prices: "
        f"{full_availability['profit_at_reference_prices_eur_per_day']:,.3f} EUR/day"
    )
    print("  NOTE: this is a local nonconvex NLP solution; multi-start comparison is required.")

    if not args.no_export:
        export_strategic_solution(
            model,
            args.output_dir,
            str(results.solver.status),
            str(termination),
            offer_reclear,
            full_availability,
        )
        print(f"\nWrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
