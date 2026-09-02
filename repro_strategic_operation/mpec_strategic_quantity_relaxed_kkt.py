"""Strategic price-and-quantity BESS MPEC with relaxed lower-level KKT.

One active investor chooses nodal BESS power and energy capacity together with
two hourly bid pairs: a maximum charging quantity and willingness-to-pay, and
a maximum discharging quantity and sell offer.  Quantities are availability
ceilings, not schedules: the lower-level ISO selects realised storage dispatch
and enforces the cyclic state-of-charge equations.  Setting
``strategic_prices=False`` fixes both price profiles to zero and exactly retains
the maintained quantity-only formulation.

For fixed upper-level bids the ISO solves

    min generation_cost
        + sum(offer_discharge * P_discharge
              - bid_charge * P_charge).

The active investor is settled only on realised dispatch at the nodal LMP.
Submitted prices are therefore merit-order parameters, not pay-as-bid
payments.  The restriction

    offer_discharge >= bid_charge / eta**2

excludes a negative-cost same-hour storage cycle.

The active investor's complete one-hour quantity offers must be deliverable
from the anticipated beginning-of-hour SOC. Physical degradation remains a
private cost in the investor's upper-level profit and is not added to the ISO
bid-cost objective.

Lower-level optimality is represented by primal feasibility, dual
feasibility/stationarity, and the Scholtes relaxation

    0 <= primal_or_slack * dual_or_reduced_cost <= epsilon.

The formulation is therefore an approximate, nonconvex NLP rather than an
exact KKT reformulation.  Use :func:`diagnostics` to audit every product and
the implied primal-dual gap after solving.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pyomo.environ as pyo

from mpec_strong_duality import InvestorConfig, build_model as build_capacity_model
from primal_market_clearing_model import (
    MarketData,
    build_primal_market_clearing_model,
)


DEFAULT_COMPLEMENTARITY_EPSILON = 1.0e-3
DEFAULT_BID_PRICE_BOUND = 500.0
DEFAULT_PROXIMAL_PRICE_SCALE = 10.0

QuantityProfile = Mapping[str, Mapping[tuple[str, int], float]]
PriceProfile = Mapping[str, Mapping[tuple[str, int], float]]


def build_model(
    data: MarketData,
    *,
    investor: InvestorConfig,
    rival_charge_bid_mw: QuantityProfile | None = None,
    rival_discharge_bid_mw: QuantityProfile | None = None,
    initial_charge_bid_mw: Mapping[tuple[str, int], float] | None = None,
    initial_discharge_bid_mw: Mapping[tuple[str, int], float] | None = None,
    strategic_prices: bool = False,
    rival_charge_bid_eur_per_mwh: PriceProfile | None = None,
    rival_discharge_offer_eur_per_mwh: PriceProfile | None = None,
    initial_charge_bid_eur_per_mwh: Mapping[tuple[str, int], float] | None = None,
    initial_discharge_offer_eur_per_mwh: Mapping[tuple[str, int], float] | None = None,
    bid_price_bound: float = DEFAULT_BID_PRICE_BOUND,
    proximal_charge_bid_mw: Mapping[tuple[str, int], float] | None = None,
    proximal_discharge_bid_mw: Mapping[tuple[str, int], float] | None = None,
    proximal_charge_bid_eur_per_mwh: Mapping[tuple[str, int], float] | None = None,
    proximal_discharge_offer_eur_per_mwh: Mapping[tuple[str, int], float] | None = None,
    proximal_price_scale: float = DEFAULT_PROXIMAL_PRICE_SCALE,
    complementarity_epsilon: float = DEFAULT_COMPLEMENTARITY_EPSILON,
    **kwargs: object,
) -> pyo.ConcreteModel:
    """Build one strategic price-quantity-and-capacity best-response MPEC.

    Rival capacities, quantities, and prices are fixed. Missing rival quantity
    entries default to full installed power; missing prices default to zero.
    """

    epsilon = float(complementarity_epsilon)
    if epsilon < 0.0:
        raise ValueError("complementarity_epsilon must be non-negative.")
    price_bound = float(bid_price_bound)
    price_scale = float(proximal_price_scale)
    if price_bound <= 0.0 or price_scale <= 0.0:
        raise ValueError("Bid-price bound and proximal price scale must be positive.")

    # Extend the maintained moving L1 capacity regularizer below with direct
    # L1 deviations of quantities and scaled prices. Quantity-only calls retain
    # the established capacity-only behavior without alteration.
    proximal_coefficient = float(kwargs.get("proximal_penalty", 0.0))
    proximal_power = kwargs.get("proximal_power")
    proximal_energy = kwargs.get("proximal_energy")

    model = build_capacity_model(data, investor=investor, **kwargs)
    model.name = (
        f"Strategic-price-quantity relaxed-KKT MPEC [{investor.investor_id}]"
        if strategic_prices
        else f"Strategic-quantity relaxed-KKT MPEC [{investor.investor_id}]"
    )
    model.strong_duality.deactivate()

    active = investor.investor_id
    rival_ids = set(model._rival_ids)
    charge_input = rival_charge_bid_mw or {}
    discharge_input = rival_discharge_bid_mw or {}
    charge_price_input = rival_charge_bid_eur_per_mwh or {}
    discharge_price_input = rival_discharge_offer_eur_per_mwh or {}
    if (
        set(charge_input) - rival_ids
        or set(discharge_input) - rival_ids
        or set(charge_price_input) - rival_ids
        or set(discharge_price_input) - rival_ids
    ):
        raise ValueError("Strategic bid mappings contain an unknown rival investor.")

    fixed_rival_charge: dict[tuple[str, str, int], float] = {}
    fixed_rival_discharge: dict[tuple[str, str, int], float] = {}
    fixed_rival_charge_price: dict[tuple[str, str, int], float] = {}
    fixed_rival_discharge_price: dict[tuple[str, str, int], float] = {}
    for unit, node in model.IN:
        if unit == active:
            continue
        installed_power = model._rival_power[unit][node]
        for time in model.T:
            key = node, int(time)
            charge = float(charge_input.get(unit, {}).get(key, installed_power))
            discharge = float(
                discharge_input.get(unit, {}).get(key, installed_power)
            )
            if not 0.0 <= charge <= installed_power + 1.0e-9:
                raise ValueError(
                    f"Rival charging bid outside [0, installed MW] for "
                    f"{unit}, {node}, {time}."
                )
            if not 0.0 <= discharge <= installed_power + 1.0e-9:
                raise ValueError(
                    f"Rival discharging bid outside [0, installed MW] for "
                    f"{unit}, {node}, {time}."
                )
            fixed_rival_charge[unit, node, int(time)] = min(
                charge, installed_power
            )
            fixed_rival_discharge[unit, node, int(time)] = min(
                discharge, installed_power
            )
            charge_price = float(charge_price_input.get(unit, {}).get(key, 0.0))
            discharge_price = float(
                discharge_price_input.get(unit, {}).get(key, 0.0)
            )
            if max(abs(charge_price), abs(discharge_price)) > price_bound + 1.0e-9:
                raise ValueError(
                    f"Rival price outside [-{price_bound:g}, {price_bound:g}] "
                    f"for {unit}, {node}, {time}."
                )
            if discharge_price + 1.0e-9 < charge_price / (data.eta**2):
                raise ValueError(
                    "Rival prices permit a negative-cost storage cycle for "
                    f"{unit}, {node}, {time}."
                )
            fixed_rival_charge_price[unit, node, int(time)] = charge_price
            fixed_rival_discharge_price[unit, node, int(time)] = discharge_price

    initial_charge = initial_charge_bid_mw or {}
    initial_discharge = initial_discharge_bid_mw or {}
    initial_charge_price = initial_charge_bid_eur_per_mwh or {}
    initial_discharge_price = initial_discharge_offer_eur_per_mwh or {}

    def bid_bounds(_: pyo.ConcreteModel, node: str, __: int):
        return 0.0, model._headroom[node]

    def bid_initial(
        values: Mapping[tuple[str, int], float], node: str, time: int
    ) -> float:
        value = float(values.get((node, int(time)), 0.0))
        return min(model._headroom[node], max(0.0, value))

    model.ChargeBidMW = pyo.Var(
        model.N,
        model.T,
        bounds=bid_bounds,
        initialize=lambda _, n, t: bid_initial(initial_charge, n, int(t)),
    )
    model.DischargeBidMW = pyo.Var(
        model.N,
        model.T,
        bounds=bid_bounds,
        initialize=lambda _, n, t: bid_initial(initial_discharge, n, int(t)),
    )

    def price_initial(
        values: Mapping[tuple[str, int], float], node: str, time: int
    ) -> float:
        value = float(values.get((node, int(time)), 0.0))
        return min(price_bound, max(-price_bound, value))

    model.BidChargeEURPerMWh = pyo.Var(
        model.N,
        model.T,
        bounds=(-price_bound, price_bound),
        initialize=lambda _, n, t: price_initial(
            initial_charge_price, n, int(t)
        ),
    )
    model.OfferDischargeEURPerMWh = pyo.Var(
        model.N,
        model.T,
        bounds=(-price_bound, price_bound),
        initialize=lambda _, n, t: price_initial(
            initial_discharge_price, n, int(t)
        ),
    )
    model.bid_price_cycle_consistency = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.OfferDischargeEURPerMWh[n, t]
        >= m.BidChargeEURPerMWh[n, t] / (data.eta**2),
    )
    if not strategic_prices:
        for node in model.N:
            for time in model.T:
                model.BidChargeEURPerMWh[node, time].fix(0.0)
                model.OfferDischargeEURPerMWh[node, time].fix(0.0)

    # Upper-level technical feasibility of the active investor's offered MW.
    # With one-hour intervals, these ensure that the entire bid could be
    # activated from the anticipated beginning-of-hour SOC.
    model.charge_bid_installed_power = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.ChargeBidMW[n, t] <= m.X_power[n],
    )
    model.discharge_bid_installed_power = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.DischargeBidMW[n, t] <= m.X_power[n],
    )
    model.charge_bid_energy_feasibility = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: data.eta * m.ChargeBidMW[n, t]
        <= m.X_energy[n] - m.SOC[active, n, t - 1],
    )
    model.discharge_bid_energy_feasibility = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.DischargeBidMW[n, t] / data.eta
        <= m.SOC[active, n, t - 1],
    )

    def available_charge(unit: str, node: str, time: int):
        return (
            model.ChargeBidMW[node, time]
            if unit == active
            else fixed_rival_charge[unit, node, int(time)]
        )

    def available_discharge(unit: str, node: str, time: int):
        return (
            model.DischargeBidMW[node, time]
            if unit == active
            else fixed_rival_discharge[unit, node, int(time)]
        )

    def charge_price(unit: str, node: str, time: int):
        return (
            model.BidChargeEURPerMWh[node, time]
            if unit == active
            else fixed_rival_charge_price[unit, node, int(time)]
        )

    def discharge_price(unit: str, node: str, time: int):
        return (
            model.OfferDischargeEURPerMWh[node, time]
            if unit == active
            else fixed_rival_discharge_price[unit, node, int(time)]
        )

    def installed_power(unit: str, node: str):
        return (
            model.X_power[node]
            if unit == active
            else model._rival_power[unit][node]
        )

    def installed_energy(unit: str, node: str):
        return (
            model.X_energy[node]
            if unit == active
            else model._rival_energy[unit][node]
        )

    # Replace full-capacity dispatch bounds by the strategic hourly bids.
    model.del_component(model.charge_power_bound)
    model.del_component(model.discharge_power_bound)
    model.charge_power_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.P_charge[i, n, t]
        <= available_charge(i, n, int(t)),
    )
    model.discharge_power_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.P_discharge[i, n, t]
        <= available_discharge(i, n, int(t)),
    )
    model.shared_inverter_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.P_charge[i, n, t]
        + m.P_discharge[i, n, t]
        <= installed_power(i, n),
    )
    model.kappa_power = pyo.Var(
        model.IN,
        model.T,
        bounds=(-float(kwargs.get("dual_bound", 10_000.0)), 0.0),
        initialize=0.0,
    )

    # Submitted prices replace the former zero storage coefficient in the ISO
    # stationarity equations. Quantities remain upper dispatch bounds.
    model.del_component(model.charge_stationarity)
    model.del_component(model.discharge_stationarity)
    model.charge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: -m.lam[n, t]
        + m.rho_ch[i, n, t]
        + m.kappa_power[i, n, t]
        - data.eta * m.gam[i, n, t]
        + charge_price(i, n, int(t))
        <= 0.0,
    )
    model.discharge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.lam[n, t]
        + m.sig_dis[i, n, t]
        + m.kappa_power[i, n, t]
        + m.gam[i, n, t] / data.eta
        <= discharge_price(i, n, int(t)),
    )

    model.lower_level_bid_cost = pyo.Expression(
        expr=sum(
            discharge_price(i, n, int(t)) * model.P_discharge[i, n, t]
            - charge_price(i, n, int(t)) * model.P_charge[i, n, t]
            for i, n in model.IN
            for t in model.T
        )
    )
    model.primal_objective.set_value(
        sum(
            data.generation_cost[g] * model.P_gen[g, t]
            for g, t in model.GT
        )
        + model.lower_level_bid_cost
    )
    model.dual_objective.set_value(
        sum(
            data.demand_el[n, t] * model.lam[n, t]
            for n in model.N
            for t in model.T
        )
        + sum(
            data.generation_capacity[g, t] * model.nu_gen[g, t]
            for g, t in model.GT
        )
        + sum(
            data.line_limit[l]
            * (model.mu_up[l, t] - model.mu_dn[l, t])
            for l in model.L
            for t in model.T
        )
        + sum(
            available_charge(i, n, int(t)) * model.rho_ch[i, n, t]
            + available_discharge(i, n, int(t))
            * model.sig_dis[i, n, t]
            + installed_power(i, n) * model.kappa_power[i, n, t]
            for i, n in model.IN
            for t in model.T
        )
        + sum(
            installed_energy(i, n) * model.del_soc[i, n, tau]
            for i, n in model.IN
            for tau in model.T_SOC
        )
    )

    eta = data.eta
    last_t = max(data.times)

    def flow(m: pyo.ConcreteModel, line: str, time: int):
        return sum(
            data.ptdf[line, node] * m.NetInjection[node, time]
            for node in m.N
        )

    def gen_reduced_cost(m: pyo.ConcreteModel, generator: str, time: int):
        return (
            data.generation_cost[generator]
            - sum(m.lam[node, time] for node in m._gen_nodes[generator])
            - m.nu_gen[generator, time]
        )

    def charge_reduced_cost(
        m: pyo.ConcreteModel, unit: str, node: str, time: int
    ):
        return (
            m.lam[node, time]
            - m.rho_ch[unit, node, time]
            - m.kappa_power[unit, node, time]
            + eta * m.gam[unit, node, time]
            - charge_price(unit, node, int(time))
        )

    def discharge_reduced_cost(
        m: pyo.ConcreteModel, unit: str, node: str, time: int
    ):
        return (
            -m.lam[node, time]
            - m.sig_dis[unit, node, time]
            - m.kappa_power[unit, node, time]
            - m.gam[unit, node, time] / eta
            + discharge_price(unit, node, int(time))
        )

    def soc_reduced_cost(
        m: pyo.ConcreteModel, unit: str, node: str, soc_time: int
    ):
        stationarity_lhs = m.del_soc[unit, node, soc_time]
        if soc_time in m.T:
            stationarity_lhs += m.gam[unit, node, soc_time]
        if soc_time + 1 in m.T:
            stationarity_lhs -= m.gam[unit, node, soc_time + 1]
        if soc_time == 0:
            stationarity_lhs += m.rho_per[unit, node]
        if soc_time == last_t:
            stationarity_lhs -= m.rho_per[unit, node]
        return -stationarity_lhs

    product_names: list[str] = []

    def add_relaxed_product(
        name: str,
        index_sets: tuple[pyo.Set, ...],
        product_rule,
    ) -> None:
        product = pyo.Expression(*index_sets, rule=product_rule)
        model.add_component(f"{name}_product", product)
        model.add_component(
            name,
            pyo.Constraint(
                *index_sets,
                rule=lambda m, *key: pyo.inequality(
                    0.0, product[key], epsilon
                ),
            ),
        )
        product_names.append(f"{name}_product")

    add_relaxed_product(
        "relaxed_comp_gen_lower",
        (model.GT,),
        lambda m, g, t: m.P_gen[g, t] * gen_reduced_cost(m, g, t),
    )

    if strategic_prices and proximal_coefficient > 0.0:
        references = {
            "charging quantity": proximal_charge_bid_mw,
            "discharging quantity": proximal_discharge_bid_mw,
            "charging price": proximal_charge_bid_eur_per_mwh,
            "discharging price": proximal_discharge_offer_eur_per_mwh,
        }
        missing = next(
            (name for name, values in references.items() if values is None),
            None,
        )
        if missing is not None:
            raise ValueError(
                f"A positive full-strategy proximal penalty requires a {missing} reference."
            )
        time_count = max(1, len(tuple(model.T)))
        model.charge_quantity_dev_pos = pyo.Var(
            model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0
        )
        model.charge_quantity_dev_neg = pyo.Var(
            model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0
        )
        model.discharge_quantity_dev_pos = pyo.Var(
            model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0
        )
        model.discharge_quantity_dev_neg = pyo.Var(
            model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0
        )
        model.charge_price_dev_pos = pyo.Var(
            model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0
        )
        model.charge_price_dev_neg = pyo.Var(
            model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0
        )
        model.discharge_price_dev_pos = pyo.Var(
            model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0
        )
        model.discharge_price_dev_neg = pyo.Var(
            model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0
        )
        model.charge_quantity_deviation = pyo.Constraint(
            model.N,
            model.T,
            rule=lambda m, n, t: m.ChargeBidMW[n, t]
            - float(proximal_charge_bid_mw[n, int(t)])
            == m.charge_quantity_dev_pos[n, t]
            - m.charge_quantity_dev_neg[n, t],
        )
        model.discharge_quantity_deviation = pyo.Constraint(
            model.N,
            model.T,
            rule=lambda m, n, t: m.DischargeBidMW[n, t]
            - float(proximal_discharge_bid_mw[n, int(t)])
            == m.discharge_quantity_dev_pos[n, t]
            - m.discharge_quantity_dev_neg[n, t],
        )
        model.charge_price_deviation = pyo.Constraint(
            model.N,
            model.T,
            rule=lambda m, n, t: m.BidChargeEURPerMWh[n, t]
            - float(proximal_charge_bid_eur_per_mwh[n, int(t)])
            == m.charge_price_dev_pos[n, t]
            - m.charge_price_dev_neg[n, t],
        )
        model.discharge_price_deviation = pyo.Constraint(
            model.N,
            model.T,
            rule=lambda m, n, t: m.OfferDischargeEURPerMWh[n, t]
            - float(proximal_discharge_offer_eur_per_mwh[n, int(t)])
            == m.discharge_price_dev_pos[n, t]
            - m.discharge_price_dev_neg[n, t],
        )
        model.bid_strategy_regularizer = pyo.Expression(
            expr=proximal_coefficient
            / (2.0 * time_count)
            * sum(
                model.charge_quantity_dev_pos[n, t]
                + model.charge_quantity_dev_neg[n, t]
                + model.discharge_quantity_dev_pos[n, t]
                + model.discharge_quantity_dev_neg[n, t]
                + (
                    model.charge_price_dev_pos[n, t]
                    + model.charge_price_dev_neg[n, t]
                    + model.discharge_price_dev_pos[n, t]
                    + model.discharge_price_dev_neg[n, t]
                )
                / price_scale
                for n in model.N
                for t in model.T
            )
        )
        capacity_regularizer = model.regularizer.expr
        model.regularizer.set_value(
            capacity_regularizer + model.bid_strategy_regularizer
        )
        model.profit.set_value(model.unregularized_profit - model.regularizer)
        model.objective.set_value(model.profit)
        model._proximal_charge_bid_mw = dict(proximal_charge_bid_mw)
        model._proximal_discharge_bid_mw = dict(proximal_discharge_bid_mw)
        model._proximal_charge_bid_eur_per_mwh = dict(
            proximal_charge_bid_eur_per_mwh
        )
        model._proximal_discharge_offer_eur_per_mwh = dict(
            proximal_discharge_offer_eur_per_mwh
        )
        model._proximal_power = dict(proximal_power)
        model._proximal_energy = dict(proximal_energy)
    add_relaxed_product(
        "relaxed_comp_charge_lower",
        (model.IN, model.T),
        lambda m, i, n, t: m.P_charge[i, n, t]
        * charge_reduced_cost(m, i, n, t),
    )
    add_relaxed_product(
        "relaxed_comp_discharge_lower",
        (model.IN, model.T),
        lambda m, i, n, t: m.P_discharge[i, n, t]
        * discharge_reduced_cost(m, i, n, t),
    )
    add_relaxed_product(
        "relaxed_comp_soc_lower",
        (model.IN, model.T_SOC),
        lambda m, i, n, tau: m.SOC[i, n, tau]
        * soc_reduced_cost(m, i, n, tau),
    )
    add_relaxed_product(
        "relaxed_comp_gen_upper",
        (model.GT,),
        lambda m, g, t: (data.generation_capacity[g, t] - m.P_gen[g, t])
        * (-m.nu_gen[g, t]),
    )
    add_relaxed_product(
        "relaxed_comp_line_upper",
        (model.L, model.T),
        lambda m, l, t: (data.line_limit[l] - flow(m, l, t))
        * (-m.mu_up[l, t]),
    )
    add_relaxed_product(
        "relaxed_comp_line_lower",
        (model.L, model.T),
        lambda m, l, t: (flow(m, l, t) + data.line_limit[l])
        * m.mu_dn[l, t],
    )
    add_relaxed_product(
        "relaxed_comp_charge_bid_upper",
        (model.IN, model.T),
        lambda m, i, n, t: (
            available_charge(i, n, int(t)) - m.P_charge[i, n, t]
        )
        * (-m.rho_ch[i, n, t]),
    )
    add_relaxed_product(
        "relaxed_comp_discharge_bid_upper",
        (model.IN, model.T),
        lambda m, i, n, t: (
            available_discharge(i, n, int(t)) - m.P_discharge[i, n, t]
        )
        * (-m.sig_dis[i, n, t]),
    )
    add_relaxed_product(
        "relaxed_comp_shared_inverter_upper",
        (model.IN, model.T),
        lambda m, i, n, t: (
            installed_power(i, n)
            - m.P_charge[i, n, t]
            - m.P_discharge[i, n, t]
        )
        * (-m.kappa_power[i, n, t]),
    )
    add_relaxed_product(
        "relaxed_comp_soc_upper",
        (model.IN, model.T_SOC),
        lambda m, i, n, tau: (
            installed_energy(i, n) - m.SOC[i, n, tau]
        )
        * (-m.del_soc[i, n, tau]),
    )

    model._lower_level_optimality = "relaxed-kkt"
    model._strategic_quantity = True
    model._strategic_prices = bool(strategic_prices)
    model._strategic_price_quantity = bool(strategic_prices)
    model._bid_price_bound = price_bound
    model._proximal_price_scale = price_scale
    model._complementarity_epsilon = epsilon
    model._rival_charge_bid_mw = fixed_rival_charge
    model._rival_discharge_bid_mw = fixed_rival_discharge
    model._rival_charge_bid_eur_per_mwh = fixed_rival_charge_price
    model._rival_discharge_offer_eur_per_mwh = fixed_rival_discharge_price
    model._relaxed_kkt_product_components = tuple(product_names)
    return model


def initialise_lower_level(model: pyo.ConcreteModel, data: MarketData) -> None:
    """Seed the MPEC from its exact fixed-bid lower-level LP.

    The initialization LP uses the current investment, quantity, and price
    values, including the shared inverter limit. If the previous Jacobi
    quantity profile is not deliverable from the LP's selected SOC trajectory,
    only the initial quantity values are reduced to the corresponding one-hour
    energy limits and the LP is resolved. This changes the NLP starting point,
    never a model bound or equation.
    """

    active = model._active_id
    units = [active, *model._rival_ids]
    x_power: dict[tuple[str, str], float] = {}
    x_energy: dict[tuple[str, str], float] = {}
    for unit in units:
        for node in data.nodes:
            if unit == active:
                x_power[unit, node] = float(pyo.value(model.X_power[node]))
                x_energy[unit, node] = float(pyo.value(model.X_energy[node]))
            else:
                x_power[unit, node] = model._rival_power[unit][node]
                x_energy[unit, node] = model._rival_energy[unit][node]

    fixed_data = replace(
        data,
        storage_units=units,
        x_power=x_power,
        x_energy=x_energy,
    )
    lower = build_primal_market_clearing_model(
        fixed_data, include_load_shed=False
    )

    def charge_bid(unit: str, node: str, time: int) -> float:
        return (
            float(pyo.value(model.ChargeBidMW[node, time]))
            if unit == active
            else model._rival_charge_bid_mw.get((unit, node, int(time)), 0.0)
        )

    def discharge_bid(unit: str, node: str, time: int) -> float:
        return (
            float(pyo.value(model.DischargeBidMW[node, time]))
            if unit == active
            else model._rival_discharge_bid_mw.get((unit, node, int(time)), 0.0)
        )

    def charge_price(unit: str, node: str, time: int) -> float:
        return (
            float(pyo.value(model.BidChargeEURPerMWh[node, time]))
            if unit == active
            else model._rival_charge_bid_eur_per_mwh.get(
                (unit, node, int(time)), 0.0
            )
        )

    def discharge_price(unit: str, node: str, time: int) -> float:
        return (
            float(pyo.value(model.OfferDischargeEURPerMWh[node, time]))
            if unit == active
            else model._rival_discharge_offer_eur_per_mwh.get(
                (unit, node, int(time)), 0.0
            )
        )

    lower.del_component(lower.charge_power_bound)
    lower.del_component(lower.discharge_power_bound)
    lower.charge_power_bound = pyo.Constraint(
        lower.I,
        lower.N,
        lower.T,
        rule=lambda m, i, n, t: m.P_charge[i, n, t]
        <= charge_bid(i, n, int(t)),
    )
    lower.discharge_power_bound = pyo.Constraint(
        lower.I,
        lower.N,
        lower.T,
        rule=lambda m, i, n, t: m.P_discharge[i, n, t]
        <= discharge_bid(i, n, int(t)),
    )
    lower.shared_inverter_bound = pyo.Constraint(
        lower.I,
        lower.N,
        lower.T,
        rule=lambda m, i, n, t: m.P_charge[i, n, t]
        + m.P_discharge[i, n, t]
        <= fixed_data.x_power[i, n],
    )
    lower.objective.set_value(
        sum(
            data.generation_cost[g] * lower.P_gen[g, t]
            for g in lower.G
            for t in lower.T
        )
        + sum(
            discharge_price(i, n, int(t)) * lower.P_discharge[i, n, t]
            - charge_price(i, n, int(t)) * lower.P_charge[i, n, t]
            for i in lower.I
            for n in lower.N
            for t in lower.T
        )
    )

    solver = pyo.SolverFactory("highs")
    result = solver.solve(lower, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return

    # A damped Jacobi profile is close to the previous best response but its
    # bids need not be deliverable from the particular (possibly degenerate)
    # SOC trajectory selected by this fixed-bid LP.  Retain as much of that
    # profile as the selected beginning-of-hour SOC permits.  The preceding LP
    # dispatch remains feasible after this reduction in the usual
    # non-simultaneous operating solution, so one repeat normally suffices.
    for _ in range(3):
        changed = False
        for node in model.N:
            installed_power = float(pyo.value(model.X_power[node]))
            installed_energy = float(pyo.value(model.X_energy[node]))
            for time in model.T:
                soc_before = float(pyo.value(lower.SOC[active, node, time - 1]))
                max_charge = min(
                    installed_power,
                    max(0.0, (installed_energy - soc_before) / data.eta),
                )
                max_discharge = min(
                    installed_power,
                    max(0.0, data.eta * soc_before),
                )
                old_charge = float(pyo.value(model.ChargeBidMW[node, time]))
                old_discharge = float(
                    pyo.value(model.DischargeBidMW[node, time])
                )
                new_charge = min(old_charge, max_charge)
                new_discharge = min(old_discharge, max_discharge)
                if old_charge - new_charge > 1.0e-9:
                    model.ChargeBidMW[node, time].set_value(new_charge)
                    lower.charge_power_bound[active, node, time].set_value(
                        lower.P_charge[active, node, time] <= new_charge
                    )
                    changed = True
                if old_discharge - new_discharge > 1.0e-9:
                    model.DischargeBidMW[node, time].set_value(new_discharge)
                    lower.discharge_power_bound[active, node, time].set_value(
                        lower.P_discharge[active, node, time] <= new_discharge
                    )
                    changed = True
        if not changed:
            break
        result = solver.solve(lower, tee=False)
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            return

    def set_seed(variable: pyo.Var, raw_value: float) -> None:
        value = float(raw_value)
        if variable.lb is not None:
            value = max(value, float(pyo.value(variable.lb)))
        if variable.ub is not None:
            value = min(value, float(pyo.value(variable.ub)))
        variable.set_value(value)

    for generator, time in model.GT:
        set_seed(
            model.P_gen[generator, time],
            pyo.value(lower.P_gen[generator, time])
        )
        set_seed(
            model.nu_gen[generator, time],
            lower.dual[lower.generation_capacity_bound[generator, time]]
        )
    for node in model.N:
        for time in model.T:
            set_seed(
                model.NetInjection[node, time],
                pyo.value(lower.NetInjection[node, time])
            )
            set_seed(
                model.lam[node, time],
                lower.dual[lower.nodal_balance[node, time]]
            )
    for time in model.T:
        set_seed(model.lam_sys[time], lower.dual[lower.system_balance[time]])
    for line in model.L:
        for time in model.T:
            set_seed(
                model.mu_up[line, time],
                lower.dual[lower.line_upper_bound[line, time]]
            )
            set_seed(
                model.mu_dn[line, time],
                lower.dual[lower.line_lower_bound[line, time]]
            )
    for unit, node in model.IN:
        for time in model.T:
            key = unit, node, time
            set_seed(model.P_charge[key], pyo.value(lower.P_charge[key]))
            set_seed(model.P_discharge[key], pyo.value(lower.P_discharge[key]))
            set_seed(model.rho_ch[key], lower.dual[lower.charge_power_bound[key]])
            set_seed(
                model.sig_dis[key],
                lower.dual[lower.discharge_power_bound[key]]
            )
            set_seed(
                model.kappa_power[key],
                lower.dual[lower.shared_inverter_bound[key]]
            )
            set_seed(model.gam[key], lower.dual[lower.soc_transition[key]])
        for soc_time in model.T_SOC:
            key = unit, node, soc_time
            set_seed(model.SOC[key], pyo.value(lower.SOC[key]))
            set_seed(
                model.del_soc[key],
                lower.dual[lower.soc_capacity_bound[key]]
            )
        set_seed(
            model.rho_per[unit, node],
            lower.dual[lower.soc_periodicity[unit, node]]
        )

    # Keep the moving-L1 auxiliary variables consistent if the warm-start LP
    # had to reduce a quantity seed to its SOC-deliverable value.
    if hasattr(model, "charge_quantity_dev_pos"):
        for node in model.N:
            power_difference = float(pyo.value(model.X_power[node])) - float(
                model._proximal_power[node]
            )
            energy_difference = float(pyo.value(model.X_energy[node])) - float(
                model._proximal_energy[node]
            )
            model.power_dev_pos[node].set_value(max(0.0, power_difference))
            model.power_dev_neg[node].set_value(max(0.0, -power_difference))
            model.energy_dev_pos[node].set_value(max(0.0, energy_difference))
            model.energy_dev_neg[node].set_value(max(0.0, -energy_difference))
        deviation_profiles = (
            (
                model.ChargeBidMW,
                model._proximal_charge_bid_mw,
                model.charge_quantity_dev_pos,
                model.charge_quantity_dev_neg,
            ),
            (
                model.DischargeBidMW,
                model._proximal_discharge_bid_mw,
                model.discharge_quantity_dev_pos,
                model.discharge_quantity_dev_neg,
            ),
            (
                model.BidChargeEURPerMWh,
                model._proximal_charge_bid_eur_per_mwh,
                model.charge_price_dev_pos,
                model.charge_price_dev_neg,
            ),
            (
                model.OfferDischargeEURPerMWh,
                model._proximal_discharge_offer_eur_per_mwh,
                model.discharge_price_dev_pos,
                model.discharge_price_dev_neg,
            ),
        )
        for strategy, reference, positive, negative in deviation_profiles:
            for node in model.N:
                for time in model.T:
                    difference = float(pyo.value(strategy[node, time])) - float(
                        reference[node, int(time)]
                    )
                    positive[node, time].set_value(max(0.0, difference))
                    negative[node, time].set_value(max(0.0, -difference))


def diagnostics(model: pyo.ConcreteModel) -> dict[str, float | int]:
    """Evaluate complementarity products and the implied duality gap."""

    products = [
        float(pyo.value(component[index]))
        for name in model._relaxed_kkt_product_components
        for component in (getattr(model, name),)
        for index in component
    ]
    epsilon = float(model._complementarity_epsilon)
    minimum = min(products, default=0.0)
    maximum = max(products, default=0.0)
    gap = float(pyo.value(model.primal_objective - model.dual_objective))
    return {
        "count": len(products),
        "epsilon": epsilon,
        "minimum_product": minimum,
        "maximum_product": maximum,
        "maximum_upper_bound_violation": max(0.0, maximum - epsilon),
        "maximum_nonnegativity_violation": max(0.0, -minimum),
        "primal_dual_gap_eur_per_day": gap,
        "absolute_primal_dual_gap_eur_per_day": abs(gap),
    }
