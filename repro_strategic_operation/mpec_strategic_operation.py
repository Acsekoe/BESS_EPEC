"""Capacity-and-price strategic-operation MPEC without quantity withholding.

The active investor selects nodal BESS MW/MWh together with hourly charging
buy-bids and discharging sell-offers. The ISO always retains access to the full
installed MW capacity: bid quantities are not upper-level decisions. Rival
capacities and two-sided prices are fixed during one best response.

For fixed prices the lower level is a linear market-clearing problem. It is
embedded through the same primal feasibility, dual feasibility, and strong
duality system as the maintained capacity-only strong-duality formulation.
"""

from __future__ import annotations

from collections.abc import Mapping

import pyomo.environ as pyo

from mpec_strong_duality import InvestorConfig, build_model as build_capacity_model
from primal_market_clearing_model import MarketData


PriceProfile = Mapping[str, Mapping[tuple[str, int], float]]


def build_model(
    data: MarketData,
    *,
    investor: InvestorConfig,
    rival_bid_charge: PriceProfile | None = None,
    rival_offer_discharge: PriceProfile | None = None,
    initial_bid_charge: Mapping[tuple[str, int], float] | None = None,
    initial_offer_discharge: Mapping[tuple[str, int], float] | None = None,
    bid_price_bound: float = 500.0,
    **kwargs: object,
) -> pyo.ConcreteModel:
    """Build one strategic-operation best response with full MW availability."""

    if bid_price_bound <= 0.0:
        raise ValueError("bid_price_bound must be positive.")
    kwargs.pop("big_m_dual", None)
    model = build_capacity_model(data, investor=investor, **kwargs)
    model.name = f"Strategic-operation MPEC [{investor.investor_id}]"

    active = investor.investor_id
    rival_ids = set(model._rival_ids)
    rival_charge_input = rival_bid_charge or {}
    rival_discharge_input = rival_offer_discharge or {}
    if set(rival_charge_input) - rival_ids or set(rival_discharge_input) - rival_ids:
        raise ValueError("Strategic price mappings contain an unknown rival investor.")

    bound = float(bid_price_bound)
    fixed_rival_charge: dict[tuple[str, str, int], float] = {}
    fixed_rival_discharge: dict[tuple[str, str, int], float] = {}
    for unit, node in model.IN:
        if unit == active:
            continue
        truthful_half_cost = min(bound, 0.5 * model._unit_degradation[unit])
        for time in model.T:
            key = node, int(time)
            charge = float(
                rival_charge_input.get(unit, {}).get(key, -truthful_half_cost)
            )
            discharge = float(
                rival_discharge_input.get(unit, {}).get(key, truthful_half_cost)
            )
            if not -bound <= charge <= bound:
                raise ValueError(
                    f"Rival charging bid outside bounds for {unit}, {node}, {time}."
                )
            if not -bound <= discharge <= bound:
                raise ValueError(
                    f"Rival discharge offer outside bounds for {unit}, {node}, {time}."
                )
            if discharge + 1e-9 < charge / (data.eta**2):
                raise ValueError(
                    "Rival prices permit a negative-cost same-hour storage loop for "
                    f"{unit}, {node}, {time}."
                )
            fixed_rival_charge[unit, node, int(time)] = charge
            fixed_rival_discharge[unit, node, int(time)] = discharge

    truthful_active_half_cost = 0.5 * investor.degradation_eur_per_mwh
    initial_charge = initial_bid_charge or {}
    initial_discharge = initial_offer_discharge or {}
    model.BidCharge = pyo.Var(
        model.N,
        model.T,
        bounds=(-bound, bound),
        initialize=lambda _, n, t: min(
            bound,
            max(-bound, float(initial_charge.get((n, int(t)), -truthful_active_half_cost))),
        ),
    )
    model.OfferDischarge = pyo.Var(
        model.N,
        model.T,
        bounds=(-bound, bound),
        initialize=lambda _, n, t: min(
            bound,
            max(-bound, float(initial_discharge.get((n, int(t)), truthful_active_half_cost))),
        ),
    )
    model.bid_cycle_consistency = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.OfferDischarge[n, t]
        >= m.BidCharge[n, t] / (data.eta**2),
    )

    def charge_bid(unit: str, node: str, time: int):
        return (
            model.BidCharge[node, time]
            if unit == active
            else fixed_rival_charge[unit, node, int(time)]
        )

    def discharge_offer(unit: str, node: str, time: int):
        return (
            model.OfferDischarge[node, time]
            if unit == active
            else fixed_rival_discharge[unit, node, int(time)]
        )

    model.del_component(model.charge_stationarity)
    model.del_component(model.discharge_stationarity)
    model.charge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: -m.lam[n, t]
        + m.rho_ch[i, n, t]
        - data.eta * m.gam[i, n, t]
        <= -charge_bid(i, n, int(t)),
    )
    model.discharge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.lam[n, t]
        + m.sig_dis[i, n, t]
        + m.gam[i, n, t] / data.eta
        <= discharge_offer(i, n, int(t)),
    )

    model.lower_level_bid_cost = pyo.Expression(
        expr=sum(
            discharge_offer(unit, node, int(time))
            * model.P_discharge[unit, node, time]
            - charge_bid(unit, node, int(time))
            * model.P_charge[unit, node, time]
            for unit, node in model.IN
            for time in model.T
        )
    )
    model.primal_objective.set_value(
        model.primal_objective.expr
        - model.lower_level_degradation
        + model.lower_level_bid_cost
    )
    model.del_component(model.strong_duality)
    model.strong_duality = pyo.Constraint(
        expr=model.primal_objective == model.dual_objective
    )

    model._strategic_operation = True
    model._bid_price_bound = bound
    model._rival_bid_charge = fixed_rival_charge
    model._rival_offer_discharge = fixed_rival_discharge
    return model
