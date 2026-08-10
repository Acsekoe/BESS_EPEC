"""Capacity-only single-investor MPEC using KKT conditions and binary Big-M."""

from __future__ import annotations

from collections.abc import Callable

import pyomo.environ as pyo

from mpec_strong_duality import InvestorConfig, build_model as build_strong_model
from primal_market_clearing_model import MarketData


def _flow(model: pyo.ConcreteModel, data: MarketData, line: str, time: int):
    return sum(data.ptdf[line, n] * model.NetInjection[n, time] for n in model.N)


def _add_complementarity(
    model: pyo.ConcreteModel,
    name: str,
    index: list[tuple],
    slack: Callable,
    dual_magnitude: Callable,
    maximum_slack: Callable,
    big_m_dual: float,
) -> None:
    """Encode ``0 <= slack _|_ dual_magnitude >= 0`` with one binary."""

    binary = pyo.Var(index, domain=pyo.Binary)
    model.add_component(f"{name}_binary", binary)
    model.add_component(
        f"{name}_slack_bigm",
        pyo.Constraint(
            index,
            rule=lambda mm, *key: slack(mm, *key)
            <= maximum_slack(*key) * binary[key],
        ),
    )
    model.add_component(
        f"{name}_dual_bigm",
        pyo.Constraint(
            index,
            rule=lambda mm, *key: dual_magnitude(mm, *key)
            <= big_m_dual * (1.0 - binary[key]),
        ),
    )


def build_model(
    data: MarketData,
    *,
    investor: InvestorConfig,
    big_m_dual: float = 800.0,
    price_bound: float = 500.0,
    **kwargs: object,
) -> pyo.ConcreteModel:
    """Build the exact KKT/Big-M MILP for one investor and fixed rivals.

    Investment remains continuous.  Storage revenue is replaced by an exact
    system-payment identity; fixed rivals are removed through their KKT
    scarcity rents.  Generator portfolio rent is ``-capacity * nu_gen``.
    """

    if big_m_dual <= 0.0:
        raise ValueError("big_m_dual must be positive.")
    if price_bound > big_m_dual:
        raise ValueError("big_m_dual must be at least as large as price_bound.")

    kwargs.pop("dual_bound", None)
    m = build_strong_model(
        data,
        investor=investor,
        price_bound=price_bound,
        dual_bound=big_m_dual,
        **kwargs,
    )
    m.strong_duality.deactivate()
    m.objective.deactivate()

    active = m._active_id
    eta = data.eta
    last_t = max(data.times)
    cap = data.generation_capacity
    line_limit = data.line_limit
    generation_pairs = list(m.GT)
    line_hours = [(l, t) for l in m.L for t in m.T]
    storage_hours = [(i, n, t) for i, n in m.IN for t in m.T]
    storage_states = [(i, n, tau) for i, n in m.IN for tau in m.T_SOC]

    def power_limit(i: str, n: str) -> float:
        return m._headroom[n] if i == active else m._rival_power[i][n]

    def energy_limit(i: str, n: str) -> float:
        return (
            investor.ratio_max * m._headroom[n]
            if i == active
            else m._rival_energy[i][n]
        )

    def reduced_cost_gen(mm: pyo.ConcreteModel, g: str, t: int):
        return data.generation_cost[g] - sum(mm.lam[n, t] for n in m._gen_nodes[g]) - mm.nu_gen[g, t]

    def reduced_cost_charge(mm: pyo.ConcreteModel, i: str, n: str, t: int):
        return 0.5 * m._unit_degradation[i] + mm.lam[n, t] - mm.rho_ch[i, n, t] + eta * mm.gam[i, n, t]

    def reduced_cost_discharge(mm: pyo.ConcreteModel, i: str, n: str, t: int):
        return 0.5 * m._unit_degradation[i] - mm.lam[n, t] - mm.sig_dis[i, n, t] - mm.gam[i, n, t] / eta

    def reduced_cost_soc(mm: pyo.ConcreteModel, i: str, n: str, tau: int):
        expression = mm.del_soc[i, n, tau]
        if tau in mm.T:
            expression += mm.gam[i, n, tau]
        if tau + 1 in mm.T:
            expression -= mm.gam[i, n, tau + 1]
        if tau == 0:
            expression += mm.rho_per[i, n]
        if tau == last_t:
            expression -= mm.rho_per[i, n]
        return -expression

    _add_complementarity(
        m, "gen_variable", generation_pairs,
        lambda mm, g, t: mm.P_gen[g, t], reduced_cost_gen,
        lambda g, t: cap[g, t], big_m_dual,
    )
    _add_complementarity(
        m, "charge_variable", storage_hours,
        lambda mm, i, n, t: mm.P_charge[i, n, t], reduced_cost_charge,
        lambda i, n, t: power_limit(i, n), big_m_dual,
    )
    _add_complementarity(
        m, "discharge_variable", storage_hours,
        lambda mm, i, n, t: mm.P_discharge[i, n, t], reduced_cost_discharge,
        lambda i, n, t: power_limit(i, n), big_m_dual,
    )
    _add_complementarity(
        m, "soc_variable", storage_states,
        lambda mm, i, n, tau: mm.SOC[i, n, tau], reduced_cost_soc,
        lambda i, n, tau: energy_limit(i, n), big_m_dual,
    )

    _add_complementarity(
        m, "generation_capacity", generation_pairs,
        lambda mm, g, t: cap[g, t] - mm.P_gen[g, t],
        lambda mm, g, t: -mm.nu_gen[g, t],
        lambda g, t: cap[g, t], big_m_dual,
    )
    _add_complementarity(
        m, "line_upper", line_hours,
        lambda mm, l, t: line_limit[l] - _flow(mm, data, l, t),
        lambda mm, l, t: -mm.mu_up[l, t],
        lambda l, t: 2.0 * line_limit[l], big_m_dual,
    )
    _add_complementarity(
        m, "line_lower", line_hours,
        lambda mm, l, t: _flow(mm, data, l, t) + line_limit[l],
        lambda mm, l, t: mm.mu_dn[l, t],
        lambda l, t: 2.0 * line_limit[l], big_m_dual,
    )
    _add_complementarity(
        m, "charge_capacity", storage_hours,
        lambda mm, i, n, t: (mm.X_power[n] if i == active else m._rival_power[i][n])
        - mm.P_charge[i, n, t],
        lambda mm, i, n, t: -mm.rho_ch[i, n, t],
        lambda i, n, t: power_limit(i, n), big_m_dual,
    )
    _add_complementarity(
        m, "discharge_capacity", storage_hours,
        lambda mm, i, n, t: (mm.X_power[n] if i == active else m._rival_power[i][n])
        - mm.P_discharge[i, n, t],
        lambda mm, i, n, t: -mm.sig_dis[i, n, t],
        lambda i, n, t: power_limit(i, n), big_m_dual,
    )
    _add_complementarity(
        m, "soc_capacity", storage_states,
        lambda mm, i, n, tau: (mm.X_energy[n] if i == active else m._rival_energy[i][n])
        - mm.SOC[i, n, tau],
        lambda mm, i, n, tau: -mm.del_soc[i, n, tau],
        lambda i, n, tau: energy_limit(i, n), big_m_dual,
    )

    m.total_storage_revenue_linear = pyo.Expression(
        expr=sum(line_limit[l] * (m.mu_up[l, t] - m.mu_dn[l, t]) for l in m.L for t in m.T)
        - sum(data.generation_cost[g] * m.P_gen[g, t] for g, t in m.GT)
        + sum(cap[g, t] * m.nu_gen[g, t] for g, t in m.GT)
        + sum(data.demand_el[n, t] * m.lam[n, t] for n in m.N for t in m.T)
    )
    m.rival_operating_profit_linear = pyo.Expression(
        expr=-sum(
            m._rival_power[i][n] * (m.rho_ch[i, n, t] + m.sig_dis[i, n, t])
            for i, n in m.IN
            if i != active
            for t in m.T
        )
        - sum(
            m._rival_energy[i][n] * m.del_soc[i, n, tau]
            for i, n in m.IN
            if i != active
            for tau in m.T_SOC
        )
    )
    m.active_operating_profit_linear = pyo.Expression(
        expr=m.total_storage_revenue_linear
        - m.lower_level_degradation
        - m.rival_operating_profit_linear
    )
    m.generation_rent_linear = pyo.Expression(
        expr=-sum(
            share * cap[g, t] * m.nu_gen[g, t]
            for g, share in investor.owned_generation_shares.items()
            for t in m.T
            if share and (g, t) in m.GT
        )
    )
    m.kkt_unregularized_profit = pyo.Expression(
        expr=m.active_operating_profit_linear + m.generation_rent_linear - m.daily_capex
    )
    m.kkt_profit = pyo.Expression(expr=m.kkt_unregularized_profit - m.regularizer)
    m.kkt_objective = pyo.Objective(expr=m.kkt_profit, sense=pyo.maximize)
    m._big_m_dual = big_m_dual
    return m
