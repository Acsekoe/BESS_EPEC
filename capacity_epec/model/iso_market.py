"""The ISO's day-ahead market for a *fixed* set of storage capacities.

The ISO minimises system cost — generation offers plus storage degradation
plus a small quadratic demand-adjustment term that pins down a unique price —
subject to nodal balance, PTDF line limits, generation limits and the storage
power, energy and cyclic-SOC constraints.

The problem is a convex QP, so its duals are well defined: the dual of nodal
balance is the LMP, and those LMPs are what settle every investor.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyomo.environ as pyo
import afrr

from investors import InvestorConfig, daily_investment_cost
from market_data import MarketData
from solvers import SolverSettings, solve_market_qp


Capacities = dict[tuple[str, str], float]


def build_market(
    data: MarketData,
    power: Capacities,
    energy: Capacities,
    degradation: dict[str, float],
) -> pyo.ConcreteModel:
    """Build the ISO cost-minimisation QP for one fixed capacity profile.

    ``power`` and ``energy`` are keyed by ``(investor_id, node)``; the storage
    units of the market are exactly the keys of ``degradation``.
    """

    units = list(degradation)
    missing = [
        key
        for unit in units
        for node in data.nodes
        for key in ((unit, node),)
        if key not in power or key not in energy
    ]
    if missing:
        raise ValueError(f"Missing capacities for {missing[:5]}.")

    m = pyo.ConcreteModel(name="ISO market clearing")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.G = pyo.Set(initialize=data.generators, ordered=True)
    m.I = pyo.Set(initialize=units, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)

    m.P_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    m.P_charge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals)
    m.P_discharge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals)
    m.SOC = pyo.Var(m.I, m.N, m.T_SOC, domain=pyo.NonNegativeReals)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals)
    # Elastic demand term: it prices the balance constraint uniquely without
    # giving anyone a strategic quantity to withhold.  It is a *demand-side*
    # device, so it is held at zero wherever there is no demand to adjust --
    # see MarketData.demand_is_adjustable.
    m.DemandAdjustment = pyo.Var(m.N, m.T, domain=pyo.Reals)
    for n in m.N:
        for t in m.T:
            if not data.demand_is_adjustable(n, t):
                m.DemandAdjustment[n, t].fix(0.0)

    m.objective = pyo.Objective(
        expr=sum(data.offer(g) * m.P_gen[g, t] for g in m.G for t in m.T)
        + sum(
            0.5 * degradation[i] * (m.P_charge[i, n, t] + m.P_discharge[i, n, t])
            for i in m.I
            for n in m.N
            for t in m.T
        )
        + 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(m.DemandAdjustment[n, t] ** 2 for n in m.N for t in m.T),
        sense=pyo.minimize,
    )

    # The dual of this constraint is the LMP.
    m.nodal_balance = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda mm, n, t: sum(mm.P_gen[g, t] for g in data.generators_at_node.get(n, []))
        + sum(mm.P_discharge[i, n, t] - mm.P_charge[i, n, t] for i in mm.I)
        + mm.DemandAdjustment[n, t]
        - data.demand_el[n, t]
        == mm.NetInjection[n, t],
    )
    m.system_balance = pyo.Constraint(
        m.T, rule=lambda mm, t: sum(mm.NetInjection[n, t] for n in mm.N) == 0.0
    )
    m.generation_capacity_bound = pyo.Constraint(
        m.G, m.T, rule=lambda mm, g, t: mm.P_gen[g, t] <= data.generation_capacity[g, t]
    )

    def flow(mm: pyo.ConcreteModel, line: str, t: int):
        return sum(data.ptdf[line, n] * mm.NetInjection[n, t] for n in mm.N)

    m.line_upper_bound = pyo.Constraint(
        m.L, m.T, rule=lambda mm, l, t: flow(mm, l, t) <= data.line_limit[l]
    )
    m.line_lower_bound = pyo.Constraint(
        m.L, m.T, rule=lambda mm, l, t: flow(mm, l, t) >= -data.line_limit[l]
    )
    m.charge_power_bound = pyo.Constraint(
        m.I, m.N, m.T, rule=lambda mm, i, n, t: mm.P_charge[i, n, t] <= power[i, n]
    )
    m.discharge_power_bound = pyo.Constraint(
        m.I, m.N, m.T, rule=lambda mm, i, n, t: mm.P_discharge[i, n, t] <= power[i, n]
    )
    m.soc_transition = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda mm, i, n, t: mm.SOC[i, n, t]
        == mm.SOC[i, n, t - 1]
        + data.eta * mm.P_charge[i, n, t]
        - mm.P_discharge[i, n, t] / data.eta,
    )
    m.soc_capacity_bound = pyo.Constraint(
        m.I, m.N, m.T_SOC, rule=lambda mm, i, n, tau: mm.SOC[i, n, tau] <= energy[i, n]
    )
    m.soc_periodicity = pyo.Constraint(
        m.I,
        m.N,
        rule=lambda mm, i, n: mm.SOC[i, n, 0] == mm.SOC[i, n, data.last_time],
    )
    if data.afrr_enabled:
        afrr.add_primal(m, data, [(i, n) for i in units for n in data.nodes],
                        lambda mm, i, n: power[i, n], lambda mm, i, n: energy[i, n], degradation)
        m.objective.set_value(m.objective.expr + m.afrr_total_cost)
    return m


def clear_market(
    data: MarketData,
    settings: SolverSettings,
    power: Capacities,
    energy: Capacities,
    degradation: dict[str, float],
) -> pyo.ConcreteModel:
    """Build and solve the market; raises if it does not clear."""

    market = build_market(data, power, energy, degradation)
    solve_market_qp(market, settings)
    return market


def lmp(market: pyo.ConcreteModel, node: str, time: int) -> float:
    """Locational marginal price: the dual of nodal balance."""

    return float(market.dual[market.nodal_balance[node, time]])


@dataclass(frozen=True)
class Settlement:
    """One investor's daily profit at cleared prices, decomposed."""

    investor: str
    total_power_mw: float
    total_energy_mwh: float
    storage_settlement: float
    degradation: float
    storage_operating_surplus: float
    owned_generation_rent: float
    capex: float
    profit: float
    afrr_capacity_revenue: float = 0.0
    afrr_expected_degradation: float = 0.0
    owned_generation_afrr_surplus: float = 0.0


def settle(
    market: pyo.ConcreteModel,
    data: MarketData,
    investor: InvestorConfig,
    power: Capacities,
    energy: Capacities,
) -> Settlement:
    """Value one investor's position at the cleared LMPs.

    Storage is settled at its own node; owned renewable generation earns the
    rent above its true economic cost, which is what makes an investor care
    about prices it does not itself set.
    """

    unit = investor.investor_id
    generator_node = {
        generator: nodes[0]
        for generator, nodes in data.nodes_by_generator().items()
        if nodes
    }
    storage_settlement = sum(
        lmp(market, node, time)
        * (
            float(pyo.value(market.P_discharge[unit, node, time]))
            - float(pyo.value(market.P_charge[unit, node, time]))
        )
        for node in data.nodes
        for time in data.times
    )
    generation_rent = sum(
        share
        * (lmp(market, generator_node[generator], time) - data.generation_cost[generator])
        * float(pyo.value(market.P_gen[generator, time]))
        for generator, share in investor.owned_generation_shares.items()
        for time in data.times
    )
    degradation = (
        0.5
        * investor.degradation_eur_per_mwh
        * sum(
            float(pyo.value(market.P_charge[unit, node, time]))
            + float(pyo.value(market.P_discharge[unit, node, time]))
            for node in data.nodes
            for time in data.times
        )
    )
    capex = float(
        daily_investment_cost(
            investor,
            (power[unit, node] for node in data.nodes),
            (energy[unit, node] for node in data.nodes),
        )
    )
    reserve_revenue = afrr.reserve_revenue(market, data, unit) if data.afrr_enabled else 0.0
    reserve_wear = (sum(float(pyo.value(market.afrr_expected_degradation[unit, n]))
                       for n in data.nodes) if data.afrr_enabled else 0.0)
    generator_reserve = (afrr.owned_reserve_surplus(market, data, investor.owned_generation_shares)
                         if data.afrr_enabled else 0.0)
    degradation += reserve_wear
    return Settlement(
        investor=unit,
        total_power_mw=sum(power[unit, node] for node in data.nodes),
        total_energy_mwh=sum(energy[unit, node] for node in data.nodes),
        storage_settlement=storage_settlement,
        degradation=degradation,
        storage_operating_surplus=storage_settlement + reserve_revenue - degradation,
        owned_generation_rent=generation_rent,
        capex=capex,
        profit=storage_settlement + reserve_revenue + generation_rent + generator_reserve - degradation - capex,
        afrr_capacity_revenue=reserve_revenue,
        afrr_expected_degradation=reserve_wear,
        owned_generation_afrr_surplus=generator_reserve,
    )
