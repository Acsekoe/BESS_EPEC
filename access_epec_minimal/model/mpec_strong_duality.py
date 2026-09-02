"""Capacity-only single-investor MPEC using exact lower-level strong duality.

One investor chooses nodal BESS power and energy capacity while all rival
capacities are fixed. A small quadratic demand-adjustment term makes the LMP
selection unique without introducing another strategic variable.
"""

from __future__ import annotations

from collections.abc import Mapping

import pyomo.environ as pyo

from investors import InvestorConfig, capital_recovery_factor
from primal_market_clearing_model import MarketData, effective_generation_offer


def _generator_nodes(data: MarketData) -> dict[str, list[str]]:
    result = {g: [] for g in data.generators}
    for n in data.nodes:
        for g in data.generators_at_node.get(n, []):
            result[g].append(n)
    return result


def _normalise_rivals(
    data: MarketData,
    active_id: str,
    rival_power: Mapping[str, Mapping[str, float]] | None,
    rival_energy: Mapping[str, Mapping[str, float]] | None,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    power = rival_power or {}
    energy = rival_energy or {}
    if set(power) != set(energy):
        raise ValueError("Rival power and energy identifiers must match.")
    if active_id in power:
        raise ValueError(f"Active investor {active_id} cannot also be a rival.")
    ids = list(power)
    normal_power = {
        i: {n: float(power[i].get(n, 0.0)) for n in data.nodes} for i in ids
    }
    normal_energy = {
        i: {n: float(energy[i].get(n, 0.0)) for n in data.nodes} for i in ids
    }
    for i in ids:
        for n in data.nodes:
            if normal_power[i][n] < 0.0 or normal_energy[i][n] < 0.0:
                raise ValueError(f"Negative rival capacity for {i} at {n}.")
            if normal_power[i][n] == 0.0 and normal_energy[i][n] > 1e-9:
                raise ValueError(f"Rival {i} has energy but no power at {n}.")
    return ids, normal_power, normal_energy


def build_model(
    data: MarketData,
    *,
    investor: InvestorConfig,
    rival_power: Mapping[str, Mapping[str, float]] | None = None,
    rival_energy: Mapping[str, Mapping[str, float]] | None = None,
    node_limit_mw: float = 1_000.0,
    initial_power_mw: float = 0.0,
    initial_ratio_hours: float = 2.0,
    price_bound: float = 500.0,
    dual_bound: float = 10_000.0,
    rival_degradation_eur_per_mwh: float = 15.0,
    sparse_capacity_tol: float = 1e-8,
    proximal_power: Mapping[str, float] | None = None,
    proximal_energy: Mapping[str, float] | None = None,
    proximal_penalty: float = 0.0,
    proximal_energy_scale: float = 2.0,
    **_: object,
) -> pyo.ConcreteModel:
    """Build the capacity-choice MPEC for one investor and fixed rivals."""

    if node_limit_mw <= 0.0 or price_bound <= 0.0 or dual_bound <= 0.0:
        raise ValueError("Capacity and dual bounds must be positive.")
    if not 0.0 < data.eta <= 1.0:
        raise ValueError("Storage efficiency must be in (0, 1].")
    if not 0.0 <= investor.wacc or investor.lifetime_years <= 0:
        raise ValueError("Invalid investor financing parameters.")
    if not 0.0 <= investor.ratio_min <= investor.ratio_max:
        raise ValueError("Invalid energy-to-power ratio bounds.")
    if proximal_penalty < 0.0 or proximal_energy_scale <= 0.0:
        raise ValueError("Invalid proximal regularizer settings.")
    unknown_owned = set(investor.owned_generation_shares) - set(data.generators)
    if unknown_owned:
        raise ValueError(f"Unknown owned generators: {sorted(unknown_owned)}")
    if any(not 0.0 <= share <= 1.0 for share in investor.owned_generation_shares.values()):
        raise ValueError("Generator ownership shares must be in [0, 1].")

    rivals, rival_p, rival_e = _normalise_rivals(
        data, investor.investor_id, rival_power, rival_energy
    )
    aggregate_rival = {
        n: sum(rival_p[i][n] for i in rivals) for n in data.nodes
    }
    if any(value > node_limit_mw + 1e-9 for value in aggregate_rival.values()):
        raise ValueError("Rival capacity exceeds a nodal connection limit.")
    headroom = {n: max(0.0, node_limit_mw - aggregate_rival[n]) for n in data.nodes}

    active = investor.investor_id
    storage_pairs = [(active, n) for n in data.nodes]
    storage_pairs += [
        (i, n)
        for i in rivals
        for n in data.nodes
        if rival_p[i][n] > sparse_capacity_tol
    ]
    storage_at_node = {
        n: [i for i, node in storage_pairs if node == n] for n in data.nodes
    }
    degradation = {active: investor.degradation_eur_per_mwh}
    degradation.update({i: rival_degradation_eur_per_mwh for i in rivals})

    gen_nodes = _generator_nodes(data)
    generation_pairs = [
        (g, t)
        for g in data.generators
        for t in data.times
        if data.generation_capacity[g, t] > 1e-8
    ]
    generators_at_node_time = {
        (n, t): [g for g in data.generators_at_node.get(n, []) if (g, t) in generation_pairs]
        for n in data.nodes
        for t in data.times
    }
    last_t = max(data.times)
    eta = data.eta
    crf_daily = capital_recovery_factor(investor.wacc, investor.lifetime_years) / 365.25

    m = pyo.ConcreteModel(name=f"Strong-duality MPEC [{active}]")
    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.GT = pyo.Set(dimen=2, initialize=generation_pairs, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.IN = pyo.Set(dimen=2, initialize=storage_pairs, ordered=True)

    initial_ratio = min(investor.ratio_max, max(investor.ratio_min, initial_ratio_hours))
    m.X_power = pyo.Var(
        m.N,
        domain=pyo.NonNegativeReals,
        initialize=lambda _, n: min(max(0.0, initial_power_mw), headroom[n]),
    )
    m.X_energy = pyo.Var(
        m.N,
        domain=pyo.NonNegativeReals,
        initialize=lambda _, n: initial_ratio * min(max(0.0, initial_power_mw), headroom[n]),
    )
    m.investment_headroom = pyo.Constraint(m.N, rule=lambda mm, n: mm.X_power[n] <= headroom[n])
    m.energy_ratio_min = pyo.Constraint(
        m.N, rule=lambda mm, n: mm.X_energy[n] >= investor.ratio_min * mm.X_power[n]
    )
    m.energy_ratio_max = pyo.Constraint(
        m.N, rule=lambda mm, n: mm.X_energy[n] <= investor.ratio_max * mm.X_power[n]
    )

    m.P_gen = pyo.Var(m.GT, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_charge = pyo.Var(m.IN, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_discharge = pyo.Var(m.IN, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.SOC = pyo.Var(m.IN, m.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)
    m.DemandAdjustment = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)

    m.lam = pyo.Var(m.N, m.T, bounds=(-price_bound, price_bound), initialize=80.0)
    m.lam_sys = pyo.Var(m.T, bounds=(-price_bound, price_bound), initialize=80.0)
    m.nu_gen = pyo.Var(m.GT, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_up = pyo.Var(m.L, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_dn = pyo.Var(m.L, m.T, bounds=(0.0, dual_bound), initialize=0.0)
    m.rho_ch = pyo.Var(m.IN, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.sig_dis = pyo.Var(m.IN, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.gam = pyo.Var(m.IN, m.T, bounds=(-dual_bound, dual_bound), initialize=0.0)
    m.del_soc = pyo.Var(m.IN, m.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.rho_per = pyo.Var(m.IN, bounds=(-dual_bound, dual_bound), initialize=0.0)

    def unit_power(mm: pyo.ConcreteModel, i: str, n: str):
        return mm.X_power[n] if i == active else rival_p[i][n]

    def unit_energy(mm: pyo.ConcreteModel, i: str, n: str):
        return mm.X_energy[n] if i == active else rival_e[i][n]

    def nodal_balance(mm: pyo.ConcreteModel, n: str, t: int):
        return (
            sum(mm.P_gen[g, t] for g in generators_at_node_time[n, t])
            + sum(mm.P_discharge[i, n, t] - mm.P_charge[i, n, t] for i in storage_at_node[n])
            + mm.DemandAdjustment[n, t]
            - data.demand_el[n, t]
            == mm.NetInjection[n, t]
        )

    m.nodal_balance = pyo.Constraint(m.N, m.T, rule=nodal_balance)
    m.system_balance = pyo.Constraint(
        m.T, rule=lambda mm, t: sum(mm.NetInjection[n, t] for n in mm.N) == 0.0
    )
    m.generation_capacity_bound = pyo.Constraint(
        m.GT, rule=lambda mm, g, t: mm.P_gen[g, t] <= data.generation_capacity[g, t]
    )
    m.line_upper_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda mm, l, t: sum(data.ptdf[l, n] * mm.NetInjection[n, t] for n in mm.N)
        <= data.line_limit[l],
    )
    m.line_lower_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda mm, l, t: sum(data.ptdf[l, n] * mm.NetInjection[n, t] for n in mm.N)
        >= -data.line_limit[l],
    )
    m.charge_power_bound = pyo.Constraint(
        m.IN, m.T, rule=lambda mm, i, n, t: mm.P_charge[i, n, t] <= unit_power(mm, i, n)
    )
    m.discharge_power_bound = pyo.Constraint(
        m.IN, m.T, rule=lambda mm, i, n, t: mm.P_discharge[i, n, t] <= unit_power(mm, i, n)
    )
    m.soc_transition = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda mm, i, n, t: mm.SOC[i, n, t]
        == mm.SOC[i, n, t - 1] + eta * mm.P_charge[i, n, t] - mm.P_discharge[i, n, t] / eta,
    )
    m.soc_capacity_bound = pyo.Constraint(
        m.IN, m.T_SOC, rule=lambda mm, i, n, tau: mm.SOC[i, n, tau] <= unit_energy(mm, i, n)
    )
    m.soc_periodicity = pyo.Constraint(
        m.IN, rule=lambda mm, i, n: mm.SOC[i, n, 0] == mm.SOC[i, n, last_t]
    )

    m.gen_stationarity = pyo.Constraint(
        m.GT,
        rule=lambda mm, g, t: sum(mm.lam[n, t] for n in gen_nodes[g]) + mm.nu_gen[g, t]
        <= effective_generation_offer(data, g),
    )
    m.charge_stationarity = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda mm, i, n, t: -mm.lam[n, t] + mm.rho_ch[i, n, t] - eta * mm.gam[i, n, t]
        <= 0.5 * degradation[i],
    )
    m.discharge_stationarity = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda mm, i, n, t: mm.lam[n, t] + mm.sig_dis[i, n, t] + mm.gam[i, n, t] / eta
        <= 0.5 * degradation[i],
    )
    m.netinjection_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda mm, n, t: -mm.lam[n, t]
        + mm.lam_sys[t]
        + sum(data.ptdf[l, n] * (mm.mu_up[l, t] + mm.mu_dn[l, t]) for l in mm.L)
        == 0.0,
    )
    m.demand_adjustment_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda mm, n, t: data.demand_adjustment_penalty_eur_per_mw2
        * mm.DemandAdjustment[n, t]
        == mm.lam[n, t],
    )

    def soc_stationarity(mm: pyo.ConcreteModel, i: str, n: str, tau: int):
        expression = mm.del_soc[i, n, tau]
        if tau in mm.T:
            expression += mm.gam[i, n, tau]
        if tau + 1 in mm.T:
            expression -= mm.gam[i, n, tau + 1]
        if tau == 0:
            expression += mm.rho_per[i, n]
        if tau == last_t:
            expression -= mm.rho_per[i, n]
        return expression <= 0.0

    m.soc_stationarity = pyo.Constraint(m.IN, m.T_SOC, rule=soc_stationarity)

    m.lower_level_degradation = pyo.Expression(
        expr=sum(
            0.5 * degradation[i] * (m.P_charge[i, n, t] + m.P_discharge[i, n, t])
            for i, n in m.IN
            for t in m.T
        )
    )
    m.primal_objective = pyo.Expression(
        expr=sum(effective_generation_offer(data, g) * m.P_gen[g, t] for g, t in m.GT)
        + m.lower_level_degradation
        + 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(m.DemandAdjustment[n, t] ** 2 for n in m.N for t in m.T)
    )
    m.dual_objective = pyo.Expression(
        expr=sum(data.demand_el[n, t] * m.lam[n, t] for n in m.N for t in m.T)
        + sum(data.generation_capacity[g, t] * m.nu_gen[g, t] for g, t in m.GT)
        + sum(data.line_limit[l] * (m.mu_up[l, t] - m.mu_dn[l, t]) for l in m.L for t in m.T)
        + sum(unit_power(m, i, n) * (m.rho_ch[i, n, t] + m.sig_dis[i, n, t]) for i, n in m.IN for t in m.T)
        + sum(unit_energy(m, i, n) * m.del_soc[i, n, tau] for i, n in m.IN for tau in m.T_SOC)
        - 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(m.DemandAdjustment[n, t] ** 2 for n in m.N for t in m.T)
    )
    m.strong_duality = pyo.Constraint(expr=m.primal_objective == m.dual_objective)

    m.spot_revenue = pyo.Expression(
        expr=sum(
            m.lam[n, t] * (m.P_discharge[active, n, t] - m.P_charge[active, n, t])
            for n in m.N
            for t in m.T
        )
    )
    m.generation_rent = pyo.Expression(
        expr=sum(
            share * (m.lam[gen_nodes[g][0], t] - data.generation_cost[g]) * m.P_gen[g, t]
            for g, share in investor.owned_generation_shares.items()
            for t in m.T
            if share and (g, t) in m.GT
        )
    )
    m.active_degradation = pyo.Expression(
        expr=0.5
        * investor.degradation_eur_per_mwh
        * sum(m.P_charge[active, n, t] + m.P_discharge[active, n, t] for n in m.N for t in m.T)
    )
    m.daily_capex = pyo.Expression(
        expr=crf_daily
        * sum(
            investor.cost_power_eur_per_mw * m.X_power[n]
            + investor.cost_energy_eur_per_mwh * m.X_energy[n]
            for n in m.N
        )
    )
    m.unregularized_profit = pyo.Expression(
        expr=m.spot_revenue + m.generation_rent - m.active_degradation - m.daily_capex
    )

    if proximal_penalty > 0.0:
        if proximal_power is None or proximal_energy is None:
            raise ValueError("A positive proximal penalty requires both capacity centres.")
        # Moving quadratic proximal term.  Energy deviations are expressed in
        # equivalent MW using the configured duration scale so that the MW and
        # MWh decisions enter the regularizer at comparable magnitudes.
        regularizer = 0.5 * proximal_penalty * sum(
            (m.X_power[n] - float(proximal_power[n])) ** 2
            + (
                (m.X_energy[n] - float(proximal_energy[n]))
                / proximal_energy_scale
            )
            ** 2
            for n in m.N
        )
    else:
        regularizer = 0.0
    m.regularizer = pyo.Expression(expr=regularizer)
    m.profit = pyo.Expression(expr=m.unregularized_profit - m.regularizer)
    m.objective = pyo.Objective(expr=m.profit, sense=pyo.maximize)

    m._market_data = data
    m._investor = investor
    m._active_id = active
    m._rival_ids = tuple(rivals)
    m._rival_power = rival_p
    m._rival_energy = rival_e
    m._headroom = headroom
    m._unit_degradation = degradation
    m._gen_nodes = gen_nodes
    m._node_limit_mw = node_limit_mw
    m._lower_level_optimality = "strong-duality"
    return m


def primal_dual_gap(model: pyo.ConcreteModel) -> float:
    return abs(pyo.value(model.primal_objective) - pyo.value(model.dual_objective))


def diagnostics(model: pyo.ConcreteModel) -> dict[str, float | int]:
    """Evaluate the individual KKT products of a solved exact MPEC."""

    data = model._market_data
    active = model._active_id
    eta = data.eta
    last_t = max(data.times)

    def unit_power(unit: str, node: str):
        return model.X_power[node] if unit == active else model._rival_power[unit][node]

    def unit_energy(unit: str, node: str):
        return model.X_energy[node] if unit == active else model._rival_energy[unit][node]

    def flow(line: str, time_: int):
        return sum(
            data.ptdf[line, node] * model.NetInjection[node, time_]
            for node in model.N
        )

    products = []
    for generator, time_ in model.GT:
        reduced_cost = (
            effective_generation_offer(data, generator)
            - sum(model.lam[node, time_] for node in model._gen_nodes[generator])
            - model.nu_gen[generator, time_]
        )
        products.extend(
            (
                model.P_gen[generator, time_] * reduced_cost,
                (data.generation_capacity[generator, time_] - model.P_gen[generator, time_])
                * (-model.nu_gen[generator, time_]),
            )
        )
    for line in model.L:
        for time_ in model.T:
            line_flow = flow(line, time_)
            products.extend(
                (
                    (data.line_limit[line] - line_flow) * (-model.mu_up[line, time_]),
                    (line_flow + data.line_limit[line]) * model.mu_dn[line, time_],
                )
            )
    for unit, node in model.IN:
        for time_ in model.T:
            charge_reduced_cost = (
                0.5 * model._unit_degradation[unit]
                + model.lam[node, time_]
                - model.rho_ch[unit, node, time_]
                + eta * model.gam[unit, node, time_]
            )
            discharge_reduced_cost = (
                0.5 * model._unit_degradation[unit]
                - model.lam[node, time_]
                - model.sig_dis[unit, node, time_]
                - model.gam[unit, node, time_] / eta
            )
            products.extend(
                (
                    model.P_charge[unit, node, time_] * charge_reduced_cost,
                    model.P_discharge[unit, node, time_] * discharge_reduced_cost,
                    (unit_power(unit, node) - model.P_charge[unit, node, time_])
                    * (-model.rho_ch[unit, node, time_]),
                    (unit_power(unit, node) - model.P_discharge[unit, node, time_])
                    * (-model.sig_dis[unit, node, time_]),
                )
            )
        for soc_time in model.T_SOC:
            stationarity_lhs = model.del_soc[unit, node, soc_time]
            if soc_time in model.T:
                stationarity_lhs += model.gam[unit, node, soc_time]
            if soc_time + 1 in model.T:
                stationarity_lhs -= model.gam[unit, node, soc_time + 1]
            if soc_time == 0:
                stationarity_lhs += model.rho_per[unit, node]
            if soc_time == last_t:
                stationarity_lhs -= model.rho_per[unit, node]
            products.extend(
                (
                    model.SOC[unit, node, soc_time] * (-stationarity_lhs),
                    (unit_energy(unit, node) - model.SOC[unit, node, soc_time])
                    * (-model.del_soc[unit, node, soc_time]),
                )
            )

    values = [float(pyo.value(product)) for product in products]
    minimum = min(values, default=0.0)
    maximum = max(values, default=0.0)
    gap = float(pyo.value(model.primal_objective - model.dual_objective))
    return {
        "count": len(values),
        "minimum_product": minimum,
        "maximum_product": maximum,
        "maximum_upper_bound_violation": max(0.0, maximum),
        "maximum_nonnegativity_violation": max(0.0, -minimum),
        "primal_dual_gap_eur_per_day": gap,
        "absolute_primal_dual_gap_eur_per_day": abs(gap),
    }
