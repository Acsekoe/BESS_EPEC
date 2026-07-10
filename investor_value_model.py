"""Single-investor primal-dual strong-duality bid model.

The lower-level spot market is represented by primal feasibility, explicit dual
feasibility, and one strong-duality equality. This avoids complementarity pairs
while still enforcing optimality of the LP lower level.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pyomo.environ as pyo

from auction import Bid
from primal_market_clearing_model import MarketData, value


DEFAULT_BESS_COST_POWER_EUR_PER_MW = 6_600.0
DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH = 18_800.0
DEFAULT_DEGRADATION_EUR_PER_MWH = 15.0
DEFAULT_LIFETIME_YEARS = 15


@dataclass(frozen=True)
class InvestorBidModelConfig:
    investor_id: str
    wacc: float
    max_request_mw_per_node: float = 70.0
    ep_ratio_hours: float = 2.0
    bid_fraction_of_value: float = 0.5
    max_bid_price_eur_per_mw: float = 50.0


@dataclass(frozen=True)
class InvestorBidModelResult:
    investor_id: str
    objective_value: float
    x_power: Mapping[str, float]
    x_energy: Mapping[str, float]
    node_profit_before_access: Mapping[str, float]
    bids: tuple[Bid, ...]


def capital_recovery_factor(wacc: float, lifetime_years: int = DEFAULT_LIFETIME_YEARS) -> float:
    return wacc * (1.0 + wacc) ** lifetime_years / ((1.0 + wacc) ** lifetime_years - 1.0)


def _nodes_of_generator(data: MarketData) -> dict[str, list[str]]:
    gen_nodes: dict[str, list[str]] = {generator: [] for generator in data.generators}
    for node in data.nodes:
        for generator in data.generators_at_node.get(node, []):
            gen_nodes.setdefault(generator, []).append(node)
    return gen_nodes


def _fixed_capacity(
    target_investor: str,
    variable_by_node: pyo.Var,
    fixed_capacities: Mapping[tuple[str, str], float],
    investor: str,
    node: str,
) -> pyo.Expression:
    if investor == target_investor:
        return variable_by_node[node]
    return float(fixed_capacities.get((investor, node), 0.0))


def build_investor_bid_model(
    data: MarketData,
    config: InvestorBidModelConfig,
    all_investor_ids: list[str],
    fixed_competitor_x_power: Mapping[tuple[str, str], float],
) -> pyo.ConcreteModel:
    """Build one optimistic investor problem with LP lower-level optimality."""

    target = config.investor_id
    if target not in all_investor_ids:
        raise ValueError(f"Target investor {target!r} must be in all_investor_ids.")

    gen_nodes = _nodes_of_generator(data)
    last_t = max(data.times)
    eta = data.eta
    crf_daily = capital_recovery_factor(config.wacc) / 365.25

    m = pyo.ConcreteModel(name=f"Investor bid model {target}")

    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.G = pyo.Set(initialize=data.generators, ordered=True)
    m.I = pyo.Set(initialize=all_investor_ids, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)

    # Upper-level capacity variables for the investor being solved.
    m.X_power = pyo.Var(
        m.N,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, config.max_request_mw_per_node),
        initialize=min(1.0, config.max_request_mw_per_node),
    )
    m.X_energy = pyo.Var(m.N, domain=pyo.NonNegativeReals, initialize=config.ep_ratio_hours)
    m.energy_power_ratio = pyo.Constraint(
        m.N,
        rule=lambda model, node: model.X_energy[node] == config.ep_ratio_hours * model.X_power[node],
    )

    # Lower-level primal variables.
    m.P_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_shed = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_charge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_discharge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.SOC = pyo.Var(m.I, m.N, m.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)

    # Lower-level dual variables.
    m.lam = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=70.0)
    m.lam_sys = pyo.Var(m.T, domain=pyo.Reals, initialize=70.0)
    m.nu_gen = pyo.Var(m.G, m.T, domain=pyo.NonPositiveReals, initialize=0.0)
    m.mu_up = pyo.Var(m.L, m.T, domain=pyo.NonPositiveReals, initialize=0.0)
    m.mu_dn = pyo.Var(m.L, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.rho_ch = pyo.Var(m.I, m.N, m.T, domain=pyo.NonPositiveReals, initialize=0.0)
    m.sig_dis = pyo.Var(m.I, m.N, m.T, domain=pyo.NonPositiveReals, initialize=0.0)
    m.gam = pyo.Var(m.I, m.N, m.T, domain=pyo.Reals, initialize=0.0)
    m.del_soc = pyo.Var(m.I, m.N, m.T_SOC, domain=pyo.NonPositiveReals, initialize=0.0)
    m.rho_per = pyo.Var(m.I, m.N, domain=pyo.Reals, initialize=0.0)
    m.xi_shed = pyo.Var(m.N, m.T, domain=pyo.NonPositiveReals, initialize=0.0)

    def x_power(model: pyo.ConcreteModel, investor: str, node: str) -> pyo.Expression:
        return _fixed_capacity(target, model.X_power, fixed_competitor_x_power, investor, node)

    def x_energy(model: pyo.ConcreteModel, investor: str, node: str) -> pyo.Expression:
        if investor == target:
            return model.X_energy[node]
        return config.ep_ratio_hours * float(fixed_competitor_x_power.get((investor, node), 0.0))

    # Primal feasibility.
    def nodal_balance_rule(model: pyo.ConcreteModel, node: str, time: int) -> pyo.Expression:
        generators = data.generators_at_node.get(node, [])
        storage_net = sum(
            model.P_discharge[investor, node, time] - model.P_charge[investor, node, time]
            for investor in model.I
        )
        return (
            sum(model.P_gen[generator, time] for generator in generators)
            + storage_net
            + model.P_shed[node, time]
            - data.demand_el[node, time]
            == model.NetInjection[node, time]
        )

    m.nodal_balance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)
    m.system_balance = pyo.Constraint(
        m.T,
        rule=lambda model, time: sum(model.NetInjection[node, time] for node in model.N) == 0.0,
    )
    m.generation_capacity_bound = pyo.Constraint(
        m.G,
        m.T,
        rule=lambda model, generator, time: model.P_gen[generator, time] <= data.generation_capacity[generator, time],
    )
    m.line_upper_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, line, time: (
            sum(data.ptdf[line, node] * model.NetInjection[node, time] for node in model.N)
            <= data.line_limit[line]
        ),
    )
    m.line_lower_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, line, time: (
            sum(data.ptdf[line, node] * model.NetInjection[node, time] for node in model.N)
            >= -data.line_limit[line]
        ),
    )
    m.charge_power_bound = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, investor, node, time: model.P_charge[investor, node, time] <= x_power(model, investor, node),
    )
    m.discharge_power_bound = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, investor, node, time: model.P_discharge[investor, node, time] <= x_power(model, investor, node),
    )
    m.soc_transition = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, investor, node, time: model.SOC[investor, node, time]
        == (
            model.SOC[investor, node, time - 1]
            + eta * model.P_charge[investor, node, time]
            - model.P_discharge[investor, node, time] / eta
        ),
    )
    m.soc_capacity_bound = pyo.Constraint(
        m.I,
        m.N,
        m.T_SOC,
        rule=lambda model, investor, node, tau: model.SOC[investor, node, tau] <= x_energy(model, investor, node),
    )
    m.soc_periodicity = pyo.Constraint(
        m.I,
        m.N,
        rule=lambda model, investor, node: model.SOC[investor, node, 0] == model.SOC[investor, node, last_t],
    )
    m.load_shed_bound = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, node, time: model.P_shed[node, time] <= data.demand_el[node, time],
    )

    # Dual feasibility.
    m.gen_stationarity = pyo.Constraint(
        m.G,
        m.T,
        rule=lambda model, generator, time: (
            sum(model.lam[node, time] for node in gen_nodes.get(generator, []))
            + model.nu_gen[generator, time]
            <= data.generation_cost[generator]
        ),
    )
    m.shed_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, node, time: model.lam[node, time] + model.xi_shed[node, time] <= data.voll,
    )
    m.charge_stationarity = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, investor, node, time: (
            -model.lam[node, time]
            + model.rho_ch[investor, node, time]
            - eta * model.gam[investor, node, time]
            <= 0.0
        ),
    )
    m.discharge_stationarity = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, investor, node, time: (
            model.lam[node, time]
            + model.sig_dis[investor, node, time]
            + model.gam[investor, node, time] / eta
            <= 0.0
        ),
    )
    m.netinjection_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, node, time: (
            -model.lam[node, time]
            + model.lam_sys[time]
            + sum(data.ptdf[line, node] * (model.mu_up[line, time] + model.mu_dn[line, time]) for line in model.L)
            == 0.0
        ),
    )

    def soc_stationarity_rule(model: pyo.ConcreteModel, investor: str, node: str, tau: int) -> pyo.Expression:
        expr = model.del_soc[investor, node, tau]
        if tau in model.T:
            expr = expr + model.gam[investor, node, tau]
        if (tau + 1) in model.T:
            expr = expr - model.gam[investor, node, tau + 1]
        if tau == 0:
            expr = expr + model.rho_per[investor, node]
        if tau == last_t:
            expr = expr - model.rho_per[investor, node]
        return expr <= 0.0

    m.soc_stationarity = pyo.Constraint(m.I, m.N, m.T_SOC, rule=soc_stationarity_rule)

    def primal_objective_rule(model: pyo.ConcreteModel) -> pyo.Expression:
        return sum(
            data.generation_cost[generator] * model.P_gen[generator, time]
            for generator in model.G
            for time in model.T
        ) + sum(
            data.voll * model.P_shed[node, time]
            for node in model.N
            for time in model.T
        )

    def dual_objective_rule(model: pyo.ConcreteModel) -> pyo.Expression:
        demand_terms = sum(
            data.demand_el[node, time] * (model.lam[node, time] + model.xi_shed[node, time])
            for node in model.N
            for time in model.T
        )
        gen_cap_terms = sum(
            data.generation_capacity[generator, time] * model.nu_gen[generator, time]
            for generator in model.G
            for time in model.T
        )
        line_terms = sum(
            data.line_limit[line] * (model.mu_up[line, time] - model.mu_dn[line, time])
            for line in model.L
            for time in model.T
        )
        power_terms = sum(
            x_power(model, investor, node) * (model.rho_ch[investor, node, time] + model.sig_dis[investor, node, time])
            for investor in model.I
            for node in model.N
            for time in model.T
        )
        energy_terms = sum(
            x_energy(model, investor, node) * model.del_soc[investor, node, tau]
            for investor in model.I
            for node in model.N
            for tau in model.T_SOC
        )
        return demand_terms + gen_cap_terms + line_terms + power_terms + energy_terms

    m.primal_objective_expr = pyo.Expression(rule=primal_objective_rule)
    m.dual_objective_expr = pyo.Expression(rule=dual_objective_rule)
    m.strong_duality = pyo.Constraint(expr=m.primal_objective_expr == m.dual_objective_expr)

    def spot_revenue_rule(model: pyo.ConcreteModel) -> pyo.Expression:
        return sum(
            model.lam[node, time]
            * (model.P_discharge[target, node, time] - model.P_charge[target, node, time])
            for node in model.N
            for time in model.T
        )

    def degradation_cost_rule(model: pyo.ConcreteModel) -> pyo.Expression:
        return 0.5 * DEFAULT_DEGRADATION_EUR_PER_MWH * sum(
            model.P_charge[target, node, time] + model.P_discharge[target, node, time]
            for node in model.N
            for time in model.T
        )

    def capex_daily_rule(model: pyo.ConcreteModel) -> pyo.Expression:
        return crf_daily * sum(
            DEFAULT_BESS_COST_POWER_EUR_PER_MW * model.X_power[node]
            + DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH * model.X_energy[node]
            for node in model.N
        )

    m.spot_revenue_expr = pyo.Expression(rule=spot_revenue_rule)
    m.degradation_cost_expr = pyo.Expression(rule=degradation_cost_rule)
    m.capex_daily_expr = pyo.Expression(rule=capex_daily_rule)
    m.investor_profit_before_access = pyo.Expression(
        expr=m.spot_revenue_expr - m.degradation_cost_expr - m.capex_daily_expr
    )
    m.objective = pyo.Objective(expr=m.investor_profit_before_access, sense=pyo.maximize)

    return m


def solve_investor_bid_model(
    data: MarketData,
    config: InvestorBidModelConfig,
    all_investor_ids: list[str],
    fixed_competitor_x_power: Mapping[tuple[str, str], float],
    solver_name: str = "ipopt",
) -> InvestorBidModelResult:
    model = build_investor_bid_model(data, config, all_investor_ids, fixed_competitor_x_power)
    solver_kwargs = {}
    if solver_name == "ipopt":
        idaes_ipopt = Path(os.environ.get("LOCALAPPDATA", "")) / "idaes" / "bin" / "ipopt.exe"
        if idaes_ipopt.exists():
            solver_kwargs["executable"] = str(idaes_ipopt)
            solver_kwargs["solver_io"] = "nl"
    solver = pyo.SolverFactory(solver_name, **solver_kwargs)
    if not solver.available(exception_flag=False):
        raise RuntimeError(f"Solver {solver_name!r} is not available for investor bid model.")

    if solver_name == "ipopt":
        solver.options["max_iter"] = 1500
        solver.options["tol"] = 1e-4
        solver.options["acceptable_tol"] = 1e-3
        solver.options["linear_solver"] = "ma97"
        solver.options["max_cpu_time"] = 30
        solver.options["print_level"] = 0

    results = solver.solve(model, tee=False)
    termination = results.solver.termination_condition
    if termination not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}:
        raise RuntimeError(f"Investor bid model failed for {config.investor_id}: termination={termination}.")

    x_power = {str(node): max(0.0, value(model.X_power[node])) for node in model.N}
    x_energy = {str(node): max(0.0, value(model.X_energy[node])) for node in model.N}
    node_profit: dict[str, float] = {}
    bids: list[Bid] = []

    crf_daily = capital_recovery_factor(config.wacc) / 365.25
    for node in model.N:
        node_str = str(node)
        spot = sum(
            value(model.lam[node, time])
            * (value(model.P_discharge[config.investor_id, node, time]) - value(model.P_charge[config.investor_id, node, time]))
            for time in model.T
        )
        degradation = 0.5 * DEFAULT_DEGRADATION_EUR_PER_MWH * sum(
            value(model.P_charge[config.investor_id, node, time])
            + value(model.P_discharge[config.investor_id, node, time])
            for time in model.T
        )
        capex = crf_daily * (
            DEFAULT_BESS_COST_POWER_EUR_PER_MW * x_power[node_str]
            + DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH * x_energy[node_str]
        )
        profit = spot - degradation - capex
        node_profit[node_str] = profit

        if x_power[node_str] <= 1e-6 or profit <= 0.0:
            bid_price = 0.0
            quantity = 0.0
        else:
            bid_price = min(
                config.max_bid_price_eur_per_mw,
                config.bid_fraction_of_value * profit / x_power[node_str],
            )
            quantity = x_power[node_str]

        bids.append(
            Bid(
                investor=config.investor_id,
                node=node_str,
                quantity_mw=quantity,
                price_eur_per_mw=max(0.0, bid_price),
            )
        )

    return InvestorBidModelResult(
        investor_id=config.investor_id,
        objective_value=value(model.objective),
        x_power=x_power,
        x_energy=x_energy,
        node_profit_before_access=node_profit,
        bids=tuple(bids),
    )
