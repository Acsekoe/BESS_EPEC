"""Single-investor MPEC using primal-dual optimality and strong duality.

This is the reduced proof model before rebuilding the EPEC. It represents one
8% WACC investor with no competitor storage assets. The lower-level spot-market
LP is embedded through primal feasibility, dual feasibility, and one strong
duality equality.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import pyomo.environ as pyo

from dual_market_clearing_model import build_dual_market_clearing_model
from primal_market_clearing_model import (
    DEFAULT_DATA_PATH,
    MarketData,
    build_primal_market_clearing_model,
    get_solver,
    load_market_data,
    solve_model,
    value,
)
from solver_utils import get_ipopt_solver


MODEL_NAME = "Single Investor Primal-Dual MPEC"
INVESTOR_ID = "I1"
EXISTING_ID = "E0"
# Experiment inputs (demand tiers, alternative capacities, ...) live in a
# separate JSON so the baseline market_data.json stays the untouched benchmark.
EXPERIMENT_DATA_PATH = DEFAULT_DATA_PATH.with_name("market_data_experiment.json")
DEFAULT_WACC = 0.08
DEFAULT_LIFETIME_YEARS = 15
DEFAULT_NODE_LIMIT_MW = 100.0
DEFAULT_RATIO_MIN = 2.0
DEFAULT_RATIO_MAX = 8.0
DEFAULT_BESS_COST_POWER_EUR_PER_MW = 6_600.0
DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH = 18_800.0
DEFAULT_DEGRADATION_EUR_PER_MWH = 15.0
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "single_investor_mpec"


def capital_recovery_factor(wacc: float, lifetime_years: int = DEFAULT_LIFETIME_YEARS) -> float:
    return wacc * (1.0 + wacc) ** lifetime_years / ((1.0 + wacc) ** lifetime_years - 1.0)


def single_storage_data(
    data: MarketData,
    power_mw: float,
    ratio_hours: float,
    existing_power_mw: float = 0.0,
    existing_ratio_hours: float = 2.0,
) -> MarketData:
    """Return data with the active investor (and optional existing fleet) as storage units."""

    units = [INVESTOR_ID]
    x_power = {(INVESTOR_ID, node): float(power_mw) for node in data.nodes}
    x_energy = {(INVESTOR_ID, node): float(power_mw) * ratio_hours for node in data.nodes}
    if existing_power_mw > 0.0:
        units.append(EXISTING_ID)
        for node in data.nodes:
            x_power[(EXISTING_ID, node)] = float(existing_power_mw)
            x_energy[(EXISTING_ID, node)] = float(existing_power_mw) * existing_ratio_hours
    return replace(data, storage_units=units, x_power=x_power, x_energy=x_energy)


def fixed_storage_data_from_solution(model: pyo.ConcreteModel) -> MarketData:
    """Return lower-level data with storage capacities fixed at the MPEC solution."""

    data: MarketData = model._market_data
    units = [INVESTOR_ID]
    x_power = {(INVESTOR_ID, node): max(0.0, value(model.X_power[node])) for node in data.nodes}
    x_energy = {(INVESTOR_ID, node): max(0.0, value(model.X_energy[node])) for node in data.nodes}
    if model._existing_power_mw > 0.0:
        units.append(EXISTING_ID)
        existing_energy_mwh = model._existing_power_mw * model._existing_ratio_hours
        for node in data.nodes:
            x_power[(EXISTING_ID, node)] = model._existing_power_mw
            x_energy[(EXISTING_ID, node)] = existing_energy_mwh
    return replace(data, storage_units=units, x_power=x_power, x_energy=x_energy)


def _nodes_of_generator(data: MarketData) -> dict[str, list[str]]:
    gen_nodes: dict[str, list[str]] = {generator: [] for generator in data.generators}
    for node in data.nodes:
        for generator in data.generators_at_node.get(node, []):
            gen_nodes.setdefault(generator, []).append(node)
    return gen_nodes


def build_single_investor_mpec(
    data: MarketData,
    *,
    wacc: float = DEFAULT_WACC,
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW,
    ratio_min: float = DEFAULT_RATIO_MIN,
    ratio_max: float = DEFAULT_RATIO_MAX,
    initial_power_mw: float = 10.0,
    initial_ratio_hours: float = DEFAULT_RATIO_MIN,
    fixed_power_mw: float | None = None,
    dual_bound_scale: float = 10.0,
    existing_power_mw: float = 0.0,
    existing_ratio_hours: float = 2.0,
) -> pyo.ConcreteModel:
    """Build the one-investor MPEC.

    ``existing_power_mw`` places an exogenous, non-strategic BESS unit of that
    size at every node inside the lower-level market clearing. It consumes part
    of the shared nodal connection limit, so the investor can only add up to
    ``node_limit_mw - existing_power_mw`` per node.
    """

    if dual_bound_scale <= 0.0:
        raise ValueError("dual_bound_scale must be positive.")
    if existing_power_mw < 0.0:
        raise ValueError("existing_power_mw must be non-negative.")
    if existing_power_mw > node_limit_mw:
        raise ValueError("existing_power_mw exceeds the nodal connection limit.")
    invest_limit_mw = node_limit_mw - existing_power_mw
    existing_energy_mwh = existing_power_mw * existing_ratio_hours
    storage_units = [INVESTOR_ID] + ([EXISTING_ID] if existing_power_mw > 0.0 else [])

    gen_nodes = _nodes_of_generator(data)
    last_t = max(data.times)
    eta = data.eta
    crf_daily = capital_recovery_factor(wacc) / 365.25
    # Congestion duals can exceed VOLL under the PTDF formulation because a
    # line shadow price maps into nodal scarcity through PTDF coefficients.
    dual_bound = dual_bound_scale * data.voll

    m = pyo.ConcreteModel(name=MODEL_NAME)

    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.G = pyo.Set(initialize=data.generators, ordered=True)
    m.I = pyo.Set(initialize=storage_units, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)
    m.K = pyo.Set(initialize=data.demand_tiers, ordered=True)

    # Upper-level investment variables. Existing capacity consumes part of the
    # shared nodal connection limit, shrinking the investor's headroom.
    init_power = min(max(initial_power_mw, 0.0), invest_limit_mw)
    init_energy = max(initial_ratio_hours, ratio_min) * init_power
    m.X_power = pyo.Var(m.N, bounds=(0.0, invest_limit_mw), initialize=init_power)
    m.X_energy = pyo.Var(m.N, bounds=(0.0, ratio_max * invest_limit_mw), initialize=init_energy)
    m.energy_ratio_min = pyo.Constraint(m.N, rule=lambda model, n: model.X_energy[n] >= ratio_min * model.X_power[n])
    m.energy_ratio_max = pyo.Constraint(m.N, rule=lambda model, n: model.X_energy[n] <= ratio_max * model.X_power[n])

    if fixed_power_mw is not None:
        fixed_power = min(max(float(fixed_power_mw), 0.0), invest_limit_mw)
        for node in data.nodes:
            m.X_power[node].fix(fixed_power)
            m.X_energy[node].fix(max(initial_ratio_hours, ratio_min) * fixed_power)

    # Lower-level primal variables.
    m.P_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_shed = pyo.Var(m.K, m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_charge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_discharge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.SOC = pyo.Var(m.I, m.N, m.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)

    # Lower-level dual variables with broad finite bounds for Ipopt stability.
    m.lam = pyo.Var(m.N, m.T, bounds=(-dual_bound, dual_bound), initialize=80.0)
    m.lam_sys = pyo.Var(m.T, bounds=(-dual_bound, dual_bound), initialize=80.0)
    m.nu_gen = pyo.Var(m.G, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_up = pyo.Var(m.L, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_dn = pyo.Var(m.L, m.T, bounds=(0.0, dual_bound), initialize=0.0)
    m.rho_ch = pyo.Var(m.I, m.N, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.sig_dis = pyo.Var(m.I, m.N, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.gam = pyo.Var(m.I, m.N, m.T, bounds=(-dual_bound, dual_bound), initialize=0.0)
    m.del_soc = pyo.Var(m.I, m.N, m.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.rho_per = pyo.Var(m.I, m.N, bounds=(-dual_bound, dual_bound), initialize=0.0)
    m.xi_shed = pyo.Var(m.K, m.N, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)

    # Primal feasibility.
    def nodal_balance_rule(model: pyo.ConcreteModel, node: str, time: int) -> pyo.Expression:
        storage_net = sum(
            model.P_discharge[unit, node, time] - model.P_charge[unit, node, time] for unit in model.I
        )
        return (
            sum(model.P_gen[generator, time] for generator in data.generators_at_node.get(node, []))
            + storage_net
            + sum(model.P_shed[k, node, time] for k in model.K)
            - data.demand_el[node, time]
            == model.NetInjection[node, time]
        )

    m.nodal_balance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)
    m.system_balance = pyo.Constraint(m.T, rule=lambda model, t: sum(model.NetInjection[n, t] for n in model.N) == 0.0)
    m.generation_capacity_bound = pyo.Constraint(
        m.G,
        m.T,
        rule=lambda model, g, t: model.P_gen[g, t] <= data.generation_capacity[g, t],
    )
    m.line_upper_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, l, t: sum(data.ptdf[l, n] * model.NetInjection[n, t] for n in model.N) <= data.line_limit[l],
    )
    m.line_lower_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, l, t: sum(data.ptdf[l, n] * model.NetInjection[n, t] for n in model.N) >= -data.line_limit[l],
    )
    def unit_power_limit(model: pyo.ConcreteModel, i: str, n: str):
        return model.X_power[n] if i == INVESTOR_ID else existing_power_mw

    m.charge_power_bound = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: model.P_charge[i, n, t] <= unit_power_limit(model, i, n),
    )
    m.discharge_power_bound = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: model.P_discharge[i, n, t] <= unit_power_limit(model, i, n),
    )
    m.soc_transition = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: model.SOC[i, n, t]
        == model.SOC[i, n, t - 1] + eta * model.P_charge[i, n, t] - model.P_discharge[i, n, t] / eta,
    )
    m.soc_capacity_bound = pyo.Constraint(
        m.I,
        m.N,
        m.T_SOC,
        rule=lambda model, i, n, tau: model.SOC[i, n, tau]
        <= (model.X_energy[n] if i == INVESTOR_ID else existing_energy_mwh),
    )
    m.soc_periodicity = pyo.Constraint(
        m.I,
        m.N,
        rule=lambda model, i, n: model.SOC[i, n, 0] == model.SOC[i, n, last_t],
    )
    m.load_shed_bound = pyo.Constraint(
        m.K,
        m.N,
        m.T,
        rule=lambda model, k, n, t: model.P_shed[k, n, t] <= data.tier_share[k] * data.demand_el[n, t],
    )

    # Dual feasibility.
    m.gen_stationarity = pyo.Constraint(
        m.G,
        m.T,
        rule=lambda model, g, t: sum(model.lam[n, t] for n in gen_nodes.get(g, [])) + model.nu_gen[g, t]
        <= data.generation_cost[g],
    )
    m.shed_stationarity = pyo.Constraint(
        m.K,
        m.N,
        m.T,
        rule=lambda model, k, n, t: model.lam[n, t] + model.xi_shed[k, n, t] <= data.tier_wtp[k],
    )
    m.charge_stationarity = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: -model.lam[n, t] + model.rho_ch[i, n, t] - eta * model.gam[i, n, t] <= 0.0,
    )
    m.discharge_stationarity = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: model.lam[n, t] + model.sig_dis[i, n, t] + model.gam[i, n, t] / eta <= 0.0,
    )
    m.netinjection_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, n, t: -model.lam[n, t]
        + model.lam_sys[t]
        + sum(data.ptdf[l, n] * (model.mu_up[l, t] + model.mu_dn[l, t]) for l in model.L)
        == 0.0,
    )

    def soc_stationarity_rule(model: pyo.ConcreteModel, i: str, n: str, tau: int) -> pyo.Expression:
        expr = model.del_soc[i, n, tau]
        if tau in model.T:
            expr = expr + model.gam[i, n, tau]
        if (tau + 1) in model.T:
            expr = expr - model.gam[i, n, tau + 1]
        if tau == 0:
            expr = expr + model.rho_per[i, n]
        if tau == last_t:
            expr = expr - model.rho_per[i, n]
        return expr <= 0.0

    m.soc_stationarity = pyo.Constraint(m.I, m.N, m.T_SOC, rule=soc_stationarity_rule)

    # Strong duality.
    m.primal_objective_expr = pyo.Expression(
        expr=sum(data.generation_cost[g] * m.P_gen[g, t] for g in m.G for t in m.T)
        + sum(data.tier_wtp[k] * m.P_shed[k, n, t] for k in m.K for n in m.N for t in m.T)
    )
    m.dual_objective_expr = pyo.Expression(
        expr=sum(
            data.demand_el[n, t]
            * (m.lam[n, t] + sum(data.tier_share[k] * m.xi_shed[k, n, t] for k in m.K))
            for n in m.N
            for t in m.T
        )
        + sum(data.generation_capacity[g, t] * m.nu_gen[g, t] for g in m.G for t in m.T)
        + sum(data.line_limit[l] * (m.mu_up[l, t] - m.mu_dn[l, t]) for l in m.L for t in m.T)
        + sum(
            unit_power_limit(m, i, n) * (m.rho_ch[i, n, t] + m.sig_dis[i, n, t])
            for i in m.I
            for n in m.N
            for t in m.T
        )
        + sum(
            (m.X_energy[n] if i == INVESTOR_ID else existing_energy_mwh) * m.del_soc[i, n, tau]
            for i in m.I
            for n in m.N
            for tau in m.T_SOC
        )
    )
    m.strong_duality = pyo.Constraint(expr=m.primal_objective_expr == m.dual_objective_expr)

    # Upper-level investor objective.
    m.spot_revenue_expr = pyo.Expression(
        expr=sum(
            m.lam[n, t] * (m.P_discharge[INVESTOR_ID, n, t] - m.P_charge[INVESTOR_ID, n, t])
            for n in m.N
            for t in m.T
        )
    )
    m.degradation_cost_expr = pyo.Expression(
        expr=0.5
        * DEFAULT_DEGRADATION_EUR_PER_MWH
        * sum(m.P_charge[INVESTOR_ID, n, t] + m.P_discharge[INVESTOR_ID, n, t] for n in m.N for t in m.T)
    )
    m.capex_daily_expr = pyo.Expression(
        expr=crf_daily
        * sum(
            DEFAULT_BESS_COST_POWER_EUR_PER_MW * m.X_power[n]
            + DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH * m.X_energy[n]
            for n in m.N
        )
    )
    m.investor_profit_expr = pyo.Expression(expr=m.spot_revenue_expr - m.degradation_cost_expr - m.capex_daily_expr)
    m.objective = pyo.Objective(expr=m.investor_profit_expr, sense=pyo.maximize)

    m._market_data = data
    m._existing_power_mw = existing_power_mw
    m._existing_ratio_hours = existing_ratio_hours
    return m


def initialize_from_lp(model: pyo.ConcreteModel, data: MarketData, ratio_hours: float, solver_name: str | None) -> None:
    """Warm-start primal and dual variables from LP solves at current capacity."""

    avg_power = sum(value(model.X_power[n]) for n in model.N) / max(1, len(list(model.N)))
    lp_data = single_storage_data(
        data,
        avg_power,
        ratio_hours,
        existing_power_mw=model._existing_power_mw,
        existing_ratio_hours=model._existing_ratio_hours,
    )

    primal = build_primal_market_clearing_model(lp_data)
    primal_results = solve_model(primal, solver_name)
    if primal_results.solver.termination_condition == pyo.TerminationCondition.optimal:
        for g in model.G:
            for t in model.T:
                model.P_gen[g, t].set_value(value(primal.P_gen[g, t]))
        for n in model.N:
            for t in model.T:
                for k in model.K:
                    model.P_shed[k, n, t].set_value(value(primal.P_shed[k, n, t]))
                model.NetInjection[n, t].set_value(value(primal.NetInjection[n, t]))
        for i in model.I:
            for n in model.N:
                for t in model.T:
                    model.P_charge[i, n, t].set_value(value(primal.P_charge[i, n, t]))
                    model.P_discharge[i, n, t].set_value(value(primal.P_discharge[i, n, t]))
                for tau in model.T_SOC:
                    model.SOC[i, n, tau].set_value(value(primal.SOC[i, n, tau]))

    dual = build_dual_market_clearing_model(lp_data)
    _, lp_solver = get_solver(solver_name)
    dual_results = lp_solver.solve(dual, tee=False)
    if dual_results.solver.termination_condition == pyo.TerminationCondition.optimal:
        for n in model.N:
            for t in model.T:
                model.lam[n, t].set_value(value(dual.lam[n, t]))
        for t in model.T:
            model.lam_sys[t].set_value(value(dual.lam_sys[t]))
        for g in model.G:
            for t in model.T:
                model.nu_gen[g, t].set_value(value(dual.nu_gen[g, t]))
        for l in model.L:
            for t in model.T:
                model.mu_up[l, t].set_value(value(dual.mu_up[l, t]))
                model.mu_dn[l, t].set_value(value(dual.mu_dn[l, t]))
        for i in model.I:
            for n in model.N:
                for t in model.T:
                    model.rho_ch[i, n, t].set_value(value(dual.rho_ch[i, n, t]))
                    model.sig_dis[i, n, t].set_value(value(dual.sig_dis[i, n, t]))
                    model.gam[i, n, t].set_value(value(dual.gam[i, n, t]))
                for tau in model.T_SOC:
                    model.del_soc[i, n, tau].set_value(value(dual.del_soc[i, n, tau]))
                model.rho_per[i, n].set_value(value(dual.rho_per[i, n]))
        for n in model.N:
            for t in model.T:
                for k in model.K:
                    model.xi_shed[k, n, t].set_value(value(dual.xi_shed[k, n, t]))


def max_strong_duality_gap(model: pyo.ConcreteModel) -> float:
    return abs(value(model.primal_objective_expr) - value(model.dual_objective_expr))


def line_flow(model: pyo.ConcreteModel, line: str, time: int) -> float:
    data: MarketData = model._market_data
    return sum(data.ptdf[line, node] * value(model.NetInjection[node, time]) for node in model.N)


def compute_reference_settlement(model: pyo.ConcreteModel, solver_name: str | None) -> dict[str, object]:
    """Settle the MPEC solution against standalone lower-level LP reference prices."""

    fixed_data = fixed_storage_data_from_solution(model)
    reference = build_primal_market_clearing_model(fixed_data)
    solver_label, lp_solver = get_solver(solver_name)
    print(f"Solving reference settlement LLP with {solver_label}...")
    results = lp_solver.solve(reference, tee=False)
    termination = results.solver.termination_condition
    if termination != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Reference settlement LLP did not solve optimally (termination={termination}).")

    reference_lambda: dict[tuple[str, int], float] = {}
    for node in reference.N:
        for time in reference.T:
            dual = reference.dual.get(reference.nodal_balance[node, time], None)
            if dual is None:
                raise RuntimeError("Reference settlement LLP did not return nodal-balance duals.")
            reference_lambda[(node, time)] = float(dual)

    lp_charge = sum(value(reference.P_charge[INVESTOR_ID, n, t]) for n in reference.N for t in reference.T)
    lp_discharge = sum(value(reference.P_discharge[INVESTOR_ID, n, t]) for n in reference.N for t in reference.T)
    lp_revenue = sum(
        reference_lambda[n, t]
        * (value(reference.P_discharge[INVESTOR_ID, n, t]) - value(reference.P_charge[INVESTOR_ID, n, t]))
        for n in reference.N
        for t in reference.T
    )
    lp_degradation = 0.5 * DEFAULT_DEGRADATION_EUR_PER_MWH * (lp_charge + lp_discharge)
    capex = value(model.capex_daily_expr)
    lp_profit = lp_revenue - lp_degradation - capex

    mpec_dispatch_revenue = sum(
        reference_lambda[n, t] * (value(model.P_discharge[INVESTOR_ID, n, t]) - value(model.P_charge[INVESTOR_ID, n, t]))
        for n in model.N
        for t in model.T
    )
    mpec_dispatch_profit = mpec_dispatch_revenue - value(model.degradation_cost_expr) - capex

    return {
        "solver": solver_label,
        "solver_status": str(results.solver.status),
        "termination": str(termination),
        "lower_level_objective_eur_per_day": value(reference.objective),
        "spot_revenue_at_reference_prices_eur_per_day": lp_revenue,
        "degradation_cost_at_reference_dispatch_eur_per_day": lp_degradation,
        "profit_at_reference_prices_eur_per_day": lp_profit,
        "optimistic_minus_reference_profit_eur_per_day": value(model.investor_profit_expr) - lp_profit,
        "mpec_dispatch_spot_revenue_at_reference_prices_eur_per_day": mpec_dispatch_revenue,
        "mpec_dispatch_profit_at_reference_prices_eur_per_day": mpec_dispatch_profit,
        "optimistic_minus_mpec_dispatch_reference_profit_eur_per_day": value(model.investor_profit_expr)
        - mpec_dispatch_profit,
        "reference_total_charge_mwh": lp_charge,
        "reference_total_discharge_mwh": lp_discharge,
        "reference_total_storage_throughput_mwh": lp_charge + lp_discharge,
        "reference_lambda_min_eur_per_mwh": min(reference_lambda.values()),
        "reference_lambda_max_eur_per_mwh": max(reference_lambda.values()),
        "reference_lambda": reference_lambda,
        "reference_model": reference,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_solution(
    model: pyo.ConcreteModel,
    output_dir: Path,
    solver_status: str,
    termination: str,
    reference_settlement: dict[str, object] | None = None,
) -> None:
    """Write detailed MPEC solution diagnostics to CSV/JSON files."""

    data: MarketData = model._market_data
    output_dir.mkdir(parents=True, exist_ok=True)

    total_charge = sum(value(model.P_charge[INVESTOR_ID, n, t]) for n in model.N for t in model.T)
    total_discharge = sum(value(model.P_discharge[INVESTOR_ID, n, t]) for n in model.N for t in model.T)
    total_shed = sum(value(model.P_shed[k, n, t]) for k in model.K for n in model.N for t in model.T)
    shed_by_tier = {
        k: sum(value(model.P_shed[k, n, t]) for n in model.N for t in model.T) for k in model.K
    }
    total_power = sum(value(model.X_power[n]) for n in model.N)
    total_energy = sum(value(model.X_energy[n]) for n in model.N)
    throughput = total_charge + total_discharge

    summary = {
        "model": MODEL_NAME,
        "investor": INVESTOR_ID,
        "wacc": DEFAULT_WACC,
        "solver_status": solver_status,
        "termination": termination,
        "profit_eur_per_day": value(model.investor_profit_expr),
        "spot_revenue_eur_per_day": value(model.spot_revenue_expr),
        "degradation_cost_eur_per_day": value(model.degradation_cost_expr),
        "capex_daily_eur_per_day": value(model.capex_daily_expr),
        "lower_level_primal_objective_eur_per_day": value(model.primal_objective_expr),
        "lower_level_dual_objective_eur_per_day": value(model.dual_objective_expr),
        "strong_duality_gap": max_strong_duality_gap(model),
        "total_power_mw": total_power,
        "total_energy_mwh": total_energy,
        "total_charge_mwh": total_charge,
        "total_discharge_mwh": total_discharge,
        "total_storage_throughput_mwh": throughput,
        "total_load_shed_mwh": total_shed,
        "load_shed_by_tier_mwh": shed_by_tier,
        "demand_tiers": [
            {"tier": k, "share": data.tier_share[k], "wtp_eur_per_mwh": data.tier_wtp[k]} for k in model.K
        ],
        "equivalent_cycles_throughput_over_2e": throughput / (2.0 * total_energy) if total_energy > 0.0 else 0.0,
        "equivalent_cycles_discharge_over_e": total_discharge / total_energy if total_energy > 0.0 else 0.0,
        "line_limits_mw": {line: data.line_limit[line] for line in data.lines},
        "existing_power_mw_per_node": model._existing_power_mw,
        "existing_ratio_hours": model._existing_ratio_hours,
        "storage_units": list(model.I),
    }
    if reference_settlement is not None:
        summary.update(
            {
                "reference_settlement_solver": reference_settlement["solver"],
                "reference_settlement_solver_status": reference_settlement["solver_status"],
                "reference_settlement_termination": reference_settlement["termination"],
                "reference_lower_level_objective_eur_per_day": reference_settlement[
                    "lower_level_objective_eur_per_day"
                ],
                "spot_revenue_at_reference_prices_eur_per_day": reference_settlement[
                    "spot_revenue_at_reference_prices_eur_per_day"
                ],
                "degradation_cost_at_reference_dispatch_eur_per_day": reference_settlement[
                    "degradation_cost_at_reference_dispatch_eur_per_day"
                ],
                "profit_at_reference_prices_eur_per_day": reference_settlement[
                    "profit_at_reference_prices_eur_per_day"
                ],
                "optimistic_minus_reference_profit_eur_per_day": reference_settlement[
                    "optimistic_minus_reference_profit_eur_per_day"
                ],
                "mpec_dispatch_spot_revenue_at_reference_prices_eur_per_day": reference_settlement[
                    "mpec_dispatch_spot_revenue_at_reference_prices_eur_per_day"
                ],
                "mpec_dispatch_profit_at_reference_prices_eur_per_day": reference_settlement[
                    "mpec_dispatch_profit_at_reference_prices_eur_per_day"
                ],
                "optimistic_minus_mpec_dispatch_reference_profit_eur_per_day": reference_settlement[
                    "optimistic_minus_mpec_dispatch_reference_profit_eur_per_day"
                ],
                "reference_total_charge_mwh": reference_settlement["reference_total_charge_mwh"],
                "reference_total_discharge_mwh": reference_settlement["reference_total_discharge_mwh"],
                "reference_total_storage_throughput_mwh": reference_settlement[
                    "reference_total_storage_throughput_mwh"
                ],
                "reference_lambda_min_eur_per_mwh": reference_settlement["reference_lambda_min_eur_per_mwh"],
                "reference_lambda_max_eur_per_mwh": reference_settlement["reference_lambda_max_eur_per_mwh"],
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_csv(
        output_dir / "investment_by_node.csv",
        ["node", "x_power_mw", "x_energy_mwh", "energy_power_ratio_h"],
        [
            {
                "node": n,
                "x_power_mw": value(model.X_power[n]),
                "x_energy_mwh": value(model.X_energy[n]),
                "energy_power_ratio_h": value(model.X_energy[n]) / value(model.X_power[n]) if value(model.X_power[n]) > 1e-9 else 0.0,
            }
            for n in model.N
        ],
    )

    _write_csv(
        output_dir / "node_hour_balance_prices.csv",
        [
            "hour",
            "node",
            "demand_mw",
            "generation_mw",
            "p_charge_mw",
            "p_discharge_mw",
            "storage_net_injection_mw",
            "load_shed_mw",
            "net_injection_mw",
            "lambda_eur_per_mwh",
        ],
        [
            {
                "hour": t,
                "node": n,
                "demand_mw": data.demand_el[n, t],
                "generation_mw": sum(value(model.P_gen[g, t]) for g in data.generators_at_node.get(n, [])),
                "p_charge_mw": sum(value(model.P_charge[i, n, t]) for i in model.I),
                "p_discharge_mw": sum(value(model.P_discharge[i, n, t]) for i in model.I),
                "storage_net_injection_mw": sum(
                    value(model.P_discharge[i, n, t]) - value(model.P_charge[i, n, t]) for i in model.I
                ),
                "load_shed_mw": sum(value(model.P_shed[k, n, t]) for k in model.K),
                "net_injection_mw": value(model.NetInjection[n, t]),
                "lambda_eur_per_mwh": value(model.lam[n, t]),
            }
            for t in model.T
            for n in model.N
        ],
    )

    _write_csv(
        output_dir / "shed_tier_hour_duals.csv",
        ["hour", "node", "tier", "wtp_eur_per_mwh", "shed_mw", "bound_mw", "xi_shed_dual"],
        [
            {
                "hour": t,
                "node": n,
                "tier": k,
                "wtp_eur_per_mwh": data.tier_wtp[k],
                "shed_mw": value(model.P_shed[k, n, t]),
                "bound_mw": data.tier_share[k] * data.demand_el[n, t],
                "xi_shed_dual": value(model.xi_shed[k, n, t]),
            }
            for t in model.T
            for n in model.N
            for k in model.K
        ],
    )

    _write_csv(
        output_dir / "generator_hour_dispatch_duals.csv",
        ["hour", "generator", "dispatch_mw", "capacity_mw", "marginal_cost_eur_per_mwh", "nu_gen_dual"],
        [
            {
                "hour": t,
                "generator": g,
                "dispatch_mw": value(model.P_gen[g, t]),
                "capacity_mw": data.generation_capacity[g, t],
                "marginal_cost_eur_per_mwh": data.generation_cost[g],
                "nu_gen_dual": value(model.nu_gen[g, t]),
            }
            for t in model.T
            for g in model.G
        ],
    )

    _write_csv(
        output_dir / "line_hour_flows_duals.csv",
        [
            "hour",
            "line",
            "flow_mw",
            "limit_mw",
            "abs_utilization",
            "mu_upper_dual",
            "mu_lower_dual",
        ],
        [
            {
                "hour": t,
                "line": l,
                "flow_mw": line_flow(model, l, t),
                "limit_mw": data.line_limit[l],
                "abs_utilization": abs(line_flow(model, l, t)) / data.line_limit[l] if data.line_limit[l] > 0.0 else 0.0,
                "mu_upper_dual": value(model.mu_up[l, t]),
                "mu_lower_dual": value(model.mu_dn[l, t]),
            }
            for t in model.T
            for l in model.L
        ],
    )

    _write_csv(
        output_dir / "storage_hour_operation_duals.csv",
        [
            "unit",
            "hour",
            "node",
            "soc_start_mwh",
            "soc_end_mwh",
            "p_charge_mw",
            "p_discharge_mw",
            "net_injection_mw",
            "lambda_eur_per_mwh",
            "spot_revenue_eur",
            "degradation_cost_eur",
            "rho_charge_dual",
            "sigma_discharge_dual",
            "gamma_soc_transition_dual",
        ],
        [
            {
                "unit": i,
                "hour": t,
                "node": n,
                "soc_start_mwh": value(model.SOC[i, n, t - 1]),
                "soc_end_mwh": value(model.SOC[i, n, t]),
                "p_charge_mw": value(model.P_charge[i, n, t]),
                "p_discharge_mw": value(model.P_discharge[i, n, t]),
                "net_injection_mw": value(model.P_discharge[i, n, t]) - value(model.P_charge[i, n, t]),
                "lambda_eur_per_mwh": value(model.lam[n, t]),
                "spot_revenue_eur": value(model.lam[n, t])
                * (value(model.P_discharge[i, n, t]) - value(model.P_charge[i, n, t])),
                "degradation_cost_eur": 0.5
                * DEFAULT_DEGRADATION_EUR_PER_MWH
                * (value(model.P_charge[i, n, t]) + value(model.P_discharge[i, n, t])),
                "rho_charge_dual": value(model.rho_ch[i, n, t]),
                "sigma_discharge_dual": value(model.sig_dis[i, n, t]),
                "gamma_soc_transition_dual": value(model.gam[i, n, t]),
            }
            for i in model.I
            for t in model.T
            for n in model.N
        ],
    )

    _write_csv(
        output_dir / "soc_hour_duals.csv",
        ["unit", "soc_hour", "node", "soc_mwh", "delta_soc_capacity_dual", "rho_periodicity_dual"],
        [
            {
                "unit": i,
                "soc_hour": tau,
                "node": n,
                "soc_mwh": value(model.SOC[i, n, tau]),
                "delta_soc_capacity_dual": value(model.del_soc[i, n, tau]),
                "rho_periodicity_dual": value(model.rho_per[i, n]),
            }
            for i in model.I
            for tau in model.T_SOC
            for n in model.N
        ],
    )

    _write_csv(
        output_dir / "system_hour_duals.csv",
        ["hour", "lambda_system", "system_net_injection_residual_mw"],
        [
            {
                "hour": t,
                "lambda_system": value(model.lam_sys[t]),
                "system_net_injection_residual_mw": sum(value(model.NetInjection[n, t]) for n in model.N),
            }
            for t in model.T
        ],
    )

    if reference_settlement is not None:
        reference: pyo.ConcreteModel = reference_settlement["reference_model"]  # type: ignore[assignment]
        reference_lambda: dict[tuple[str, int], float] = reference_settlement["reference_lambda"]  # type: ignore[assignment]

        _write_csv(
            output_dir / "reference_node_hour_prices.csv",
            ["hour", "node", "lambda_reference_eur_per_mwh"],
            [
                {
                    "hour": t,
                    "node": n,
                    "lambda_reference_eur_per_mwh": reference_lambda[n, t],
                }
                for t in reference.T
                for n in reference.N
            ],
        )

        _write_csv(
            output_dir / "reference_storage_hour_operation.csv",
            [
                "unit",
                "hour",
                "node",
                "p_charge_mw",
                "p_discharge_mw",
                "net_injection_mw",
                "lambda_reference_eur_per_mwh",
                "spot_revenue_eur",
                "degradation_cost_eur",
            ],
            [
                {
                    "unit": i,
                    "hour": t,
                    "node": n,
                    "p_charge_mw": value(reference.P_charge[i, n, t]),
                    "p_discharge_mw": value(reference.P_discharge[i, n, t]),
                    "net_injection_mw": value(reference.P_discharge[i, n, t]) - value(reference.P_charge[i, n, t]),
                    "lambda_reference_eur_per_mwh": reference_lambda[n, t],
                    "spot_revenue_eur": reference_lambda[n, t]
                    * (value(reference.P_discharge[i, n, t]) - value(reference.P_charge[i, n, t])),
                    "degradation_cost_eur": 0.5
                    * DEFAULT_DEGRADATION_EUR_PER_MWH
                    * (value(reference.P_charge[i, n, t]) + value(reference.P_discharge[i, n, t])),
                }
                for i in reference.I
                for t in reference.T
                for n in reference.N
            ],
        )


def print_solution_summary(model: pyo.ConcreteModel, reference_settlement: dict[str, object] | None = None) -> None:
    print("\nSingle-investor MPEC solution")
    print(f"  investor: {INVESTOR_ID}")
    print(f"  WACC: {DEFAULT_WACC:.2%}")
    print(f"  profit: {value(model.investor_profit_expr):,.4f} EUR/day")
    print(f"  spot revenue: {value(model.spot_revenue_expr):,.4f} EUR/day")
    print(f"  degradation cost: {value(model.degradation_cost_expr):,.4f} EUR/day")
    print(f"  CAPEX daily: {value(model.capex_daily_expr):,.4f} EUR/day")
    print(f"  lower-level primal objective: {value(model.primal_objective_expr):,.4f}")
    print(f"  lower-level dual objective: {value(model.dual_objective_expr):,.4f}")
    print(f"  strong-duality gap: {max_strong_duality_gap(model):.6e}")

    print("\nInvestment by node")
    for node in model.N:
        print(f"  {node}: X_power={value(model.X_power[node]):9.4f} MW, X_energy={value(model.X_energy[node]):9.4f} MWh")

    if reference_settlement is not None:
        print("\nReference-price settlement")
        print(f"  solver: {reference_settlement['solver']}")
        print(
            "  reference LP dispatch profit: "
            f"{reference_settlement['profit_at_reference_prices_eur_per_day']:,.4f} EUR/day"
        )
        print(
            "  reference LP dispatch spot revenue: "
            f"{reference_settlement['spot_revenue_at_reference_prices_eur_per_day']:,.4f} EUR/day"
        )
        print(
            "  MPEC dispatch at reference prices profit: "
            f"{reference_settlement['mpec_dispatch_profit_at_reference_prices_eur_per_day']:,.4f} EUR/day"
        )
        print(
            "  optimistic - reference LP profit gap: "
            f"{reference_settlement['optimistic_minus_reference_profit_eur_per_day']:,.4f} EUR/day"
        )
        print(
            "  reference lambda range: "
            f"{reference_settlement['reference_lambda_min_eur_per_mwh']:,.4f} to "
            f"{reference_settlement['reference_lambda_max_eur_per_mwh']:,.4f} EUR/MWh"
        )


def print_lambda_and_line_duals(model: pyo.ConcreteModel) -> None:
    data: MarketData = model._market_data

    print("\nNodal lambda by hour")
    for time in model.T:
        for node in model.N:
            print(f"  hour={time:>2}, node={node}: lambda={value(model.lam[node, time]):12.6f} EUR/MWh")

    print("\nLine congestion duals by hour")
    for time in model.T:
        for line in model.L:
            flow = line_flow(model, line, time)
            limit = data.line_limit[line]
            mu_upper = value(model.mu_up[line, time])
            mu_lower = value(model.mu_dn[line, time])
            print(
                f"  hour={time:>2}, line={line}: "
                f"flow={flow:12.6f} MW, limit={limit:12.6f} MW, "
                f"mu_upper={mu_upper:12.6f}, mu_lower={mu_lower:12.6f}, "
                f"mu_net={mu_upper + mu_lower:12.6f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--data", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--lp-solver", default="appsi_highs")
    parser.add_argument("--tee", action="store_true", help="Show Ipopt output.")
    parser.add_argument("--initial-power-mw", type=float, default=10.0)
    parser.add_argument("--initial-ratio-hours", type=float, default=DEFAULT_RATIO_MIN)
    parser.add_argument("--fixed-power-mw", type=float, default=None, help="Fix all nodal power capacities for validation.")
    parser.add_argument(
        "--existing-power-mw",
        type=float,
        default=0.0,
        help="Exogenous non-strategic BESS power at every node; reduces the investor's connection headroom.",
    )
    parser.add_argument(
        "--existing-ratio-hours",
        type=float,
        default=2.0,
        help="Energy-to-power ratio of the exogenous existing BESS fleet.",
    )
    parser.add_argument("--max-cpu-time", type=float, default=120.0)
    parser.add_argument(
        "--dual-bound-scale",
        type=float,
        default=10.0,
        help="Finite bound multiplier for lower-level dual variables relative to VOLL.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-export", action="store_true", help="Do not write detailed CSV/JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.data)
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR

    model = build_single_investor_mpec(
        data,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        fixed_power_mw=args.fixed_power_mw,
        dual_bound_scale=args.dual_bound_scale,
        existing_power_mw=args.existing_power_mw,
        existing_ratio_hours=args.existing_ratio_hours,
    )
    initialize_from_lp(model, data, args.initial_ratio_hours, args.lp_solver)

    solver = get_ipopt_solver({"max_cpu_time": args.max_cpu_time})
    results = solver.solve(model, tee=args.tee)
    termination = results.solver.termination_condition
    print(f"Solver status: {results.solver.status}")
    print(f"Termination: {termination}")
    if termination != pyo.TerminationCondition.optimal:
        print("MPEC solve did not terminate optimally.")
        return 1

    reference_settlement = compute_reference_settlement(model, args.lp_solver)

    print_solution_summary(model, reference_settlement)
    print_lambda_and_line_duals(model)
    if not args.no_export:
        export_solution(model, output_dir, str(results.solver.status), str(termination), reference_settlement)
        print(f"\nWrote detailed MPEC outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
