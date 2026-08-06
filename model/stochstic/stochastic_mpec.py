"""Capacity-only stochastic single-investor MPEC.

One merchant investor chooses a common nodal BESS MW/MWh fleet. Each renewable
scenario has an independent fixed-demand spot-market recourse problem embedded
through primal feasibility, dual feasibility, and exact strong duality. The
upper objective maximizes probability-weighted optimistic spot profit less one
daily annualized CAPEX payment.

The model is deliberately isolated from the maintained deterministic MPEC. It
has no rivals, portfolio generation, demand response, dispatch regularization,
strategic quantity offers, or price penalty.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import pyomo.environ as pyo


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
PRIMAL_DUAL_DIR = MODEL_DIR / "Primal and dual problems"
for candidate in (SCRIPT_DIR, MODEL_DIR, PRIMAL_DUAL_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from primal_market_clearing_model import MarketData, load_market_data  # noqa: E402
from run_experiment import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_NODE_LIMIT_MW,
    DEFAULT_SCENARIO_PATH,
    RenewableScenario,
    get_solver,
    load_scenarios,
    scenario_market_data,
    value,
    write_csv,
)
from single_investor_mpec import (  # noqa: E402
    DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH,
    DEFAULT_BESS_COST_POWER_EUR_PER_MW,
    DEFAULT_DEGRADATION_EUR_PER_MWH,
    DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    DEFAULT_LIFETIME_YEARS,
    DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    DEFAULT_RATIO_MAX,
    DEFAULT_RATIO_MIN,
    DEFAULT_SOLVER_TOL,
    DEFAULT_WACC,
    build_fixed_demand_primal_model,
    capital_recovery_factor,
    fixed_demand_reference_lambda,
)
from solver_utils import get_ipopt_solver  # noqa: E402


INVESTOR_ID = "I1"
DEFAULT_INITIAL_CAPACITIES = None
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "stochastic_mpec_default"
GENERATION_CAPACITY_TOL_MW = 1.0e-8


def generator_nodes(data: MarketData) -> dict[str, str]:
    result: dict[str, str] = {}
    for node, generators in data.generators_at_node.items():
        for generator in generators:
            if generator in result:
                raise ValueError(f"Generator {generator} is assigned to multiple nodes.")
            result[generator] = node
    missing = set(data.generators) - set(result)
    if missing:
        raise ValueError(f"Generators without a node: {sorted(missing)}")
    return result


def load_initial_fleet(
    path: Path | None,
    nodes,
    *,
    fallback_power_mw: float,
    fallback_ratio_hours: float,
) -> tuple[dict[str, float], dict[str, float]]:
    power = {str(node): fallback_power_mw for node in nodes}
    energy = {str(node): fallback_power_mw * fallback_ratio_hours for node in nodes}
    if path is None or not path.is_file():
        return power, energy
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {str(row["node"]): row for row in csv.DictReader(handle)}
    for node in power:
        if node not in rows:
            raise ValueError(f"Initial-capacity file is missing node {node}.")
        power[node] = float(rows[node]["x_power_mw"])
        energy[node] = float(rows[node]["x_energy_mwh"])
    return power, energy


def build_stochastic_mpec(
    base: MarketData,
    scenarios: tuple[RenewableScenario, ...],
    *,
    node_limit_mw: float,
    initial_power_by_node: dict[str, float],
    initial_energy_by_node: dict[str, float],
    wacc: float = DEFAULT_WACC,
    price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    dual_bound_eur_per_mwh: float = DEFAULT_DUAL_BOUND_EUR_PER_MWH,
) -> pyo.ConcreteModel:
    if node_limit_mw <= 0.0:
        raise ValueError("node_limit_mw must be positive.")
    if price_bound_eur_per_mwh <= 0.0 or dual_bound_eur_per_mwh <= 0.0:
        raise ValueError("Price and dual bounds must be positive.")
    scenario_data = {s.name: scenario_market_data(base, s) for s in scenarios}
    probabilities = {s.name: s.probability for s in scenarios}
    gen_node = generator_nodes(base)
    scenario_generation_pairs = {
        s.name: tuple(
            (g, int(t))
            for g in base.generators
            for t in base.times
            if scenario_data[s.name].generation_capacity[g, t]
            > GENERATION_CAPACITY_TOL_MW
        )
        for s in scenarios
    }
    scenario_generators_at_node_time = {
        (s.name, n, int(t)): tuple(
            g
            for g in base.generators_at_node.get(n, [])
            if (g, int(t)) in scenario_generation_pairs[s.name]
        )
        for s in scenarios
        for n in base.nodes
        for t in base.times
    }
    scenario_generation_triples = tuple(
        (s.name, g, t)
        for s in scenarios
        for g, t in scenario_generation_pairs[s.name]
    )
    last_hour = max(base.times)
    daily_crf = capital_recovery_factor(wacc, DEFAULT_LIFETIME_YEARS) / 365.25

    model = pyo.ConcreteModel(name="Stochastic capacity-only merchant MPEC")
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.S = pyo.Set(initialize=[s.name for s in scenarios], ordered=True)
    model.N = pyo.Set(initialize=base.nodes, ordered=True)
    model.G = pyo.Set(initialize=base.generators, ordered=True)
    model.SGT = pyo.Set(dimen=3, initialize=scenario_generation_triples, ordered=True)
    model.T = pyo.Set(initialize=base.times, ordered=True)
    model.T_SOC = pyo.Set(initialize=base.soc_times, ordered=True)
    model.L = pyo.Set(initialize=base.lines, ordered=True)

    model.X_power = pyo.Var(
        model.N,
        bounds=(0.0, node_limit_mw),
        initialize=lambda m, n: min(node_limit_mw, max(0.0, initial_power_by_node[str(n)])),
    )
    model.X_energy = pyo.Var(
        model.N,
        bounds=(0.0, DEFAULT_RATIO_MAX * node_limit_mw),
        initialize=lambda m, n: min(
            DEFAULT_RATIO_MAX * node_limit_mw,
            max(0.0, initial_energy_by_node[str(n)]),
        ),
    )
    model.energy_ratio_min = pyo.Constraint(
        model.N,
        rule=lambda m, n: m.X_energy[n] >= DEFAULT_RATIO_MIN * m.X_power[n],
    )
    model.energy_ratio_max = pyo.Constraint(
        model.N,
        rule=lambda m, n: m.X_energy[n] <= DEFAULT_RATIO_MAX * m.X_power[n],
    )

    model.P_gen = pyo.Var(model.SGT, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_charge = pyo.Var(model.S, model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_discharge = pyo.Var(model.S, model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.SOC = pyo.Var(model.S, model.N, model.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    model.NetInjection = pyo.Var(model.S, model.N, model.T, domain=pyo.Reals, initialize=0.0)

    price_bound = float(price_bound_eur_per_mwh)
    dual_bound = float(dual_bound_eur_per_mwh)
    model.lam = pyo.Var(model.S, model.N, model.T, bounds=(-price_bound, price_bound), initialize=60.0)
    model.lam_sys = pyo.Var(model.S, model.T, bounds=(-price_bound, price_bound), initialize=60.0)
    model.nu_gen = pyo.Var(model.SGT, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.mu_up = pyo.Var(model.S, model.L, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.mu_dn = pyo.Var(model.S, model.L, model.T, bounds=(0.0, dual_bound), initialize=0.0)
    model.rho_ch = pyo.Var(model.S, model.N, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.sig_dis = pyo.Var(model.S, model.N, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.gam = pyo.Var(model.S, model.N, model.T, bounds=(-dual_bound, dual_bound), initialize=0.0)
    model.del_soc = pyo.Var(model.S, model.N, model.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.rho_per = pyo.Var(model.S, model.N, bounds=(-dual_bound, dual_bound), initialize=0.0)

    def nodal_balance(m, s, n, t):
        data = scenario_data[s]
        return (
            sum(
                m.P_gen[s, g, t]
                for g in scenario_generators_at_node_time[s, n, int(t)]
            )
            + m.P_discharge[s, n, t]
            - m.P_charge[s, n, t]
            - data.demand_el[n, t]
            == m.NetInjection[s, n, t]
        )

    model.nodal_balance = pyo.Constraint(model.S, model.N, model.T, rule=nodal_balance)
    model.system_balance = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: sum(m.NetInjection[s, n, t] for n in m.N) == 0.0,
    )
    model.generation_capacity_bound = pyo.Constraint(
        model.SGT,
        rule=lambda m, s, g, t: m.P_gen[s, g, t]
        <= scenario_data[s].generation_capacity[g, t],
    )

    def flow(m, s, line, hour):
        data = scenario_data[s]
        return sum(data.ptdf[line, n] * m.NetInjection[s, n, hour] for n in m.N)

    model.line_upper_bound = pyo.Constraint(
        model.S,
        model.L,
        model.T,
        rule=lambda m, s, line, hour: flow(m, s, line, hour)
        <= scenario_data[s].line_limit[line],
    )
    model.line_lower_bound = pyo.Constraint(
        model.S,
        model.L,
        model.T,
        rule=lambda m, s, line, hour: flow(m, s, line, hour)
        >= -scenario_data[s].line_limit[line],
    )
    model.charge_power_bound = pyo.Constraint(
        model.S,
        model.N,
        model.T,
        rule=lambda m, s, n, t: m.P_charge[s, n, t] <= m.X_power[n],
    )
    model.discharge_power_bound = pyo.Constraint(
        model.S,
        model.N,
        model.T,
        rule=lambda m, s, n, t: m.P_discharge[s, n, t] <= m.X_power[n],
    )
    model.soc_transition = pyo.Constraint(
        model.S,
        model.N,
        model.T,
        rule=lambda m, s, n, t: m.SOC[s, n, t]
        == m.SOC[s, n, t - 1]
        + base.eta * m.P_charge[s, n, t]
        - m.P_discharge[s, n, t] / base.eta,
    )
    model.soc_capacity_bound = pyo.Constraint(
        model.S,
        model.N,
        model.T_SOC,
        rule=lambda m, s, n, tau: m.SOC[s, n, tau] <= m.X_energy[n],
    )
    model.soc_periodicity = pyo.Constraint(
        model.S,
        model.N,
        rule=lambda m, s, n: m.SOC[s, n, 0] == m.SOC[s, n, last_hour],
    )

    model.gen_stationarity = pyo.Constraint(
        model.SGT,
        rule=lambda m, s, g, t: m.lam[s, gen_node[g], t] + m.nu_gen[s, g, t]
        <= scenario_data[s].generation_cost[g],
    )
    model.charge_stationarity = pyo.Constraint(
        model.S,
        model.N,
        model.T,
        rule=lambda m, s, n, t: -m.lam[s, n, t]
        + m.rho_ch[s, n, t]
        - base.eta * m.gam[s, n, t]
        <= 0.5 * DEFAULT_DEGRADATION_EUR_PER_MWH,
    )
    model.discharge_stationarity = pyo.Constraint(
        model.S,
        model.N,
        model.T,
        rule=lambda m, s, n, t: m.lam[s, n, t]
        + m.sig_dis[s, n, t]
        + m.gam[s, n, t] / base.eta
        <= 0.5 * DEFAULT_DEGRADATION_EUR_PER_MWH,
    )
    model.netinjection_stationarity = pyo.Constraint(
        model.S,
        model.N,
        model.T,
        rule=lambda m, s, n, t: -m.lam[s, n, t]
        + m.lam_sys[s, t]
        + sum(
            scenario_data[s].ptdf[line, n]
            * (m.mu_up[s, line, t] + m.mu_dn[s, line, t])
            for line in m.L
        )
        == 0.0,
    )

    def soc_stationarity(m, s, n, tau):
        expression = m.del_soc[s, n, tau]
        if tau in m.T:
            expression += m.gam[s, n, tau]
        if (tau + 1) in m.T:
            expression -= m.gam[s, n, tau + 1]
        if tau == 0:
            expression += m.rho_per[s, n]
        if tau == last_hour:
            expression -= m.rho_per[s, n]
        return expression <= 0.0

    model.soc_stationarity = pyo.Constraint(
        model.S, model.N, model.T_SOC, rule=soc_stationarity
    )

    model.scenario_primal_objective = pyo.Expression(
        model.S,
        rule=lambda m, s: sum(
            scenario_data[s].generation_cost[g] * m.P_gen[s, g, t]
            for ss, g, t in m.SGT
            if ss == s
        )
        + 0.5
        * DEFAULT_DEGRADATION_EUR_PER_MWH
        * sum(
            m.P_charge[s, n, t] + m.P_discharge[s, n, t]
            for n in m.N
            for t in m.T
        ),
    )
    model.scenario_dual_objective = pyo.Expression(
        model.S,
        rule=lambda m, s: sum(
            scenario_data[s].demand_el[n, t] * m.lam[s, n, t]
            for n in m.N
            for t in m.T
        )
        + sum(
            scenario_data[s].generation_capacity[g, t] * m.nu_gen[s, g, t]
            for ss, g, t in m.SGT
            if ss == s
        )
        + sum(
            scenario_data[s].line_limit[line]
            * (m.mu_up[s, line, t] - m.mu_dn[s, line, t])
            for line in m.L
            for t in m.T
        )
        + sum(
            m.X_power[n] * (m.rho_ch[s, n, t] + m.sig_dis[s, n, t])
            for n in m.N
            for t in m.T
        )
        + sum(
            m.X_energy[n] * m.del_soc[s, n, tau]
            for n in m.N
            for tau in m.T_SOC
        ),
    )
    model.strong_duality = pyo.Constraint(
        model.S,
        rule=lambda m, s: m.scenario_primal_objective[s]
        == m.scenario_dual_objective[s],
    )

    model.scenario_spot_revenue = pyo.Expression(
        model.S,
        rule=lambda m, s: sum(
            m.lam[s, n, t]
            * (m.P_discharge[s, n, t] - m.P_charge[s, n, t])
            for n in m.N
            for t in m.T
        ),
    )
    model.scenario_degradation_cost = pyo.Expression(
        model.S,
        rule=lambda m, s: 0.5
        * DEFAULT_DEGRADATION_EUR_PER_MWH
        * sum(
            m.P_charge[s, n, t] + m.P_discharge[s, n, t]
            for n in m.N
            for t in m.T
        ),
    )
    model.expected_operating_profit = pyo.Expression(
        expr=sum(
            probabilities[s]
            * (model.scenario_spot_revenue[s] - model.scenario_degradation_cost[s])
            for s in model.S
        )
    )
    model.capex_daily = pyo.Expression(
        expr=daily_crf
        * sum(
            DEFAULT_BESS_COST_POWER_EUR_PER_MW * model.X_power[n]
            + DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH * model.X_energy[n]
            for n in model.N
        )
    )
    model.investor_profit = pyo.Expression(
        expr=model.expected_operating_profit - model.capex_daily
    )
    model.objective = pyo.Objective(expr=model.investor_profit, sense=pyo.maximize)

    model._base_data = base
    model._scenario_data = scenario_data
    model._probabilities = probabilities
    model._scenario_generation_pairs = scenario_generation_pairs
    model._gen_node = gen_node
    model._node_limit_mw = node_limit_mw
    model._wacc = wacc
    model._price_bound_eur_per_mwh = price_bound
    model._dual_bound_eur_per_mwh = dual_bound
    return model


def fleet_market_data(data: MarketData, power: dict[str, float], energy: dict[str, float]) -> MarketData:
    return replace(
        data,
        storage_units=[INVESTOR_ID],
        x_power={(INVESTOR_ID, n): power[n] for n in data.nodes},
        x_energy={(INVESTOR_ID, n): energy[n] for n in data.nodes},
    )


def initialize_from_reference_markets(model) -> None:
    power = {str(n): value(model.X_power[n]) for n in model.N}
    energy = {str(n): value(model.X_energy[n]) for n in model.N}
    solver = get_solver("highs")
    for scenario in model.S:
        data = fleet_market_data(model._scenario_data[scenario], power, energy)
        reference = build_fixed_demand_primal_model(
            data,
            storage_degradation_eur_per_mwh={
                INVESTOR_ID: DEFAULT_DEGRADATION_EUR_PER_MWH
            },
            dispatch_regularization_eur_per_mw2h=0.0,
        )
        result = solver.solve(reference)
        if str(result.solver.termination_condition) != "optimal":
            continue
        prices = fixed_demand_reference_lambda(reference)
        for g, t in model._scenario_generation_pairs[scenario]:
            model.P_gen[scenario, g, t].set_value(
                max(0.0, value(reference.P_gen[g, t]))
            )
            model.nu_gen[scenario, g, t].set_value(
                min(0.0, data.generation_cost[g] - prices[model._gen_node[g], t])
            )
        for n in model.N:
            for t in model.T:
                model.P_charge[scenario, n, t].set_value(
                    max(0.0, value(reference.P_charge[INVESTOR_ID, n, t]))
                )
                model.P_discharge[scenario, n, t].set_value(
                    max(0.0, value(reference.P_discharge[INVESTOR_ID, n, t]))
                )
                model.NetInjection[scenario, n, t].set_value(
                    value(reference.NetInjection[n, t])
                )
                model.lam[scenario, n, t].set_value(prices[n, t])
                model.gam[scenario, n, t].set_value(-prices[n, t])
            for tau in model.T_SOC:
                model.SOC[scenario, n, tau].set_value(
                    max(0.0, value(reference.SOC[INVESTOR_ID, n, tau]))
                )
        for t in model.T:
            model.lam_sys[scenario, t].set_value(
                sum(prices[n, t] for n in model.N) / len(model.N)
            )


def reference_settlement(model):
    power = {str(n): value(model.X_power[n]) for n in model.N}
    energy = {str(n): value(model.X_energy[n]) for n in model.N}
    solver = get_solver("highs")
    scenario_rows = []
    price_rows = []
    operation_rows = []
    expected_operating_profit = 0.0
    for scenario in model.S:
        data = fleet_market_data(model._scenario_data[scenario], power, energy)
        reference = build_fixed_demand_primal_model(
            data,
            storage_degradation_eur_per_mwh={
                INVESTOR_ID: DEFAULT_DEGRADATION_EUR_PER_MWH
            },
            dispatch_regularization_eur_per_mw2h=0.0,
        )
        result = solver.solve(reference)
        termination = str(result.solver.termination_condition)
        if termination != "optimal":
            raise RuntimeError(
                f"Reference market for scenario {scenario} terminated {termination}."
            )
        prices = fixed_demand_reference_lambda(reference)
        revenue = sum(
            prices[n, t]
            * (
                value(reference.P_discharge[INVESTOR_ID, n, t])
                - value(reference.P_charge[INVESTOR_ID, n, t])
            )
            for n in reference.N
            for t in reference.T
        )
        degradation = 0.5 * DEFAULT_DEGRADATION_EUR_PER_MWH * sum(
            value(reference.P_charge[INVESTOR_ID, n, t])
            + value(reference.P_discharge[INVESTOR_ID, n, t])
            for n in reference.N
            for t in reference.T
        )
        operating_profit = revenue - degradation
        expected_operating_profit += model._probabilities[scenario] * operating_profit
        embedded_prices = {
            (str(n), int(t)): value(model.lam[scenario, n, t])
            for n in model.N
            for t in model.T
        }
        max_price_difference = max(
            abs(embedded_prices[str(n), int(t)] - prices[n, t])
            for n in reference.N
            for t in reference.T
        )
        max_charge_difference = max(
            abs(
                value(model.P_charge[scenario, n, t])
                - value(reference.P_charge[INVESTOR_ID, n, t])
            )
            for n in reference.N
            for t in reference.T
        )
        max_discharge_difference = max(
            abs(
                value(model.P_discharge[scenario, n, t])
                - value(reference.P_discharge[INVESTOR_ID, n, t])
            )
            for n in reference.N
            for t in reference.T
        )
        scenario_rows.append(
            {
                "scenario": str(scenario),
                "probability": model._probabilities[scenario],
                "embedded_spot_revenue_eur_per_day": value(
                    model.scenario_spot_revenue[scenario]
                ),
                "embedded_degradation_eur_per_day": value(
                    model.scenario_degradation_cost[scenario]
                ),
                "embedded_operating_profit_eur_per_day": value(
                    model.scenario_spot_revenue[scenario]
                    - model.scenario_degradation_cost[scenario]
                ),
                "reference_spot_revenue_eur_per_day": revenue,
                "reference_degradation_eur_per_day": degradation,
                "reference_operating_profit_eur_per_day": operating_profit,
                "max_embedded_vs_reference_lmp_diff_eur_per_mwh": max_price_difference,
                "max_embedded_vs_reference_charge_diff_mw": max_charge_difference,
                "max_embedded_vs_reference_discharge_diff_mw": max_discharge_difference,
                "embedded_lower_level_objective_eur_per_day": value(
                    model.scenario_primal_objective[scenario]
                ),
                "reference_lower_level_objective_eur_per_day": value(
                    reference.objective
                ),
                "strong_duality_gap_eur_per_day": abs(
                    value(model.scenario_primal_objective[scenario])
                    - value(model.scenario_dual_objective[scenario])
                ),
            }
        )
        price_rows.extend(
            {
                "scenario": str(scenario),
                "hour": int(t),
                "node": str(n),
                "embedded_lambda_eur_per_mwh": embedded_prices[str(n), int(t)],
                "reference_lambda_eur_per_mwh": prices[n, t],
            }
            for t in reference.T
            for n in reference.N
        )
        operation_rows.extend(
            {
                "scenario": str(scenario),
                "hour": int(t),
                "node": str(n),
                "embedded_charge_mw": value(model.P_charge[scenario, n, t]),
                "embedded_discharge_mw": value(model.P_discharge[scenario, n, t]),
                "reference_charge_mw": value(
                    reference.P_charge[INVESTOR_ID, n, t]
                ),
                "reference_discharge_mw": value(
                    reference.P_discharge[INVESTOR_ID, n, t]
                ),
            }
            for t in reference.T
            for n in reference.N
        )
    return expected_operating_profit, scenario_rows, price_rows, operation_rows


def diagnostics(model) -> dict[str, float]:
    return {
        "max_strong_duality_gap_eur_per_day": max(
            abs(
                value(model.scenario_primal_objective[s])
                - value(model.scenario_dual_objective[s])
            )
            for s in model.S
        ),
        "max_nodal_balance_residual_mw": max(
            abs(value(model.nodal_balance[s, n, t].body))
            for s in model.S
            for n in model.N
            for t in model.T
        ),
        "max_system_balance_residual_mw": max(
            abs(sum(value(model.NetInjection[s, n, t]) for n in model.N))
            for s in model.S
            for t in model.T
        ),
    }


def export_results(
    output_dir: Path,
    model,
    summary: dict,
    scenario_rows: list[dict],
    price_rows: list[dict],
    operation_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        output_dir / "capacities.csv",
        ["node", "x_power_mw", "x_energy_mwh", "ratio_hours"],
        (
            {
                "node": str(n),
                "x_power_mw": value(model.X_power[n]),
                "x_energy_mwh": value(model.X_energy[n]),
                "ratio_hours": (
                    value(model.X_energy[n]) / value(model.X_power[n])
                    if value(model.X_power[n]) > 1.0e-9
                    else 0.0
                ),
            }
            for n in model.N
        ),
    )
    write_csv(output_dir / "scenario_summary.csv", list(scenario_rows[0]), scenario_rows)
    write_csv(output_dir / "node_hour_prices.csv", list(price_rows[0]), price_rows)
    write_csv(output_dir / "storage_operation.csv", list(operation_rows[0]), operation_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--only-scenario", default=None)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--wacc", type=float, default=DEFAULT_WACC)
    parser.add_argument("--initial-capacities", type=Path, default=DEFAULT_INITIAL_CAPACITIES)
    parser.add_argument("--initial-power-mw", type=float, default=10.0)
    parser.add_argument("--initial-ratio-hours", type=float, default=2.0)
    parser.add_argument("--price-bound-eur-per-mwh", type=float, default=DEFAULT_PRICE_BOUND_EUR_PER_MWH)
    parser.add_argument("--dual-bound-eur-per-mwh", type=float, default=DEFAULT_DUAL_BOUND_EUR_PER_MWH)
    parser.add_argument("--solver-tol", type=float, default=DEFAULT_SOLVER_TOL)
    parser.add_argument("--max-cpu-time", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, scenarios = load_scenarios(args.scenarios, args.only_scenario)
    base = load_market_data(args.data)
    initial_power, initial_energy = load_initial_fleet(
        args.initial_capacities,
        base.nodes,
        fallback_power_mw=args.initial_power_mw,
        fallback_ratio_hours=args.initial_ratio_hours,
    )
    model = build_stochastic_mpec(
        base,
        scenarios,
        node_limit_mw=args.node_limit_mw,
        initial_power_by_node=initial_power,
        initial_energy_by_node=initial_energy,
        wacc=args.wacc,
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
    )
    initialize_from_reference_markets(model)
    print(
        f"Stochastic merchant MPEC: {len(scenarios)} scenarios, "
        f"initial fleet {sum(initial_power.values()):.2f} MW / "
        f"{sum(initial_energy.values()):.2f} MWh"
    )
    result = get_ipopt_solver(
        {
            "max_cpu_time": args.max_cpu_time,
            "tol": args.solver_tol,
            "acceptable_tol": args.solver_tol,
        }
    ).solve(model, tee=args.tee)
    termination = str(result.solver.termination_condition)
    print(f"Solver termination: {termination}")
    if termination != "optimal":
        print("The stochastic MPEC did not terminate optimally; no result was exported.")
        return 1

    reference_expected_operating_profit, scenario_rows, price_rows, operation_rows = (
        reference_settlement(model)
    )
    diagnostic = diagnostics(model)
    capex = value(model.capex_daily)
    summary = {
        "model": "capacity-only stochastic single-investor optimistic MPEC",
        "termination": termination,
        "scenarios": [s.__dict__ for s in scenarios],
        "node_limit_mw": args.node_limit_mw,
        "wacc": args.wacc,
        "price_bound_eur_per_mwh": args.price_bound_eur_per_mwh,
        "dual_bound_eur_per_mwh": args.dual_bound_eur_per_mwh,
        "solver_tol": args.solver_tol,
        "initial_capacity_source": (
            str(args.initial_capacities)
            if args.initial_capacities is not None
            else "uniform numerical seed"
        ),
        "initial_total_power_mw": sum(initial_power.values()),
        "initial_total_energy_mwh": sum(initial_energy.values()),
        "total_power_mw": sum(value(model.X_power[n]) for n in model.N),
        "total_energy_mwh": sum(value(model.X_energy[n]) for n in model.N),
        "capex_eur_per_day": capex,
        "optimistic_expected_operating_profit_eur_per_day": value(
            model.expected_operating_profit
        ),
        "optimistic_expected_profit_eur_per_day": value(model.investor_profit),
        "reference_expected_operating_profit_eur_per_day": reference_expected_operating_profit,
        "reference_expected_profit_eur_per_day": reference_expected_operating_profit
        - capex,
        **diagnostic,
    }
    print(
        f"  BESS={summary['total_power_mw']:.2f} MW / "
        f"{summary['total_energy_mwh']:.2f} MWh"
    )
    print(
        f"  optimistic expected profit={summary['optimistic_expected_profit_eur_per_day']:,.2f} EUR/day"
    )
    print(
        f"  reference-price expected profit={summary['reference_expected_profit_eur_per_day']:,.2f} EUR/day"
    )
    print(
        f"  max strong-duality gap={summary['max_strong_duality_gap_eur_per_day']:.3e}; "
        f"max balance residual={max(summary['max_nodal_balance_residual_mw'], summary['max_system_balance_residual_mw']):.3e} MW"
    )
    if not args.no_export:
        export_results(
            args.output_dir,
            model,
            summary,
            scenario_rows,
            price_rows,
            operation_rows,
        )
        print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
