"""Illustrative renewable-scenario LLP and stochastic central-planner experiment.

The deterministic market data and maintained models are not modified. The LLP
is cleared independently in every scenario with the fixed storage fleet stored
in the input JSON. The stochastic planner chooses one common nodal MW/MWh fleet
before renewable availability is known, then dispatches independently in every
scenario and minimizes expected daily resource cost plus daily storage CAPEX.

This is two-stage investment/dispatch uncertainty with perfect information at
dispatch. It is not a multistage forecast-error, balancing, or reserve model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyomo.environ as pyo


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
PRIMAL_DUAL_DIR = MODEL_DIR / "Primal and dual problems"
for candidate in (MODEL_DIR, PRIMAL_DUAL_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from primal_market_clearing_model import MarketData, load_market_data  # noqa: E402
from single_investor_mpec import (  # noqa: E402
    DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH,
    DEFAULT_BESS_COST_POWER_EUR_PER_MW,
    DEFAULT_DEGRADATION_EUR_PER_MWH,
    DEFAULT_LIFETIME_YEARS,
    DEFAULT_RATIO_MAX,
    DEFAULT_RATIO_MIN,
    DEFAULT_WACC,
    build_fixed_demand_primal_model,
    capital_recovery_factor,
    fixed_demand_reference_lambda,
)


DEFAULT_DATA_PATH = (
    MODEL_DIR / "data" / "processed" / "market_data_IEEE_9Bus_distributed_congestion.json"
)
DEFAULT_SCENARIO_PATH = SCRIPT_DIR / "scenarios.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "default"
DEFAULT_NODE_LIMIT_MW = 200.0
CORRIDOR_LINES = ("L46", "L69", "L98", "L78")


@dataclass(frozen=True)
class RenewableScenario:
    name: str
    probability: float
    pv_scale: float
    wind_scale: float


def value(component) -> float:
    return float(pyo.value(component))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_scenarios(path: Path, only_scenario: str | None = None) -> tuple[str, tuple[RenewableScenario, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = tuple(
        RenewableScenario(
            name=str(row["name"]),
            probability=float(row["probability"]),
            pv_scale=float(row["pv_scale"]),
            wind_scale=float(row["wind_scale"]),
        )
        for row in raw["scenarios"]
    )
    if only_scenario is not None:
        scenarios = tuple(s for s in scenarios if s.name == only_scenario)
        if not scenarios:
            raise ValueError(f"Unknown scenario {only_scenario!r}.")
        scenarios = tuple(replace(s, probability=1.0) for s in scenarios)
    names = [s.name for s in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("Scenario names must be unique.")
    if any(s.probability <= 0.0 for s in scenarios):
        raise ValueError("Scenario probabilities must be positive.")
    if any(s.pv_scale < 0.0 or s.wind_scale < 0.0 for s in scenarios):
        raise ValueError("Renewable scale factors must be non-negative.")
    if abs(sum(s.probability for s in scenarios) - 1.0) > 1.0e-9:
        raise ValueError("Scenario probabilities must sum to one.")
    return str(raw.get("description", "")), scenarios


def scenario_market_data(base: MarketData, scenario: RenewableScenario) -> MarketData:
    generation_capacity = dict(base.generation_capacity)
    for generator in base.generators:
        upper = str(generator).upper()
        if "PV" in upper:
            scale = scenario.pv_scale
        elif "WIND" in upper:
            scale = scenario.wind_scale
        else:
            continue
        for hour in base.times:
            generation_capacity[generator, hour] *= scale
    return replace(base, generation_capacity=generation_capacity)


def get_solver(name: str):
    solver = pyo.SolverFactory(name)
    if not solver.available(exception_flag=False):
        raise RuntimeError(f"Requested solver {name!r} is unavailable.")
    return solver


def renewable_generators(data: MarketData) -> tuple[str, ...]:
    return tuple(g for g in data.generators if data.generation_cost[g] == 0.0)


def line_dual_magnitudes(model, probability: float = 1.0, scenario: str | None = None):
    for line in model.L:
        for hour in model.T:
            if scenario is None:
                upper = model.line_upper_bound[line, hour]
                lower = model.line_lower_bound[line, hour]
            else:
                upper = model.line_upper_bound[scenario, line, hour]
                lower = model.line_lower_bound[scenario, line, hour]
            up = abs(float(model.dual.get(upper, 0.0))) / probability
            down = abs(float(model.dual.get(lower, 0.0))) / probability
            yield line, int(hour), up, down


def run_llp_scenarios(
    base: MarketData,
    scenarios: Sequence[RenewableScenario],
    solver_name: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    solver = get_solver(solver_name)
    summaries: list[dict] = []
    prices: list[dict] = []
    line_duals: list[dict] = []
    dispatch: list[dict] = []
    for scenario in scenarios:
        data = scenario_market_data(base, scenario)
        model = build_fixed_demand_primal_model(
            data,
            dispatch_regularization_eur_per_mw2h=0.0,
        )
        result = solver.solve(model)
        termination = str(result.solver.termination_condition)
        if termination != "optimal":
            raise RuntimeError(f"LLP scenario {scenario.name} terminated {termination}.")
        lam = fixed_demand_reference_lambda(model)
        renewables = renewable_generators(data)
        curtailment = sum(
            max(0.0, data.generation_capacity[g, t] - value(model.P_gen[g, t]))
            for g in renewables
            for t in model.T
        )
        hourly_spreads = [
            max(lam[n, t] for n in model.N) - min(lam[n, t] for n in model.N)
            for t in model.T
        ]
        scenario_line_duals = list(line_dual_magnitudes(model))
        summaries.append(
            {
                "scenario": scenario.name,
                "probability": scenario.probability,
                "pv_scale": scenario.pv_scale,
                "wind_scale": scenario.wind_scale,
                "operating_cost_eur_per_day": value(model.objective),
                "renewable_curtailment_mwh": curtailment,
                "lambda_min_eur_per_mwh": min(lam.values()),
                "lambda_max_eur_per_mwh": max(lam.values()),
                "max_hourly_nodal_spread_eur_per_mwh": max(hourly_spreads),
                "positive_corridor_dual_hours": sum(
                    max(up, down) > 1.0e-6
                    for line, _, up, down in scenario_line_duals
                    if line in CORRIDOR_LINES
                ),
            }
        )
        prices.extend(
            {
                "scenario": scenario.name,
                "probability": scenario.probability,
                "hour": int(t),
                "node": str(n),
                "lambda_eur_per_mwh": lam[n, t],
            }
            for t in model.T
            for n in model.N
        )
        line_duals.extend(
            {
                "scenario": scenario.name,
                "hour": hour,
                "line": line,
                "upper_dual_abs": up,
                "lower_dual_abs": down,
            }
            for line, hour, up, down in scenario_line_duals
        )
        dispatch.extend(
            {
                "scenario": scenario.name,
                "hour": int(t),
                "generator": str(g),
                "available_mw": data.generation_capacity[g, t],
                "dispatch_mw": value(model.P_gen[g, t]),
                "curtailment_mw": (
                    max(0.0, data.generation_capacity[g, t] - value(model.P_gen[g, t]))
                    if g in renewables
                    else 0.0
                ),
            }
            for t in model.T
            for g in model.G
        )
    return summaries, prices, line_duals, dispatch


def build_stochastic_planner(
    base: MarketData,
    scenarios: Sequence[RenewableScenario],
    *,
    node_limit_mw: float,
    wacc: float,
) -> pyo.ConcreteModel:
    if node_limit_mw <= 0.0:
        raise ValueError("node_limit_mw must be positive.")
    if not 0.0 <= wacc < 1.0:
        raise ValueError("wacc must be in [0, 1).")
    data_by_scenario = {s.name: scenario_market_data(base, s) for s in scenarios}
    probabilities = {s.name: s.probability for s in scenarios}
    last_hour = max(base.times)
    daily_crf = capital_recovery_factor(wacc, DEFAULT_LIFETIME_YEARS) / 365.25

    model = pyo.ConcreteModel(name="Two-stage stochastic BESS central planner")
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.S = pyo.Set(initialize=[s.name for s in scenarios], ordered=True)
    model.N = pyo.Set(initialize=base.nodes, ordered=True)
    model.G = pyo.Set(initialize=base.generators, ordered=True)
    model.T = pyo.Set(initialize=base.times, ordered=True)
    model.T_SOC = pyo.Set(initialize=base.soc_times, ordered=True)
    model.L = pyo.Set(initialize=base.lines, ordered=True)

    model.X_power = pyo.Var(model.N, bounds=(0.0, node_limit_mw), initialize=0.0)
    model.X_energy = pyo.Var(
        model.N,
        bounds=(0.0, DEFAULT_RATIO_MAX * node_limit_mw),
        initialize=0.0,
    )
    model.energy_ratio_min = pyo.Constraint(
        model.N,
        rule=lambda m, n: m.X_energy[n] >= DEFAULT_RATIO_MIN * m.X_power[n],
    )
    model.energy_ratio_max = pyo.Constraint(
        model.N,
        rule=lambda m, n: m.X_energy[n] <= DEFAULT_RATIO_MAX * m.X_power[n],
    )

    model.P_gen = pyo.Var(model.S, model.G, model.T, domain=pyo.NonNegativeReals)
    model.P_charge = pyo.Var(model.S, model.N, model.T, domain=pyo.NonNegativeReals)
    model.P_discharge = pyo.Var(model.S, model.N, model.T, domain=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.S, model.N, model.T_SOC, domain=pyo.NonNegativeReals)
    model.NetInjection = pyo.Var(model.S, model.N, model.T, domain=pyo.Reals)

    def nodal_balance(m, s, n, t):
        data = data_by_scenario[s]
        return (
            sum(m.P_gen[s, g, t] for g in data.generators_at_node.get(n, []))
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
        model.S,
        model.G,
        model.T,
        rule=lambda m, s, g, t: m.P_gen[s, g, t]
        <= data_by_scenario[s].generation_capacity[g, t],
    )

    def flow(m, s, line, hour):
        data = data_by_scenario[s]
        return sum(data.ptdf[line, n] * m.NetInjection[s, n, hour] for n in m.N)

    model.line_upper_bound = pyo.Constraint(
        model.S,
        model.L,
        model.T,
        rule=lambda m, s, line, hour: flow(m, s, line, hour)
        <= data_by_scenario[s].line_limit[line],
    )
    model.line_lower_bound = pyo.Constraint(
        model.S,
        model.L,
        model.T,
        rule=lambda m, s, line, hour: flow(m, s, line, hour)
        >= -data_by_scenario[s].line_limit[line],
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

    model.scenario_generation_cost = pyo.Expression(
        model.S,
        rule=lambda m, s: sum(
            data_by_scenario[s].generation_cost[g] * m.P_gen[s, g, t]
            for g in m.G
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
    model.expected_operating_cost = pyo.Expression(
        expr=sum(
            probabilities[s]
            * (model.scenario_generation_cost[s] + model.scenario_degradation_cost[s])
            for s in model.S
        )
    )
    model.storage_capex = pyo.Expression(
        expr=daily_crf
        * sum(
            DEFAULT_BESS_COST_POWER_EUR_PER_MW * model.X_power[n]
            + DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH * model.X_energy[n]
            for n in model.N
        )
    )
    model.objective = pyo.Objective(
        expr=model.expected_operating_cost + model.storage_capex,
        sense=pyo.minimize,
    )
    model._base_data = base
    model._scenario_data = data_by_scenario
    model._probabilities = probabilities
    model._node_limit_mw = node_limit_mw
    model._wacc = wacc
    return model


def stochastic_prices(model) -> dict[tuple[str, str, int], float]:
    raw = {
        (s, n, int(t)): float(model.dual[model.nodal_balance[s, n, t]])
        / model._probabilities[s]
        for s in model.S
        for n in model.N
        for t in model.T
    }
    sign = 1.0 if sum(raw.values()) >= 0.0 else -1.0
    return {key: sign * dual for key, dual in raw.items()}


def solve_stochastic_planner(
    base: MarketData,
    scenarios: Sequence[RenewableScenario],
    *,
    node_limit_mw: float,
    wacc: float,
    solver_name: str,
):
    model = build_stochastic_planner(
        base,
        scenarios,
        node_limit_mw=node_limit_mw,
        wacc=wacc,
    )
    result = get_solver(solver_name).solve(model)
    termination = str(result.solver.termination_condition)
    if termination != "optimal":
        raise RuntimeError(f"Stochastic planner terminated {termination}.")
    return model, stochastic_prices(model)


def planner_results(model, prices):
    capacities = []
    for node in model.N:
        power = value(model.X_power[node])
        energy = value(model.X_energy[node])
        capacities.append(
            {
                "node": str(node),
                "x_power_mw": power,
                "x_energy_mwh": energy,
                "ratio_hours": energy / power if power > 1.0e-9 else 0.0,
                "share_of_node_limit": power / model._node_limit_mw,
            }
        )
    scenario_rows = []
    for scenario in model.S:
        data = model._scenario_data[scenario]
        renewables = renewable_generators(data)
        scenario_line_duals = list(
            line_dual_magnitudes(
                model,
                probability=model._probabilities[scenario],
                scenario=str(scenario),
            )
        )
        curtailment = sum(
            max(
                0.0,
                data.generation_capacity[g, t] - value(model.P_gen[scenario, g, t]),
            )
            for g in renewables
            for t in model.T
        )
        scenario_prices = [prices[scenario, n, int(t)] for n in model.N for t in model.T]
        scenario_rows.append(
            {
                "scenario": str(scenario),
                "probability": model._probabilities[scenario],
                "generation_cost_eur_per_day": value(model.scenario_generation_cost[scenario]),
                "degradation_cost_eur_per_day": value(model.scenario_degradation_cost[scenario]),
                "renewable_curtailment_mwh": curtailment,
                "lambda_min_eur_per_mwh": min(scenario_prices),
                "lambda_max_eur_per_mwh": max(scenario_prices),
                "max_hourly_nodal_spread_eur_per_mwh": max(
                    max(prices[scenario, n, int(t)] for n in model.N)
                    - min(prices[scenario, n, int(t)] for n in model.N)
                    for t in model.T
                ),
                "positive_corridor_dual_hours": sum(
                    max(up, down) > 1.0e-6
                    for line, _, up, down in scenario_line_duals
                    if line in CORRIDOR_LINES
                ),
                "max_corridor_dual_abs": max(
                    (
                        max(up, down)
                        for line, _, up, down in scenario_line_duals
                        if line in CORRIDOR_LINES
                    ),
                    default=0.0,
                ),
            }
        )
    max_system_balance_residual = max(
        abs(sum(value(model.NetInjection[s, n, t]) for n in model.N))
        for s in model.S
        for t in model.T
    )
    max_nodal_balance_residual = max(
        abs(value(model.nodal_balance[s, n, t].body))
        for s in model.S
        for n in model.N
        for t in model.T
    )
    summary = {
        "model": "two-stage stochastic central planner with perfect-information dispatch recourse",
        "wacc": model._wacc,
        "node_limit_mw": model._node_limit_mw,
        "total_power_mw": sum(row["x_power_mw"] for row in capacities),
        "total_energy_mwh": sum(row["x_energy_mwh"] for row in capacities),
        "expected_operating_cost_eur_per_day": value(model.expected_operating_cost),
        "storage_capex_eur_per_day": value(model.storage_capex),
        "expected_system_cost_eur_per_day": value(model.objective),
        "expected_renewable_curtailment_mwh": sum(
            row["probability"] * row["renewable_curtailment_mwh"]
            for row in scenario_rows
        ),
        "max_system_balance_residual_mw": max_system_balance_residual,
        "max_nodal_balance_residual_mw": max_nodal_balance_residual,
    }
    return summary, capacities, scenario_rows


def export_results(
    output_dir: Path,
    *,
    config: dict,
    llp_summaries: list[dict],
    llp_prices: list[dict],
    llp_line_duals: list[dict],
    llp_dispatch: list[dict],
    independent_planners: list[dict],
    independent_planner_capacities: list[dict],
    planner_model,
    planner_prices,
    planner_summary: dict,
    planner_capacities: list[dict],
    planner_scenarios: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "stochastic_planner_summary.json").write_text(
        json.dumps(planner_summary, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "llp_summary.csv", list(llp_summaries[0]), llp_summaries)
    write_csv(output_dir / "llp_node_hour_prices.csv", list(llp_prices[0]), llp_prices)
    write_csv(output_dir / "llp_line_hour_duals.csv", list(llp_line_duals[0]), llp_line_duals)
    write_csv(output_dir / "llp_generation_dispatch.csv", list(llp_dispatch[0]), llp_dispatch)
    write_csv(
        output_dir / "independent_planner_summary.csv",
        list(independent_planners[0]),
        independent_planners,
    )
    write_csv(
        output_dir / "independent_planner_capacities.csv",
        list(independent_planner_capacities[0]),
        independent_planner_capacities,
    )
    write_csv(
        output_dir / "stochastic_planner_capacities.csv",
        list(planner_capacities[0]),
        planner_capacities,
    )
    write_csv(
        output_dir / "stochastic_planner_scenario_summary.csv",
        list(planner_scenarios[0]),
        planner_scenarios,
    )
    write_csv(
        output_dir / "stochastic_planner_node_hour_prices.csv",
        ["scenario", "probability", "hour", "node", "lambda_eur_per_mwh"],
        (
            {
                "scenario": str(s),
                "probability": planner_model._probabilities[s],
                "hour": int(t),
                "node": str(n),
                "lambda_eur_per_mwh": planner_prices[s, n, int(t)],
            }
            for s in planner_model.S
            for t in planner_model.T
            for n in planner_model.N
        ),
    )
    write_csv(
        output_dir / "stochastic_planner_storage_operation.csv",
        ["scenario", "probability", "hour", "node", "p_charge_mw", "p_discharge_mw", "soc_mwh"],
        (
            {
                "scenario": str(s),
                "probability": planner_model._probabilities[s],
                "hour": int(t),
                "node": str(n),
                "p_charge_mw": value(planner_model.P_charge[s, n, t]),
                "p_discharge_mw": value(planner_model.P_discharge[s, n, t]),
                "soc_mwh": value(planner_model.SOC[s, n, t]),
            }
            for s in planner_model.S
            for t in planner_model.T
            for n in planner_model.N
        ),
    )
    write_csv(
        output_dir / "stochastic_planner_line_hour_duals.csv",
        ["scenario", "probability", "hour", "line", "upper_dual_abs", "lower_dual_abs"],
        (
            {
                "scenario": str(s),
                "probability": planner_model._probabilities[s],
                "hour": hour,
                "line": line,
                "upper_dual_abs": up,
                "lower_dual_abs": down,
            }
            for s in planner_model.S
            for line, hour, up, down in line_dual_magnitudes(
                planner_model,
                probability=planner_model._probabilities[s],
                scenario=str(s),
            )
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--only-scenario", default=None)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--wacc", type=float, default=DEFAULT_WACC)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    description, scenarios = load_scenarios(args.scenarios, args.only_scenario)
    base = load_market_data(args.data)
    fixed_storage_mw = sum(base.x_power.values())
    fixed_storage_mwh = sum(base.x_energy.values())
    print(description)
    print(
        f"Fixed-fleet LLP: {len(scenarios)} scenarios; input fleet "
        f"{fixed_storage_mw:.3f} MW / {fixed_storage_mwh:.3f} MWh"
    )
    llp_summaries, llp_prices, llp_line_duals, llp_dispatch = run_llp_scenarios(
        base, scenarios, args.solver
    )
    for row in llp_summaries:
        print(
            f"  {row['scenario']}: cost={row['operating_cost_eur_per_day']:,.2f} EUR/day, "
            f"renewable curtailment={row['renewable_curtailment_mwh']:,.2f} MWh, "
            f"LMP=[{row['lambda_min_eur_per_mwh']:.2f}, {row['lambda_max_eur_per_mwh']:.2f}]"
        )
    expected_llp_cost = sum(
        row["probability"] * row["operating_cost_eur_per_day"] for row in llp_summaries
    )
    expected_llp_curtailment = sum(
        row["probability"] * row["renewable_curtailment_mwh"] for row in llp_summaries
    )
    print(
        f"  expected fixed-fleet cost={expected_llp_cost:,.2f} EUR/day; "
        f"expected curtailment={expected_llp_curtailment:,.2f} MWh/day"
    )

    independent_planners = []
    independent_planner_capacities = []
    print("Independent perfect-foresight planner benchmarks:")
    for scenario in scenarios:
        isolated = replace(scenario, probability=1.0)
        model, prices = solve_stochastic_planner(
            base,
            (isolated,),
            node_limit_mw=args.node_limit_mw,
            wacc=args.wacc,
            solver_name=args.solver,
        )
        summary, capacity_rows, scenario_rows = planner_results(model, prices)
        row = {
            "scenario": scenario.name,
            "pv_scale": scenario.pv_scale,
            "wind_scale": scenario.wind_scale,
            "total_power_mw": summary["total_power_mw"],
            "total_energy_mwh": summary["total_energy_mwh"],
            "system_cost_eur_per_day": summary["expected_system_cost_eur_per_day"],
            "renewable_curtailment_mwh": scenario_rows[0]["renewable_curtailment_mwh"],
        }
        independent_planners.append(row)
        independent_planner_capacities.extend(
            {"scenario": scenario.name, **capacity_row}
            for capacity_row in capacity_rows
        )
        print(
            f"  {scenario.name}: {row['total_power_mw']:.2f} MW / "
            f"{row['total_energy_mwh']:.2f} MWh; residual curtailment "
            f"{row['renewable_curtailment_mwh']:.2f} MWh"
        )

    print("Shared-investment stochastic planner:")
    planner_model, planner_prices = solve_stochastic_planner(
        base,
        scenarios,
        node_limit_mw=args.node_limit_mw,
        wacc=args.wacc,
        solver_name=args.solver,
    )
    planner_summary, planner_capacities, planner_scenarios = planner_results(
        planner_model, planner_prices
    )
    expected_perfect_foresight_cost = sum(
        scenario.probability
        * next(
            row["system_cost_eur_per_day"]
            for row in independent_planners
            if row["scenario"] == scenario.name
        )
        for scenario in scenarios
    )
    planner_summary.update(
        {
            "expected_no_storage_cost_eur_per_day": expected_llp_cost,
            "expected_no_storage_curtailment_mwh": expected_llp_curtailment,
            "expected_perfect_foresight_cost_eur_per_day": expected_perfect_foresight_cost,
            "stochastic_savings_vs_no_storage_eur_per_day": expected_llp_cost
            - planner_summary["expected_system_cost_eur_per_day"],
            "expected_value_of_perfect_information_eur_per_day": planner_summary[
                "expected_system_cost_eur_per_day"
            ]
            - expected_perfect_foresight_cost,
        }
    )
    print(
        f"  total BESS={planner_summary['total_power_mw']:.2f} MW / "
        f"{planner_summary['total_energy_mwh']:.2f} MWh"
    )
    print(
        f"  expected system cost={planner_summary['expected_system_cost_eur_per_day']:,.2f} EUR/day; "
        f"expected residual curtailment={planner_summary['expected_renewable_curtailment_mwh']:.2f} MWh/day"
    )
    print(
        f"  savings vs no storage={planner_summary['stochastic_savings_vs_no_storage_eur_per_day']:,.2f} EUR/day; "
        f"perfect-information gap={planner_summary['expected_value_of_perfect_information_eur_per_day']:,.2f} EUR/day"
    )
    for row in planner_capacities:
        if row["x_power_mw"] > 1.0e-4:
            print(
                f"  {row['node']}: {row['x_power_mw']:.2f} MW / "
                f"{row['x_energy_mwh']:.2f} MWh"
            )

    config = {
        "data_path": str(args.data),
        "scenario_path": str(args.scenarios),
        "scenario_description": description,
        "scenarios": [s.__dict__ for s in scenarios],
        "solver": args.solver,
        "node_limit_mw": args.node_limit_mw,
        "wacc": args.wacc,
        "fixed_demand": True,
        "dispatch_regularization_eur_per_mw2h": 0.0,
        "storage_recoursed_by_scenario": True,
        "investment_shared_across_scenarios": True,
        "perfect_information_at_dispatch": True,
    }
    if not args.no_export:
        export_results(
            args.output_dir,
            config=config,
            llp_summaries=llp_summaries,
            llp_prices=llp_prices,
            llp_line_duals=llp_line_duals,
            llp_dispatch=llp_dispatch,
            independent_planners=independent_planners,
            independent_planner_capacities=independent_planner_capacities,
            planner_model=planner_model,
            planner_prices=planner_prices,
            planner_summary=planner_summary,
            planner_capacities=planner_capacities,
            planner_scenarios=planner_scenarios,
        )
        print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
