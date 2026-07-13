"""Standalone Pyomo implementation of the primal spot-market LLP.

The model follows the deterministic lower-level market-clearing formulation in
Overleaf_Alex/model_extension.tex.

Default run:
    python primal_market_clearing_model.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import pyomo.environ as pyo


MODEL_NAME = "Primal Spot Market Clearing Model"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "market_data.json"


@dataclass(frozen=True)
class MarketData:
    nodes: Sequence[str]
    generators: Sequence[str]
    storage_units: Sequence[str]
    times: Sequence[int]
    soc_times: Sequence[int]
    lines: Sequence[str]
    generators_at_node: Mapping[str, Sequence[str]]
    generation_cost: Mapping[str, float]
    generation_capacity: Mapping[Tuple[str, int], float]
    demand_el: Mapping[Tuple[str, int], float]
    voll: float
    x_power: Mapping[Tuple[str, str], float]
    x_energy: Mapping[Tuple[str, str], float]
    line_limit: Mapping[str, float]
    ptdf: Mapping[Tuple[str, str], float]
    eta: float


def _tuple_map(records: Sequence[Mapping[str, Any]], key_fields: Sequence[str], value_field: str) -> dict[tuple[Any, ...], float]:
    return {
        tuple(record[field] for field in key_fields): float(record[value_field])
        for record in records
    }


def load_market_data(path: Path = DEFAULT_DATA_PATH) -> MarketData:
    """Load processed benchmark data produced by prepare_data.py."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    generation_capacity = _tuple_map(raw["generation_capacity"], ["generator", "hour"], "capacity_mw")
    demand_el = _tuple_map(raw["demand_el"], ["node", "hour"], "demand_mw")
    x_power = _tuple_map(raw["x_power"], ["storage_unit", "node"], "power_mw")
    x_energy = _tuple_map(raw["x_energy"], ["storage_unit", "node"], "energy_mwh")
    ptdf = _tuple_map(raw["ptdf"], ["line", "node"], "ptdf")

    return MarketData(
        nodes=[str(node) for node in raw["nodes"]],
        generators=[str(generator) for generator in raw["generators"]],
        storage_units=[str(unit) for unit in raw["storage_units"]],
        times=[int(hour) for hour in raw["times"]],
        soc_times=[int(hour) for hour in raw["soc_times"]],
        lines=[str(line) for line in raw["lines"]],
        generators_at_node={
            str(node): [str(generator) for generator in generators]
            for node, generators in raw["generators_at_node"].items()
        },
        generation_cost={str(generator): float(cost) for generator, cost in raw["generation_cost"].items()},
        generation_capacity=generation_capacity,
        demand_el=demand_el,
        voll=float(raw["voll"]),
        x_power=x_power,
        x_energy=x_energy,
        line_limit={str(line): float(limit) for line, limit in raw["line_limit"].items()},
        ptdf=ptdf,
        eta=float(raw["eta"]),
    )


def _validate_data(data: MarketData) -> None:
    if not 0.0 < data.eta <= 1.0:
        raise ValueError("eta must be in (0, 1].")
    for n in data.nodes:
        if n not in data.generators_at_node:
            raise ValueError(f"Missing generators_at_node entry for node {n}.")
    for t in data.times:
        for n in data.nodes:
            if (n, t) not in data.demand_el:
                raise ValueError(f"Missing demand_el for node {n}, hour {t}.")
        for g in data.generators:
            if (g, t) not in data.generation_capacity:
                raise ValueError(f"Missing generation_capacity for generator {g}, hour {t}.")
        for l in data.lines:
            if l not in data.line_limit:
                raise ValueError(f"Missing line limit for line {l}.")
            for n in data.nodes:
                if (l, n) not in data.ptdf:
                    raise ValueError(f"Missing PTDF for line {l}, node {n}.")
    for i in data.storage_units:
        for n in data.nodes:
            if (i, n) not in data.x_power:
                raise ValueError(f"Missing x_power for {i}, {n}.")
            if (i, n) not in data.x_energy:
                raise ValueError(f"Missing x_energy for {i}, {n}.")


def build_primal_market_clearing_model(data: MarketData) -> pyo.ConcreteModel:
    """Build the deterministic primal lower-level spot-market LP."""

    _validate_data(data)

    m = pyo.ConcreteModel(name=MODEL_NAME)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.G = pyo.Set(initialize=data.generators, ordered=True)
    m.I = pyo.Set(initialize=data.storage_units, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)

    m.P_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    m.P_shed = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)
    m.P_charge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals)
    m.P_discharge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals)
    m.SOC = pyo.Var(m.I, m.N, m.T_SOC, domain=pyo.NonNegativeReals)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals)

    def objective_rule(model: pyo.ConcreteModel) -> pyo.Expression:
        generation_cost = sum(
            data.generation_cost[g] * model.P_gen[g, t]
            for g in model.G
            for t in model.T
        )
        load_shed_cost = sum(
            data.voll * model.P_shed[n, t]
            for n in model.N
            for t in model.T
        )
        return generation_cost + load_shed_cost

    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def nodal_balance_rule(model: pyo.ConcreteModel, n: str, t: int) -> pyo.Expression:
        generators = data.generators_at_node.get(n, [])
        storage_net = sum(
            model.P_discharge[i, n, t] - model.P_charge[i, n, t]
            for i in model.I
        )
        return (
            sum(model.P_gen[g, t] for g in generators)
            + storage_net
            + model.P_shed[n, t]
            - data.demand_el[n, t]
            == model.NetInjection[n, t]
        )

    m.nodal_balance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)

    def system_balance_rule(model: pyo.ConcreteModel, t: int) -> pyo.Expression:
        return sum(model.NetInjection[n, t] for n in model.N) == 0.0

    m.system_balance = pyo.Constraint(m.T, rule=system_balance_rule)

    def generation_capacity_rule(model: pyo.ConcreteModel, g: str, t: int) -> pyo.Expression:
        return model.P_gen[g, t] <= data.generation_capacity[g, t]

    m.generation_capacity_bound = pyo.Constraint(m.G, m.T, rule=generation_capacity_rule)

    def line_upper_rule(model: pyo.ConcreteModel, l: str, t: int) -> pyo.Expression:
        flow = sum(data.ptdf[l, n] * model.NetInjection[n, t] for n in model.N)
        return flow <= data.line_limit[l]

    def line_lower_rule(model: pyo.ConcreteModel, l: str, t: int) -> pyo.Expression:
        flow = sum(data.ptdf[l, n] * model.NetInjection[n, t] for n in model.N)
        return flow >= -data.line_limit[l]

    m.line_upper_bound = pyo.Constraint(m.L, m.T, rule=line_upper_rule)
    m.line_lower_bound = pyo.Constraint(m.L, m.T, rule=line_lower_rule)

    def charge_power_rule(model: pyo.ConcreteModel, i: str, n: str, t: int) -> pyo.Expression:
        return model.P_charge[i, n, t] <= data.x_power[i, n]

    def discharge_power_rule(model: pyo.ConcreteModel, i: str, n: str, t: int) -> pyo.Expression:
        return model.P_discharge[i, n, t] <= data.x_power[i, n]

    m.charge_power_bound = pyo.Constraint(m.I, m.N, m.T, rule=charge_power_rule)
    m.discharge_power_bound = pyo.Constraint(m.I, m.N, m.T, rule=discharge_power_rule)

    def soc_transition_rule(model: pyo.ConcreteModel, i: str, n: str, t: int) -> pyo.Expression:
        return model.SOC[i, n, t] == (
            model.SOC[i, n, t - 1]
            + data.eta * model.P_charge[i, n, t]
            - model.P_discharge[i, n, t] / data.eta
        )

    m.soc_transition = pyo.Constraint(m.I, m.N, m.T, rule=soc_transition_rule)

    def soc_capacity_rule(model: pyo.ConcreteModel, i: str, n: str, tau: int) -> pyo.Expression:
        return model.SOC[i, n, tau] <= data.x_energy[i, n]

    m.soc_capacity_bound = pyo.Constraint(m.I, m.N, m.T_SOC, rule=soc_capacity_rule)

    last_t = max(data.times)

    def soc_periodicity_rule(model: pyo.ConcreteModel, i: str, n: str) -> pyo.Expression:
        return model.SOC[i, n, 0] == model.SOC[i, n, last_t]

    m.soc_periodicity = pyo.Constraint(m.I, m.N, rule=soc_periodicity_rule)

    def load_shed_bound_rule(model: pyo.ConcreteModel, n: str, t: int) -> pyo.Expression:
        return model.P_shed[n, t] <= data.demand_el[n, t]

    m.load_shed_bound = pyo.Constraint(m.N, m.T, rule=load_shed_bound_rule)

    m._market_data = data
    return m


def get_solver(preferred: str | None = None) -> Tuple[str, pyo.SolverFactory]:
    candidates = [preferred] if preferred else []
    candidates.extend(["appsi_highs", "highs", "glpk", "cbc"])
    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        solver = pyo.SolverFactory(name)
        if solver.available(exception_flag=False):
            return name, solver
    raise RuntimeError(
        "No LP solver is available. Install highspy, HiGHS, GLPK, or CBC. "
        "Example: python -m pip install highspy"
    )


def solve_model(model: pyo.ConcreteModel, solver_name: str | None = None) -> pyo.SolverResults:
    name, solver = get_solver(solver_name)
    print(f"Solving {model.name} with {name}...")
    return solver.solve(model, tee=False)


def value(obj: pyo.Var | pyo.Expression) -> float:
    return float(pyo.value(obj))


def max_abs(values: Iterable[float]) -> float:
    values = list(values)
    return max((abs(v) for v in values), default=0.0)


def run_sanity_checks(model: pyo.ConcreteModel) -> Dict[str, float]:
    data: MarketData = model._market_data
    checks: Dict[str, float] = {}

    checks["system_balance_max_abs_MW"] = max_abs(
        sum(value(model.NetInjection[n, t]) for n in model.N)
        for t in model.T
    )
    checks["charge_power_violation_MW"] = max(
        max(0.0, value(model.P_charge[i, n, t]) - data.x_power[i, n])
        for i in model.I
        for n in model.N
        for t in model.T
    )
    checks["discharge_power_violation_MW"] = max(
        max(0.0, value(model.P_discharge[i, n, t]) - data.x_power[i, n])
        for i in model.I
        for n in model.N
        for t in model.T
    )
    checks["soc_capacity_violation_MWh"] = max(
        max(0.0, value(model.SOC[i, n, tau]) - data.x_energy[i, n])
        for i in model.I
        for n in model.N
        for tau in model.T_SOC
    )
    checks["soc_periodicity_max_abs_MWh"] = max_abs(
        value(model.SOC[i, n, 0]) - value(model.SOC[i, n, max(data.times)])
        for i in model.I
        for n in model.N
    )
    checks["load_shed_bound_violation_MW"] = max(
        max(0.0, value(model.P_shed[n, t]) - data.demand_el[n, t])
        for n in model.N
        for t in model.T
    )
    return checks


def print_solution_summary(model: pyo.ConcreteModel, checks: Mapping[str, float]) -> None:
    print(f"\nObjective value: {value(model.objective):,.4f}")
    print("Formulation: deterministic spot-market clearing only.")

    print("\nSystem-wide lambda by time (dual of sum_n NetInjection[n,t] == 0):")
    for t in model.T:
        dual = model.dual.get(model.system_balance[t], None)
        if dual is None:
            print(f"  t={t}: lambda_sys=not available")
        else:
            print(f"  t={t}: lambda_sys={dual:10.4f}")

    print("\nNodal balance duals by time:")
    for t in model.T:
        duals = []
        for n in model.N:
            dual = model.dual.get(model.nodal_balance[n, t], None)
            duals.append(f"{n}=not available" if dual is None else f"{n}={dual:10.4f}")
        print(f"  t={t}: " + ", ".join(duals))

    print("\nDispatch by time:")
    for t in model.T:
        gen = sum(value(model.P_gen[g, t]) for g in model.G)
        shed = sum(value(model.P_shed[n, t]) for n in model.N)
        charge = sum(value(model.P_charge[i, n, t]) for i in model.I for n in model.N)
        discharge = sum(value(model.P_discharge[i, n, t]) for i in model.I for n in model.N)
        print(
            f"  t={t}: gen={gen:8.3f}, shed={shed:7.3f}, "
            f"charge={charge:7.3f}, discharge={discharge:7.3f}"
        )

    print("\nSanity checks:")
    for name, residual in checks.items():
        status = "OK" if residual <= 1e-5 else "CHECK"
        print(f"  {status:5s} {name}: {residual:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--solver", default=None, help="Optional Pyomo solver name, e.g. appsi_highs, glpk, cbc.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Processed data JSON from prepare_data.py.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.data)
    model = build_primal_market_clearing_model(data)
    results = solve_model(model, args.solver)

    termination = results.solver.termination_condition
    print(f"Solver termination: {termination}")
    if termination not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}:
        print("Solve did not return an optimal or feasible solution.")
        return 1

    checks = run_sanity_checks(model)
    print_solution_summary(model, checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
