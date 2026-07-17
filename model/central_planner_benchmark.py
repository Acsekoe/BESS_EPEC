"""Central-planner (ISO) BESS siting/sizing benchmark for the spot-market EPEC.

This is the cost-minimizing counterpart to the strategic EPEC. A single planner
chooses BESS power/energy capacity at every node *and* the spot-market dispatch
jointly, minimizing total system resource cost:

    generation cost + curtailment cost + annualized storage CAPEX + degradation

subject to the same lower-level clearing physics the EPEC embeds (nodal balance,
system balance, PTDF line limits, generator caps, SOC dynamics/periodicity), the
same shared nodal connection limit, and the same E/P envelope.

It is deliberately *not* an MPEC: with a single welfare/cost objective there is
no game, so the problem is one convex QP (quadratic only in the curtailment
term). It gives the first-best siting/sizing to compare against the EPEC result:
total MW/MWh, where storage goes, system cost, and the efficiency gap ("price of
anarchy"). Ownership and market prices are irrelevant to the planner - generation
rent is a transfer, not a resource cost, so it never appears here.

Run:
    python central_planner_benchmark.py --data data/processed/market_data_9bus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyomo.environ as pyo

from primal_market_clearing_model import MarketData, load_market_data, value
from single_investor_mpec import (
    DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH,
    DEFAULT_BESS_COST_POWER_EUR_PER_MW,
    DEFAULT_DEGRADATION_EUR_PER_MWH,
    DEFAULT_LIFETIME_YEARS,
    DEFAULT_NODE_LIMIT_MW,
    DEFAULT_RATIO_MAX,
    DEFAULT_RATIO_MIN,
    DEFAULT_WACC,
    QuadraticDemandCurve,
    capital_recovery_factor,
    default_quadratic_demand_curve,
    quadratic_reference_lambda,
    reference_system_price,
)
from single_investor_mpec_results import _write_csv
from solver_utils import get_ipopt_solver

EXPERIMENT_DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "market_data_euro.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "central_planner"


def build_central_planner_model(
    data: MarketData,
    quad: QuadraticDemandCurve,
    *,
    wacc: float = DEFAULT_WACC,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW,
    ratio_min: float = DEFAULT_RATIO_MIN,
    ratio_max: float = DEFAULT_RATIO_MAX,
    cost_power_eur_per_mw: float = DEFAULT_BESS_COST_POWER_EUR_PER_MW,
    cost_energy_eur_per_mwh: float = DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH,
    degradation_eur_per_mwh: float = DEFAULT_DEGRADATION_EUR_PER_MWH,
) -> pyo.ConcreteModel:
    """Build the single-level cost-minimizing planner QP with endogenous storage.

    One aggregate storage fleet per node (no ownership index): the planner does
    not care who builds what. Uses a single social discount rate ``wacc`` rather
    than the investors' heterogeneous WACCs.
    """

    eta = data.eta
    last_t = max(data.times)
    crf_daily = capital_recovery_factor(wacc, lifetime_years) / 365.25

    m = pyo.ConcreteModel(name="Central Planner BESS Benchmark")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.G = pyo.Set(initialize=data.generators, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)

    # Investment variables: aggregate storage per node, capped by the shared
    # nodal connection limit and the E/P envelope.
    m.X_power = pyo.Var(m.N, bounds=lambda mm, n: (0.0, node_limit_mw), initialize=0.0)
    m.X_energy = pyo.Var(m.N, bounds=lambda mm, n: (0.0, ratio_max * node_limit_mw), initialize=0.0)
    m.energy_ratio_min = pyo.Constraint(m.N, rule=lambda mm, n: mm.X_energy[n] >= ratio_min * mm.X_power[n])
    m.energy_ratio_max = pyo.Constraint(m.N, rule=lambda mm, n: mm.X_energy[n] <= ratio_max * mm.X_power[n])

    # Dispatch variables (single aggregate storage per node -> node-indexed).
    m.P_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_shed = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_charge = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_discharge = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.SOC = pyo.Var(m.N, m.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)

    # Clearing physics (identical to the EPEC lower level).
    def nodal_balance_rule(mm, n, t):
        return (
            sum(mm.P_gen[g, t] for g in data.generators_at_node.get(n, []))
            + mm.P_discharge[n, t]
            - mm.P_charge[n, t]
            + mm.P_shed[n, t]
            - data.demand_el[n, t]
            == mm.NetInjection[n, t]
        )

    m.nodal_balance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)
    m.system_balance = pyo.Constraint(m.T, rule=lambda mm, t: sum(mm.NetInjection[n, t] for n in mm.N) == 0.0)
    m.generation_capacity_bound = pyo.Constraint(
        m.G, m.T, rule=lambda mm, g, t: mm.P_gen[g, t] <= data.generation_capacity[g, t]
    )
    m.line_upper_bound = pyo.Constraint(
        m.L, m.T,
        rule=lambda mm, l, t: sum(data.ptdf[l, n] * mm.NetInjection[n, t] for n in mm.N) <= data.line_limit[l],
    )
    m.line_lower_bound = pyo.Constraint(
        m.L, m.T,
        rule=lambda mm, l, t: sum(data.ptdf[l, n] * mm.NetInjection[n, t] for n in mm.N) >= -data.line_limit[l],
    )
    m.charge_power_bound = pyo.Constraint(m.N, m.T, rule=lambda mm, n, t: mm.P_charge[n, t] <= mm.X_power[n])
    m.discharge_power_bound = pyo.Constraint(m.N, m.T, rule=lambda mm, n, t: mm.P_discharge[n, t] <= mm.X_power[n])
    m.soc_transition = pyo.Constraint(
        m.N, m.T,
        rule=lambda mm, n, t: mm.SOC[n, t]
        == mm.SOC[n, t - 1] + eta * mm.P_charge[n, t] - mm.P_discharge[n, t] / eta,
    )
    m.soc_capacity_bound = pyo.Constraint(m.N, m.T_SOC, rule=lambda mm, n, tau: mm.SOC[n, tau] <= mm.X_energy[n])
    m.soc_periodicity = pyo.Constraint(m.N, rule=lambda mm, n: mm.SOC[n, 0] == mm.SOC[n, last_t])
    m.load_shed_bound = pyo.Constraint(m.N, m.T, rule=lambda mm, n, t: mm.P_shed[n, t] <= data.demand_el[n, t])

    # Total system resource cost (the planner objective).
    m.generation_cost_expr = pyo.Expression(
        expr=sum(data.generation_cost[g] * m.P_gen[g, t] for g in m.G for t in m.T)
    )
    m.curtailment_cost_expr = pyo.Expression(
        expr=sum(
            quad.alpha * m.P_shed[n, t]
            + 0.5 * quad.quad_coefficient(data.demand_el[n, t]) * m.P_shed[n, t] ** 2
            for n in m.N
            for t in m.T
        )
    )
    m.storage_capex_expr = pyo.Expression(
        expr=crf_daily * sum(cost_power_eur_per_mw * m.X_power[n] + cost_energy_eur_per_mwh * m.X_energy[n] for n in m.N)
    )
    m.degradation_cost_expr = pyo.Expression(
        expr=0.5 * degradation_eur_per_mwh * sum(m.P_charge[n, t] + m.P_discharge[n, t] for n in m.N for t in m.T)
    )
    m.system_cost_expr = pyo.Expression(
        expr=m.generation_cost_expr + m.curtailment_cost_expr + m.storage_capex_expr + m.degradation_cost_expr
    )
    m.objective = pyo.Objective(expr=m.system_cost_expr, sense=pyo.minimize)

    m._market_data = data
    m._quad_demand = quad
    m._wacc = wacc
    m._node_limit_mw = node_limit_mw
    return m


def summarize(model: pyo.ConcreteModel, quad: QuadraticDemandCurve) -> dict:
    data: MarketData = model._market_data
    lam = quadratic_reference_lambda(model, quad)
    sys_price = reference_system_price(model, lam)

    nodes = list(model.N)
    per_node = {}
    for n in nodes:
        p = value(model.X_power[n])
        e = value(model.X_energy[n])
        charge = sum(value(model.P_charge[n, t]) for t in model.T)
        discharge = sum(value(model.P_discharge[n, t]) for t in model.T)
        per_node[n] = {
            "x_power_mw": p,
            "x_energy_mwh": e,
            "ratio_hours": e / p if p > 1e-9 else 0.0,
            "charge_mwh": charge,
            "discharge_mwh": discharge,
            "share_of_node_limit": p / model._node_limit_mw,
        }
    return {
        "wacc": model._wacc,
        "system_cost_eur_per_day": value(model.system_cost_expr),
        "generation_cost_eur_per_day": value(model.generation_cost_expr),
        "curtailment_cost_eur_per_day": value(model.curtailment_cost_expr),
        "storage_capex_eur_per_day": value(model.storage_capex_expr),
        "degradation_cost_eur_per_day": value(model.degradation_cost_expr),
        "total_power_mw": sum(per_node[n]["x_power_mw"] for n in nodes),
        "total_energy_mwh": sum(per_node[n]["x_energy_mwh"] for n in nodes),
        "lambda_min_eur_per_mwh": min(lam.values()),
        "lambda_max_eur_per_mwh": max(lam.values()),
        "system_price_min_eur_per_mwh": min(sys_price.values()),
        "system_price_max_eur_per_mwh": max(sys_price.values()),
        "per_node": per_node,
        "_lambda": lam,
    }


def print_summary(summary: dict) -> None:
    print("\nCentral-planner benchmark result")
    print(f"  social WACC: {summary['wacc']:.1%}")
    print(f"  total BESS: {summary['total_power_mw']:.2f} MW / {summary['total_energy_mwh']:.2f} MWh")
    print(
        f"  system cost: {summary['system_cost_eur_per_day']:,.2f} EUR/day "
        f"(gen {summary['generation_cost_eur_per_day']:,.0f} + curtail {summary['curtailment_cost_eur_per_day']:,.0f} "
        f"+ storage capex {summary['storage_capex_eur_per_day']:,.0f} + degrad {summary['degradation_cost_eur_per_day']:,.0f})"
    )
    print(
        f"  nodal LMP range: {summary['lambda_min_eur_per_mwh']:,.2f} to {summary['lambda_max_eur_per_mwh']:,.2f} EUR/MWh"
    )
    print("  per-node storage [MW / MWh]:")
    for n, r in summary["per_node"].items():
        if r["x_power_mw"] > 1e-3:
            print(
                f"    {n}: {r['x_power_mw']:7.2f} MW / {r['x_energy_mwh']:8.2f} MWh "
                f"(E/P {r['ratio_hours']:.1f}h, {r['share_of_node_limit']*100:.0f}% of limit)"
            )


def export(output_dir: Path, model: pyo.ConcreteModel, summary: dict, data_path: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lam = summary["_lambda"]
    summary_json = {k: v for k, v in summary.items() if k != "_lambda"}
    summary_json["data_path"] = str(data_path)
    summary_json["node_limit_mw"] = model._node_limit_mw
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    _write_csv(
        output_dir / "final_capacities.csv",
        ["node", "x_power_mw", "x_energy_mwh", "ratio_hours", "charge_mwh", "discharge_mwh", "share_of_node_limit"],
        [{"node": n, **{k: r[k] for k in ("x_power_mw", "x_energy_mwh", "ratio_hours", "charge_mwh", "discharge_mwh", "share_of_node_limit")}}
         for n, r in summary["per_node"].items()],
    )
    _write_csv(
        output_dir / "node_hour_prices.csv",
        ["hour", "node", "lambda_eur_per_mwh"],
        [{"hour": t, "node": n, "lambda_eur_per_mwh": lam[n, t]} for t in model.T for n in model.N],
    )
    _write_csv(
        output_dir / "storage_hour_operation.csv",
        ["hour", "node", "p_charge_mw", "p_discharge_mw", "soc_mwh", "lambda_eur_per_mwh"],
        [
            {
                "hour": t,
                "node": n,
                "p_charge_mw": value(model.P_charge[n, t]),
                "p_discharge_mw": value(model.P_discharge[n, t]),
                "soc_mwh": value(model.SOC[n, t]),
                "lambda_eur_per_mwh": lam[n, t],
            }
            for t in model.T
            for n in model.N
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Central-planner BESS siting/sizing benchmark")
    parser.add_argument("--data", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--wacc", type=float, default=DEFAULT_WACC, help="Single social discount rate.")
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--max-cpu-time", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Optional label appended to the output folder name.")
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.data)
    quad = default_quadratic_demand_curve()
    print(
        f"Central-planner benchmark: {len(data.nodes)} nodes, social WACC {args.wacc:.1%}, "
        f"node limit {args.node_limit_mw:.0f} MW"
    )
    print(f"Quadratic demand curve: marginal WTP = {quad.alpha:,.2f} + {quad.beta:,.2f} * curtailed_share EUR/MWh")

    model = build_central_planner_model(data, quad, wacc=args.wacc, node_limit_mw=args.node_limit_mw)
    results = get_ipopt_solver({"max_cpu_time": args.max_cpu_time}).solve(model, tee=args.tee)
    termination = str(results.solver.termination_condition)
    print(f"Solver termination: {termination}")
    if termination != "optimal":
        print("Planner solve did not terminate optimally.")
        return 1

    summary = summarize(model, quad)
    print_summary(summary)
    if not args.no_export:
        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            output_dir = DEFAULT_OUTPUT_DIR
            if args.tag:
                output_dir = output_dir.with_name(output_dir.name + f"_{args.tag}")
        export(output_dir, model, summary, args.data)
        print(f"\nWrote central-planner outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
