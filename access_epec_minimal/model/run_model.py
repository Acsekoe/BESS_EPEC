"""Run the clean four-investor IEEE-9 BESS capacity game.

Power and energy capacity are the only strategic variables. The ISO dispatch
is competitive and embedded through a selectable relaxed-KKT or exact
strong-duality formulation. No access auction, access bids, quantity offers,
or operational price bids are present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pyomo.environ as pyo

from investors import InvestorConfig, capital_recovery_factor
from jacobi_diagonalization import (
    BestResponseResult,
    JacobiConfig,
    JacobiResult,
    audit_state,
    four_investors,
    initial_state,
    run_jacobi,
    solve_exact_market_model,
)
from primal_market_clearing_model import (
    MarketData,
    build_primal_market_clearing_model,
    effective_generation_offer,
    load_market_data,
)


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = MODEL_DIR / "input" / "market_data.json"
DEFAULT_OUTPUT = MODEL_DIR / "output" / "capacity_only_high_limits"


def fixed_capacity_data(
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
) -> MarketData:
    return replace(
        data,
        storage_units=[investor.investor_id for investor in investors],
        x_power=dict(power),
        x_energy=dict(energy),
    )


def clear_exact_market(
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
    config: JacobiConfig,
) -> pyo.ConcreteModel:
    """Clear the ordinary competitive market for one fixed capacity profile."""

    fixed = fixed_capacity_data(data, investors, power, energy)
    model = build_primal_market_clearing_model(fixed, include_load_shed=False)
    model.objective.deactivate()
    degradation = {
        investor.investor_id: investor.degradation_eur_per_mwh
        for investor in investors
    }
    model.objective_with_degradation = pyo.Objective(
        expr=sum(
            effective_generation_offer(data, generator)
            * model.P_gen[generator, time_]
            for generator in model.G
            for time_ in model.T
        )
        + sum(
            0.5
            * degradation[investor_id]
            * (
                model.P_charge[investor_id, node, time_]
                + model.P_discharge[investor_id, node, time_]
            )
            for investor_id in model.I
            for node in model.N
            for time_ in model.T
        )
        + 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(
            model.DemandAdjustment[node, time_] ** 2
            for node in model.N
            for time_ in model.T
        ),
        sense=pyo.minimize,
    )
    result = solve_exact_market_model(model, config)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Exact market reclear failed: {result.solver.termination_condition}")
    return model


def market_profit(
    market: pyo.ConcreteModel,
    data: MarketData,
    investor: InvestorConfig,
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
) -> float:
    investor_id = investor.investor_id
    generator_nodes = {
        generator: next(
            node
            for node in data.nodes
            if generator in data.generators_at_node.get(node, [])
        )
        for generator in data.generators
    }
    storage_revenue = sum(
        float(market.dual[market.nodal_balance[node, time_]])
        * (
            float(pyo.value(market.P_discharge[investor_id, node, time_]))
            - float(pyo.value(market.P_charge[investor_id, node, time_]))
        )
        for node in data.nodes
        for time_ in data.times
    )
    generation_rent = sum(
        share
        * (
            float(market.dual[market.nodal_balance[generator_nodes[generator], time_]])
            - data.generation_cost[generator]
        )
        * float(pyo.value(market.P_gen[generator, time_]))
        for generator, share in investor.owned_generation_shares.items()
        for time_ in data.times
    )
    degradation = 0.5 * investor.degradation_eur_per_mwh * sum(
        float(pyo.value(market.P_charge[investor_id, node, time_]))
        + float(pyo.value(market.P_discharge[investor_id, node, time_]))
        for node in data.nodes
        for time_ in data.times
    )
    daily_crf = capital_recovery_factor(investor.wacc, investor.lifetime_years) / 365.25
    capex = daily_crf * sum(
        investor.cost_power_eur_per_mw * power[investor_id, node]
        + investor.cost_energy_eur_per_mwh * energy[investor_id, node]
        for node in data.nodes
    )
    return storage_revenue + generation_rent - degradation - capex


def _checkpoint(path: Path, state: JacobiResult, config: JacobiConfig) -> None:
    payload = {
        "format_version": 1,
        "formulation": f"capacity-only-{config.formulation}",
        "sweep": state.sweep,
        "converged": state.converged,
        "stable_sweeps": state.stable_sweeps,
        "power": [
            {"investor": key[0], "node": key[1], "mw": value}
            for key, value in sorted(state.power.items())
        ],
        "energy": [
            {"investor": key[0], "node": key[1], "mwh": value}
            for key, value in sorted(state.energy.items())
        ],
        "node_limit_mw": config.node_limit_mw,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_capacities(
    path: Path,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    state: JacobiResult,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["investor", "node", "power_mw", "energy_mwh", "duration_hours"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for investor in investors:
            for node in data.nodes:
                key = investor.investor_id, node
                power = state.power[key]
                writer.writerow(
                    {
                        "investor": investor.investor_id,
                        "node": node,
                        "power_mw": power,
                        "energy_mwh": state.energy[key],
                        "duration_hours": state.energy[key] / power if power > 1.0e-9 else None,
                    }
                )


def _capacity_trajectory_rows(
    sweep: int,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    nodal_totals = {
        node: sum(power[investor.investor_id, node] for investor in investors)
        for node in data.nodes
    }
    for investor in investors:
        investor_id = investor.investor_id
        for node in data.nodes:
            key = investor_id, node
            power_mw = power[key]
            energy_mwh = energy[key]
            nodal_total = nodal_totals[node]
            rows.append(
                {
                    "sweep": sweep,
                    "investor": investor_id,
                    "node": node,
                    "power_mw": power_mw,
                    "energy_mwh": energy_mwh,
                    "duration_hours": (
                        energy_mwh / power_mw if power_mw > 1.0e-9 else None
                    ),
                    "nodal_total_power_mw": nodal_total,
                    "investor_nodal_power_share": (
                        power_mw / nodal_total if nodal_total > 1.0e-9 else None
                    ),
                }
            )
    return rows


def _capacity_total_rows(
    sweep: int,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    system_power = sum(power.values())
    for investor in investors:
        investor_id = investor.investor_id
        total_power = sum(power[investor_id, node] for node in data.nodes)
        total_energy = sum(energy[investor_id, node] for node in data.nodes)
        rows.append(
            {
                "sweep": sweep,
                "investor": investor_id,
                "total_power_mw": total_power,
                "total_energy_mwh": total_energy,
                "portfolio_duration_hours": (
                    total_energy / total_power if total_power > 1.0e-9 else None
                ),
                "investor_system_power_share": (
                    total_power / system_power if system_power > 1.0e-9 else None
                ),
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_market(path: Path, market: pyo.ConcreteModel, data: MarketData) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "time",
            "node",
            "demand_mw",
            "lmp_eur_per_mwh",
            "generation_mw",
            "charge_mw",
            "discharge_mw",
            "net_injection_mw",
            "demand_adjustment_mw",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for time_ in data.times:
            for node in data.nodes:
                writer.writerow(
                    {
                        "time": time_,
                        "node": node,
                        "demand_mw": data.demand_el[node, time_],
                        "lmp_eur_per_mwh": float(
                            market.dual[market.nodal_balance[node, time_]]
                        ),
                        "generation_mw": sum(
                            float(pyo.value(market.P_gen[generator, time_]))
                            for generator in data.generators_at_node.get(node, [])
                        ),
                        "charge_mw": sum(
                            float(pyo.value(market.P_charge[investor_id, node, time_]))
                            for investor_id in market.I
                        ),
                        "discharge_mw": sum(
                            float(pyo.value(market.P_discharge[investor_id, node, time_]))
                            for investor_id in market.I
                        ),
                        "net_injection_mw": float(
                            pyo.value(market.NetInjection[node, time_])
                        ),
                        "demand_adjustment_mw": float(
                            pyo.value(market.DemandAdjustment[node, time_])
                        ),
                    }
                )


def _audit(
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    config: JacobiConfig,
    state: JacobiResult,
) -> tuple[list[dict[str, object]], pyo.ConcreteModel]:
    common_market = clear_exact_market(
        data, investors, state.power, state.energy, config
    )
    current_profit = {
        investor.investor_id: market_profit(
            common_market, data, investor, state.power, state.energy
        )
        for investor in investors
    }
    responses = audit_state(data, config, state)
    rows = []
    for investor in investors:
        investor_id = investor.investor_id
        response: BestResponseResult = responses[investor_id]
        candidate_power = dict(state.power)
        candidate_energy = dict(state.energy)
        for node in data.nodes:
            candidate_power[investor_id, node] = response.proposed_power[node]
            candidate_energy[investor_id, node] = response.proposed_energy[node]
        if response.outcome.has_solution:
            candidate_market = clear_exact_market(
                data, investors, candidate_power, candidate_energy, config
            )
            recleared_profit = market_profit(
                candidate_market, data, investor, candidate_power, candidate_energy
            )
        else:
            recleared_profit = float("nan")
        rows.append(
            {
                "investor": investor_id,
                "optimal": response.outcome.optimal,
                "termination": response.outcome.termination,
                "max_power_deviation_mw": max(
                    abs(response.proposed_power[node] - state.power[investor_id, node])
                    for node in data.nodes
                ),
                "max_energy_deviation_mwh": max(
                    abs(response.proposed_energy[node] - state.energy[investor_id, node])
                    for node in data.nodes
                ),
                "embedded_profit_eur_per_day": response.profit_eur_per_day,
                "recleared_profit_eur_per_day": recleared_profit,
                "embedded_reclear_profit_gap_eur_per_day": abs(
                    response.profit_eur_per_day - recleared_profit
                ),
                "current_recleared_profit_eur_per_day": current_profit[investor_id],
                "profitable_deviation_eur_per_day": max(
                    0.0, recleared_profit - current_profit[investor_id]
                ),
                "complementarity_max_product": response.complementarity_max_product,
                "complementarity_max_violation": response.complementarity_max_violation,
                "absolute_primal_dual_gap_eur_per_day": abs(
                    response.primal_dual_gap_eur_per_day
                ),
            }
        )
    return rows, common_market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--formulation",
        choices=("relaxed-kkt", "strong-duality"),
        default="relaxed-kkt",
    )
    parser.add_argument("--node-limit-mw", type=float, default=1_000.0)
    parser.add_argument("--max-sweeps", type=int, default=60)
    parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument("--tolerance-mw", type=float, default=0.5)
    parser.add_argument("--tolerance-mwh", type=float, default=1.0)
    parser.add_argument("--consecutive-sweeps", type=int, default=2)
    parser.add_argument("--initial-power-mw", type=float, default=5.0)
    parser.add_argument("--initial-ratio-hours", type=float, default=3.0)
    parser.add_argument("--complementarity-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--proximal-penalty", type=float, default=0.01)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--ipopt-linear-solver", default="ma57")
    parser.add_argument("--max-solver-iterations", type=int, default=3_000)
    parser.add_argument("--max-solve-seconds", type=float, default=600.0)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.data)
    investors = four_investors(data)
    config = JacobiConfig(
        investors=investors,
        formulation=args.formulation,
        node_limit_mw=args.node_limit_mw,
        max_sweeps=args.max_sweeps,
        damping=args.damping,
        tolerance_mw=args.tolerance_mw,
        tolerance_mwh=args.tolerance_mwh,
        consecutive_sweeps=args.consecutive_sweeps,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        complementarity_epsilon=args.complementarity_epsilon,
        proximal_penalty=args.proximal_penalty,
        parallel_workers=args.parallel_workers,
        ipopt_linear_solver=args.ipopt_linear_solver,
        max_solver_iterations=args.max_solver_iterations,
        max_solve_seconds=args.max_solve_seconds,
        solver_tolerance=args.solver_tolerance,
        tee=args.tee,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    run_config = {
        "formulation": f"capacity-only-{config.formulation}",
        "strategic_variables": ["nodal_power_capacity_mw", "nodal_energy_capacity_mwh"],
        "access_auction": False,
        "operational_bidding": False,
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        **{
            key: value
            for key, value in asdict(config).items()
            if key not in {"investors", "ipopt_executable"}
        },
        "investors": [
            {
                **asdict(investor),
                "owned_generation_shares": dict(investor.owned_generation_shares),
            }
            for investor in investors
        ],
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    state_initial = initial_state(data, config)
    capacity_trajectory = _capacity_trajectory_rows(
        0, data, investors, state_initial.power, state_initial.energy
    )
    capacity_totals_trajectory = _capacity_total_rows(
        0, data, investors, state_initial.power, state_initial.energy
    )
    _write_rows(
        args.output_dir / "capacity_by_investor_node_by_sweep.csv",
        capacity_trajectory,
    )
    _write_rows(
        args.output_dir / "capacity_totals_by_investor_by_sweep.csv",
        capacity_totals_trajectory,
    )

    def on_sweep(state: JacobiResult) -> None:
        row = state.history[-1]
        print(
            f"sweep={state.sweep:03d} "
            f"power_residual={row['max_raw_power_deviation_mw']:.4f} MW "
            f"energy_residual={row['max_raw_energy_deviation_mwh']:.4f} MWh "
            f"total_power={row['total_power_mw']:.3f} MW "
            f"optimal={row['all_best_responses_optimal']}",
            flush=True,
        )
        _write_history(args.output_dir / "history.csv", state.history)
        _write_capacities(
            args.output_dir / "current_capacities.csv", data, investors, state
        )
        capacity_trajectory.extend(
            _capacity_trajectory_rows(
                state.sweep, data, investors, state.power, state.energy
            )
        )
        capacity_totals_trajectory.extend(
            _capacity_total_rows(
                state.sweep, data, investors, state.power, state.energy
            )
        )
        _write_rows(
            args.output_dir / "capacity_by_investor_node_by_sweep.csv",
            capacity_trajectory,
        )
        _write_rows(
            args.output_dir / "capacity_totals_by_investor_by_sweep.csv",
            capacity_totals_trajectory,
        )
        _checkpoint(args.output_dir / "checkpoint.json", state, config)

    print(
        "Starting clean capacity-only EPEC: "
        f"formulation={config.formulation}, "
        f"node_limit={config.node_limit_mw:g} MW, "
        f"investors={len(investors)}, workers={config.parallel_workers}",
        flush=True,
    )
    state = run_jacobi(data, config, initial=state_initial, on_sweep=on_sweep)
    audit_rows, market = _audit(data, investors, config, state)
    _write_capacities(args.output_dir / "final_capacities.csv", data, investors, state)
    _write_market(args.output_dir / "final_market.csv", market, data)
    with (args.output_dir / "final_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    nodal_totals = {
        node: sum(state.power[investor.investor_id, node] for investor in investors)
        for node in data.nodes
    }
    summary = {
        "formulation": f"capacity-only-{config.formulation}",
        "strategic_variables": ["power_capacity", "energy_capacity"],
        "access_auction": False,
        "operational_bidding": False,
        "converged": state.converged,
        "sweeps": state.sweep,
        "stop_reason": state.stop_reason,
        "node_limit_mw": config.node_limit_mw,
        "maximum_nodal_capacity_mw": max(nodal_totals.values()),
        "maximum_nodal_limit_utilisation": max(nodal_totals.values()) / config.node_limit_mw,
        "total_power_mw": sum(state.power.values()),
        "total_energy_mwh": sum(state.energy.values()),
        "final_raw_power_residual_mw": max(
            row["max_power_deviation_mw"] for row in audit_rows
        ),
        "final_raw_energy_residual_mwh": max(
            row["max_energy_deviation_mwh"] for row in audit_rows
        ),
        "final_profitable_deviation_eur_per_day": max(
            row["profitable_deviation_eur_per_day"] for row in audit_rows
        ),
        "final_embedded_reclear_profit_gap_eur_per_day": max(
            row["embedded_reclear_profit_gap_eur_per_day"] for row in audit_rows
        ),
        "final_complementarity_violation": max(
            row["complementarity_max_violation"] for row in audit_rows
        ),
        "nodal_power_mw": nodal_totals,
        "capacity_trajectory_file": "capacity_by_investor_node_by_sweep.csv",
        "investor_totals_trajectory_file": "capacity_totals_by_investor_by_sweep.csv",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _checkpoint(args.output_dir / "checkpoint.json", state, config)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if state.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
