"""Fixed-fleet sensitivity of finite-gamma prices against the exact market.

The experiment preserves every exported investor-node MW/MWh pair, clears the
exact hard-balance market, selects its minimum-norm LMP on the exact dual-
optimal face, and compares matched soft primal/dual clears across gamma values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import pyomo.environ as pyo

try:
    from .common import MODEL_DIR, load_calibrated_case, solve_ipopt
    from .dual_tikhonov_llp import build_tikhonov_dual_llp
    from .primal_llp import build_primal_llp
    from .soft_balance_llp import soft_balance_prices, solve_matched_soft_market
except ImportError:  # Direct execution.
    from common import MODEL_DIR, load_calibrated_case, solve_ipopt
    from dual_tikhonov_llp import build_tikhonov_dual_llp
    from primal_llp import build_primal_llp
    from soft_balance_llp import soft_balance_prices, solve_matched_soft_market

from single_investor_mpec import InvestorConfig, capital_recovery_factor


DEFAULT_FLEETS = {
    "with_penalty": (
        MODEL_DIR
        / "output"
        / "jacobi_tikhonov_runs"
        / "tikhonov_epec_capacity_100iters"
    ),
    "no_penalty": (
        MODEL_DIR
        / "output"
        / "jacobi_tikhonov_runs"
        / "tikhonov_capacity_no_penalty_d025"
    ),
}
DEFAULT_GAMMAS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in known:
                fieldnames.append(field)
                known.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_fleet(run_dir: Path):
    with (run_dir / "run_config.json").open(encoding="utf-8") as handle:
        run_config = json.load(handle)
    with (run_dir / "final_capacities.csv").open(newline="", encoding="utf-8") as handle:
        capacity_rows = list(csv.DictReader(handle))

    investors = [InvestorConfig(**item) for item in run_config["investors"]]
    units = [investor.investor_id for investor in investors]
    data = load_calibrated_case(Path(run_config["data_path"]))
    by_key = {(row["investor"], row["node"]): row for row in capacity_rows}
    missing = [
        (unit, node)
        for unit in units
        for node in data.nodes
        if (unit, node) not in by_key
    ]
    if missing:
        raise ValueError(f"Missing exported capacities: {missing}")

    x_power = {
        (unit, node): max(0.0, float(by_key[unit, node]["x_power_mw"]))
        for unit in units
        for node in data.nodes
    }
    x_energy = {
        (unit, node): max(0.0, float(by_key[unit, node]["x_energy_mwh"]))
        for unit in units
        for node in data.nodes
    }
    fixed_data = replace(
        data,
        storage_units=units,
        x_power=x_power,
        x_energy=x_energy,
    )
    degradation = {
        investor.investor_id: investor.degradation_eur_per_mwh
        for investor in investors
    }
    return fixed_data, investors, degradation, run_config, capacity_rows


def _termination(results) -> str:
    return str(results.solver.termination_condition)


def _storage_value(model, component_name: str, unit: str, node: str, hour: int) -> float:
    component = getattr(model, component_name)
    index = (unit, node, hour)
    return pyo.value(component[index]) if index in component else 0.0


def _prices_from_dual(model) -> dict[tuple[str, int], float]:
    return {
        (str(node), int(hour)): pyo.value(model.lam[node, hour])
        for node in model.N
        for hour in model.T
    }


def _aggregate_storage_net(model, units: Iterable[str]) -> dict[tuple[str, int], float]:
    return {
        (str(node), int(hour)): sum(
            _storage_value(model, "P_discharge", unit, str(node), int(hour))
            - _storage_value(model, "P_charge", unit, str(node), int(hour))
            for unit in units
        )
        for node in model.N
        for hour in model.T
    }


def _generator_dispatch(model) -> dict[tuple[str, int], float]:
    return {
        (str(generator), int(hour)): pyo.value(model.P_gen[generator, hour])
        for generator, hour in model.GT
    }


def _market_costs(model) -> tuple[float, float]:
    return (
        pyo.value(model.generation_cost_expr),
        pyo.value(model.degradation_cost_expr),
    )


def _investor_economics(
    fleet_label: str,
    case_label: str,
    model,
    prices: dict[tuple[str, int], float],
    data,
    investors: list[InvestorConfig],
) -> list[dict[str, Any]]:
    units = [investor.investor_id for investor in investors]
    generator_node = {
        generator: node
        for node in data.nodes
        for generator in data.generators_at_node.get(node, [])
    }
    rows: list[dict[str, Any]] = []
    for investor in investors:
        unit = investor.investor_id
        charge = sum(
            _storage_value(model, "P_charge", unit, str(node), int(hour))
            for node in model.N
            for hour in model.T
        )
        discharge = sum(
            _storage_value(model, "P_discharge", unit, str(node), int(hour))
            for node in model.N
            for hour in model.T
        )
        spot_revenue = sum(
            prices[str(node), int(hour)]
            * (
                _storage_value(model, "P_discharge", unit, str(node), int(hour))
                - _storage_value(model, "P_charge", unit, str(node), int(hour))
            )
            for node in model.N
            for hour in model.T
        )
        generation_rent = 0.0
        for generator, share in investor.owned_generation_shares.items():
            node = generator_node.get(generator)
            if node is None or share == 0.0:
                continue
            generation_rent += share * sum(
                (prices[node, int(hour)] - data.generation_cost[generator])
                * pyo.value(model.P_gen[generator, hour])
                for hour in model.T
                if (generator, hour) in model.P_gen
            )
        degradation = 0.5 * investor.degradation_eur_per_mwh * (charge + discharge)
        crf_daily = (
            capital_recovery_factor(investor.wacc, investor.lifetime_years) / 365.25
        )
        capex = crf_daily * sum(
            investor.cost_power_eur_per_mw * data.x_power[unit, str(node)]
            + investor.cost_energy_eur_per_mwh * data.x_energy[unit, str(node)]
            for node in model.N
        )

        proportional_revenue = 0.0
        proportional_throughput = 0.0
        for node in model.N:
            node = str(node)
            total_power = sum(data.x_power[other, node] for other in units)
            share = data.x_power[unit, node] / total_power if total_power > 1.0e-9 else 0.0
            for hour in model.T:
                hour = int(hour)
                aggregate_charge = sum(
                    _storage_value(model, "P_charge", other, node, hour)
                    for other in units
                )
                aggregate_discharge = sum(
                    _storage_value(model, "P_discharge", other, node, hour)
                    for other in units
                )
                proportional_revenue += (
                    share
                    * prices[node, hour]
                    * (aggregate_discharge - aggregate_charge)
                )
                proportional_throughput += share * (
                    aggregate_charge + aggregate_discharge
                )
        proportional_profit = (
            proportional_revenue
            + generation_rent
            - 0.5 * investor.degradation_eur_per_mwh * proportional_throughput
            - capex
        )
        settled_profit = spot_revenue + generation_rent - degradation - capex
        rows.append(
            {
                "fleet": fleet_label,
                "case": case_label,
                "investor": unit,
                "total_power_mw": sum(data.x_power[unit, str(node)] for node in model.N),
                "total_energy_mwh": sum(data.x_energy[unit, str(node)] for node in model.N),
                "charge_mwh": charge,
                "discharge_mwh": discharge,
                "throughput_mwh": charge + discharge,
                "spot_revenue_eur_per_day": spot_revenue,
                "generation_rent_eur_per_day": generation_rent,
                "degradation_eur_per_day": degradation,
                "capex_eur_per_day": capex,
                "profit_eur_per_day": settled_profit,
                "capacity_proportional_profit_eur_per_day": proportional_profit,
                "dispatch_attribution_band_eur_per_day": abs(
                    settled_profit - proportional_profit
                ),
            }
        )
    return rows


def _detail_rows(
    fleet_label: str,
    case_label: str,
    gamma: float | None,
    model,
    prices: dict[tuple[str, int], float],
    hard_prices: dict[tuple[str, int], float],
    units: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    price_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    for node in model.N:
        node = str(node)
        for hour in model.T:
            hour = int(hour)
            residual = (
                pyo.value(model.balance_residual[node, hour])
                if hasattr(model, "balance_residual")
                else 0.0
            )
            price_rows.append(
                {
                    "fleet": fleet_label,
                    "case": case_label,
                    "gamma": gamma,
                    "node": node,
                    "hour": hour,
                    "lambda_eur_per_mwh": prices[node, hour],
                    "hard_min_norm_lambda_eur_per_mwh": hard_prices[node, hour],
                    "lambda_difference_eur_per_mwh": (
                        prices[node, hour] - hard_prices[node, hour]
                    ),
                    "balance_residual_mw": residual,
                }
            )
            for unit in units:
                charge = _storage_value(model, "P_charge", unit, node, hour)
                discharge = _storage_value(model, "P_discharge", unit, node, hour)
                operation_rows.append(
                    {
                        "fleet": fleet_label,
                        "case": case_label,
                        "gamma": gamma,
                        "investor": unit,
                        "node": node,
                        "hour": hour,
                        "p_charge_mw": charge,
                        "p_discharge_mw": discharge,
                        "net_injection_mw": discharge - charge,
                    }
                )
    for generator, hour in model.GT:
        generator_rows.append(
            {
                "fleet": fleet_label,
                "case": case_label,
                "gamma": gamma,
                "generator": str(generator),
                "hour": int(hour),
                "p_gen_mw": pyo.value(model.P_gen[generator, hour]),
            }
        )
    return price_rows, operation_rows, generator_rows


def _case_summary(
    *,
    fleet_label: str,
    case_label: str,
    gamma: float | None,
    primal,
    primal_termination: str,
    dual_termination: str,
    solve_seconds: float,
    prices: dict[tuple[str, int], float],
    hard_prices: dict[tuple[str, int], float],
    hard_storage_net: dict[tuple[str, int], float],
    hard_generation: dict[tuple[str, int], float],
    units: list[str],
    primal_objective: float,
    unregularized_dual_objective: float,
    strong_duality_gap: float,
) -> dict[str, Any]:
    generation_cost, degradation_cost = _market_costs(primal)
    residuals = (
        {
            (str(node), int(hour)): pyo.value(primal.balance_residual[node, hour])
            for node in primal.N
            for hour in primal.T
        }
        if hasattr(primal, "balance_residual")
        else {(str(node), int(hour)): 0.0 for node in primal.N for hour in primal.T}
    )
    system_residuals = {
        int(hour): sum(residuals[str(node), int(hour)] for node in primal.N)
        for hour in primal.T
    }
    price_errors = [abs(prices[key] - hard_prices[key]) for key in hard_prices]
    storage_net = _aggregate_storage_net(primal, units)
    storage_node_errors = [
        abs(storage_net[key] - hard_storage_net[key]) for key in hard_storage_net
    ]
    system_storage_errors = [
        abs(
            sum(storage_net[str(node), int(hour)] for node in primal.N)
            - sum(hard_storage_net[str(node), int(hour)] for node in primal.N)
        )
        for hour in primal.T
    ]
    generation = _generator_dispatch(primal)
    generation_errors = [
        abs(generation.get(key, 0.0) - hard_generation.get(key, 0.0))
        for key in set(generation) | set(hard_generation)
    ]
    soft_penalty = (
        pyo.value(primal.soft_balance_penalty_expr)
        if hasattr(primal, "soft_balance_penalty_expr")
        else 0.0
    )
    return {
        "fleet": fleet_label,
        "case": case_label,
        "gamma": gamma,
        "primal_termination": primal_termination,
        "dual_termination": dual_termination,
        "solve_seconds": solve_seconds,
        "primal_objective_eur_per_day": primal_objective,
        "unpenalized_market_cost_eur_per_day": generation_cost + degradation_cost,
        "generation_cost_eur_per_day": generation_cost,
        "degradation_cost_eur_per_day": degradation_cost,
        "soft_balance_penalty_eur_per_day": soft_penalty,
        "unregularized_dual_objective_eur_per_day": unregularized_dual_objective,
        "strong_duality_gap_eur_per_day": strong_duality_gap,
        "max_abs_nodal_balance_residual_mw": max(
            (abs(value) for value in residuals.values()), default=0.0
        ),
        "max_abs_hourly_system_balance_residual_mw": max(
            (abs(value) for value in system_residuals.values()), default=0.0
        ),
        "lambda_min_eur_per_mwh": min(prices.values()),
        "lambda_max_eur_per_mwh": max(prices.values()),
        "max_abs_lambda_diff_vs_hard_eur_per_mwh": max(price_errors, default=0.0),
        "mean_abs_lambda_diff_vs_hard_eur_per_mwh": (
            sum(price_errors) / len(price_errors) if price_errors else 0.0
        ),
        "max_abs_storage_node_hour_net_diff_vs_hard_mw": max(
            storage_node_errors, default=0.0
        ),
        "max_abs_storage_system_hour_net_diff_vs_hard_mw": max(
            system_storage_errors, default=0.0
        ),
        "max_abs_generation_dispatch_diff_vs_hard_mw": max(
            generation_errors, default=0.0
        ),
    }


def run_fleet(
    fleet_label: str,
    run_dir: Path,
    gammas: tuple[float, ...],
    solver_tol: float,
    max_cpu_time: float,
) -> dict[str, list[dict[str, Any]]]:
    data, investors, degradation, run_config, capacity_rows = _read_fleet(run_dir)
    units = [investor.investor_id for investor in investors]
    price_bound = float(run_config["price_bound_eur_per_mwh"])
    dual_bound = float(run_config["dual_bound_eur_per_mwh"])

    started = time.perf_counter()
    hard_primal = build_primal_llp(
        data,
        degradation_eur_per_mwh_by_unit=degradation,
    )
    hard_primal_results = solve_ipopt(
        hard_primal,
        solver_tol=solver_tol,
        max_cpu_time=max_cpu_time,
        tee=False,
    )
    hard_primal_termination = _termination(hard_primal_results)
    if hard_primal_termination != "optimal":
        raise RuntimeError(
            f"{fleet_label}: hard primal terminated {hard_primal_termination}"
        )
    hard_objective = pyo.value(hard_primal.objective)
    hard_dual = build_tikhonov_dual_llp(
        data,
        degradation_eur_per_mwh_by_unit=degradation,
        gamma=1.0,
        primary_optimum_value=hard_objective,
        price_bound_eur_per_mwh=price_bound,
        other_dual_bound=dual_bound,
    )
    hard_dual_results = solve_ipopt(
        hard_dual,
        solver_tol=solver_tol,
        max_cpu_time=max_cpu_time,
        tee=False,
    )
    hard_dual_termination = _termination(hard_dual_results)
    if hard_dual_termination != "optimal":
        raise RuntimeError(
            f"{fleet_label}: hard minimum-norm dual terminated {hard_dual_termination}"
        )
    hard_seconds = time.perf_counter() - started
    hard_prices = _prices_from_dual(hard_dual)
    hard_storage_net = _aggregate_storage_net(hard_primal, units)
    hard_generation = _generator_dispatch(hard_primal)
    hard_dual_value = pyo.value(hard_dual.unregularized_dual_objective_expr)

    summaries = [
        _case_summary(
            fleet_label=fleet_label,
            case_label="hard_min_norm",
            gamma=None,
            primal=hard_primal,
            primal_termination=hard_primal_termination,
            dual_termination=hard_dual_termination,
            solve_seconds=hard_seconds,
            prices=hard_prices,
            hard_prices=hard_prices,
            hard_storage_net=hard_storage_net,
            hard_generation=hard_generation,
            units=units,
            primal_objective=hard_objective,
            unregularized_dual_objective=hard_dual_value,
            strong_duality_gap=hard_objective - hard_dual_value,
        )
    ]
    economics = _investor_economics(
        fleet_label,
        "hard_min_norm",
        hard_primal,
        hard_prices,
        data,
        investors,
    )
    price_rows, operation_rows, generator_rows = _detail_rows(
        fleet_label,
        "hard_min_norm",
        None,
        hard_primal,
        hard_prices,
        hard_prices,
        units,
    )

    for gamma in gammas:
        case_label = f"gamma_{gamma:.0e}".replace("+", "")
        print(f"{fleet_label}: solving {case_label}", flush=True)
        started = time.perf_counter()
        soft_primal, soft_dual, diagnostics = solve_matched_soft_market(
            data,
            degradation_eur_per_mwh_by_unit=degradation,
            gamma=gamma,
            solver_tol=solver_tol,
            max_cpu_time=max_cpu_time,
            price_bound_eur_per_mwh=price_bound,
            other_dual_bound=dual_bound,
            tee=False,
        )
        solve_seconds = time.perf_counter() - started
        primal_termination = str(diagnostics["primal_termination"])
        dual_termination = str(diagnostics["dual_termination"])
        if primal_termination != "optimal" or dual_termination != "optimal":
            summaries.append(
                {
                    "fleet": fleet_label,
                    "case": case_label,
                    "gamma": gamma,
                    "primal_termination": primal_termination,
                    "dual_termination": dual_termination,
                    "solve_seconds": solve_seconds,
                }
            )
            continue
        soft_prices = soft_balance_prices(soft_primal)
        dual_prices = _prices_from_dual(soft_dual)
        primal_objective = pyo.value(soft_primal.soft_objective)
        dual_objective = pyo.value(soft_dual.regularized_objective)
        summary = _case_summary(
            fleet_label=fleet_label,
            case_label=case_label,
            gamma=gamma,
            primal=soft_primal,
            primal_termination=primal_termination,
            dual_termination=dual_termination,
            solve_seconds=solve_seconds,
            prices=soft_prices,
            hard_prices=hard_prices,
            hard_storage_net=hard_storage_net,
            hard_generation=hard_generation,
            units=units,
            primal_objective=primal_objective,
            unregularized_dual_objective=pyo.value(
                soft_dual.unregularized_dual_objective_expr
            ),
            strong_duality_gap=primal_objective - dual_objective,
        )
        summary["max_abs_soft_primal_vs_dual_lambda_eur_per_mwh"] = max(
            abs(soft_prices[key] - dual_prices[key]) for key in soft_prices
        )
        summary["max_abs_h_plus_gamma_lambda_mw"] = max(
            abs(
                pyo.value(soft_primal.balance_residual[node, hour])
                + gamma * dual_prices[str(node), int(hour)]
            )
            for node in soft_primal.N
            for hour in soft_primal.T
        )
        summaries.append(summary)
        economics.extend(
            _investor_economics(
                fleet_label,
                case_label,
                soft_primal,
                soft_prices,
                data,
                investors,
            )
        )
        new_prices, new_operations, new_generators = _detail_rows(
            fleet_label,
            case_label,
            gamma,
            soft_primal,
            soft_prices,
            hard_prices,
            units,
        )
        price_rows.extend(new_prices)
        operation_rows.extend(new_operations)
        generator_rows.extend(new_generators)

    return {
        "summary": summaries,
        "investor_economics": economics,
        "node_hour_prices": price_rows,
        "storage_operation": operation_rows,
        "generator_dispatch": generator_rows,
        "fixed_fleet": [
            {"fleet": fleet_label, **row} for row in capacity_rows
        ],
    }


def _parse_fleet(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Fleet must use LABEL=RUN_DIRECTORY.")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Fleet label cannot be empty.")
    return label, Path(raw_path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fleet",
        action="append",
        type=_parse_fleet,
        help="LABEL=directory containing final_capacities.csv and run_config.json",
    )
    parser.add_argument(
        "--gamma",
        nargs="+",
        type=float,
        default=list(DEFAULT_GAMMAS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR / "output" / "gamma_sens",
    )
    parser.add_argument("--solver-tol", type=float, default=1.0e-7)
    parser.add_argument("--max-cpu-time", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.solver_tol <= 0.0 or args.max_cpu_time <= 0.0:
        raise SystemExit("Solver tolerance and CPU limit must be positive.")
    gammas = tuple(float(gamma) for gamma in args.gamma)
    if not gammas or any(not math.isfinite(gamma) or gamma <= 0.0 for gamma in gammas):
        raise SystemExit("Every gamma must be finite and positive.")
    fleets = dict(args.fleet) if args.fleet else DEFAULT_FLEETS
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "summary": [],
        "investor_economics": [],
        "node_hour_prices": [],
        "storage_operation": [],
        "generator_dispatch": [],
        "fixed_fleet": [],
    }
    for fleet_label, run_dir in fleets.items():
        print(f"{fleet_label}: reading {run_dir}", flush=True)
        result = run_fleet(
            fleet_label,
            Path(run_dir),
            gammas,
            args.solver_tol,
            args.max_cpu_time,
        )
        for key in combined:
            combined[key].extend(result[key])

    for key, rows in combined.items():
        _write_csv(output_dir / f"{key}.csv", rows)
    run_record = {
        "experiment": "fixed-fleet gamma sensitivity",
        "fleets": {label: str(Path(path).resolve()) for label, path in fleets.items()},
        "gammas": gammas,
        "hard_price_rule": "minimum lambda L2 norm on exact dual-optimal face",
        "solver_tol": args.solver_tol,
        "max_cpu_time_seconds_per_solve": args.max_cpu_time,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(combined["summary"], indent=2), encoding="utf-8"
    )
    print(f"Wrote sensitivity outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
