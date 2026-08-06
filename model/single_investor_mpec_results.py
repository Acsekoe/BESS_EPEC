"""Reporting and export helpers for the single-investor MPEC."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyomo.environ as pyo

from primal_market_clearing_model import MarketData, value


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _line_flow(model: pyo.ConcreteModel, line: str, time: int) -> float:
    data: MarketData = model._market_data
    return sum(data.ptdf[line, node] * value(model.NetInjection[node, time]) for node in model.N)


def _strong_duality_gap(model: pyo.ConcreteModel) -> float:
    return abs(value(model.primal_objective_expr) - value(model.dual_objective_expr))


def _headroom_shadow_price(model: pyo.ConcreteModel, node: str) -> float:
    raw = float(model.dual.get(model.investment_headroom[node], float("nan")))
    return 0.0 if abs(raw) < 1e-8 else raw


def _storage_pairs(model: pyo.ConcreteModel) -> list[tuple[str, str]]:
    return list(model.IN) if hasattr(model, "IN") else [(i, n) for i in model.I for n in model.N]


def _storage_units_at_node(model: pyo.ConcreteModel, node: str) -> list[str]:
    return [unit for unit, pair_node in _storage_pairs(model) if pair_node == node]


def _max_abs_components(
    model: pyo.ConcreteModel, component_names: tuple[str, ...]
) -> tuple[str | None, float]:
    largest_name: str | None = None
    largest_value = 0.0
    for component_name in component_names:
        component = getattr(model, component_name, None)
        if component is None:
            continue
        for index, variable in component.items():
            magnitude = abs(value(variable))
            if magnitude > largest_value:
                largest_name = f"{component_name}[{index}]"
                largest_value = magnitude
    return largest_name, largest_value


PRICE_DUAL_COMPONENTS = ("lam", "lam_sys")
INTERNAL_DUAL_COMPONENTS = (
        "nu_gen",
        "mu_up",
        "mu_dn",
        "rho_ch",
        "sig_dis",
        "gam",
        "del_soc",
        "rho_per",
        "xi_shed",
)


def export_solution(
    model: pyo.ConcreteModel,
    output_dir: Path,
    solver_status: str,
    termination: str,
    reference_settlement: dict[str, object] | None = None,
) -> None:
    """Write the active quadratic MPEC solution to CSV/JSON files."""

    data: MarketData = model._market_data
    output_dir.mkdir(parents=True, exist_ok=True)

    investor_id = model._investor_id
    quad = model._quad_demand
    use_demand_curve = getattr(model, "_use_demand_curve", True)
    degradation_cost = model._degradation_eur_per_mwh
    max_price_name, max_price_value = _max_abs_components(model, PRICE_DUAL_COMPONENTS)
    max_internal_name, max_internal_value = _max_abs_components(model, INTERNAL_DUAL_COMPONENTS)
    if max_price_value >= max_internal_value:
        max_dual_name, max_dual_value = max_price_name, max_price_value
        max_dual_bound = model._price_bound_eur_per_mwh
    else:
        max_dual_name, max_dual_value = max_internal_name, max_internal_value
        max_dual_bound = model._dual_bound_eur_per_mwh

    total_charge = sum(value(model.P_charge[investor_id, n, t]) for n in model.N for t in model.T)
    total_discharge = sum(value(model.P_discharge[investor_id, n, t]) for n in model.N for t in model.T)
    total_shed = (
        sum(value(model.P_shed[n, t]) for n in model.N for t in model.T)
        if use_demand_curve
        else 0.0
    )
    total_power = sum(value(model.X_power[n]) for n in model.N)
    total_energy = sum(value(model.X_energy[n]) for n in model.N)
    throughput = total_charge + total_discharge

    summary = {
        "model": model.name,
        "investor": investor_id,
        "wacc": model._wacc,
        "solver_status": solver_status,
        "termination": termination,
        "profit_eur_per_day": value(model.investor_profit_expr),
        "spot_revenue_eur_per_day": value(model.spot_revenue_expr),
        "degradation_cost_eur_per_day": value(model.degradation_cost_expr),
        "capex_daily_eur_per_day": value(model.capex_daily_expr),
        "access_shadow_price_eur_per_mw_day": {
            n: _headroom_shadow_price(model, n) for n in model.N
        },
        "lower_level_primal_objective_eur_per_day": value(model.primal_objective_expr),
        "lower_level_dual_objective_eur_per_day": value(model.dual_objective_expr),
        "strong_duality_gap": _strong_duality_gap(model),
        "total_power_mw": total_power,
        "total_energy_mwh": total_energy,
        "total_charge_mwh": total_charge,
        "total_discharge_mwh": total_discharge,
        "total_storage_throughput_mwh": throughput,
        "total_load_shed_mwh": total_shed,
        "equivalent_cycles_throughput_over_2e": throughput / (2.0 * total_energy) if total_energy > 0.0 else 0.0,
        "equivalent_cycles_discharge_over_e": total_discharge / total_energy if total_energy > 0.0 else 0.0,
        "line_limits_mw": {line: data.line_limit[line] for line in data.lines},
        "existing_power_mw_per_node": model._existing_power_mw,
        "existing_ratio_hours": model._existing_ratio_hours,
        "storage_units": list(model.I),
        "rival_representation": "separate_battery_per_investor_with_nodal_mw_mwh",
        "rival_power_mw_by_unit": model._rival_power_mw_by_unit,
        "rival_energy_mwh_by_unit": model._rival_energy_mwh_by_unit,
        "demand_model": "quadratic" if use_demand_curve else "fixed_no_load_shed",
        "load_shed_allowed": use_demand_curve,
        "dispatch_regularization_eur_per_mw2h": model._dispatch_regularization_eur_per_mw2h,
        "solver_tol": model._solver_tol,
        "price_bound_eur_per_mwh": model._price_bound_eur_per_mwh,
        "dual_bound_eur_per_mwh": model._dual_bound_eur_per_mwh,
        "lambda_l2_penalty_coefficient": model._lambda_l2_penalty_coefficient,
        "lambda_l2_norm": value(model.lambda_l2_norm_expr),
        "lambda_l2_penalty_eur_per_day": value(model.lambda_l2_penalty_expr),
        "penalized_objective_eur_per_day": value(model.penalized_investor_objective_expr),
        "max_abs_price_dual": max_price_value,
        "max_abs_price_dual_name": max_price_name,
        "max_abs_price_dual_fraction_of_bound": (
            max_price_value / model._price_bound_eur_per_mwh
        ),
        "max_abs_internal_dual": max_internal_value,
        "max_abs_internal_dual_name": max_internal_name,
        "max_abs_internal_dual_fraction_of_bound": (
            max_internal_value / model._dual_bound_eur_per_mwh
        ),
        "max_abs_embedded_dual": max_dual_value,
        "max_abs_embedded_dual_name": max_dual_name,
        "max_abs_embedded_dual_applied_bound": max_dual_bound,
        "max_abs_embedded_dual_fraction_of_bound": max_dual_value / max_dual_bound,
        "embedded_storage_node_pairs": len(_storage_pairs(model)),
        "sparse_rival_capacity_tol_mw_mwh": model._sparse_rival_capacity_tol,
        "embedded_positive_capacity_generator_hours": len(model.GT),
        "sparse_generation_capacity_tol_mw": model._sparse_generation_capacity_tol,
        "quadratic_demand_alpha_eur_per_mwh": quad.alpha if use_demand_curve else None,
        "quadratic_demand_beta_eur_per_mwh_per_share": quad.beta if use_demand_curve else None,
        "quadratic_demand_source": "fixed_default" if use_demand_curve else None,
    }
    lower_level_optimality = getattr(model, "_lower_level_optimality", "strong-duality")
    if lower_level_optimality == "relaxed-kkt":
        from single_investor_mpec_relaxed_kkt import relaxed_kkt_diagnostics

        summary["lower_level_optimality"] = "relaxed-kkt"
        summary["relaxed_kkt_complementarity"] = relaxed_kkt_diagnostics(model)
    elif lower_level_optimality == "iso-min-norm-dual":
        from single_investor_mpec_min_norm_prices import min_norm_price_diagnostics

        summary["lower_level_optimality"] = "iso-min-norm-dual"
        summary["iso_min_norm_price_selection"] = min_norm_price_diagnostics(model)
    else:
        summary["lower_level_optimality"] = "strong-duality"

    if reference_settlement is not None:
        summary.update(
            {
                "reference_settlement_solver": reference_settlement["solver"],
                "reference_settlement_problem": reference_settlement.get("problem"),
                "reference_settlement_solver_status": reference_settlement["solver_status"],
                "reference_settlement_termination": reference_settlement["termination"],
                "reference_lower_level_objective_eur_per_day": reference_settlement[
                    "lower_level_objective_eur_per_day"
                ],
                "mpec_lambda_max_abs_diff_vs_reference_eur_per_mwh": reference_settlement[
                    "mpec_lambda_max_abs_diff_vs_reference_eur_per_mwh"
                ],
                "reference_lambda_solver_dual_max_abs_diff": reference_settlement[
                    "reference_lambda_solver_dual_max_abs_diff"
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
        [
            "node",
            "x_power_mw",
            "x_energy_mwh",
            "energy_power_ratio_h",
            "private_headroom_limit_mw",
            "private_headroom_slack_mw",
            "headroom_binding",
            "access_shadow_price_eur_per_mw_day",
            "headroom_complementarity_residual_eur_per_day",
        ],
        [
            {
                "node": n,
                "x_power_mw": value(model.X_power[n]),
                "x_energy_mwh": value(model.X_energy[n]),
                "energy_power_ratio_h": value(model.X_energy[n]) / value(model.X_power[n])
                if value(model.X_power[n]) > 1e-9
                else 0.0,
                "private_headroom_limit_mw": model._invest_limit_mw[n],
                "private_headroom_slack_mw": max(
                    0.0, model._invest_limit_mw[n] - value(model.X_power[n])
                ),
                "headroom_binding": abs(
                    model._invest_limit_mw[n] - value(model.X_power[n])
                ) <= 1e-5,
                "access_shadow_price_eur_per_mw_day": _headroom_shadow_price(model, n),
                "headroom_complementarity_residual_eur_per_day": (
                    _headroom_shadow_price(model, n)
                    * max(0.0, model._invest_limit_mw[n] - value(model.X_power[n]))
                ),
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
                "generation_mw": sum(
                    value(model.P_gen[g, t])
                    for g in data.generators_at_node.get(n, [])
                    if (g, t) in model.GT
                ),
                "p_charge_mw": sum(value(model.P_charge[i, n, t]) for i in _storage_units_at_node(model, n)),
                "p_discharge_mw": sum(value(model.P_discharge[i, n, t]) for i in _storage_units_at_node(model, n)),
                "storage_net_injection_mw": sum(
                    value(model.P_discharge[i, n, t]) - value(model.P_charge[i, n, t])
                    for i in _storage_units_at_node(model, n)
                ),
                "load_shed_mw": value(model.P_shed[n, t]) if use_demand_curve else 0.0,
                "net_injection_mw": value(model.NetInjection[n, t]),
                "lambda_eur_per_mwh": value(model.lam[n, t]),
            }
            for t in model.T
            for n in model.N
        ],
    )

    _write_csv(
        output_dir / "shed_hour_duals.csv",
        ["hour", "node", "shed_mw", "bound_mw", "marginal_wtp_eur_per_mwh", "xi_shed_dual"],
        [
            {
                "hour": t,
                "node": n,
                "shed_mw": value(model.P_shed[n, t]) if use_demand_curve else 0.0,
                "bound_mw": data.demand_el[n, t],
                "marginal_wtp_eur_per_mwh": quad.marginal_wtp(value(model.P_shed[n, t]), data.demand_el[n, t])
                if use_demand_curve
                else None,
                "xi_shed_dual": value(model.xi_shed[n, t]) if use_demand_curve else None,
            }
            for t in model.T
            for n in model.N
        ],
    )

    _write_csv(
        output_dir / "generator_hour_dispatch_duals.csv",
        ["hour", "generator", "dispatch_mw", "capacity_mw", "marginal_cost_eur_per_mwh", "nu_gen_dual"],
        [
            {
                "hour": t,
                "generator": g,
                "dispatch_mw": value(model.P_gen[g, t]) if (g, t) in model.GT else 0.0,
                "capacity_mw": data.generation_capacity[g, t],
                "marginal_cost_eur_per_mwh": data.generation_cost[g],
                "nu_gen_dual": value(model.nu_gen[g, t]) if (g, t) in model.GT else None,
            }
            for t in model.T
            for g in model.G
        ],
    )

    _write_csv(
        output_dir / "line_hour_flows_duals.csv",
        ["hour", "line", "flow_mw", "limit_mw", "abs_utilization", "mu_upper_dual", "mu_lower_dual"],
        [
            {
                "hour": t,
                "line": l,
                "flow_mw": _line_flow(model, l, t),
                "limit_mw": data.line_limit[l],
                "abs_utilization": abs(_line_flow(model, l, t)) / data.line_limit[l]
                if data.line_limit[l] > 0.0
                else 0.0,
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
                * degradation_cost
                * (value(model.P_charge[i, n, t]) + value(model.P_discharge[i, n, t])),
                "rho_charge_dual": value(model.rho_ch[i, n, t]),
                "sigma_discharge_dual": value(model.sig_dis[i, n, t]),
                "gamma_soc_transition_dual": value(model.gam[i, n, t]),
            }
            for i, n in _storage_pairs(model)
            for t in model.T
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
            for i, n in _storage_pairs(model)
            for tau in model.T_SOC
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

    if reference_settlement is None:
        return

    reference: pyo.ConcreteModel = reference_settlement["reference_model"]  # type: ignore[assignment]
    reference_lambda: dict[tuple[str, int], float] = reference_settlement["reference_lambda"]  # type: ignore[assignment]

    _write_csv(
        output_dir / "reference_node_hour_prices.csv",
        ["hour", "node", "lambda_reference_eur_per_mwh"],
        [
            {"hour": t, "node": n, "lambda_reference_eur_per_mwh": reference_lambda[n, t]}
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
                * degradation_cost
                * (value(reference.P_charge[i, n, t]) + value(reference.P_discharge[i, n, t])),
            }
            for i in reference.I
            for t in reference.T
            for n in reference.N
        ],
    )


def print_solution_summary(model: pyo.ConcreteModel, reference_settlement: dict[str, object] | None = None) -> None:
    print("\nSingle-investor MPEC solution")
    print(f"  investor: {model._investor_id}")
    print(f"  WACC: {model._wacc:.2%}")
    print(f"  profit: {value(model.investor_profit_expr):,.4f} EUR/day")
    print(f"  spot revenue: {value(model.spot_revenue_expr):,.4f} EUR/day")
    print(f"  degradation cost: {value(model.degradation_cost_expr):,.4f} EUR/day")
    print(f"  CAPEX daily: {value(model.capex_daily_expr):,.4f} EUR/day")
    print(f"  lower-level primal objective: {value(model.primal_objective_expr):,.4f}")
    print(f"  lower-level dual objective: {value(model.dual_objective_expr):,.4f}")
    print(f"  strong-duality gap: {_strong_duality_gap(model):.6e}")

    print("\nInvestment by node")
    for node in model.N:
        print(
            f"  {node}: X_power={value(model.X_power[node]):9.4f} MW, "
            f"X_energy={value(model.X_energy[node]):9.4f} MWh, "
            f"access shadow={_headroom_shadow_price(model, node):9.4f} EUR/MW/day"
        )

    if reference_settlement is None:
        return

    print("\nReference-price settlement")
    print(f"  solver: {reference_settlement['solver']} ({reference_settlement.get('problem', 'QP')})")
    problem = reference_settlement.get("problem", "QP")
    print(f"  reference {problem} dispatch profit: {reference_settlement['profit_at_reference_prices_eur_per_day']:,.4f} EUR/day")
    print(
        f"  reference {problem} dispatch spot revenue: "
        f"{reference_settlement['spot_revenue_at_reference_prices_eur_per_day']:,.4f} EUR/day"
    )
    print(
        "  MPEC dispatch at reference prices profit: "
        f"{reference_settlement['mpec_dispatch_profit_at_reference_prices_eur_per_day']:,.4f} EUR/day"
    )
    print(
        f"  optimistic - reference {problem} profit gap: "
        f"{reference_settlement['optimistic_minus_reference_profit_eur_per_day']:,.4f} EUR/day"
    )
    print(
        "  reference lambda range: "
        f"{reference_settlement['reference_lambda_min_eur_per_mwh']:,.4f} to "
        f"{reference_settlement['reference_lambda_max_eur_per_mwh']:,.4f} EUR/MWh"
    )
    print(
        "  max |lambda_MPEC - lambda_reference|: "
        f"{reference_settlement['mpec_lambda_max_abs_diff_vs_reference_eur_per_mwh']:,.6f} EUR/MWh"
    )
    if reference_settlement["reference_lambda_solver_dual_max_abs_diff"] is not None:
        print(
            "  reference lambda solver-dual cross-check: "
            f"{reference_settlement['reference_lambda_solver_dual_max_abs_diff']:,.6f} EUR/MWh"
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
            flow = _line_flow(model, line, time)
            limit = data.line_limit[line]
            mu_upper = value(model.mu_up[line, time])
            mu_lower = value(model.mu_dn[line, time])
            print(
                f"  hour={time:>2}, line={line}: "
                f"flow={flow:12.6f} MW, limit={limit:12.6f} MW, "
                f"mu_upper={mu_upper:12.6f}, mu_lower={mu_lower:12.6f}, "
                f"mu_net={mu_upper + mu_lower:12.6f}"
            )
