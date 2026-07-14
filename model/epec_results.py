"""Joint settlement and exports for the diagonalization EPEC.

After (attempted) convergence, one lower-level clearing QP with every
investor's converged fleet produces the settlement prices. Because identical-
efficiency fleets are interchangeable in dispatch, per-investor settled profit
carries a dispatch-attribution ambiguity band: the joint-QP dispatch split
versus a capacity-proportional split of each node's aggregate storage rent.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyomo.environ as pyo

from primal_market_clearing_model import MarketData, value
from single_investor_mpec import (
    QuadraticDemandCurve,
    _solver_dual_cross_check,
    build_quadratic_primal_model,
    capital_recovery_factor,
    quadratic_reference_lambda,
)
from single_investor_mpec_results import _write_csv
from solver_utils import get_ipopt_solver


def _daily_capex(cfg_investor, state, nodes: list[str]) -> float:
    crf_daily = capital_recovery_factor(cfg_investor.wacc, cfg_investor.lifetime_years) / 365.25
    return crf_daily * sum(
        cfg_investor.cost_power_eur_per_mw * state.x_power[cfg_investor.investor_id, n]
        + cfg_investor.cost_energy_eur_per_mwh * state.x_energy[cfg_investor.investor_id, n]
        for n in nodes
    )


def compute_joint_settlement(data: MarketData, quad: QuadraticDemandCurve, state, cfg) -> dict:
    """Clear the market once with all converged fleets and settle every investor."""

    nodes = list(data.nodes)
    units = [inv.investor_id for inv in cfg.investors]
    joint_data = replace(
        data,
        storage_units=units,
        x_power={(i, n): max(0.0, state.x_power[i, n]) for i in units for n in nodes},
        x_energy={(i, n): max(0.0, state.x_energy[i, n]) for i in units for n in nodes},
    )
    reference = build_quadratic_primal_model(joint_data, quad)
    results = get_ipopt_solver().solve(reference, tee=False)
    termination = str(results.solver.termination_condition)
    if termination != "optimal":
        raise RuntimeError(f"Joint settlement QP did not solve optimally (termination={termination}).")

    lam = quadratic_reference_lambda(reference, quad)
    dual_cross_check = _solver_dual_cross_check(reference, lam)

    investors_out: dict[str, dict] = {}
    for inv in cfg.investors:
        i = inv.investor_id
        charge = sum(value(reference.P_charge[i, n, t]) for n in reference.N for t in reference.T)
        discharge = sum(value(reference.P_discharge[i, n, t]) for n in reference.N for t in reference.T)
        revenue = sum(
            lam[n, t] * (value(reference.P_discharge[i, n, t]) - value(reference.P_charge[i, n, t]))
            for n in reference.N
            for t in reference.T
        )
        degradation = 0.5 * inv.degradation_eur_per_mwh * (charge + discharge)
        capex = _daily_capex(inv, state, nodes)
        settled_profit = revenue - degradation - capex

        # Capacity-proportional attribution of each node-hour's aggregate
        # storage rent: the other end of the dispatch-degeneracy band.
        alt_revenue = 0.0
        alt_throughput = 0.0
        for n in reference.N:
            total_power = sum(state.x_power[j, n] for j in units)
            share = state.x_power[i, n] / total_power if total_power > 1e-9 else 0.0
            for t in reference.T:
                agg_net = sum(
                    value(reference.P_discharge[j, n, t]) - value(reference.P_charge[j, n, t]) for j in units
                )
                agg_thru = sum(
                    value(reference.P_discharge[j, n, t]) + value(reference.P_charge[j, n, t]) for j in units
                )
                alt_revenue += share * lam[n, t] * agg_net
                alt_throughput += share * agg_thru
        alt_profit = alt_revenue - 0.5 * inv.degradation_eur_per_mwh * alt_throughput - capex

        belief = next(
            (
                row["profit_belief_eur_per_day"]
                for row in reversed(state.history)
                if row["investor"] == i and row["termination"] == "optimal"
            ),
            float("nan"),
        )
        model = state.final_models.get(i)
        lambda_diff = (
            max(abs(value(model.lam[n, t]) - lam[n, t]) for n in model.N for t in model.T)
            if model is not None
            else None
        )
        investors_out[i] = {
            "wacc": inv.wacc,
            "total_power_mw": sum(state.x_power[i, n] for n in nodes),
            "total_energy_mwh": sum(state.x_energy[i, n] for n in nodes),
            "settled_spot_revenue_eur_per_day": revenue,
            "settled_degradation_eur_per_day": degradation,
            "capex_daily_eur_per_day": capex,
            "settled_profit_eur_per_day": settled_profit,
            "capacity_proportional_profit_eur_per_day": alt_profit,
            "dispatch_attribution_band_eur_per_day": abs(settled_profit - alt_profit),
            "last_profit_belief_eur_per_day": belief,
            "belief_minus_settled_eur_per_day": belief - settled_profit,
            "mpec_lambda_max_abs_diff_vs_joint_eur_per_mwh": lambda_diff,
            "throughput_mwh": charge + discharge,
        }

    node_shares = {
        n: {
            **{i: state.x_power[i, n] for i in units},
            "total_mw": sum(state.x_power[i, n] for i in units),
            "limit_mw": cfg.node_limit_mw,
        }
        for n in nodes
    }
    return {
        "termination": termination,
        "joint_lower_level_objective_eur_per_day": value(reference.quad_objective),
        "lambda_solver_dual_max_abs_diff": dual_cross_check,
        "lambda_min_eur_per_mwh": min(lam.values()),
        "lambda_max_eur_per_mwh": max(lam.values()),
        "investors": investors_out,
        "node_shares": node_shares,
        "reference_lambda": lam,
        "reference_model": reference,
    }


def print_epec_summary(state, cfg, settlement: dict) -> None:
    print("\nEPEC result")
    print(f"  update rule: {cfg.update_rule}, damping: {cfg.damping}")
    print(f"  status: {state.stop_reason}")
    print(f"  projection events: {len(state.projection_events)}")
    print(
        "  joint settlement lambda range: "
        f"{settlement['lambda_min_eur_per_mwh']:,.4f} to {settlement['lambda_max_eur_per_mwh']:,.4f} EUR/MWh"
    )
    for i, row in settlement["investors"].items():
        print(
            f"  {i} (WACC {row['wacc']:.1%}): {row['total_power_mw']:8.2f} MW / {row['total_energy_mwh']:9.2f} MWh"
            f"  settled {row['settled_profit_eur_per_day']:12,.2f} EUR/day"
            f"  (belief-settled {row['belief_minus_settled_eur_per_day']:+10,.2f},"
            f" attribution band {row['dispatch_attribution_band_eur_per_day']:8,.2f})"
        )
    print("  per-node power shares [MW]:")
    for n, shares in settlement["node_shares"].items():
        parts = ", ".join(f"{i}={shares[i]:.2f}" for i in shares if i not in ("total_mw", "limit_mw"))
        print(f"    {n}: {parts}  (total {shares['total_mw']:.2f} / limit {shares['limit_mw']:.0f})")


def export_epec_results(
    output_dir: Path, data: MarketData, state, cfg, settlement: dict, data_path: Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = list(data.nodes)
    units = [inv.investor_id for inv in cfg.investors]

    run_config = {
        "data_path": str(data_path),
        "update_rule": cfg.update_rule,
        "damping": cfg.damping,
        "max_iters": cfg.max_iters,
        "tol_rel": cfg.tol_rel,
        "floor_mw": cfg.floor_mw,
        "floor_mwh": cfg.floor_mwh,
        "seed_power_mw": cfg.seed_power_mw,
        "seed_ratio_hours": cfg.seed_ratio_hours,
        "node_limit_mw": cfg.node_limit_mw,
        "max_cpu_time": cfg.max_cpu_time,
        "dual_bound_scale": cfg.dual_bound_scale,
        "investors": [
            {
                "investor_id": inv.investor_id,
                "wacc": inv.wacc,
                "lifetime_years": inv.lifetime_years,
                "cost_power_eur_per_mw": inv.cost_power_eur_per_mw,
                "cost_energy_eur_per_mwh": inv.cost_energy_eur_per_mwh,
                "degradation_eur_per_mwh": inv.degradation_eur_per_mwh,
                "ratio_min": inv.ratio_min,
                "ratio_max": inv.ratio_max,
            }
            for inv in cfg.investors
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    _write_csv(
        output_dir / "iteration_history.csv",
        [
            "iteration",
            "investor",
            "termination",
            "solve_seconds",
            "profit_belief_eur_per_day",
            "strong_duality_gap",
            "total_power_mw",
            "total_energy_mwh",
            "max_rel_delta_power",
            "max_rel_delta_energy",
            "max_undamped_delta_power_mw",
        ],
        state.history,
    )
    _write_csv(
        output_dir / "capacity_trajectory.csv",
        ["iteration", "investor", "node", "x_power_mw", "x_energy_mwh", "proposed_x_power_mw", "headroom_mw"],
        state.trajectory,
    )
    _write_csv(
        output_dir / "projection_events.csv",
        ["iteration", "node", "total_before_mw", "scale"],
        state.projection_events,
    )
    _write_csv(
        output_dir / "final_capacities.csv",
        ["investor", "node", "x_power_mw", "x_energy_mwh", "ratio_hours", "share_of_node_limit"],
        [
            {
                "investor": i,
                "node": n,
                "x_power_mw": state.x_power[i, n],
                "x_energy_mwh": state.x_energy[i, n],
                "ratio_hours": state.x_energy[i, n] / state.x_power[i, n] if state.x_power[i, n] > 1e-9 else 0.0,
                "share_of_node_limit": state.x_power[i, n] / cfg.node_limit_mw,
            }
            for i in units
            for n in nodes
        ],
    )

    reference: pyo.ConcreteModel = settlement["reference_model"]
    lam: dict[tuple[str, int], float] = settlement["reference_lambda"]
    _write_csv(
        output_dir / "joint_node_hour_prices.csv",
        ["hour", "node", "lambda_joint_eur_per_mwh"],
        [
            {"hour": t, "node": n, "lambda_joint_eur_per_mwh": lam[n, t]}
            for t in reference.T
            for n in reference.N
        ],
    )
    _write_csv(
        output_dir / "joint_storage_hour_operation.csv",
        ["unit", "hour", "node", "p_charge_mw", "p_discharge_mw", "net_injection_mw", "lambda_joint_eur_per_mwh", "spot_revenue_eur"],
        [
            {
                "unit": i,
                "hour": t,
                "node": n,
                "p_charge_mw": value(reference.P_charge[i, n, t]),
                "p_discharge_mw": value(reference.P_discharge[i, n, t]),
                "net_injection_mw": value(reference.P_discharge[i, n, t]) - value(reference.P_charge[i, n, t]),
                "lambda_joint_eur_per_mwh": lam[n, t],
                "spot_revenue_eur": lam[n, t]
                * (value(reference.P_discharge[i, n, t]) - value(reference.P_charge[i, n, t])),
            }
            for i in reference.I
            for t in reference.T
            for n in reference.N
        ],
    )

    settlement_json = {k: v for k, v in settlement.items() if k not in ("reference_model", "reference_lambda")}
    (output_dir / "joint_settlement.json").write_text(json.dumps(settlement_json, indent=2), encoding="utf-8")

    summary = {
        "converged": state.converged,
        "stop_reason": state.stop_reason,
        "iterations": state.iteration,
        "update_rule": cfg.update_rule,
        "damping": cfg.damping,
        "tol_rel": cfg.tol_rel,
        "projection_event_count": len(state.projection_events),
        "investors": settlement_json["investors"],
        "node_shares": settlement_json["node_shares"],
        "joint_lambda_min_eur_per_mwh": settlement["lambda_min_eur_per_mwh"],
        "joint_lambda_max_eur_per_mwh": settlement["lambda_max_eur_per_mwh"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    from single_investor_mpec_results import export_solution

    for i, model in state.final_models.items():
        if model is None:
            continue
        export_solution(model, output_dir / f"investor_{i}", "ok", "optimal", None)
