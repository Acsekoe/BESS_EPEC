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
    build_fixed_demand_primal_model,
    build_quadratic_primal_model,
    capital_recovery_factor,
    fixed_demand_reference_lambda,
    investment_headroom_shadow_price,
    quadratic_reference_lambda,
    reference_system_price,
)
from single_investor_mpec_results import _write_csv
from solver_utils import get_ipopt_solver


#### Export schemas
# -----------------------------------------------------------------------------

ITERATION_HISTORY_FIELDS = [
    "iteration",
    "regularization_stage",
    "dual_tikhonov_gamma",
    "proximal_coefficient_eur_per_mw2_day",
    "complementarity_epsilon",
    "investor",
    "termination",
    "solve_seconds",
    "preparation_seconds",
    "attempt_count",
    "retry_status",
    "optimistic_mpec_profit_eur_per_day",
    "max_access_shadow_price_eur_per_mw_day",
    "strong_duality_gap",
    "max_complementarity_product",
    "sum_complementarity_products",
    "max_original_nodal_balance_residual_mw",
    "total_absolute_original_nodal_balance_residual_mwh",
    "max_hourly_system_balance_residual_mw",
    "signed_primal_dual_gap_eur_per_day",
    "total_power_mw",
    "total_energy_mwh",
    "max_rel_delta_power",
    "max_rel_delta_energy",
    "abs_capacity_step_mw_equivalent",
    "abs_offer_step_mw",
    "abs_price_step_eur_per_mwh",
    "converged_sweep_streak",
    "max_undamped_delta_power_mw",
    "max_undamped_delta_energy_mwh",
    "max_undamped_rel_delta_power",
    "max_undamped_rel_delta_energy",
    "undamped_capacity_residual_mw_equivalent",
    "undamped_offer_residual_mw",
    "undamped_price_residual_eur_per_mwh",
]

CAPACITY_TRAJECTORY_FIELDS = [
    "iteration",
    "investor",
    "node",
    "x_power_mw",
    "x_energy_mwh",
    "proposed_x_power_mw",
    "private_headroom_limit_mw",
    "private_headroom_slack_mw",
    "access_shadow_price_eur_per_mw_day",
    "headroom_complementarity_residual_eur_per_day",
    "headroom_mw",
]

#### Joint market settlement
# -----------------------------------------------------------------------------

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
    degradation = {inv.investor_id: inv.degradation_eur_per_mwh for inv in cfg.investors}
    if cfg.use_demand_curve:
        reference = build_quadratic_primal_model(
            joint_data,
            quad,
            storage_degradation_eur_per_mwh=degradation,
            dispatch_regularization_eur_per_mw2h=cfg.dispatch_regularization_eur_per_mw2h,
        )
    else:
        reference = build_fixed_demand_primal_model(
            joint_data,
            storage_degradation_eur_per_mwh=degradation,
            dispatch_regularization_eur_per_mw2h=cfg.dispatch_regularization_eur_per_mw2h,
            demand_expansion=getattr(cfg, "demand_expansion", None),
        )
    results = get_ipopt_solver(
        {
            "max_cpu_time": cfg.max_cpu_time,
            "tol": cfg.solver_tol,
            "acceptable_tol": cfg.solver_tol,
        }
    ).solve(reference, tee=False)
    termination = str(results.solver.termination_condition)
    if termination != "optimal":
        raise RuntimeError(f"Joint settlement QP did not solve optimally (termination={termination}).")

    lam = (
        quadratic_reference_lambda(reference, quad)
        if cfg.use_demand_curve
        else fixed_demand_reference_lambda(reference)
    )
    dual_cross_check = _solver_dual_cross_check(reference, lam)

    # Settlement price: nodal LMP by default, or the uniform per-hour system
    # price (broadcast to every node) when the run uses zonal settlement, so
    # the reported profit matches the price each investor optimized against.
    if cfg.system_price_settlement:
        sys_price = reference_system_price(reference, lam)
        settle_price = {(n, t): sys_price[t] for n in reference.N for t in reference.T}
    else:
        settle_price = lam

    # Generator -> its node, for crediting portfolio investors their owned
    # share of each existing generator's inframarginal rent at settlement prices.
    gen_node = {g: n for n in nodes for g in data.generators_at_node.get(n, [])}

    investors_out: dict[str, dict] = {}
    for inv in cfg.investors:
        i = inv.investor_id
        charge = sum(value(reference.P_charge[i, n, t]) for n in reference.N for t in reference.T)
        discharge = sum(value(reference.P_discharge[i, n, t]) for n in reference.N for t in reference.T)
        revenue = sum(
            settle_price[n, t]
            * (value(reference.P_discharge[i, n, t]) - value(reference.P_charge[i, n, t]))
            for n in reference.N
            for t in reference.T
        )
        generation_rent = 0.0
        for g, share in inv.owned_generation_shares.items():
            n = gen_node.get(g)
            if n is None or share == 0.0:
                continue
            mc = data.generation_cost[g]
            generation_rent += share * sum(
                (settle_price[n, t] - mc) * value(reference.P_gen[g, t]) for t in reference.T
        )
        degradation = 0.5 * inv.degradation_eur_per_mwh * (charge + discharge)
        capex = _daily_capex(inv, state, nodes)
        settled_profit = revenue + generation_rent - degradation - capex

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
                alt_revenue += share * settle_price[n, t] * agg_net
                alt_throughput += share * agg_thru
        # Generation rent is unambiguously owned (one generator, one investor),
        # so it enters both attribution variants unchanged.
        alt_profit = (
            alt_revenue
            + generation_rent
            - 0.5 * inv.degradation_eur_per_mwh * alt_throughput
            - capex
        )

        optimistic_profit = next(
            (
                row["optimistic_mpec_profit_eur_per_day"]
                for row in reversed(state.history)
                if row["investor"] == i and row["termination"] == "optimal"
            ),
            float("nan"),
        )
        model = state.final_models.get(i)
        if model is None:
            selected_prices = getattr(state, "final_selected_prices", {}).get(i)
            access_shadow_prices = getattr(
                state, "final_access_shadow_prices", {}
            ).get(i)
            lambda_diff = (
                max(
                    abs(selected_prices[n, int(t)] - settle_price[n, t])
                    for n in nodes
                    for t in reference.T
                )
                if selected_prices
                else None
            )
        elif cfg.system_price_settlement:
            access_shadow_prices = {
                n: investment_headroom_shadow_price(model, n) for n in model.N
            }
            lambda_diff = max(
                abs(value(model.lam_sys[t]) - settle_price[n, t])
                for n in model.N
                for t in model.T
            )
        else:
            access_shadow_prices = {
                n: investment_headroom_shadow_price(model, n) for n in model.N
            }
            lambda_diff = max(
                abs(value(model.lam[n, t]) - settle_price[n, t]) for n in model.N for t in model.T
            )
        investors_out[i] = {
            "wacc": inv.wacc,
            "total_power_mw": sum(state.x_power[i, n] for n in nodes),
            "total_energy_mwh": sum(state.x_energy[i, n] for n in nodes),
            "settled_spot_revenue_eur_per_day": revenue,
            "settled_generation_rent_eur_per_day": generation_rent,
            "owned_generation_shares": dict(inv.owned_generation_shares),
            "settled_degradation_eur_per_day": degradation,
            "capex_daily_eur_per_day": capex,
            "settled_profit_eur_per_day": settled_profit,
            "capacity_proportional_profit_eur_per_day": alt_profit,
            "dispatch_attribution_band_eur_per_day": abs(settled_profit - alt_profit),
            "last_optimistic_mpec_profit_eur_per_day": optimistic_profit,
            "optimistic_mpec_minus_settled_eur_per_day": optimistic_profit - settled_profit,
            "mpec_lambda_max_abs_diff_vs_joint_eur_per_mwh": lambda_diff,
            "last_best_response_access_shadow_price_eur_per_mw_day": access_shadow_prices,
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
    max_node_overload = max(
        max(0.0, shares["total_mw"] - shares["limit_mw"])
        for shares in node_shares.values()
    )
    simultaneous_dispatch = [
        min(
            value(reference.P_charge[i, n, t]),
            value(reference.P_discharge[i, n, t]),
        )
        for i in reference.I
        for n in reference.N
        for t in reference.T
    ]
    return {
        "termination": termination,
        "joint_lower_level_objective_eur_per_day": value(
            reference.quad_objective if cfg.use_demand_curve else reference.objective
        ),
        "lambda_solver_dual_max_abs_diff": dual_cross_check,
        "lambda_min_eur_per_mwh": min(settle_price.values()),
        "lambda_max_eur_per_mwh": max(settle_price.values()),
        "settlement_price_basis": "system" if cfg.system_price_settlement else "nodal",
        "investors": investors_out,
        "node_shares": node_shares,
        "max_node_overload_mw": max_node_overload,
        "shared_limit_feasible": max_node_overload <= 1e-6,
        "simultaneous_charge_discharge_mwh": sum(simultaneous_dispatch),
        "max_simultaneous_charge_discharge_mw": max(simultaneous_dispatch, default=0.0),
        "reference_lambda": settle_price,
        "reference_model": reference,
    }


def print_epec_summary(state, cfg, settlement: dict) -> None:
    print("\nEPEC result")
    print(f"  update rule: {cfg.update_rule}, damping: {cfg.damping}")
    print(f"  status: {state.stop_reason}")
    print(f"  projection events: {len(state.projection_events)}")
    if not state.converged:
        print("  WARNING: this is a diagnostic settlement of a nonconverged iterate, not an equilibrium result.")
    if not settlement["shared_limit_feasible"]:
        print(
            "  WARNING: diagnostic settlement uses a fleet that exceeds a shared nodal limit by "
            f"up to {settlement['max_node_overload_mw']:.3f} MW; it is not a feasible equilibrium."
        )
    print(
        "  joint settlement lambda range: "
        f"{settlement['lambda_min_eur_per_mwh']:,.4f} to {settlement['lambda_max_eur_per_mwh']:,.4f} EUR/MWh"
    )
    if settlement.get("max_simultaneous_charge_discharge_mw", 0.0) > 1e-5:
        print(
            "  WARNING: joint re-clear has simultaneous charge/discharge up to "
            f"{settlement['max_simultaneous_charge_discharge_mw']:.6f} MW "
            f"({settlement['simultaneous_charge_discharge_mwh']:.6f} MWh total)."
        )
    for i, row in settlement["investors"].items():
        gen_rent = row.get("settled_generation_rent_eur_per_day", 0.0)
        gen_str = f" gen rent {gen_rent:11,.2f}," if gen_rent else ""
        print(
            f"  {i} (WACC {row['wacc']:.1%}): {row['total_power_mw']:8.2f} MW / {row['total_energy_mwh']:9.2f} MWh"
            f"  settled {row['settled_profit_eur_per_day']:12,.2f} EUR/day"
            f" ({gen_str} optimistic-settled {row['optimistic_mpec_minus_settled_eur_per_day']:+10,.2f},"
            f" attribution band {row['dispatch_attribution_band_eur_per_day']:8,.2f})"
        )
        shadow_prices = row.get("last_best_response_access_shadow_price_eur_per_mw_day") or {}
        positive_shadows = {n: price for n, price in shadow_prices.items() if price > 1e-6}
        if positive_shadows:
            formatted = ", ".join(f"{n}={price:,.2f}" for n, price in positive_shadows.items())
            print(f"      endogenous access shadow values [EUR/MW/day]: {formatted}")
    print("  per-node power shares [MW]:")
    non_investor_keys = ("total_mw", "limit_mw")
    for n, shares in settlement["node_shares"].items():
        parts = ", ".join(f"{i}={shares[i]:.2f}" for i in shares if i not in non_investor_keys)
        print(f"    {n}: {parts}  (total {shares['total_mw']:.2f} / limit {shares['limit_mw']:.0f})")


#### Checkpoint export
# -----------------------------------------------------------------------------

def _model_variant(cfg) -> str:
    clean = (
        cfg.lower_level_optimality == "strong-duality"
        and not cfg.use_demand_curve
        and getattr(cfg, "demand_expansion", None) is None
        and cfg.dispatch_regularization_eur_per_mw2h == 0.0
        and cfg.lambda_l2_penalty_coefficient == 0.0
    )
    return (
        "clean-fixed-demand-exact-strong-duality"
        if clean
        else f"experimental-{cfg.lower_level_optimality}"
    )


def _demand_metadata(cfg) -> dict[str, float | str | None]:
    expansion = getattr(cfg, "demand_expansion", None)
    if cfg.use_demand_curve:
        label = "quadratic-curtailment"
    elif expansion is not None:
        label = "quadratic-expansion"
    else:
        label = "fixed"
    return {
        "demand_model": label,
        "demand_expansion_reference_price_eur_per_mwh": (
            getattr(expansion, "reference_price_eur_per_mwh", None)
        ),
        "demand_expansion_elasticity": getattr(expansion, "elasticity", None),
    }

def export_epec_checkpoint(output_dir: Path, state, cfg) -> None:
    """Persist lightweight traces after every completed EPEC iteration.

    This deliberately avoids the joint market settlement and per-investor
    model exports. If a run is interrupted during a later solve, the files
    retain the last fully completed iteration and can be inspected directly.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "iteration_history.csv", ITERATION_HISTORY_FIELDS, state.history)
    _write_csv(output_dir / "capacity_trajectory.csv", CAPACITY_TRAJECTORY_FIELDS, state.trajectory)
    nodes = sorted({node for _, node in state.x_power})
    node_total_power = {
        node: sum(state.x_power[inv.investor_id, node] for inv in cfg.investors)
        for node in nodes
    }
    node_excess = {node: total - cfg.node_limit_mw for node, total in node_total_power.items()}
    checkpoint = {
        "model_variant": _model_variant(cfg),
        **_demand_metadata(cfg),
        "status": state.stop_reason or f"in progress after iteration {state.iteration}",
        "converged": state.converged,
        "iteration": state.iteration,
        "starting_iteration": cfg.starting_iteration,
        "resume_from": cfg.resume_from,
        "update_rule": cfg.update_rule,
        "parallel_workers": cfg.strategic_parallel_workers,
        "investor_solve_order": [inv.investor_id for inv in cfg.investors],
        "initialization_method": state.initialization_method,
        "initializer_summary": state.initializer_summary,
        "damping": cfg.damping,
        "capacity_cleanup_tol_mw_mwh": cfg.capacity_cleanup_tol_mw_mwh,
        "rival_sparsity_tol_mw": (
            None if hasattr(state, "offer_charge") else cfg.rival_sparsity_tol_mw
        ),
        "price_bound_eur_per_mwh": cfg.price_bound_eur_per_mwh,
        "dual_bound_eur_per_mwh": cfg.dual_bound_eur_per_mwh,
        "lambda_l2_penalty_coefficient": cfg.lambda_l2_penalty_coefficient,
        "lower_level_optimality": cfg.lower_level_optimality,
        "iso_min_norm_complementarity_epsilon": (
            cfg.iso_min_norm_complementarity_epsilon
            if cfg.lower_level_optimality == "iso-min-norm-dual"
            else None
        ),
        "complementarity_epsilon": (
            cfg.complementarity_epsilon
            if cfg.lower_level_optimality == "relaxed-kkt"
            else None
        ),
        "complementarity_formulation": (
            cfg.complementarity_formulation
            if cfg.lower_level_optimality == "relaxed-kkt"
            else None
        ),
        "complementarity_shift": (
            cfg.complementarity_shift
            if cfg.lower_level_optimality == "relaxed-kkt"
            else None
        ),
        "dual_tikhonov_gamma": (
            cfg.dual_tikhonov_gamma
            if cfg.lower_level_optimality
            in ("relaxed-kkt", "tikhonov-strong-duality")
            else None
        ),
        "proximal_penalty_eur_per_mw2_day": (
            cfg.strategic_proximal_penalty_eur_per_mw2_day
        ),
        "proximal_energy_scale_hours": cfg.strategic_proximal_energy_scale_hours,
        "proximal_penalty_step_eur_per_mw2_day": (
            cfg.strategic_proximal_penalty_step_eur_per_mw2_day
        ),
        "proximal_penalty_step_iterations": (
            cfg.strategic_proximal_penalty_step_iterations
        ),
        "proximal_penalty_initial_zero_iterations": (
            cfg.strategic_proximal_penalty_initial_zero_iterations
        ),
        "node_limit_mw": cfg.node_limit_mw,
        "node_total_power_mw": node_total_power,
        "node_excess_mw": node_excess,
        "max_node_overload_mw": max((max(0.0, value) for value in node_excess.values()), default=0.0),
        "x_power_mw": {
            f"{investor}|{node}": value
            for (investor, node), value in state.x_power.items()
        },
        "x_energy_mwh": {
            f"{investor}|{node}": value
            for (investor, node), value in state.x_energy.items()
        },
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


#### Final result export
# -----------------------------------------------------------------------------

def export_epec_results(
    output_dir: Path, data: MarketData, state, cfg, settlement: dict, data_path: Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = list(data.nodes)
    units = [inv.investor_id for inv in cfg.investors]
    reference_quad = getattr(settlement["reference_model"], "_quad_demand", None)

    run_config = {
        "model_variant": _model_variant(cfg),
        **_demand_metadata(cfg),
        "data_path": str(data_path),
        "update_rule": cfg.update_rule,
        "parallel_workers": cfg.strategic_parallel_workers,
        "investor_solve_order": [inv.investor_id for inv in cfg.investors],
        "initialization_method": state.initialization_method,
        "settlement_price_basis": "system" if cfg.system_price_settlement else "nodal",
        "dual_selection": (
            "iso_minimum_norm_lmp_secondary_qp"
            if cfg.lower_level_optimality == "iso-min-norm-dual"
            else (
                "exact_tikhonov_matched_primal_dual_strong_duality"
                if cfg.lower_level_optimality == "tikhonov-strong-duality"
                else (
                    "direct_tikhonov_regularized_dual_kkt"
                    if cfg.lower_level_optimality == "relaxed-kkt"
                    and cfg.dual_tikhonov_gamma > 0.0
                    else (
                        "leader_lambda_l2_penalty"
                        if cfg.lambda_l2_penalty_coefficient > 0.0
                        else "optimistic_mpec_no_price_penalty"
                    )
                )
            )
        ),
        "lambda_l2_penalty_coefficient": cfg.lambda_l2_penalty_coefficient,
        "lower_level_optimality": cfg.lower_level_optimality,
        "iso_min_norm_complementarity_epsilon": (
            cfg.iso_min_norm_complementarity_epsilon
            if cfg.lower_level_optimality == "iso-min-norm-dual"
            else None
        ),
        "complementarity_epsilon": (
            cfg.complementarity_epsilon
            if cfg.lower_level_optimality == "relaxed-kkt"
            else None
        ),
        "complementarity_formulation": (
            cfg.complementarity_formulation
            if cfg.lower_level_optimality == "relaxed-kkt"
            else None
        ),
        "complementarity_shift": (
            cfg.complementarity_shift
            if cfg.lower_level_optimality == "relaxed-kkt"
            else None
        ),
        "dual_tikhonov_gamma": (
            cfg.dual_tikhonov_gamma
            if cfg.lower_level_optimality
            in ("relaxed-kkt", "tikhonov-strong-duality")
            else None
        ),
        "proximal_penalty_eur_per_mw2_day": (
            cfg.strategic_proximal_penalty_eur_per_mw2_day
        ),
        "proximal_energy_scale_hours": cfg.strategic_proximal_energy_scale_hours,
        "proximal_penalty_step_eur_per_mw2_day": (
            cfg.strategic_proximal_penalty_step_eur_per_mw2_day
        ),
        "proximal_penalty_step_iterations": (
            cfg.strategic_proximal_penalty_step_iterations
        ),
        "proximal_penalty_initial_zero_iterations": (
            cfg.strategic_proximal_penalty_initial_zero_iterations
        ),
        "quadratic_demand_alpha_eur_per_mwh": reference_quad.alpha if reference_quad is not None else None,
        "quadratic_demand_beta_eur_per_mwh_per_share": reference_quad.beta if reference_quad is not None else None,
        "dispatch_regularization_eur_per_mw2h": cfg.dispatch_regularization_eur_per_mw2h,
        "solver_tol": cfg.solver_tol,
        "rival_representation": "separate_battery_per_investor_with_nodal_mw_mwh",
        "rival_sparsity_tol_mw": (
            None if hasattr(state, "offer_charge") else cfg.rival_sparsity_tol_mw
        ),
        "full_investor_model_exports_available": bool(state.final_models),
        "embedded_sparsity": (
            "active_investor_all_nodes; rivals_only_positive_mw_or_mwh; "
            "generators_only_positive_capacity_hours"
            if hasattr(state, "offer_charge")
            else "active_investor_all_nodes; rival_node_blocks_only_above_configured_power_threshold; "
            "generators_only_positive_capacity_hours"
        ),
        "fixed_demand_shedding_block_omitted": not cfg.use_demand_curve,
        "capacity_cleanup_tol_mw_mwh": cfg.capacity_cleanup_tol_mw_mwh,
        "automatic_jacobi_initializer": cfg.automatic_jacobi_initializer,
        "jacobi_initializer_snapshot_power_mw": (
            cfg.jacobi_initializer_snapshot_power_mw
        ),
        "jacobi_initializer_snapshot_ratio_hours": (
            cfg.jacobi_initializer_snapshot_ratio_hours
        ),
        "damping": cfg.damping,
        "max_iters": cfg.max_iters,
        "starting_iteration": cfg.starting_iteration,
        "resume_from": cfg.resume_from,
        "tol_rel": cfg.tol_rel,
        "floor_mw": cfg.floor_mw,
        "floor_mwh": cfg.floor_mwh,
        "seed_power_mw": cfg.seed_power_mw,
        "seed_ratio_hours": cfg.seed_ratio_hours,
        "node_limit_mw": cfg.node_limit_mw,
        "max_cpu_time": cfg.max_cpu_time,
        "price_bound_eur_per_mwh": cfg.price_bound_eur_per_mwh,
        "dual_bound_eur_per_mwh": cfg.dual_bound_eur_per_mwh,
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
                "owned_generation_shares": dict(inv.owned_generation_shares),
            }
            for inv in cfg.investors
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    export_epec_checkpoint(output_dir, state, cfg)
    _write_csv(
        output_dir / "projection_events.csv",
        ["iteration", "node", "total_before_mw", "scale"],
        state.projection_events,
    )
    _write_csv(
        output_dir / "final_capacities.csv",
        [
            "investor",
            "node",
            "x_power_mw",
            "x_energy_mwh",
            "ratio_hours",
            "share_of_node_limit",
            "last_best_response_access_shadow_price_eur_per_mw_day",
        ],
        [
            {
                "investor": i,
                "node": n,
                "x_power_mw": state.x_power[i, n],
                "x_energy_mwh": state.x_energy[i, n],
                "ratio_hours": state.x_energy[i, n] / state.x_power[i, n] if state.x_power[i, n] > 1e-9 else 0.0,
                "share_of_node_limit": state.x_power[i, n] / cfg.node_limit_mw,
                "last_best_response_access_shadow_price_eur_per_mw_day": (
                    investment_headroom_shadow_price(state.final_models[i], n)
                    if state.final_models.get(i) is not None
                    else getattr(state, "final_access_shadow_prices", {})
                    .get(i, {})
                    .get(n, float("nan"))
                ),
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
    selected_prices = getattr(state, "final_selected_prices", {})
    if selected_prices:
        _write_csv(
            output_dir / "last_best_response_prices.csv",
            [
                "investor",
                "hour",
                "node",
                "lambda_best_response_eur_per_mwh",
                "lambda_joint_eur_per_mwh",
                "difference_eur_per_mwh",
            ],
            [
                {
                    "investor": investor,
                    "hour": int(t),
                    "node": n,
                    "lambda_best_response_eur_per_mwh": prices[n, int(t)],
                    "lambda_joint_eur_per_mwh": lam[n, int(t)],
                    "difference_eur_per_mwh": prices[n, int(t)] - lam[n, int(t)],
                }
                for investor, prices in selected_prices.items()
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
        "model_variant": _model_variant(cfg),
        **_demand_metadata(cfg),
        "converged": state.converged,
        "stop_reason": state.stop_reason,
        "iterations": state.iteration,
        "starting_iteration": cfg.starting_iteration,
        "additional_max_iters": cfg.max_iters,
        "resume_from": cfg.resume_from,
        "update_rule": cfg.update_rule,
        "parallel_workers": cfg.strategic_parallel_workers,
        "investor_solve_order": [inv.investor_id for inv in cfg.investors],
        "initialization_method": state.initialization_method,
        "initializer_summary": state.initializer_summary,
        "settlement_price_basis": "system" if cfg.system_price_settlement else "nodal",
        "lower_level_optimality": cfg.lower_level_optimality,
        "iso_min_norm_complementarity_epsilon": (
            cfg.iso_min_norm_complementarity_epsilon
            if cfg.lower_level_optimality == "iso-min-norm-dual"
            else None
        ),
        "dual_tikhonov_gamma": (
            cfg.dual_tikhonov_gamma
            if cfg.lower_level_optimality
            in ("relaxed-kkt", "tikhonov-strong-duality")
            else None
        ),
        "proximal_penalty_eur_per_mw2_day": (
            cfg.strategic_proximal_penalty_eur_per_mw2_day
        ),
        "proximal_penalty_step_eur_per_mw2_day": (
            cfg.strategic_proximal_penalty_step_eur_per_mw2_day
        ),
        "proximal_penalty_step_iterations": (
            cfg.strategic_proximal_penalty_step_iterations
        ),
        "proximal_penalty_initial_zero_iterations": (
            cfg.strategic_proximal_penalty_initial_zero_iterations
        ),
        "complementarity_epsilon": (
            cfg.complementarity_epsilon
            if cfg.lower_level_optimality == "relaxed-kkt"
            else None
        ),
        "complementarity_formulation": (
            cfg.complementarity_formulation
            if cfg.lower_level_optimality == "relaxed-kkt"
            else None
        ),
        "last_sweep_max_original_nodal_balance_residual_mw": max(
            (
                row.get("max_original_nodal_balance_residual_mw", 0.0)
                for row in state.history
                if row.get("iteration") == state.iteration
            ),
            default=0.0,
        ),
        "last_sweep_max_hourly_system_balance_residual_mw": max(
            (
                row.get("max_hourly_system_balance_residual_mw", 0.0)
                for row in state.history
                if row.get("iteration") == state.iteration
            ),
            default=0.0,
        ),
        "dispatch_regularization_eur_per_mw2h": cfg.dispatch_regularization_eur_per_mw2h,
        "solver_tol": cfg.solver_tol,
        "rival_representation": "separate_battery_per_investor_with_nodal_mw_mwh",
        "rival_sparsity_tol_mw": (
            None if hasattr(state, "offer_charge") else cfg.rival_sparsity_tol_mw
        ),
        "full_investor_model_exports_available": bool(state.final_models),
        "damping": cfg.damping,
        "tol_rel": cfg.tol_rel,
        "projection_event_count": len(state.projection_events),
        "investors": settlement_json["investors"],
        "node_shares": settlement_json["node_shares"],
        "max_node_overload_mw": settlement["max_node_overload_mw"],
        "shared_limit_feasible": settlement["shared_limit_feasible"],
        "joint_lambda_min_eur_per_mwh": settlement["lambda_min_eur_per_mwh"],
        "joint_lambda_max_eur_per_mwh": settlement["lambda_max_eur_per_mwh"],
        "simultaneous_charge_discharge_mwh": settlement.get(
            "simultaneous_charge_discharge_mwh", 0.0
        ),
        "max_simultaneous_charge_discharge_mw": settlement.get(
            "max_simultaneous_charge_discharge_mw", 0.0
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    from single_investor_mpec_results import export_solution

    for i, model in state.final_models.items():
        if model is None:
            continue
        export_solution(model, output_dir / f"investor_{i}", "ok", "optimal", None)
