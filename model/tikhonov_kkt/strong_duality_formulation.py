"""Exact strong-duality MPEC for the finite-gamma soft-balance market.

The lower-level primal and dual are the matched pair

    min C(x) + ||h||^2/(2*gamma)
    max D(y) - (gamma/2)||lambda||^2.

Primal feasibility, dual feasibility, and their objective equality replace
all explicit complementarity relations.  There is no Scholtes epsilon.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

try:
    from . import common as _common  # Sets maintained-model import paths.
    from .soft_balance_llp import solve_matched_soft_market, soft_balance_prices
except ImportError:
    import common as _common
    from soft_balance_llp import solve_matched_soft_market, soft_balance_prices

from primal_market_clearing_model import MarketData, value
from single_investor_mpec import (
    build_single_investor_mpec,
    fixed_storage_data_from_solution,
)


MODEL_NAME = "Single Investor Exact Tikhonov Strong-Duality MPEC"


def build_single_investor_tikhonov_strong_duality_mpec(
    data: MarketData,
    *,
    dual_tikhonov_gamma: float,
    **mpec_kwargs: Any,
) -> pyo.ConcreteModel:
    """Build the exact finite-gamma MPEC without complementarity products."""

    gamma = float(dual_tikhonov_gamma)
    if gamma <= 0.0:
        raise ValueError("dual_tikhonov_gamma must be positive.")
    if bool(mpec_kwargs.get("use_demand_curve", False)):
        raise ValueError(
            "The quadratic load-curtailment block is not supported here. The "
            "price-responsive demand_expansion block is."
        )
    if float(mpec_kwargs.get("lambda_l2_penalty_coefficient", 0.0)) != 0.0:
        raise ValueError(
            "Leader-side lambda penalties must be zero in the Tikhonov strong-duality experiment."
        )

    m = build_single_investor_mpec(data, **mpec_kwargs)
    return apply_tikhonov_strong_duality(m, data, gamma=gamma)


def apply_tikhonov_strong_duality(
    m: pyo.ConcreteModel,
    data: MarketData,
    *,
    gamma: float,
) -> pyo.ConcreteModel:
    """Apply the matched finite-gamma formulation to an assembled MPEC.

    The caller may modify the lower-level primal and dual first (for example,
    to install strategic storage quantity and price bids).  Freezing the
    unregularized objective pair here ensures the soft primal, regularized
    dual, and diagnostics all describe that modified market.
    """

    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    if bool(getattr(m, "_use_demand_curve", False)):
        raise ValueError(
            "The quadratic load-curtailment block is not supported here. The "
            "price-responsive demand_expansion block is."
        )
    if float(getattr(m, "_lambda_l2_penalty_coefficient", 0.0)) != 0.0:
        raise ValueError(
            "Leader-side lambda penalties must be zero in the Tikhonov strong-duality experiment."
        )
    if hasattr(m, "balance_residual"):
        raise ValueError("The MPEC already has a soft nodal-balance formulation.")

    m.name = MODEL_NAME
    m.nodal_balance.deactivate()

    storage_units_at_node = {
        node: [unit for unit, candidate_node in m.IN if candidate_node == node]
        for node in m.N
    }
    generators_at_node_time = {
        (node, int(time)): [
            generator
            for generator in data.generators_at_node.get(node, [])
            if (generator, int(time)) in m.GT
        ]
        for node in m.N
        for time in m.T
    }

    def original_balance_residual(model, node, time):
        return (
            sum(
                model.P_gen[generator, time]
                for generator in generators_at_node_time[node, int(time)]
            )
            + sum(
                model.P_discharge[unit, node, time]
                - model.P_charge[unit, node, time]
                for unit in storage_units_at_node[node]
            )
            - data.demand_el[node, time]
            - (
                model.E_extra[node, time]
                if hasattr(model, "EB") and (node, int(time)) in model.EB
                else 0.0
            )
            - model.NetInjection[node, time]
        )

    m.original_nodal_balance_residual = pyo.Expression(
        m.N, m.T, rule=original_balance_residual
    )
    m.balance_residual = pyo.Var(
        m.N, m.T, domain=pyo.Reals, initialize=-gamma * 50.0
    )
    m.soft_nodal_balance = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, n, t: model.original_nodal_balance_residual[n, t]
        - model.balance_residual[n, t]
        == 0.0,
    )
    # This stationarity equation is implied by exact primal/dual feasibility
    # and strong duality.  Keeping it explicitly prevents small bound
    # violations across thousands of nonnegative gap terms from cancelling in
    # the one global objective-equality constraint.
    m.tikhonov_price_relation = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, n, t: model.balance_residual[n, t]
        + gamma * model.lam[n, t]
        == 0.0,
    )

    base_primal_expr = m.primal_objective_expr.expr
    base_dual_expr = m.dual_objective_expr.expr
    m.unpenalized_primal_objective_expr = pyo.Expression(
        expr=base_primal_expr
    )
    m.unregularized_dual_objective_expr = pyo.Expression(
        expr=base_dual_expr
    )
    m.soft_balance_penalty_expr = pyo.Expression(
        expr=sum(
            m.balance_residual[n, t] ** 2
            for n in m.N
            for t in m.T
        )
        / (2.0 * gamma)
    )
    m.tikhonov_dual_penalty_expr = pyo.Expression(
        expr=0.5 * gamma * m.lambda_l2_norm_expr
    )
    m.primal_objective_expr.set_value(
        m.unpenalized_primal_objective_expr + m.soft_balance_penalty_expr
    )
    m.dual_objective_expr.set_value(
        m.unregularized_dual_objective_expr
        - m.tikhonov_dual_penalty_expr
    )
    m.strong_duality.set_value(
        m.primal_objective_expr == m.dual_objective_expr
    )

    m._lower_level_optimality = "tikhonov-strong-duality"
    m._dual_tikhonov_gamma = gamma
    return m


def strong_duality_diagnostics(
    model: pyo.ConcreteModel,
) -> dict[str, float | str]:
    """Evaluate finite-gamma feasibility, duality, and price diagnostics."""

    if not hasattr(model, "balance_residual"):
        # Exact market clearing: the nodal balance holds as an equality, so
        # every soft-market residual is structurally zero.
        prices = [value(model.lam[n, t]) for n in model.N for t in model.T]
        gap = value(model.primal_objective_expr) - value(model.dual_objective_expr)
        return {
            "lower_level_optimality": "exact-strong-duality",
            "dual_tikhonov_gamma": 0.0,
            "matched_strong_duality_gap_eur_per_day": gap,
            "unregularized_primal_dual_gap_eur_per_day": gap,
            "soft_balance_penalty_eur_per_day": 0.0,
            "tikhonov_dual_penalty_eur_per_day": 0.0,
            "max_abs_soft_nodal_balance_constraint_residual_mw": 0.0,
            "max_abs_h_plus_gamma_lambda_mw": 0.0,
            "max_abs_original_nodal_balance_residual_mw": 0.0,
            "total_abs_original_nodal_balance_residual_mwh": 0.0,
            "max_abs_hourly_system_balance_residual_mw": 0.0,
            "lambda_min_eur_per_mwh": min(prices, default=0.0),
            "lambda_max_eur_per_mwh": max(prices, default=0.0),
        }
    gamma = float(model._dual_tikhonov_gamma)
    original_residuals = [
        value(model.original_nodal_balance_residual[n, t])
        for n in model.N
        for t in model.T
    ]
    soft_constraint_residuals = [
        value(model.original_nodal_balance_residual[n, t])
        - value(model.balance_residual[n, t])
        for n in model.N
        for t in model.T
    ]
    price_relation_residuals = [
        value(model.balance_residual[n, t])
        + gamma * value(model.lam[n, t])
        for n in model.N
        for t in model.T
    ]
    hourly_system_residuals = [
        sum(value(model.original_nodal_balance_residual[n, t]) for n in model.N)
        for t in model.T
    ]
    prices = [value(model.lam[n, t]) for n in model.N for t in model.T]
    return {
        "lower_level_optimality": str(model._lower_level_optimality),
        "dual_tikhonov_gamma": gamma,
        "matched_strong_duality_gap_eur_per_day": value(
            model.primal_objective_expr
        )
        - value(model.dual_objective_expr),
        "unregularized_primal_dual_gap_eur_per_day": value(
            model.unpenalized_primal_objective_expr
        )
        - value(model.unregularized_dual_objective_expr),
        "soft_balance_penalty_eur_per_day": value(
            model.soft_balance_penalty_expr
        ),
        "tikhonov_dual_penalty_eur_per_day": value(
            model.tikhonov_dual_penalty_expr
        ),
        "max_abs_soft_nodal_balance_constraint_residual_mw": max(
            (abs(item) for item in soft_constraint_residuals), default=0.0
        ),
        "max_abs_h_plus_gamma_lambda_mw": max(
            (abs(item) for item in price_relation_residuals), default=0.0
        ),
        "max_abs_original_nodal_balance_residual_mw": max(
            (abs(item) for item in original_residuals), default=0.0
        ),
        "total_abs_original_nodal_balance_residual_mwh": sum(
            abs(item) for item in original_residuals
        ),
        "max_abs_hourly_system_balance_residual_mw": max(
            (abs(item) for item in hourly_system_residuals), default=0.0
        ),
        "lambda_min_eur_per_mwh": min(prices, default=0.0),
        "lambda_max_eur_per_mwh": max(prices, default=0.0),
    }


def _copy_component_values(target, source) -> None:
    for index in target:
        item = target[index]
        raw = value(source[index]) if index in source else 0.0
        if item.lb is not None:
            raw = max(value(item.lb), raw)
        if item.ub is not None:
            raw = min(value(item.ub), raw)
        item.set_value(raw)


def initialize_mpec_from_soft_market(
    model: pyo.ConcreteModel,
    *,
    solver_tol: float,
    max_cpu_time: float,
    gamma: float | None = None,
    tee: bool = False,
) -> dict[str, float | str]:
    """Warm-start an MPEC from independently solved matched soft LLPs.

    ``gamma`` defaults to the model's own regularization. An exact MPEC has
    none, so callers must supply a small warm-start value there.
    """

    warm_gamma = float(
        gamma if gamma is not None else model._dual_tikhonov_gamma
    )
    fixed_data = fixed_storage_data_from_solution(model)
    primal, dual, diagnostics = solve_matched_soft_market(
        fixed_data,
        degradation_eur_per_mwh_by_unit=(
            model._storage_degradation_eur_per_mwh
        ),
        gamma=warm_gamma,
        solver_tol=solver_tol,
        max_cpu_time=max_cpu_time,
        price_bound_eur_per_mwh=model._price_bound_eur_per_mwh,
        other_dual_bound=model._dual_bound_eur_per_mwh,
        demand_expansion=getattr(model, "_demand_expansion", None),
        tee=tee,
    )
    if diagnostics.get("primal_termination") != "optimal" or diagnostics.get(
        "dual_termination"
    ) != "optimal":
        return diagnostics

    _copy_component_values(model.P_gen, primal.P_gen)
    _copy_component_values(model.P_charge, primal.P_charge)
    _copy_component_values(model.P_discharge, primal.P_discharge)
    _copy_component_values(model.SOC, primal.SOC)
    _copy_component_values(model.NetInjection, primal.NetInjection)
    if hasattr(model, "balance_residual"):
        _copy_component_values(model.balance_residual, primal.balance_residual)
    if hasattr(model, "E_extra"):
        _copy_component_values(model.E_extra, primal.E_extra)
        _copy_component_values(model.xi_extra, dual.xi_extra)

    for component_name in (
        "lam",
        "lam_sys",
        "nu_gen",
        "mu_up",
        "mu_dn",
        "rho_ch",
        "sig_dis",
        "gam",
        "del_soc",
        "rho_per",
    ):
        _copy_component_values(
            getattr(model, component_name), getattr(dual, component_name)
        )
    return diagnostics


def audit_mpec_prices_against_soft_market(
    model: pyo.ConcreteModel,
    *,
    gamma: float,
    solver_tol: float,
    max_cpu_time: float,
    tee: bool = False,
) -> dict[str, float | str]:
    """Re-solve the unique soft market at the MPEC's proposed capacity fleet."""

    fixed_data = fixed_storage_data_from_solution(model)
    primal, dual, diagnostics = solve_matched_soft_market(
        fixed_data,
        degradation_eur_per_mwh_by_unit=(
            model._storage_degradation_eur_per_mwh
        ),
        gamma=gamma,
        solver_tol=solver_tol,
        max_cpu_time=max_cpu_time,
        price_bound_eur_per_mwh=model._price_bound_eur_per_mwh,
        other_dual_bound=model._dual_bound_eur_per_mwh,
        demand_expansion=getattr(model, "_demand_expansion", None),
        tee=tee,
    )
    audit = dict(diagnostics)
    audit["mpec_total_power_mw"] = sum(value(model.X_power[n]) for n in model.N)
    audit["mpec_total_energy_mwh"] = sum(value(model.X_energy[n]) for n in model.N)
    if diagnostics.get("primal_termination") != "optimal" or diagnostics.get(
        "dual_termination"
    ) != "optimal":
        return audit

    primal_prices = soft_balance_prices(primal)
    audit.update(
        {
            "max_abs_mpec_vs_soft_primal_lambda_eur_per_mwh": max(
                abs(value(model.lam[n, t]) - primal_prices[str(n), int(t)])
                for n in model.N
                for t in model.T
            ),
            "max_abs_mpec_vs_soft_dual_lambda_eur_per_mwh": max(
                abs(value(model.lam[n, t]) - value(dual.lam[n, t]))
                for n in model.N
                for t in model.T
            ),
        }
    )
    return audit
