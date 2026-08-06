"""Single-investor MPEC with directly regularized lower-level KKT conditions.

This module reuses the maintained primal, dual, investment, and upper-level
objective blocks from :mod:`single_investor_mpec`. It replaces the lower-level
strong-duality equality with individual smooth complementarity relaxations

    primal_or_dual_slack * complementary_quantity <= epsilon_comp.

No binary variables or Big-M disjunctions are introduced. The formulation
supports the separate fixed rival batteries used by capacity diagonalization.
At epsilon_comp=0 the default Scholtes form is the exact,
constraint-degenerate MPCC; positive values are intended for Ipopt experiments
and should be audited through the exported complementarity and primal-dual-gap
diagnostics.

Two explicitly experimental regularizations are optional:

* ``dual_tikhonov_gamma > 0`` replaces exact nodal balance ``h = 0`` by
  ``h + gamma * lambda = 0``. With this model's sign convention, that is the
  first-order condition of a dual objective regularized by
  ``-(gamma / 2) * ||lambda||^2``. It permits physical imbalance at finite
  gamma and is therefore a homotopy approximation, not the original ISO
  clearing problem.
* ``complementarity_formulation='central-path'`` replaces each Scholtes upper
  bound by ``(slack + shift) * dual == epsilon``. This is a perturbed KKT
  diagnostic; with positive shift it is not the exact logarithmic central path.

Both options default off so archived relaxed-KKT runs remain reproducible.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

try:
    from .. import common as _common  # Sets the maintained model import paths.
except ImportError:
    from tikhonov_kkt import common as _common

from primal_market_clearing_model import MarketData, value
from single_investor_mpec import _nodes_of_generator, build_single_investor_mpec


MODEL_NAME = "Single Investor Directly Regularized KKT MPEC"
DEFAULT_COMPLEMENTARITY_EPSILON = 1.0e-3
DEFAULT_COMPLEMENTARITY_SHIFT = 1.0e-8
COMPLEMENTARITY_FORMULATIONS = ("scholtes", "central-path")


def build_single_investor_relaxed_kkt_mpec(
    data: MarketData,
    *,
    complementarity_epsilon: float = DEFAULT_COMPLEMENTARITY_EPSILON,
    complementarity_formulation: str = "scholtes",
    complementarity_shift: float = DEFAULT_COMPLEMENTARITY_SHIFT,
    dual_tikhonov_gamma: float = 0.0,
    **mpec_kwargs: Any,
) -> pyo.ConcreteModel:
    """Build the MPEC with optional direct lower-level KKT regularization."""

    epsilon = float(complementarity_epsilon)
    if epsilon < 0.0:
        raise ValueError("complementarity_epsilon must be non-negative.")
    formulation = str(complementarity_formulation)
    if formulation not in COMPLEMENTARITY_FORMULATIONS:
        raise ValueError(
            "complementarity_formulation must be one of "
            f"{COMPLEMENTARITY_FORMULATIONS}, received {formulation!r}."
        )
    shift = float(complementarity_shift)
    if shift < 0.0:
        raise ValueError("complementarity_shift must be non-negative.")
    if formulation == "central-path" and epsilon <= 0.0:
        raise ValueError("central-path complementarity requires a positive epsilon.")
    gamma = float(dual_tikhonov_gamma)
    if gamma < 0.0:
        raise ValueError("dual_tikhonov_gamma must be non-negative.")

    m = build_single_investor_mpec(data, **mpec_kwargs)
    m.strong_duality.deactivate()
    m.name = MODEL_NAME

    inv = m._investor_config
    inv_id = m._investor_id
    eta = data.eta
    dispatch_reg = m._dispatch_regularization_eur_per_mw2h
    storage_degradation = m._storage_degradation_eur_per_mwh
    gen_nodes = _nodes_of_generator(data)
    last_t = max(data.times)
    use_central_path = formulation == "central-path"

    storage_units_at_node = {
        node: [unit for unit, candidate_node in m.IN if candidate_node == node]
        for node in m.N
    }
    generators_at_node_time = {
        (node, time): [
            generator
            for generator in data.generators_at_node.get(node, [])
            if (generator, time) in m.GT
        ]
        for node in m.N
        for time in m.T
    }

    def original_nodal_balance_residual(
        model: pyo.ConcreteModel, node: str, time: int
    ) -> pyo.Expression:
        storage_net = sum(
            model.P_discharge[unit, node, time] - model.P_charge[unit, node, time]
            for unit in storage_units_at_node[node]
        )
        return (
            sum(
                model.P_gen[generator, time]
                for generator in generators_at_node_time[node, time]
            )
            + storage_net
            + (model.P_shed[node, time] if model._use_demand_curve else 0.0)
            - data.demand_el[node, time]
            - model.NetInjection[node, time]
        )

    m.original_nodal_balance_residual = pyo.Expression(
        m.N, m.T, rule=original_nodal_balance_residual
    )
    if gamma > 0.0:
        # The economic lambda used by this model has the opposite sign of the
        # raw equality multiplier in L = C - lambda*h. Maximizing the dual
        # objective minus (gamma/2)||lambda||^2 therefore gives h+gamma*lambda=0.
        m.nodal_balance.deactivate()
        m.tikhonov_regularized_nodal_balance = pyo.Constraint(
            m.N,
            m.T,
            rule=lambda model, n, t: model.original_nodal_balance_residual[n, t]
            + gamma * model.lam[n, t]
            == 0.0,
        )

    def regularized_product(slack, dual_magnitude):
        return (slack + shift) * dual_magnitude if use_central_path else slack * dual_magnitude

    def complementarity_relation(product):
        return product == epsilon if use_central_path else product <= epsilon

    def unit_power_limit(model: pyo.ConcreteModel, unit: str, node: str):
        if unit == inv_id:
            return model.X_power[node]
        return model._rival_power_mw_by_unit[unit][node]

    def unit_energy_limit(model: pyo.ConcreteModel, unit: str, node: str):
        if unit == inv_id:
            return model.X_energy[node]
        return model._rival_energy_mwh_by_unit[unit][node]

    def flow(model: pyo.ConcreteModel, line: str, time: int):
        return sum(data.ptdf[line, node] * model.NetInjection[node, time] for node in model.N)

    # Non-negative primal variable paired with its non-negative reduced cost.
    m.relaxed_comp_gen_lower_product = pyo.Expression(
        m.GT,
        rule=lambda model, g, t: regularized_product(
            model.P_gen[g, t],
            data.generation_cost[g]
            + dispatch_reg * model.P_gen[g, t]
            - sum(model.lam[n, t] for n in gen_nodes.get(g, []))
            - model.nu_gen[g, t],
        ),
    )
    m.relaxed_comp_gen_lower = pyo.Constraint(
        m.GT,
        rule=lambda model, g, t: complementarity_relation(
            model.relaxed_comp_gen_lower_product[g, t]
        ),
    )
    m.relaxed_comp_charge_lower_product = pyo.Expression(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: regularized_product(
            model.P_charge[i, n, t],
            0.5 * storage_degradation[i]
            + dispatch_reg * model.P_charge[i, n, t]
            + model.lam[n, t]
            - model.rho_ch[i, n, t]
            + eta * model.gam[i, n, t],
        ),
    )
    m.relaxed_comp_charge_lower = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: complementarity_relation(
            model.relaxed_comp_charge_lower_product[i, n, t]
        ),
    )
    m.relaxed_comp_discharge_lower_product = pyo.Expression(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: regularized_product(
            model.P_discharge[i, n, t],
            0.5 * storage_degradation[i]
            + dispatch_reg * model.P_discharge[i, n, t]
            - model.lam[n, t]
            - model.sig_dis[i, n, t]
            - model.gam[i, n, t] / eta,
        ),
    )
    m.relaxed_comp_discharge_lower = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: complementarity_relation(
            model.relaxed_comp_discharge_lower_product[i, n, t]
        ),
    )

    def soc_reduced_cost(model: pyo.ConcreteModel, i: str, n: str, tau: int):
        stationarity_lhs = model.del_soc[i, n, tau]
        if tau in model.T:
            stationarity_lhs += model.gam[i, n, tau]
        if (tau + 1) in model.T:
            stationarity_lhs -= model.gam[i, n, tau + 1]
        if tau == 0:
            stationarity_lhs += model.rho_per[i, n]
        if tau == last_t:
            stationarity_lhs -= model.rho_per[i, n]
        return dispatch_reg * model.SOC[i, n, tau] - stationarity_lhs

    m.relaxed_comp_soc_lower_product = pyo.Expression(
        m.IN,
        m.T_SOC,
        rule=lambda model, i, n, tau: regularized_product(
            model.SOC[i, n, tau], soc_reduced_cost(model, i, n, tau)
        ),
    )
    m.relaxed_comp_soc_lower = pyo.Constraint(
        m.IN,
        m.T_SOC,
        rule=lambda model, i, n, tau: complementarity_relation(
            model.relaxed_comp_soc_lower_product[i, n, tau]
        ),
    )

    # Primal upper-bound slack paired with the sign-oriented dual magnitude.
    m.relaxed_comp_gen_upper_product = pyo.Expression(
        m.GT,
        rule=lambda model, g, t: regularized_product(
            data.generation_capacity[g, t] - model.P_gen[g, t],
            -model.nu_gen[g, t],
        ),
    )
    m.relaxed_comp_gen_upper = pyo.Constraint(
        m.GT,
        rule=lambda model, g, t: complementarity_relation(
            model.relaxed_comp_gen_upper_product[g, t]
        ),
    )
    m.relaxed_comp_line_upper_product = pyo.Expression(
        m.L,
        m.T,
        rule=lambda model, l, t: regularized_product(
            data.line_limit[l] - flow(model, l, t), -model.mu_up[l, t]
        ),
    )
    m.relaxed_comp_line_upper = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, l, t: complementarity_relation(
            model.relaxed_comp_line_upper_product[l, t]
        ),
    )
    m.relaxed_comp_line_lower_product = pyo.Expression(
        m.L,
        m.T,
        rule=lambda model, l, t: regularized_product(
            flow(model, l, t) + data.line_limit[l], model.mu_dn[l, t]
        ),
    )
    m.relaxed_comp_line_lower = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, l, t: complementarity_relation(
            model.relaxed_comp_line_lower_product[l, t]
        ),
    )
    m.relaxed_comp_charge_upper_product = pyo.Expression(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: regularized_product(
            unit_power_limit(model, i, n) - model.P_charge[i, n, t],
            -model.rho_ch[i, n, t],
        ),
    )
    m.relaxed_comp_charge_upper = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: complementarity_relation(
            model.relaxed_comp_charge_upper_product[i, n, t]
        ),
    )
    m.relaxed_comp_discharge_upper_product = pyo.Expression(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: regularized_product(
            unit_power_limit(model, i, n) - model.P_discharge[i, n, t],
            -model.sig_dis[i, n, t],
        ),
    )
    m.relaxed_comp_discharge_upper = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: complementarity_relation(
            model.relaxed_comp_discharge_upper_product[i, n, t]
        ),
    )
    m.relaxed_comp_soc_upper_product = pyo.Expression(
        m.IN,
        m.T_SOC,
        rule=lambda model, i, n, tau: regularized_product(
            unit_energy_limit(model, i, n) - model.SOC[i, n, tau],
            -model.del_soc[i, n, tau],
        ),
    )
    m.relaxed_comp_soc_upper = pyo.Constraint(
        m.IN,
        m.T_SOC,
        rule=lambda model, i, n, tau: complementarity_relation(
            model.relaxed_comp_soc_upper_product[i, n, tau]
        ),
    )

    product_components = [
        "relaxed_comp_gen_lower_product",
        "relaxed_comp_gen_upper_product",
        "relaxed_comp_line_upper_product",
        "relaxed_comp_line_lower_product",
        "relaxed_comp_charge_lower_product",
        "relaxed_comp_charge_upper_product",
        "relaxed_comp_discharge_lower_product",
        "relaxed_comp_discharge_upper_product",
        "relaxed_comp_soc_lower_product",
        "relaxed_comp_soc_upper_product",
    ]

    if m._use_demand_curve:
        quad = m._quad_demand
        m.relaxed_comp_shed_lower_product = pyo.Expression(
            m.N,
            m.T,
            rule=lambda model, n, t: regularized_product(
                model.P_shed[n, t],
                quad.alpha
                + (quad.quad_coefficient(data.demand_el[n, t]) + dispatch_reg)
                * model.P_shed[n, t]
                - model.lam[n, t]
                - model.xi_shed[n, t],
            ),
        )
        m.relaxed_comp_shed_lower = pyo.Constraint(
            m.N,
            m.T,
            rule=lambda model, n, t: complementarity_relation(
                model.relaxed_comp_shed_lower_product[n, t]
            ),
        )
        m.relaxed_comp_shed_upper_product = pyo.Expression(
            m.N,
            m.T,
            rule=lambda model, n, t: regularized_product(
                data.demand_el[n, t] - model.P_shed[n, t],
                -model.xi_shed[n, t],
            ),
        )
        m.relaxed_comp_shed_upper = pyo.Constraint(
            m.N,
            m.T,
            rule=lambda model, n, t: complementarity_relation(
                model.relaxed_comp_shed_upper_product[n, t]
            ),
        )
        product_components.extend(
            ["relaxed_comp_shed_lower_product", "relaxed_comp_shed_upper_product"]
        )

    m.relaxed_kkt_total_complementarity_expr = pyo.Expression(
        expr=sum(
            product[index]
            for name in product_components
            for product in (getattr(m, name),)
            for index in product
        )
    )
    m._lower_level_optimality = "relaxed-kkt"
    m._complementarity_epsilon = epsilon
    m._complementarity_formulation = formulation
    m._complementarity_shift = shift if use_central_path else 0.0
    m._dual_tikhonov_gamma = gamma
    m._relaxed_kkt_product_components = tuple(product_components)
    return m


def relaxed_kkt_diagnostics(model: pyo.ConcreteModel) -> dict[str, float | int | str]:
    """Evaluate every complementarity product in a solved relaxed-KKT model."""

    epsilon = float(model._complementarity_epsilon)
    products = [
        value(component[index])
        for name in model._relaxed_kkt_product_components
        for component in (getattr(model, name),)
        for index in component
    ]
    formulation = str(getattr(model, "_complementarity_formulation", "scholtes"))
    residuals = [abs(product - epsilon) for product in products]
    nodal_residuals = [
        value(model.original_nodal_balance_residual[n, t])
        for n in model.N
        for t in model.T
    ]
    hourly_system_residuals = [
        sum(value(model.original_nodal_balance_residual[n, t]) for n in model.N)
        for t in model.T
    ]
    return {
        "count": len(products),
        "epsilon": epsilon,
        "formulation": formulation,
        "complementarity_shift": float(getattr(model, "_complementarity_shift", 0.0)),
        "dual_tikhonov_gamma": float(getattr(model, "_dual_tikhonov_gamma", 0.0)),
        "minimum_product": min(products, default=0.0),
        "maximum_product": max(products, default=0.0),
        "sum_products": sum(products),
        "maximum_equality_residual": (
            max(residuals, default=0.0) if formulation == "central-path" else 0.0
        ),
        "maximum_upper_bound_violation": max(
            (max(0.0, product - epsilon) for product in products),
            default=0.0,
        ),
        "maximum_absolute_original_nodal_balance_residual_mw": max(
            (abs(residual) for residual in nodal_residuals), default=0.0
        ),
        "total_absolute_original_nodal_balance_residual_mwh": sum(
            abs(residual) for residual in nodal_residuals
        ),
        "maximum_absolute_hourly_system_balance_residual_mw": max(
            (abs(residual) for residual in hourly_system_residuals), default=0.0
        ),
        "primal_dual_objective_gap_eur_per_day": value(model.primal_objective_expr)
        - value(model.dual_objective_expr),
    }
