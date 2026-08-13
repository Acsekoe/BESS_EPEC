"""Capacity-only MPEC with Scholtes-relaxed lower-level KKT conditions.

The lower-level market is the same linear program used by the maintained
strong-duality formulation.  Strong duality is replaced by the individual
complementarity inequalities

    0 <= primal_or_slack * dual_or_reduced_cost <= epsilon.

For positive ``epsilon`` this is a smooth NLP that can be solved by Ipopt.  It
is an approximation of the exact KKT system, so solved models must be audited
through the exported product and primal-dual-gap diagnostics.
"""

from __future__ import annotations

import pyomo.environ as pyo

from mpec_strong_duality import build_model as build_strong_duality_model
from primal_market_clearing_model import MarketData


DEFAULT_COMPLEMENTARITY_EPSILON = 1.0e-3


def build_model(
    data: MarketData,
    *,
    complementarity_epsilon: float = DEFAULT_COMPLEMENTARITY_EPSILON,
    **kwargs: object,
) -> pyo.ConcreteModel:
    """Build one investor's capacity MPEC with relaxed KKT products."""

    epsilon = float(complementarity_epsilon)
    if epsilon < 0.0:
        raise ValueError("complementarity_epsilon must be non-negative.")

    m = build_strong_duality_model(data, **kwargs)
    m.name = f"Relaxed-KKT MPEC [{m._active_id}]"
    m.strong_duality.deactivate()

    active = m._active_id
    eta = data.eta
    last_t = max(data.times)

    def unit_power(model: pyo.ConcreteModel, unit: str, node: str):
        return (
            model.X_power[node]
            if unit == active
            else model._rival_power[unit][node]
        )

    def unit_energy(model: pyo.ConcreteModel, unit: str, node: str):
        return (
            model.X_energy[node]
            if unit == active
            else model._rival_energy[unit][node]
        )

    def flow(model: pyo.ConcreteModel, line: str, time: int):
        return sum(
            data.ptdf[line, node] * model.NetInjection[node, time]
            for node in model.N
        )

    def gen_reduced_cost(model: pyo.ConcreteModel, generator: str, time: int):
        return (
            data.generation_cost[generator]
            - sum(model.lam[node, time] for node in model._gen_nodes[generator])
            - model.nu_gen[generator, time]
        )

    def charge_reduced_cost(
        model: pyo.ConcreteModel, unit: str, node: str, time: int
    ):
        return (
            0.5 * model._unit_degradation[unit]
            + model.lam[node, time]
            - model.rho_ch[unit, node, time]
            + eta * model.gam[unit, node, time]
        )

    def discharge_reduced_cost(
        model: pyo.ConcreteModel, unit: str, node: str, time: int
    ):
        return (
            0.5 * model._unit_degradation[unit]
            - model.lam[node, time]
            - model.sig_dis[unit, node, time]
            - model.gam[unit, node, time] / eta
        )

    def soc_reduced_cost(
        model: pyo.ConcreteModel, unit: str, node: str, soc_time: int
    ):
        stationarity_lhs = model.del_soc[unit, node, soc_time]
        if soc_time in model.T:
            stationarity_lhs += model.gam[unit, node, soc_time]
        if soc_time + 1 in model.T:
            stationarity_lhs -= model.gam[unit, node, soc_time + 1]
        if soc_time == 0:
            stationarity_lhs += model.rho_per[unit, node]
        if soc_time == last_t:
            stationarity_lhs -= model.rho_per[unit, node]
        return -stationarity_lhs

    def add_relaxed_product(
        name: str,
        index_sets: tuple[pyo.Set, ...],
        product_rule,
    ) -> None:
        product = pyo.Expression(*index_sets, rule=product_rule)
        m.add_component(f"{name}_product", product)
        m.add_component(
            name,
            pyo.Constraint(
                *index_sets,
                rule=lambda model, *key: pyo.inequality(
                    0.0, product[key], epsilon
                ),
            ),
        )

    add_relaxed_product(
        "relaxed_comp_gen_lower",
        (m.GT,),
        lambda model, g, t: model.P_gen[g, t]
        * gen_reduced_cost(model, g, t),
    )
    add_relaxed_product(
        "relaxed_comp_charge_lower",
        (m.IN, m.T),
        lambda model, i, n, t: model.P_charge[i, n, t]
        * charge_reduced_cost(model, i, n, t),
    )
    add_relaxed_product(
        "relaxed_comp_discharge_lower",
        (m.IN, m.T),
        lambda model, i, n, t: model.P_discharge[i, n, t]
        * discharge_reduced_cost(model, i, n, t),
    )
    add_relaxed_product(
        "relaxed_comp_soc_lower",
        (m.IN, m.T_SOC),
        lambda model, i, n, tau: model.SOC[i, n, tau]
        * soc_reduced_cost(model, i, n, tau),
    )
    add_relaxed_product(
        "relaxed_comp_gen_upper",
        (m.GT,),
        lambda model, g, t: (
            data.generation_capacity[g, t] - model.P_gen[g, t]
        )
        * (-model.nu_gen[g, t]),
    )
    add_relaxed_product(
        "relaxed_comp_line_upper",
        (m.L, m.T),
        lambda model, l, t: (data.line_limit[l] - flow(model, l, t))
        * (-model.mu_up[l, t]),
    )
    add_relaxed_product(
        "relaxed_comp_line_lower",
        (m.L, m.T),
        lambda model, l, t: (flow(model, l, t) + data.line_limit[l])
        * model.mu_dn[l, t],
    )
    add_relaxed_product(
        "relaxed_comp_charge_upper",
        (m.IN, m.T),
        lambda model, i, n, t: (
            unit_power(model, i, n) - model.P_charge[i, n, t]
        )
        * (-model.rho_ch[i, n, t]),
    )
    add_relaxed_product(
        "relaxed_comp_discharge_upper",
        (m.IN, m.T),
        lambda model, i, n, t: (
            unit_power(model, i, n) - model.P_discharge[i, n, t]
        )
        * (-model.sig_dis[i, n, t]),
    )
    add_relaxed_product(
        "relaxed_comp_soc_upper",
        (m.IN, m.T_SOC),
        lambda model, i, n, tau: (
            unit_energy(model, i, n) - model.SOC[i, n, tau]
        )
        * (-model.del_soc[i, n, tau]),
    )

    m._lower_level_optimality = "relaxed-kkt"
    m._complementarity_epsilon = epsilon
    m._relaxed_kkt_product_components = tuple(
        f"{name}_product"
        for name in (
            "relaxed_comp_gen_lower",
            "relaxed_comp_charge_lower",
            "relaxed_comp_discharge_lower",
            "relaxed_comp_soc_lower",
            "relaxed_comp_gen_upper",
            "relaxed_comp_line_upper",
            "relaxed_comp_line_lower",
            "relaxed_comp_charge_upper",
            "relaxed_comp_discharge_upper",
            "relaxed_comp_soc_upper",
        )
    )
    return m


def diagnostics(model: pyo.ConcreteModel) -> dict[str, float | int]:
    """Evaluate all complementarity products in a solved relaxed-KKT MPEC."""

    products = [
        float(pyo.value(component[index]))
        for name in model._relaxed_kkt_product_components
        for component in (getattr(model, name),)
        for index in component
    ]
    epsilon = float(model._complementarity_epsilon)
    minimum = min(products, default=0.0)
    maximum = max(products, default=0.0)
    gap = float(pyo.value(model.primal_objective - model.dual_objective))
    return {
        "count": len(products),
        "epsilon": epsilon,
        "minimum_product": minimum,
        "maximum_product": maximum,
        "maximum_upper_bound_violation": max(0.0, maximum - epsilon),
        "maximum_nonnegativity_violation": max(0.0, -minimum),
        "primal_dual_gap_eur_per_day": gap,
        "absolute_primal_dual_gap_eur_per_day": abs(gap),
    }
