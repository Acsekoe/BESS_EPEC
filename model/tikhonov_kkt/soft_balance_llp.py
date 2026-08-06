"""Matched soft-balance primal and Tikhonov-regularized dual LLP.

For positive ``gamma`` the primal penalizes the original nodal-balance
residual ``h`` by ``||h||^2 / (2*gamma)``.  Its exact dual is the ordinary LLP
dual with objective ``D - (gamma/2)||lambda||^2``.  This module keeps both
models together so finite-gamma primal/dual comparisons use a matched pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pyomo.environ as pyo

try:
    from .common import DEFAULT_DATA_PATH, load_fixed_storage_case, solve_ipopt
    from .dual_tikhonov_llp import build_tikhonov_dual_llp
    from .primal_llp import build_primal_llp
except ImportError:
    from common import DEFAULT_DATA_PATH, load_fixed_storage_case, solve_ipopt
    from dual_tikhonov_llp import build_tikhonov_dual_llp
    from primal_llp import build_primal_llp


# USER CONTROLS ---------------------------------------------------------------
DATA_PATH = DEFAULT_DATA_PATH
STORAGE_ID = "BESS_FIXED"
STORAGE_POWER_MW_BY_NODE = {"N6": 50.0, "N8": 50.0}
STORAGE_DURATION_HOURS = 4.0
DEGRADATION_EUR_PER_MWH = 15.0
TIKHONOV_GAMMA = 1.0e-3
PRICE_BOUND_EUR_PER_MWH = 500.0
OTHER_DUAL_BOUND = 10_000.0
SOLVER_TOL = 1.0e-7
MAX_CPU_TIME_SECONDS = 120.0
TEE = False
# -----------------------------------------------------------------------------


def build_soft_balance_primal_llp(
    data,
    *,
    degradation_eur_per_mwh_by_unit: Mapping[str, float],
    gamma: float,
    demand_expansion=None,
) -> pyo.ConcreteModel:
    """Build ``min C(x) + ||h(x)||^2/(2*gamma)``."""

    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    m = build_primal_llp(
        data,
        degradation_eur_per_mwh_by_unit=degradation_eur_per_mwh_by_unit,
        demand_expansion=demand_expansion,
    )
    m.name = "Soft-balance primal LLP"
    m.nodal_balance.deactivate()
    m.objective.deactivate()

    storage_at_node = {
        node: [unit for unit, pair_node in m.IN if pair_node == node]
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
                for unit in storage_at_node[node]
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
        m.N, m.T, domain=pyo.Reals, initialize=0.0
    )
    m.soft_nodal_balance = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, n, t: model.original_nodal_balance_residual[n, t]
        - model.balance_residual[n, t]
        == 0.0,
    )
    m.unpenalized_primal_objective_expr = pyo.Expression(
        expr=m.generation_cost_expr
        + m.degradation_cost_expr
        - m.demand_expansion_utility_expr
    )
    m.soft_balance_penalty_expr = pyo.Expression(
        expr=sum(
            m.balance_residual[n, t] ** 2
            for n in m.N
            for t in m.T
        )
        / (2.0 * gamma)
    )
    m.soft_objective = pyo.Objective(
        expr=m.unpenalized_primal_objective_expr
        + m.soft_balance_penalty_expr,
        sense=pyo.minimize,
    )
    m._tikhonov_gamma = gamma
    return m


def soft_balance_prices(model: pyo.ConcreteModel) -> dict[tuple[str, int], float]:
    """Recover the unique economic prices from ``h = -gamma*lambda``."""

    gamma = float(model._tikhonov_gamma)
    return {
        (str(n), int(t)): -pyo.value(model.balance_residual[n, t]) / gamma
        for n in model.N
        for t in model.T
    }


def solve_matched_soft_market(
    data,
    *,
    degradation_eur_per_mwh_by_unit: Mapping[str, float],
    gamma: float,
    solver_tol: float,
    max_cpu_time: float,
    price_bound_eur_per_mwh: float = PRICE_BOUND_EUR_PER_MWH,
    other_dual_bound: float = OTHER_DUAL_BOUND,
    demand_expansion=None,
    tee: bool = False,
) -> tuple[pyo.ConcreteModel, pyo.ConcreteModel, dict[str, float | str]]:
    """Solve the matched primal and dual and return cross-check diagnostics."""

    primal = build_soft_balance_primal_llp(
        data,
        degradation_eur_per_mwh_by_unit=degradation_eur_per_mwh_by_unit,
        gamma=gamma,
        demand_expansion=demand_expansion,
    )
    primal_results = solve_ipopt(
        primal,
        solver_tol=solver_tol,
        max_cpu_time=max_cpu_time,
        tee=tee,
    )
    dual = build_tikhonov_dual_llp(
        data,
        degradation_eur_per_mwh_by_unit=degradation_eur_per_mwh_by_unit,
        gamma=gamma,
        price_bound_eur_per_mwh=price_bound_eur_per_mwh,
        other_dual_bound=other_dual_bound,
        demand_expansion=demand_expansion,
    )
    dual_results = solve_ipopt(
        dual,
        solver_tol=solver_tol,
        max_cpu_time=max_cpu_time,
        tee=tee,
    )

    primal_termination = str(primal_results.solver.termination_condition)
    dual_termination = str(dual_results.solver.termination_condition)
    diagnostics: dict[str, float | str] = {
        "primal_termination": primal_termination,
        "dual_termination": dual_termination,
    }
    if primal_termination != "optimal" or dual_termination != "optimal":
        return primal, dual, diagnostics

    primal_prices = soft_balance_prices(primal)
    dual_prices = {
        (str(n), int(t)): pyo.value(dual.lam[n, t])
        for n in dual.N
        for t in dual.T
    }
    primal_value = pyo.value(primal.soft_objective)
    dual_value = pyo.value(dual.regularized_objective)
    diagnostics.update(
        {
            "primal_regularized_objective_eur_per_day": primal_value,
            "dual_regularized_objective_eur_per_day": dual_value,
            "absolute_strong_duality_gap_eur_per_day": abs(
                primal_value - dual_value
            ),
            "max_abs_primal_vs_dual_lambda_eur_per_mwh": max(
                abs(primal_prices[key] - dual_prices[key])
                for key in primal_prices
            ),
            "max_abs_original_nodal_balance_residual_mw": max(
                abs(pyo.value(primal.balance_residual[n, t]))
                for n in primal.N
                for t in primal.T
            ),
            "max_abs_h_plus_gamma_lambda_mw": max(
                abs(
                    pyo.value(primal.balance_residual[n, t])
                    + gamma * dual_prices[str(n), int(t)]
                )
                for n in primal.N
                for t in primal.T
            ),
        }
    )
    return primal, dual, diagnostics


def main() -> int:
    data = load_fixed_storage_case(
        Path(DATA_PATH),
        storage_id=STORAGE_ID,
        power_mw_by_node=STORAGE_POWER_MW_BY_NODE,
        duration_hours=STORAGE_DURATION_HOURS,
    )
    _, _, diagnostics = solve_matched_soft_market(
        data,
        degradation_eur_per_mwh_by_unit={
            STORAGE_ID: DEGRADATION_EUR_PER_MWH
        },
        gamma=TIKHONOV_GAMMA,
        solver_tol=SOLVER_TOL,
        max_cpu_time=MAX_CPU_TIME_SECONDS,
        tee=TEE,
    )
    for key, value in diagnostics.items():
        print(f"{key}={value}")
    return 0 if diagnostics.get("primal_termination") == diagnostics.get("dual_termination") == "optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
