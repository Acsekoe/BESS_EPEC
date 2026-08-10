"""Tikhonov-regularized dual LLP, with optional primary-optimal-face lock."""

from __future__ import annotations

from pathlib import Path

import pyomo.environ as pyo

try:
    from .common import DEFAULT_DATA_PATH, load_fixed_storage_case, solve_ipopt
    from .dual_llp import build_dual_llp
    from .primal_llp import build_primal_llp
except ImportError:
    from common import DEFAULT_DATA_PATH, load_fixed_storage_case, solve_ipopt
    from dual_llp import build_dual_llp
    from primal_llp import build_primal_llp


# USER CONTROLS ---------------------------------------------------------------
DATA_PATH = DEFAULT_DATA_PATH
STORAGE_ID = "BESS_FIXED"
STORAGE_POWER_MW_BY_NODE = {"N6": 50.0, "N8": 50.0}
STORAGE_DURATION_HOURS = 4.0
DEGRADATION_EUR_PER_MWH = 15.0
TIKHONOV_GAMMA = 1.0e-3
LOCK_TO_PRIMAL_OPTIMUM = False
PRICE_BOUND_EUR_PER_MWH = 500.0
OTHER_DUAL_BOUND = 10_000.0
SOLVER_TOL = 1.0e-6
MAX_CPU_TIME_SECONDS = 120.0
TEE = False
# -----------------------------------------------------------------------------


def build_tikhonov_dual_llp(
    data,
    *,
    degradation_eur_per_mwh_by_unit,
    gamma: float,
    primary_optimum_value: float | None = None,
    price_bound_eur_per_mwh: float = PRICE_BOUND_EUR_PER_MWH,
    other_dual_bound: float = OTHER_DUAL_BOUND,
    demand_expansion=None,
) -> pyo.ConcreteModel:
    """Build ``max D(lambda,...) - gamma/2 ||lambda||²``.

    Without ``primary_optimum_value``, this is the direct Tikhonov dual. Its KKT
    system perturbs the primal nodal balances. With a supplied primary optimum,
    the unregularized dual objective is locked to the primary-optimal face and
    the quadratic term only selects prices within that face.
    """

    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    m = build_dual_llp(
        data,
        degradation_eur_per_mwh_by_unit=degradation_eur_per_mwh_by_unit,
        price_bound_eur_per_mwh=price_bound_eur_per_mwh,
        other_dual_bound=other_dual_bound,
        demand_expansion=demand_expansion,
    )
    m.name = "Tikhonov-regularized dual LLP"
    m.objective.deactivate()
    m.lambda_squared_norm_expr = pyo.Expression(
        expr=sum(m.lam[n, t] ** 2 for n in m.N for t in m.T)
    )
    m.tikhonov_penalty_expr = pyo.Expression(
        expr=0.5 * gamma * m.lambda_squared_norm_expr
    )
    if primary_optimum_value is not None:
        m.primary_optimal_face_lock = pyo.Constraint(
            expr=m.unregularized_dual_objective_expr
            == float(primary_optimum_value)
        )
    m.regularized_objective = pyo.Objective(
        expr=m.unregularized_dual_objective_expr - m.tikhonov_penalty_expr,
        sense=pyo.maximize,
    )
    m._tikhonov_gamma = gamma
    m._primary_optimum_value = primary_optimum_value
    return m


def main() -> int:
    data = load_fixed_storage_case(
        Path(DATA_PATH),
        storage_id=STORAGE_ID,
        power_mw_by_node=STORAGE_POWER_MW_BY_NODE,
        duration_hours=STORAGE_DURATION_HOURS,
    )
    degradation = {STORAGE_ID: DEGRADATION_EUR_PER_MWH}
    primary_optimum = None
    if LOCK_TO_PRIMAL_OPTIMUM:
        primal = build_primal_llp(
            data, degradation_eur_per_mwh_by_unit=degradation
        )
        primal_results = solve_ipopt(
            primal,
            solver_tol=SOLVER_TOL,
            max_cpu_time=MAX_CPU_TIME_SECONDS,
            tee=TEE,
        )
        if str(primal_results.solver.termination_condition) != "optimal":
            print(
                "primal lock solve failed: "
                f"{primal_results.solver.termination_condition}"
            )
            return 1
        primary_optimum = pyo.value(primal.objective)

    model = build_tikhonov_dual_llp(
        data,
        degradation_eur_per_mwh_by_unit=degradation,
        gamma=TIKHONOV_GAMMA,
        primary_optimum_value=primary_optimum,
    )
    results = solve_ipopt(
        model,
        solver_tol=SOLVER_TOL,
        max_cpu_time=MAX_CPU_TIME_SECONDS,
        tee=TEE,
    )
    termination = str(results.solver.termination_condition)
    print(f"termination={termination}")
    if termination != "optimal":
        return 1
    dual_value = pyo.value(model.unregularized_dual_objective_expr)
    penalty = pyo.value(model.tikhonov_penalty_expr)
    prices = [pyo.value(model.lam[n, t]) for n in model.N for t in model.T]
    print(f"locked_to_primary_optimum={LOCK_TO_PRIMAL_OPTIMUM}")
    print(f"unregularized dual objective={dual_value:,.6f} EUR/day")
    print(f"Tikhonov penalty={penalty:,.6f}")
    print(f"regularized objective={dual_value - penalty:,.6f}")
    if primary_optimum is not None:
        print(f"primary-dual gap={primary_optimum - dual_value:.6e}")
    print(f"lambda range=[{min(prices):.6f}, {max(prices):.6f}] EUR/MWh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

