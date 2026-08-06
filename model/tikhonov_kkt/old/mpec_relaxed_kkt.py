"""Single-investor MPEC with direct Tikhonov and relaxed KKT conditions."""

from __future__ import annotations

from pathlib import Path

import pyomo.environ as pyo

try:
    from ..common import DEFAULT_DATA_PATH, load_calibrated_case, solve_ipopt
except ImportError:
    from tikhonov_kkt.common import DEFAULT_DATA_PATH, load_calibrated_case, solve_ipopt

from single_investor_mpec import (
    InvestorConfig,
    QuadraticDemandCurve,
    initialize_from_reference_dispatch,
)
try:
    from .kkt_formulation import (
        build_single_investor_relaxed_kkt_mpec,
        relaxed_kkt_diagnostics,
    )
except ImportError:
    from tikhonov_kkt.old.kkt_formulation import (
        build_single_investor_relaxed_kkt_mpec,
        relaxed_kkt_diagnostics,
    )


# USER CONTROLS ---------------------------------------------------------------
DATA_PATH = DEFAULT_DATA_PATH
INVESTOR_ID = "I1"
WACC = 0.08
NODE_LIMIT_MW = 200.0
INITIAL_POWER_MW_PER_NODE = 10.0
INITIAL_DURATION_HOURS = 4.0
PRICE_BOUND_EUR_PER_MWH = 500.0
OTHER_DUAL_BOUND = 10_000.0

# Direct dual regularization: max D - (gamma/2) * sum(lambda**2).
# At finite gamma its KKT equation is original_balance + gamma*lambda == 0,
# so this is an approximation with explicitly measured physical imbalance.
DUAL_TIKHONOV_GAMMA = 1.0e-3

# Scholtes relaxation: every non-negative complementarity product satisfies
# slack * multiplier <= epsilon.
COMPLEMENTARITY_EPSILON = 1.0e-3
COMPLEMENTARITY_FORMULATION = "scholtes"  # "scholtes" or "central-path"
COMPLEMENTARITY_SHIFT = 1.0e-8

DISPATCH_REGULARIZATION_EUR_PER_MW2H = 0.0
SOLVER_TOL = 1.0e-4
MAX_CPU_TIME_SECONDS = 180.0
TEE = False
# -----------------------------------------------------------------------------


def build_experimental_mpec(data) -> pyo.ConcreteModel:
    investor = InvestorConfig(investor_id=INVESTOR_ID, wacc=WACC)
    return build_single_investor_relaxed_kkt_mpec(
        data,
        quad_demand=QuadraticDemandCurve(alpha=100.0, beta=0.1),
        investor=investor,
        node_limit_mw=NODE_LIMIT_MW,
        initial_power_mw=INITIAL_POWER_MW_PER_NODE,
        initial_ratio_hours=INITIAL_DURATION_HOURS,
        price_bound_eur_per_mwh=PRICE_BOUND_EUR_PER_MWH,
        dual_bound_eur_per_mwh=OTHER_DUAL_BOUND,
        use_demand_curve=False,
        dispatch_regularization_eur_per_mw2h=(
            DISPATCH_REGULARIZATION_EUR_PER_MW2H
        ),
        solver_tol=SOLVER_TOL,
        complementarity_epsilon=COMPLEMENTARITY_EPSILON,
        complementarity_formulation=COMPLEMENTARITY_FORMULATION,
        complementarity_shift=COMPLEMENTARITY_SHIFT,
        dual_tikhonov_gamma=DUAL_TIKHONOV_GAMMA,
    )


def main() -> int:
    data = load_calibrated_case(Path(DATA_PATH))
    model = build_experimental_mpec(data)
    initialize_from_reference_dispatch(model, data, INITIAL_DURATION_HOURS)
    results = solve_ipopt(
        model,
        solver_tol=SOLVER_TOL,
        max_cpu_time=MAX_CPU_TIME_SECONDS,
        tee=TEE,
    )
    termination = str(results.solver.termination_condition)
    print(f"termination={termination}")
    diagnostics = relaxed_kkt_diagnostics(model)
    print(f"investor profit={pyo.value(model.investor_profit_expr):,.6f} EUR/day")
    print(
        "investment="
        f"{sum(pyo.value(model.X_power[n]) for n in model.N):.6f} MW / "
        f"{sum(pyo.value(model.X_energy[n]) for n in model.N):.6f} MWh"
    )
    for key, value in diagnostics.items():
        print(f"{key}={value}")
    return 0 if termination == "optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
