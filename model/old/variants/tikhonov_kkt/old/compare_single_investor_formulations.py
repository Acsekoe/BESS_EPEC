"""Compare exact strong duality and relaxed KKT at the same finite gamma."""

from __future__ import annotations

import json
from pathlib import Path

import pyomo.environ as pyo

try:
    from ..common import DEFAULT_DATA_PATH, MODEL_DIR, load_calibrated_case, solve_ipopt
    from .kkt_formulation import (
        build_single_investor_relaxed_kkt_mpec,
        relaxed_kkt_diagnostics,
    )
    from ..mpec_strong_duality import (
        DUAL_TIKHONOV_GAMMA,
        INITIAL_DURATION_HOURS,
        INITIAL_POWER_MW_PER_NODE,
        INVESTOR_ID,
        NODE_LIMIT_MW,
        OTHER_DUAL_BOUND,
        PRICE_BOUND_EUR_PER_MWH,
        WACC,
        solve_experimental_mpec,
    )
    from ..strong_duality_formulation import (
        audit_mpec_prices_against_soft_market,
        initialize_mpec_from_soft_market,
    )
except ImportError:
    from tikhonov_kkt.common import DEFAULT_DATA_PATH, MODEL_DIR, load_calibrated_case, solve_ipopt
    from tikhonov_kkt.old.kkt_formulation import (
        build_single_investor_relaxed_kkt_mpec,
        relaxed_kkt_diagnostics,
    )
    from tikhonov_kkt.mpec_strong_duality import (
        DUAL_TIKHONOV_GAMMA,
        INITIAL_DURATION_HOURS,
        INITIAL_POWER_MW_PER_NODE,
        INVESTOR_ID,
        NODE_LIMIT_MW,
        OTHER_DUAL_BOUND,
        PRICE_BOUND_EUR_PER_MWH,
        WACC,
        solve_experimental_mpec,
    )
    from tikhonov_kkt.strong_duality_formulation import (
        audit_mpec_prices_against_soft_market,
        initialize_mpec_from_soft_market,
    )

from single_investor_mpec import InvestorConfig, QuadraticDemandCurve


# USER CONTROLS ---------------------------------------------------------------
OUTPUT_DIR = MODEL_DIR / "output" / "tikhonov_single_investor_formulation_comparison"
COMPLEMENTARITY_EPSILON = 1.0e-3
EXACT_SOLVER_TOL = 1.0e-6
RELAXED_SOLVER_TOL = 1.0e-4
AUDIT_SOLVER_TOL = 1.0e-8
MAX_CPU_TIME_SECONDS = 300.0
TEE = False
# -----------------------------------------------------------------------------


def build_relaxed_mpec(data) -> pyo.ConcreteModel:
    return build_single_investor_relaxed_kkt_mpec(
        data,
        quad_demand=QuadraticDemandCurve(alpha=100.0, beta=0.1),
        investor=InvestorConfig(investor_id=INVESTOR_ID, wacc=WACC),
        node_limit_mw=NODE_LIMIT_MW,
        initial_power_mw=INITIAL_POWER_MW_PER_NODE,
        initial_ratio_hours=INITIAL_DURATION_HOURS,
        price_bound_eur_per_mwh=PRICE_BOUND_EUR_PER_MWH,
        dual_bound_eur_per_mwh=OTHER_DUAL_BOUND,
        use_demand_curve=False,
        solver_tol=RELAXED_SOLVER_TOL,
        complementarity_epsilon=COMPLEMENTARITY_EPSILON,
        complementarity_formulation="scholtes",
        dual_tikhonov_gamma=DUAL_TIKHONOV_GAMMA,
    )


def main() -> int:
    data = load_calibrated_case(Path(DEFAULT_DATA_PATH))

    print("Solving exact Tikhonov strong-duality MPEC...", flush=True)
    _, exact_summary = solve_experimental_mpec(data)

    print("Solving relaxed-KKT MPEC at the same gamma...", flush=True)
    relaxed = build_relaxed_mpec(data)
    relaxed_initialization = initialize_mpec_from_soft_market(
        relaxed,
        solver_tol=RELAXED_SOLVER_TOL,
        max_cpu_time=MAX_CPU_TIME_SECONDS,
        tee=False,
    )
    relaxed_results = solve_ipopt(
        relaxed,
        solver_tol=RELAXED_SOLVER_TOL,
        max_cpu_time=MAX_CPU_TIME_SECONDS,
        tee=TEE,
    )
    relaxed_termination = str(relaxed_results.solver.termination_condition)
    relaxed_summary = {
        "termination": relaxed_termination,
        "initialization": relaxed_initialization,
        "investor_profit_eur_per_day": pyo.value(relaxed.investor_profit_expr),
        "investment_power_mw": sum(pyo.value(relaxed.X_power[n]) for n in relaxed.N),
        "investment_energy_mwh": sum(pyo.value(relaxed.X_energy[n]) for n in relaxed.N),
        "relaxed_kkt_diagnostics": relaxed_kkt_diagnostics(relaxed),
    }
    if relaxed_termination == "optimal":
        relaxed_summary["same_fleet_soft_market_audit"] = (
            audit_mpec_prices_against_soft_market(
                relaxed,
                gamma=DUAL_TIKHONOV_GAMMA,
                solver_tol=AUDIT_SOLVER_TOL,
                max_cpu_time=MAX_CPU_TIME_SECONDS,
                tee=False,
            )
        )

    comparison = {
        "gamma": DUAL_TIKHONOV_GAMMA,
        "relaxed_complementarity_epsilon": COMPLEMENTARITY_EPSILON,
        "exact_strong_duality": exact_summary,
        "relaxed_kkt": relaxed_summary,
    }
    if exact_summary["termination"] == relaxed_summary["termination"] == "optimal":
        comparison["exact_minus_relaxed"] = {
            "investment_power_mw": exact_summary["investment_power_mw"]
            - relaxed_summary["investment_power_mw"],
            "investment_energy_mwh": exact_summary["investment_energy_mwh"]
            - relaxed_summary["investment_energy_mwh"],
            "investor_profit_eur_per_day": exact_summary[
                "investor_profit_eur_per_day"
            ]
            - relaxed_summary["investor_profit_eur_per_day"],
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "summary.json"
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"Wrote comparison to {output_path}")
    return 0 if exact_summary["termination"] == relaxed_summary["termination"] == "optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
