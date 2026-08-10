"""Runnable single-investor strong-duality MPEC with price-responsive demand.

Each investor is solved once against the same market. ``DUAL_TIKHONOV_GAMMA``
selects the lower-level treatment: a positive value uses the finite-gamma soft
market, and ``None`` uses exact market clearing (primal feasibility, dual
feasibility, and the strong-duality equality, with no soft balance residual).

Exact clearing is available because the demand-expansion block makes the lower
level strictly convex in demand, which pins the nodal price to the marginal
willingness to pay wherever the block is interior.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyomo.environ as pyo

try:
    from .common import DEFAULT_DATA_PATH, MODEL_DIR, load_calibrated_case, solve_ipopt
    from .strong_duality_formulation import (
        audit_mpec_prices_against_soft_market,
        build_single_investor_tikhonov_strong_duality_mpec,
        initialize_mpec_from_soft_market,
        strong_duality_diagnostics,
    )
except ImportError:
    from common import DEFAULT_DATA_PATH, MODEL_DIR, load_calibrated_case, solve_ipopt
    from strong_duality_formulation import (
        audit_mpec_prices_against_soft_market,
        build_single_investor_tikhonov_strong_duality_mpec,
        initialize_mpec_from_soft_market,
        strong_duality_diagnostics,
    )

from single_investor_mpec import (
    DemandExpansionCurve,
    InvestorConfig,
    QuadraticDemandCurve,
    build_single_investor_mpec,
)


# USER CONTROLS ---------------------------------------------------------------
DATA_PATH = DEFAULT_DATA_PATH
OUTPUT_DIR = MODEL_DIR / "output" / "tikhonov_single_investor_elastic_demand"
NODE_LIMIT_MW = 200.0
INITIAL_POWER_MW_PER_NODE = 10.0
INITIAL_DURATION_HOURS = 4.0
PRICE_BOUND_EUR_PER_MWH = 500.0
OTHER_DUAL_BOUND = 10_000.0
# None runs exact market clearing; a positive float runs the soft market.
DUAL_TIKHONOV_GAMMA: float | None = None
# Warm start only. A small value makes the soft primal's 1/(2*gamma) penalty
# near-singular and costs far more time than the starting point is worth.
WARM_START_GAMMA = 1.0e-3
DEMAND_REFERENCE_PRICE_EUR_PER_MWH = 60.0
DEMAND_ELASTICITY = 0.20
DISPATCH_REGULARIZATION_EUR_PER_MW2H = 0.0
SOLVER_TOL = 1.0e-6
AUDIT_SOLVER_TOL = 1.0e-8
MAX_CPU_TIME_SECONDS = 600.0
TEE = True
# Restrict to a subset, e.g. ("I1",), to watch one solve. Empty runs all.
ONLY_INVESTOR_IDS: tuple[str, ...] = ()

# The four thesis investors. Portfolio shares stay empty here: this script is
# the isolated single-investor best response, not the EPEC.
INVESTORS = (
    InvestorConfig(investor_id="I1", wacc=0.08),
    InvestorConfig(investor_id="I2", wacc=0.12),
    InvestorConfig(investor_id="I3", wacc=0.08),
    InvestorConfig(investor_id="I4", wacc=0.08),
)
# -----------------------------------------------------------------------------


def demand_expansion_curve() -> DemandExpansionCurve:
    return DemandExpansionCurve(
        reference_price_eur_per_mwh=DEMAND_REFERENCE_PRICE_EUR_PER_MWH,
        elasticity=DEMAND_ELASTICITY,
    )


def build_experimental_mpec(data, investor: InvestorConfig) -> pyo.ConcreteModel:
    kwargs = dict(
        quad_demand=QuadraticDemandCurve(alpha=100.0, beta=0.1),
        demand_expansion=demand_expansion_curve(),
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
    )
    if DUAL_TIKHONOV_GAMMA is None:
        return build_single_investor_mpec(data, **kwargs)
    return build_single_investor_tikhonov_strong_duality_mpec(
        data, dual_tikhonov_gamma=DUAL_TIKHONOV_GAMMA, **kwargs
    )


def demand_response_diagnostics(model: pyo.ConcreteModel) -> dict[str, float]:
    """Report how far realized demand sits from its reference and its cap."""

    band = model._demand_expansion_band_mw
    reference_mwh = sum(
        model._market_data.demand_el[n, t] for n in model.N for t in model.T
    )
    extra = {key: pyo.value(model.E_extra[key]) for key in band}
    served_above_reference = sum(extra.values())
    withheld_vs_free_power = sum(band[key] - extra[key] for key in band)
    return {
        "reference_demand_mwh_per_day": reference_mwh,
        "expansion_band_mwh_per_day": sum(band.values()),
        "realized_expansion_mwh_per_day": served_above_reference,
        "realized_expansion_share_of_reference": (
            served_above_reference / reference_mwh if reference_mwh else 0.0
        ),
        "price_withheld_expansion_mwh_per_day": withheld_vs_free_power,
        "price_withheld_share_of_band": (
            withheld_vs_free_power / sum(band.values()) if band else 0.0
        ),
        "curtailed_demand_below_reference_mwh_per_day": 0.0,
    }


def solve_experimental_mpec(
    data, investor: InvestorConfig
) -> tuple[pyo.ConcreteModel, dict]:
    model = build_experimental_mpec(data, investor)
    initialization = initialize_mpec_from_soft_market(
        model,
        solver_tol=SOLVER_TOL,
        max_cpu_time=MAX_CPU_TIME_SECONDS,
        gamma=(
            WARM_START_GAMMA
            if DUAL_TIKHONOV_GAMMA is None
            else DUAL_TIKHONOV_GAMMA
        ),
        tee=False,
    )
    results = solve_ipopt(
        model,
        solver_tol=SOLVER_TOL,
        max_cpu_time=MAX_CPU_TIME_SECONDS,
        tee=TEE,
    )
    termination = str(results.solver.termination_condition)
    summary = {
        "investor_id": investor.investor_id,
        "wacc": investor.wacc,
        "termination": termination,
        "lower_level": (
            "exact" if DUAL_TIKHONOV_GAMMA is None else "soft-gamma"
        ),
        "dual_tikhonov_gamma": DUAL_TIKHONOV_GAMMA,
        "demand_reference_price_eur_per_mwh": DEMAND_REFERENCE_PRICE_EUR_PER_MWH,
        "demand_elasticity": DEMAND_ELASTICITY,
        "initialization": initialization,
        "investor_profit_eur_per_day": pyo.value(model.investor_profit_expr),
        "investment_power_mw": sum(
            pyo.value(model.X_power[n]) for n in model.N
        ),
        "investment_energy_mwh": sum(
            pyo.value(model.X_energy[n]) for n in model.N
        ),
        "investment_power_mw_by_node": {
            str(n): pyo.value(model.X_power[n]) for n in model.N
        },
        "investment_energy_mwh_by_node": {
            str(n): pyo.value(model.X_energy[n]) for n in model.N
        },
        "strong_duality_diagnostics": strong_duality_diagnostics(model),
        "demand_response": demand_response_diagnostics(model),
    }
    if termination == "optimal" and DUAL_TIKHONOV_GAMMA is not None:
        summary["same_fleet_soft_market_audit"] = (
            audit_mpec_prices_against_soft_market(
                model,
                gamma=DUAL_TIKHONOV_GAMMA,
                solver_tol=AUDIT_SOLVER_TOL,
                max_cpu_time=MAX_CPU_TIME_SECONDS,
                tee=False,
            )
        )
    return model, summary


def main() -> int:
    data = load_calibrated_case(Path(DATA_PATH))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    selected = [
        investor
        for investor in INVESTORS
        if not ONLY_INVESTOR_IDS or investor.investor_id in ONLY_INVESTOR_IDS
    ]
    for investor in selected:
        started = time.time()
        _, summary = solve_experimental_mpec(data, investor)
        summary["solve_seconds"] = time.time() - started
        summaries.append(summary)
        demand = summary["demand_response"]
        diag = summary["strong_duality_diagnostics"]
        print(
            f"{summary['investor_id']} wacc={investor.wacc:.0%} "
            f"{summary['termination']:>10s} | "
            f"{summary['investment_power_mw']:8.3f} MW / "
            f"{summary['investment_energy_mwh']:9.3f} MWh | "
            f"profit={summary['investor_profit_eur_per_day']:10.2f} EUR/day | "
            f"SD gap={diag['matched_strong_duality_gap_eur_per_day']:+.2e} | "
            f"lam=[{diag['lambda_min_eur_per_mwh']:.2f}, "
            f"{diag['lambda_max_eur_per_mwh']:.2f}] | "
            f"extra demand={demand['realized_expansion_mwh_per_day']:.1f} MWh "
            f"({demand['realized_expansion_share_of_reference']:.2%}) | "
            f"{summary['solve_seconds']:.0f}s",
            flush=True,
        )
    output_path = OUTPUT_DIR / "summary.json"
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {len(summaries)} investor summaries to {output_path}")
    return 0 if all(s["termination"] == "optimal" for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
