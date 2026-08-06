"""Solve one standalone exact strong-duality MPEC per thesis investor.

Each solve has one strategic investor and no rival BESS.  The resulting price
vectors therefore answer which market that investor anticipates at its own
optimal fleet; they are not four best responses to a common rival snapshot.
"""

from __future__ import annotations

import csv
import json
import time
from itertools import combinations
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

from epec_diagonalization import four_investor_portfolio_profiles
from single_investor_mpec import QuadraticDemandCurve


# USER CONTROLS ---------------------------------------------------------------
DATA_PATH = DEFAULT_DATA_PATH
OUTPUT_DIR = MODEL_DIR / "output" / "tikhonov_four_standalone_strong_duality"
NODE_LIMIT_MW = 200.0
INITIAL_POWER_MW_PER_NODE = 10.0
INITIAL_DURATION_HOURS = 4.0
PRICE_BOUND_EUR_PER_MWH = 500.0
OTHER_DUAL_BOUND = 10_000.0
DUAL_TIKHONOV_GAMMA = 1.0e-3
DISPATCH_REGULARIZATION_EUR_PER_MW2H = 0.0
SOLVER_TOL = 1.0e-6
AUDIT_SOLVER_TOL = 1.0e-8
MAX_CPU_TIME_SECONDS = 300.0
TEE = False
# -----------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_mpec(data, investor) -> pyo.ConcreteModel:
    return build_single_investor_tikhonov_strong_duality_mpec(
        data,
        dual_tikhonov_gamma=DUAL_TIKHONOV_GAMMA,
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
    )


def _price_key(node: object, hour: object) -> tuple[str, int]:
    return str(node), int(hour)


def _write_partial_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    data = load_calibrated_case(Path(DATA_PATH))
    investors = four_investor_portfolio_profiles(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_summary: dict[str, object] = {
        "experiment": "four standalone single-investor exact strong-duality MPECs",
        "interpretation": (
            "Each investor optimizes with no rival BESS. Cross-investor prices can "
            "differ because the four MPECs choose different fleets. The same-fleet "
            "soft-market audit tests price consistency within each MPEC."
        ),
        "dual_tikhonov_gamma": DUAL_TIKHONOV_GAMMA,
        "investors": {},
    }
    price_vectors: dict[str, dict[tuple[str, int], float]] = {}
    capacity_rows: list[dict[str, object]] = []
    all_optimal = True

    for position, investor in enumerate(investors, start=1):
        investor_id = investor.investor_id
        print(
            f"[{position}/{len(investors)}] Solving standalone exact MPEC for "
            f"{investor_id} (WACC={100.0 * investor.wacc:.1f}%, "
            f"owned generators={len(investor.owned_generation_shares)})...",
            flush=True,
        )
        started = time.perf_counter()
        model = _build_mpec(data, investor)
        initialization = initialize_mpec_from_soft_market(
            model,
            solver_tol=SOLVER_TOL,
            max_cpu_time=MAX_CPU_TIME_SECONDS,
            tee=False,
        )
        results = solve_ipopt(
            model,
            solver_tol=SOLVER_TOL,
            max_cpu_time=MAX_CPU_TIME_SECONDS,
            tee=TEE,
        )
        termination = str(results.solver.termination_condition)
        all_optimal = all_optimal and termination == "optimal"

        power_by_node = {
            str(node): pyo.value(model.X_power[node]) for node in model.N
        }
        energy_by_node = {
            str(node): pyo.value(model.X_energy[node]) for node in model.N
        }
        investor_summary: dict[str, object] = {
            "termination": termination,
            "wacc": investor.wacc,
            "owned_generation_shares": dict(investor.owned_generation_shares),
            "solve_elapsed_seconds_including_initialization": (
                time.perf_counter() - started
            ),
            "initialization": initialization,
            "investor_profit_eur_per_day": pyo.value(model.investor_profit_expr),
            "investment_power_mw": sum(power_by_node.values()),
            "investment_energy_mwh": sum(energy_by_node.values()),
            "investment_power_mw_by_node": power_by_node,
            "investment_energy_mwh_by_node": energy_by_node,
            "strong_duality_diagnostics": strong_duality_diagnostics(model),
        }

        if termination == "optimal":
            print(f"[{investor_id}] Auditing prices at its proposed fleet...", flush=True)
            investor_summary["same_fleet_soft_market_audit"] = (
                audit_mpec_prices_against_soft_market(
                    model,
                    gamma=DUAL_TIKHONOV_GAMMA,
                    solver_tol=AUDIT_SOLVER_TOL,
                    max_cpu_time=MAX_CPU_TIME_SECONDS,
                    tee=False,
                )
            )

        price_vectors[investor_id] = {
            _price_key(node, hour): pyo.value(model.lam[node, hour])
            for node in model.N
            for hour in model.T
        }
        run_summary["investors"][investor_id] = investor_summary
        for node in model.N:
            capacity_rows.append(
                {
                    "investor": investor_id,
                    "node": str(node),
                    "power_mw": power_by_node[str(node)],
                    "energy_mwh": energy_by_node[str(node)],
                }
            )
        _write_partial_summary(OUTPUT_DIR / "summary.json", run_summary)
        print(
            f"[{investor_id}] {termination}: {sum(power_by_node.values()):.6f} MW, "
            f"{sum(energy_by_node.values()):.6f} MWh, "
            f"profit={pyo.value(model.investor_profit_expr):.6f} EUR/day",
            flush=True,
        )

    investor_ids = [investor.investor_id for investor in investors]
    common_keys = sorted(
        set.intersection(*(set(price_vectors[item]) for item in investor_ids)),
        key=lambda item: (item[1], item[0]),
    )
    price_rows: list[dict[str, object]] = []
    for node, hour in common_keys:
        row: dict[str, object] = {"hour": hour, "node": node}
        prices = []
        for investor_id in investor_ids:
            price = price_vectors[investor_id][node, hour]
            row[f"lambda_{investor_id}_eur_per_mwh"] = price
            prices.append(price)
        row["cross_investor_range_eur_per_mwh"] = max(prices) - min(prices)
        price_rows.append(row)

    pairwise_rows: list[dict[str, object]] = []
    for left, right in combinations(investor_ids, 2):
        differences = [
            abs(price_vectors[left][key] - price_vectors[right][key])
            for key in common_keys
        ]
        pairwise_rows.append(
            {
                "investor_left": left,
                "investor_right": right,
                "max_abs_price_difference_eur_per_mwh": max(differences),
                "mean_abs_price_difference_eur_per_mwh": (
                    sum(differences) / len(differences)
                ),
            }
        )
    run_summary["pairwise_anticipated_price_differences"] = pairwise_rows
    run_summary["max_hour_node_cross_investor_price_range_eur_per_mwh"] = max(
        row["cross_investor_range_eur_per_mwh"] for row in price_rows
    )

    _write_csv(OUTPUT_DIR / "anticipated_prices.csv", price_rows)
    _write_csv(OUTPUT_DIR / "capacities.csv", capacity_rows)
    _write_csv(OUTPUT_DIR / "pairwise_price_differences.csv", pairwise_rows)
    _write_partial_summary(OUTPUT_DIR / "summary.json", run_summary)
    print(json.dumps(run_summary, indent=2), flush=True)
    print(f"Wrote four-investor comparison to {OUTPUT_DIR}", flush=True)
    return 0 if all_optimal else 1


if __name__ == "__main__":
    raise SystemExit(main())
