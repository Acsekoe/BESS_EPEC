"""Strategic-operation MPEC with exact finite-gamma strong duality.

The active investor chooses nodal BESS MW/MWh, hourly charging and discharging
quantity offers, and (optionally) two-sided bid prices.  For fixed upper-level
decisions the ISO dispatch is represented by the matched Tikhonov pair

    min C_bid(x) + ||h||^2 / (2*gamma)
    max D_bid(y) - (gamma/2) ||lambda||^2.

Primal feasibility, dual feasibility, ``h + gamma*lambda = 0``, and strong
duality replace complementarity products.  Physical degradation remains in
the investor profit even when submitted bid prices replace it in the ISO
objective.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyomo.environ as pyo

try:
    from .common import DEFAULT_DATA_PATH, MODEL_DIR, load_calibrated_case, solve_ipopt
    from .dual_tikhonov_llp import build_tikhonov_dual_llp
    from .soft_balance_llp import build_soft_balance_primal_llp, soft_balance_prices
    from .strong_duality_formulation import (
        apply_tikhonov_strong_duality,
        strong_duality_diagnostics,
    )
except ImportError:
    from common import DEFAULT_DATA_PATH, MODEL_DIR, load_calibrated_case, solve_ipopt
    from dual_tikhonov_llp import build_tikhonov_dual_llp
    from soft_balance_llp import build_soft_balance_primal_llp, soft_balance_prices
    from strong_duality_formulation import (
        apply_tikhonov_strong_duality,
        strong_duality_diagnostics,
    )

from ieee9_strategic_operation_mpec import (
    build_ieee9_strategic_operation_mpec,
    offer_metrics,
)
from single_investor_mpec import (
    InvestorConfig,
    QuadraticDemandCurve,
    fixed_storage_data_from_solution,
)


MODEL_NAME = "Strategic Operation Exact Tikhonov Strong-Duality MPEC"

# USER CONTROLS ---------------------------------------------------------------
DATA_PATH = DEFAULT_DATA_PATH
OUTPUT_DIR = (
    MODEL_DIR
    / "output"
    / "tikhonov_strategic_operation_strong_duality_gamma1e-3"
)
INVESTOR_ID = "I1"
WACC = 0.08
NODE_LIMIT_MW = 200.0
INITIAL_POWER_MW_PER_NODE = 10.0
INITIAL_DURATION_HOURS = 4.0
PRICE_BOUND_EUR_PER_MWH = 500.0
BID_PRICE_BOUND_EUR_PER_MWH = 500.0
OTHER_DUAL_BOUND = 10_000.0
DUAL_TIKHONOV_GAMMA = 1.0e-3
STRATEGIC_BID_PRICES = True
DISPATCH_REGULARIZATION_EUR_PER_MW2H = 0.0
SOLVER_TOL = 1.0e-6
AUDIT_SOLVER_TOL = 1.0e-8
MAX_CPU_TIME_SECONDS = 300.0
TEE = False
# -----------------------------------------------------------------------------


def build_strategic_operation_tikhonov_mpec(
    data,
    *,
    dual_tikhonov_gamma: float,
    **strategic_kwargs: Any,
) -> pyo.ConcreteModel:
    """Build the strategic MPEC, then apply the matched Tikhonov pair.

    ``strategic_kwargs`` intentionally mirrors
    :func:`build_ieee9_strategic_operation_mpec`, including separate rival
    capacities, quantity offers, and bid prices for EPEC best responses.
    """

    if bool(strategic_kwargs.get("use_demand_curve", False)):
        raise ValueError("The Tikhonov strategic MPEC currently supports fixed demand only.")
    if float(strategic_kwargs.get("dispatch_regularization_eur_per_mw2h", 0.0)) != 0.0:
        raise ValueError(
            "Use zero dispatch regularization; the finite-gamma balance term is the "
            "only lower-level regularizer in this formulation."
        )

    model = build_ieee9_strategic_operation_mpec(data, **strategic_kwargs)
    apply_tikhonov_strong_duality(
        model,
        data,
        gamma=dual_tikhonov_gamma,
    )
    model.name = MODEL_NAME
    return model


def _strategy_snapshot(model: pyo.ConcreteModel) -> dict[str, object]:
    """Freeze the MPEC fleet and bids for an independent ISO solve."""

    investor = model._investor_id
    quantities_charge: dict[tuple[str, str, int], float] = {}
    quantities_discharge: dict[tuple[str, str, int], float] = {}
    bid_prices_charge: dict[tuple[str, str, int], float] = {}
    offer_prices_discharge: dict[tuple[str, str, int], float] = {}

    for unit, node in model.IN:
        for time in model.T:
            key = str(unit), str(node), int(time)
            if unit == investor:
                quantities_charge[key] = pyo.value(model.Q_offer_charge[node, time])
                quantities_discharge[key] = pyo.value(
                    model.Q_offer_discharge[node, time]
                )
            else:
                quantities_charge[key] = model._rival_offer_charge_mw_by_unit[
                    unit, node, int(time)
                ]
                quantities_discharge[key] = (
                    model._rival_offer_discharge_mw_by_unit[unit, node, int(time)]
                )

            if getattr(model, "_strategic_price_bids", False):
                if unit == investor:
                    bid_prices_charge[key] = pyo.value(model.p_bid_charge[node, time])
                    offer_prices_discharge[key] = pyo.value(
                        model.p_offer_discharge[node, time]
                    )
                else:
                    bid_prices_charge[key] = (
                        model._rival_bid_price_charge_eur_per_mwh_by_unit[
                            unit, node, int(time)
                        ]
                    )
                    offer_prices_discharge[key] = (
                        model._rival_offer_price_discharge_eur_per_mwh_by_unit[
                            unit, node, int(time)
                        ]
                    )
            else:
                half_degradation = 0.5 * model._storage_degradation_eur_per_mwh[unit]
                bid_prices_charge[key] = -half_degradation
                offer_prices_discharge[key] = half_degradation

    return {
        "quantity_charge_mw": quantities_charge,
        "quantity_discharge_mw": quantities_discharge,
        "bid_price_charge_eur_per_mwh": bid_prices_charge,
        "offer_price_discharge_eur_per_mwh": offer_prices_discharge,
        "strategic_bid_prices": bool(
            getattr(model, "_strategic_price_bids", False)
        ),
    }


def _build_strategic_soft_primal(
    data,
    *,
    degradation_eur_per_mwh_by_unit: Mapping[str, float],
    strategy: Mapping[str, object],
    gamma: float,
) -> pyo.ConcreteModel:
    model = build_soft_balance_primal_llp(
        data,
        degradation_eur_per_mwh_by_unit=degradation_eur_per_mwh_by_unit,
        gamma=gamma,
    )
    quantity_charge = strategy["quantity_charge_mw"]
    quantity_discharge = strategy["quantity_discharge_mw"]

    model.del_component(model.charge_power_bound)
    model.del_component(model.discharge_power_bound)
    model.charge_power_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.P_charge[i, n, t]
        <= quantity_charge[str(i), str(n), int(t)],
    )
    model.discharge_power_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.P_discharge[i, n, t]
        <= quantity_discharge[str(i), str(n), int(t)],
    )

    if strategy["strategic_bid_prices"]:
        charge_price = strategy["bid_price_charge_eur_per_mwh"]
        discharge_price = strategy["offer_price_discharge_eur_per_mwh"]
        model.strategic_bid_cost_expr = pyo.Expression(
            expr=sum(
                discharge_price[str(i), str(n), int(t)]
                * model.P_discharge[i, n, t]
                - charge_price[str(i), str(n), int(t)]
                * model.P_charge[i, n, t]
                for i, n in model.IN
                for t in model.T
            )
        )
        model.unpenalized_primal_objective_expr.set_value(
            model.generation_cost_expr + model.strategic_bid_cost_expr
        )
        model.soft_objective.set_value(
            model.unpenalized_primal_objective_expr
            + model.soft_balance_penalty_expr
        )
    return model


def _build_strategic_tikhonov_dual(
    data,
    *,
    degradation_eur_per_mwh_by_unit: Mapping[str, float],
    strategy: Mapping[str, object],
    gamma: float,
    price_lower_bound_eur_per_mwh: float,
    price_upper_bound_eur_per_mwh: float,
    other_dual_bound: float,
) -> pyo.ConcreteModel:
    symmetric_price_bound = max(
        abs(price_lower_bound_eur_per_mwh), abs(price_upper_bound_eur_per_mwh)
    )
    model = build_tikhonov_dual_llp(
        data,
        degradation_eur_per_mwh_by_unit=degradation_eur_per_mwh_by_unit,
        gamma=gamma,
        price_bound_eur_per_mwh=symmetric_price_bound,
        other_dual_bound=other_dual_bound,
    )
    for component in (model.lam, model.lam_sys):
        for index in component:
            component[index].setlb(price_lower_bound_eur_per_mwh)
            component[index].setub(price_upper_bound_eur_per_mwh)

    charge_price = strategy["bid_price_charge_eur_per_mwh"]
    discharge_price = strategy["offer_price_discharge_eur_per_mwh"]
    eta = data.eta
    model.del_component(model.charge_stationarity)
    model.del_component(model.discharge_stationarity)
    model.charge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: -m.lam[n, t]
        + m.rho_ch[i, n, t]
        - eta * m.gam[i, n, t]
        <= -charge_price[str(i), str(n), int(t)],
    )
    model.discharge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.lam[n, t]
        + m.sig_dis[i, n, t]
        + m.gam[i, n, t] / eta
        <= discharge_price[str(i), str(n), int(t)],
    )

    original_power_term = sum(
        data.x_power[i, n]
        * (model.rho_ch[i, n, t] + model.sig_dis[i, n, t])
        for i, n in model.IN
        for t in model.T
    )
    quantity_charge = strategy["quantity_charge_mw"]
    quantity_discharge = strategy["quantity_discharge_mw"]
    offered_power_term = sum(
        quantity_charge[str(i), str(n), int(t)] * model.rho_ch[i, n, t]
        + quantity_discharge[str(i), str(n), int(t)] * model.sig_dis[i, n, t]
        for i, n in model.IN
        for t in model.T
    )
    model.unregularized_dual_objective_expr.set_value(
        model.unregularized_dual_objective_expr.expr
        - original_power_term
        + offered_power_term
    )
    model.regularized_objective.set_value(
        model.unregularized_dual_objective_expr - model.tikhonov_penalty_expr
    )
    return model


def solve_matched_strategic_soft_market(
    model: pyo.ConcreteModel,
    *,
    solver_tol: float,
    max_cpu_time: float,
    tee: bool = False,
) -> tuple[pyo.ConcreteModel, pyo.ConcreteModel, dict[str, float | str]]:
    """Independently solve the fixed-fleet, fixed-bid matched market pair."""

    fixed_data = fixed_storage_data_from_solution(model)
    strategy = _strategy_snapshot(model)
    degradation = model._storage_degradation_eur_per_mwh
    gamma = float(model._dual_tikhonov_gamma)
    primal = _build_strategic_soft_primal(
        fixed_data,
        degradation_eur_per_mwh_by_unit=degradation,
        strategy=strategy,
        gamma=gamma,
    )
    primal_results = solve_ipopt(
        primal,
        solver_tol=solver_tol,
        max_cpu_time=max_cpu_time,
        tee=tee,
    )
    dual = _build_strategic_tikhonov_dual(
        fixed_data,
        degradation_eur_per_mwh_by_unit=degradation,
        strategy=strategy,
        gamma=gamma,
        price_lower_bound_eur_per_mwh=model._price_lower_bound_eur_per_mwh,
        price_upper_bound_eur_per_mwh=model._price_upper_bound_eur_per_mwh,
        other_dual_bound=model._dual_bound_eur_per_mwh,
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
                abs(primal_prices[str(n), int(t)] - pyo.value(dual.lam[n, t]))
                for n in dual.N
                for t in dual.T
            ),
            "max_abs_original_nodal_balance_residual_mw": max(
                abs(pyo.value(primal.balance_residual[n, t]))
                for n in primal.N
                for t in primal.T
            ),
            "max_abs_h_plus_gamma_lambda_mw": max(
                abs(
                    pyo.value(primal.balance_residual[n, t])
                    + gamma * pyo.value(dual.lam[n, t])
                )
                for n in dual.N
                for t in dual.T
            ),
        }
    )
    return primal, dual, diagnostics


def _copy_component_values(target, source) -> None:
    for index in target:
        item = target[index]
        raw = pyo.value(source[index]) if index in source else 0.0
        if item.lb is not None:
            raw = max(pyo.value(item.lb), raw)
        if item.ub is not None:
            raw = min(pyo.value(item.ub), raw)
        item.set_value(raw)


def initialize_strategic_mpec_from_soft_market(
    model: pyo.ConcreteModel,
    *,
    solver_tol: float,
    max_cpu_time: float,
    tee: bool = False,
) -> dict[str, float | str]:
    """Warm-start all embedded primal/dual variables from the matched market."""

    primal, dual, diagnostics = solve_matched_strategic_soft_market(
        model,
        solver_tol=solver_tol,
        max_cpu_time=max_cpu_time,
        tee=tee,
    )
    if diagnostics.get("primal_termination") != "optimal" or diagnostics.get(
        "dual_termination"
    ) != "optimal":
        return diagnostics

    for component_name in (
        "P_gen",
        "P_charge",
        "P_discharge",
        "SOC",
        "NetInjection",
        "balance_residual",
    ):
        _copy_component_values(
            getattr(model, component_name), getattr(primal, component_name)
        )
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


def audit_strategic_mpec_against_soft_market(
    model: pyo.ConcreteModel,
    *,
    solver_tol: float,
    max_cpu_time: float,
    tee: bool = False,
) -> dict[str, float | str]:
    """Re-clear the exact soft market at the MPEC's fleet and submitted bids."""

    primal, dual, diagnostics = solve_matched_strategic_soft_market(
        model,
        solver_tol=solver_tol,
        max_cpu_time=max_cpu_time,
        tee=tee,
    )
    audit = dict(diagnostics)
    if diagnostics.get("primal_termination") != "optimal" or diagnostics.get(
        "dual_termination"
    ) != "optimal":
        return audit

    primal_prices = soft_balance_prices(primal)
    audit.update(
        {
            "max_abs_mpec_vs_soft_primal_lambda_eur_per_mwh": max(
                abs(pyo.value(model.lam[n, t]) - primal_prices[str(n), int(t)])
                for n in model.N
                for t in model.T
            ),
            "max_abs_mpec_vs_soft_dual_lambda_eur_per_mwh": max(
                abs(pyo.value(model.lam[n, t]) - pyo.value(dual.lam[n, t]))
                for n in model.N
                for t in model.T
            ),
            "max_abs_charge_dispatch_difference_mw": max(
                abs(
                    pyo.value(model.P_charge[i, n, t])
                    - (pyo.value(primal.P_charge[i, n, t]) if (i, n) in primal.IN else 0.0)
                )
                for i, n in model.IN
                for t in model.T
            ),
            "max_abs_discharge_dispatch_difference_mw": max(
                abs(
                    pyo.value(model.P_discharge[i, n, t])
                    - (
                        pyo.value(primal.P_discharge[i, n, t])
                        if (i, n) in primal.IN
                        else 0.0
                    )
                )
                for i, n in model.IN
                for t in model.T
            ),
        }
    )
    return audit


def build_experimental_mpec(data) -> pyo.ConcreteModel:
    return build_strategic_operation_tikhonov_mpec(
        data,
        dual_tikhonov_gamma=DUAL_TIKHONOV_GAMMA,
        quad_demand=QuadraticDemandCurve(alpha=100.0, beta=0.1),
        investor=InvestorConfig(investor_id=INVESTOR_ID, wacc=WACC),
        node_limit_mw=NODE_LIMIT_MW,
        initial_power_mw=INITIAL_POWER_MW_PER_NODE,
        initial_ratio_hours=INITIAL_DURATION_HOURS,
        price_bound_eur_per_mwh=PRICE_BOUND_EUR_PER_MWH,
        dual_bound_eur_per_mwh=OTHER_DUAL_BOUND,
        use_demand_curve=False,
        dispatch_regularization_eur_per_mw2h=(
            DISPATCH_REGULARIZATION_EUR_PER_MW2H
        ),
        strategic_bid_prices=STRATEGIC_BID_PRICES,
        bid_price_bound_eur_per_mwh=BID_PRICE_BOUND_EUR_PER_MWH,
        proximal_penalty_eur_per_mw2_day=0.0,
        strategic_epsilon_penalty=0.0,
        solver_tol=SOLVER_TOL,
        initialize_model=False,
    )


def solve_experimental_mpec(data) -> tuple[pyo.ConcreteModel, dict]:
    model = build_experimental_mpec(data)
    initialization = initialize_strategic_mpec_from_soft_market(
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
    summary = {
        "termination": termination,
        "initialization": initialization,
        "investor_profit_eur_per_day": pyo.value(model.investor_profit_expr),
        "investment_power_mw": sum(pyo.value(model.X_power[n]) for n in model.N),
        "investment_energy_mwh": sum(
            pyo.value(model.X_energy[n]) for n in model.N
        ),
        "strategic_bid_prices": STRATEGIC_BID_PRICES,
        "offer_metrics": offer_metrics(model),
        "strong_duality_diagnostics": strong_duality_diagnostics(model),
    }
    if termination == "optimal":
        summary["same_strategy_soft_market_audit"] = (
            audit_strategic_mpec_against_soft_market(
                model,
                solver_tol=AUDIT_SOLVER_TOL,
                max_cpu_time=MAX_CPU_TIME_SECONDS,
                tee=False,
            )
        )
    return model, summary


def main() -> int:
    data = load_calibrated_case(Path(DATA_PATH))
    _, summary = solve_experimental_mpec(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote strategic exact strong-duality MPEC summary to {output_path}")
    return 0 if summary["termination"] == "optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
