"""IEEE-9 single-investor MPEC with strategic hourly quantity offers.

The investor chooses nodal BESS MW/MWh and the charge/discharge power made
available in every node-hour. The ISO retains control of accepted operation and
clears the full PTDF market with battery SOC dynamics. The convex lower level is
embedded through primal feasibility, dual feasibility/stationarity, and strong
duality.

This is an experimental bridge between the maintained capacity-only MPEC and a
future price-and-quantity bidding model. Offer prices remain truthful here: the
ISO objective uses the physical degradation cost, while strategic behavior
enters only through quantity withholding.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pyomo.environ as pyo

from single_investor_mpec import (
    DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
    DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    DEFAULT_INITIAL_POWER_MW,
    DEFAULT_INITIAL_RATIO_HOURS,
    DEFAULT_NODE_LIMIT_MW,
    DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    DEFAULT_SOLVER_TOL,
    InvestorConfig,
    build_fixed_demand_primal_model,
    build_single_investor_mpec,
    compute_reference_settlement,
    default_quadratic_demand_curve,
    fixed_demand_reference_lambda,
    fixed_storage_data_from_solution,
    initialize_from_reference_dispatch,
    load_market_data,
)
from single_investor_mpec_results import _write_csv, export_solution
from solver_utils import get_ipopt_solver


MODEL_NAME = "IEEE-9 Strategic Storage Quantity-Offer MPEC"
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (
    MODEL_DIR
    / "data"
    / "processed"
    / "market_data_IEEE_9Bus_distributed_congestion.json"
)
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / "strategic_operation_mpec" / "ieee9_single_investor"


def apply_generator_calibration(
    data,
    *,
    conventional_capacity_adder_mw: float = 0.0,
    peaker_node: str | None = None,
    peaker_capacity_mw: float = 0.0,
    peaker_cost_eur_per_mwh: float = 95.0,
):
    """Return a sensitivity copy with optional conventional and peaker capacity.

    A sufficiently large peaker gives scarcity an economic marginal cost. Merely
    enlarging the existing linear generators cannot pin the price when every
    available resource and strategic offer is binding.
    """

    if conventional_capacity_adder_mw < 0.0:
        raise ValueError("Conventional capacity adder must be non-negative.")
    if peaker_capacity_mw < 0.0 or peaker_cost_eur_per_mwh < 0.0:
        raise ValueError("Peaker capacity and marginal cost must be non-negative.")
    if peaker_capacity_mw > 0.0 and peaker_node not in data.nodes:
        raise ValueError(f"Unknown peaker node: {peaker_node}")

    conventional = [g for g in data.generators if str(g).startswith("G_IEEE")]
    generation_capacity = dict(data.generation_capacity)
    for generator in conventional:
        for time in data.times:
            generation_capacity[generator, time] += conventional_capacity_adder_mw

    generators = list(data.generators)
    generators_at_node = {
        node: list(node_generators)
        for node, node_generators in data.generators_at_node.items()
    }
    generation_cost = dict(data.generation_cost)
    peaker_id = None
    if peaker_capacity_mw > 0.0:
        peaker_id = f"G_PEAKER_{peaker_node}"
        if peaker_id in generators:
            raise ValueError(f"Peaker {peaker_id} already exists in the dataset.")
        generators.append(peaker_id)
        generators_at_node[peaker_node].append(peaker_id)
        generation_cost[peaker_id] = peaker_cost_eur_per_mwh
        for time in data.times:
            generation_capacity[peaker_id, time] = peaker_capacity_mw

    calibrated = replace(
        data,
        generators=generators,
        generators_at_node=generators_at_node,
        generation_cost=generation_cost,
        generation_capacity=generation_capacity,
    )
    config = {
        "conventional_generators": conventional,
        "conventional_capacity_adder_mw_each": conventional_capacity_adder_mw,
        "peaker_id": peaker_id,
        "peaker_node": peaker_node if peaker_id is not None else None,
        "peaker_capacity_mw": peaker_capacity_mw if peaker_id is not None else 0.0,
        "peaker_cost_eur_per_mwh": peaker_cost_eur_per_mwh if peaker_id is not None else None,
    }
    return calibrated, config


def add_strategic_quantity_offers(
    model: pyo.ConcreteModel,
    *,
    rival_offer_charge_mw_by_unit: Mapping[str, Mapping[tuple[str, int], float]] | None = None,
    rival_offer_discharge_mw_by_unit: Mapping[str, Mapping[tuple[str, int], float]] | None = None,
) -> None:
    """Replace installed-power dispatch bounds with strategic hourly offers.

    The original model's lower-level dual objective contains
    ``X_power * (rho_charge + sigma_discharge)`` because installed power is the
    RHS of both dispatch bounds. Once the investor can withhold quantity, those
    strong-duality terms must use the corresponding hourly offers instead.
    """

    investor = model._investor_id
    rival_charge_input = rival_offer_charge_mw_by_unit or {}
    rival_discharge_input = rival_offer_discharge_mw_by_unit or {}
    rival_ids = set(model._rival_ids)
    if set(rival_charge_input) - rival_ids or set(rival_discharge_input) - rival_ids:
        raise ValueError("Strategic offer mappings contain an unknown rival investor.")

    fixed_rival_charge: dict[tuple[str, str, int], float] = {}
    fixed_rival_discharge: dict[tuple[str, str, int], float] = {}
    for unit, node in model.IN:
        if unit == investor:
            continue
        capacity = model._rival_power_mw_by_unit[unit][node]
        for time in model.T:
            charge = float(rival_charge_input.get(unit, {}).get((node, int(time)), capacity))
            discharge = float(rival_discharge_input.get(unit, {}).get((node, int(time)), capacity))
            if not -1e-9 <= charge <= capacity + 1e-7:
                raise ValueError(f"Rival charge offer exceeds capacity for {unit}, {node}, {time}.")
            if not -1e-9 <= discharge <= capacity + 1e-7:
                raise ValueError(f"Rival discharge offer exceeds capacity for {unit}, {node}, {time}.")
            fixed_rival_charge[unit, node, int(time)] = max(0.0, min(charge, capacity))
            fixed_rival_discharge[unit, node, int(time)] = max(0.0, min(discharge, capacity))

    original_power_bound_dual_term = sum(
        (
            model.X_power[n]
            if unit == investor
            else model._rival_power_mw_by_unit[unit][n]
        )
        * (model.rho_ch[unit, n, t] + model.sig_dis[unit, n, t])
        for unit, n in model.IN
        for t in model.T
    )

    model.Q_offer_charge = pyo.Var(
        model.N,
        model.T,
        domain=pyo.NonNegativeReals,
        initialize=lambda m, n, t: pyo.value(m.X_power[n]),
    )
    model.Q_offer_discharge = pyo.Var(
        model.N,
        model.T,
        domain=pyo.NonNegativeReals,
        initialize=lambda m, n, t: pyo.value(m.X_power[n]),
    )

    # Upper-level feasibility: an investor cannot offer more than it installs.
    model.offer_charge_capacity = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.Q_offer_charge[n, t] <= m.X_power[n],
    )
    model.offer_discharge_capacity = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: m.Q_offer_discharge[n, t] <= m.X_power[n],
    )

    # Lower-level feasibility: the ISO selects accepted dispatch inside the
    # submitted offer. All network and SOC constraints remain unchanged.
    model.del_component(model.charge_power_bound)
    model.del_component(model.discharge_power_bound)
    model.charge_power_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, unit, n, t: m.P_charge[unit, n, t]
        <= (
            m.Q_offer_charge[n, t]
            if unit == investor
            else fixed_rival_charge[unit, n, int(t)]
        ),
    )
    model.discharge_power_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, unit, n, t: m.P_discharge[unit, n, t]
        <= (
            m.Q_offer_discharge[n, t]
            if unit == investor
            else fixed_rival_discharge[unit, n, int(t)]
        ),
    )

    strategic_offer_dual_term = sum(
        (
            model.Q_offer_charge[n, t]
            if unit == investor
            else fixed_rival_charge[unit, n, int(t)]
        )
        * model.rho_ch[unit, n, t]
        + (
            model.Q_offer_discharge[n, t]
            if unit == investor
            else fixed_rival_discharge[unit, n, int(t)]
        )
        * model.sig_dis[unit, n, t]
        for unit, n in model.IN
        for t in model.T
    )
    model.dual_objective_expr.set_value(
        model.dual_objective_expr.expr
        - original_power_bound_dual_term
        + strategic_offer_dual_term
    )
    model._strategic_quantity_offers = True
    model._rival_offer_charge_mw_by_unit = fixed_rival_charge
    model._rival_offer_discharge_mw_by_unit = fixed_rival_discharge


def build_ieee9_strategic_operation_mpec(
    data,
    *,
    initial_power_mw: float = DEFAULT_INITIAL_POWER_MW,
    initial_ratio_hours: float = DEFAULT_INITIAL_RATIO_HOURS,
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW,
    investor: InvestorConfig | None = None,
    rival_power_mw_by_unit: Mapping[str, Mapping[str, float]] | None = None,
    rival_energy_mwh_by_unit: Mapping[str, Mapping[str, float]] | None = None,
    rival_offer_charge_mw_by_unit: Mapping[str, Mapping[tuple[str, int], float]] | None = None,
    rival_offer_discharge_mw_by_unit: Mapping[str, Mapping[tuple[str, int], float]] | None = None,
    rival_degradation_eur_per_mwh_by_unit: Mapping[str, float] | None = None,
    price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    dual_bound_eur_per_mwh: float = DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    dispatch_regularization_eur_per_mw2h: float = DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
    system_price_settlement: bool = False,
    solver_tol: float = DEFAULT_SOLVER_TOL,
    initialize_model: bool = True,
) -> pyo.ConcreteModel:
    """Build the strategic-operation MPEC on a supplied market dataset."""

    model = build_single_investor_mpec(
        data,
        initial_power_mw=initial_power_mw,
        initial_ratio_hours=initial_ratio_hours,
        node_limit_mw=node_limit_mw,
        price_bound_eur_per_mwh=price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=dual_bound_eur_per_mwh,
        quad_demand=default_quadratic_demand_curve(),
        use_demand_curve=False,
        investor=investor or InvestorConfig(),
        rival_power_mw_by_unit=rival_power_mw_by_unit,
        rival_energy_mwh_by_unit=rival_energy_mwh_by_unit,
        rival_degradation_eur_per_mwh_by_unit=rival_degradation_eur_per_mwh_by_unit,
        dispatch_regularization_eur_per_mw2h=dispatch_regularization_eur_per_mw2h,
        system_price_settlement=system_price_settlement,
        solver_tol=solver_tol,
    )
    model.name = MODEL_NAME
    add_strategic_quantity_offers(
        model,
        rival_offer_charge_mw_by_unit=rival_offer_charge_mw_by_unit,
        rival_offer_discharge_mw_by_unit=rival_offer_discharge_mw_by_unit,
    )
    if initialize_model:
        initialize_from_reference_dispatch(model, data, initial_ratio_hours)

        # The reference initialization uses the full installed power, which is
        # also the starting offer. Keep the relationship explicit afterwards.
        for n in model.N:
            for t in model.T:
                model.Q_offer_charge[n, t].set_value(pyo.value(model.X_power[n]))
                model.Q_offer_discharge[n, t].set_value(pyo.value(model.X_power[n]))
    model._initial_power_mw = initial_power_mw
    model._initial_ratio_hours = initial_ratio_hours
    return model


def solve_offer_reclear(model: pyo.ConcreteModel) -> dict[str, object]:
    """Independently re-clear the ISO problem with investment and offers fixed."""

    fixed_data = fixed_storage_data_from_solution(model)
    reference = build_fixed_demand_primal_model(
        fixed_data,
        storage_degradation_eur_per_mwh=model._storage_degradation_eur_per_mwh,
        dispatch_regularization_eur_per_mw2h=model._dispatch_regularization_eur_per_mw2h,
    )
    investor = model._investor_id
    reference.offer_charge_bound = pyo.Constraint(
        reference.N,
        reference.T,
        rule=lambda m, n, t: m.P_charge[investor, n, t]
        <= pyo.value(model.Q_offer_charge[n, t]),
    )
    reference.offer_discharge_bound = pyo.Constraint(
        reference.N,
        reference.T,
        rule=lambda m, n, t: m.P_discharge[investor, n, t]
        <= pyo.value(model.Q_offer_discharge[n, t]),
    )
    results = get_ipopt_solver(
        {"tol": model._solver_tol, "acceptable_tol": model._solver_tol}
    ).solve(reference, tee=False)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            "Offer-constrained ISO re-clear failed: "
            f"{results.solver.termination_condition}"
        )

    prices = fixed_demand_reference_lambda(reference)
    reference_objective = pyo.value(reference.objective)
    embedded_objective = pyo.value(model.primal_objective_expr)
    max_dispatch_difference = max(
        max(
            abs(
                pyo.value(model.P_charge[investor, n, t])
                - pyo.value(reference.P_charge[investor, n, t])
            ),
            abs(
                pyo.value(model.P_discharge[investor, n, t])
                - pyo.value(reference.P_discharge[investor, n, t])
            ),
        )
        for n in model.N
        for t in model.T
    )
    max_price_difference = max(
        abs(pyo.value(model.lam[n, t]) - prices[n, t])
        for n in model.N
        for t in model.T
    )
    revenue = sum(
        prices[n, t]
        * (
            pyo.value(reference.P_discharge[investor, n, t])
            - pyo.value(reference.P_charge[investor, n, t])
        )
        for n in reference.N
        for t in reference.T
    )
    degradation = 0.5 * model._degradation_eur_per_mwh * sum(
        pyo.value(reference.P_charge[investor, n, t])
        + pyo.value(reference.P_discharge[investor, n, t])
        for n in reference.N
        for t in reference.T
    )
    profit = revenue - degradation - pyo.value(model.capex_daily_expr)
    return {
        "reference_model": reference,
        "reference_lambda": prices,
        "solver_status": str(results.solver.status),
        "termination": str(results.solver.termination_condition),
        "lower_level_objective_eur_per_day": reference_objective,
        "embedded_primal_objective_eur_per_day": embedded_objective,
        "objective_difference_eur_per_day": embedded_objective - reference_objective,
        "max_dispatch_difference_mw": max_dispatch_difference,
        "max_price_difference_eur_per_mwh": max_price_difference,
        "spot_revenue_eur_per_day": revenue,
        "degradation_cost_eur_per_day": degradation,
        "profit_eur_per_day": profit,
    }


def offer_metrics(model: pyo.ConcreteModel) -> dict[str, float | int]:
    investor = model._investor_id
    total_power = sum(pyo.value(model.X_power[n]) for n in model.N)
    denominator = len(list(model.T)) * total_power
    charge_offer = sum(
        pyo.value(model.Q_offer_charge[n, t]) for n in model.N for t in model.T
    )
    discharge_offer = sum(
        pyo.value(model.Q_offer_discharge[n, t]) for n in model.N for t in model.T
    )
    charge_accepted = sum(
        pyo.value(model.P_charge[investor, n, t]) for n in model.N for t in model.T
    )
    discharge_accepted = sum(
        pyo.value(model.P_discharge[investor, n, t]) for n in model.N for t in model.T
    )
    return {
        "charge_offer_capacity_hours_mwh": charge_offer,
        "discharge_offer_capacity_hours_mwh": discharge_offer,
        "charge_offer_fraction_of_installed_capacity_hours": (
            charge_offer / denominator if denominator > 0.0 else 0.0
        ),
        "discharge_offer_fraction_of_installed_capacity_hours": (
            discharge_offer / denominator if denominator > 0.0 else 0.0
        ),
        "accepted_charge_mwh": charge_accepted,
        "accepted_discharge_mwh": discharge_accepted,
        "binding_charge_offer_node_hours": sum(
            pyo.value(model.X_power[n]) > 1e-4
            and abs(
                pyo.value(model.Q_offer_charge[n, t])
                - pyo.value(model.P_charge[investor, n, t])
            )
            <= 1e-4
            for n in model.N
            for t in model.T
        ),
        "binding_discharge_offer_node_hours": sum(
            pyo.value(model.X_power[n]) > 1e-4
            and abs(
                pyo.value(model.Q_offer_discharge[n, t])
                - pyo.value(model.P_discharge[investor, n, t])
            )
            <= 1e-4
            for n in model.N
            for t in model.T
        ),
    }


def export_strategic_solution(
    model: pyo.ConcreteModel,
    output_dir: Path,
    solver_status: str,
    termination: str,
    offer_reclear: dict[str, object],
    full_availability: dict[str, object],
) -> None:
    """Export standard MPEC results plus strategic offers and diagnostics."""

    export_solution(
        model,
        output_dir,
        solver_status,
        termination,
        full_availability,
    )
    investor = model._investor_id
    _write_csv(
        output_dir / "strategic_quantity_offers.csv",
        [
            "hour",
            "node",
            "installed_power_mw",
            "charge_offer_mw",
            "accepted_charge_mw",
            "discharge_offer_mw",
            "accepted_discharge_mw",
            "embedded_lambda_eur_per_mwh",
        ],
        [
            {
                "hour": t,
                "node": n,
                "installed_power_mw": pyo.value(model.X_power[n]),
                "charge_offer_mw": pyo.value(model.Q_offer_charge[n, t]),
                "accepted_charge_mw": pyo.value(model.P_charge[investor, n, t]),
                "discharge_offer_mw": pyo.value(model.Q_offer_discharge[n, t]),
                "accepted_discharge_mw": pyo.value(model.P_discharge[investor, n, t]),
                "embedded_lambda_eur_per_mwh": pyo.value(model.lam[n, t]),
            }
            for t in model.T
            for n in model.N
        ],
    )
    _write_csv(
        output_dir / "offer_reclear_prices.csv",
        ["hour", "node", "lambda_offer_reclear_eur_per_mwh"],
        [
            {
                "hour": t,
                "node": n,
                "lambda_offer_reclear_eur_per_mwh": offer_reclear["reference_lambda"][n, t],
            }
            for t in model.T
            for n in model.N
        ],
    )
    full_availability_prices = full_availability["reference_lambda"]
    _write_csv(
        output_dir / "lmp_withholding_comparison.csv",
        [
            "hour",
            "node",
            "lambda_strategic_offer_eur_per_mwh",
            "lambda_full_availability_eur_per_mwh",
            "strategic_minus_full_availability_eur_per_mwh",
        ],
        [
            {
                "hour": t,
                "node": n,
                "lambda_strategic_offer_eur_per_mwh": pyo.value(model.lam[n, t]),
                "lambda_full_availability_eur_per_mwh": full_availability_prices[n, t],
                "strategic_minus_full_availability_eur_per_mwh": (
                    pyo.value(model.lam[n, t]) - full_availability_prices[n, t]
                ),
            }
            for t in model.T
            for n in model.N
        ],
    )
    plot_lmp_withholding_comparison(model, full_availability_prices, output_dir)

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": "strategic_hourly_quantity_offers_with_iso_dispatch",
            "data_path": getattr(model, "_data_path", None),
            "initial_power_mw_per_node": model._initial_power_mw,
            "initial_ratio_hours": model._initial_ratio_hours,
            "generator_calibration": getattr(model, "_generator_calibration", None),
            "offer_metrics": offer_metrics(model),
            "offer_reclear_has_material_primal_or_dual_nonuniqueness": (
                offer_reclear["max_dispatch_difference_mw"] > 1e-3
                or offer_reclear["max_price_difference_eur_per_mwh"] > 1.0
            ),
            "offer_reclear": {
                key: value
                for key, value in offer_reclear.items()
                if key not in {"reference_model", "reference_lambda"}
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_lmp_withholding_comparison(
    model: pyo.ConcreteModel,
    full_availability_prices,
    output_dir: Path,
) -> None:
    """Plot hourly LMP evolution with strategic offers and full availability."""

    import matplotlib.pyplot as plt

    hours = list(model.T)
    strategic = {
        node: [pyo.value(model.lam[node, time]) for time in hours]
        for node in model.N
    }
    full = {
        node: [full_availability_prices[node, time] for time in hours]
        for node in model.N
    }
    strategic_min = [min(strategic[node][k] for node in model.N) for k in range(len(hours))]
    strategic_max = [max(strategic[node][k] for node in model.N) for k in range(len(hours))]
    full_min = [min(full[node][k] for node in model.N) for k in range(len(hours))]
    full_max = [max(full[node][k] for node in model.N) for k in range(len(hours))]

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True)
    axes[0].fill_between(hours, strategic_min, strategic_max, color="#d95f02", alpha=0.22)
    axes[0].plot(hours, strategic_max, color="#d95f02", linewidth=1.8, label="Strategic nodal maximum")
    axes[0].plot(hours, strategic_min, color="#d95f02", linewidth=1.0, linestyle=":", label="Strategic nodal minimum")
    axes[0].fill_between(hours, full_min, full_max, color="#1f77b4", alpha=0.18)
    axes[0].plot(hours, full_max, color="#1f77b4", linewidth=1.8, label="Full-availability nodal maximum")
    axes[0].plot(hours, full_min, color="#1f77b4", linewidth=1.0, linestyle=":", label="Full-availability nodal minimum")
    axes[0].set_ylabel("LMP [EUR/MWh]")
    axes[0].set_title("System-wide nodal LMP envelope")
    axes[0].legend(ncol=2, fontsize=8)

    for node, color in (("N5", "#b2182b"), ("N8", "#2166ac")):
        axes[1].plot(hours, strategic[node], color=color, linewidth=1.8, label=f"{node} strategic")
        axes[1].plot(hours, full[node], color=color, linewidth=1.5, linestyle="--", label=f"{node} full availability")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("LMP [EUR/MWh]")
    axes[1].set_title("Congested load node N5 and storage node N8")
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].set_xticks(hours)

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(min(hours), max(hours))
    fig.tight_layout()
    fig.savefig(output_dir / "lmp_with_vs_without_strategic_withholding.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "lmp_with_vs_without_strategic_withholding.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--initial-power-mw", type=float, default=DEFAULT_INITIAL_POWER_MW)
    parser.add_argument("--initial-ratio-hours", type=float, default=4.0)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--conventional-capacity-adder-mw", type=float, default=0.0)
    parser.add_argument("--peaker-node", choices=[f"N{k}" for k in range(1, 10)], default=None)
    parser.add_argument("--peaker-capacity-mw", type=float, default=0.0)
    parser.add_argument("--peaker-cost-eur-per-mwh", type=float, default=95.0)
    parser.add_argument("--solver-tol", type=float, default=DEFAULT_SOLVER_TOL)
    parser.add_argument(
        "--dispatch-regularization",
        type=float,
        default=DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
        help="Optional neutral lower-level quadratic tie-break in EUR/(MW^2 h).",
    )
    parser.add_argument("--max-cpu-time", type=float, default=180.0)
    parser.add_argument("--price-bound-eur-per-mwh", type=float, default=DEFAULT_PRICE_BOUND_EUR_PER_MWH)
    parser.add_argument("--dual-bound-eur-per-mwh", type=float, default=DEFAULT_DUAL_BOUND_EUR_PER_MWH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dispatch_regularization < 0.0:
        raise SystemExit("--dispatch-regularization must be non-negative.")
    if args.solver_tol <= 0.0:
        raise SystemExit("--solver-tol must be positive.")
    base_data = load_market_data(args.data)
    data, generator_calibration = apply_generator_calibration(
        base_data,
        conventional_capacity_adder_mw=args.conventional_capacity_adder_mw,
        peaker_node=args.peaker_node,
        peaker_capacity_mw=args.peaker_capacity_mw,
        peaker_cost_eur_per_mwh=args.peaker_cost_eur_per_mwh,
    )
    model = build_ieee9_strategic_operation_mpec(
        data,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        node_limit_mw=args.node_limit_mw,
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
        dispatch_regularization_eur_per_mw2h=args.dispatch_regularization,
        solver_tol=args.solver_tol,
    )
    model._data_path = str(args.data)
    model._generator_calibration = generator_calibration
    solver = get_ipopt_solver(
        {
            "max_cpu_time": args.max_cpu_time,
            "tol": args.solver_tol,
            "acceptable_tol": args.solver_tol,
        }
    )
    results = solver.solve(model, tee=args.tee)
    termination = results.solver.termination_condition
    print(f"Solver status: {results.solver.status}")
    print(f"Termination: {termination}")
    if termination != pyo.TerminationCondition.optimal:
        return 1

    offer_reclear = solve_offer_reclear(model)
    full_availability = compute_reference_settlement(model)
    strong_duality_gap = abs(
        pyo.value(model.primal_objective_expr) - pyo.value(model.dual_objective_expr)
    )

    print("\nStrategic-operation MPEC")
    print(f"  profit: {pyo.value(model.investor_profit_expr):,.3f} EUR/day")
    print(
        "  investment: "
        f"{sum(pyo.value(model.X_power[n]) for n in model.N):,.3f} MW / "
        f"{sum(pyo.value(model.X_energy[n]) for n in model.N):,.3f} MWh"
    )
    for n in model.N:
        if pyo.value(model.X_power[n]) > 1e-3:
            print(
                f"  {n}: {pyo.value(model.X_power[n]):,.3f} MW / "
                f"{pyo.value(model.X_energy[n]):,.3f} MWh"
            )
    print(f"  strong-duality gap: {strong_duality_gap:.3e}")
    print(
        "  offer re-clear lower-level objective difference: "
        f"{offer_reclear['objective_difference_eur_per_day']:.3e} EUR/day"
    )
    if (
        offer_reclear["max_dispatch_difference_mw"] > 1e-3
        or offer_reclear["max_price_difference_eur_per_mwh"] > 1.0
    ):
        print(
            "  nonunique re-clear diagnostic (dispatch/price): "
            f"{offer_reclear['max_dispatch_difference_mw']:.3f} MW / "
            f"{offer_reclear['max_price_difference_eur_per_mwh']:.3f} EUR/MWh"
        )
    metrics = offer_metrics(model)
    print(
        "  offered charge/discharge capacity-hours: "
        f"{metrics['charge_offer_fraction_of_installed_capacity_hours']:.1%} / "
        f"{metrics['discharge_offer_fraction_of_installed_capacity_hours']:.1%}"
    )
    print(
        "  full-availability counterfactual profit at reference prices: "
        f"{full_availability['profit_at_reference_prices_eur_per_day']:,.3f} EUR/day"
    )
    print("  NOTE: this is a local nonconvex NLP solution; multi-start comparison is required.")

    if not args.no_export:
        export_strategic_solution(
            model,
            args.output_dir,
            str(results.solver.status),
            str(termination),
            offer_reclear,
            full_availability,
        )
        print(f"\nWrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
