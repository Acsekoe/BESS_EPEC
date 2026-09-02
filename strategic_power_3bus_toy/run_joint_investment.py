"""Run the one-stage continuous investment-and-availability toy game.

Each investor chooses installed power, installed energy, and hourly availability
in the same best response. This is deliberately not the sequential two-stage
investment/operation game described in ``summary.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pyomo.environ as pyo

from model import (
    BestResponse,
    default_data,
    full_availability,
    installed_capacities,
    market_profit,
    solve_best_response,
    solve_market,
)


ROOT = Path(__file__).resolve().parent


def solve_response(args, data, investor_id, power, energy, charge, discharge) -> BestResponse:
    return solve_best_response(
        data,
        investor_id,
        charge,
        discharge,
        power_capacity_mw=power,
        energy_capacity_mwh=energy,
        endogenous_investment=True,
        complementarity_epsilon=args.complementarity_epsilon,
        availability_tie_breaker_eur_per_mw=args.availability_tie_breaker,
        strategic_charge=args.two_sided,
        max_solve_seconds=args.max_solve_seconds,
        tee=args.tee,
    )


def deviations(response, power, energy, charge, discharge) -> dict[str, float]:
    investor_id = response.investor_id
    return {
        "power_mw": abs(float(response.power_mw) - power[investor_id]),
        "energy_mwh": abs(float(response.energy_mwh) - energy[investor_id]),
        "offer_mw": max(
            max(
                abs(response.charge_offer_mw[t] - charge[investor_id, t]),
                abs(response.discharge_offer_mw[t] - discharge[investor_id, t]),
            )
            for t in response.charge_offer_mw
        ),
    }


def apply_damped_response(args, data, response, power, energy, charge, discharge) -> None:
    investor_id = response.investor_id
    investor = next(item for item in data.investors if item.investor_id == investor_id)
    weight = args.damping
    power[investor_id] = (
        (1.0 - weight) * power[investor_id] + weight * float(response.power_mw)
    )
    energy[investor_id] = (
        (1.0 - weight) * energy[investor_id] + weight * float(response.energy_mwh)
    )
    energy[investor_id] = min(
        investor.duration_max_hours * power[investor_id],
        max(investor.duration_min_hours * power[investor_id], energy[investor_id]),
    )
    for time in data.times:
        charge[investor_id, time] = min(
            power[investor_id],
            max(
                0.0,
                (1.0 - weight) * charge[investor_id, time]
                + weight * response.charge_offer_mw[time],
            ),
        )
        discharge[investor_id, time] = min(
            power[investor_id],
            max(
                0.0,
                (1.0 - weight) * discharge[investor_id, time]
                + weight * response.discharge_offer_mw[time],
            ),
        )


def write_outputs(output_dir, data, power, energy, charge, discharge, market, history, audits, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    with (output_dir / "final_investments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["investor", "power_mw", "energy_mwh", "duration_hours"],
        )
        writer.writeheader()
        for investor in data.investors:
            investor_id = investor.investor_id
            writer.writerow(
                {
                    "investor": investor_id,
                    "power_mw": power[investor_id],
                    "energy_mwh": energy[investor_id],
                    "duration_hours": (
                        energy[investor_id] / power[investor_id]
                        if power[investor_id] > 1.0e-9
                        else None
                    ),
                }
            )

    with (output_dir / "final_strategies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "investor",
                "time",
                "power_mw",
                "charge_offer_mw",
                "discharge_offer_mw",
                "discharge_withheld_mw",
            ],
        )
        writer.writeheader()
        for investor in data.investors:
            investor_id = investor.investor_id
            for time in data.times:
                writer.writerow(
                    {
                        "investor": investor_id,
                        "time": time,
                        "power_mw": power[investor_id],
                        "charge_offer_mw": charge[investor_id, time],
                        "discharge_offer_mw": discharge[investor_id, time],
                        "discharge_withheld_mw": max(
                            0.0, power[investor_id] - discharge[investor_id, time]
                        ),
                    }
                )

    with (output_dir / "final_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "investor",
            "optimal",
            "termination",
            "best_response_power_mw",
            "best_response_energy_mwh",
            "best_response_duration_hours",
            "power_deviation_mw",
            "energy_deviation_mwh",
            "offer_deviation_mw",
            "recleared_profit_eur_per_day",
            "profitable_deviation_eur_per_day",
            "maximum_lmp_reclear_gap_eur_per_mwh",
            "embedded_reclear_profit_gap_eur_per_day",
            "maximum_complementarity_violation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for response, residual, profit_gain in audits:
            writer.writerow(
                {
                    "investor": response.investor_id,
                    "optimal": response.optimal,
                    "termination": response.termination,
                    "best_response_power_mw": response.power_mw,
                    "best_response_energy_mwh": response.energy_mwh,
                    "best_response_duration_hours": (
                        response.energy_mwh / response.power_mw
                        if response.power_mw is not None and response.power_mw > 1.0e-9
                        else None
                    ),
                    "power_deviation_mw": residual["power_mw"],
                    "energy_deviation_mwh": residual["energy_mwh"],
                    "offer_deviation_mw": residual["offer_mw"],
                    "recleared_profit_eur_per_day": response.recleared_profit_eur_per_day,
                    "profitable_deviation_eur_per_day": profit_gain,
                    "maximum_lmp_reclear_gap_eur_per_mwh": response.maximum_lmp_reclear_gap_eur_per_mwh,
                    "embedded_reclear_profit_gap_eur_per_day": response.embedded_reclear_profit_gap_eur_per_day,
                    "maximum_complementarity_violation": response.maximum_complementarity_violation,
                }
            )

    generator_by_node = {
        node: [g.generator_id for g in data.generators if g.node == node] for node in data.nodes
    }
    investor_by_node = {
        node: [i.investor_id for i in data.investors if i.node == node] for node in data.nodes
    }
    with (output_dir / "final_market.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "time",
            "node",
            "demand_mw",
            "lmp_eur_per_mwh",
            "generation_mw",
            "charge_mw",
            "discharge_mw",
            "net_injection_mw",
            "demand_adjustment_mw",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for time in data.times:
            for node in data.nodes:
                writer.writerow(
                    {
                        "time": time,
                        "node": node,
                        "demand_mw": data.demand_mw[node, time],
                        "lmp_eur_per_mwh": float(market.dual[market.nodal_balance[node, time]]),
                        "generation_mw": sum(
                            float(pyo.value(market.PGen[g, time]))
                            for g in generator_by_node[node]
                        ),
                        "charge_mw": sum(
                            float(pyo.value(market.PCharge[i, time]))
                            for i in investor_by_node[node]
                        ),
                        "discharge_mw": sum(
                            float(pyo.value(market.PDischarge[i, time]))
                            for i in investor_by_node[node]
                        ),
                        "net_injection_mw": float(pyo.value(market.NetInjection[node, time])),
                        "demand_adjustment_mw": float(
                            pyo.value(market.DemandAdjustment[node, time])
                        ),
                    }
                )


def run(args) -> int:
    data = default_data()
    power, energy = installed_capacities(data)
    charge, discharge = full_availability(data)
    investor_ids = [investor.investor_id for investor in data.investors]
    history = []
    stable_sweeps = 0
    converged = False
    final_audits = []

    for sweep in range(1, args.max_sweeps + 1):
        shift = (sweep - 1) % len(investor_ids) if args.rotate_order else 0
        order = investor_ids[shift:] + investor_ids[:shift]
        update_optimal = True
        maximum_update = {"power_mw": 0.0, "energy_mwh": 0.0, "offer_mw": 0.0}

        for investor_id in order:
            response = solve_response(
                args, data, investor_id, power, energy, charge, discharge
            )
            update_optimal = update_optimal and response.optimal
            residual = deviations(response, power, energy, charge, discharge)
            for key in maximum_update:
                maximum_update[key] = max(maximum_update[key], residual[key])
            if response.optimal:
                apply_damped_response(
                    args, data, response, power, energy, charge, discharge
                )

        market = solve_market(
            data,
            charge,
            discharge,
            power_capacity_mw=power,
            energy_capacity_mwh=energy,
        )
        final_audits = []
        audit_optimal = True
        maximum_nash = {"power_mw": 0.0, "energy_mwh": 0.0, "offer_mw": 0.0}
        maximum_profit_gain = 0.0
        maximum_lmp_gap = 0.0
        maximum_profit_gap = 0.0
        maximum_comp_violation = 0.0
        for investor_id in investor_ids:
            response = solve_response(
                args, data, investor_id, power, energy, charge, discharge
            )
            residual = deviations(response, power, energy, charge, discharge)
            profit_gain = max(
                0.0,
                response.recleared_profit_eur_per_day - market_profit(market, investor_id),
            )
            final_audits.append((response, residual, profit_gain))
            audit_optimal = audit_optimal and response.optimal
            for key in maximum_nash:
                maximum_nash[key] = max(maximum_nash[key], residual[key])
            maximum_profit_gain = max(maximum_profit_gain, profit_gain)
            maximum_lmp_gap = max(
                maximum_lmp_gap, response.maximum_lmp_reclear_gap_eur_per_mwh
            )
            maximum_profit_gap = max(
                maximum_profit_gap, response.embedded_reclear_profit_gap_eur_per_day
            )
            maximum_comp_violation = max(
                maximum_comp_violation, response.maximum_complementarity_violation
            )

        stable = (
            audit_optimal
            and maximum_nash["power_mw"] <= args.power_tolerance_mw
            and maximum_nash["energy_mwh"] <= args.energy_tolerance_mwh
            and maximum_nash["offer_mw"] <= args.offer_tolerance_mw
            and maximum_lmp_gap <= args.price_audit_tolerance
            and maximum_profit_gap <= args.profit_audit_tolerance
            and maximum_profit_gain <= args.nash_profit_tolerance
        )
        stable_sweeps = stable_sweeps + 1 if stable else 0
        row = {
            "sweep": sweep,
            "order": "->".join(order),
            "all_updates_optimal": update_optimal,
            "all_audits_optimal": audit_optimal,
            "max_update_power_mw": maximum_update["power_mw"],
            "max_update_energy_mwh": maximum_update["energy_mwh"],
            "max_update_offer_mw": maximum_update["offer_mw"],
            "max_nash_power_mw": maximum_nash["power_mw"],
            "max_nash_energy_mwh": maximum_nash["energy_mwh"],
            "max_nash_offer_mw": maximum_nash["offer_mw"],
            "max_profitable_deviation_eur_per_day": maximum_profit_gain,
            "max_lmp_reclear_gap_eur_per_mwh": maximum_lmp_gap,
            "max_embedded_reclear_profit_gap_eur_per_day": maximum_profit_gap,
            "max_complementarity_violation": maximum_comp_violation,
            "stable_sweeps": stable_sweeps,
        }
        for investor_id in investor_ids:
            row[f"power_{investor_id}_mw"] = power[investor_id]
            row[f"energy_{investor_id}_mwh"] = energy[investor_id]
            row[f"profit_{investor_id}_eur_per_day"] = market_profit(market, investor_id)
        history.append(row)
        print(
            f"sweep={sweep:02d} order={row['order']} "
            f"K={maximum_nash['power_mw']:.4f} MW "
            f"E={maximum_nash['energy_mwh']:.4f} MWh "
            f"offer={maximum_nash['offer_mw']:.4f} MW "
            f"profit_gain={maximum_profit_gain:.4f} EUR/day "
            f"price_audit={maximum_lmp_gap:.4f} EUR/MWh"
        )
        if stable_sweeps >= args.consecutive_sweeps:
            converged = True
            break

    final_market = solve_market(
        data,
        charge,
        discharge,
        power_capacity_mw=power,
        energy_capacity_mwh=energy,
    )
    maximum_withholding = {
        investor_id: max(
            max(0.0, power[investor_id] - discharge[investor_id, time])
            for time in data.times
        )
        for investor_id in investor_ids
    }
    summary = {
        "formulation": "one-stage simultaneous continuous investment and availability game",
        "sequential_game": False,
        "strategic_charge_availability": args.two_sided,
        "strategic_discharge_availability": True,
        "duration_bounds_hours": [2.0, 8.0],
        "individual_power_upper_bound_mw": 30.0,
        "sweeps": len(history),
        "converged": converged,
        "stable_sweeps": stable_sweeps,
        "final_investments": {
            investor_id: {
                "power_mw": power[investor_id],
                "energy_mwh": energy[investor_id],
                "duration_hours": (
                    energy[investor_id] / power[investor_id]
                    if power[investor_id] > 1.0e-9
                    else None
                ),
            }
            for investor_id in investor_ids
        },
        "maximum_discharge_withholding_mw": maximum_withholding,
        "final_profits_eur_per_day": {
            investor_id: market_profit(final_market, investor_id)
            for investor_id in investor_ids
        },
        "final_audit": {
            "max_nash_power_mw": history[-1]["max_nash_power_mw"],
            "max_nash_energy_mwh": history[-1]["max_nash_energy_mwh"],
            "max_nash_offer_mw": history[-1]["max_nash_offer_mw"],
            "max_profitable_deviation_eur_per_day": history[-1][
                "max_profitable_deviation_eur_per_day"
            ],
            "max_lmp_reclear_gap_eur_per_mwh": history[-1][
                "max_lmp_reclear_gap_eur_per_mwh"
            ],
            "max_embedded_reclear_profit_gap_eur_per_day": history[-1][
                "max_embedded_reclear_profit_gap_eur_per_day"
            ],
        },
    }
    if not args.no_output:
        write_outputs(
            args.output_dir,
            data,
            power,
            energy,
            charge,
            discharge,
            final_market,
            history,
            final_audits,
            summary,
        )
    print(json.dumps(summary, indent=2))
    return 0 if converged else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sweeps", type=int, default=30)
    parser.add_argument("--damping", type=float, default=0.35)
    parser.add_argument("--power-tolerance-mw", type=float, default=0.05)
    parser.add_argument("--energy-tolerance-mwh", type=float, default=0.10)
    parser.add_argument("--offer-tolerance-mw", type=float, default=0.05)
    parser.add_argument("--consecutive-sweeps", type=int, default=2)
    parser.add_argument("--complementarity-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--availability-tie-breaker", type=float, default=1.0e-3)
    parser.add_argument("--price-audit-tolerance", type=float, default=0.02)
    parser.add_argument("--profit-audit-tolerance", type=float, default=0.25)
    parser.add_argument("--nash-profit-tolerance", type=float, default=0.50)
    parser.add_argument("--two-sided", action="store_true")
    parser.add_argument("--max-solve-seconds", type=float, default=60.0)
    parser.add_argument("--fixed-order", dest="rotate_order", action="store_false")
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "output" / "joint_investment"
    )
    parser.set_defaults(rotate_order=True)
    args = parser.parse_args()
    if args.max_sweeps <= 0 or not 0.0 < args.damping <= 1.0:
        parser.error("max-sweeps must be positive and damping must lie in (0, 1].")
    tolerances = (
        args.power_tolerance_mw,
        args.energy_tolerance_mwh,
        args.offer_tolerance_mw,
        args.price_audit_tolerance,
        args.profit_audit_tolerance,
        args.nash_profit_tolerance,
    )
    if any(value < 0.0 for value in tolerances) or args.consecutive_sweeps <= 0:
        parser.error("Tolerances must be nonnegative and consecutive-sweeps positive.")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
