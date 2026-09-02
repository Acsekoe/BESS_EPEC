"""Run damped Gauss-Seidel on the three-bus strategic-power toy model."""

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
    market_profit,
    solve_best_response,
    solve_market,
)


ROOT = Path(__file__).resolve().parent


def _maximum_deviation(
    response: BestResponse,
    charge: dict[tuple[str, int], float],
    discharge: dict[tuple[str, int], float],
) -> float:
    investor = response.investor_id
    return max(
        [
            abs(response.charge_offer_mw[t] - charge[investor, t])
            for t in response.charge_offer_mw
        ]
        + [
            abs(response.discharge_offer_mw[t] - discharge[investor, t])
            for t in response.discharge_offer_mw
        ]
    )


def _solve_response(args, data, investor_id, charge, discharge) -> BestResponse:
    return solve_best_response(
        data,
        investor_id,
        charge,
        discharge,
        complementarity_epsilon=args.complementarity_epsilon,
        availability_tie_breaker_eur_per_mw=args.availability_tie_breaker,
        strategic_charge=args.two_sided,
        max_solve_seconds=args.max_solve_seconds,
        tee=args.tee,
    )


def _write_outputs(output_dir: Path, data, charge, discharge, market, history, audits, summary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    with (output_dir / "final_strategies.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["investor", "time", "charge_offer_mw", "discharge_offer_mw"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for investor in data.investors:
            for time in data.times:
                writer.writerow(
                    {
                        "investor": investor.investor_id,
                        "time": time,
                        "charge_offer_mw": charge[investor.investor_id, time],
                        "discharge_offer_mw": discharge[investor.investor_id, time],
                    }
                )

    with (output_dir / "final_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "investor",
            "optimal",
            "termination",
            "max_raw_deviation_mw",
            "profit_eur_per_day",
            "max_complementarity_product",
            "max_complementarity_violation",
            "absolute_primal_dual_gap_eur_per_day",
            "recleared_profit_eur_per_day",
            "embedded_reclear_profit_gap_eur_per_day",
            "maximum_lmp_reclear_gap_eur_per_mwh",
            "profitable_deviation_eur_per_day",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for response, deviation, profit_gain in audits:
            writer.writerow(
                {
                    "investor": response.investor_id,
                    "optimal": response.optimal,
                    "termination": response.termination,
                    "max_raw_deviation_mw": deviation,
                    "profit_eur_per_day": response.profit_eur_per_day,
                    "max_complementarity_product": response.maximum_complementarity_product,
                    "max_complementarity_violation": response.maximum_complementarity_violation,
                    "absolute_primal_dual_gap_eur_per_day": response.absolute_primal_dual_gap_eur_per_day,
                    "recleared_profit_eur_per_day": response.recleared_profit_eur_per_day,
                    "embedded_reclear_profit_gap_eur_per_day": response.embedded_reclear_profit_gap_eur_per_day,
                    "maximum_lmp_reclear_gap_eur_per_mwh": response.maximum_lmp_reclear_gap_eur_per_mwh,
                    "profitable_deviation_eur_per_day": profit_gain,
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
                            float(pyo.value(market.PGen[g, time])) for g in generator_by_node[node]
                        ),
                        "charge_mw": sum(
                            float(pyo.value(market.PCharge[i, time])) for i in investor_by_node[node]
                        ),
                        "discharge_mw": sum(
                            float(pyo.value(market.PDischarge[i, time])) for i in investor_by_node[node]
                        ),
                        "net_injection_mw": float(pyo.value(market.NetInjection[node, time])),
                        "demand_adjustment_mw": float(
                            pyo.value(market.DemandAdjustment[node, time])
                        ),
                    }
                )


def run(args) -> int:
    data = default_data()
    charge, discharge = full_availability(data)
    investor_ids = [investor.investor_id for investor in data.investors]
    history: list[dict[str, object]] = []
    stable_sweeps = 0
    converged = False
    final_audits: list[tuple[BestResponse, float, float]] = []

    for sweep in range(1, args.max_sweeps + 1):
        shift = (sweep - 1) % len(investor_ids) if args.rotate_order else 0
        order = investor_ids[shift:] + investor_ids[:shift]
        update_optimal = True
        maximum_update_deviation = 0.0
        update_seconds = 0.0

        for investor_id in order:
            response = _solve_response(args, data, investor_id, charge, discharge)
            update_seconds += response.solve_seconds
            update_optimal = update_optimal and response.optimal
            deviation = _maximum_deviation(response, charge, discharge)
            maximum_update_deviation = max(maximum_update_deviation, deviation)
            if response.optimal:
                for time in data.times:
                    charge[investor_id, time] = (
                        (1.0 - args.damping) * charge[investor_id, time]
                        + args.damping * response.charge_offer_mw[time]
                    )
                    discharge[investor_id, time] = (
                        (1.0 - args.damping) * discharge[investor_id, time]
                        + args.damping * response.discharge_offer_mw[time]
                    )

        market = solve_market(data, charge, discharge)
        final_audits = []
        audit_optimal = True
        maximum_nash_deviation = 0.0
        maximum_comp_violation = 0.0
        maximum_gap = 0.0
        maximum_profit_reclear_gap = 0.0
        maximum_lmp_reclear_gap = 0.0
        maximum_profitable_deviation = 0.0
        audit_seconds = 0.0
        for investor_id in investor_ids:
            response = _solve_response(args, data, investor_id, charge, discharge)
            audit_seconds += response.solve_seconds
            deviation = _maximum_deviation(response, charge, discharge)
            profit_gain = max(
                0.0,
                response.recleared_profit_eur_per_day - market_profit(market, investor_id),
            )
            final_audits.append((response, deviation, profit_gain))
            audit_optimal = audit_optimal and response.optimal
            maximum_nash_deviation = max(maximum_nash_deviation, deviation)
            maximum_comp_violation = max(
                maximum_comp_violation, response.maximum_complementarity_violation
            )
            maximum_gap = max(maximum_gap, response.absolute_primal_dual_gap_eur_per_day)
            maximum_profit_reclear_gap = max(
                maximum_profit_reclear_gap,
                response.embedded_reclear_profit_gap_eur_per_day,
            )
            maximum_lmp_reclear_gap = max(
                maximum_lmp_reclear_gap,
                response.maximum_lmp_reclear_gap_eur_per_mwh,
            )
            maximum_profitable_deviation = max(maximum_profitable_deviation, profit_gain)

        stable = (
            audit_optimal
            and maximum_nash_deviation <= args.tolerance_mw
            and maximum_profit_reclear_gap <= args.profit_audit_tolerance
            and maximum_lmp_reclear_gap <= args.price_audit_tolerance
            and maximum_profitable_deviation <= args.nash_profit_tolerance
        )
        stable_sweeps = stable_sweeps + 1 if stable else 0
        row: dict[str, object] = {
            "sweep": sweep,
            "order": "->".join(order),
            "all_updates_optimal": update_optimal,
            "all_audits_optimal": audit_optimal,
            "max_update_raw_deviation_mw": maximum_update_deviation,
            "max_nash_deviation_mw": maximum_nash_deviation,
            "max_complementarity_violation": maximum_comp_violation,
            "max_absolute_primal_dual_gap_eur_per_day": maximum_gap,
            "max_embedded_reclear_profit_gap_eur_per_day": maximum_profit_reclear_gap,
            "max_lmp_reclear_gap_eur_per_mwh": maximum_lmp_reclear_gap,
            "max_profitable_deviation_eur_per_day": maximum_profitable_deviation,
            "market_objective_eur_per_day": float(pyo.value(market.objective)),
            "update_solve_seconds": update_seconds,
            "audit_solve_seconds": audit_seconds,
            "stable_sweeps": stable_sweeps,
        }
        for investor_id in investor_ids:
            row[f"profit_{investor_id}_eur_per_day"] = market_profit(market, investor_id)
        history.append(row)
        print(
            f"sweep={sweep:02d} order={row['order']} "
            f"update_residual={maximum_update_deviation:.4f} MW "
            f"nash_residual={maximum_nash_deviation:.4f} MW "
            f"price_audit={maximum_lmp_reclear_gap:.4f} EUR/MWh "
            f"profit_gain={maximum_profitable_deviation:.4f} EUR/day "
            f"optimal={audit_optimal}"
        )
        if stable_sweeps >= args.consecutive_sweeps:
            converged = True
            break

    final_market = solve_market(data, charge, discharge)
    summary = {
        "formulation": "fixed-capacity strategic hourly charge/discharge availability",
        "strategic_charge_availability": args.two_sided,
        "strategic_discharge_availability": True,
        "algorithm": "damped Gauss-Seidel with simultaneous Nash audit after each sweep",
        "sweeps": len(history),
        "converged": converged,
        "stop_reason": (
            "Nash, LMP-reclear, and profit-reclear audits passed for "
            f"{stable_sweeps} consecutive sweeps"
            if converged
            else "maximum sweeps reached"
        ),
        "damping": args.damping,
        "tolerance_mw": args.tolerance_mw,
        "consecutive_sweeps": args.consecutive_sweeps,
        "complementarity_epsilon": args.complementarity_epsilon,
        "availability_tie_breaker_eur_per_mw": args.availability_tie_breaker,
        "price_audit_tolerance_eur_per_mwh": args.price_audit_tolerance,
        "profit_audit_tolerance_eur_per_day": args.profit_audit_tolerance,
        "nash_profit_tolerance_eur_per_day": args.nash_profit_tolerance,
        "final_max_nash_deviation_mw": history[-1]["max_nash_deviation_mw"],
        "final_all_audits_optimal": history[-1]["all_audits_optimal"],
        "final_max_lmp_reclear_gap_eur_per_mwh": history[-1][
            "max_lmp_reclear_gap_eur_per_mwh"
        ],
        "final_max_embedded_reclear_profit_gap_eur_per_day": history[-1][
            "max_embedded_reclear_profit_gap_eur_per_day"
        ],
        "final_max_profitable_deviation_eur_per_day": history[-1][
            "max_profitable_deviation_eur_per_day"
        ],
        "final_market_objective_eur_per_day": float(pyo.value(final_market.objective)),
        "final_profits_eur_per_day": {
            investor_id: market_profit(final_market, investor_id) for investor_id in investor_ids
        },
        "investors": {
            investor.investor_id: {
                "node": investor.node,
                "power_mw": investor.power_mw,
                "energy_mwh": investor.energy_mwh,
                "owned_generation_shares": dict(investor.owned_generation_shares),
            }
            for investor in data.investors
        },
    }
    if not args.no_output:
        _write_outputs(args.output_dir, data, charge, discharge, final_market, history, final_audits, summary)
    print(json.dumps(summary, indent=2))
    return 0 if converged else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sweeps", type=int, default=15)
    parser.add_argument("--damping", type=float, default=0.35)
    parser.add_argument("--tolerance-mw", type=float, default=0.10)
    parser.add_argument("--consecutive-sweeps", type=int, default=2)
    parser.add_argument("--complementarity-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--availability-tie-breaker", type=float, default=1.0e-3)
    parser.add_argument("--price-audit-tolerance", type=float, default=0.02)
    parser.add_argument("--profit-audit-tolerance", type=float, default=0.25)
    parser.add_argument("--nash-profit-tolerance", type=float, default=0.50)
    parser.add_argument(
        "--two-sided",
        action="store_true",
        help="Also make charging availability strategic; the default only withholds discharge.",
    )
    parser.add_argument("--max-solve-seconds", type=float, default=60.0)
    parser.add_argument("--fixed-order", dest="rotate_order", action="store_false")
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "default")
    parser.set_defaults(rotate_order=True)
    args = parser.parse_args()
    if args.max_sweeps <= 0 or not 0.0 < args.damping <= 1.0:
        parser.error("max-sweeps must be positive and damping must lie in (0, 1].")
    if (
        args.tolerance_mw < 0.0
        or args.consecutive_sweeps <= 0
        or args.availability_tie_breaker < 0.0
        or args.price_audit_tolerance < 0.0
        or args.profit_audit_tolerance < 0.0
        or args.nash_profit_tolerance < 0.0
    ):
        parser.error("tolerance must be nonnegative and consecutive-sweeps positive.")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
