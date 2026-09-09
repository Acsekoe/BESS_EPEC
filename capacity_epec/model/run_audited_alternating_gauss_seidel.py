"""Run fixed- or alternating-order Gauss--Seidel experiments with exact audits.

The capacity update itself is deliberately cheap: each investor solves once
from the incumbent in sequence.  Capacity distance is reported but never used
to declare convergence.  After every complete sweep, a zero-proximal
multistart audit computes exact-recleared regret for all investors.  The run
stops successfully only when maximum regret and the numerical diagnostics pass.

By default this development experiment starts from the profile historically
called the sweep-254 profile.  Its provenance includes the scripted I3
conditional scan followed by 30 Jacobi sweeps; the input file and its SHA-256
are written to ``run_config.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import pyomo

import capacity_game as game
import reporting
from investors import three_investors
from iso_market import settle
from market_data import load_market_data
from solvers import SolverSettings, ipopt_path


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = MODEL_DIR / "input" / "market_data_smoothed.json"
DEFAULT_START = MODEL_DIR / "output" / "from_i3_scan_optimum" / "final_capacities.csv"
DEFAULT_OUTPUT = MODEL_DIR / "output" / "audited_alternating_gs_from_sweep254"


def rotating_order(investor_ids: tuple[str, ...], sweep: int) -> tuple[str, ...]:
    """Rotate deterministically so every investor moves last once per cycle."""

    if not investor_ids:
        raise ValueError("At least one investor is required.")
    offset = (sweep - 1) % len(investor_ids)
    return investor_ids[offset:] + investor_ids[:offset]


def read_capacities(path: Path) -> tuple[game.Capacities, game.Capacities]:
    power: game.Capacities = {}
    energy: game.Capacities = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row["investor"], row["node"]
            power[key] = float(row["power_mw"])
            energy[key] = float(row["energy_mwh"])
    if not power:
        raise ValueError(f"No capacities found in {path}.")
    return power, energy


def audit_numerically_valid(report: game.AuditReport) -> bool:
    """Require solved best responses and trustworthy numerical diagnostics."""

    if not report.valid or not all(row.optimal for row in report.rows):
        return False
    limits = {
        "embedded_reclear_profit_gap_eur_per_day": (
            report.config.audit_reclear_gap_tolerance_eur_per_day
        ),
        "complementarity_max_violation": report.config.solver.tolerance,
        "max_bound_violation": report.config.solver.tolerance,
    }
    return all(
        (value := report.worst(field)) is not None and value <= limit
        for field, limit in limits.items()
    )


def audit_passes(report: game.AuditReport, epsilon: float) -> bool:
    """Regret-based acceptance without any capacity-distance requirement."""

    regret = report.worst("profitable_deviation_eur_per_day")
    return (
        audit_numerically_valid(report)
        and regret is not None
        and regret <= epsilon
    )


def audit_row(
    sweep: int,
    order: tuple[str, ...],
    state: game.GameState,
    report: game.AuditReport,
    sweep_seconds: float,
    audit_seconds: float,
    max_update_power: float,
    max_update_energy: float,
) -> dict[str, object]:
    regrets = {
        row.investor: row.profitable_deviation_eur_per_day for row in report.rows
    }
    row: dict[str, object] = {
        "sweep": sweep,
        "order": "->".join(order) if order else "initial",
        "maximum_regret_eur_per_day": report.worst(
            "profitable_deviation_eur_per_day"
        ),
        "sum_regret_eur_per_day": (
            sum(regrets.values())
            if regrets and all(math.isfinite(value) for value in regrets.values())
            else math.nan
        ),
        "audit_valid": report.valid,
        "audit_numerically_valid": audit_numerically_valid(report),
        "audit_passed": audit_passes(
            report, report.config.audit_profit_tolerance_eur_per_day
        ),
        "maximum_embedded_reclear_gap_eur_per_day": report.worst(
            "embedded_reclear_profit_gap_eur_per_day"
        ),
        "maximum_complementarity_violation": report.worst(
            "complementarity_max_violation"
        ),
        "maximum_bound_violation": report.worst("max_bound_violation"),
        "max_applied_power_update_mw": max_update_power,
        "max_applied_energy_update_mwh": max_update_energy,
        "total_power_mw": state.total_power_mw(),
        "total_energy_mwh": state.total_energy_mwh(),
        "sweep_seconds": sweep_seconds,
        "audit_seconds": audit_seconds,
    }
    for unit, regret in regrets.items():
        row[f"regret_{unit}_eur_per_day"] = regret
    return row


def write_progress(out: Path, **values: object) -> None:
    reporting.write_json(
        out / "progress.json",
        {"updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"), **values},
    )


def write_checkpoint(
    out: Path,
    data,
    config: game.GameConfig,
    algorithm: str,
    state: game.GameState,
    history: list[dict[str, object]],
    best_sweep: int,
    best_power: game.Capacities,
    best_energy: game.Capacities,
    report: game.AuditReport,
) -> None:
    reporting.write_rows(out / "history.csv", history)
    reporting.write_rows(
        out / "current_capacities.csv",
        reporting.capacity_rows(data, config.investors, state.power, state.energy),
    )
    reporting.write_rows(
        out / "best_capacities.csv",
        reporting.capacity_rows(data, config.investors, best_power, best_energy),
    )
    reporting.write_json(
        out / "checkpoint.json",
        {
            **reporting.checkpoint(state, config),
            "algorithm": algorithm,
            "best_audited_sweep": best_sweep,
            "latest_maximum_regret_eur_per_day": history[-1][
                "maximum_regret_eur_per_day"
            ],
            "latest_sum_regret_eur_per_day": history[-1][
                "sum_regret_eur_per_day"
            ],
        },
    )
    reporting.write_rows(
        out / f"audit_sweep_{state.sweep:03d}.csv",
        [asdict(row) for row in report.rows],
    )
    reporting.write_json(
        out / f"audit_starts_sweep_{state.sweep:03d}.json",
        {
            unit: list(response.start_records)
            for unit, response in state.responses.items()
        },
    )
    if report.market is not None:
        reporting.write_rows(
            out / "current_market.csv", reporting.market_rows(report.market, data)
        )
        reporting.write_rows(
            out / "profit_decomposition.csv",
            [
                asdict(settle(report.market, data, investor, state.power, state.energy))
                for investor in config.investors
            ],
        )


def run(args: argparse.Namespace) -> int:
    data_path = args.data.resolve()
    start_path = args.initial_capacities.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = load_market_data(data_path)
    investors = three_investors(data)
    config = game.GameConfig(
        investors=investors,
        lower_level=args.formulation,
        complementarity_epsilon=args.complementarity_epsilon,
        max_sweeps=args.sweeps,
        damping=args.damping,
        proximal_penalty=0.0,
        multistart_every_sweeps=0,
        refinement_starts=args.audit_refinement_starts,
        audit_profit_tolerance_eur_per_day=args.epsilon_eur_per_day,
        audit_reclear_gap_tolerance_eur_per_day=(
            args.audit_reclear_gap_tolerance_eur_per_day
        ),
        parallel_workers=args.parallel_workers,
        solver=SolverSettings(
            linear_solver=args.ipopt_linear_solver,
            max_iterations=args.max_solver_iterations,
            max_seconds=args.max_solve_seconds,
            tolerance=args.solver_tolerance,
            tee=args.tee,
        ),
    )
    initial = read_capacities(start_path)
    state = game.initial_state(data, config, initial)
    base_ids = config.investor_ids
    algorithm = (
        "audited-fixed-order-gauss-seidel"
        if args.order_mode == "fixed"
        else "audited-alternating-gauss-seidel"
    )
    investors_by_id = {investor.investor_id: investor for investor in investors}
    history: list[dict[str, object]] = []
    best_key = (math.inf, math.inf)
    best_sweep = 0
    best_power, best_energy = dict(state.power), dict(state.energy)

    executable = ipopt_path(config.solver)
    reporting.write_json(
        out / "run_config.json",
        {
            **reporting.run_config(
                config, data_path, hashlib.sha256(data_path.read_bytes()).hexdigest()
            ),
            "algorithm": algorithm,
            "capacity_distance_is_stopping_criterion": False,
            "audit_after_every_sweep": True,
            "order_mode": args.order_mode,
            "sweep_orders_first_cycle": [
                list(
                    base_ids
                    if args.order_mode == "fixed"
                    else rotating_order(base_ids, sweep)
                )
                for sweep in range(1, len(base_ids) + 1)
            ],
            "initial_capacities": {
                "file": str(start_path),
                "sha256": hashlib.sha256(start_path.read_bytes()).hexdigest(),
                "historical_label": "sweep-254 final Jacobi iterate",
                "provenance_note": (
                    "Includes the scripted I3 conditional scan followed by 30 "
                    "Jacobi sweeps; not a pure uninterrupted 254-sweep chain."
                ),
            },
            "python_version": platform.python_version(),
            "pyomo_version": pyomo.__version__,
            "ipopt_executable": str(executable) if executable else None,
            "code_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), Path(game.__file__))
            },
        },
    )

    write_progress(out, phase="initial_audit", sweep=0)
    audit_started = time.perf_counter()
    report = game.audit_equilibrium(data, config, state)
    audit_seconds = time.perf_counter() - audit_started
    initial_row = audit_row(
        0, (), state, report, 0.0, audit_seconds, 0.0, 0.0
    )
    history.append(initial_row)
    if audit_numerically_valid(report):
        best_key = (
            float(initial_row["maximum_regret_eur_per_day"]),
            float(initial_row["sum_regret_eur_per_day"]),
        )
    write_checkpoint(
        out,
        data,
        config,
        algorithm,
        state,
        history,
        best_sweep,
        best_power,
        best_energy,
        report,
    )
    print(
        f"audit sweep=000 max_regret={initial_row['maximum_regret_eur_per_day']:.6g} "
        f"sum_regret={initial_row['sum_regret_eur_per_day']:.6g}",
        flush=True,
    )

    for sweep in range(1, args.sweeps + 1):
        order = (
            base_ids
            if args.order_mode == "fixed"
            else rotating_order(base_ids, sweep)
        )
        sweep_started = time.perf_counter()
        max_power_update = 0.0
        max_energy_update = 0.0
        cheap_responses: dict[str, game.BestResponse] = {}
        for position, unit in enumerate(order, start=1):
            write_progress(
                out,
                phase="cheap_best_response",
                sweep=sweep,
                order=list(order),
                investor=unit,
                investor_position=position,
                investors_in_sweep=len(order),
            )
            response = game.best_response(
                data,
                config,
                investors_by_id[unit],
                dict(state.power),
                dict(state.energy),
                use_multistart=False,
            )
            cheap_responses[unit] = response
            if not response.outcome.has_solution:
                continue
            for node in data.nodes:
                key = unit, node
                old_power, old_energy = state.power[key], state.energy[key]
                new_power = game._damped(old_power, response.power[node], args.damping)
                new_energy = game._damped(old_energy, response.energy[node], args.damping)
                max_power_update = max(max_power_update, abs(new_power - old_power))
                max_energy_update = max(max_energy_update, abs(new_energy - old_energy))
                state.power[key] = new_power
                state.energy[key] = new_energy
                if state.power[key] < config.cleanup_tolerance:
                    state.power[key] = 0.0
                    state.energy[key] = 0.0
        sweep_seconds = time.perf_counter() - sweep_started
        reporting.write_json(
            out / f"cheap_responses_sweep_{sweep:03d}.json",
            {
                unit: {
                    "termination": response.outcome.termination,
                    "optimal": response.outcome.optimal,
                    "selected_start": response.start_label,
                    "recleared_profit_eur_per_day": (
                        response.recleared_profit_eur_per_day
                    ),
                    "starts": list(response.start_records),
                }
                for unit, response in cheap_responses.items()
            },
        )

        state.sweep = sweep
        write_progress(out, phase="full_multistart_audit", sweep=sweep, order=list(order))
        audit_started = time.perf_counter()
        report = game.audit_equilibrium(data, config, state)
        audit_seconds = time.perf_counter() - audit_started
        row = audit_row(
            sweep,
            order,
            state,
            report,
            sweep_seconds,
            audit_seconds,
            max_power_update,
            max_energy_update,
        )
        history.append(row)
        key = (
            float(row["maximum_regret_eur_per_day"]),
            float(row["sum_regret_eur_per_day"]),
        )
        if row["audit_numerically_valid"] and key < best_key:
            best_key = key
            best_sweep = sweep
            best_power, best_energy = dict(state.power), dict(state.energy)
        write_checkpoint(
            out,
            data,
            config,
            algorithm,
            state,
            history,
            best_sweep,
            best_power,
            best_energy,
            report,
        )
        print(
            f"audit sweep={sweep:03d} order={'->'.join(order)} "
            f"max_regret={row['maximum_regret_eur_per_day']:.6g} "
            f"sum_regret={row['sum_regret_eur_per_day']:.6g} "
            f"passed={row['audit_passed']}",
            flush=True,
        )
        if row["audit_passed"]:
            state.converged = True
            state.stop_reason = (
                f"maximum exact-recleared regret <= {args.epsilon_eur_per_day:g} "
                "EUR/day with valid diagnostics"
            )
            break

    if not state.converged:
        state.stop_reason = f"completed {state.sweep} audited alternating sweeps"
    reporting.write_rows(
        out / "final_capacities.csv",
        reporting.capacity_rows(data, config.investors, state.power, state.energy),
    )
    reporting.write_json(
        out / "summary.json",
        {
            "algorithm": algorithm,
            "converged": state.converged,
            "global_best_response_certified": False,
            "sweeps": state.sweep,
            "best_audited_sweep": best_sweep,
            "best_maximum_regret_eur_per_day": best_key[0],
            "best_sum_regret_eur_per_day": best_key[1],
            "final_maximum_regret_eur_per_day": history[-1][
                "maximum_regret_eur_per_day"
            ],
            "final_sum_regret_eur_per_day": history[-1][
                "sum_regret_eur_per_day"
            ],
            "stop_reason": state.stop_reason,
        },
    )
    write_progress(
        out,
        phase="complete",
        sweep=state.sweep,
        converged=state.converged,
        stop_reason=state.stop_reason,
    )
    return 0 if state.converged else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--initial-capacities", type=Path, default=DEFAULT_START)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument(
        "--order-mode",
        choices=("alternating", "fixed"),
        default="alternating",
        help="Rotate the first mover each sweep or retain I1->I2->I3.",
    )
    parser.add_argument("--epsilon-eur-per-day", type=float, default=20.0)
    parser.add_argument("--audit-refinement-starts", type=int, default=3)
    parser.add_argument(
        "--audit-reclear-gap-tolerance-eur-per-day", type=float, default=2.0
    )
    parser.add_argument(
        "--formulation", choices=("relaxed-kkt", "strong-duality"), default="relaxed-kkt"
    )
    parser.add_argument("--complementarity-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--ipopt-linear-solver", default="ma57")
    parser.add_argument("--max-solver-iterations", type=int, default=3_000)
    parser.add_argument("--max-solve-seconds", type=float, default=600.0)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args()
    if args.sweeps <= 0:
        parser.error("--sweeps must be positive")
    if not 0.0 < args.damping <= 1.0:
        parser.error("--damping must lie in (0, 1]")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
