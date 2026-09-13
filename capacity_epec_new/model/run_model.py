"""Run the three-investor IEEE-9 BESS capacity game.

Nodal power and energy capacity are the only strategic variables.  Conditional
on them the ISO dispatches at minimum system cost, and that dispatch is
embedded in each investor's problem through relaxed KKT conditions or exact
strong duality.

    python model/run_model.py --max-sweeps 400 --damping 0.25

Exit status is 0 only for an equilibrium candidate: the iteration converged
*and* the zero-proximal audit passed.  Anything else exits 1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import reporting
from capacity_game import (
    CONVERGENCE_METRICS,
    GameConfig,
    GameState,
    audit_equilibrium,
    initial_state,
    run_gauss_seidel,
    run_jacobi,
)
from investors import three_investors
from iso_market import settle
from market_data import load_market_data
from solvers import SolverSettings


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "capacity_game"
DEFAULT_DATA = Path(__file__).resolve().parent / "input" / "market_data_smoothed.json"


def parse_node_costs(entries: list[str] | None, option: str) -> dict[str, float]:
    """Parse repeated NODE=COEFFICIENT entries."""

    result: dict[str, float] = {}
    for entry in entries or ():
        node, separator, raw_value = entry.partition("=")
        if not separator or not node.strip():
            raise SystemExit(f"{option}: expected NODE=COEFFICIENT, got {entry!r}")
        try:
            value = float(raw_value)
        except ValueError:
            raise SystemExit(f"{option}: non-numeric coefficient in {entry!r}") from None
        if value < 0.0:
            raise SystemExit(f"{option}: coefficients cannot be negative")
        result[node.strip()] = value
    return result


def parse_generation_ownership(entries: list[str] | None) -> dict[str, dict[str, float]] | None:
    """Parse INVESTOR=GEN[:SHARE][,GEN[:SHARE]] entries into a share mapping."""

    if not entries:
        return None
    ownership: dict[str, dict[str, float]] = {}
    for entry in entries:
        unit, _, spec = entry.partition("=")
        unit = unit.strip()
        if not unit or not _:
            raise SystemExit(f"--generation-ownership: expected INVESTOR=GEN..., got {entry!r}")
        shares = ownership.setdefault(unit, {})
        for token in (t.strip() for t in spec.split(",") if t.strip()):
            generator, _, share = token.partition(":")
            try:
                shares[generator.strip()] = float(share) if share else 1.0
            except ValueError:
                raise SystemExit(
                    f"--generation-ownership: {token!r} has a non-numeric share"
                ) from None
    return ownership


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    formulation = parser.add_argument_group("formulation")
    formulation.add_argument(
        "--market",
        choices=("energy-only",),
        default="energy-only",
        help="Energy-only market (the only formulation in this minimal project).",
    )
    formulation.add_argument(
        "--formulation",
        choices=("relaxed-kkt", "strong-duality", "relaxed-strong-duality"),
        default="relaxed-kkt",
        help=(
            "Lower-level optimality: every complementarity product <= epsilon "
            "(relaxed-kkt), primal objective == dual objective (strong-duality), or "
            "primal objective - dual objective <= epsilon (relaxed-strong-duality)."
        ),
    )
    formulation.add_argument(
        "--complementarity-epsilon",
        type=float,
        default=1.0e-6,
        help=(
            "relaxed-kkt: bound on each complementarity product. "
            "relaxed-strong-duality: bound on the total primal-dual gap in EUR/day. "
            "Ignored by strong-duality."
        ),
    )
    formulation.add_argument(
        "--demand-adjustment-penalty-eur-per-mw2",
        type=float,
        default=0.0,
        help=(
            "Effective rho for the ISO's quadratic demand adjustment. Zero (the "
            "reference default) fixes DemandAdjustment to zero everywhere and "
            "models inelastic demand. A positive rho introduces real demand "
            "elasticity: DemandAdjustment=LMP/rho, which can change congestion "
            "regimes and profits; it is not a neutral numerical tie-breaker."
        ),
    )
    formulation.add_argument(
        "--quadratic-cost-power-eur-per-mw2",
        type=float,
        default=0.0,
        help=(
            "Common portfolio power-cost curvature; adds 0.5*qP*(sum_n X_power[n])^2 "
            "to the OVERNIGHT cost, which the CRF then annualises by ~3.2e-4."
        ),
    )
    formulation.add_argument(
        "--quadratic-cost-energy-eur-per-mwh2",
        type=float,
        default=0.0,
        help=(
            "Common portfolio energy-cost curvature; adds 0.5*qE*(sum_n X_energy[n])^2 "
            "to the overnight cost, annualised by the same ~3.2e-4 factor."
        ),
    )
    formulation.add_argument(
        "--quadratic-cost-power-per-node-eur-per-mw2",
        type=float,
        default=0.0,
        help=(
            "PER-NODE power-cost curvature; adds 0.5*qP*sum_n(X_power[n]^2) to the "
            "overnight cost. Unlike --quadratic-cost-power-eur-per-mw2 (which curves "
            "on the portfolio TOTAL and has identical marginal cost at every node, so "
            "it cannot break ties between nodes), this raises a node's own marginal "
            "cost with that node's own capacity and so directly discourages "
            "concentrating capacity on a few nodes."
        ),
    )
    formulation.add_argument(
        "--quadratic-cost-energy-per-node-eur-per-mwh2",
        type=float,
        default=0.0,
        help="PER-NODE energy-cost curvature; adds 0.5*qE*sum_n(X_energy[n]^2). See above.",
    )
    formulation.add_argument(
        "--quadratic-cost-power-by-node-eur-per-mw2",
        nargs="+",
        default=None,
        metavar="NODE=Q",
        help=(
            "Node-specific overrides of the common per-node qP, e.g. "
            "N1=515 N6=5125 N8=5125. Missing nodes use the common value."
        ),
    )
    formulation.add_argument(
        "--quadratic-cost-energy-by-node-eur-per-mwh2",
        nargs="+",
        default=None,
        metavar="NODE=Q",
        help="Node-specific overrides of the common per-node qE.",
    )

    ownership = parser.add_argument_group("generation ownership")
    ownership.add_argument(
        "--generation-ownership",
        nargs="+",
        default=None,
        metavar="INVESTOR=GEN[:SHARE][,GEN[:SHARE]]",
        help=(
            "Override who owns which generation, e.g. "
            "--generation-ownership I1=RES_Wind_N1 I2=RES_PV_N6 I3=RES_PV_N8. "
            "Default: I2 owns all wind and I3 all PV. Ownership changes who captures "
            "price support from congestion relief, but does not by itself establish "
            "a unique split or a continuum; that must be tested with unrestricted "
            "best responses. Shares default to 1.0 and may not sum above 1.0 for any "
            "generator."
        ),
    )

    access = parser.add_argument_group("nodal BESS connection")
    access.add_argument(
        "--node-limit-mw",
        type=float,
        default=None,
        help=(
            "Uniform aggregate BESS MW cap at every node; default uses the input limits. "
            "Pass 'inf' to switch the shared cap off in the MPEC, the candidate screen "
            "and the exact reclear at once."
        ),
    )

    iteration = parser.add_argument_group("iteration")
    iteration.add_argument(
        "--update-scheme",
        choices=("jacobi", "gauss-seidel"),
        default="jacobi",
        help="Simultaneous Jacobi updates or sequential Gauss-Seidel updates.",
    )
    iteration.add_argument(
        "--gauss-seidel-order",
        nargs="+",
        default=None,
        metavar="INVESTOR",
        help=(
            "Fixed sequential order containing every investor exactly once, e.g. "
            "I2 I3 I1. Valid only with --update-scheme gauss-seidel."
        ),
    )
    iteration.add_argument(
        "--gauss-seidel-rotate-first-mover",
        action="store_true",
        help=(
            "Cyclically rotate the fixed order after each sweep so every investor "
            "moves first once per cycle. Valid only for Gauss-Seidel."
        ),
    )
    iteration.add_argument("--max-sweeps", type=int, default=400)
    iteration.add_argument("--damping", type=float, default=0.25)
    iteration.add_argument("--consecutive-sweeps", type=int, default=2)
    iteration.add_argument("--tolerance-mw", type=float, default=0.5)
    iteration.add_argument("--tolerance-mwh", type=float, default=1.0)
    iteration.add_argument(
        "--convergence-metric",
        choices=CONVERGENCE_METRICS,
        default="regret",
        help=(
            "What has to settle before the run stops. 'capacity': raw best-response "
            "MW/MWh deviations (not a Nash gap, and the wrong question when equilibria "
            "form a set - the iterate slides along it at constant payoff). 'regret': "
            "each profitable unilateral deviation relative to that investor's own "
            "profit, taking the largest ratio; this is the Nash condition itself and "
            "the same absolute deviation quantity the "
            "final audit reports. 'profit': incumbent payoffs stop moving between "
            "sweeps - cheapest, but flat payoffs also occur in the dead zone between "
            "branches, so it is a diagnostic and not a certificate. 'regret' and "
            "'profit' cost one extra market clear per sweep and need --update-scheme "
            "jacobi."
        ),
    )
    iteration.add_argument(
        "--tolerance-relative-regret",
        type=float,
        default=1.0e-3,
        help="Stop threshold for --convergence-metric regret (default 1e-3 = 0.1%%).",
    )
    iteration.add_argument(
        "--tolerance-profit-eur-per-day",
        type=float,
        default=1.0,
        help="Stop threshold for --convergence-metric profit.",
    )
    iteration.add_argument(
        "--allow-negative-profit-stop",
        dest="require_individual_rationality",
        action="store_false",
        help=(
            "Permit stopping with an investor below zero profit. Off by default: a "
            "loss-making investor means the run parked between branches."
        ),
    )
    parser.set_defaults(require_individual_rationality=True)
    iteration.add_argument("--proximal-penalty", type=float, default=0.0)
    iteration.add_argument("--initial-power-mw", type=float, default=5.0)
    iteration.add_argument("--initial-ratio-hours", type=float, default=3.0)
    start = iteration.add_mutually_exclusive_group()
    start.add_argument(
        "--symmetric-initialization", dest="symmetric_initialization", action="store_true"
    )
    start.add_argument(
        "--asymmetric-initialization", dest="symmetric_initialization", action="store_false"
    )
    parser.set_defaults(symmetric_initialization=True)
    iteration.add_argument(
        "--initial-capacities",
        type=Path,
        default=None,
        help=(
            "CSV with investor,node,power_mw,energy_mwh to start from (the format of "
            "final_capacities.csv). Overrides the symmetric and historical starts."
        ),
    )

    search = parser.add_argument_group("best-response search")
    search.add_argument(
        "--multistart-every-sweeps",
        type=int,
        default=0,
        help="Multistart on sweep 1 and every N sweeps; 0 disables it.",
    )
    search.add_argument("--refinement-starts", type=int, default=3)

    audit = parser.add_argument_group("audit")
    audit.add_argument("--audit-profit-tolerance-eur-per-day", type=float, default=1.0)
    audit.add_argument(
        "--audit-relative-regret-tolerance",
        type=float,
        default=1.0e-2,
        help=(
            "Best-found stationary-candidate threshold relative to each investor's "
            "own absolute incumbent payoff (default 0.01 = 1%%). Reported "
            "separately; it is not a global Nash certificate."
        ),
    )
    audit.add_argument("--audit-reclear-gap-tolerance-eur-per-day", type=float, default=10.0)
    audit.add_argument("--audit-primal-dual-gap-tolerance-eur-per-day", type=float, default=10.0)
    audit.add_argument("--audit-artificial-bound-utilization-limit", type=float, default=0.99)

    solver = parser.add_argument_group("solver")
    solver.add_argument("--parallel-workers", type=int, default=4)
    solver.add_argument("--ipopt-linear-solver", default="ma57")
    solver.add_argument("--max-solver-iterations", type=int, default=3_000)
    solver.add_argument("--max-solve-seconds", type=float, default=600.0)
    solver.add_argument("--solver-tolerance", type=float, default=1.0e-6)
    solver.add_argument("--tee", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace, data) -> GameConfig:
    if (
        args.quadratic_cost_power_eur_per_mw2 < 0.0
        or args.quadratic_cost_energy_eur_per_mwh2 < 0.0
        or args.quadratic_cost_power_per_node_eur_per_mw2 < 0.0
        or args.quadratic_cost_energy_per_node_eur_per_mwh2 < 0.0
    ):
        raise ValueError("Quadratic investment-cost coefficients cannot be negative.")
    if args.node_limit_mw is not None and not args.node_limit_mw > 0.0:
        raise ValueError("The node limit must be positive.")
    return GameConfig(
        investors=three_investors(
            data,
            ownership=parse_generation_ownership(args.generation_ownership),
            quadratic_cost_power_eur_per_mw2=args.quadratic_cost_power_eur_per_mw2,
            quadratic_cost_energy_eur_per_mwh2=args.quadratic_cost_energy_eur_per_mwh2,
            quadratic_cost_power_per_node_eur_per_mw2=args.quadratic_cost_power_per_node_eur_per_mw2,
            quadratic_cost_energy_per_node_eur_per_mwh2=args.quadratic_cost_energy_per_node_eur_per_mwh2,
            quadratic_cost_power_by_node_eur_per_mw2=parse_node_costs(
                args.quadratic_cost_power_by_node_eur_per_mw2,
                "--quadratic-cost-power-by-node-eur-per-mw2",
            ),
            quadratic_cost_energy_by_node_eur_per_mwh2=parse_node_costs(
                args.quadratic_cost_energy_by_node_eur_per_mwh2,
                "--quadratic-cost-energy-by-node-eur-per-mwh2",
            ),
        ),
        lower_level=args.formulation,
        market_design=args.market,
        complementarity_epsilon=args.complementarity_epsilon,
        max_sweeps=args.max_sweeps,
        damping=args.damping,
        consecutive_sweeps=args.consecutive_sweeps,
        tolerance_mw=args.tolerance_mw,
        tolerance_mwh=args.tolerance_mwh,
        convergence_metric=args.convergence_metric,
        tolerance_relative_regret=args.tolerance_relative_regret,
        tolerance_profit_eur_per_day=args.tolerance_profit_eur_per_day,
        require_individual_rationality=args.require_individual_rationality,
        gauss_seidel_order=(
            tuple(args.gauss_seidel_order) if args.gauss_seidel_order else None
        ),
        gauss_seidel_rotate_first_mover=args.gauss_seidel_rotate_first_mover,
        multistart_every_sweeps=args.multistart_every_sweeps,
        refinement_starts=args.refinement_starts,
        proximal_penalty=args.proximal_penalty,
        uniform_node_connection_limit_mw=args.node_limit_mw,
        symmetric_initialization=args.symmetric_initialization,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        audit_profit_tolerance_eur_per_day=args.audit_profit_tolerance_eur_per_day,
        audit_relative_regret_tolerance=args.audit_relative_regret_tolerance,
        audit_reclear_gap_tolerance_eur_per_day=args.audit_reclear_gap_tolerance_eur_per_day,
        audit_primal_dual_gap_tolerance_eur_per_day=(
            args.audit_primal_dual_gap_tolerance_eur_per_day
        ),
        audit_artificial_bound_utilization_limit=(
            args.audit_artificial_bound_utilization_limit
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


def main() -> int:
    args = parse_args()
    if args.update_scheme != "gauss-seidel" and (
        args.gauss_seidel_order or args.gauss_seidel_rotate_first_mover
    ):
        raise SystemExit(
            "--gauss-seidel-order and --gauss-seidel-rotate-first-mover require "
            "--update-scheme gauss-seidel"
        )
    data = load_market_data(args.data)
    if args.demand_adjustment_penalty_eur_per_mw2 is not None:
        data = replace(
            data,
            demand_adjustment_penalty_eur_per_mw2=args.demand_adjustment_penalty_eur_per_mw2,
        )
    config = build_config(args, data)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # --initial-capacities overrides the symmetric start but never reaches
    # GameConfig, so record it explicitly: without this the run_config of a
    # warm-started run still claims symmetric_initialization.
    warm_start = (
        {
            "file": str(args.initial_capacities.resolve()),
            "sha256": hashlib.sha256(args.initial_capacities.read_bytes()).hexdigest(),
        }
        if args.initial_capacities is not None
        else None
    )
    reporting.write_json(
        out / "run_config.json",
        {
            **reporting.run_config(
                config,
                args.data,
                hashlib.sha256(args.data.read_bytes()).hexdigest(),
                demand_adjustment_penalty_eur_per_mw2=(
                    data.demand_adjustment_penalty_eur_per_mw2
                ),
            ),
            "update_scheme": args.update_scheme,
            "initial_capacities": warm_start,
            "start_is_symmetric": warm_start is None and config.symmetric_initialization,
        },
    )

    state = initial_state(data, config, read_capacities(args.initial_capacities))
    nodal_trajectory = reporting.capacity_rows(
        data, config.investors, state.power, state.energy, sweep=0
    )
    total_trajectory = reporting.capacity_total_rows(
        data, config.investors, state.power, state.energy, sweep=0
    )

    def on_sweep(state: GameState) -> None:
        row = state.history[-1]
        print(
            f"sweep={state.sweep:03d} "
            f"power_residual={row['max_raw_power_deviation_mw']:.4f} MW "
            f"energy_residual={row['max_raw_energy_deviation_mwh']:.4f} MWh "
            f"total_power={row['total_power_mw']:.3f} MW "
            f"optimal={row['all_best_responses_optimal']}",
            flush=True,
        )
        nodal_trajectory.extend(
            reporting.capacity_rows(
                data, config.investors, state.power, state.energy, sweep=state.sweep
            )
        )
        total_trajectory.extend(
            reporting.capacity_total_rows(
                data, config.investors, state.power, state.energy, sweep=state.sweep
            )
        )
        reporting.write_rows(out / "history.csv", state.history)
        reporting.write_rows(out / "capacity_by_investor_node_by_sweep.csv", nodal_trajectory)
        reporting.write_rows(out / "capacity_totals_by_investor_by_sweep.csv", total_trajectory)
        reporting.write_rows(
            out / "current_capacities.csv",
            reporting.capacity_rows(data, config.investors, state.power, state.energy),
        )
        reporting.write_json(out / "checkpoint.json", reporting.checkpoint(state, config))
        reporting.write_json(
            out / f"starts_sweep_{state.sweep:03d}.json",
            {unit: response.start_records for unit, response in state.responses.items()},
        )

    print(
        f"Capacity EPEC: market={config.market_design}, formulation={config.lower_level}, "
        f"investors={len(config.investors)}, workers={config.parallel_workers}",
        flush=True,
    )
    runner = run_jacobi if args.update_scheme == "jacobi" else run_gauss_seidel
    state = runner(data, config, initial=state, on_sweep=on_sweep)
    report = audit_equilibrium(data, config, state)

    reporting.write_rows(
        out / "final_capacities.csv",
        reporting.capacity_rows(data, config.investors, state.power, state.energy),
    )
    reporting.write_rows(out / "final_audit.csv", [asdict(row) for row in report.rows])
    reporting.write_json(
        out / "starts_final_audit.json",
        {unit: response.start_records for unit, response in state.responses.items()},
    )
    if report.market is not None:
        reporting.write_rows(out / "final_market.csv", reporting.market_rows(report.market, data))
        reporting.write_rows(
            out / "profit_decomposition.csv",
            [
                asdict(settle(report.market, data, investor, state.power, state.energy))
                for investor in config.investors
            ],
        )

    summary = build_summary(data, config, state, report)
    reporting.write_json(out / "summary.json", summary)
    reporting.write_json(out / "checkpoint.json", reporting.checkpoint(state, config))
    print(reporting.json_dumps(summary, indent=2), flush=True)
    return 0 if summary["converged"] else 1


def read_capacities(path: Path | None):
    """Read a starting profile from a `final_capacities.csv`-shaped file."""

    if path is None:
        return None
    power: dict[tuple[str, str], float] = {}
    energy: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["investor"], row["node"])
            power[key] = float(row["power_mw"])
            energy[key] = float(row["energy_mwh"])
    if not power:
        raise ValueError(f"No capacity rows in {path}.")
    return power, energy


def build_summary(data, config: GameConfig, state: GameState, report) -> dict[str, object]:
    equilibrium_candidate = state.converged and report.passed
    stop_reason = state.stop_reason
    if state.converged and not report.passed:
        stop_reason = "iteration converged but the zero-proximal final audit failed"
    nodal_power = {
        node: sum(state.power[unit, node] for unit in config.investor_ids)
        for node in data.nodes
    }
    node_limits = config.node_limits(data)
    return {
        "formulation": f"capacity-only-{config.lower_level}",
        "strategic_variables": ["power_capacity", "energy_capacity"],
        "operational_bidding": False,
        "market_design": config.market_design,
        "converged": equilibrium_candidate,
        "iteration_converged": state.converged,
        "local_zero_proximal_audit_valid": report.valid,
        "local_zero_proximal_audit_passed": report.passed,
        "local_zero_proximal_audit_numerically_valid": report.numerically_valid,
        # This is intentionally separate from ``converged``: IPOPT plus
        # multistart supplies best-found local deviations, not global-response
        # bounds.  It is an epsilon-stationary candidate, never a certificate.
        "best_found_epsilon_stationary_candidate": report.passes_relative_regret(),
        "relative_regret_tolerance": config.audit_relative_regret_tolerance,
        "final_max_relative_regret": report.max_relative_regret,
        "final_relative_regret_by_investor": report.relative_regret_by_investor,
        # IPOPT certifies local solutions only; multistart is a search, not a proof.
        "global_best_response_certified": False,
        "final_market_valid": report.market is not None,
        "sweeps": state.sweep,
        "stop_reason": stop_reason,
        "total_power_mw": state.total_power_mw(),
        "total_energy_mwh": state.total_energy_mwh(),
        "maximum_nodal_capacity_mw": max(nodal_power.values()),
        "nodal_power_mw": nodal_power,
        "shared_node_cap_enforced": config.shared_node_cap_enforced,
        "node_connection_limit_mw": node_limits,
        "nodal_connection_utilization": {
            node: nodal_power[node] / node_limits[node] for node in data.nodes
        },
        "final_raw_power_residual_mw": report.worst("max_power_deviation_mw"),
        "final_raw_energy_residual_mwh": report.worst("max_energy_deviation_mwh"),
        "final_profitable_deviation_eur_per_day": report.worst(
            "profitable_deviation_eur_per_day"
        ),
        "final_embedded_reclear_profit_gap_eur_per_day": report.worst(
            "embedded_reclear_profit_gap_eur_per_day"
        ),
        "final_complementarity_violation": report.worst("complementarity_max_violation"),
        "capacity_trajectory_file": "capacity_by_investor_node_by_sweep.csv",
        "investor_totals_trajectory_file": "capacity_totals_by_investor_by_sweep.csv",
    }


if __name__ == "__main__":
    raise SystemExit(main())
