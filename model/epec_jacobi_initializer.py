"""Build a feasible Gauss-Seidel starting point from one Jacobi best-response sweep.

Every investor responds to the same capacity snapshot. The default snapshot has
zero rival storage, while Ipopt starts from a separate positive numerical guess.
Raw desired MW/MWh are exported before a transparent proportional nodal
projection. The projected fleet is written as a checkpoint accepted by
``epec_diagonalization.py --resume-from``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
_PRIMAL_DUAL_DIR = _MODEL_DIR / "Primal and dual problems"
if _PRIMAL_DUAL_DIR.is_dir() and str(_PRIMAL_DUAL_DIR) not in sys.path:
    sys.path.append(str(_PRIMAL_DUAL_DIR))

from primal_market_clearing_model import load_market_data
from epec_diagonalization import (
    DEFAULT_DAMPING,
    DEFAULT_CAPACITY_CLEANUP_TOL_MW_MWH,
    DEFAULT_FLOOR_MW,
    DEFAULT_FLOOR_MWH,
    DEFAULT_NODE_LIMIT_MW,
    DEFAULT_SOLVER_TOL,
    DEFAULT_TOL_REL,
    EpecConfig,
    EpecState,
    clean_capacity_pair,
    four_investor_portfolio_profiles,
    separate_rival_capacities,
    solve_best_response,
)
from single_investor_mpec import (
    DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    EXPERIMENT_DATA_PATH,
    default_quadratic_demand_curve,
)
from single_investor_mpec_results import export_solution


DEFAULT_OUTPUT_DIR = _MODEL_DIR / "output" / "epec" / "initializer_jacobi_zero_snapshot"


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a projected EPEC starting checkpoint from one Jacobi sweep"
    )
    parser.add_argument("--data", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument(
        "--rival-snapshot-power-mw",
        type=float,
        default=0.0,
        help="Common economic snapshot MW per investor-node seen by every Jacobi response.",
    )
    parser.add_argument("--rival-snapshot-ratio-hours", type=float, default=2.0)
    parser.add_argument(
        "--mpec-initial-power-mw",
        type=float,
        default=10.0,
        help="Numerical Ipopt initial guess, independent of the economic rival snapshot.",
    )
    parser.add_argument("--mpec-initial-ratio-hours", type=float, default=2.0)
    parser.add_argument("--demand-model", choices=["fixed", "quadratic"], default="fixed")
    parser.add_argument("--dispatch-regularization", type=float, default=0.0)
    parser.add_argument("--solver-tol", type=float, default=DEFAULT_SOLVER_TOL)
    parser.add_argument(
        "--capacity-cleanup-tol",
        type=float,
        default=DEFAULT_CAPACITY_CLEANUP_TOL_MW_MWH,
        help="Set projected pairs to zero when both MW and MWh do not exceed this tolerance.",
    )
    parser.add_argument(
        "--price-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    )
    parser.add_argument(
        "--dual-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    )
    parser.add_argument("--max-cpu-time", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dual_bound_eur_per_mwh <= 0.0:
        raise SystemExit("--dual-bound-eur-per-mwh must be positive.")
    if args.price_bound_eur_per_mwh <= 0.0:
        raise SystemExit("--price-bound-eur-per-mwh must be positive.")
    if args.node_limit_mw <= 0.0:
        raise SystemExit("--node-limit-mw must be positive.")
    if args.rival_snapshot_power_mw < 0.0 or args.mpec_initial_power_mw < 0.0:
        raise SystemExit("Snapshot and initial-guess power must be non-negative.")
    if args.dispatch_regularization < 0.0:
        raise SystemExit("--dispatch-regularization must be non-negative.")
    if args.solver_tol <= 0.0:
        raise SystemExit("--solver-tol must be positive.")
    if args.capacity_cleanup_tol < 0.0:
        raise SystemExit("--capacity-cleanup-tol must be non-negative.")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty; choose a clean path: {output_dir}")

    data = load_market_data(args.data)
    investors = four_investor_portfolio_profiles(data)
    nodes = list(data.nodes)
    n_investors = len(investors)
    if args.rival_snapshot_power_mw * n_investors > args.node_limit_mw + 1e-9:
        raise SystemExit("The common rival snapshot violates the shared nodal connection limit.")
    for inv in investors:
        if not inv.ratio_min <= args.rival_snapshot_ratio_hours <= inv.ratio_max:
            raise SystemExit("--rival-snapshot-ratio-hours violates an investor E/P envelope.")
        if not inv.ratio_min <= args.mpec_initial_ratio_hours <= inv.ratio_max:
            raise SystemExit("--mpec-initial-ratio-hours violates an investor E/P envelope.")

    use_demand_curve = args.demand_model == "quadratic"
    cfg = EpecConfig(
        investors=investors,
        node_limit_mw=args.node_limit_mw,
        update_rule="jacobi",
        damping=1.0,
        max_iters=1,
        tol_rel=DEFAULT_TOL_REL,
        floor_mw=DEFAULT_FLOOR_MW,
        floor_mwh=DEFAULT_FLOOR_MWH,
        seed_power_mw=args.rival_snapshot_power_mw,
        seed_ratio_hours=args.rival_snapshot_ratio_hours,
        max_cpu_time=args.max_cpu_time,
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
        system_price_settlement=False,
        use_demand_curve=use_demand_curve,
        dispatch_regularization_eur_per_mw2h=args.dispatch_regularization,
        solver_tol=args.solver_tol,
        capacity_cleanup_tol_mw_mwh=args.capacity_cleanup_tol,
    )
    snapshot = EpecState(
        x_power={
            (inv.investor_id, n): args.rival_snapshot_power_mw
            for inv in investors
            for n in nodes
        },
        x_energy={
            (inv.investor_id, n): args.rival_snapshot_power_mw * args.rival_snapshot_ratio_hours
            for inv in investors
            for n in nodes
        },
    )
    guess_power = {n: args.mpec_initial_power_mw for n in nodes}
    guess_energy = {
        n: args.mpec_initial_power_mw * args.mpec_initial_ratio_hours for n in nodes
    }
    quad = default_quadratic_demand_curve()
    responses = []
    print(
        "Jacobi initializer: common snapshot "
        f"{args.rival_snapshot_power_mw:g} MW/node, numerical guess "
        f"{args.mpec_initial_power_mw:g} MW/node, demand={args.demand_model}, "
        f"regularization={args.dispatch_regularization:.3e}, solver_tol={args.solver_tol:.1e}"
    )
    for investor in investors:
        rival_power, rival_energy = separate_rival_capacities(
            snapshot, cfg, nodes, investor.investor_id
        )
        response = solve_best_response(
            data,
            quad,
            cfg,
            investor,
            rival_power,
            rival_energy,
            {n: snapshot.x_power[investor.investor_id, n] for n in nodes},
            {n: snapshot.x_energy[investor.investor_id, n] for n in nodes},
            initial_guess_power=guess_power,
            initial_guess_energy=guess_energy,
            tee=args.tee,
        )
        responses.append(response)
        print(
            f"  {investor.investor_id}: {response.termination}, "
            f"desired {sum(response.proposed_power.values()):.3f} MW / "
            f"{sum(response.proposed_energy.values()):.3f} MWh"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "method": "one_jacobi_sweep_then_proportional_nodal_projection",
        "data_path": str(args.data.resolve()),
        "investors": [inv.investor_id for inv in investors],
        "node_limit_mw": args.node_limit_mw,
        "rival_snapshot_power_mw_per_investor_node": args.rival_snapshot_power_mw,
        "rival_snapshot_ratio_hours": args.rival_snapshot_ratio_hours,
        "mpec_initial_power_mw_per_node": args.mpec_initial_power_mw,
        "mpec_initial_ratio_hours": args.mpec_initial_ratio_hours,
        "demand_model": args.demand_model,
        "dispatch_regularization_eur_per_mw2h": args.dispatch_regularization,
        "solver_tol": args.solver_tol,
        "price_bound_eur_per_mwh": args.price_bound_eur_per_mwh,
        "dual_bound_eur_per_mwh": args.dual_bound_eur_per_mwh,
        "max_cpu_time": args.max_cpu_time,
        "rival_representation": "separate_battery_per_investor_with_nodal_mw_mwh",
        "embedded_sparsity": "active_investor_all_nodes; rivals_only_positive_mw_or_mwh; generators_only_positive_capacity_hours",
        "fixed_demand_shedding_block_omitted": not use_demand_curve,
        "capacity_cleanup_tol_mw_mwh": args.capacity_cleanup_tol,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    response_summary = {
        response.investor_id: {
            "termination": response.termination,
            "solve_seconds": response.solve_seconds,
            "desired_power_mw": sum(response.proposed_power.values()),
            "desired_energy_mwh": sum(response.proposed_energy.values()),
            "optimistic_mpec_profit_eur_per_day": response.optimistic_mpec_profit_eur_per_day,
            "strong_duality_gap": response.strong_duality_gap,
        }
        for response in responses
    }
    if not all(response.ok for response in responses):
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "checkpoint_created": False,
                    "reason": "at least one Jacobi best response was not optimal",
                    "responses": response_summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("No projected checkpoint was created because at least one response failed.")
        return 1

    raw_rows: list[dict] = []
    projected_rows: list[dict] = []
    projected_power: dict[tuple[str, str], float] = {}
    projected_energy: dict[tuple[str, str], float] = {}
    node_summary: dict[str, dict] = {}
    for n in nodes:
        desired_total = sum(response.proposed_power[n] for response in responses)
        scale = min(1.0, args.node_limit_mw / desired_total) if desired_total > 0.0 else 1.0
        projected_total = 0.0
        for response in responses:
            investor_id = response.investor_id
            desired_power = response.proposed_power[n]
            desired_energy = response.proposed_energy[n]
            power = scale * desired_power
            energy = scale * desired_energy
            power, energy = clean_capacity_pair(
                power, energy, args.capacity_cleanup_tol
            )
            projected_power[investor_id, n] = power
            projected_energy[investor_id, n] = energy
            projected_total += power
            raw_rows.append(
                {
                    "investor": investor_id,
                    "node": n,
                    "desired_power_mw": desired_power,
                    "desired_energy_mwh": desired_energy,
                    "desired_ratio_hours": desired_energy / desired_power if desired_power > 1e-9 else 0.0,
                    "private_headroom_limit_mw": response.private_headroom_limit_mw[n],
                    "access_shadow_price_eur_per_mw_day": response.access_shadow_price_eur_per_mw_day[n],
                }
            )
            projected_rows.append(
                {
                    "investor": investor_id,
                    "node": n,
                    "projection_scale": scale,
                    "projected_power_mw": power,
                    "projected_energy_mwh": energy,
                    "projected_ratio_hours": energy / power if power > 1e-9 else 0.0,
                }
            )
        node_summary[n] = {
            "desired_total_power_mw": desired_total,
            "projection_scale": scale,
            "projected_total_power_mw": projected_total,
            "limit_mw": args.node_limit_mw,
        }

    _write_csv(
        output_dir / "raw_desired_capacities.csv",
        [
            "investor",
            "node",
            "desired_power_mw",
            "desired_energy_mwh",
            "desired_ratio_hours",
            "private_headroom_limit_mw",
            "access_shadow_price_eur_per_mw_day",
        ],
        raw_rows,
    )
    _write_csv(
        output_dir / "projected_initial_capacities.csv",
        [
            "investor",
            "node",
            "projection_scale",
            "projected_power_mw",
            "projected_energy_mwh",
            "projected_ratio_hours",
        ],
        projected_rows,
    )

    checkpoint = {
        "status": "projected Jacobi initializer; ready for Gauss-Seidel resume",
        "converged": False,
        "iteration": 0,
        "starting_iteration": 0,
        "resume_from": None,
        "update_rule": "jacobi_initializer",
        "damping": DEFAULT_DAMPING,
        "price_bound_eur_per_mwh": args.price_bound_eur_per_mwh,
        "dual_bound_eur_per_mwh": args.dual_bound_eur_per_mwh,
        "capacity_cleanup_tol_mw_mwh": args.capacity_cleanup_tol,
        "node_limit_mw": args.node_limit_mw,
        "node_total_power_mw": {
            n: sum(projected_power[inv.investor_id, n] for inv in investors) for n in nodes
        },
        "node_excess_mw": {
            n: sum(projected_power[inv.investor_id, n] for inv in investors) - args.node_limit_mw
            for n in nodes
        },
        "max_node_overload_mw": max(
            max(
                0.0,
                sum(projected_power[inv.investor_id, n] for inv in investors)
                - args.node_limit_mw,
            )
            for n in nodes
        ),
        "x_power_mw": {
            f"{investor}|{node}": value
            for (investor, node), value in projected_power.items()
        },
        "x_energy_mwh": {
            f"{investor}|{node}": value
            for (investor, node), value in projected_energy.items()
        },
    }
    (output_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_created": True,
                "interpretation": "feasible initialization heuristic, not an equilibrium",
                "responses": response_summary,
                "nodes": node_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for response in responses:
        export_solution(
            response.model,
            output_dir / f"investor_{response.investor_id}",
            "ok",
            response.termination,
            None,
        )

    print(f"Projected initializer written to {output_dir}")
    print(
        "Resume with: python model/epec_diagonalization.py "
        f"--resume-from \"{output_dir}\" --output-dir <new-seidel-output-dir>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
