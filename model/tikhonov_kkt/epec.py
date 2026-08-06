"""Unified finite-gamma EPEC runner for capacity and strategic operation.

Both modes use exact Tikhonov strong duality and support simultaneous
Gauss--Jacobi best responses in multiple worker processes.  Capacity mode
chooses only BESS MW/MWh. Strategic-operation mode additionally chooses hourly
charge/discharge quantities and, by default, two-sided bid prices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import jacobi_epec as capacity_driver
    from .common import DEFAULT_DATA_PATH, MODEL_DIR
except ImportError:
    import jacobi_epec as capacity_driver
    from common import DEFAULT_DATA_PATH, MODEL_DIR

from epec_strategic_operation_diagonalization import main as strategic_main


DEFAULT_GAMMA = 1.0e-3
DEFAULT_PARALLEL_WORKERS = 2
DEFAULT_MAX_ITERS = 40


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-gamma Tikhonov EPEC in capacity or strategic-operation mode."
    )
    parser.add_argument(
        "--mode",
        choices=["capacity", "strategic-operation"],
        default="capacity",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument(
        "--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS
    )
    parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument("--node-limit-mw", type=float, default=200.0)
    parser.add_argument("--max-cpu-time", type=float, default=300.0)
    parser.add_argument("--solver-tol", type=float, default=1.0e-6)
    parser.add_argument("--price-bound-eur-per-mwh", type=float, default=500.0)
    parser.add_argument("--dual-bound-eur-per-mwh", type=float, default=10_000.0)
    parser.add_argument(
        "--strategic-bid-prices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strategic mode: choose two-sided bid prices as well as quantities.",
    )
    parser.add_argument(
        "--bid-price-bound-eur-per-mwh", type=float, default=500.0
    )
    parser.add_argument(
        "--strategic-epsilon-penalty", type=float, default=0.0
    )
    parser.add_argument(
        "--proximal-penalty-eur-per-mw2-day", type=float, default=0.0
    )
    parser.add_argument(
        "--proximal-penalty-step-eur-per-mw2-day",
        type=float,
        default=1.0,
        help=(
            "Strategic mode staircase increment; zero disables the default "
            "staircase. A positive fixed proximal penalty takes precedence."
        ),
    )
    parser.add_argument(
        "--proximal-penalty-zero-iters",
        type=int,
        default=10,
        help="Strategic mode initial zero-penalty iterations.",
    )
    parser.add_argument(
        "--proximal-penalty-step-iters",
        type=int,
        default=5,
        help="Strategic mode iterations per staircase coefficient.",
    )
    initializer_group = parser.add_mutually_exclusive_group()
    initializer_group.add_argument(
        "--skip-jacobi-initializer",
        dest="skip_jacobi_initializer",
        action="store_true",
        help=(
            "Strategic mode: start directly from the zero-capacity economic "
            "snapshot (the default)."
        ),
    )
    initializer_group.add_argument(
        "--use-jacobi-initializer",
        dest="skip_jacobi_initializer",
        action="store_false",
        help="Strategic mode: run the extra projected best-response initializer.",
    )
    parser.set_defaults(skip_jacobi_initializer=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-stage-number", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> None:
    if args.gamma <= 0.0:
        raise SystemExit("--gamma must be positive.")
    if args.max_iters <= 0:
        raise SystemExit("--max-iters must be positive.")
    if args.parallel_workers <= 0:
        raise SystemExit("--parallel-workers must be positive.")
    if not 0.0 < args.damping <= 1.0:
        raise SystemExit("--damping must be in (0, 1].")
    if args.tee and args.parallel_workers > 1:
        raise SystemExit("--tee is not supported when parallel workers exceed one.")
    if args.resume_stage_number <= 0:
        raise SystemExit("--resume-stage-number must be positive.")
    if (
        args.proximal_penalty_eur_per_mw2_day < 0.0
        or args.proximal_penalty_step_eur_per_mw2_day < 0.0
    ):
        raise SystemExit("Proximal penalty coefficients must be non-negative.")
    if args.proximal_penalty_zero_iters < 0:
        raise SystemExit("--proximal-penalty-zero-iters must be non-negative.")
    if args.proximal_penalty_step_iters <= 0:
        raise SystemExit("--proximal-penalty-step-iters must be positive.")


def _run_capacity(args: argparse.Namespace) -> int:
    if args.no_export:
        raise SystemExit("Capacity mode does not support --no-export.")
    capacity_driver.DATA_PATH = args.data
    capacity_driver.OUTPUT_DIR = args.output_dir or (
        MODEL_DIR / "output" / "tikhonov_epec_capacity"
    )
    capacity_driver.RESUME_FROM = args.resume_from
    capacity_driver.RESUME_STAGE_NUMBER = args.resume_stage_number
    capacity_driver.REGULARIZATION_STAGES = ((args.gamma, args.max_iters),)
    capacity_driver.NODE_LIMIT_MW = args.node_limit_mw
    capacity_driver.DAMPING = args.damping
    capacity_driver.PRICE_BOUND_EUR_PER_MWH = args.price_bound_eur_per_mwh
    capacity_driver.OTHER_DUAL_BOUND = args.dual_bound_eur_per_mwh
    capacity_driver.SOLVER_TOL = args.solver_tol
    capacity_driver.MAX_CPU_TIME_SECONDS_PER_INVESTOR = args.max_cpu_time
    capacity_driver.PARALLEL_WORKERS = args.parallel_workers
    capacity_driver.TEE = args.tee
    capacity_driver.PROXIMAL_PENALTY_EUR_PER_MW2_DAY = (
        args.proximal_penalty_eur_per_mw2_day
    )
    capacity_driver.PROXIMAL_PENALTY_STEP_EUR_PER_MW2_DAY = (
        0.0
        if args.proximal_penalty_eur_per_mw2_day > 0.0
        else args.proximal_penalty_step_eur_per_mw2_day
    )
    capacity_driver.PROXIMAL_PENALTY_INITIAL_ZERO_ITERATIONS = (
        args.proximal_penalty_zero_iters
    )
    capacity_driver.PROXIMAL_PENALTY_STEP_ITERATIONS = (
        args.proximal_penalty_step_iters
    )
    return capacity_driver.main()


def _run_strategic_operation(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or (
        MODEL_DIR / "output" / "tikhonov_epec_strategic_operation"
    )
    staircase_step = (
        0.0
        if args.proximal_penalty_eur_per_mw2_day > 0.0
        else args.proximal_penalty_step_eur_per_mw2_day
    )
    strategic_argv = [
        "--data",
        str(args.data),
        "--update-rule",
        "jacobi",
        "--parallel-workers",
        str(args.parallel_workers),
        "--damping",
        str(args.damping),
        "--node-limit-mw",
        str(args.node_limit_mw),
        "--max-iters",
        str(args.max_iters),
        "--max-cpu-time",
        str(args.max_cpu_time),
        "--solver-tol",
        str(args.solver_tol),
        "--price-bound-eur-per-mwh",
        str(args.price_bound_eur_per_mwh),
        "--dual-bound-eur-per-mwh",
        str(args.dual_bound_eur_per_mwh),
        "--bid-price-bound-eur-per-mwh",
        str(args.bid_price_bound_eur_per_mwh),
        "--strategic-epsilon-penalty",
        str(args.strategic_epsilon_penalty),
        "--proximal-penalty-eur-per-mw2-day",
        str(args.proximal_penalty_eur_per_mw2_day),
        "--proximal-penalty-step-eur-per-mw2-day",
        str(staircase_step),
        "--proximal-penalty-zero-iters",
        str(args.proximal_penalty_zero_iters),
        "--proximal-penalty-step-iters",
        str(args.proximal_penalty_step_iters),
        "--lower-level-optimality",
        "tikhonov-strong-duality",
        "--dual-tikhonov-gamma",
        str(args.gamma),
        "--output-dir",
        str(output_dir),
    ]
    if args.strategic_bid_prices:
        strategic_argv.append("--strategic-bid-prices")
    if args.skip_jacobi_initializer:
        strategic_argv.append("--skip-jacobi-initializer")
    if args.resume_from is not None:
        strategic_argv.extend(("--resume-from", str(args.resume_from)))
    if args.tee:
        strategic_argv.append("--tee")
    if args.no_export:
        strategic_argv.append("--no-export")
    return strategic_main(strategic_argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate(args)
    print(
        f"Tikhonov EPEC mode={args.mode}, gamma={args.gamma:.3e}, "
        f"workers={args.parallel_workers}, max_iters={args.max_iters}",
        flush=True,
    )
    if args.mode == "capacity":
        return _run_capacity(args)
    return _run_strategic_operation(args)


if __name__ == "__main__":
    raise SystemExit(main())
