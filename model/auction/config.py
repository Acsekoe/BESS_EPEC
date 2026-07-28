"""Command-line configuration for the clean auction EPEC workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


AUCTION_DIR = Path(__file__).resolve().parent
MODEL_DIR = AUCTION_DIR.parent
DEFAULT_IEEE9_DATA_PATH = MODEL_DIR / "data" / "processed" / "market_data_IEEE_9Bus_congestion.json"
DEFAULT_BALANCED_BIDS_PATH = AUCTION_DIR / "data" / "auction_mpec_cases" / "balanced_competition.json"
DEFAULT_SINGLE_MPEC_OUTPUT_PATH = AUCTION_DIR / "output" / "single_investor_auction_mpec" / "summary.json"
DEFAULT_GAUSS_SEIDEL_OUTPUT_DIR = AUCTION_DIR / "output" / "gauss_seidel" / "portfolio4"

DEFAULT_INVESTOR_ORDER = ("I1", "I2", "I3", "I4")
DEFAULT_NODE_LIMIT_MW = 100.0
DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY = 100.0
DEFAULT_MAX_CPU_TIME_SECONDS = 120.0
DEFAULT_PRICE_BOUND_EUR_PER_MWH = 500.0
DEFAULT_DUAL_BOUND_EUR_PER_MWH = 10_000.0
DEFAULT_CAPACITY_CLEANUP_TOL_MW = 1.0e-4
DEFAULT_SOLVER_TOL = 1.0e-4
DEFAULT_BID_PRICE_TICK_EUR_PER_MW_DAY = 0.0
DEFAULT_TIE_BREAK_EPSILON_EUR_PER_MW_DAY = 0.001
DEFAULT_PAYMENT_RULE = "uniform"
DEFAULT_AUCTION_QUADRATIC_EPSILON_EUR_PER_MW2_DAY = 1.0e-3
DEFAULT_CLEARING_PRICE_TOLERANCE_EUR_PER_MW_DAY = 0.05


@dataclass(frozen=True)
class SingleMpecCliConfig:
    data_path: Path
    active_investor: str
    active_node: str | None
    rival_bids_path: Path | None
    node_limit_mw: float
    max_bid_price_eur_per_mw_day: float
    initial_bid_quantity_mw: float
    initial_bid_price_eur_per_mw_day: float
    initial_duration_hours: float
    max_cpu_time: float
    price_bound_eur_per_mwh: float
    dual_bound_eur_per_mwh: float
    capacity_cleanup_tol_mw: float
    solver_tol: float
    bid_price_tick_eur_per_mw_day: float
    tie_break_epsilon_eur_per_mw_day: float
    payment_rule: str
    auction_quadratic_epsilon_eur_per_mw2_day: float
    output_path: Path
    tee: bool


@dataclass(frozen=True)
class GaussSeidelConfig:
    data_path: Path
    initial_bids_path: Path
    output_dir: Path
    update_rule: str = "seidel"
    investor_order: tuple[str, ...] = DEFAULT_INVESTOR_ORDER
    active_nodes: tuple[str, ...] | None = None
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW
    max_bid_price_eur_per_mw_day: float = DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY
    max_iterations: int = 20
    max_cpu_time: float = DEFAULT_MAX_CPU_TIME_SECONDS
    price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH
    dual_bound_eur_per_mwh: float = DEFAULT_DUAL_BOUND_EUR_PER_MWH
    capacity_cleanup_tol_mw: float = DEFAULT_CAPACITY_CLEANUP_TOL_MW
    solver_tol: float = DEFAULT_SOLVER_TOL
    bid_price_tick_eur_per_mw_day: float = DEFAULT_BID_PRICE_TICK_EUR_PER_MW_DAY
    tie_break_epsilon_eur_per_mw_day: float = DEFAULT_TIE_BREAK_EPSILON_EUR_PER_MW_DAY
    payment_rule: str = DEFAULT_PAYMENT_RULE
    auction_quadratic_epsilon_eur_per_mw2_day: float = (
        DEFAULT_AUCTION_QUADRATIC_EPSILON_EUR_PER_MW2_DAY
    )
    clearing_price_tolerance_eur_per_mw_day: float = (
        DEFAULT_CLEARING_PRICE_TOLERANCE_EUR_PER_MW_DAY
    )
    zero_bid_numerical_quantity_mw: float = 70.0
    zero_bid_numerical_price_eur_per_mw_day: float = 0.01
    quantity_tolerance_mw: float = 0.05
    award_tolerance_mw: float = 0.05
    price_tolerance_eur_per_mw_day: float = 0.001
    duration_tolerance_hours: float = 0.01
    max_consecutive_failures: int = 3
    resume_path: Path | None = None
    tee: bool = False


def _add_common_numerical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-cpu-time", type=float, default=DEFAULT_MAX_CPU_TIME_SECONDS)
    parser.add_argument(
        "--price-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_PRICE_BOUND_EUR_PER_MWH,
        help="Absolute bound for lambda and lambda_system (default: 500).",
    )
    parser.add_argument(
        "--dual-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_DUAL_BOUND_EUR_PER_MWH,
        help="Absolute bound for every other embedded follower dual (default: 10000).",
    )
    parser.add_argument(
        "--capacity-cleanup-tol",
        type=float,
        default=DEFAULT_CAPACITY_CLEANUP_TOL_MW,
        help="Bid quantities at or below this tolerance are normalized to zero.",
    )
    parser.add_argument("--solver-tol", type=float, default=DEFAULT_SOLVER_TOL)
    parser.add_argument(
        "--bid-price-tick",
        type=float,
        default=DEFAULT_BID_PRICE_TICK_EUR_PER_MW_DAY,
        help="Exact external bid-price grid; use 0.01 for tick enumeration, zero for continuous bids.",
    )
    parser.add_argument(
        "--tie-break-epsilon",
        type=float,
        default=DEFAULT_TIE_BREAK_EPSILON_EUR_PER_MW_DAY,
        help="Deterministic merit offset for equal raw bids; it is not included in payment.",
    )
    parser.add_argument(
        "--payment-rule",
        choices=("uniform", "pay_as_bid"),
        default=DEFAULT_PAYMENT_RULE,
        help=(
            "uniform: every award pays the nodal clearing price (auction capacity dual); "
            "pay_as_bid: legacy rule where every award pays its own raw bid."
        ),
    )
    parser.add_argument(
        "--auction-quadratic-epsilon",
        type=float,
        default=DEFAULT_AUCTION_QUADRATIC_EPSILON_EUR_PER_MW2_DAY,
        help=(
            "Strictly concave auction regularization in EUR/MW^2/day; it makes the "
            "allocation unique and splits exact price ties pro rata. Used only with "
            "the uniform payment rule."
        ),
    )


def parse_single_mpec_cli(argv: Sequence[str] | None = None) -> SingleMpecCliConfig:
    parser = argparse.ArgumentParser(
        description="Solve one continuous strategic access-bid MPEC with auction and spot followers."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_IEEE9_DATA_PATH)
    parser.add_argument("--active-investor", choices=DEFAULT_INVESTOR_ORDER, default="I1")
    parser.add_argument(
        "--active-node",
        default=None,
        help="Restrict the strategic bid to one node; omitted enables every node.",
    )
    parser.add_argument("--rival-bids", type=Path, default=None)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument(
        "--max-bid-price",
        type=float,
        default=DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY,
    )
    parser.add_argument("--initial-bid-quantity", type=float, default=10.0)
    parser.add_argument("--initial-bid-price", type=float, default=10.0)
    parser.add_argument("--initial-duration", type=float, default=4.0)
    _add_common_numerical_arguments(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_SINGLE_MPEC_OUTPUT_PATH)
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args(argv)
    return SingleMpecCliConfig(
        data_path=args.data,
        active_investor=args.active_investor,
        active_node=args.active_node,
        rival_bids_path=args.rival_bids,
        node_limit_mw=args.node_limit_mw,
        max_bid_price_eur_per_mw_day=args.max_bid_price,
        initial_bid_quantity_mw=args.initial_bid_quantity,
        initial_bid_price_eur_per_mw_day=args.initial_bid_price,
        initial_duration_hours=args.initial_duration,
        max_cpu_time=args.max_cpu_time,
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
        capacity_cleanup_tol_mw=args.capacity_cleanup_tol,
        solver_tol=args.solver_tol,
        bid_price_tick_eur_per_mw_day=args.bid_price_tick,
        tie_break_epsilon_eur_per_mw_day=args.tie_break_epsilon,
        payment_rule=args.payment_rule,
        auction_quadratic_epsilon_eur_per_mw2_day=args.auction_quadratic_epsilon,
        output_path=args.output,
        tee=args.tee,
    )


def parse_gauss_seidel_cli(argv: Sequence[str] | None = None) -> GaussSeidelConfig:
    parser = argparse.ArgumentParser(
        description="Gauss-Jacobi or Gauss-Seidel diagonalization of the clean auction EPEC."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_IEEE9_DATA_PATH)
    parser.add_argument("--initial-bids", type=Path, default=DEFAULT_BALANCED_BIDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GAUSS_SEIDEL_OUTPUT_DIR)
    parser.add_argument(
        "--update-rule",
        choices=("jacobi", "seidel"),
        default="seidel",
        help="Jacobi applies sealed-bid best responses simultaneously; Seidel applies them immediately.",
    )
    parser.add_argument(
        "--investor-order",
        nargs="+",
        choices=DEFAULT_INVESTOR_ORDER,
        default=list(DEFAULT_INVESTOR_ORDER),
    )
    parser.add_argument("--active-nodes", nargs="+", default=None)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument(
        "--max-bid-price",
        type=float,
        default=DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY,
    )
    parser.add_argument("--max-iterations", type=int, default=20)
    _add_common_numerical_arguments(parser)
    parser.add_argument(
        "--zero-bid-numerical-quantity",
        type=float,
        default=70.0,
        help=(
            "Positive variable initialization used when an investor's economic bid quantity "
            "is zero; it does not change the saved bid state or impose a lower bound."
        ),
    )
    parser.add_argument(
        "--zero-bid-numerical-price",
        type=float,
        default=0.01,
        help=(
            "Positive price initialization used with a zero economic bid; it does not "
            "change the saved bid state or impose a lower bound."
        ),
    )
    parser.add_argument("--quantity-tol", type=float, default=0.05)
    parser.add_argument("--award-tol", type=float, default=0.05)
    parser.add_argument(
        "--clearing-price-tol",
        type=float,
        default=DEFAULT_CLEARING_PRICE_TOLERANCE_EUR_PER_MW_DAY,
        help=(
            "Uniform rule only: maximum allowed gap between each embedded nodal "
            "clearing price and the common re-clear price at payment-relevant nodes."
        ),
    )
    parser.add_argument(
        "--price-tol",
        type=float,
        default=0.001,
        help="Raw-bid convergence tolerance; keep this below --bid-price-tick in grid mode.",
    )
    parser.add_argument("--duration-tol", type=float, default=0.01)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args(argv)
    return GaussSeidelConfig(
        data_path=args.data,
        initial_bids_path=args.initial_bids,
        output_dir=args.output_dir,
        update_rule=args.update_rule,
        investor_order=tuple(args.investor_order),
        active_nodes=None if args.active_nodes is None else tuple(args.active_nodes),
        node_limit_mw=args.node_limit_mw,
        max_bid_price_eur_per_mw_day=args.max_bid_price,
        max_iterations=args.max_iterations,
        max_cpu_time=args.max_cpu_time,
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
        capacity_cleanup_tol_mw=args.capacity_cleanup_tol,
        solver_tol=args.solver_tol,
        bid_price_tick_eur_per_mw_day=args.bid_price_tick,
        tie_break_epsilon_eur_per_mw_day=args.tie_break_epsilon,
        payment_rule=args.payment_rule,
        auction_quadratic_epsilon_eur_per_mw2_day=args.auction_quadratic_epsilon,
        clearing_price_tolerance_eur_per_mw_day=args.clearing_price_tol,
        zero_bid_numerical_quantity_mw=args.zero_bid_numerical_quantity,
        zero_bid_numerical_price_eur_per_mw_day=args.zero_bid_numerical_price,
        quantity_tolerance_mw=args.quantity_tol,
        award_tolerance_mw=args.award_tol,
        price_tolerance_eur_per_mw_day=args.price_tol,
        duration_tolerance_hours=args.duration_tol,
        max_consecutive_failures=args.max_consecutive_failures,
        resume_path=args.resume,
        tee=args.tee,
    )
