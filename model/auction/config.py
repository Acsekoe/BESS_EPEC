"""Central command-line configuration for auction-MPEC workflows.

Model formulation stays in ``single_investor_auction_mpec.py`` and the
Gauss-Seidel algorithm stays in ``gauss_seidel.py``. This module owns their
paths, run defaults, argument parsers, and immutable run-configuration objects.
"""

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
DEFAULT_GAUSS_SEIDEL_OUTPUT_DIR = AUCTION_DIR / "output" / "gauss_seidel"

DEFAULT_INVESTOR_ORDER = ("I1", "I2", "I3", "I4")
DEFAULT_NODE_LIMIT_MW = 100.0
DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY = 100.0
DEFAULT_MAX_CPU_TIME_SECONDS = 60.0
DEFAULT_DUAL_BOUND_SCALE = 10.0


@dataclass(frozen=True)
class SingleMpecCliConfig:
    data_path: Path
    active_investor: str
    active_node: str | None
    rival_bids_path: Path | None
    node_limit_mw: float
    min_bid_price_eur_per_mw_day: float
    max_bid_price_eur_per_mw_day: float
    initial_bid_quantity_mw: float
    initial_bid_price_eur_per_mw_day: float
    initial_duration_hours: float
    max_cpu_time: float
    dual_bound_scale: float
    output_path: Path
    tee: bool


@dataclass(frozen=True)
class GaussSeidelConfig:
    data_path: Path
    initial_bids_path: Path
    output_dir: Path
    investor_order: tuple[str, ...] = DEFAULT_INVESTOR_ORDER
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW
    min_bid_price_eur_per_mw_day: float = 0.0
    max_bid_price_eur_per_mw_day: float = DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY
    max_iterations: int = 20
    max_cpu_time: float = DEFAULT_MAX_CPU_TIME_SECONDS
    dual_bound_scale: float = DEFAULT_DUAL_BOUND_SCALE
    damping: float = 1.0
    quantity_tolerance_mw: float = 0.05
    price_tolerance_eur_per_mw_day: float = 0.05
    duration_tolerance_hours: float = 0.01
    award_tolerance_mw: float = 0.05
    active_award_zero_tolerance_mw: float = 1.0e-4
    outside_option_tolerance_eur_per_day: float = 0.01
    max_consecutive_failures: int = 3
    use_outside_option: bool = True
    resume_path: Path | None = None
    tee: bool = False


def parse_single_mpec_cli(argv: Sequence[str] | None = None) -> SingleMpecCliConfig:
    parser = argparse.ArgumentParser(
        description="Solve one strategic investor's embedded auction-and-spot MPEC."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_IEEE9_DATA_PATH)
    parser.add_argument(
        "--active-investor",
        choices=DEFAULT_INVESTOR_ORDER,
        default="I1",
        help="Thesis investor whose bid is optimized.",
    )
    parser.add_argument(
        "--active-node",
        default=None,
        help="Restrict the strategic investor to one node; omit to open all nodes.",
    )
    parser.add_argument(
        "--rival-bids",
        type=Path,
        default=None,
        help="JSON rival-bid vector; the built-in demonstration is used when omitted.",
    )
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument(
        "--min-bid-price",
        type=float,
        default=0.0,
        help="Optional active-bid floor in EUR/MW/day; zero preserves the stated auction.",
    )
    parser.add_argument(
        "--max-bid-price",
        type=float,
        default=DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY,
        help="Upper bound on the active access bid in EUR/MW/day.",
    )
    parser.add_argument("--initial-bid-quantity", type=float, default=10.0)
    parser.add_argument("--initial-bid-price", type=float, default=10.0)
    parser.add_argument("--initial-duration", type=float, default=4.0)
    parser.add_argument("--max-cpu-time", type=float, default=DEFAULT_MAX_CPU_TIME_SECONDS)
    parser.add_argument("--dual-bound-scale", type=float, default=DEFAULT_DUAL_BOUND_SCALE)
    parser.add_argument("--output", type=Path, default=DEFAULT_SINGLE_MPEC_OUTPUT_PATH)
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args(argv)
    return SingleMpecCliConfig(
        data_path=args.data,
        active_investor=args.active_investor,
        active_node=args.active_node,
        rival_bids_path=args.rival_bids,
        node_limit_mw=args.node_limit_mw,
        min_bid_price_eur_per_mw_day=args.min_bid_price,
        max_bid_price_eur_per_mw_day=args.max_bid_price,
        initial_bid_quantity_mw=args.initial_bid_quantity,
        initial_bid_price_eur_per_mw_day=args.initial_bid_price,
        initial_duration_hours=args.initial_duration,
        max_cpu_time=args.max_cpu_time,
        dual_bound_scale=args.dual_bound_scale,
        output_path=args.output,
        tee=args.tee,
    )


def parse_gauss_seidel_cli(argv: Sequence[str] | None = None) -> GaussSeidelConfig:
    parser = argparse.ArgumentParser(
        description="Gauss-Seidel diagonalization of the four-investor auction MPECs."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_IEEE9_DATA_PATH)
    parser.add_argument("--initial-bids", type=Path, default=DEFAULT_BALANCED_BIDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GAUSS_SEIDEL_OUTPUT_DIR)
    parser.add_argument(
        "--investor-order",
        nargs="+",
        choices=DEFAULT_INVESTOR_ORDER,
        default=list(DEFAULT_INVESTOR_ORDER),
    )
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--min-bid-price", type=float, default=0.0)
    parser.add_argument("--max-bid-price", type=float, default=DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-cpu-time", type=float, default=DEFAULT_MAX_CPU_TIME_SECONDS)
    parser.add_argument("--dual-bound-scale", type=float, default=DEFAULT_DUAL_BOUND_SCALE)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--quantity-tol", type=float, default=0.05)
    parser.add_argument("--price-tol", type=float, default=0.05)
    parser.add_argument("--duration-tol", type=float, default=0.01)
    parser.add_argument("--award-tol", type=float, default=0.05)
    parser.add_argument("--active-award-zero-tol", type=float, default=1.0e-4)
    parser.add_argument("--outside-option-tol", type=float, default=0.01)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument(
        "--skip-outside-option",
        action="store_true",
        help="Skip the explicit zero-bid comparison; intended only for diagnostics.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args(argv)
    return GaussSeidelConfig(
        data_path=args.data,
        initial_bids_path=args.initial_bids,
        output_dir=args.output_dir,
        investor_order=tuple(args.investor_order),
        node_limit_mw=args.node_limit_mw,
        min_bid_price_eur_per_mw_day=args.min_bid_price,
        max_bid_price_eur_per_mw_day=args.max_bid_price,
        max_iterations=args.max_iterations,
        max_cpu_time=args.max_cpu_time,
        dual_bound_scale=args.dual_bound_scale,
        damping=args.damping,
        quantity_tolerance_mw=args.quantity_tol,
        price_tolerance_eur_per_mw_day=args.price_tol,
        duration_tolerance_hours=args.duration_tol,
        award_tolerance_mw=args.award_tol,
        active_award_zero_tolerance_mw=args.active_award_zero_tol,
        outside_option_tolerance_eur_per_day=args.outside_option_tol,
        max_consecutive_failures=args.max_consecutive_failures,
        use_outside_option=not args.skip_outside_option,
        resume_path=args.resume,
        tee=args.tee,
    )
