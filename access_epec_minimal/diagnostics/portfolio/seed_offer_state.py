"""Throwaway: build an offer-seeded resume directory from a finished run.

Copies the source run dir, re-solves I3/I4 zero-proximal offer BRs at its
final state (capacities/storage frozen), and overwrites their rows in
final_generation_offers.csv with the proposals. The Jacobi loop can then be
resumed from a state where the offer channel is active.

Usage: python seed_offer_state.py <source_run_dir> <seeded_dir> [--blend 1.0]
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

MODEL = Path(r"d:/Alexander/Studium/EEG/Complementarity Modelling/BESS_EPEC/access_epec_minimal/model")
sys.path.insert(0, str(MODEL))

from investors import split_portfolio_investors
from primal_market_clearing_model import load_market_data
from run_portfolio_bid_game import (
    PortfolioGameConfig,
    load_generation_offers,
    solve_portfolio_response,
    _generation_offer_rows,
)
from run_hourly_bid_game import load_capacities, load_prices, _write_rows


parser = argparse.ArgumentParser(
    description="Seed renewable offers from zero-proximal best responses."
)
parser.add_argument("source_run_dir", type=Path)
parser.add_argument("seeded_dir", type=Path)
parser.add_argument(
    "--blend",
    type=float,
    default=1.0,
    help="Fraction of the zero-proximal offer move to apply, in (0, 1].",
)
args = parser.parse_args()
if not 0.0 < args.blend <= 1.0:
    raise SystemExit("--blend must be in (0, 1].")

source = args.source_run_dir.resolve()
target = args.seeded_dir.resolve()
if target.exists():
    raise SystemExit(f"target exists: {target}")
shutil.copytree(source, target)

run_config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
data = load_market_data(MODEL / "input" / "market_data_strategic_generation.json")
investors = split_portfolio_investors(
    data, include_i2=bool(run_config.get("include_i2", False))
)
power, energy = load_capacities(source / "final_capacities.csv", data, investors)
charge, discharge = load_prices(source / "final_hourly_bids.csv", data, investors)
generation = load_generation_offers(source / "final_generation_offers.csv", data, investors)

config = PortfolioGameConfig(
    investors=investors,
    capacity_fixed=True,
    strategic_generation=True,
    strategic_storage=False,
    proximal_generation_penalty=0.0,
    parallel_workers=1,
)

seeded = dict(generation)
seed_report: dict[str, object] = {"blend": args.blend, "investors": {}}
for inv_id in ("I3", "I4"):
    investor = next(i for i in investors if i.investor_id == inv_id)
    response = solve_portfolio_response(
        data, config, investor, power, energy, charge, discharge, generation)
    if not response.optimal:
        raise SystemExit(f"{inv_id} BR not optimal: {response.termination}")
    for (g, t), v in response.generation_offer.items():
        old = generation[inv_id, g, t]
        seeded[inv_id, g, t] = old + args.blend * (v - old)
    dev = max(abs(response.generation_offer[g, t] - generation[inv_id, g, t])
              for (i2, g, t) in generation if i2 == inv_id)
    applied_dev = args.blend * dev
    seed_report["investors"][inv_id] = {
        "zero_proximal_max_move_eur_per_mwh": dev,
        "applied_max_move_eur_per_mwh": applied_dev,
        "claimed_profit_eur_per_day": response.profit_eur_per_day,
    }
    print(
        f"{inv_id}: BR optimal, zero-prox move {dev:.2f} EUR/MWh, "
        f"applied {applied_dev:.2f}, claimed {response.profit_eur_per_day:,.1f}"
    )

_write_rows(target / "final_generation_offers.csv",
            _generation_offer_rows(None, data, investors, seeded))
with (target / "offer_seed_config.json").open("w", encoding="utf-8") as handle:
    json.dump(seed_report, handle, indent=2)
print(f"seeded state written to {target}")
