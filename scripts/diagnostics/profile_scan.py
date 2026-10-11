"""Unique-price profit of one investor along scaled copies of a capacity profile.

Each point re-clears the market at fixed capacities with balancing eps > 0
(unique prices, solution_space.unique_price_reclear) and reports the
investor's settlement profit. Costs come from --costs; the capacities are
taken from --caps (the --source investor's entry, default --investor).
--pv-offer sets the offer price of both PV plants (negative = costly curtailment).

Run with the repo root as first argument, e.g.
python scripts/diagnostics/profile_scan.py . --costs model/input/capacities_capex155.json
    --investor I4 --caps workflow/capex_2026-10-11/I4_capex120_relaxed.json
    --pv-offer -20 --factors 0 0.999 1 1.001
"""
import argparse
import copy
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--data", type=Path, default=Path("model/input/market_data.json"))
parser.add_argument("--costs", type=Path, required=True)
parser.add_argument("--investor", required=True)
parser.add_argument("--caps", type=Path, required=True)
parser.add_argument("--source")
parser.add_argument("--eps", type=float, default=1e-4)
parser.add_argument("--pv-offer", type=float)
parser.add_argument("--factors", type=float, nargs="+", default=[0.0, 0.999, 1.0, 1.001])
args = parser.parse_args()
sys.path.insert(0, str(args.root / "model"))
from prepare_input import load_case
from solution_space import unique_price_reclear, settlement_profit

data, profile = load_case(args.root / args.data, args.root / args.costs)
if args.pv_offer is not None:
    for g in ("RES_PV_N6", "RES_PV_N8"):
        data["generation_cost"][g] = args.pv_offer
caps = json.loads((args.root / args.caps).read_text())[args.source or args.investor]
for factor in args.factors:
    q = copy.deepcopy(profile)
    for key in ("power_mw", "energy_mwh"):
        q[args.investor][key] = {n: v * factor for n, v in caps[key].items()}
    p, price = unique_price_reclear(data, q, args.eps)
    print(f"{args.investor} x{factor:<6} {sum(q[args.investor]['power_mw'].values()):8.3f} MW "
          f"{sum(q[args.investor]['energy_mwh'].values()):8.3f} MWh -> profit "
          f"{settlement_profit(p, args.investor, price):9.0f} EUR/day | N6 price h13 {price['N6', 13]:7.2f}", flush=True)
