"""Throwaway: is truthful-offers a real feature of run E or a local artifact?

Re-solve I4's (and I3's) BR at run E's final state with proximal_generation_
penalty = 0 from two starts: (a) the run's own state (truthful offers),
(b) offers seeded with the run-A midday markup pattern. Validate every
proposal by exact reclear.
"""
import sys
from pathlib import Path

MODEL = Path(r"d:/Alexander/Studium/EEG/Complementarity Modelling/BESS_EPEC/access_epec_minimal/model")
sys.path.insert(0, str(MODEL))

import exact_reclear
from dataclasses import replace as dc_replace
from investors import split_portfolio_investors
from primal_market_clearing_model import load_market_data
from run_portfolio_bid_game import (
    PortfolioGameConfig,
    load_generation_offers,
    solve_portfolio_response,
    strategic_generation_pairs,
)
from run_hourly_bid_game import load_capacities, load_prices

RUN = MODEL / "output" / "portfolio_full_epec_damp015_40_sweeps"
RUN_A = MODEL / "output" / "portfolio_gen_only_40_sweeps"

data = load_market_data(MODEL / "input" / "market_data_strategic_generation.json")
investors = split_portfolio_investors(data)
ids = [i.investor_id for i in investors]
power, energy = load_capacities(RUN / "final_capacities.csv", data, investors)
charge, discharge = load_prices(RUN / "final_hourly_bids.csv", data, investors)
generation = load_generation_offers(RUN / "final_generation_offers.csv", data, investors)
gen_a = load_generation_offers(RUN_A / "final_generation_offers.csv", data, investors)

config = PortfolioGameConfig(
    investors=investors,
    capacity_fixed=True,          # audit offers at frozen capacities
    strategic_generation=True,
    strategic_storage=False,      # freeze storage bids at E state
    proximal_generation_penalty=0.0,
    parallel_workers=1,
)

def reclear_profit(gen_offers):
    flat = {(g, int(t)): v for (_, g, t), v in gen_offers.items()}
    res = exact_reclear.clear(
        data, investor_ids=ids, power=power, energy=energy,
        charge_bid=charge, discharge_offer=discharge, generation_offer=flat)
    return {r["investor"]: float(r["profit_eur_per_day"])
            for r in exact_reclear.profit_decomposition(res, investors)}

state_profit = reclear_profit(generation)
print("recleared at E final state:", {k: round(v, 1) for k, v in state_profit.items()})

for inv_id in ("I3", "I4"):
    investor = next(i for i in investors if i.investor_id == inv_id)
    for label, snapshot in (("truthful-start", generation),
                            ("runA-seeded", {**generation, **{k: v for k, v in gen_a.items() if k[0] == inv_id}})):
        response = solve_portfolio_response(
            data, config, investor, power, energy, charge, discharge, snapshot)
        if not response.optimal:
            print(f"[{inv_id} {label}] NOT OPTIMAL: {response.termination}")
            continue
        proposed = dict(generation)
        for (g, t), v in response.generation_offer.items():
            proposed[inv_id, g, t] = v
        dev = max(abs(response.generation_offer[g, t] - generation[inv_id, g, t])
                  for (i2, g, t) in generation if i2 == inv_id)
        rp = reclear_profit(proposed)
        print(f"[{inv_id} {label}] optimal, offer dev {dev:8.2f}, claimed {response.profit_eur_per_day:12,.1f}, "
              f"recleared {rp[inv_id]:12,.1f} (state {state_profit[inv_id]:12,.1f}, gain {rp[inv_id]-state_profit[inv_id]:+9.1f})")
