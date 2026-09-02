"""Throwaway audit: run A final state, zero-proximal best responses.

For each investor, re-solve the portfolio BR at the final Jacobi state with
proximal_generation_penalty = 0, sequentially in this process. Report raw
deviation, claimed profit, and exact-recleared profit of the proposal.
"""
import csv
import sys
from dataclasses import replace
from pathlib import Path

MODEL = Path(r"d:/Alexander/Studium/EEG/Complementarity Modelling/BESS_EPEC/access_epec_minimal/model")
sys.path.insert(0, str(MODEL))

import exact_reclear
from investors import split_portfolio_investors
from primal_market_clearing_model import load_market_data
from run_portfolio_bid_game import (
    PortfolioGameConfig,
    load_generation_offers,
    solve_portfolio_response,
    truthful_generation_offers,
)
from run_hourly_bid_game import load_capacities, load_prices

RUN = MODEL / "output" / "portfolio_gen_only_40_sweeps"

data = load_market_data(MODEL / "input" / "market_data_strategic_generation.json")
investors = split_portfolio_investors(data)
ids = [i.investor_id for i in investors]
power, energy = load_capacities(RUN / "final_capacities.csv", data, investors)
charge, discharge = load_prices(RUN / "final_hourly_bids.csv", data, investors)
generation = load_generation_offers(RUN / "final_generation_offers.csv", data, investors)

config = PortfolioGameConfig(
    investors=investors,
    capacity_fixed=True,
    strategic_generation=True,
    strategic_storage=False,
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
print("recleared profits at final state:", {k: round(v, 1) for k, v in state_profit.items()})

for investor in investors:
    response = solve_portfolio_response(
        data, config, investor, power, energy, charge, discharge, generation)
    print(f"\n[{investor.investor_id}] termination={response.termination} "
          f"optimal={response.optimal} seconds={response.seconds:.1f}")
    if not response.optimal:
        continue
    deviation = max(
        (abs(response.generation_offer[g, t] - generation[investor.investor_id, g, t])
         for (inv, g, t) in generation if inv == investor.investor_id),
        default=0.0,
    )
    print(f"  raw zero-proximal offer deviation: {deviation:.4f} EUR/MWh")
    print(f"  claimed profit: {response.profit_eur_per_day:,.1f}")
    proposed = dict(generation)
    for (g, t), v in response.generation_offer.items():
        proposed[investor.investor_id, g, t] = v
    dev_profit = reclear_profit(proposed)
    own = investor.investor_id
    print(f"  recleared profit of deviation: {dev_profit[own]:,.1f} "
          f"(vs state {state_profit[own]:,.1f}; exact gain {dev_profit[own] - state_profit[own]:+,.1f})")
