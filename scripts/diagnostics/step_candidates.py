"""Step-based start candidates for the capacity MPEC (prototype; not used by model/run.py).

Best responses tend to sit at a step: the storage size at which a pocket's
curtailed zero-cost energy is absorbed (workflow/summary_2026-10-09_13-33.md).
This script evaluates the investor's exact profit (unique-price reclear, Ipopt)
at candidates around those steps, refines the best by coordinate search, and
writes it as a capacity profile for active_set_start.py --start-profile.
Heuristic: the result is a good start, not a proven best response.

1. No-storage market (active investor at zero, rivals fixed). At node n and
   hour t the absorbable energy is the curtailment of zero-cost plants whose
   node price is below 1 EUR/MWh and within 0.5 EUR/MWh of the price at n
   (same pocket). Step energy E(n) = eta * daily sum.
2. Single-node candidates: energy f * E(n) for f in FRACTIONS (FALLBACK_MWH
   where E(n) = 0) at the durations in DURATIONS; keep the best --keep per node.
3. Combine the kept candidates of every node that beats zero storage, then
   refine the best combination: move one node's MW by +-step and climb its MWh
   to the best value for that MW (steps 10, 1, 0.1, 0.01 for both). The inner
   climb follows a ridge where the energy must just fill (V6, I1): there the
   best MWh rises by only about 2 MWh per MW, so single-coordinate moves stall.

python scripts/diagnostics/step_candidates.py . --data model/input/ieee5_v6_market_data.json --profile model/input/ieee5_v6_capacities.json --investor I1 --output model/output/<new_dir>/start.json
"""
import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path

FRACTIONS = (0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
FALLBACK_MWH = (50.0, 100.0, 200.0, 400.0)
DURATIONS = (2.0, 3.0, 4.0, 6.0, 8.0)
STEPS = (10.0, 1.0, 0.1, 0.01)

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--data", type=Path, default=Path("model/input/ieee5_market_data.json"))
parser.add_argument("--profile", type=Path, default=Path("model/input/ieee5_capacities.json"))
parser.add_argument("--investor", default="I1")
parser.add_argument("--nodes", nargs="+", help="Candidate nodes (default: all)")
parser.add_argument("--balancing-eps", type=float, default=1e-4)
parser.add_argument("--keep", type=int, default=3, help="Single-node candidates kept per node")
parser.add_argument("--output", type=Path, help="Write the best capacity as a profile JSON")
args = parser.parse_args()
sys.path.insert(0, str(args.root / "model"))
import pyomo.environ as pyo
from prepare_input import load_case
from solution_space import settlement_profit, unique_price_reclear

data, profile = load_case(args.root / args.data, args.root / args.profile)
i = args.investor
nodes = args.nodes or data["nodes"]
ratio_min, ratio_max = profile[i]["ratio_min"], profile[i]["ratio_max"]
cache = {}


def with_capacity(capacity):
    trial = copy.deepcopy(profile)
    trial[i]["power_mw"] = {n: c[0] for n, c in capacity.items()}
    trial[i]["energy_mwh"] = {n: c[1] for n, c in capacity.items()}
    return trial


def profit(capacity):
    """Exact profit at {node: (MW, MWh)}, energy clipped to the duration bounds."""
    capacity = {n: (round(p, 6), e) for n, (p, e) in capacity.items() if p > 1e-6}
    capacity = {n: (p, min(max(round(e, 6), ratio_min * p), ratio_max * p)) for n, (p, e) in capacity.items()}
    key = tuple(sorted(capacity.items()))
    if key not in cache:
        try:
            market, price = unique_price_reclear(data, with_capacity(capacity), args.balancing_eps)
            cache[key] = settlement_profit(market, i, price)
        except RuntimeError as error:  # Ipopt failure: skip this candidate, report it
            print(f"    reclear failed at {show(capacity)}: {error}", flush=True)
            cache[key] = float("-inf")
    return cache[key], capacity


def show(capacity):
    return {n: (round(p, 3), round(e, 3)) for n, (p, e) in capacity.items()}


def climb_energy(value, capacity, n):
    """Best MWh at node n for its current MW, other capacities fixed."""
    for step in STEPS:
        improved = True
        while improved:
            improved = False
            p, e = capacity[n]
            for trial in ((p, e + step), (p, e - step)):
                new_value, new_capacity = profit({**capacity, n: trial})
                if new_value > value + 1e-6:
                    value, capacity, improved = new_value, new_capacity, True
                    break
    return value, capacity


start_time = time.time()

# 1. Steps from the no-storage market.
market, price = unique_price_reclear(data, with_capacity({}), args.balancing_eps)
available = {(r["generator"], r["hour"]): r["capacity_mw"] for r in data["generation_capacity"]}
plant_node = {g: n for n, plants in data["generators_at_node"].items() for g in plants}
zero_cost = [g for g in data["generators"] if data["generation_cost"][g] == 0]
step_mwh = {}
for n in nodes:
    absorbable = 0.0
    for t in data["times"]:
        for g in zero_cost:
            pocket_price = price[plant_node[g], t]
            if pocket_price < 1.0 and abs(price[n, t] - pocket_price) <= 0.5:
                absorbable += available[g, t] - pyo.value(market.generation[g, t])
    step_mwh[n] = data["eta"] * absorbable
zero, _ = profit({})
print(f"{i}: no storage {zero:.2f}; step energy per node [MWh]: "
      + ", ".join(f"{n} {e:.1f}" for n, e in step_mwh.items()), flush=True)

# 2. Single-node candidates around each step.
kept = {}
for n in nodes:
    energies = [f * step_mwh[n] for f in FRACTIONS] if step_mwh[n] > 0 else FALLBACK_MWH
    results = sorted((profit({n: (e / d, e)}) for e in energies for d in DURATIONS), key=lambda r: -r[0])
    kept[n] = [capacity for value, capacity in results[:args.keep] if value > zero]
    print(f"  {n}: best single {results[0][0]:.2f} at {show(results[0][1])}", flush=True)

# 3. Combinations of the kept candidates, then coordinate search.
options = [[{}] + [{n: c[n]} for c in kept[n]] for n in nodes if kept[n]]
combos = [profit({k: v for part in combo for k, v in part.items()}) for combo in itertools.product(*options)]
value, capacity = max(combos, key=lambda r: r[0]) if combos else (zero, {})
print(f"best of {len(combos)} combinations: {value:.2f} at {show(capacity)}", flush=True)
for n in list(capacity):
    value, capacity = climb_energy(value, capacity, n)
for step in STEPS:
    improved = True
    while improved:
        improved = False
        for n in list(capacity):
            p, e = capacity[n]
            for trial_power in (p + step, p - step):
                if trial_power <= 0:
                    continue
                new_value, new_capacity = climb_energy(*profit({**capacity, n: (trial_power, e)}), n)
                if new_value > value + 1e-6:
                    value, capacity, improved = new_value, new_capacity, True
                    break
    print(f"  refined at step {step:g}: {value:.2f} at {show(capacity)}", flush=True)
print(f"best start: {value:.2f} at {show(capacity)}; {len(cache)} exact reclears, {time.time() - start_time:.0f} s")
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(with_capacity(capacity), indent=2) + "\n", encoding="utf-8")
