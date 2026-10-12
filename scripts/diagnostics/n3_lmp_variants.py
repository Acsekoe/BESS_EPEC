"""Do N3 and N9 differ? LMPs and storage profit under edited line limits and loads.

N3 is radial (only line L93 to N9), so LMP(N3) - LMP(N9) is nonzero only in
hours where L93 binds. Each variant edits market_data in memory (nothing in
model/input changes) and reclears with balancing eps (unique prices):
  - without storage: hours in which the N3 and N9 prices differ;
  - with --mw MW / 4*MW MWh of I1 storage at N3 versus at N9: I1's profit.
A battery larger than the binding line never uses its full capacity and then
earns exactly zero operating margin; keep --mw below the L93 limit.

Variants are JSON: {"limits": {"L93": 40}, "move_load": {"N5": ["N3", 0.1, [17, 18]]}}
moves the share of N5's demand to N3 in the listed hours (all hours if omitted).

python scripts/diagnostics/n3_lmp_variants.py . --mw 20 base
    '{"move_load": {"N5": ["N3", 0.1, [17,18,19,20,21,22]]}, "limits": {"L93": 40}}'
"""
import argparse
import copy
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("variants", nargs="+", help="'base' or a JSON variant")
parser.add_argument("--data", type=Path, default=Path("model/input/market_data.json"))
parser.add_argument("--costs", type=Path, default=Path("model/input/capacities_capex120_curtail.json"))
parser.add_argument("--eps", type=float, default=1e-4)
parser.add_argument("--mw", type=float, default=20.0, help="I1 test storage power; energy is 4 * MW")
args = parser.parse_args()
sys.path.insert(0, str(args.root / "model"))
import pyomo.environ as pyo
from prepare_input import load_case
from solution_space import settlement_profit, unique_price_reclear

base_data, base_profile = load_case(args.root / args.data, args.root / args.costs)


def edited(spec):
    data = copy.deepcopy(base_data)
    for line, limit in spec.get("limits", {}).items():
        data["line_limit"][line] = limit
    for source, (target, share, *hours) in spec.get("move_load", {}).items():
        for row in data["demand_el"]:
            if row["node"] == source and (not hours or row["hour"] in hours[0]):
                moved = share * row["demand_mw"]
                row["demand_mw"] -= moved
                next(r for r in data["demand_el"] if r["node"] == target and r["hour"] == row["hour"])["demand_mw"] += moved
    return data


def with_storage(node):
    profile = copy.deepcopy(base_profile)
    profile["I1"]["power_mw"], profile["I1"]["energy_mwh"] = {node: args.mw}, {node: 4 * args.mw}
    return profile


for name in args.variants:
    data = edited({} if name == "base" else json.loads(name))
    primal, price = unique_price_reclear(data, base_profile, args.eps)
    diff = {t: price["N3", t] - price["N9", t] for t in data["times"]}
    profit = {}
    for node in ("N3", "N9"):
        m, p = unique_price_reclear(data, with_storage(node), args.eps)
        profit[node] = settlement_profit(m, "I1", p)
    print(name)
    print(f"  no storage: max |LMP N3 - N9| {max(map(abs, diff.values())):.2f}, "
          f"G3 max {max(pyo.value(primal.generation['G_IEEE3', t]) for t in data['times']):.1f} MW, "
          f"|L93| max {max(abs(pyo.value(primal.line_flow['L93', t])) for t in data['times']):.1f} MW, "
          f"shed {sum(pyo.value(v) for v in primal.load_shed.values()):.1f} MWh")
    print(f"  differing hours: {({t: round(d, 2) for t, d in diff.items() if abs(d) > 1e-3})}")
    print(f"  I1 {args.mw:g} MW / {4 * args.mw:g} MWh: profit at N3 {profit['N3']:.2f}, at N9 {profit['N9']:.2f}")
