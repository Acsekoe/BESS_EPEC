"""Probe a candidate equilibrium with small capacity moves (exact reclears only).

Each case changes one or more investors' capacities and reports every changed
investor's profit gain and the rivals' profit changes (unique-price reclear,
Ipopt, no capacity price). Local Ipopt best responses miss moves that cross a
pocket step (workflow/summary_2026-10-13_15-01.md, update 16:04); this checks
them directly. A probe without a profitable move is not a proof.

A change is INVESTOR.NODE=MW/MWh (set) or INVESTOR.NODE+=MW/MWh (add; negative
values remove). Changes in one case are comma-separated. Without --case: for
each RES owner and each node with storage, add 0.01/0.04 and 1/4, remove 1 MW
and remove 0.5 MWh. --prices prints the hourly price at those nodes per case.

python scripts/diagnostics/equilibrium_probe.py . --profile workflow/equilibria_2026-10-13/ieee5_v6_equilibrium_B.json --prices N2
python scripts/diagnostics/equilibrium_probe.py . --profile workflow/equilibria_2026-10-13/ieee5_v6_equilibrium_A.json --case "I3.N2=0/0" --case "I3.N5=129.122/557.089" --case "I3.N2=0/0,I3.N5=129.122/557.089"
"""
import argparse
import copy
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--data", type=Path, default=Path("model/input/ieee5_v6_market_data.json"))
parser.add_argument("--profile", type=Path, required=True)
parser.add_argument("--balancing-eps", type=float, default=1e-4)
parser.add_argument("--case", action="append", default=[])
parser.add_argument("--prices", nargs="*", default=[])
args = parser.parse_args()

sys.path.insert(0, str(args.root / "model"))
from capacity_clearing import exact_profits  # noqa: E402
from prepare_input import load_case  # noqa: E402
from solution_space import unique_price_reclear  # noqa: E402

CHANGE = re.compile(r"^(\w+)\.(\w+)(\+?=)([-+\d.eE]+)/([-+\d.eE]+)$")
DEFAULT_MOVES = ("+=0.01/0.04", "+=1/4", "+=-1/0", "+=0/-0.5")

data, base = load_case(args.root / args.data, args.root / args.profile)


def apply(profile, case):
    profile, investors = copy.deepcopy(profile), []
    for change in case.split(","):
        match = CHANGE.match(change.strip())
        if not match:
            raise ValueError(f"Bad change {change!r}; use I.N=MW/MWh or I.N+=MW/MWh.")
        i, n, op, mw, mwh = match.groups()
        for field, value in (("power_mw", float(mw)), ("energy_mwh", float(mwh))):
            old = profile[i][field].get(n, 0.0) if op == "+=" else 0.0
            profile[i][field][n] = max(old + value, 0.0)
        investors.append(i)
    return profile, sorted(set(investors))


def default_cases():
    owners = [i for i, c in base.items() if c.get("owned_generation_shares")]
    nodes = [n for n in data["nodes"] if any(c["power_mw"].get(n, 0.0) > 1e-6 for c in base.values())]
    return [f"{i}.{n}{move}" for i in owners for n in nodes for move in DEFAULT_MOVES]


def report(name, profile, investors):
    profits = {i: v["market_profit"] for i, v in exact_profits(data, profile, None, args.balancing_eps).items()}
    own = " ".join(f"{i} {profits[i] - reference[i]:+10.1f}" for i in investors)
    rivals = " ".join(f"{i} {profits[i] - reference[i]:+10.1f}" for i in profits if i not in investors)
    print(f"{name:42s} gain {own} | rivals {rivals}", flush=True)
    if args.prices:
        _, price = unique_price_reclear(data, profile, args.balancing_eps)
        for n in args.prices:
            print(f"{'':42s} price {n}: " + " ".join(f"{price[n, t]:.1f}" for t in data["times"]))


reference = {i: v["market_profit"] for i, v in exact_profits(data, base, None, args.balancing_eps).items()}
print(f"base profits (EUR/day): " + " ".join(f"{i} {v:.1f}" for i, v in reference.items()))
report("base", base, [])
for case in args.case or default_cases():
    profile, investors = apply(base, case)
    report(case, profile, investors)
