"""Payoff table of hosting-right allocations (run.py clearing --allocation).

Each run directory holds one assignment of the lots at a node; its summary.json
has every investor's exact market profit (after capex, no payment). From the
table:
    WTP to win the whole node instead of rival j:  profit_i(i holds all) - profit_i(j holds all)
    WTP against the node staying closed:           profit_i(i holds all) - profit_i(closed)
The first is what i would bid when someone fills the node anyway; the second
also counts the price uplift that any storage there gives an owner.

python scripts/diagnostics/lot_table.py model/output/lots_N5 --node N5 --closed closed --output <file>.json
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("runs", type=Path, help="Folder with one run directory per allocation.")
parser.add_argument("--node", default="N5")
parser.add_argument("--closed", default="closed", help="Run directory with the node closed.")
parser.add_argument("--output", type=Path)
args = parser.parse_args()

table = {}
for summary_path in sorted(args.runs.glob("*/summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile = json.loads((summary_path.parent / "capacities.json").read_text(encoding="utf-8"))
    rights = {i: r.get(args.node, 0.0) for i, r in summary["allocation"].items() if r.get(args.node, 0.0) > 0}
    table[summary_path.parent.name] = dict(
        rights=rights, converged=summary["inner_converged"],
        profit={i: v["market_profit"] for i, v in summary["profits"].items()},
        power_mw={i: {n: round(v, 2) for n, v in c["power_mw"].items() if v > 1e-3} for i, c in profile.items()},
        energy_mwh={i: {n: round(v, 1) for n, v in c["energy_mwh"].items() if v > 1e-3} for i, c in profile.items()})

investors = list(next(iter(table.values()))["profit"])
print(f"{'allocation':14s} {'conv.':5s} " + " ".join(f"{i:>9s}" for i in investors) + f" {'sum':>9s}  storage MW / MWh")
for name, row in table.items():
    built = "; ".join(f"{i} " + ", ".join(f"{n} {p}/{row['energy_mwh'][i][n]}" for n, p in mw.items())
                      for i, mw in row["power_mw"].items() if mw)
    print(f"{name:14s} {str(row['converged']):5s} " + " ".join(f"{row['profit'][i]:9.0f}" for i in investors)
          + f" {sum(row['profit'].values()):9.0f}  {built}")

# Runs in which one investor holds every right at the node.
sole = {next(iter(row["rights"])): row for row in table.values() if len(row["rights"]) == 1}
wtp = {}
for i, own in sole.items():
    wtp[i] = {f"vs {j}": own["profit"][i] - row["profit"][i] for j, row in sole.items() if j != i}
    if args.closed in table:
        wtp[i]["vs closed"] = own["profit"][i] - table[args.closed]["profit"][i]
print(f"\nWTP for all rights at {args.node} [EUR/day]")
for i, values in wtp.items():
    print(f"  {i}: " + ", ".join(f"{k} {v:.0f}" for k, v in values.items()))
if args.output:
    args.output.write_text(json.dumps(dict(node=args.node, table=table, wtp=wtp), indent=1) + "\n", encoding="utf-8")
