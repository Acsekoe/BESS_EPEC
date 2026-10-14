"""Pocket prices and profits along one storage size, for one flexible demand scale.

One investor (default the merchant I1) holds a battery of --duration hours at
--node; its power runs over --powers. Each point re-clears the market with
unique prices (balancing eps, Ipopt; solution_space.unique_price_reclear) at
flexible demand scale --scale (run.py --flexible-demand-scale; 0 = none). Rivals
hold nothing. Writes OUT_prices.csv (scale, power, node, hour, price) and
OUT_profits.csv (scale, power, investor, profit after capex). Prices do not
depend on who owns the battery (truthful bids, equal degradation and dispatch
cost); profits do: the holder's profit is its storage arbitrage, the RES
owners' changes are their margin effects.

python scripts/diagnostics/flex_size_scan.py . --node N5 --duration 4.027 --scale 1 --output model/output/flex_scan/N5_s1
python scripts/diagnostics/plot_flex_size_scan.py model/output/flex_scan --output model/output/flex_scan
"""
import argparse
import copy
import csv
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--data", type=Path, default=Path("model/input/five_bus_market_data.json"))
parser.add_argument("--profile", type=Path, default=Path("model/input/five_bus_capacities.json"))
parser.add_argument("--node", required=True)
parser.add_argument("--investor", default="I1")
parser.add_argument("--duration", type=float, required=True, help="MWh per MW of the scanned battery.")
parser.add_argument("--powers", type=float, nargs="+", required=True)
parser.add_argument("--scale", type=float, required=True)
parser.add_argument("--balancing-eps", type=float, default=1e-4)
parser.add_argument("--output", type=Path, required=True, help="Path prefix for the two CSV files.")
args = parser.parse_args()

sys.path.insert(0, str(args.root / "model"))
from prepare_input import load_case, scale_flexible_demand  # noqa: E402
from solution_space import settlement_profit, unique_price_reclear  # noqa: E402

data, base = load_case(args.root / args.data, args.root / args.profile)
data = scale_flexible_demand(data, args.scale)
args.output.parent.mkdir(parents=True, exist_ok=True)
with open(f"{args.output}_prices.csv", "w", newline="") as fp, open(f"{args.output}_profits.csv", "w", newline="") as fq:
    prices, profits = csv.writer(fp), csv.writer(fq)
    prices.writerow(["scale", "node_scanned", "power_mw", "energy_mwh", "node", "hour", "price_eur_per_mwh"])
    profits.writerow(["scale", "node_scanned", "power_mw", "energy_mwh", "investor", "profit_eur_per_day"])
    for power in args.powers:
        profile = copy.deepcopy(base)
        profile[args.investor]["power_mw"] = {args.node: power} if power > 0 else {}
        profile[args.investor]["energy_mwh"] = {args.node: args.duration * power} if power > 0 else {}
        market, price = unique_price_reclear(data, profile, args.balancing_eps)
        key = [args.scale, args.node, power, args.duration * power]
        for (n, t), value in price.items():
            prices.writerow(key + [n, t, value])
        for i in profile:
            profits.writerow(key + [i, settlement_profit(market, i, price)])
        print(f"scale {args.scale} {args.node} {power} MW done", flush=True)
