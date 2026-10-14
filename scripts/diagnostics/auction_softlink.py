"""Uniform-price capacity auction soft-linked to the market model (test).

Hosting capacity H at one node is auctioned. Each round:
  1. bids: for every investor and every price p on --prices, its best response
     against the rivals' current holdings with capacity price p at the node
     (relaxed MPEC, Ipopt, from its current capacity and from 50 MW, plus
     building nothing; best exact net profit). Its MW at the node is its
     demand at p, a truthful bid curve.
  2. clearing: the uniform price p* where total demand equals H (linear
     interpolation between grid prices; p* = 0 if demand at 0 fits).
  3. allocation: each investor's best response at p*; if their total MW at
     the node exceeds H (lumpy demand), every holding there is scaled down
     pro rata (energy with it).
The allocation becomes the rivals' holdings of the next round (Jacobi: all bid
against the same state, so there is no move order). Pure Jacobi (--damping 1)
cycled with period 2 on the 5-bus system: all investors rushed into the
unlimited N2 together, then all left. --damping a moves each holding only a
of the way to the new allocation. A fixed point with
truthful bids is the uniform-price (variational) equilibrium that run.py
clearing searches by bisection. Local best responses; not a proof.

python scripts/diagnostics/auction_softlink.py . --node N5 --hosting 100 --rounds 4 --output model/output/scarce_N5/auction
"""
import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from capacity_clearing import EPSILONS, exact_profits, net_profit  # noqa: E402
from mpec import chosen_profile  # noqa: E402
from mpec_relaxed import build_mpec_relaxed, initialize_from_market, solve_relaxed, start_profile  # noqa: E402
from prepare_input import load_case, scale_flexible_demand  # noqa: E402


def best_response(job):
    """Best of building nothing and two relaxed MPEC starts at capacity price p."""
    data, profile, i, node, p, eps, seconds = job
    prices = {node: p}
    zero = copy.deepcopy(profile)
    zero[i]["power_mw"], zero[i]["energy_mwh"] = {}, {}
    candidates = [zero]
    builds = any(v > 1e-6 for v in profile[i]["power_mw"].values())
    for source in ([profile] if builds else []) + [None]:
        m = build_mpec_relaxed(data, profile, i, balancing_eps=eps, capacity_price=prices)
        initialize_from_market(m, data, start_profile(m, profile, start_power=50.0, source=source))
        history = solve_relaxed(m, EPSILONS, seconds=seconds)
        if len(history) == len(EPSILONS) and history[-1]["status"] == "optimal":
            candidates.append(chosen_profile(m))
    best = max(candidates, key=lambda c: net_profit(data, c, i, prices, eps))
    return i, p, {f: best[i][f] for f in ("power_mw", "energy_mwh")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--data", type=Path, default=Path("model/input/five_bus_market_data.json"))
    parser.add_argument("--profile", type=Path, default=Path("model/input/five_bus_capacities.json"))
    parser.add_argument("--node", required=True)
    parser.add_argument("--hosting", type=float, required=True)
    parser.add_argument("--prices", type=float, nargs="+", default=[0, 50, 100, 200, 300, 500, 800, 1200])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--balancing-eps", type=float, default=1e-4)
    parser.add_argument("--flexible-demand-scale", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--damping", type=float, default=1.0, help="Share of the step to the new allocation.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, profile = load_case(args.root / args.data, args.root / args.profile)
    data = scale_flexible_demand(data, args.flexible_demand_scale)
    args.output.mkdir(parents=True, exist_ok=True)
    node, eps, log = args.node, args.balancing_eps, []
    for round_ in range(1, args.rounds + 1):
        jobs = [(data, profile, i, node, p, eps, args.seconds) for i in profile for p in args.prices]
        with ProcessPoolExecutor(args.workers) as pool:
            bids = {(i, p): choice for i, p, choice in pool.map(best_response, jobs)}
        demand = {p: sum(bids[i, p]["power_mw"].get(node, 0.0) for i in profile) for p in args.prices}
        prices = sorted(args.prices)
        if demand[prices[0]] <= args.hosting:
            clearing = prices[0]
        elif demand[prices[-1]] > args.hosting:
            clearing = prices[-1]
        else:
            k = next(k for k in range(len(prices) - 1) if demand[prices[k + 1]] <= args.hosting)
            lo, hi = prices[k], prices[k + 1]
            clearing = lo + (demand[lo] - args.hosting) / (demand[lo] - demand[hi]) * (hi - lo)
        with ProcessPoolExecutor(args.workers) as pool:
            final = {i: choice for i, _, choice in pool.map(
                best_response, [(data, profile, i, node, clearing, eps, args.seconds) for i in profile])}
        total = sum(c["power_mw"].get(node, 0.0) for c in final.values())
        ration = min(1.0, args.hosting / total) if total > 0 else 1.0
        new, a = copy.deepcopy(profile), args.damping
        for i, c in final.items():
            for field in ("power_mw", "energy_mwh"):
                target = {n: v * (ration if n == node else 1.0) for n, v in c[field].items()}
                new[i][field] = {n: (1 - a) * profile[i][field].get(n, 0.0) + a * target.get(n, 0.0)
                                 for n in data["nodes"]}
        profits = exact_profits(data, new, {node: clearing}, eps)
        change = max(abs(new[i]["power_mw"].get(n, 0.0) - profile[i]["power_mw"].get(n, 0.0))
                     for i in profile for n in data["nodes"])
        record = dict(round=round_, damping=a, clearing_price=clearing, demand_mw=demand, rationing=ration, max_change_mw=change,
                      bids={i: {str(p): round(bids[i, p]["power_mw"].get(node, 0.0), 3) for p in args.prices}
                            for i in profile},
                      allocation={i: {n: round(v, 3) for n, v in new[i]["power_mw"].items() if v > 1e-6}
                                  for i in profile},
                      net_profit={i: v["market_profit"] - v["capacity_payment"] for i, v in profits.items()})
        log.append(record)
        print(json.dumps({k: record[k] for k in ("round", "clearing_price", "demand_mw", "rationing",
                                                 "max_change_mw", "allocation")}), flush=True)
        profile = new
        (args.output / "auction_log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
        (args.output / "capacities.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        if change < 0.5:
            break


if __name__ == "__main__":
    main()
