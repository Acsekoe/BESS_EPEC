"""Independent commands: market, ranges, and one-investor capacity mpec."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pyomo.environ as pyo

from prepare_input import INPUT, load_case
from primal_llp import build_primal, dispatch_variables
from dual_llp import build_dual, dual_profit
from solve import require_optimal, residual, solve
from mpec import build_mpec, chosen_profile, direct_profit, complementarity_pairs
from solution_space import analyze, extrema, face_models


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def export_market(out, m, data):
    write_csv(out / "dispatch.csv", [dict(coordinate=name, value=pyo.value(variable))
                                       for name, variable in dispatch_variables(m)])
    write_csv(out / "prices.csv", [dict(node=n, hour=t,
        price_eur_per_mwh=float(m.dual[m.nodal_balance[n, t]]))
        for n in data["nodes"] for t in data["times"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("market", "ranges", "mpec"))
    parser.add_argument("--data", type=Path, default=INPUT / "market_data.json")
    parser.add_argument("--profile", type=Path, default=INPUT / "capacities.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--investor", default="I1")
    parser.add_argument("--nodes", nargs="+", help="Restrict the active investor's possible investment locations.")
    parser.add_argument("--node-limit", type=float, default=1000)
    parser.add_argument("--dual-m", type=float, default=100000)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--face-tolerance", type=float, default=1e-6, help="Absolute market objective tolerance in EUR.")
    parser.add_argument("--dispatch", choices=("none", "storage", "all"), default="storage")
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.mip_gap < 0 or args.face_tolerance < 0:
        parser.error("Time must be positive; MIP gap and face tolerance must be nonnegative.")
    data, profile = load_case(args.data, args.profile)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory so a new run cannot leave stale results.")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    write_json(args.output / "run_config.json", config)
    write_json(args.output / "input_profile.json", profile)
    if args.command == "market":
        m = build_primal(data, profile)
        require_optimal(m, seconds=args.seconds)
        dual = build_dual(data, profile)
        require_optimal(dual, seconds=args.seconds)
        summary = dict(status="optimal", market_cost_eur=pyo.value(m.market_cost),
            dual_value_eur=pyo.value(dual.dual_value),
            primal_dual_gap_eur=pyo.value(m.market_cost) - pyo.value(dual.dual_value),
            primal_residual=residual(m), dual_residual=residual(dual),
            load_shed_mwh=sum(pyo.value(variable) for variable in m.load_shed.values()))
        export_market(args.output, m, data)
    elif args.command == "ranges":
        records, summary = analyze(data, profile, tolerance=args.face_tolerance, dispatch=args.dispatch, progress=print)
        write_csv(args.output / "ranges.csv", records)
    else:
        m = build_mpec(data, profile, args.investor, nodes=args.nodes,
                            node_limit=args.node_limit, dual_m=args.dual_m)
        market_variable_count = sum(1 for _ in dispatch_variables(m))
        binary_count = sum(v.is_binary() for v in m.component_data_objects(pyo.Var))
        print(f"Solving one-investor MPEC: {market_variable_count} market variables, {binary_count} binaries", flush=True)
        result = solve(m, seconds=args.seconds, mip_gap=args.mip_gap)
        status = str(result.solver.termination_condition)
        summary = dict(status=status, investor=args.investor, dual_big_m=args.dual_m,
                       investment_nodes=args.nodes or data["nodes"],
                       solver_lower_bound=float(result.problem.lower_bound) if result.problem.lower_bound is not None and abs(result.problem.lower_bound) < float("inf") else None,
                       solver_upper_bound=float(result.problem.upper_bound) if result.problem.upper_bound is not None and abs(result.problem.upper_bound) < float("inf") else None)
        if status != "optimal":
            summary["interpretation"] = "No proven optimal MPEC result loaded. Increase the time limit or reduce the case. No equilibrium claim."
            write_json(args.output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return 2
        selected = chosen_profile(m)
        write_json(args.output / "capacities.json", selected)
        reclear, dual, face_summary = face_models(data, selected, args.face_tolerance)
        export_market(args.output, reclear, data)
        # Solver's selected reclear prices are only one dual optimum; report
        # the full payoff interval before interpreting a settlement difference.
        payoff = extrema(dual, dual_profit(dual, args.investor))
        pairs = list(complementarity_pairs(m))
        summary.update(profit_eur_per_day=pyo.value(m.profit), direct_profit_eur_per_day=direct_profit(m),
            profit_identity_error_eur=abs(pyo.value(m.profit) - direct_profit(m)),
            embedded_market_cost_eur=pyo.value(m.market_cost), recleared_market_cost_eur=face_summary["market_cost_eur"],
            reclear_cost_gap_eur=pyo.value(m.market_cost) - face_summary["market_cost_eur"],
            primal_dual_gap_eur=pyo.value(m.market_cost - m.dual_value),
            maximum_constraint_violation=residual(m),
            maximum_complementarity_product=max(abs(pyo.value(slack * multiplier)) for slack, multiplier in pairs),
            largest_dual_multiplier=max(pyo.value(multiplier) for _, multiplier in pairs),
            dual_m_saturation_count=sum(pyo.value(multiplier) >= 0.999 * args.dual_m for _, multiplier in pairs),
            fixed_capacity_profit_range=payoff,
            interpretation="Optimistic best response for the specified rival profile and strategy bounds, conditional on dual Big-M validity. Not a multi-investor equilibrium.")
        write_csv(args.output / "embedded_dispatch.csv", [dict(coordinate=name, value=pyo.value(variable))
                                                          for name, variable in dispatch_variables(m)])
        write_csv(args.output / "embedded_prices.csv", [dict(node=n, hour=t, price_eur_per_mwh=pyo.value(m.price[n, t]))
                   for n in data["nodes"] for t in data["times"]])
    if args.command == "market":
        summary["verification_passed"] = (abs(summary["primal_dual_gap_eur"]) <= 1e-5
            and max(summary["primal_residual"], summary["dual_residual"]) <= 1e-5)
    elif args.command == "mpec":
        summary["verification_passed"] = (summary["profit_identity_error_eur"] <= 1e-4
            and abs(summary["reclear_cost_gap_eur"]) <= 1e-4
            and abs(summary["primal_dual_gap_eur"]) <= 1e-4
            and summary["maximum_constraint_violation"] <= 1e-5
            and summary["maximum_complementarity_product"] <= 1e-4)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("verification_passed", True) else 3


if __name__ == "__main__":
    raise SystemExit(main())
