"""Independent commands: market, ranges, one-investor capacity mpec (Big-M,
Gurobi), and mpec_relaxed (relaxed complementarity, Ipopt, local)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pyomo.environ as pyo

from prepare_input import INPUT, load_case
from primal_llp import build_primal, dispatch_variables
from dual_llp import build_dual, dual_profit
from solve import make_solver, nlp_solver_version, require_optimal, residual, solve, solver_version
from mpec import build_mpec, chosen_profile, complementarity_pairs
from mpec_relaxed import build_mpec_relaxed, initialize_from_market, solve_relaxed
from solution_space import analyze, extrema, face_models, settlement_profit, unique_price_reclear


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def export_market(out, m, data, price=None):
    """price=None reads the LP duals of the nodal balance."""
    write_csv(out / "dispatch.csv", [dict(coordinate=name, value=pyo.value(variable))
                                       for name, variable in dispatch_variables(m)])
    write_csv(out / "prices.csv", [dict(node=n, hour=t,
        price_eur_per_mwh=float(m.dual[m.nodal_balance[n, t]]) if price is None else price[n, t])
        for n in data["nodes"] for t in data["times"]])


def mpec_results(out, data, m, args):
    """Export a solved MPEC (Big-M or relaxed), reclear its capacity exactly, and
    return the shared checks."""
    selected = chosen_profile(m)
    write_json(out / "capacities.json", selected)
    if args.balancing_eps > 0:
        # Unique prices: the profit range is one point, the settlement at those prices.
        reclear, price = unique_price_reclear(data, selected, args.balancing_eps)
        export_market(out, reclear, data, price)
        unique = settlement_profit(reclear, args.investor, price)
        face_summary = dict(market_cost_eur=pyo.value(reclear.market_cost))
        payoff = dict(minimum=unique, maximum=unique, minimum_status="unique_prices",
                      maximum_status="unique_prices", width=0.0)
    else:
        reclear, dual, face_summary = face_models(data, selected, args.face_tolerance)
        export_market(out, reclear, data)
        # Solver's selected reclear prices are only one dual optimum; report
        # the full payoff interval before interpreting a settlement difference.
        payoff = extrema(dual, dual_profit(dual, args.investor))
    pairs = list(complementarity_pairs(m))
    write_csv(out / "embedded_dispatch.csv", [dict(coordinate=name, value=pyo.value(variable))
                                              for name, variable in dispatch_variables(m)])
    write_csv(out / "embedded_prices.csv", [dict(node=n, hour=t, price_eur_per_mwh=pyo.value(m.price[n, t]))
               for n in data["nodes"] for t in data["times"]])
    # Direct settlement and strong-duality profit agree only at an exact KKT point.
    return dict(profit_eur_per_day=pyo.value(m.profit), linear_profit_eur_per_day=pyo.value(m.profit_linear),
        profit_identity_error_eur=abs(pyo.value(m.profit - m.profit_linear)),
        embedded_market_cost_eur=pyo.value(m.market_cost), recleared_market_cost_eur=face_summary["market_cost_eur"],
        reclear_cost_gap_eur=pyo.value(m.market_cost) - face_summary["market_cost_eur"],
        primal_dual_gap_eur=pyo.value(m.market_cost - m.dual_value),
        maximum_constraint_violation=residual(m),
        complementarity_pair_count=len(pairs),
        maximum_complementarity_product=max(abs(pyo.value(slack * multiplier)) for slack, multiplier in pairs),
        largest_dual_multiplier=max(pyo.value(multiplier) for _, multiplier in pairs),
        dual_m_saturation_count=sum(pyo.value(multiplier) >= 0.999 * args.dual_m for _, multiplier in pairs),
        fixed_capacity_profit_range=payoff)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("market", "ranges", "mpec", "mpec_relaxed"))
    parser.add_argument("--data", type=Path, default=INPUT / "market_data.json")
    parser.add_argument("--profile", type=Path, default=INPUT / "capacities.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--investor", default="I1")
    parser.add_argument("--nodes", nargs="+", help="Restrict the active investor's possible investment locations.")
    parser.add_argument("--node-limit", type=float, default=1000)
    parser.add_argument("--dual-m", type=float, default=100000)
    parser.add_argument("--objective", choices=("bilinear", "linear"), default="bilinear",
                        help="MPEC objective: direct nodal settlement, or its strong-duality linear form.")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--face-tolerance", type=float, default=1e-6, help="Absolute market objective tolerance in EUR.")
    parser.add_argument("--dispatch", choices=("none", "storage", "all"), default="storage")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1, 1e-1, 1e-2, 1e-3, 1e-4],
                        help="mpec_relaxed: decreasing complementarity product bounds; the last is final.")
    parser.add_argument("--start-power", type=float, default=50,
                        help="mpec_relaxed: start capacity [MW] at each investment node.")
    parser.add_argument("--balancing-eps", type=float, default=0.0,
                        help="Nodal balancing slope [MW per EUR/MWh]; > 0 makes prices unique (QP market). 0 is the LP.")
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.mip_gap < 0 or args.face_tolerance < 0:
        parser.error("Time must be positive; MIP gap and face tolerance must be nonnegative.")
    if min(args.epsilons) <= 0 or any(a <= b for a, b in zip(args.epsilons, args.epsilons[1:])):
        parser.error("Epsilons must be positive and strictly decreasing.")
    if args.start_power <= 0:
        parser.error("The start capacity must be positive.")
    if args.balancing_eps < 0:
        parser.error("The balancing slope must be nonnegative.")
    data, profile = load_case(args.data, args.profile)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory so a new run cannot leave stale results.")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    config["solver"], config["solver_version"] = "gurobi", solver_version()
    if args.command == "mpec_relaxed" or args.balancing_eps > 0:
        # Ipopt solves the relaxed MPEC and, with balancing, the unique-price reclear.
        config["nlp_solver"], config["nlp_solver_version"] = "ipopt", nlp_solver_version()
    write_json(args.output / "run_config.json", config)
    write_json(args.output / "input_profile.json", profile)
    if args.command == "market":
        m = build_primal(data, profile, args.balancing_eps)
        require_optimal(m, seconds=args.seconds)
        dual = build_dual(data, profile, args.balancing_eps)
        require_optimal(dual, seconds=args.seconds)
        summary = dict(status="optimal", market_cost_eur=pyo.value(m.market_cost),
            dual_value_eur=pyo.value(dual.dual_value),
            primal_dual_gap_eur=pyo.value(m.market_cost) - pyo.value(dual.dual_value),
            primal_residual=residual(m), dual_residual=residual(dual),
            load_shed_mwh=sum(pyo.value(variable) for variable in m.load_shed.values()))
        export_market(args.output, m, data)
    elif args.command == "ranges":
        records, summary = analyze(data, profile, tolerance=args.face_tolerance, dispatch=args.dispatch, progress=print,
                                   balancing_eps=args.balancing_eps)
        write_csv(args.output / "ranges.csv", records)
    elif args.command == "mpec":
        m = build_mpec(data, profile, args.investor, nodes=args.nodes,
                            node_limit=args.node_limit, dual_m=args.dual_m, objective=args.objective,
                       balancing_eps=args.balancing_eps)
        market_variable_count = sum(1 for _ in dispatch_variables(m))
        binary_count = sum(v.is_binary() for v in m.component_data_objects(pyo.Var))
        print(f"Solving one-investor MPEC ({args.objective} objective): "
              f"{market_variable_count} market variables, {binary_count} binaries", flush=True)
        solver = make_solver()
        solver.options["LogFile"] = str(args.output / "gurobi.log")
        result = solve(m, seconds=args.seconds, mip_gap=args.mip_gap, solver=solver)
        status = str(result.solver.termination_condition)
        summary = dict(status=status, investor=args.investor, dual_big_m=args.dual_m,
                       objective=args.objective, solver="gurobi", solver_version=config["solver_version"],
                       investment_nodes=args.nodes or data["nodes"],
                       solver_lower_bound=float(result.problem.lower_bound) if result.problem.lower_bound is not None and abs(result.problem.lower_bound) < float("inf") else None,
                       solver_upper_bound=float(result.problem.upper_bound) if result.problem.upper_bound is not None and abs(result.problem.upper_bound) < float("inf") else None)
        if status != "optimal":
            summary["interpretation"] = "No proven optimal MPEC result loaded. Increase the time limit or reduce the case. No equilibrium claim."
            write_json(args.output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return 2
        summary.update(mpec_results(args.output, data, m, args),
            interpretation="Optimistic best response for the specified rival profile and strategy bounds, conditional on dual Big-M validity. Not a multi-investor equilibrium.")
    else:
        m = build_mpec_relaxed(data, profile, args.investor, nodes=args.nodes, node_limit=args.node_limit,
                               dual_m=args.dual_m, objective=args.objective, epsilon=args.epsilons[0],
                               balancing_eps=args.balancing_eps)
        print(f"Solving relaxed one-investor MPEC ({args.objective} objective) with Ipopt: "
              f"epsilon {args.epsilons[0]:g} -> {args.epsilons[-1]:g}", flush=True)
        write_json(args.output / "start_profile.json", initialize_from_market(m, data, profile, args.start_power))
        _, history = solve_relaxed(m, args.epsilons, seconds=args.seconds, log_dir=args.output)
        status = history[-1]["status"]
        summary = dict(status=status, investor=args.investor, dual_big_m=args.dual_m,
                       objective=args.objective, solver="ipopt", solver_version=config["nlp_solver_version"],
                       investment_nodes=args.nodes or data["nodes"], epsilon=args.epsilons[-1],
                       start_power_mw=args.start_power, continuation=history)
        if status != "optimal":
            summary["interpretation"] = "Continuation stopped before the final epsilon; no result exported. Try another --start-power or more epsilon steps. No equilibrium claim."
            write_json(args.output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return 2
        summary.update(mpec_results(args.output, data, m, args))
        exact = summary["fixed_capacity_profit_range"]["maximum"]
        summary.update(relaxed_minus_exact_optimistic_profit_eur=None if exact is None else summary["profit_eur_per_day"] - exact,
            interpretation="Local optimum of the relaxed MPEC from one start, not a proven best response. Products are only bounded by epsilon; compare the exact fixed-capacity profit range. Not a multi-investor equilibrium.")
    if args.command == "market":
        summary["verification_passed"] = (abs(summary["primal_dual_gap_eur"]) <= 1e-5
            and max(summary["primal_residual"], summary["dual_residual"]) <= 1e-5)
    elif args.command == "mpec":
        summary["verification_passed"] = (summary["profit_identity_error_eur"] <= 1e-4
            and abs(summary["reclear_cost_gap_eur"]) <= 1e-4
            and abs(summary["primal_dual_gap_eur"]) <= 1e-4
            and summary["maximum_constraint_violation"] <= 1e-5
            and summary["maximum_complementarity_product"] <= 1e-4)
    elif args.command == "mpec_relaxed":
        # The primal-dual gap is the sum of all products, so at most pairs * epsilon.
        gap_limit = summary["complementarity_pair_count"] * args.epsilons[-1] + 1e-6
        summary["verification_passed"] = (summary["maximum_constraint_violation"] <= 1e-6
            and summary["maximum_complementarity_product"] <= args.epsilons[-1] + 1e-6
            and -1e-6 <= summary["primal_dual_gap_eur"] <= gap_limit
            and -1e-6 <= summary["reclear_cost_gap_eur"] <= gap_limit)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("verification_passed", True) else 3


if __name__ == "__main__":
    raise SystemExit(main())
