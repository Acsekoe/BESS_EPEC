"""Independent commands: market, ranges, the one-investor capacity mpec (SOS1
complementarity, Gurobi), mpec_relaxed (relaxed complementarity, Ipopt,
local), clearing (hosting capacity shared by a uniform price, all investors), and
verify (Gurobi check that a capacity profile is an equilibrium).
--strategic-bids adds the investor's hourly charge bids and discharge offers;
mpec then selects the bids closest to truthful before the reclear."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pyomo.environ as pyo

from prepare_input import INPUT, load_case, scale_flexible_demand
from primal_llp import build_primal, dispatch_variables
from dual_llp import build_dual, dual_profit
from solve import load_incumbent, make_solver, nlp_solver_version, require_optimal, residual, solver_version
from mpec import build_mpec, chosen_profile, complementarity_pairs, complete_sos1_start, solve_mpec
from mpec_relaxed import build_mpec_relaxed, initialize_from_market, solve_relaxed, start_profile
from solution_space import analyze, extrema, face_models, settlement_profit, unique_price_reclear
from capacity_clearing import clear_capacity, exact_profits, inner_equilibrium, verify_equilibrium


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
    """Export a solved MPEC, reclear its capacity exactly, and return the checks."""
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
        fixed_capacity_profit_range=payoff)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("market", "ranges", "mpec", "mpec_relaxed", "clearing", "verify"))
    parser.add_argument("--data", type=Path, default=INPUT / "five_bus_market_data.json")
    parser.add_argument("--profile", type=Path, default=INPUT / "five_bus_capacities.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--investor", default="I1")
    parser.add_argument("--nodes", nargs="+", help="Restrict the active investor's possible investment locations "
                        "(verify: the deviations checked).")
    parser.add_argument("--node-limit", type=float, default=1000)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--face-tolerance", type=float, default=1e-6, help="Absolute market objective tolerance in EUR.")
    parser.add_argument("--dispatch", choices=("none", "storage", "all"), default="storage")
    parser.add_argument("--balancing-eps", type=float, default=0.0,
                        help="Nodal balancing slope [MW per EUR/MWh]; > 0 makes prices unique (QP market). 0 is the LP.")
    parser.add_argument("--flexible-demand-scale", type=float, default=1.0,
                        help="Multiplies every flexible demand's max_mw in the market data (same willingness to pay); "
                             "0 removes flexible demand.")
    parser.add_argument("--strategic-bids", action="store_true",
                        help="mpec, mpec_relaxed: the investor also chooses hourly charge bids and discharge offers per node.")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                        help="mpec_relaxed: decreasing complementarity product bounds; the last is final.")
    parser.add_argument("--start-power", type=float, default=50,
                        help="mpec_relaxed: start MW (minimum duration) at each investment node.")
    parser.add_argument("--start-profile", type=Path,
                        help="mpec_relaxed: start from the investor's MW, MWh and bids in this profile instead. "
                             "mpec: Gurobi MIP start at that capacity (e.g. an mpec_relaxed capacities.json).")
    parser.add_argument("--hosting", nargs="+", default=[], metavar="NODE=MW",
                        help="clearing: hosting capacity of the scarce node(s); other nodes keep --node-limit.")
    parser.add_argument("--first-come", action="store_true",
                        help="clearing: share the --hosting limit first come, first served (rivals' MW reduce the "
                             "headroom) instead of by a uniform price.")
    parser.add_argument("--access-slope", nargs="+", default=[], metavar="NODE=SLOPE",
                        help="mpec, mpec_relaxed, clearing, verify: quadratic access cost; every MW at NODE pays "
                             "SLOPE [EUR/MW/day per MW] times the node's total MW.")
    parser.add_argument("--order", nargs="+", help="clearing: best-response order (default: profile order).")
    parser.add_argument("--price-start", type=float, default=50, help="clearing: first price tried [EUR/MW/day].")
    parser.add_argument("--price-tolerance", type=float, default=0.5, help="clearing: final bracket width [EUR/MW/day].")
    parser.add_argument("--rounds", type=int, default=6, help="clearing: Gauss-Seidel rounds per price.")
    parser.add_argument("--gain-tolerance", type=float, default=1.0,
                        help="clearing: an investor moves only for a larger profit gain [EUR/day]; "
                             "verify: largest gain that still counts as no profitable deviation.")
    parser.add_argument("--threads", type=int, help="verify: Gurobi threads per solve (default: all).")
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.mip_gap < 0 or args.face_tolerance < 0:
        parser.error("Time must be positive; MIP gap and face tolerance must be nonnegative.")
    if args.balancing_eps < 0:
        parser.error("The balancing slope must be nonnegative.")
    if min(args.epsilons) <= 0 or any(a <= b for a, b in zip(args.epsilons, args.epsilons[1:])):
        parser.error("Epsilons must be positive and strictly decreasing.")
    if args.strategic_bids and args.balancing_eps == 0:
        # With LP prices the bid margin depends on a non-unique dispatch, so the
        # exact profit at the chosen bids would be an interval over a joint face.
        parser.error("--strategic-bids needs --balancing-eps > 0 (unique prices for the exact reclear).")
    if args.flexible_demand_scale < 0:
        parser.error("The flexible demand scale must be nonnegative.")
    data, profile = load_case(args.data, args.profile)
    data = scale_flexible_demand(data, args.flexible_demand_scale)
    access_slope = {k: float(v) for k, v in (item.split("=") for item in args.access_slope)} or None
    if access_slope and set(access_slope) - set(data["nodes"]):
        parser.error("Unknown access slope node.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory so a new run cannot leave stale results.")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    config["solver"], config["solver_version"] = "gurobi", solver_version()
    if args.command == "mpec_relaxed" or args.balancing_eps > 0:
        # Ipopt solves the relaxed MPEC and, with balancing, the unique-price reclear.
        config["nlp_solver"], config["nlp_solver_version"] = "ipopt", nlp_solver_version()
    if args.start_profile is not None:
        config["start_profile_sha256"] = hashlib.sha256(args.start_profile.read_bytes()).hexdigest()
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
            load_shed_mwh=sum(pyo.value(variable) for variable in m.load_shed.values()),
            unused_generation_mwh={g: sum(pyo.value(m.curtailment[g, t]) for t in m.timesteps) for g in m.generators},
            flexible_demand_mwh=sum(pyo.value(variable) for variable in m.flexible_demand.values()),
            flexible_demand_benefit_eur=pyo.value(m.flexible_demand_benefit_total))
        export_market(args.output, m, data)
        summary["verification_passed"] = (abs(summary["primal_dual_gap_eur"]) <= 1e-5 + 1e-9 * abs(summary["market_cost_eur"])
            and max(summary["primal_residual"], summary["dual_residual"]) <= 1e-5)
    elif args.command == "clearing":
        if args.balancing_eps <= 0:
            parser.error("clearing compares exact profits: it needs --balancing-eps > 0 (unique prices).")
        hosting = {k: float(v) for k, v in (item.split("=") for item in args.hosting)}
        if set(hosting) - set(data["nodes"]):
            parser.error("Unknown hosting node.")
        order = args.order or list(profile)
        if args.first_come:
            print(f"First come, first served: hosting {hosting}, order {order}", flush=True)
            limits = {n: hosting.get(n, args.node_limit) for n in data["nodes"]}
            current, converged, history = inner_equilibrium(
                data, profile, limits, None, order=order, balancing_eps=args.balancing_eps,
                strategic_bids=args.strategic_bids, rounds=args.rounds, seconds=args.seconds,
                gain_tolerance=args.gain_tolerance, access_slope=access_slope,
                log=lambda text: print(text, flush=True))
            result = dict(mechanism="first come, first served", hosting_mw=hosting, profile=current,
                          inner_converged=converged,
                          profits=exact_profits(data, current, None, args.balancing_eps, access_slope))
            evaluations = [dict(prices=None, converged=converged, history=history, profile=current)]
        else:
            print(f"Clearing hosting capacity {hosting}, order {order}, access slope {access_slope}", flush=True)
            result, evaluations = clear_capacity(
                data, profile, hosting, order=order, balancing_eps=args.balancing_eps, node_limit=args.node_limit,
                strategic_bids=args.strategic_bids,
                price_start=args.price_start, price_tolerance=args.price_tolerance, rounds=args.rounds,
                seconds=args.seconds, gain_tolerance=args.gain_tolerance, access_slope=access_slope,
                log=lambda text: print(text, flush=True))
        write_json(args.output / "capacities.json", result.pop("profile"))
        write_json(args.output / "clearing_log.json", evaluations)
        summary = dict(result, investors=order, strategic_bids=args.strategic_bids, access_slope=access_slope,
            best_response="bid MPEC, Gurobi SOS1 (proven)" if args.strategic_bids
                          else "relaxed capacity MPEC, Ipopt (local), and building nothing",
            evaluation_count=len(evaluations),
            interpretation=("First come, first served: Gauss-Seidel best responses in the given order, rivals' MW "
                            "reduce each investor's headroom. inner_converged: a full round without profitable "
                            "deviation found (capacity only: by a local method). Depends on the order; not proven "
                            "unique." if args.first_come else
                            "Uniform hosting-capacity price (variational equilibrium of the shared node limit), "
                            "found by bisection on the price with Gauss-Seidel bid-MPEC best responses. "
                            "inner_converged: a full round without profitable deviation found at the final price "
                            "(capacity only: by a local method). "
                            "unused_mw > 0 at a narrow bracket means demand jumps across the limit (no clearing "
                            "price). With access_slope every MW also pays slope * node total. "
                            "One start and order; not proven unique."))
        summary["verification_passed"] = summary["inner_converged"]
    elif args.command == "verify":
        if args.balancing_eps <= 0:
            parser.error("verify compares exact profits: it needs --balancing-eps > 0 (unique prices).")
        print(f"Verifying {args.profile} as an equilibrium with Gurobi", flush=True)
        overall, records = verify_equilibrium(
            data, profile, balancing_eps=args.balancing_eps, node_limit=args.node_limit,
            strategic_bids=args.strategic_bids, nodes=args.nodes, seconds=args.seconds, mip_gap=args.mip_gap,
            threads=args.threads, tolerance=args.gain_tolerance, access_slope=access_slope,
            log=lambda text: print(text, flush=True))
        summary = dict(equilibrium=overall, investors=records, strategic_bids=args.strategic_bids,
            deviation_nodes=args.nodes or data["nodes"],
            interpretation="proven: every investor's Gurobi upper bound is within the tolerance of its current "
                           "exact profit, so no profitable deviation exists (rivals fixed, optimistic MPEC). "
                           "refuted: a better deviation was found. open: the bound did not close in time.")
        summary["verification_passed"] = overall == "proven"
    elif args.command == "ranges":
        records, summary = analyze(data, profile, tolerance=args.face_tolerance, dispatch=args.dispatch, progress=print,
                                   balancing_eps=args.balancing_eps)
        write_csv(args.output / "ranges.csv", records)
    elif args.command == "mpec_relaxed":
        m = build_mpec_relaxed(data, profile, args.investor, nodes=args.nodes, node_limit=args.node_limit,
                               balancing_eps=args.balancing_eps, strategic_bids=args.strategic_bids,
                               epsilon=args.epsilons[0], access_slope=access_slope)
        source = None if args.start_profile is None else load_case(args.data, args.start_profile)[1]
        start = start_profile(m, profile, start_power=args.start_power, source=source)
        write_json(args.output / "start_profile.json", start)
        initialize_from_market(m, data, start)
        print(f"Solving {m.name} with Ipopt: epsilon {args.epsilons[0]:g} -> {args.epsilons[-1]:g}", flush=True)
        history = solve_relaxed(m, args.epsilons, seconds=args.seconds, log_dir=args.output)
        status = history[-1]["status"]
        summary = dict(status=status, investor=args.investor, solver="ipopt", solver_version=config["nlp_solver_version"],
                       strategic_bids=args.strategic_bids, investment_nodes=args.nodes or data["nodes"],
                       epsilon=args.epsilons[-1], continuation=history)
        if len(history) < len(args.epsilons) or status != "optimal":
            summary["interpretation"] = "Continuation stopped before the final epsilon; no result exported. Try another start or more epsilon steps. No equilibrium claim."
            write_json(args.output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return 2
        summary.update(mpec_results(args.output, data, m, args))
        exact = summary["fixed_capacity_profit_range"]["maximum"]
        summary.update(exact_profit_eur_per_day=exact,
            relaxed_minus_exact_profit_eur=None if exact is None else summary["linear_profit_eur_per_day"] - exact,
            interpretation="Local optimum of the relaxed MPEC from one start, not a proven best response. "
                           "Products are only bounded by epsilon; exact_profit_eur_per_day is the investor's profit "
                           "in the exactly recleared market at the chosen capacity (and bids). Not a multi-investor equilibrium.")
        # The relaxation bounds the signed products. A negative one is a primal
        # tolerance (1e-8) times a large multiplier and is covered by the violation check.
        summary["maximum_signed_product"] = max(pyo.value(s * mu) for s, mu in complementarity_pairs(m))
        # At epsilon > 0 market cost and dual value differ by the sum of all products.
        summary["verification_passed"] = (summary["maximum_signed_product"] <= args.epsilons[-1] + 1e-6
            and summary["maximum_constraint_violation"] <= 1e-6
            and abs(summary["primal_dual_gap_eur"]) <= summary["complementarity_pair_count"] * args.epsilons[-1] + 1e-6)
    else:
        m = build_mpec(data, profile, args.investor, nodes=args.nodes, node_limit=args.node_limit,
                       balancing_eps=args.balancing_eps, strategic_bids=args.strategic_bids, access_slope=access_slope)
        market_variable_count = sum(1 for _ in dispatch_variables(m))
        sos_count = sum(1 for _ in m.component_data_objects(pyo.SOSConstraint))
        print(f"Solving one-investor MPEC: {market_variable_count} market variables, {sos_count} SOS1 sets", flush=True)
        if args.start_profile is not None:
            # MIP start: the fixed-capacity market KKT point at the start capacity.
            source = load_case(args.data, args.start_profile)[1]
            start = start_profile(m, profile, source=source)
            write_json(args.output / "start_profile.json", start)
            initialize_from_market(m, data, start)
            start_status = complete_sos1_start(m)
            print(f"MIP start from {args.start_profile}: polish {start_status}", flush=True)
        solver = make_solver()
        solver.options["LogFile"] = str(args.output / "gurobi.log")
        # With bids, a second solve selects the bids closest to truthful, so the
        # exported bids and the reclear below are definite.
        result, (incumbent, bound), selection = solve_mpec(m, seconds=args.seconds, mip_gap=args.mip_gap,
                                                           solver=solver, warmstart=args.start_profile is not None)
        status = str(result.solver.termination_condition)
        summary = dict(status=status, investor=args.investor, solver="gurobi", solver_version=config["solver_version"],
                       strategic_bids=args.strategic_bids, investment_nodes=args.nodes or data["nodes"],
                       solver_lower_bound=incumbent, solver_upper_bound=bound)  # maximization: incumbent <= bound
        if args.start_profile is not None:
            summary["mip_start"] = dict(profile=str(args.start_profile), polish_status=start_status)
        if selection is not None:
            summary["bid_selection"] = selection
        # A time-limited run exports Gurobi's incumbent: a feasible capacity
        # whose exact profit bounds the best response from below.
        proven = status == "optimal"
        if not proven and not load_incumbent(solver):
            summary["interpretation"] = "No MPEC solution found. Increase the time limit, give --start-profile, or reduce the case. No equilibrium claim."
            write_json(args.output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return 2
        summary.update(mpec_results(args.output, data, m, args),
            interpretation="Optimistic best response for the specified rival profile and strategy bounds, with SOS1 complementarity and no multiplier bound. Not a multi-investor equilibrium."
            if proven else "Gurobi's incumbent at the time limit, not a proven best response: the optimum lies between "
            "solver_lower_bound and solver_upper_bound. Exported as a candidate (start profile or deviation); bids, if "
            "any, are not selected. Not a multi-investor equilibrium.")
        exact = summary["fixed_capacity_profit_range"]["maximum"]
        if args.balancing_eps > 0:
            summary["linear_minus_exact_profit_eur"] = summary["linear_profit_eur_per_day"] - exact
        # 1e-4 EUR plus 1e-8 of the money involved: Gurobi's 1e-6 feasibility
        # tolerance times prices of 1e3-1e4 EUR/MWh gives errors of about 1e-3 EUR.
        money = abs(summary["embedded_market_cost_eur"]) + abs(summary["linear_profit_eur_per_day"])
        summary["verification_tolerance_eur"] = tolerance = 1e-4 + 1e-8 * money
        summary["verification_passed"] = ((selection is None or selection["status"] == "optimal")
            and summary["profit_identity_error_eur"] <= tolerance
            and abs(summary["reclear_cost_gap_eur"]) <= tolerance
            and abs(summary["primal_dual_gap_eur"]) <= tolerance
            and summary["maximum_constraint_violation"] <= 1e-5
            and summary["maximum_complementarity_product"] <= 1e-4)
        if not proven:
            write_json(args.output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return 2
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("verification_passed", True) else 3


if __name__ == "__main__":
    raise SystemExit(main())
