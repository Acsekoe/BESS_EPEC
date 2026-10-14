"""Hosting capacity shared by a uniform clearing price (variational equilibrium).

Node n can host H_n MW of storage power in total. Instead of first come, first
served (rivals' MW reduce the headroom), every investor pays the same price p_n
[EUR/MW/day] per MW at n and may alone build up to H_n. The price clears the node:

    0 <= p_n   complementary to   H_n - sum_i X_power[i, n] >= 0

With one price for all investors this is the variational equilibrium of the
shared constraint (one common multiplier; Rosen 1965, Harker 1991). The energy
market is unchanged; p_n enters each investor's MPEC as a linear fee.

Algorithm (one scarce node):
    inner: at fixed prices, Gauss-Seidel best responses. Capacity only (truthful
           bids): relaxed MPEC (Ipopt) from the investor's current capacity and
           from start_power, plus building nothing; local, so a round without
           moves means no improvement found, not a proof. With
           strategic bids: the bid MPEC (solve_mpec, Gurobi), proven. An
           investor moves only if the best candidate beats its current exact
           profit (unique-price reclear) by more than gain_tolerance.
    outer: bisection on p_n between a price where the node is over-full and
           one where it is not, each step warm-started from the last one.

If the node's demand jumps across H_n (lumpy investment), the bracket shrinks
while the node stays partly empty at the upper price: no uniform price clears
it. The result can depend on the start and the order; it is not unique by proof.

Two other ways to share a node use the same Gauss-Seidel loop:
    first come, first served: prices None; rivals' MW reduce each investor's
           headroom, so whoever holds capacity keeps it (inner_equilibrium).
    quadratic access cost: access_slope {node: b}; every MW at the node pays
           b * (total MW at the node), the same fee for everyone, and each
           investor sees that its own MW raise it (no hard limit, no dual to
           match across investors).
"""
import copy
import time

from mpec import build_mpec, chosen_profile, solve_mpec
from mpec_relaxed import build_mpec_relaxed, initialize_from_market, solve_relaxed, start_profile
from solution_space import settlement_profit, unique_price_reclear
from solve import load_incumbent, make_solver

EPSILONS = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def node_load(profile, n):
    return sum(c["power_mw"].get(n, 0.0) for c in profile.values())


def exact_profits(data, profile, prices, balancing_eps, access_slope=None):
    """Each investor's market profit (unique-price reclear, after capex) and its
    capacity payment (prices None: first come, first served, no payment). With
    access_slope every MW at node n also pays slope * (total MW at n)."""
    market, price = unique_price_reclear(data, profile, balancing_eps)
    fee = {n: (prices or {}).get(n, 0.0) + (access_slope or {}).get(n, 0.0) * node_load(profile, n)
           for n in data["nodes"]}
    return {i: dict(market_profit=settlement_profit(market, i, price),
                    capacity_payment=sum(fee[n] * profile[i]["power_mw"].get(n, 0.0) for n in data["nodes"]))
            for i in profile}


def best_response_candidates(data, profile, i, limits, prices, *, balancing_eps, strategic_bids, seconds,
                             start_power=50.0, access_slope=None):
    """Candidate profiles for investor i, and whether the best one is proven optimal."""
    if strategic_bids:
        m = build_mpec(data, profile, i, node_limit=limits, balancing_eps=balancing_eps,
                       strategic_bids=True, capacity_price=prices, access_slope=access_slope)
        result, _, selection = solve_mpec(m, seconds=seconds)
        ok = str(result.solver.termination_condition) == "optimal" and selection["status"] == "optimal"
        return ([chosen_profile(m)] if ok else []), ok
    # A relaxed optimum can be worse than building nothing, so zero is a candidate.
    # One start is not enough: from its current capacity an investor's local
    # optimum missed moves worth 48 000 EUR/day on V6, so it also starts fresh.
    zero = copy.deepcopy(profile)
    zero[i]["power_mw"], zero[i]["energy_mwh"] = {}, {}
    candidates = [zero]
    builds = any(v > 1e-6 for v in profile[i]["power_mw"].values())
    for source in ([profile] if builds else []) + [None]:
        m = build_mpec_relaxed(data, profile, i, node_limit=limits, balancing_eps=balancing_eps,
                               capacity_price=prices, access_slope=access_slope)
        initialize_from_market(m, data, start_profile(m, profile, start_power=start_power, source=source))
        history = solve_relaxed(m, EPSILONS, seconds=seconds)
        if len(history) == len(EPSILONS) and history[-1]["status"] == "optimal":
            candidates.append(chosen_profile(m))
    return candidates, False


def net_profit(data, profile, i, prices, balancing_eps, access_slope=None):
    profit = exact_profits(data, profile, prices, balancing_eps, access_slope)[i]
    return profit["market_profit"] - profit["capacity_payment"]


def inner_equilibrium(data, profile, limits, prices, *, order, balancing_eps, strategic_bids=True,
                      rounds=6, seconds=120, gain_tolerance=1.0, access_slope=None, log=print):
    """Gauss-Seidel best responses at fixed capacity prices. Returns the last
    profile, whether a full round had no moves (with strategic bids: all
    proven), and the history. prices None: first come, first served (rivals'
    MW reduce the headroom)."""
    history = []
    for round_ in range(1, rounds + 1):
        moved, proven = [], True
        for i in order:
            start = time.time()
            current_net = net_profit(data, profile, i, prices, balancing_eps, access_slope)
            candidates, proven_here = best_response_candidates(
                data, profile, i, limits, prices, balancing_eps=balancing_eps,
                strategic_bids=strategic_bids, seconds=seconds, access_slope=access_slope)
            scored = sorted(((net_profit(data, c, i, prices, balancing_eps, access_slope), k)
                             for k, c in enumerate(candidates)), reverse=True)
            best_net = scored[0][0] if scored else None
            gain = None if best_net is None else best_net - current_net
            status = "no candidate" if gain is None else ("optimal" if proven_here else "local")
            record = dict(round=round_, investor=i, status=status, current_net_profit=current_net,
                          best_response_net_profit=best_net, gain=gain, moved=False,
                          seconds=round(time.time() - start, 1))
            if gain is None or (strategic_bids and not proven_here):
                proven = False
            if gain is not None and gain > gain_tolerance:
                profile = candidates[scored[0][1]]
                record["moved"] = True
                moved.append(i)
            record["power_mw"] = {n: round(v, 3) for n, v in profile[i]["power_mw"].items() if v > 1e-6}
            history.append(record)
            log(f"    round {round_} {i}: {status}, gain {gain if gain is None else round(gain, 2)}, "
                f"{'moved' if record['moved'] else 'stays'}, MW {record['power_mw']}, {record['seconds']} s")
        if not moved and proven:
            return profile, True, history
    return profile, False, history


def verify_equilibrium(data, profile, *, balancing_eps, node_limit=1000.0, prices=None, strategic_bids=False,
                       nodes=None, seconds=600, mip_gap=1e-6, threads=None, tolerance=1.0, access_slope=None,
                       log=print):
    """Gurobi check of a candidate equilibrium, e.g. one found with local best
    responses: each investor's SOS1 MPEC against the others' fixed capacities.

    Verdict per investor, against its current exact net profit (reclear):
        proven   Gurobi's upper bound <= current + tolerance: no profitable deviation exists.
        refuted  Gurobi found a solution better by more than tolerance: not an equilibrium.
        open     neither within the time limit.
    The candidate is a proven equilibrium only if every investor is proven.

    Cutoff = current + tolerance: Gurobi prunes every branch that cannot beat
    it, and status cutoff (Pyomo: minFunctionValue) means proven. A MIP start at
    the candidate would add nothing, since its objective lies below the cutoff.
    nodes restricts the deviations (default: all nodes); a verdict then holds
    only for deviations at those nodes (record["deviation_nodes"])."""
    records = []
    for i in profile:
        start = time.time()
        built = [n for n, v in profile[i]["power_mw"].items() if v > 0]
        if nodes is not None and set(built) - set(nodes):
            raise ValueError(f"{i} holds capacity at {built}, outside the deviation nodes {nodes}.")
        current = net_profit(data, profile, i, prices, balancing_eps, access_slope)
        m = build_mpec(data, profile, i, nodes=nodes, node_limit=node_limit, balancing_eps=balancing_eps,
                       strategic_bids=strategic_bids, capacity_price=prices, access_slope=access_slope)
        solver = make_solver()
        if threads:
            solver.options["Threads"] = threads
        solver.options["Cutoff"] = current + tolerance
        result, (incumbent, bound), _ = solve_mpec(m, seconds=seconds, mip_gap=mip_gap, solver=solver)
        status = str(result.solver.termination_condition)
        if status == "minFunctionValue":  # Gurobi CUTOFF: nothing beats current + tolerance
            bound = current + tolerance
        # A refutation is confirmed by the exact reclear of the deviation, not
        # only by the solver's objective value.
        deviation = None
        if incumbent is not None and (status == "optimal" or load_incumbent(solver)):
            deviation = chosen_profile(m)
        exact_deviation = None if deviation is None else net_profit(data, deviation, i, prices, balancing_eps,
                                                                    access_slope)
        if bound is not None and bound <= current + tolerance:
            verdict = "proven"
        elif exact_deviation is not None and exact_deviation > current + tolerance:
            verdict = "refuted"
        else:
            verdict = "open"
        record = dict(investor=i, verdict=verdict, status=status, deviation_nodes=nodes or data["nodes"],
                      current_net_profit=current, solver_incumbent=incumbent, solver_upper_bound=bound,
                      exact_deviation_net_profit=exact_deviation,
                      best_deviation_gain=None if exact_deviation is None else exact_deviation - current,
                      largest_possible_gain=None if bound is None else bound - current,
                      seconds=round(time.time() - start, 1))
        if deviation is not None:
            record["deviation"] = {field: {n: round(v, 3) for n, v in deviation[i][field].items() if v > 1e-6}
                                   for field in ("power_mw", "energy_mwh")}
        records.append(record)
        log(f"  {i}: {verdict}, current {current:.2f}, incumbent {incumbent}, bound {bound}, {record['seconds']} s")
    verdicts = {r["verdict"] for r in records}
    overall = "refuted" if "refuted" in verdicts else ("proven" if verdicts == {"proven"} else "open")
    return overall, records


def clear_capacity(data, profile, hosting, *, order, balancing_eps, node_limit=1000.0, strategic_bids=True,
                   price_start=50.0, price_max=5000.0, price_tolerance=0.5, rounds=6, seconds=120,
                   gain_tolerance=1.0, access_slope=None, log=print):
    """hosting {node: MW} overrides node_limit at the scarce nodes. Bisection on the
    price of the one node that is over-full at zero prices."""
    limits = {n: hosting.get(n, node_limit) for n in data["nodes"]}
    evaluations = []

    def evaluate(prices, start_profile):
        log(f"  prices {prices}")
        result, converged, history = inner_equilibrium(
            data, start_profile, limits, prices, order=order, balancing_eps=balancing_eps,
            strategic_bids=strategic_bids, rounds=rounds, seconds=seconds, gain_tolerance=gain_tolerance,
            access_slope=access_slope, log=log)
        loads = {n: node_load(result, n) for n in data["nodes"]}
        evaluations.append(dict(prices=dict(prices), converged=converged, load_mw=loads, history=history, profile=result))
        log(f"  -> converged {converged}, load {({n: round(v, 2) for n, v in loads.items() if v > 1e-6})}")
        return result, converged, loads

    prices = {n: 0.0 for n in data["nodes"]}
    current, converged, loads = evaluate(prices, profile)
    scarce = [n for n in data["nodes"] if loads[n] > limits[n] + 1e-6]
    if not scarce:
        return dict(scarce_node=None, prices=prices, profile=current, inner_converged=converged,
                    profits=exact_profits(data, current, prices, balancing_eps, access_slope)), evaluations
    if len(scarce) > 1:
        raise NotImplementedError(f"Several nodes over-full at zero prices ({scarce}); bisection handles one.")
    n = scarce[0]
    low, high, high_result = 0.0, None, None  # low: over-full; high: within the limit
    price = price_start
    while high is None:
        prices[n] = price
        current, converged, loads = evaluate(prices, current)
        if loads[n] > limits[n] + 1e-6:
            low, price = price, 2 * price
            if price > price_max:
                raise RuntimeError(f"Node {n} still over-full at {low} EUR/MW/day.")
        else:
            high, high_result = price, (current, converged, loads)
    while high - low > price_tolerance:
        prices[n] = 0.5 * (low + high)
        current, converged, loads = evaluate(prices, current)
        if loads[n] > limits[n] + 1e-6:
            low = prices[n]
        else:
            high, high_result = prices[n], (current, converged, loads)
    prices[n] = high
    current, converged, loads = high_result
    return dict(scarce_node=n, hosting_mw=limits[n], prices=dict(prices), price_bracket=[low, high],
                load_mw=loads[n], unused_mw=limits[n] - loads[n], profile=current, inner_converged=converged,
                profits=exact_profits(data, current, prices, balancing_eps, access_slope)), evaluations
