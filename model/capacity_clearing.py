"""Hosting capacity shared by a uniform clearing price (variational equilibrium).

Node n can host H_n MW of storage power in total. Instead of first come, first
served (rivals' MW reduce the headroom), every investor pays the same price p_n
[EUR/MW/day] per MW at n and may alone build up to H_n. The price clears the node:

    0 <= p_n   complementary to   H_n - sum_i X_power[i, n] >= 0

With one price for all investors this is the variational equilibrium of the
shared constraint (one common multiplier; Rosen 1965, Harker 1991). The energy
market is unchanged; p_n enters each investor's MPEC as a linear fee.

Algorithm (prices at all nodes at once):
    inner: at fixed prices, Gauss-Seidel best responses. Capacity only (truthful
           bids): relaxed MPEC (Ipopt) from the investor's current capacity and
           from start_power, plus building nothing; local, so a round without
           moves means no improvement found, not a proof. With
           strategic bids: the bid MPEC (solve_mpec, Gurobi), proven. An
           investor moves only if the best candidate beats its current exact
           profit (unique-price reclear) by more than gain_tolerance.
    outer: per node, bisection on p_n between a price where the node is
           over-full and one where it is not; all nodes move together, each
           step warm-started from the last one (clear_capacity). Nodes whose
           limit never binds keep p_n = 0.

If a node's demand jumps across H_n (lumpy investment), its bracket shrinks
while the node stays partly empty at the upper price: no uniform price clears
it. The result can depend on the start and the order; it is not unique by proof.

Two other ways to share a node use the same Gauss-Seidel loop (prices None):
    first come, first served: rivals' MW reduce each investor's headroom, so
           whoever holds capacity keeps it (inner_equilibrium).
    hosting rights in lots: allocation {investor: {node: MW}}; at an
           allocated node each investor may build only its own right (0
           without one). Running every assignment of the lots gives each
           investor's profit when it wins or loses them, from which auction
           outcomes and willingness to pay follow.
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


def own_limits(allocation, i):
    """Investor i's caps from the hosting rights {investor: {node: MW}}: at every
    allocated node its own right, 0 without one. None without an allocation."""
    if not allocation:
        return None
    nodes = {n for rights in allocation.values() for n in rights}
    return {n: allocation.get(i, {}).get(n, 0.0) for n in nodes}


def exact_profits(data, profile, prices, balancing_eps):
    """Each investor's market profit (unique-price reclear, after capex) and its
    capacity payment (prices None: first come, first served or allocated
    rights, no payment)."""
    market, price = unique_price_reclear(data, profile, balancing_eps)
    return {i: dict(market_profit=settlement_profit(market, i, price),
                    capacity_payment=sum((prices or {}).get(n, 0.0) * profile[i]["power_mw"].get(n, 0.0)
                                         for n in data["nodes"]))
            for i in profile}


def best_response_candidates(data, profile, i, limits, prices, *, balancing_eps, strategic_bids, seconds,
                             start_power=50.0, own_limit=None):
    """Candidate profiles for investor i, and whether the best one is proven optimal."""
    if strategic_bids:
        m = build_mpec(data, profile, i, node_limit=limits, balancing_eps=balancing_eps,
                       strategic_bids=True, capacity_price=prices, own_limit=own_limit)
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
                               capacity_price=prices, own_limit=own_limit)
        initialize_from_market(m, data, start_profile(m, profile, start_power=start_power, source=source))
        history = solve_relaxed(m, EPSILONS, seconds=seconds)
        if len(history) == len(EPSILONS) and history[-1]["status"] == "optimal":
            candidates.append(chosen_profile(m))
    return candidates, False


def net_profit(data, profile, i, prices, balancing_eps):
    profit = exact_profits(data, profile, prices, balancing_eps)[i]
    return profit["market_profit"] - profit["capacity_payment"]


def inner_equilibrium(data, profile, limits, prices, *, order, balancing_eps, strategic_bids=True,
                      rounds=6, seconds=120, gain_tolerance=1.0, allocation=None, log=print):
    """Gauss-Seidel best responses at fixed capacity prices. Returns the last
    profile, whether a full round had no moves (with strategic bids: all
    proven), and the history. prices None: first come, first served (rivals'
    MW reduce the headroom), with allocation {investor: {node: MW}} each
    investor also within its own hosting rights."""
    history = []
    for round_ in range(1, rounds + 1):
        moved, proven = [], True
        for i in order:
            start = time.time()
            current_net = net_profit(data, profile, i, prices, balancing_eps)
            candidates, proven_here = best_response_candidates(
                data, profile, i, limits, prices, balancing_eps=balancing_eps,
                strategic_bids=strategic_bids, seconds=seconds, own_limit=own_limits(allocation, i))
            scored = sorted(((net_profit(data, c, i, prices, balancing_eps), k)
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
                       nodes=None, seconds=600, mip_gap=1e-6, threads=None, tolerance=1.0, allocation=None,
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
    only for deviations at those nodes (record["deviation_nodes"]). With
    allocation, deviations stay within each investor's hosting rights."""
    records = []
    for i in profile:
        start = time.time()
        built = [n for n, v in profile[i]["power_mw"].items() if v > 0]
        if nodes is not None and set(built) - set(nodes):
            raise ValueError(f"{i} holds capacity at {built}, outside the deviation nodes {nodes}.")
        current = net_profit(data, profile, i, prices, balancing_eps)
        m = build_mpec(data, profile, i, nodes=nodes, node_limit=node_limit, balancing_eps=balancing_eps,
                       strategic_bids=strategic_bids, capacity_price=prices, own_limit=own_limits(allocation, i))
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
        exact_deviation = None if deviation is None else net_profit(data, deviation, i, prices, balancing_eps)
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
                   price_start=50.0, price_max=5000.0, price_tolerance=0.5, max_evaluations=40, rounds=6,
                   seconds=120, gain_tolerance=1.0, log=print):
    """hosting {node: MW} overrides node_limit. One price per node, all searched
    together; each evaluation is an inner equilibrium at the current prices.

    Per node n, low[n] is the highest price known to leave n over-full and
    high[n] the lowest known to fit it (None: none yet). An over-full node's
    price doubles from price_start until a bracket exists, then bisects; a node
    that fits at a positive price bisects downwards. Prices at the other nodes
    shift a node's demand, so a bracket that no longer holds is reopened. Done
    when every node fits and every positive price is within price_tolerance of
    its low[n]; with one scarce node this is the plain bisection."""
    nodes = data["nodes"]
    limits = {n: hosting.get(n, node_limit) for n in nodes}
    prices = {n: 0.0 for n in nodes}
    low, high = {n: 0.0 for n in nodes}, {n: None for n in nodes}
    evaluations = []
    current = profile
    for _ in range(max_evaluations):
        log(f"  prices {({n: p for n, p in prices.items() if p > 0})}")
        current, converged, history = inner_equilibrium(
            data, current, limits, prices, order=order, balancing_eps=balancing_eps,
            strategic_bids=strategic_bids, rounds=rounds, seconds=seconds, gain_tolerance=gain_tolerance, log=log)
        loads = {n: node_load(current, n) for n in nodes}
        evaluations.append(dict(prices=dict(prices), converged=converged, load_mw=loads, history=history,
                                profile=current))
        log(f"  -> converged {converged}, load {({n: round(v, 2) for n, v in loads.items() if v > 1e-6})}")
        over = {n: loads[n] > limits[n] + 1e-6 for n in nodes}
        for n in nodes:
            if over[n]:
                low[n] = prices[n]
                if high[n] is not None and high[n] <= prices[n]:
                    high[n] = None  # the other prices raised n's demand: reopen upwards
            elif prices[n] > 0:
                high[n] = prices[n]
                if low[n] >= prices[n]:
                    low[n] = 0.0  # the other prices lowered n's demand: reopen downwards
        cleared = not any(over.values()) and all(prices[n] - low[n] <= price_tolerance
                                                 for n in nodes if prices[n] > 0)
        if cleared:
            break
        for n in nodes:
            if high[n] is None:
                if over[n]:
                    prices[n] = 2 * prices[n] if prices[n] > 0 else price_start
                    if prices[n] > price_max:
                        raise RuntimeError(f"Node {n} still over-full at {low[n]} EUR/MW/day.")
            elif over[n] or prices[n] - low[n] > price_tolerance:
                prices[n] = high[n] if high[n] - low[n] <= price_tolerance else 0.5 * (low[n] + high[n])
    priced = [n for n in nodes if prices[n] > 0]
    return dict(cleared=cleared, scarce_nodes=priced, hosting_mw={n: limits[n] for n in hosting},
                prices=dict(prices), price_bracket={n: [low[n], prices[n]] for n in priced},
                load_mw={n: loads[n] for n in hosting}, unused_mw={n: limits[n] - loads[n] for n in priced},
                profile=current, inner_converged=converged,
                profits=exact_profits(data, current, prices, balancing_eps)), evaluations
