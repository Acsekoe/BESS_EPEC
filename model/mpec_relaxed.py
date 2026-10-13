"""Capacity (and bid) MPEC with relaxed complementarity (Scholtes), solved by Ipopt.

    max  investor profit, strong-duality form     (6, mpec.py)
    s.t. investor decisions                       (1, mpec.py; optional strategic bids)
         LLP primal feasibility                   (2, primal_llp.py)
         LLP stationarity, multipliers mu >= 0    (3, dual_llp.py)
         relaxed complementarity                  (4)
         physical variable bounds                 (5, mpec.py)

Only block 4 differs from mpec.py. Instead of an SOS1 set, each pair's product
is bounded:

    slack * mu <= epsilon

Both factors are nonnegative, so only the upper bound is written. epsilon = 0
is the exact KKT system, where no constraint qualification holds; epsilon > 0
gives a smooth NLP without binaries (Scholtes 2001). solve_relaxed() solves a
decreasing sequence of epsilon, each warm-started from the previous solution.
The equality slack * mu = epsilon is not used: pairs whose slack is fixed at
zero (an uninstalled node) would make it infeasible.

Degenerate pairs: shedding at zero demand and a plant at zero capacity have
equal lower and upper bounds, so their two multipliers are undetermined.
Gurobi's presolve removes them; Ipopt has no presolve, so fix_degenerate_pairs()
writes each as the equality quantity = 0 with one free multiplier.

Limits:
- Ipopt returns a local optimum of the relaxed problem from one start
  (initialize_from_market); other starts can give other results.
- As epsilon -> 0, limit points are in general only C-stationary (Scholtes 2001;
  Ralph and Wright 2004), not necessarily local optima of the MPEC. This fails
  mostly at pairs with slack = mu = 0, which is where optima at a supply step sit.
- At epsilon > 0 the objective can exceed the true profit by using the
  relaxation. run.py therefore reclears the chosen capacity (and bids) exactly
  and reports that profit.
- One epsilon for all pairs, although multipliers reach about 1e4 (shedding)
  and slacks hundreds of MW.

Run independently: python model/mpec_relaxed.py --investor I1 --output ...
"""
import copy

import pyomo.environ as pyo

from prepare_input import prepare_input
from primal_llp import add_primal, build_primal
from dual_llp import add_dual, build_dual
from mpec import add_investment, add_investor_profit, add_physical_bounds
from solve import make_nlp_solver, solve_nlp


def build_mpec_relaxed(data, profile, investor, *, nodes=None, node_limit=1000.0, balancing_eps=0.0,
                       strategic_bids=False, epsilon=1.0):
    """epsilon is the initial product bound; solve_relaxed() changes it in place."""
    m = prepare_input(data, profile, active_investor=investor, investment_nodes=nodes, node_limit=node_limit,
                      balancing_eps=balancing_eps)
    m.name = f"Relaxed {'capacity and bid' if strategic_bids else 'capacity'} MPEC: {investor}"

    # 1. Investor decisions: MW and MWh (and bids) of the active investor only.
    add_investment(m, strategic_bids)

    # 2. LLP primal feasibility: balances, SOC dynamics, and all capacity limits.
    add_primal(m)

    # 3. LLP stationarity: named multipliers (mu >= 0) and dL/d(dispatch) = 0.
    add_dual(m)

    # 4. Relaxed complementarity: slack * mu <= epsilon for each bound family.
    m.epsilon = pyo.Param(initialize=epsilon, mutable=True, doc="Bound on each complementarity product")
    m.comp_gen_lower = pyo.Constraint(m.generators, m.timesteps, rule=comp_gen_lower_rule)
    m.comp_gen_upper = pyo.Constraint(m.generators, m.timesteps, rule=comp_gen_upper_rule)
    m.comp_shed_lower = pyo.Constraint(m.nodes, m.timesteps, rule=comp_shed_lower_rule)
    m.comp_shed_upper = pyo.Constraint(m.nodes, m.timesteps, rule=comp_shed_upper_rule)
    m.comp_line_upper = pyo.Constraint(m.lines, m.timesteps, rule=comp_line_upper_rule)
    m.comp_line_lower = pyo.Constraint(m.lines, m.timesteps, rule=comp_line_lower_rule)
    m.comp_charge_lower = pyo.Constraint(m.storage_pairs, m.timesteps, rule=comp_charge_lower_rule)
    m.comp_discharge_lower = pyo.Constraint(m.storage_pairs, m.timesteps, rule=comp_discharge_lower_rule)
    m.comp_power = pyo.Constraint(m.storage_pairs, m.timesteps, rule=comp_power_rule)
    m.comp_soc_lower = pyo.Constraint(m.storage_pairs, m.soc_timesteps, rule=comp_soc_lower_rule)
    m.comp_soc_upper = pyo.Constraint(m.storage_pairs, m.soc_timesteps, rule=comp_soc_upper_rule)
    fix_degenerate_pairs(m)

    # 5. Physical bounds implied by the market constraints, stated on the variables.
    add_physical_bounds(m)

    # 6. Investor objective as in mpec.py. At epsilon > 0 profit_linear differs from
    # the settlement profit by a weighted sum of the relaxed products.
    add_investor_profit(m)
    m.objective = pyo.Objective(expr=m.profit_linear, sense=pyo.maximize)
    return m


# Generation: 0 <= generation <= available capacity.
def comp_gen_lower_rule(m, g, t):
    return m.generation[g, t] * m.mu_gen_lower[g, t] <= m.epsilon


def comp_gen_upper_rule(m, g, t):
    return (m.generation_capacity[g, t] - m.generation[g, t]) * m.mu_gen_upper[g, t] <= m.epsilon


# Load shedding: 0 <= load_shed <= demand.
def comp_shed_lower_rule(m, n, t):
    return m.load_shed[n, t] * m.mu_shed_lower[n, t] <= m.epsilon


def comp_shed_upper_rule(m, n, t):
    return (m.demand[n, t] - m.load_shed[n, t]) * m.mu_shed_upper[n, t] <= m.epsilon


# Transmission: -line_limit <= line_flow <= line_limit.
def comp_line_upper_rule(m, line, t):
    return (m.line_limit[line] - m.line_flow[line, t]) * m.mu_line_upper[line, t] <= m.epsilon


def comp_line_lower_rule(m, line, t):
    return (m.line_limit[line] + m.line_flow[line, t]) * m.mu_line_lower[line, t] <= m.epsilon


# Storage power: charge >= 0, discharge >= 0, charge + discharge <= power.
def comp_charge_lower_rule(m, i, n, t):
    return m.charge[i, n, t] * m.mu_charge_lower[i, n, t] <= m.epsilon


def comp_discharge_lower_rule(m, i, n, t):
    return m.discharge[i, n, t] * m.mu_discharge_lower[i, n, t] <= m.epsilon


def comp_power_rule(m, i, n, t):
    return ((m.power_capacity[i, n] - m.charge[i, n, t] - m.discharge[i, n, t])
            * m.mu_power[i, n, t] <= m.epsilon)


# Storage energy: 0 <= SOC <= energy capacity, including t=0 and t=T.
def comp_soc_lower_rule(m, i, n, t):
    return m.soc[i, n, t] * m.mu_soc_lower[i, n, t] <= m.epsilon


def comp_soc_upper_rule(m, i, n, t):
    return (m.energy_capacity[i, n] - m.soc[i, n, t]) * m.mu_soc_upper[i, n, t] <= m.epsilon


def fix_degenerate_pairs(m):
    """0 <= quantity <= 0 (shedding at zero demand, a plant at zero capacity):
    both multipliers enter stationarity only as mu_upper - mu_lower. The quantity
    and mu_upper are fixed at zero, and mu_lower becomes the free multiplier of
    quantity = 0. Both relaxed products and the upper limit are dropped."""
    for n, t in m.load_shed:
        if pyo.value(m.demand[n, t]) == 0:
            m.load_shed[n, t].fix(0.0)
            m.load_shed_limit[n, t].deactivate()
            m.mu_shed_upper[n, t].fix(0.0)
            m.mu_shed_lower[n, t].domain = pyo.Reals
            m.comp_shed_lower[n, t].deactivate()
            m.comp_shed_upper[n, t].deactivate()
    for g, t in m.generation:
        if pyo.value(m.generation_capacity[g, t]) == 0:
            m.generation[g, t].fix(0.0)
            m.generation_limit[g, t].deactivate()
            m.mu_gen_upper[g, t].fix(0.0)
            m.mu_gen_lower[g, t].domain = pyo.Reals
            m.comp_gen_lower[g, t].deactivate()
            m.comp_gen_upper[g, t].deactivate()


def start_profile(m, profile, *, start_power=50.0, source=None):
    """The fixed-capacity market to start from: the active investor's MW, MWh and
    bids from source (another profile, e.g. a step search result), or else
    start_power MW at minimum duration at each investment node. Rivals are
    always those of profile."""
    i = m.active_investor
    start = copy.deepcopy(profile)
    if source is not None:
        for field in ("power_mw", "energy_mwh", "charge_bid_eur_per_mwh", "discharge_offer_eur_per_mwh"):
            start[i][field] = {n: value for n, value in source[i].get(field, {}).items() if n in m.investment_nodes}
    else:
        if start_power <= 0:
            raise ValueError("The start capacity must be positive so every active storage pair has a start.")
        start[i]["power_mw"] = {n: min(start_power, pyo.value(m.headroom[n])) for n in m.investment_nodes}
        start[i]["energy_mwh"] = {n: pyo.value(m.ratio_min[i]) * p for n, p in start[i]["power_mw"].items()}
    return start


def initialize_from_market(m, data, start):
    """Start at a KKT point: the fixed-capacity market primal and dual optimum at
    the start profile. From the all-zero point stationarity is violated by about
    VOLL and Ipopt does not regain feasibility. Ipopt also solves these two
    markets, so the relaxed MPEC does not need Gurobi."""
    i = m.active_investor
    eps = pyo.value(m.balancing_eps)
    primal, dual = build_primal(data, start, eps), build_dual(data, start, eps)
    for model in (primal, dual):
        status = solve_nlp(model).solver.termination_condition
        if status != pyo.TerminationCondition.optimal:
            raise RuntimeError(f"Start market for the relaxed MPEC not solved: {status}")
    for n in m.investment_nodes:
        m.X_power[n].set_value(start[i]["power_mw"].get(n, 0.0))
        m.X_energy[n].set_value(start[i]["energy_mwh"].get(n, 0.0))
    if m.component("X_charge_bid") is not None:
        truthful = 0.5 * pyo.value(m.degradation[i])
        for n in m.investment_nodes:
            for t in m.timesteps:
                m.X_charge_bid[n, t].set_value(
                    start[i].get("charge_bid_eur_per_mwh", {}).get(n, {}).get(str(t), -truthful))
                m.X_discharge_offer[n, t].set_value(
                    start[i].get("discharge_offer_eur_per_mwh", {}).get(n, {}).get(str(t), truthful))
    for variable in (m.charge, m.discharge, m.soc):  # storage absent from the start market at zero capacity
        for v in variable.values():
            v.set_value(0.0)
    # Dispatch, prices, and multipliers have the same names and indices in all models.
    for source in (primal, dual):
        for variable in source.component_objects(pyo.Var):
            target = m.component(variable.local_name)
            for index, value in variable.get_values().items():
                if index in target and value is not None and not target[index].fixed:
                    target[index].set_value(value, skip_validation=True)


def solve_relaxed(m, epsilons, *, seconds=60, solver=None, log_dir=None):
    """Solve for each epsilon in order, warm-started from the previous solution.

    Stops at the first step that is not locally optimal. Returns one history
    record per attempted step.
    """
    if solver is None:
        # Ipopt relaxes bounds by 1e-8; a quantity at -1e-8 times a multiplier of
        # about VOLL leaves a product of -1e-5. Project the result onto the bounds.
        solver = make_nlp_solver()
        solver.options["honor_original_bounds"] = "yes"
    history = []
    for k, epsilon in enumerate(epsilons):
        m.epsilon.set_value(epsilon)
        log_file = None if log_dir is None else log_dir / f"ipopt_step{k}.log"
        result = solve_nlp(m, seconds=seconds, solver=solver, log_file=log_file)
        status = str(result.solver.termination_condition)
        optimal = status == "optimal"
        history.append(dict(step=k, epsilon=epsilon, status=status,
                            linear_profit_eur_per_day=pyo.value(m.profit_linear) if optimal else None,
                            profit_eur_per_day=pyo.value(m.profit) if optimal else None))
        if not optimal:
            break
    return history


if __name__ == "__main__":
    import sys
    from run import main
    raise SystemExit(main(["mpec_relaxed", *sys.argv[1:]]))
