"""Capacity-only optimistic MPEC with relaxed complementarity (Scholtes), solved by Ipopt.

    max  investor objective                       (6, as mpec.py)
    s.t. investor constraints                     (1, as mpec.py)
         LLP primal feasibility                   (2, primal_llp.py)
         LLP stationarity, multipliers mu >= 0    (3, dual_llp.py)
         relaxed complementarity                  (4)
         variable bounds                          (5, as mpec.py)

The only difference from mpec.py is block 4. Instead of a binary Big-M
disjunction, every complementarity product is bounded:

    0 <= slack * mu <= epsilon

Both factors are nonnegative, so only the upper bound is written. epsilon = 0
is the exact KKT system, where no constraint qualification holds; epsilon > 0
gives a smooth NLP without binaries. solve_relaxed() solves a decreasing
sequence of epsilon, each warm-started from the previous solution.

The equality slack * mu = epsilon is not used: pairs whose slack is fixed at
zero (zero capacity, zero demand, an uninstalled node) make it infeasible.

Ipopt returns a local optimum of the relaxed problem: no global optimality and
only approximate complementarity. run.py reclears the selected capacity exactly
and reports the investor's exact profit interval there.

Run independently: python model/mpec_relaxed.py --investor I1 --output ...
"""
import copy
import math

import pyomo.environ as pyo

from prepare_input import prepare_input
from primal_llp import add_primal, build_primal
from dual_llp import add_dual, build_dual
from mpec import (add_bounds, duration_max_rule, duration_min_rule, energy_capacity_rule,
                  generation_margin_rule, investment_cost_rule, investment_energy_limit_rule,
                  investment_power_limit_rule, power_capacity_rule, profit_linear_rule, profit_rule,
                  storage_degradation_rule, storage_revenue_rule)
from solve import solve_nlp


def build_mpec_relaxed(data, profile, investor, *, nodes=None, node_limit=1000.0, dual_m=100000.0,
                       objective="bilinear", epsilon=1e-4, balancing_eps=0.0):
    """epsilon is the initial product bound; solve_relaxed() changes it in place.
    balancing_eps > 0 makes LLP prices unique (see primal_llp.py)."""
    if not math.isfinite(dual_m) or dual_m <= 0:
        raise ValueError("The multiplier bound must be positive and finite.")
    if objective not in ("bilinear", "linear"):
        raise ValueError("The objective must be 'bilinear' or 'linear'.")
    m = prepare_input(data, profile, active_investor=investor,
                      investment_nodes=nodes, node_limit=node_limit, balancing_eps=balancing_eps)
    m.name = f"Relaxed capacity MPEC: {investor}"

    # 1. Investor constraints: MW and MWh of the active investor only.
    m.X_power = pyo.Var(m.investment_nodes, domain=pyo.NonNegativeReals)
    m.X_energy = pyo.Var(m.investment_nodes, domain=pyo.NonNegativeReals)
    m.investment_power_limit = pyo.Constraint(m.investment_nodes, rule=investment_power_limit_rule)
    m.investment_energy_limit = pyo.Constraint(m.investment_nodes, rule=investment_energy_limit_rule)
    m.duration_min = pyo.Constraint(m.investment_nodes, rule=duration_min_rule)
    m.duration_max = pyo.Constraint(m.investment_nodes, rule=duration_max_rule)

    # Active capacity is variable; rival capacity is the fixed input parameter.
    m.power_capacity = pyo.Expression(m.storage_pairs, rule=power_capacity_rule)
    m.energy_capacity = pyo.Expression(m.storage_pairs, rule=energy_capacity_rule)

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

    # 5. Bounds: the same as mpec.py. mu <= dual_m keeps the price bounded on
    # degenerate dual faces, so both MPEC versions use the same assumption.
    m.dual_m = pyo.Param(initialize=dual_m, doc="Assumed upper bound on inequality multipliers")
    add_bounds(m)

    # 6. Investor objective: nodal settlement of the active investor's assets.
    m.storage_revenue = pyo.Expression(rule=storage_revenue_rule)
    m.storage_degradation = pyo.Expression(rule=storage_degradation_rule)
    m.generation_margin = pyo.Expression(rule=generation_margin_rule)
    m.capex = pyo.Expression(rule=investment_cost_rule)
    m.profit = pyo.Expression(rule=profit_rule)
    m.profit_linear = pyo.Expression(rule=profit_linear_rule)
    m.objective = pyo.Objective(expr=m.profit if objective == "bilinear" else m.profit_linear,
                                sense=pyo.maximize)
    # No profit identity cut: with epsilon > 0 the two forms differ by a weighted
    # sum of relaxed products, so the cut would not be valid.
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


def initialize_from_market(m, data, profile, start_power):
    """Start at a KKT point: the fixed-capacity market primal and dual optimum
    with start_power MW (minimum duration) at each investment node.

    From the all-zero point, stationarity is violated by about VOLL and Ipopt
    does not regain feasibility. The result depends on this start. Ipopt also
    solves these two LPs, so the relaxed MPEC does not need Gurobi.
    """
    if start_power <= 0:
        raise ValueError("The start capacity must be positive so every active storage pair has a start.")
    i = m.active_investor
    start = copy.deepcopy(profile)
    start[i]["power_mw"] = {n: min(start_power, pyo.value(m.headroom[n])) for n in m.investment_nodes}
    start[i]["energy_mwh"] = {n: pyo.value(m.ratio_min[i]) * p for n, p in start[i]["power_mw"].items()}
    eps = pyo.value(m.balancing_eps)
    primal, dual = build_primal(data, start, eps), build_dual(data, start, eps)
    for model in (primal, dual):
        status = solve_nlp(model).solver.termination_condition
        if status != pyo.TerminationCondition.optimal:
            raise RuntimeError(f"Start market for the relaxed MPEC not solved: {status}")
    for n in m.investment_nodes:
        m.X_power[n].set_value(start[i]["power_mw"][n])
        m.X_energy[n].set_value(start[i]["energy_mwh"][n])
    # Dispatch, prices, and multipliers have the same names and indices in all models.
    for source in (primal, dual):
        for variable in source.component_objects(pyo.Var):
            target = m.component(variable.local_name)
            for index, value in variable.get_values().items():
                if index in target and value is not None:
                    target[index].set_value(value, skip_validation=True)
    return start


def solve_relaxed(m, epsilons, *, seconds=60, solver=None, log_dir=None):
    """Solve for each epsilon in order, warm-started from the previous solution.

    Stops at the first step that is not locally optimal. Returns the last solver
    result and one history record per attempted step.
    """
    history = []
    for k, epsilon in enumerate(epsilons):
        m.epsilon.set_value(epsilon)
        log_file = None if log_dir is None else log_dir / f"ipopt_step{k}.log"
        result = solve_nlp(m, seconds=seconds, solver=solver, log_file=log_file)
        status = str(result.solver.termination_condition)
        history.append(dict(step=k, epsilon=epsilon, status=status,
                            profit_eur_per_day=pyo.value(m.profit) if status == "optimal" else None))
        if status != "optimal":
            break
    return result, history


if __name__ == "__main__":
    import sys
    from run import main
    raise SystemExit(main(["mpec_relaxed", *sys.argv[1:]]))
