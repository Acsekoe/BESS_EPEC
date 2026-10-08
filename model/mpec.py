"""Capacity-only optimistic MPEC: explicit investment and KKT/Big-M rules.

The physical LLP equations come from primal_llp.py. The stationarity equations
and named multipliers come from dual_llp.py. Every complementarity pair is
declared below: z=1 permits a positive multiplier and forces zero slack;
z=0 permits a positive slack and forces the multiplier to zero.

Run independently: python model/mpec.py --investor I1 --nodes N6 --output ...
"""
import copy
import math

import pyomo.environ as pyo

from prepare_input import prepare_input
from primal_llp import add_primal
from dual_llp import add_dual


def build_mpec(data, profile, investor, *, nodes=None, node_limit=1000.0, dual_m=100000.0):
    if not math.isfinite(dual_m) or dual_m <= 0:
        raise ValueError("The multiplier Big-M must be positive and finite.")
    m = prepare_input(data, profile, active_investor=investor,
                      investment_nodes=nodes, node_limit=node_limit)
    m.name = f"Capacity MPEC: {investor}"

    # Upper-level choices: MW and MWh of the active investor only.
    m.X_power = pyo.Var(m.investment_nodes, domain=pyo.NonNegativeReals)
    m.X_energy = pyo.Var(m.investment_nodes, domain=pyo.NonNegativeReals)
    m.investment_power_limit = pyo.Constraint(m.investment_nodes, rule=investment_power_limit_rule)
    m.investment_energy_limit = pyo.Constraint(
        m.investment_nodes,
        rule=investment_energy_limit_rule,
    )
    m.duration_min = pyo.Constraint(m.investment_nodes, rule=duration_min_rule)
    m.duration_max = pyo.Constraint(m.investment_nodes, rule=duration_max_rule)

    # Active capacity is variable; rival capacity is the fixed input parameter.
    m.power_capacity = pyo.Expression(m.storage_pairs, rule=power_capacity_rule)
    m.energy_capacity = pyo.Expression(m.storage_pairs, rule=energy_capacity_rule)
    add_primal(m)
    add_dual(m)

    # KKT complementarity: explicit binary disjunction for each bound family.
    m.dual_m = pyo.Param(initialize=dual_m, doc="Assumed upper bound on inequality multipliers")
    m.z_gen_lower = pyo.Var(m.generators, m.timesteps, domain=pyo.Binary)
    m.z_gen_upper = pyo.Var(m.generators, m.timesteps, domain=pyo.Binary)
    m.z_shed_lower = pyo.Var(m.nodes, m.timesteps, domain=pyo.Binary)
    m.z_shed_upper = pyo.Var(m.nodes, m.timesteps, domain=pyo.Binary)
    m.z_line_upper = pyo.Var(m.lines, m.timesteps, domain=pyo.Binary)
    m.z_line_lower = pyo.Var(m.lines, m.timesteps, domain=pyo.Binary)
    m.z_charge_lower = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.Binary)
    m.z_discharge_lower = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.Binary)
    m.z_power = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.Binary)
    m.z_soc_lower = pyo.Var(m.storage_pairs, m.soc_timesteps, domain=pyo.Binary)
    m.z_soc_upper = pyo.Var(m.storage_pairs, m.soc_timesteps, domain=pyo.Binary)

    m.comp_gen_lower_slack = pyo.Constraint(
        m.generators,
        m.timesteps,
        rule=comp_gen_lower_slack_rule,
    )
    m.comp_gen_lower_dual = pyo.Constraint(m.generators, m.timesteps, rule=comp_gen_lower_dual_rule)
    m.comp_gen_upper_slack = pyo.Constraint(
        m.generators,
        m.timesteps,
        rule=comp_gen_upper_slack_rule,
    )
    m.comp_gen_upper_dual = pyo.Constraint(m.generators, m.timesteps, rule=comp_gen_upper_dual_rule)
    m.comp_shed_lower_slack = pyo.Constraint(m.nodes, m.timesteps, rule=comp_shed_lower_slack_rule)
    m.comp_shed_lower_dual = pyo.Constraint(m.nodes, m.timesteps, rule=comp_shed_lower_dual_rule)
    m.comp_shed_upper_slack = pyo.Constraint(m.nodes, m.timesteps, rule=comp_shed_upper_slack_rule)
    m.comp_shed_upper_dual = pyo.Constraint(m.nodes, m.timesteps, rule=comp_shed_upper_dual_rule)
    m.comp_line_upper_slack = pyo.Constraint(m.lines, m.timesteps, rule=comp_line_upper_slack_rule)
    m.comp_line_upper_dual = pyo.Constraint(m.lines, m.timesteps, rule=comp_line_upper_dual_rule)
    m.comp_line_lower_slack = pyo.Constraint(m.lines, m.timesteps, rule=comp_line_lower_slack_rule)
    m.comp_line_lower_dual = pyo.Constraint(m.lines, m.timesteps, rule=comp_line_lower_dual_rule)
    m.comp_charge_lower_slack = pyo.Constraint(
        m.storage_pairs,
        m.timesteps,
        rule=comp_charge_lower_slack_rule,
    )
    m.comp_charge_lower_dual = pyo.Constraint(
        m.storage_pairs,
        m.timesteps,
        rule=comp_charge_lower_dual_rule,
    )
    m.comp_discharge_lower_slack = pyo.Constraint(
        m.storage_pairs,
        m.timesteps,
        rule=comp_discharge_lower_slack_rule,
    )
    m.comp_discharge_lower_dual = pyo.Constraint(
        m.storage_pairs,
        m.timesteps,
        rule=comp_discharge_lower_dual_rule,
    )
    m.comp_power_slack = pyo.Constraint(m.storage_pairs, m.timesteps, rule=comp_power_slack_rule)
    m.comp_power_dual = pyo.Constraint(m.storage_pairs, m.timesteps, rule=comp_power_dual_rule)
    m.comp_soc_lower_slack = pyo.Constraint(
        m.storage_pairs,
        m.soc_timesteps,
        rule=comp_soc_lower_slack_rule,
    )
    m.comp_soc_lower_dual = pyo.Constraint(
        m.storage_pairs,
        m.soc_timesteps,
        rule=comp_soc_lower_dual_rule,
    )
    m.comp_soc_upper_slack = pyo.Constraint(
        m.storage_pairs,
        m.soc_timesteps,
        rule=comp_soc_upper_slack_rule,
    )
    m.comp_soc_upper_dual = pyo.Constraint(
        m.storage_pairs,
        m.soc_timesteps,
        rule=comp_soc_upper_dual_rule,
    )

    # Economic objective, linearized using strong duality and storage KKT.
    m.storage_net = pyo.Expression(rule=active_storage_net_rule)
    m.generation_rent = pyo.Expression(expr=m.owned_generation_rent[investor])
    m.capex = pyo.Expression(rule=investment_cost_rule)
    m.profit = pyo.Expression(rule=profit_rule)
    m.objective = pyo.Objective(expr=m.profit, sense=pyo.maximize)
    return m


# Investment constraints and capacities seen by the ISO.
def investment_power_limit_rule(m, n):
    return m.X_power[n] <= m.headroom[n]


def investment_energy_limit_rule(m, n):
    return m.X_energy[n] <= m.ratio_max[m.active_investor] * m.headroom[n]


def duration_min_rule(m, n):
    return m.X_energy[n] >= m.ratio_min[m.active_investor] * m.X_power[n]


def duration_max_rule(m, n):
    return m.X_energy[n] <= m.ratio_max[m.active_investor] * m.X_power[n]


def power_capacity_rule(m, i, n):
    return m.X_power[n] if i == m.active_investor else m.installed_power[i, n]


def energy_capacity_rule(m, i, n):
    return m.X_energy[n] if i == m.active_investor else m.installed_energy[i, n]


# Generation: 0 <= generation <= available capacity.
def comp_gen_lower_slack_rule(m, g, t):
    return m.generation[g, t] <= m.generation_capacity[g, t] * (1 - m.z_gen_lower[g, t])


def comp_gen_lower_dual_rule(m, g, t):
    return m.mu_gen_lower[g, t] <= m.dual_m * m.z_gen_lower[g, t]


def comp_gen_upper_slack_rule(m, g, t):
    return m.generation_capacity[g, t] - m.generation[g, t] <= m.generation_capacity[g, t] * (1 - m.z_gen_upper[g, t])


def comp_gen_upper_dual_rule(m, g, t):
    return m.mu_gen_upper[g, t] <= m.dual_m * m.z_gen_upper[g, t]


# Load shedding: 0 <= load_shed <= demand.
def comp_shed_lower_slack_rule(m, n, t):
    return m.load_shed[n, t] <= m.demand[n, t] * (1 - m.z_shed_lower[n, t])


def comp_shed_lower_dual_rule(m, n, t):
    return m.mu_shed_lower[n, t] <= m.dual_m * m.z_shed_lower[n, t]


def comp_shed_upper_slack_rule(m, n, t):
    return m.demand[n, t] - m.load_shed[n, t] <= m.demand[n, t] * (1 - m.z_shed_upper[n, t])


def comp_shed_upper_dual_rule(m, n, t):
    return m.mu_shed_upper[n, t] <= m.dual_m * m.z_shed_upper[n, t]


# Transmission: -line_limit <= line_flow <= line_limit.
def comp_line_upper_slack_rule(m, line, t):
    return m.line_limit[line] - m.line_flow[line, t] <= 2 * m.line_limit[line] * (1 - m.z_line_upper[line, t])


def comp_line_upper_dual_rule(m, line, t):
    return m.mu_line_upper[line, t] <= m.dual_m * m.z_line_upper[line, t]


def comp_line_lower_slack_rule(m, line, t):
    return m.line_limit[line] + m.line_flow[line, t] <= 2 * m.line_limit[line] * (1 - m.z_line_lower[line, t])


def comp_line_lower_dual_rule(m, line, t):
    return m.mu_line_lower[line, t] <= m.dual_m * m.z_line_lower[line, t]


# Storage power: charge >= 0, discharge >= 0, charge + discharge <= power.
def comp_charge_lower_slack_rule(m, i, n, t):
    return m.charge[i, n, t] <= m.power_max[i, n] * (1 - m.z_charge_lower[i, n, t])


def comp_charge_lower_dual_rule(m, i, n, t):
    return m.mu_charge_lower[i, n, t] <= m.dual_m * m.z_charge_lower[i, n, t]


def comp_discharge_lower_slack_rule(m, i, n, t):
    return m.discharge[i, n, t] <= m.power_max[i, n] * (1 - m.z_discharge_lower[i, n, t])


def comp_discharge_lower_dual_rule(m, i, n, t):
    return m.mu_discharge_lower[i, n, t] <= m.dual_m * m.z_discharge_lower[i, n, t]


def comp_power_slack_rule(m, i, n, t):
    return (m.power_capacity[i, n] - m.charge[i, n, t] - m.discharge[i, n, t]
            <= m.power_max[i, n] * (1 - m.z_power[i, n, t]))


def comp_power_dual_rule(m, i, n, t):
    return m.mu_power[i, n, t] <= m.dual_m * m.z_power[i, n, t]


# Storage energy: 0 <= SOC <= energy capacity, including t=0 and t=T.
def comp_soc_lower_slack_rule(m, i, n, t):
    return m.soc[i, n, t] <= m.energy_max[i, n] * (1 - m.z_soc_lower[i, n, t])


def comp_soc_lower_dual_rule(m, i, n, t):
    return m.mu_soc_lower[i, n, t] <= m.dual_m * m.z_soc_lower[i, n, t]


def comp_soc_upper_slack_rule(m, i, n, t):
    return m.energy_capacity[i, n] - m.soc[i, n, t] <= m.energy_max[i, n] * (1 - m.z_soc_upper[i, n, t])


def comp_soc_upper_dual_rule(m, i, n, t):
    return m.mu_soc_upper[i, n, t] <= m.dual_m * m.z_soc_upper[i, n, t]


def active_storage_net_rule(m):
    """Active storage revenue minus degradation, using the exact payment identity.

    Demand payment = market cost + generator/line/shedding rents + storage rents.
    Subtract fixed rival storage rents to avoid the nonlinear active X*mu terms.
    """
    return (m.demand_payment - m.market_cost - m.generation_scarcity_rent
            - m.shedding_scarcity_rent - m.transmission_scarcity_rent
            - sum(m.storage_scarcity_rent[i] for i in m.investors if i != m.active_investor))


def investment_cost_rule(m):
    i = m.active_investor
    return sum(m.cost_power_daily[i] * m.X_power[n] + m.cost_energy_daily[i] * m.X_energy[n]
               for n in m.investment_nodes)


def profit_rule(m):
    return m.storage_net + m.generation_rent - m.capex


def complementarity_pairs(m):
    """Explicit (slack, multiplier) pairs for reporting residuals, not construction."""
    for g in m.generators:
        for t in m.timesteps:
            yield m.generation[g, t], m.mu_gen_lower[g, t]
            yield m.generation_capacity[g, t] - m.generation[g, t], m.mu_gen_upper[g, t]
    for n in m.nodes:
        for t in m.timesteps:
            yield m.load_shed[n, t], m.mu_shed_lower[n, t]
            yield m.demand[n, t] - m.load_shed[n, t], m.mu_shed_upper[n, t]
    for line in m.lines:
        for t in m.timesteps:
            yield m.line_limit[line] - m.line_flow[line, t], m.mu_line_upper[line, t]
            yield m.line_limit[line] + m.line_flow[line, t], m.mu_line_lower[line, t]
    for i, n in m.storage_pairs:
        for t in m.timesteps:
            yield m.charge[i, n, t], m.mu_charge_lower[i, n, t]
            yield m.discharge[i, n, t], m.mu_discharge_lower[i, n, t]
            yield m.power_capacity[i, n] - m.charge[i, n, t] - m.discharge[i, n, t], m.mu_power[i, n, t]
        for t in m.soc_timesteps:
            yield m.soc[i, n, t], m.mu_soc_lower[i, n, t]
            yield m.energy_capacity[i, n] - m.soc[i, n, t], m.mu_soc_upper[i, n, t]


def chosen_profile(m):
    result = copy.deepcopy(m._profile)
    for field, variables in (("power_mw", m.X_power), ("energy_mwh", m.X_energy)):
        result[m.active_investor][field] = {
            n: max(0.0, pyo.value(variables[n])) if n in variables else 0.0 for n in m.nodes
        }
    return result


def direct_profit(m):
    """Independent nodal settlement check of the linear profit identity."""
    investor = m.active_investor
    storage_profit = sum(
        m.price[n, t] * (m.discharge[i, n, t] - m.charge[i, n, t])
        - 0.5 * m.degradation[i] * (m.charge[i, n, t] + m.discharge[i, n, t])
        for i, n in m.storage_pairs if i == investor for t in m.timesteps
    )
    generation_profit = sum(
        m.owned_share[investor, g]
        * (m.price[pyo.value(m.generator_node[g]), t] - m.generation_cost[g]) * m.generation[g, t]
        for g in m.generators for t in m.timesteps
    )
    return pyo.value(storage_profit + generation_profit - m.capex)


if __name__ == "__main__":
    import sys
    from run import main
    raise SystemExit(main(["mpec", *sys.argv[1:]]))
