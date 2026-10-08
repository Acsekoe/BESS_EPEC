"""Capacity-only optimistic MPEC: investor problem plus LLP KKT with Big-M.

    max  investor objective                       (6, bilinear: price * quantity)
    s.t. investor constraints                     (1)
         LLP primal feasibility                   (2, primal_llp.py)
         LLP stationarity, multipliers mu >= 0    (3, dual_llp.py)
         LLP complementarity via Big-M            (4)
         variable bounds                          (5)
         profit identity: valid cut               (7, bilinear objective only)

Every complementarity pair is declared below: z=1 permits a positive multiplier
and forces zero slack; z=0 permits a positive slack and forces the multiplier to
zero. The bilinear objective needs Gurobi with NonConvex=2 (see solve.py).

Run independently: python model/mpec.py --investor I1 --nodes N6 --output ...
"""
import copy
import math

import pyomo.environ as pyo

from prepare_input import prepare_input
from primal_llp import add_primal
from dual_llp import add_dual


def build_mpec(data, profile, investor, *, nodes=None, node_limit=1000.0, dual_m=100000.0,
               objective="bilinear"):
    """objective="bilinear" maximizes the investor's nodal settlement directly;
    "linear" maximizes the strong-duality form, which is equal at every KKT point."""
    if not math.isfinite(dual_m) or dual_m <= 0:
        raise ValueError("The multiplier Big-M must be positive and finite.")
    if objective not in ("bilinear", "linear"):
        raise ValueError("The objective must be 'bilinear' or 'linear'.")
    m = prepare_input(data, profile, active_investor=investor,
                      investment_nodes=nodes, node_limit=node_limit)
    m.name = f"Capacity MPEC: {investor}"

    # 1. Investor constraints: MW and MWh of the active investor only.
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

    # 2. LLP primal feasibility: balances, SOC dynamics, and all capacity limits.
    add_primal(m)

    # 3. LLP stationarity: named multipliers (mu >= 0) and dL/d(dispatch) = 0.
    add_dual(m)

    # 4. LLP complementarity: explicit binary Big-M disjunction for each bound family.
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

    # 5. Bounds: finite ranges for both factors of every bilinear objective term.
    # Each is implied by a physical limit, or by stationarity with mu <= dual_m.
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

    # 7. Valid cut: both profit forms are equal at every KKT point, so it removes no
    # feasible solution. It limits the relaxed bilinear objective by the linear form;
    # without it, Gurobi's bound on the N6 case stays ~1e4 times too high.
    if objective == "bilinear":
        m.profit_identity = pyo.Constraint(rule=profit_identity_rule)
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


# Bounds. Gurobi's spatial branch-and-bound needs finite ranges for price and for
# the quantities it multiplies. None of these bounds removes a KKT point that
# satisfies the Big-M assumption mu <= dual_m.
def add_bounds(m):
    for (n, t), price in m.price.items():
        price.setlb(price_lower_bound(m, n))
        price.setub(price_upper_bound(m, n))
    for (g, t), generation in m.generation.items():
        generation.setub(m.generation_capacity[g, t])
    for (i, n, t), charge in m.charge.items():
        charge.setub(m.power_max[i, n])
    for (i, n, t), discharge in m.discharge.items():
        discharge.setub(m.power_max[i, n])
    for multiplier in (m.mu_gen_upper, m.mu_gen_lower, m.mu_shed_upper, m.mu_shed_lower,
                       m.mu_line_upper, m.mu_line_lower, m.mu_power, m.mu_charge_lower,
                       m.mu_discharge_lower, m.mu_soc_upper, m.mu_soc_lower):
        for mu in multiplier.values():
            mu.setub(pyo.value(m.dual_m))


def price_lower_bound(m, n):
    """Load-shed stationarity gives price = voll + mu_shed_upper - mu_shed_lower,
    and each local generator gives price = cost + mu_gen_upper - mu_gen_lower."""
    dual_m = pyo.value(m.dual_m)
    return max([pyo.value(m.voll) - dual_m]
               + [m.generation_cost[g] - dual_m for g in m.generators_at_node[n]])


def price_upper_bound(m, n):
    dual_m = pyo.value(m.dual_m)
    return min([pyo.value(m.voll) + dual_m]
               + [m.generation_cost[g] + dual_m for g in m.generators_at_node[n]])


# Investor objective: nodal settlement of the active investor's storage and generation.
def storage_revenue_rule(m):
    """Bilinear: nodal price times net discharge."""
    return sum(m.price[n, t] * (m.discharge[i, n, t] - m.charge[i, n, t])
               for i, n in m.storage_pairs if i == m.active_investor for t in m.timesteps)


def storage_degradation_rule(m):
    return sum(0.5 * m.degradation[i] * (m.charge[i, n, t] + m.discharge[i, n, t])
               for i, n in m.storage_pairs if i == m.active_investor for t in m.timesteps)


def generation_margin_rule(m):
    """Bilinear: owned share of (nodal price - marginal cost) times generation."""
    i = m.active_investor
    return sum(m.owned_share[i, g] * (m.price[pyo.value(m.generator_node[g]), t] - m.generation_cost[g])
               * m.generation[g, t]
               for g in m.generators if m.owned_share[i, g] > 0 for t in m.timesteps)


def investment_cost_rule(m):
    i = m.active_investor
    return sum(m.cost_power_daily[i] * m.X_power[n] + m.cost_energy_daily[i] * m.X_energy[n]
               for n in m.investment_nodes)


def profit_rule(m):
    return m.storage_revenue - m.storage_degradation + m.generation_margin - m.capex


def profit_linear_rule(m):
    """The same profit without products of variables; equal at every KKT point.

    Strong duality: demand payment = market cost + generator/line/shedding rents
    + storage rents. Subtracting the fixed rival storage rents leaves the active
    storage rent without the bilinear X*mu terms. Owned generation earns its
    capacity scarcity rent.
    """
    return (m.demand_payment - m.market_cost - m.generation_scarcity_rent
            - m.shedding_scarcity_rent - m.transmission_scarcity_rent
            - sum(m.storage_scarcity_rent[i] for i in m.investors if i != m.active_investor)
            + m.owned_generation_rent[m.active_investor] - m.capex)


def profit_identity_rule(m):
    """Settlement profit minus strong-duality profit is a weighted sum of the
    slack*multiplier products of all other assets and lines, which is zero under
    complementarity."""
    return m.profit == m.profit_linear


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


if __name__ == "__main__":
    import sys
    from run import main
    raise SystemExit(main(["mpec", *sys.argv[1:]]))
