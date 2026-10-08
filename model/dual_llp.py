"""Explicit dual feasibility and stationarity of primal_llp.py.

Sign convention: L = market_cost - price*nodal_balance_residual
- system_price*system_balance_residual + soc_value*soc_transition_residual
+ cyclic_value*soc_cyclic_residual + sum(mu_upper*(quantity-capacity))
- sum(mu_lower*quantity). All mu are nonnegative; equality multipliers are free.

The standalone dual maximizes dual_value. The MPEC reuses these stationarity
equations and adds complementarity in mpec.py.
"""
import pyomo.environ as pyo

from prepare_input import prepare_input
from primal_llp import add_fixed_capacities


def build_dual(data, profile):
    m = prepare_input(data, profile)
    m.name = "Dual market LLP"
    add_fixed_capacities(m)
    add_dual(m)
    m.objective = pyo.Objective(expr=m.dual_value, sense=pyo.maximize)
    return m


def add_dual(m):
    # Equality multipliers, with nodal price represented directly as the LMP.
    m.price = pyo.Var(m.nodes, m.timesteps, domain=pyo.Reals, doc="Nodal LMP [EUR/MWh]")
    m.system_price = pyo.Var(m.timesteps, domain=pyo.Reals)
    m.soc_value = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.Reals)
    m.cyclic_value = pyo.Var(m.storage_pairs, domain=pyo.Reals)

    # Multipliers for every physical upper and lower bound.
    m.mu_gen_upper = pyo.Var(m.generators, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_gen_lower = pyo.Var(m.generators, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_shed_upper = pyo.Var(m.nodes, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_shed_lower = pyo.Var(m.nodes, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_line_upper = pyo.Var(m.lines, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_line_lower = pyo.Var(m.lines, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_power = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_charge_lower = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_discharge_lower = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_soc_upper = pyo.Var(m.storage_pairs, m.soc_timesteps, domain=pyo.NonNegativeReals)
    m.mu_soc_lower = pyo.Var(m.storage_pairs, m.soc_timesteps, domain=pyo.NonNegativeReals)

    m.stationarity_generation = pyo.Constraint(
        m.generators,
        m.timesteps,
        rule=stationarity_generation_rule,
    )
    m.stationarity_load_shed = pyo.Constraint(
        m.nodes,
        m.timesteps,
        rule=stationarity_load_shed_rule,
    )
    m.stationarity_injection = pyo.Constraint(
        m.nodes,
        m.timesteps,
        rule=stationarity_injection_rule,
    )
    m.stationarity_charge = pyo.Constraint(
        m.storage_pairs,
        m.timesteps,
        rule=stationarity_charge_rule,
    )
    m.stationarity_discharge = pyo.Constraint(
        m.storage_pairs,
        m.timesteps,
        rule=stationarity_discharge_rule,
    )
    m.stationarity_soc = pyo.Constraint(
        m.storage_pairs,
        m.soc_timesteps,
        rule=stationarity_soc_rule,
    )

    m.demand_payment = pyo.Expression(rule=demand_payment_rule)
    m.generation_scarcity_rent = pyo.Expression(rule=generation_scarcity_rent_rule)
    m.shedding_scarcity_rent = pyo.Expression(rule=shedding_scarcity_rent_rule)
    m.transmission_scarcity_rent = pyo.Expression(rule=transmission_scarcity_rent_rule)
    m.storage_scarcity_rent = pyo.Expression(m.investors, rule=storage_scarcity_rent_rule)
    m.owned_generation_rent = pyo.Expression(m.investors, rule=owned_generation_rent_rule)
    m.dual_value = pyo.Expression(rule=dual_value_rule)


def stationarity_generation_rule(m, g, t):
    """dL/dGeneration = marginal cost - price + mu_upper - mu_lower = 0."""
    n = pyo.value(m.generator_node[g])
    return m.generation_cost[g] - m.price[n, t] + m.mu_gen_upper[g, t] - m.mu_gen_lower[g, t] == 0


def stationarity_load_shed_rule(m, n, t):
    return m.voll - m.price[n, t] + m.mu_shed_upper[n, t] - m.mu_shed_lower[n, t] == 0


def stationarity_injection_rule(m, n, t):
    return (
        m.price[n, t] - m.system_price[t]
        + sum(m.ptdf[line, n] * (m.mu_line_upper[line, t] - m.mu_line_lower[line, t]) for line in m.lines)
        == 0
    )


def stationarity_charge_rule(m, i, n, t):
    return (0.5 * m.degradation[i] + m.price[n, t] - m.eta * m.soc_value[i, n, t]
            + m.mu_power[i, n, t] - m.mu_charge_lower[i, n, t] == 0)


def stationarity_discharge_rule(m, i, n, t):
    return (0.5 * m.degradation[i] - m.price[n, t] + m.soc_value[i, n, t] / m.eta
            + m.mu_power[i, n, t] - m.mu_discharge_lower[i, n, t] == 0)


def stationarity_soc_rule(m, i, n, t):
    """Includes both boundary states: t=0 and t=T have cyclic multipliers."""
    marginal_value = m.mu_soc_upper[i, n, t] - m.mu_soc_lower[i, n, t]
    if t in m.timesteps:
        marginal_value += m.soc_value[i, n, t]
    if t + 1 in m.timesteps:
        marginal_value -= m.soc_value[i, n, t + 1]
    if t == 0:
        marginal_value += m.cyclic_value[i, n]
    if t == m.timesteps.last():
        marginal_value -= m.cyclic_value[i, n]
    return marginal_value == 0


def demand_payment_rule(m):
    return sum(m.demand[n, t] * m.price[n, t] for n in m.nodes for t in m.timesteps)


def generation_scarcity_rent_rule(m):
    return sum(m.generation_capacity[g, t] * m.mu_gen_upper[g, t] for g in m.generators for t in m.timesteps)


def shedding_scarcity_rent_rule(m):
    return sum(m.demand[n, t] * m.mu_shed_upper[n, t] for n in m.nodes for t in m.timesteps)


def transmission_scarcity_rent_rule(m):
    return sum(m.line_limit[line] * (m.mu_line_upper[line, t] + m.mu_line_lower[line, t])
               for line in m.lines for t in m.timesteps)


def storage_scarcity_rent_rule(m, investor):
    return (
        sum(m.power_capacity[i, n] * m.mu_power[i, n, t]
            for i, n in m.storage_pairs if i == investor for t in m.timesteps)
        + sum(m.energy_capacity[i, n] * m.mu_soc_upper[i, n, t]
              for i, n in m.storage_pairs if i == investor for t in m.soc_timesteps)
    )


def owned_generation_rent_rule(m, i):
    return sum(m.owned_share[i, g] * m.generation_capacity[g, t] * m.mu_gen_upper[g, t]
               for g in m.generators for t in m.timesteps)


def dual_value_rule(m):
    return (m.demand_payment - m.generation_scarcity_rent - m.shedding_scarcity_rent
            - m.transmission_scarcity_rent - sum(m.storage_scarcity_rent[i] for i in m.investors))


def dual_profit(m, investor):
    """Net daily payoff on a fixed-capacity dual optimal face (a linear expression)."""
    capex = sum(m.cost_power_daily[investor] * m.installed_power[investor, n]
                + m.cost_energy_daily[investor] * m.installed_energy[investor, n] for n in m.nodes)
    return m.storage_scarcity_rent[investor] + m.owned_generation_rent[investor] - capex
