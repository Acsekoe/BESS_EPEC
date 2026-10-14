"""Explicit dual feasibility and stationarity of primal_llp.py.

Sign convention: L = market_cost - price*nodal_balance_residual
- system_price*system_balance_residual + soc_value*soc_transition_residual
+ cyclic_value*soc_cyclic_residual + sum(mu_upper*(quantity-capacity))
- sum(mu_lower*quantity). All mu are nonnegative; equality multipliers are free.

The standalone dual maximizes dual_value. The MPEC reuses these stationarity
equations and adds complementarity in mpec.py.

With balancing (eps > 0), stationarity of b gives b = eps * price. The
standalone dual eliminates b, which adds -(eps/2) * sum(price^2) to dual_value:
strictly concave in price, so prices are unique. The MPEC keeps b and the
stationarity equation explicitly.

With a curtailment cost 0.5 * slope * curtailment^2, generator stationarity
gains -slope * curtailment. The standalone dual is then the Dorn dual of the QP:
it keeps generation as a free variable and adds 0.5 * slope * (capacity^2 -
generation^2) to dual_value. In the MPEC, generation is the primal variable.

With a dispatch cost 0.5 * slope * (charge^2 + discharge^2), charge and
discharge stationarity gain slope * charge and slope * discharge. The Dorn
dual keeps charge and discharge as free variables and subtracts
storage_dispatch_rent, 0.5 * slope * (charge^2 + discharge^2), from dual_value.
At a KKT point the same term is the storage's rent from its rising marginal
cost: settlement less degradation and dispatch cost = scarcity rent + this rent.

With a rising generation cost 0.5 * slope * generation^2, generator
stationarity gains slope * generation. The Dorn dual keeps generation as a
free variable and subtracts generation_dispatch_rent, 0.5 * slope *
generation^2, from dual_value; at a KKT point it is the plant's rent from its
rising marginal cost.

With flexible demand q (benefit max_price * q - 0.5 * slope * q^2), its
stationarity reads price = max_price - slope * q - mu_upper + mu_lower: inside
its bounds the price equals the willingness to pay. The Dorn dual keeps q as a
free variable and subtracts flexible_demand_surplus, max_mw * mu_upper + 0.5 *
slope * q^2, from dual_value; at a KKT point that is the load's consumer surplus,
benefit - price * q.

Storage bids replace 0.5 * d in charge and discharge stationarity: -charge_bid
and discharge_offer (truthful: -0.5 * d and 0.5 * d). They do not enter the dual
objective, so strong duality holds for any bids. A storage's settlement less
its true costs is then scarcity rent + dispatch rent + bid margin
(offer - 0.5 d) * discharge + (-bid - 0.5 d) * charge.
"""
import pyomo.environ as pyo

from prepare_input import prepare_input
from primal_llp import add_fixed_storage, curtailment_rule


def build_dual(data, profile, balancing_eps=0.0):
    m = prepare_input(data, profile, balancing_eps=balancing_eps)
    m.name = "Dual market LLP"
    add_fixed_storage(m)
    if any(pyo.value(m.curtailment_slope[g]) > 0 or pyo.value(m.generation_slope[g]) > 0 for g in m.generators):
        m.generation = pyo.Var(m.generators, m.timesteps, domain=pyo.Reals,
                               doc="Dorn dual: generation of plants with a curtailment cost or rising cost [MW]")
        m.curtailment = pyo.Expression(m.generators, m.timesteps, rule=curtailment_rule)
    if any(pyo.value(m.dispatch_slope[i]) > 0 for i, _ in m.storage_pairs):
        m.charge = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.Reals,
                           doc="Dorn dual: charging of storage with a dispatch cost [MW]")
        m.discharge = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.Reals,
                              doc="Dorn dual: discharging of storage with a dispatch cost [MW]")
    m.flexible_demand = pyo.Var(m.flexible_nodes, m.timesteps, domain=pyo.Reals,
                                doc="Dorn dual: flexible demand [MW]")
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
    m.mu_flex_upper = pyo.Var(m.flexible_nodes, m.timesteps, domain=pyo.NonNegativeReals)
    m.mu_flex_lower = pyo.Var(m.flexible_nodes, m.timesteps, domain=pyo.NonNegativeReals)
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
    m.stationarity_flexible_demand = pyo.Constraint(
        m.flexible_nodes,
        m.timesteps,
        rule=stationarity_flexible_demand_rule,
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
    # Balancing b exists only in models that also hold the primal (the MPECs).
    if m.component("balancing") is not None:
        m.stationarity_balancing = pyo.Constraint(
            m.nodes,
            m.timesteps,
            rule=stationarity_balancing_rule,
        )

    m.demand_payment = pyo.Expression(rule=demand_payment_rule)
    m.generation_scarcity_rent = pyo.Expression(rule=generation_scarcity_rent_rule)
    m.generation_dispatch_rent = pyo.Expression(rule=generation_dispatch_rent_rule)
    m.shedding_scarcity_rent = pyo.Expression(rule=shedding_scarcity_rent_rule)
    m.flexible_demand_surplus = pyo.Expression(rule=flexible_demand_surplus_rule)
    m.transmission_scarcity_rent = pyo.Expression(rule=transmission_scarcity_rent_rule)
    m.storage_scarcity_rent = pyo.Expression(m.investors, rule=storage_scarcity_rent_rule)
    m.storage_dispatch_rent = pyo.Expression(m.investors, rule=storage_dispatch_rent_rule)
    m.owned_generation_rent = pyo.Expression(m.investors, rule=owned_generation_rent_rule)
    m.balancing_dual_term = pyo.Expression(rule=balancing_dual_term_rule)
    m.curtailment_dual_term = pyo.Expression(rule=curtailment_dual_term_rule)
    m.dual_value = pyo.Expression(rule=dual_value_rule)


def stationarity_generation_rule(m, g, t):
    """dL/dGeneration = cost + rising cost - slope * curtailment - price + mu_upper - mu_lower = 0.

    The rising cost is generation_slope * generation; one more MW generated
    avoids curtailment_slope * curtailment of curtailment cost."""
    n = pyo.value(m.generator_node[g])
    rising_cost = m.generation_slope[g] * m.generation[g, t] if pyo.value(m.generation_slope[g]) > 0 else 0
    avoided_curtailment_cost = (m.curtailment_slope[g] * m.curtailment[g, t]
                                if pyo.value(m.curtailment_slope[g]) > 0 else 0)
    return (m.generation_cost[g] + rising_cost - avoided_curtailment_cost - m.price[n, t]
            + m.mu_gen_upper[g, t] - m.mu_gen_lower[g, t] == 0)


def stationarity_load_shed_rule(m, n, t):
    return m.voll - m.price[n, t] + m.mu_shed_upper[n, t] - m.mu_shed_lower[n, t] == 0


def stationarity_flexible_demand_rule(m, n, t):
    """dL/dFlexibleDemand = -(max_price - slope * q) + price + mu_upper - mu_lower = 0:
    the load consumes until its willingness to pay falls to the price."""
    return (-(m.flexible_demand_price[n] - m.flexible_demand_slope[n] * m.flexible_demand[n, t]) + m.price[n, t]
            + m.mu_flex_upper[n, t] - m.mu_flex_lower[n, t] == 0)


def stationarity_injection_rule(m, n, t):
    return (
        m.price[n, t] - m.system_price[t]
        + sum(m.ptdf[line, n] * (m.mu_line_upper[line, t] - m.mu_line_lower[line, t]) for line in m.lines)
        == 0
    )


def stationarity_charge_rule(m, i, n, t):
    """dL/dCharge = -charge_bid + slope * charge + price - eta * soc_value
    + mu_power - mu_lower = 0 (truthful bid: -0.5 * degradation)."""
    dispatch_cost = m.dispatch_slope[i] * m.charge[i, n, t] if pyo.value(m.dispatch_slope[i]) > 0 else 0
    return (-m.charge_bid[i, n, t] + dispatch_cost + m.price[n, t] - m.eta * m.soc_value[i, n, t]
            + m.mu_power[i, n, t] - m.mu_charge_lower[i, n, t] == 0)


def stationarity_discharge_rule(m, i, n, t):
    """dL/dDischarge = discharge_offer + slope * discharge - price + soc_value / eta
    + mu_power - mu_lower = 0 (truthful offer: 0.5 * degradation)."""
    dispatch_cost = m.dispatch_slope[i] * m.discharge[i, n, t] if pyo.value(m.dispatch_slope[i]) > 0 else 0
    return (m.discharge_offer[i, n, t] + dispatch_cost - m.price[n, t] + m.soc_value[i, n, t] / m.eta
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


def stationarity_balancing_rule(m, n, t):
    """dL/dBalancing = b/eps - price = 0, written as b - eps * price = 0."""
    return m.balancing[n, t] - m.balancing_eps * m.price[n, t] == 0


def demand_payment_rule(m):
    return sum(m.demand[n, t] * m.price[n, t] for n in m.nodes for t in m.timesteps)


def generation_scarcity_rent_rule(m):
    return sum(m.generation_capacity[g, t] * m.mu_gen_upper[g, t] for g in m.generators for t in m.timesteps)


def shedding_scarcity_rent_rule(m):
    return sum(m.demand[n, t] * m.mu_shed_upper[n, t] for n in m.nodes for t in m.timesteps)


def flexible_demand_surplus_rule(m):
    """max_mw * mu_upper + 0.5 * slope * q^2: the Dorn dual term of the flexible
    demand, and its consumer surplus at a KKT point (stationarity times q gives
    (max_price - price) * q = slope * q^2 + max_mw * mu_upper)."""
    return sum(m.flexible_demand_max[n] * m.mu_flex_upper[n, t]
               + 0.5 * m.flexible_demand_slope[n] * m.flexible_demand[n, t] ** 2
               for n in m.flexible_nodes for t in m.timesteps)


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


def storage_dispatch_rent_rule(m, investor):
    """0.5 * slope * (charge^2 + discharge^2): the Dorn dual term of the dispatch
    cost, and the storage's rent from its rising marginal dispatch cost."""
    if pyo.value(m.dispatch_slope[investor]) == 0:
        return 0
    return sum(0.5 * m.dispatch_slope[i] * (m.charge[i, n, t] ** 2 + m.discharge[i, n, t] ** 2)
               for i, n in m.storage_pairs if i == investor for t in m.timesteps)


def generation_dispatch_rent_rule(m):
    """0.5 * slope * generation^2: the Dorn dual term of the rising generation
    cost, and the plant's rent from its rising marginal cost (price * g - cost
    = 0.5 * slope * g^2 + capacity * mu_upper at a KKT point)."""
    return sum(0.5 * m.generation_slope[g] * m.generation[g, t] ** 2
               for g in m.generators if pyo.value(m.generation_slope[g]) > 0 for t in m.timesteps)


def owned_generation_rent_rule(m, i):
    """Owned settlement margin share * ((price - cost) * generation - owner's
    curtailment cost) at a KKT point. Stationarity and complementarity give
    (price - cost) * generation = capacity * mu_upper - slope * curtailment * generation."""
    scarcity_rent = sum(m.owned_share[i, g] * m.generation_capacity[g, t] * m.mu_gen_upper[g, t]
                        for g in m.generators for t in m.timesteps)
    curtailment_terms = sum(
        m.owned_share[i, g] * (m.curtailment_slope[g] * m.curtailment[g, t] * m.generation[g, t]
                               + 0.5 * m.owner_curtailment_slope[i, g] * m.curtailment[g, t] ** 2)
        for g in m.generators if pyo.value(m.owned_share[i, g] * m.curtailment_slope[g]) > 0
        for t in m.timesteps)
    return scarcity_rent - curtailment_terms


def balancing_dual_term_rule(m):
    """(eps/2) * sum(price^2): the conjugate of the balancing cost b^2/(2 eps)."""
    if pyo.value(m.balancing_eps) == 0:
        return 0
    return sum(0.5 * m.balancing_eps * m.price[n, t] ** 2 for n in m.nodes for t in m.timesteps)


def curtailment_dual_term_rule(m):
    """0.5 * slope * (capacity^2 - generation^2): the Dorn dual term of the
    curtailment cost, concave in generation."""
    return sum(0.5 * m.curtailment_slope[g] * (m.generation_capacity[g, t] ** 2 - m.generation[g, t] ** 2)
               for g in m.generators if pyo.value(m.curtailment_slope[g]) > 0 for t in m.timesteps)


def dual_value_rule(m):
    return (m.demand_payment - m.generation_scarcity_rent - m.generation_dispatch_rent - m.shedding_scarcity_rent
            - m.flexible_demand_surplus - m.transmission_scarcity_rent
            - sum(m.storage_scarcity_rent[i] + m.storage_dispatch_rent[i] for i in m.investors)
            - m.balancing_dual_term + m.curtailment_dual_term)


def dual_profit(m, investor):
    """Net daily payoff on a fixed-capacity dual optimal face (linear without
    dispatch or curtailment cost). Needs the investor's truthful bids: otherwise
    the payoff includes the bid margin, which depends on the dispatch."""
    for i, n in m.storage_pairs:
        for t in m.timesteps:
            if i == investor and (pyo.value(m.fixed_charge_bid[i, n, t]) != -0.5 * pyo.value(m.degradation[i])
                                  or pyo.value(m.fixed_discharge_offer[i, n, t]) != 0.5 * pyo.value(m.degradation[i])):
                raise ValueError(f"dual_profit needs truthful bids; {investor} bids strategically.")
    capex = sum(m.cost_power_daily[investor] * m.installed_power[investor, n]
                + m.cost_energy_daily[investor] * m.installed_energy[investor, n] for n in m.nodes)
    return (m.storage_scarcity_rent[investor] + m.storage_dispatch_rent[investor]
            + m.owned_generation_rent[investor] - capex)
