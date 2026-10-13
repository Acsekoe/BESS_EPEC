"""Capacity MPEC: investor problem plus LLP KKT with SOS1 complementarity.

    max  investor profit, strong-duality form     (6)
    s.t. investor decisions                       (1)
         LLP primal feasibility                   (2, primal_llp.py)
         LLP stationarity, multipliers mu >= 0    (3, dual_llp.py)
         LLP complementarity via SOS1             (4)
         physical variable bounds                 (5)

Each market inequality and its multiplier form one complementarity pair,
declared below. Each pair {slack, multiplier} is a special ordered set of
type 1: at most one member is nonzero, and Gurobi branches on the set directly
(slack = 0 or mu = 0). There are no binaries and no bound on the multipliers.
Prices are therefore unbounded too, so the objective is the strong-duality
profit, which contains no price * quantity products; the direct nodal
settlement is kept as an expression to check it.

The investor chooses MW and MWh per node. With strategic_bids it also chooses
an hourly charge bid and discharge offer per node, which replace its truthful
bids in the market (primal_llp.py). The bids enter only charge and discharge
stationarity, linearly; the strong-duality profit does not contain them, so the
MPEC keeps its structure. mpec_relaxed.py replaces block 4 by relaxed products.

Run independently: python model/mpec.py --investor I1 --nodes N9 --output ...
"""
import copy

import pyomo.environ as pyo

from prepare_input import prepare_input
from primal_llp import add_primal
from dual_llp import add_dual


def build_mpec(data, profile, investor, *, nodes=None, node_limit=1000.0, balancing_eps=0.0,
               strategic_bids=False):
    """balancing_eps > 0 makes LLP prices unique (see primal_llp.py)."""
    m = prepare_input(data, profile, active_investor=investor, investment_nodes=nodes, node_limit=node_limit,
                      balancing_eps=balancing_eps)
    m.name = f"{'Capacity and bid' if strategic_bids else 'Capacity'} MPEC: {investor}"

    # 1. Investor decisions: MW and MWh (and bids) of the active investor only.
    # Rival capacities and bids are fixed input parameters.
    add_investment(m, strategic_bids)

    # 2. LLP primal feasibility: balances, SOC dynamics, and all capacity limits.
    add_primal(m)

    # 3. LLP stationarity: named multipliers (mu >= 0) and dL/d(dispatch) = 0,
    # including balancing = eps * price when eps > 0 (free, so no complementarity).
    add_dual(m)

    # 4. LLP complementarity: one SOS1 set {slack, multiplier} per market inequality.
    add_sos1_complementarity(m)

    # 5. Physical bounds implied by the market constraints, stated on the variables.
    add_physical_bounds(m)

    # 6. Investor objective. profit is the direct nodal settlement of the active
    # investor's assets; profit_linear equals it at every KKT point and is maximized.
    add_investor_profit(m)
    m.objective = pyo.Objective(expr=m.profit_linear, sense=pyo.maximize)
    return m


def add_investment(m, strategic_bids=False):
    m.X_power = pyo.Var(m.investment_nodes, domain=pyo.NonNegativeReals)
    m.X_energy = pyo.Var(m.investment_nodes, domain=pyo.NonNegativeReals)
    m.investment_power_limit = pyo.Constraint(m.investment_nodes, rule=investment_power_limit_rule)
    m.investment_energy_limit = pyo.Constraint(
        m.investment_nodes,
        rule=investment_energy_limit_rule,
    )
    m.duration_min = pyo.Constraint(m.investment_nodes, rule=duration_min_rule)
    m.duration_max = pyo.Constraint(m.investment_nodes, rule=duration_max_rule)
    if strategic_bids:
        # Bids within the market's price range [-VOLL, VOLL]; an offer above
        # every price withholds the storage.
        bid_range = (-pyo.value(m.voll), pyo.value(m.voll))
        m.X_charge_bid = pyo.Var(m.investment_nodes, m.timesteps, bounds=bid_range,
                                 doc="Active investor's charge bid, price to buy [EUR/MWh]")
        m.X_discharge_offer = pyo.Var(m.investment_nodes, m.timesteps, bounds=bid_range,
                                      doc="Active investor's discharge offer, price to sell [EUR/MWh]")
        m.bid_no_loop = pyo.Constraint(m.investment_nodes, m.timesteps, rule=bid_no_loop_rule)

    # Active capacity and bids are variable; rival ones are fixed input parameters.
    m.power_capacity = pyo.Expression(m.storage_pairs, rule=power_capacity_rule)
    m.energy_capacity = pyo.Expression(m.storage_pairs, rule=energy_capacity_rule)
    m.charge_bid = pyo.Expression(m.storage_pairs, m.timesteps, rule=charge_bid_rule)
    m.discharge_offer = pyo.Expression(m.storage_pairs, m.timesteps, rule=discharge_offer_rule)


def add_investor_profit(m):
    m.storage_revenue = pyo.Expression(rule=storage_revenue_rule)
    m.storage_degradation = pyo.Expression(rule=storage_degradation_rule)
    m.storage_dispatch_cost = pyo.Expression(rule=storage_dispatch_cost_rule)
    m.generation_margin = pyo.Expression(rule=generation_margin_rule)
    m.capex = pyo.Expression(rule=investment_cost_rule)
    m.profit = pyo.Expression(rule=profit_rule)
    m.profit_linear = pyo.Expression(rule=profit_linear_rule)


def add_sos1_complementarity(m):
    """SOS1 set {slack, multiplier} for each bound family. SOS members must be
    variables, so a slack that is an expression gets an explicit nonnegative
    slack variable and a defining equation; lower-bound slacks are the dispatch
    variables themselves."""
    m.slack_gen_upper = pyo.Var(m.generators, m.timesteps, domain=pyo.NonNegativeReals)
    m.slack_shed_upper = pyo.Var(m.nodes, m.timesteps, domain=pyo.NonNegativeReals)
    m.slack_line_upper = pyo.Var(m.lines, m.timesteps, domain=pyo.NonNegativeReals)
    m.slack_line_lower = pyo.Var(m.lines, m.timesteps, domain=pyo.NonNegativeReals)
    m.slack_power = pyo.Var(m.storage_pairs, m.timesteps, domain=pyo.NonNegativeReals)
    m.slack_soc_upper = pyo.Var(m.storage_pairs, m.soc_timesteps, domain=pyo.NonNegativeReals)

    m.slack_gen_upper_definition = pyo.Constraint(m.generators, m.timesteps, rule=slack_gen_upper_rule)
    m.slack_shed_upper_definition = pyo.Constraint(m.nodes, m.timesteps, rule=slack_shed_upper_rule)
    m.slack_line_upper_definition = pyo.Constraint(m.lines, m.timesteps, rule=slack_line_upper_rule)
    m.slack_line_lower_definition = pyo.Constraint(m.lines, m.timesteps, rule=slack_line_lower_rule)
    m.slack_power_definition = pyo.Constraint(m.storage_pairs, m.timesteps, rule=slack_power_rule)
    m.slack_soc_upper_definition = pyo.Constraint(m.storage_pairs, m.soc_timesteps, rule=slack_soc_upper_rule)

    m.sos_gen_lower = pyo.SOSConstraint(m.generators, m.timesteps, rule=sos_gen_lower_rule, sos=1)
    m.sos_gen_upper = pyo.SOSConstraint(m.generators, m.timesteps, rule=sos_gen_upper_rule, sos=1)
    m.sos_shed_lower = pyo.SOSConstraint(m.nodes, m.timesteps, rule=sos_shed_lower_rule, sos=1)
    m.sos_shed_upper = pyo.SOSConstraint(m.nodes, m.timesteps, rule=sos_shed_upper_rule, sos=1)
    m.sos_line_upper = pyo.SOSConstraint(m.lines, m.timesteps, rule=sos_line_upper_rule, sos=1)
    m.sos_line_lower = pyo.SOSConstraint(m.lines, m.timesteps, rule=sos_line_lower_rule, sos=1)
    m.sos_charge_lower = pyo.SOSConstraint(m.storage_pairs, m.timesteps, rule=sos_charge_lower_rule, sos=1)
    m.sos_discharge_lower = pyo.SOSConstraint(m.storage_pairs, m.timesteps, rule=sos_discharge_lower_rule, sos=1)
    m.sos_power = pyo.SOSConstraint(m.storage_pairs, m.timesteps, rule=sos_power_rule, sos=1)
    m.sos_soc_lower = pyo.SOSConstraint(m.storage_pairs, m.soc_timesteps, rule=sos_soc_lower_rule, sos=1)
    m.sos_soc_upper = pyo.SOSConstraint(m.storage_pairs, m.soc_timesteps, rule=sos_soc_upper_rule, sos=1)


def add_physical_bounds(m):
    for (g, t), generation in m.generation.items():
        generation.setub(m.generation_capacity[g, t])
    for (i, n, t), charge in m.charge.items():
        charge.setub(m.power_max[i, n])
    for (i, n, t), discharge in m.discharge.items():
        discharge.setub(m.power_max[i, n])


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


def charge_bid_rule(m, i, n, t):
    if i == m.active_investor and m.component("X_charge_bid") is not None:
        return m.X_charge_bid[n, t]
    return m.fixed_charge_bid[i, n, t]


def discharge_offer_rule(m, i, n, t):
    if i == m.active_investor and m.component("X_discharge_offer") is not None:
        return m.X_discharge_offer[n, t]
    return m.fixed_discharge_offer[i, n, t]


def bid_no_loop_rule(m, n, t):
    """A same-hour loop (charge c, discharge eta^2 * c, SOC unchanged) costs
    -bid * c + offer * eta^2 * c as bid. It must not be negative, or the ISO
    would cycle the storage for the bid alone (prepare_input checks fixed bids)."""
    return m.eta ** 2 * m.X_discharge_offer[n, t] >= m.X_charge_bid[n, t]


# Slack variables of the upper-bound and line pairs.
def slack_gen_upper_rule(m, g, t):
    return m.slack_gen_upper[g, t] == m.generation_capacity[g, t] - m.generation[g, t]


def slack_shed_upper_rule(m, n, t):
    return m.slack_shed_upper[n, t] == m.demand[n, t] - m.load_shed[n, t]


def slack_line_upper_rule(m, line, t):
    return m.slack_line_upper[line, t] == m.line_limit[line] - m.line_flow[line, t]


def slack_line_lower_rule(m, line, t):
    return m.slack_line_lower[line, t] == m.line_limit[line] + m.line_flow[line, t]


def slack_power_rule(m, i, n, t):
    return m.slack_power[i, n, t] == m.power_capacity[i, n] - m.charge[i, n, t] - m.discharge[i, n, t]


def slack_soc_upper_rule(m, i, n, t):
    return m.slack_soc_upper[i, n, t] == m.energy_capacity[i, n] - m.soc[i, n, t]


# Generation: 0 <= generation <= available capacity.
def sos_gen_lower_rule(m, g, t):
    return [m.generation[g, t], m.mu_gen_lower[g, t]]


def sos_gen_upper_rule(m, g, t):
    return [m.slack_gen_upper[g, t], m.mu_gen_upper[g, t]]


# Load shedding: 0 <= load_shed <= demand.
def sos_shed_lower_rule(m, n, t):
    return [m.load_shed[n, t], m.mu_shed_lower[n, t]]


def sos_shed_upper_rule(m, n, t):
    return [m.slack_shed_upper[n, t], m.mu_shed_upper[n, t]]


# Transmission: -line_limit <= line_flow <= line_limit.
def sos_line_upper_rule(m, line, t):
    return [m.slack_line_upper[line, t], m.mu_line_upper[line, t]]


def sos_line_lower_rule(m, line, t):
    return [m.slack_line_lower[line, t], m.mu_line_lower[line, t]]


# Storage power: charge >= 0, discharge >= 0, charge + discharge <= power.
def sos_charge_lower_rule(m, i, n, t):
    return [m.charge[i, n, t], m.mu_charge_lower[i, n, t]]


def sos_discharge_lower_rule(m, i, n, t):
    return [m.discharge[i, n, t], m.mu_discharge_lower[i, n, t]]


def sos_power_rule(m, i, n, t):
    return [m.slack_power[i, n, t], m.mu_power[i, n, t]]


# Storage energy: 0 <= SOC <= energy capacity, including t=0 and t=T.
def sos_soc_lower_rule(m, i, n, t):
    return [m.soc[i, n, t], m.mu_soc_lower[i, n, t]]


def sos_soc_upper_rule(m, i, n, t):
    return [m.slack_soc_upper[i, n, t], m.mu_soc_upper[i, n, t]]


# Investor profit: nodal settlement of the active investor's storage and generation.
def storage_revenue_rule(m):
    """Bilinear: nodal price times net discharge."""
    return sum(m.price[n, t] * (m.discharge[i, n, t] - m.charge[i, n, t])
               for i, n in m.storage_pairs if i == m.active_investor for t in m.timesteps)


def storage_degradation_rule(m):
    """True degradation cost; with strategic bids the market sees the bids instead."""
    return sum(0.5 * m.degradation[i] * (m.charge[i, n, t] + m.discharge[i, n, t])
               for i, n in m.storage_pairs if i == m.active_investor for t in m.timesteps)


def storage_dispatch_cost_rule(m):
    i = m.active_investor
    if pyo.value(m.dispatch_slope[i]) == 0:
        return 0
    return sum(0.5 * m.dispatch_slope[i] * (m.charge[i, n, t] ** 2 + m.discharge[i, n, t] ** 2)
               for j, n in m.storage_pairs if j == i for t in m.timesteps)


def generation_margin_rule(m):
    """Bilinear: owned share of (nodal price - marginal cost) times generation,
    less the investor's own curtailment cost share * 0.5 * owner slope * curtailment^2."""
    i = m.active_investor
    margin = sum(m.owned_share[i, g] * (m.price[pyo.value(m.generator_node[g]), t] - m.generation_cost[g])
                 * m.generation[g, t]
                 for g in m.generators if m.owned_share[i, g] > 0 for t in m.timesteps)
    curtailment_cost = sum(m.owned_share[i, g] * 0.5 * m.owner_curtailment_slope[i, g] * m.curtailment[g, t] ** 2
                           for g in m.generators if pyo.value(m.owned_share[i, g] * m.owner_curtailment_slope[i, g]) > 0
                           for t in m.timesteps)
    return margin - curtailment_cost


def investment_cost_rule(m):
    i = m.active_investor
    return sum(m.cost_power_daily[i] * m.X_power[n] + m.cost_energy_daily[i] * m.X_energy[n]
               for n in m.investment_nodes)


def profit_rule(m):
    """Direct settlement; reported to check profit_linear, not maximized."""
    return (m.storage_revenue - m.storage_degradation - m.storage_dispatch_cost
            + m.generation_margin - m.capex)


def profit_linear_rule(m):
    """The same profit without price * quantity products; equal at every KKT point.

    Strong duality: demand payment = market cost + generator/line/shedding rents
    + storage rents. Subtracting the fixed rival storage rents leaves the active
    storage rent without the bilinear X*mu terms. Owned generation earns its
    capacity scarcity rent. The difference to profit is a weighted sum of the
    slack*multiplier products of all other assets and lines.

    With balancing, market_cost includes b^2/(2 eps) = (eps/2) sum(price^2) at
    b = eps * price, and the balancing payment price * b = eps * sum(price^2)
    is not an investor's revenue; together that is -balancing_dual_term. Both
    quadratic terms enter with a minus sign, so the objective stays concave.

    With a curtailment cost, strong duality adds curtailment_dual_term, and the
    owned margin carries -share * (slope * curtailment * generation + 0.5 * owner
    slope * curtailment^2). Per plant the generation^2 coefficient is
    -slope * (1 - share) - 0.5 * share * owner slope <= 0: still concave.

    With a dispatch cost, strong duality subtracts every storage's dispatch rent
    0.5 * slope * (charge^2 + discharge^2); the rivals' rents are removed as
    their other rents are. The active storage's charge^2 coefficient is then
    -0.5 * slope (from the market cost), a rival's -slope: still concave.

    Bids: the market cost holds every storage's as-bid cost. The active
    investor's is replaced by its true degradation cost, so its bid margin
    (bid - true cost) * quantity is part of the profit, and the product
    bid * quantity of two variables never appears (Ruiz and Conejo 2009).
    With truthful bids the two are equal and this is demand payment - market cost.
    """
    i = m.active_investor
    market_cost_at_true_cost = (
        m.generation_cost_total + m.load_shed_cost_total + m.storage_degradation
        + sum(m.storage_bid_cost[j] for j in m.investors if j != i)
        + m.dispatch_cost_total + m.curtailment_cost_total + m.balancing_cost_total)
    return (m.demand_payment - market_cost_at_true_cost - m.balancing_dual_term + m.curtailment_dual_term
            - m.generation_scarcity_rent
            - m.shedding_scarcity_rent - m.transmission_scarcity_rent
            - sum(m.storage_scarcity_rent[j] + m.storage_dispatch_rent[j]
                  for j in m.investors if j != i)
            + m.owned_generation_rent[i] - m.capex)


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
    i = m.active_investor
    result = copy.deepcopy(m._profile)
    for field, variables in (("power_mw", m.X_power), ("energy_mwh", m.X_energy)):
        result[i][field] = {
            n: max(0.0, pyo.value(variables[n])) if n in variables else 0.0 for n in m.nodes
        }
    # Solver tolerances can leave energy just outside the duration bounds; snap
    # it back so the exported profile satisfies them exactly.
    power, energy = result[i]["power_mw"], result[i]["energy_mwh"]
    for n in m.nodes:
        energy[n] = min(max(energy[n], pyo.value(m.ratio_min[i]) * power[n]), pyo.value(m.ratio_max[i]) * power[n])
    if m.component("X_charge_bid") is not None:
        bids = {n: {str(t): pyo.value(m.X_charge_bid[n, t]) for t in m.timesteps} for n in m.investment_nodes}
        # Snap the offers to the no-loop bound, which the solver meets only to its tolerance.
        offers = {n: {str(t): max(pyo.value(m.X_discharge_offer[n, t]), bids[n][str(t)] / pyo.value(m.eta) ** 2)
                      for t in m.timesteps} for n in m.investment_nodes}
        result[i]["charge_bid_eur_per_mwh"] = bids
        result[i]["discharge_offer_eur_per_mwh"] = offers
    return result


if __name__ == "__main__":
    import sys
    from run import main
    raise SystemExit(main(["mpec", *sys.argv[1:]]))
