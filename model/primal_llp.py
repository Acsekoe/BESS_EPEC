"""Primal lower-level problem: fixed-demand, one-hour DC market clearing.

Read add_primal() for named variables/constraints, then the *_rule functions
for their equations. The MPEC reuses these exact physical equations.

Optional price-responsive balancing (balancing_eps = eps > 0): a free nodal
injection b[n,t] with cost b^2/(2 eps). Its stationarity gives b = eps * price,
the dual gains -(eps/2) * sum(price^2), and every nodal price becomes unique.
The market becomes a convex QP. eps = 0 omits b: the original LP.

Optional curtailment cost (profile key curtailment_slope_eur_per_mwh_per_mw):
curtailing u = capacity - generation costs 0.5 * slope * u^2, so the plant
offers its last MW at generation_cost - slope * u. Below a congestion step the
price is then negative and rises continuously as curtailment falls. Slope 0
omits the term.

Optional dispatch cost (profile key dispatch_cost_slope_eur_per_mwh_per_mw):
charging or discharging p MW for one hour costs 0.5 * slope * p^2 on top of
the linear degradation, so the marginal cost rises by slope per MW (losses
rising with current). It is strictly convex in charge and discharge, so the
storage dispatch is unique. Slope 0 omits the term.

Storage bids: each storage enters the market with an hourly charge bid (price
to buy) and discharge offer (price to sell), so the ISO's storage cost is
offer * discharge - bid * charge. Truthful bids (the default) are -0.5 * d and
0.5 * d, which is the degradation cost. The market objective is the as-bid cost;
an investor's settlement uses its true costs (solution_space.settlement_profit).

Run independently: python model/primal_llp.py --output model/output/market_run
"""
import pyomo.environ as pyo

from prepare_input import prepare_input


def build_primal(data, profile, balancing_eps=0.0):
    m = prepare_input(data, profile, balancing_eps=balancing_eps)
    m.name = "Primal market LLP"
    add_fixed_storage(m)
    add_primal(m)
    m.objective = pyo.Objective(expr=m.market_cost, sense=pyo.minimize)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def add_fixed_storage(m):
    """Capacities and bids of every storage are the profile's input parameters."""
    m.power_capacity = pyo.Expression(m.storage_pairs, rule=fixed_power_capacity_rule)
    m.energy_capacity = pyo.Expression(m.storage_pairs, rule=fixed_energy_capacity_rule)
    m.charge_bid = pyo.Expression(m.storage_pairs, m.timesteps, rule=fixed_charge_bid_rule)
    m.discharge_offer = pyo.Expression(m.storage_pairs, m.timesteps, rule=fixed_discharge_offer_rule)


def add_primal(m):
    # Dispatch variables. Nonnegative domains are the physical lower bounds.
    m.generation = pyo.Var(
        m.generators,
        m.timesteps,
        domain=pyo.NonNegativeReals,
        doc="Generation [MW]",
    )
    m.load_shed = pyo.Var(
        m.nodes,
        m.timesteps,
        domain=pyo.NonNegativeReals,
        doc="Unserved load [MW]",
    )
    m.charge = pyo.Var(
        m.storage_pairs,
        m.timesteps,
        domain=pyo.NonNegativeReals,
        doc="Storage charging [MW]",
    )
    m.discharge = pyo.Var(
        m.storage_pairs,
        m.timesteps,
        domain=pyo.NonNegativeReals,
        doc="Storage discharging [MW]",
    )
    m.soc = pyo.Var(
        m.storage_pairs,
        m.soc_timesteps,
        domain=pyo.NonNegativeReals,
        doc="Stored energy [MWh]",
    )
    m.net_injection = pyo.Var(
        m.nodes,
        m.timesteps,
        domain=pyo.Reals,
        doc="Net injection into the network [MW]",
    )

    if pyo.value(m.balancing_eps) > 0:
        m.balancing = pyo.Var(
            m.nodes,
            m.timesteps,
            domain=pyo.Reals,
            doc="Price-responsive balancing injection, eps * price [MW]",
        )

    m.line_flow = pyo.Expression(m.lines, m.timesteps, rule=line_flow_rule)
    m.curtailment = pyo.Expression(m.generators, m.timesteps, rule=curtailment_rule)

    # Physical market equations. Each rule is written explicitly below.
    m.nodal_balance = pyo.Constraint(m.nodes, m.timesteps, rule=nodal_balance_rule)
    m.system_balance = pyo.Constraint(m.timesteps, rule=system_balance_rule)
    m.generation_limit = pyo.Constraint(m.generators, m.timesteps, rule=generation_limit_rule)
    m.load_shed_limit = pyo.Constraint(m.nodes, m.timesteps, rule=load_shed_limit_rule)
    m.line_upper_limit = pyo.Constraint(m.lines, m.timesteps, rule=line_upper_limit_rule)
    m.line_lower_limit = pyo.Constraint(m.lines, m.timesteps, rule=line_lower_limit_rule)
    m.storage_power_limit = pyo.Constraint(
        m.storage_pairs,
        m.timesteps,
        rule=storage_power_limit_rule,
    )
    m.soc_transition = pyo.Constraint(m.storage_pairs, m.timesteps, rule=soc_transition_rule)
    m.soc_limit = pyo.Constraint(m.storage_pairs, m.soc_timesteps, rule=soc_limit_rule)
    m.soc_cyclic = pyo.Constraint(m.storage_pairs, rule=soc_cyclic_rule)

    m.generation_cost_total = pyo.Expression(rule=generation_cost_rule)
    m.load_shed_cost_total = pyo.Expression(rule=load_shed_cost_rule)
    m.storage_bid_cost = pyo.Expression(m.investors, rule=storage_bid_cost_rule)
    m.dispatch_cost_total = pyo.Expression(rule=dispatch_cost_rule)
    m.curtailment_cost_total = pyo.Expression(rule=curtailment_cost_rule)
    m.balancing_cost_total = pyo.Expression(rule=balancing_cost_rule)
    m.market_cost = pyo.Expression(rule=market_cost_rule)


def fixed_power_capacity_rule(m, i, n):
    return m.installed_power[i, n]


def fixed_energy_capacity_rule(m, i, n):
    return m.installed_energy[i, n]


def fixed_charge_bid_rule(m, i, n, t):
    return m.fixed_charge_bid[i, n, t]


def fixed_discharge_offer_rule(m, i, n, t):
    return m.fixed_discharge_offer[i, n, t]


def nodal_balance_rule(m, n, t):
    """Generation + discharge - charge + shedding + balancing - injection = demand."""
    balancing = m.balancing[n, t] if pyo.value(m.balancing_eps) > 0 else 0
    return (
        sum(m.generation[g, t] for g in m.generators_at_node[n])
        + sum(m.discharge[i, n, t] - m.charge[i, n, t] for i in m.storage_at_node[n])
        + m.load_shed[n, t] + balancing - m.net_injection[n, t]
        == m.demand[n, t]
    )


def system_balance_rule(m, t):
    """The sum of all nodal injections must be zero."""
    return sum(m.net_injection[n, t] for n in m.nodes) == 0


def generation_limit_rule(m, g, t):
    return m.generation[g, t] <= m.generation_capacity[g, t]


def load_shed_limit_rule(m, n, t):
    return m.load_shed[n, t] <= m.demand[n, t]


def line_flow_rule(m, line, t):
    return sum(m.ptdf[line, n] * m.net_injection[n, t] for n in m.nodes)


def curtailment_rule(m, g, t):
    """Available but unused generation [MW]."""
    return m.generation_capacity[g, t] - m.generation[g, t]


def line_upper_limit_rule(m, line, t):
    return m.line_flow[line, t] <= m.line_limit[line]


def line_lower_limit_rule(m, line, t):
    return -m.line_flow[line, t] <= m.line_limit[line]


def storage_power_limit_rule(m, i, n, t):
    """Shared inverter: charging plus discharging cannot exceed installed MW."""
    return m.charge[i, n, t] + m.discharge[i, n, t] <= m.power_capacity[i, n]


def soc_transition_rule(m, i, n, t):
    """SOC[t] - SOC[t-1] - eta*charge[t] + discharge[t]/eta = 0."""
    return (
        m.soc[i, n, t] - m.soc[i, n, t - 1]
        - m.eta * m.charge[i, n, t] + m.discharge[i, n, t] / m.eta == 0
    )


def soc_limit_rule(m, i, n, t):
    return m.soc[i, n, t] <= m.energy_capacity[i, n]


def soc_cyclic_rule(m, i, n):
    """Free initial SOC equals final SOC: a repeated-day assumption."""
    return m.soc[i, n, 0] - m.soc[i, n, m.timesteps.last()] == 0


def generation_cost_rule(m):
    return sum(m.generation_cost[g] * m.generation[g, t] for g in m.generators for t in m.timesteps)


def load_shed_cost_rule(m):
    return sum(m.voll * m.load_shed[n, t] for n in m.nodes for t in m.timesteps)


def storage_bid_cost_rule(m, investor):
    """As-bid cost of the investor's storage: offer * discharge - bid * charge.
    With truthful bids this is the degradation cost 0.5 * d * (charge + discharge)."""
    return sum(m.discharge_offer[i, n, t] * m.discharge[i, n, t] - m.charge_bid[i, n, t] * m.charge[i, n, t]
               for i, n in m.storage_pairs if i == investor for t in m.timesteps)


def dispatch_cost_rule(m):
    """0.5 * slope * (charge^2 + discharge^2): the marginal cost rises by slope per MW."""
    return sum(0.5 * m.dispatch_slope[i] * (m.charge[i, n, t] ** 2 + m.discharge[i, n, t] ** 2)
               for i, n in m.storage_pairs if pyo.value(m.dispatch_slope[i]) > 0 for t in m.timesteps)


def curtailment_cost_rule(m):
    """0.5 * slope * curtailment^2: the marginal curtailment cost is slope * curtailment."""
    return sum(0.5 * m.curtailment_slope[g] * m.curtailment[g, t] ** 2
               for g in m.generators if pyo.value(m.curtailment_slope[g]) > 0 for t in m.timesteps)


def balancing_cost_rule(m):
    """Area under the balancing supply curve price = b/eps: b^2/(2 eps)."""
    if pyo.value(m.balancing_eps) == 0:
        return 0
    return sum(m.balancing[n, t] ** 2 / (2 * m.balancing_eps) for n in m.nodes for t in m.timesteps)


def market_cost_rule(m):
    return (m.generation_cost_total + m.load_shed_cost_total + sum(m.storage_bid_cost[i] for i in m.investors)
            + m.dispatch_cost_total + m.curtailment_cost_total + m.balancing_cost_total)


def dispatch_variables(m):
    """Named quantities for CSV output only; this does not define constraints.

    Keep the previous CSV coordinate labels so existing results stay comparable.
    """
    for g, t in m.generation:
        yield f"gen|{g}|{t}", m.generation[g, t]
    for n, t in m.net_injection:
        yield f"injection|{n}|{t}", m.net_injection[n, t]
        yield f"shed|{n}|{t}", m.load_shed[n, t]
    for i, n, t in m.charge:
        yield f"charge|{i}|{n}|{t}", m.charge[i, n, t]
        yield f"discharge|{i}|{n}|{t}", m.discharge[i, n, t]
    for i, n, t in m.soc:
        yield f"soc|{i}|{n}|{t}", m.soc[i, n, t]
    if pyo.value(m.balancing_eps) > 0:
        for n, t in m.balancing:
            yield f"balancing|{n}|{t}", m.balancing[n, t]


if __name__ == "__main__":
    import sys
    from run import main
    raise SystemExit(main(["market", *sys.argv[1:]]))
