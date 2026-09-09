"""Shared primal aFRR extension for the ISO QP and the capacity MPEC.

Offers/penalties are EUR/MW/h. Unscaled MW procurement constraints have
duals in EUR/MW/block. SOC buffers are a deterministic deliverability proxy;
expected activation enters wear only, not physical energy or settlement.
"""
import pyomo.environ as pyo

from market_data import validate_afrr


def add_primal(m, data, pairs, power, energy, degradation):
    validate_afrr(data)
    m.B = pyo.Set(initialize=sorted(data.afrr_block_hours), ordered=True)
    m.AFRR_IN = pyo.Set(dimen=2, initialize=list(pairs), ordered=True)
    # Sparse MPEC generation: a generator with zero availability in even one
    # block hour cannot promise either direction throughout that block.
    gb = [(g, b) for g in data.afrr_eligible_generators for b in m.B
          if all(data.generation_capacity[g, t] > 1e-8 for t in data.afrr_block_hours[b])]
    m.AFRR_GB = pyo.Set(dimen=2, initialize=gb, ordered=True)
    m.AFRR_GT = pyo.Set(dimen=2, initialize=[
        (g, t) for g, b in gb for t in data.afrr_block_hours[b]], ordered=True)
    # Every block includes its start and end SOC; shared endpoints belong to
    # BOTH blocks. Internal hourly endpoints are not duplicated within a block.
    m.AFRR_BK = pyo.Set(dimen=2, initialize=[
        (b, k) for b in m.B for k in range(min(data.afrr_block_hours[b])-1,
                                         max(data.afrr_block_hours[b])+1)], ordered=True)
    m.R_up = pyo.Var(m.AFRR_IN, m.B, domain=pyo.NonNegativeReals, initialize=0)
    m.R_dn = pyo.Var(m.AFRR_IN, m.B, domain=pyo.NonNegativeReals, initialize=0)
    m.r_up = pyo.Var(m.AFRR_GB, domain=pyo.NonNegativeReals, initialize=0)
    m.r_dn = pyo.Var(m.AFRR_GB, domain=pyo.NonNegativeReals, initialize=0)
    m.afrr_short_up = pyo.Var(m.B, domain=pyo.NonNegativeReals, initialize=0)
    m.afrr_short_dn = pyo.Var(m.B, domain=pyo.NonNegativeReals, initialize=0)
    for i, n in m.AFRR_IN:
        for t in m.T:
            b = data.afrr_block(t)
            m.discharge_power_bound[i, n, t].set_value(
                m.P_discharge[i, n, t] + m.R_up[i, n, b] <= power(m, i, n))
            m.charge_power_bound[i, n, t].set_value(
                m.P_charge[i, n, t] + m.R_dn[i, n, b] <= power(m, i, n))
    for g, t in m.AFRR_GT:
        m.generation_capacity_bound[g, t].set_value(
            m.P_gen[g, t] + m.r_up[g, data.afrr_block(t)] <= data.generation_capacity[g, t])
    m.afrr_gen_down_headroom = pyo.Constraint(m.AFRR_GT, rule=lambda mm, g, t:
        mm.P_gen[g, t] - mm.r_dn[g, data.afrr_block(t)] >= 0)
    m.afrr_soc_up = pyo.Constraint(m.AFRR_IN, m.AFRR_BK, rule=lambda mm, i, n, b, k:
        mm.SOC[i, n, k] - data.afrr_response_hours/data.eta * mm.R_up[i, n, b] >= 0)
    m.afrr_soc_down = pyo.Constraint(m.AFRR_IN, m.AFRR_BK, rule=lambda mm, i, n, b, k:
        mm.SOC[i, n, k] + data.afrr_response_hours*data.eta * mm.R_dn[i, n, b]
        <= energy(mm, i, n))
    for direction, demand in [('up', data.afrr_demand_up_mw), ('dn', data.afrr_demand_down_mw)]:
        reserves = getattr(m, 'R_'+direction)
        generation = getattr(m, 'r_'+direction)
        shortage = getattr(m, 'afrr_short_'+direction)
        m.add_component('afrr_balance_'+direction, pyo.Constraint(m.B, rule=lambda mm, b,
            reserves=reserves, generation=generation, shortage=shortage, demand=demand:
            sum(reserves[i, n, b] for i, n in mm.AFRR_IN)
            + sum(generation[g, bb] for g, bb in mm.AFRR_GB if bb == b)
            + shortage[b] >= demand))
    m.afrr_expected_degradation = pyo.Expression(m.AFRR_IN, rule=lambda mm, i, n:
        0.5 * degradation[i] * sum(data.afrr_duration(b) * (
            data.afrr_activation_up * mm.R_up[i, n, b]
            + data.afrr_activation_down * mm.R_dn[i, n, b]) for b in mm.B))
    m.afrr_procurement_cost = pyo.Expression(expr=sum(
        data.afrr_duration(b) * data.afrr_generator_offer_eur_per_mw_hour[g]
        * (m.r_up[g, b]+m.r_dn[g, b]) for g, b in m.AFRR_GB)
        + sum(data.afrr_duration(b) * data.afrr_penalty_eur_per_mw_hour
              * (m.afrr_short_up[b]+m.afrr_short_dn[b]) for b in m.B))
    m.afrr_total_cost = pyo.Expression(expr=m.afrr_procurement_cost
        + sum(m.afrr_expected_degradation[i, n] for i, n in m.AFRR_IN))


def price(m, data, direction, block, *, hourly=False):
    """Imported reserve dual: EUR/MW/block by default, EUR/MW/h if hourly."""
    value = float(m.dual[getattr(m, 'afrr_balance_'+direction)[block]])
    return value/data.afrr_duration(block) if hourly else value


def reserve_revenue(m, data, unit):
    return sum(price(m, data, direction, b) * pyo.value(getattr(m, 'R_'+direction)[unit, n, b])
               for direction in ('up', 'dn') for n in data.nodes for b in m.B)


def owned_reserve_surplus(m, data, shares):
    # Offers are maintained as true availability costs in this benchmark.
    return sum(share * (price(m, data, direction, b)
               - data.afrr_duration(b)*data.afrr_generator_offer_eur_per_mw_hour[g])
               * pyo.value(getattr(m, 'r_'+direction)[g, b])
               for g, share in shares.items() for b in m.B
               if (g, b) in m.AFRR_GB for direction in ('up', 'dn'))
