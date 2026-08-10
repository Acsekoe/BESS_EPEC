"""Clean fixed-demand primal lower-level market-clearing problem."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pyomo.environ as pyo

try:
    from .common import (
        DEFAULT_DATA_PATH,
        generation_pairs,
        load_fixed_storage_case,
        solve_ipopt,
        sparse_pairs,
    )
except ImportError:  # Direct execution: python model/tikhonov_kkt/primal_llp.py
    from common import (
        DEFAULT_DATA_PATH,
        generation_pairs,
        load_fixed_storage_case,
        solve_ipopt,
        sparse_pairs,
    )


# USER CONTROLS ---------------------------------------------------------------
DATA_PATH = DEFAULT_DATA_PATH
STORAGE_ID = "BESS_FIXED"
STORAGE_POWER_MW_BY_NODE = {"N6": 50.0, "N8": 50.0}
STORAGE_DURATION_HOURS = 4.0
DEGRADATION_EUR_PER_MWH = 15.0
SPARSE_STORAGE_TOL_MW = 1.0e-8
# Price-responsive demand. Set DEMAND_ELASTICITY to 0.0 for fixed demand.
DEMAND_REFERENCE_PRICE_EUR_PER_MWH = 60.0
DEMAND_ELASTICITY = 0.20
SOLVER_TOL = 1.0e-6
MAX_CPU_TIME_SECONDS = 120.0
TEE = True
# -----------------------------------------------------------------------------


def configured_demand_expansion():
    """Build the demand-expansion curve from this module's user controls."""

    if DEMAND_ELASTICITY <= 0.0:
        return None
    from single_investor_mpec import DemandExpansionCurve

    return DemandExpansionCurve(
        reference_price_eur_per_mwh=DEMAND_REFERENCE_PRICE_EUR_PER_MWH,
        elasticity=DEMAND_ELASTICITY,
    )


def demand_expansion_band(data, demand_expansion) -> dict[tuple[str, int], float]:
    """Node-hour widths of the price-responsive demand-expansion block."""

    if demand_expansion is None:
        return {}
    return {
        (n, t): demand_expansion.band_mw(data.demand_el[n, t])
        for n in data.nodes
        for t in data.times
        if demand_expansion.band_mw(data.demand_el[n, t]) > 1.0e-8
    }


def build_primal_llp(
    data,
    *,
    degradation_eur_per_mwh_by_unit: Mapping[str, float],
    sparse_storage_tol_mw: float = SPARSE_STORAGE_TOL_MW,
    demand_expansion=None,
) -> pyo.ConcreteModel:
    """Build the physical DC-OPF with fixed BESS capacities.

    With ``demand_expansion`` supplied, each load node also carries the
    low-price half of its demand curve, so the nodal price can settle at an
    intermediate value instead of jumping between free renewables and the
    storage indifference price.
    """

    storage_pairs = sparse_pairs(data, sparse_storage_tol_mw)
    gen_pairs = generation_pairs(data)
    generators_at_node_time = {
        (node, int(time)): [
            generator
            for generator in data.generators_at_node.get(node, [])
            if (generator, int(time)) in gen_pairs
        ]
        for node in data.nodes
        for time in data.times
    }
    storage_at_node = {
        node: [unit for unit, pair_node in storage_pairs if pair_node == node]
        for node in data.nodes
    }
    last_t = max(data.times)
    eta = data.eta

    m = pyo.ConcreteModel(name="Clean fixed-demand primal LLP")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.GT = pyo.Set(dimen=2, initialize=gen_pairs, ordered=True)
    m.IN = pyo.Set(dimen=2, initialize=storage_pairs, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)

    m.P_gen = pyo.Var(m.GT, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_charge = pyo.Var(m.IN, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_discharge = pyo.Var(m.IN, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.SOC = pyo.Var(m.IN, m.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)

    band = demand_expansion_band(data, demand_expansion)
    if band:
        m.EB = pyo.Set(dimen=2, initialize=sorted(band), ordered=True)
        m.E_extra = pyo.Var(m.EB, domain=pyo.NonNegativeReals, initialize=0.0)
        m.demand_expansion_bound = pyo.Constraint(
            m.EB, rule=lambda mm, n, t: mm.E_extra[n, t] <= band[n, t]
        )
        m.demand_expansion_utility_expr = pyo.Expression(
            expr=sum(
                demand_expansion.reference_price_eur_per_mwh * m.E_extra[n, t]
                - 0.5
                * demand_expansion.slope(data.demand_el[n, t])
                * m.E_extra[n, t] ** 2
                for n, t in m.EB
            )
        )
    else:
        m.demand_expansion_utility_expr = pyo.Expression(expr=0.0)

    m.generation_cost_expr = pyo.Expression(
        expr=sum(data.generation_cost[g] * m.P_gen[g, t] for g, t in m.GT)
    )
    m.degradation_cost_expr = pyo.Expression(
        expr=sum(
            0.5
            * degradation_eur_per_mwh_by_unit[i]
            * (m.P_charge[i, n, t] + m.P_discharge[i, n, t])
            for i, n in m.IN
            for t in m.T
        )
    )
    m.objective = pyo.Objective(
        expr=m.generation_cost_expr
        + m.degradation_cost_expr
        - m.demand_expansion_utility_expr,
        sense=pyo.minimize,
    )

    def nodal_balance(model, node, time):
        storage_net = sum(
            model.P_discharge[unit, node, time]
            - model.P_charge[unit, node, time]
            for unit in storage_at_node[node]
        )
        return (
            sum(
                model.P_gen[generator, time]
                for generator in generators_at_node_time[node, int(time)]
            )
            + storage_net
            - data.demand_el[node, time]
            - (model.E_extra[node, time] if (node, time) in band else 0.0)
            == model.NetInjection[node, time]
        )

    m.nodal_balance = pyo.Constraint(m.N, m.T, rule=nodal_balance)
    m.system_balance = pyo.Constraint(
        m.T,
        rule=lambda model, t: sum(model.NetInjection[n, t] for n in model.N)
        == 0.0,
    )
    m.generation_capacity_bound = pyo.Constraint(
        m.GT,
        rule=lambda model, g, t: model.P_gen[g, t]
        <= data.generation_capacity[g, t],
    )

    def flow(model, line, time):
        return sum(
            data.ptdf[line, node] * model.NetInjection[node, time]
            for node in model.N
        )

    m.line_upper_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, l, t: flow(model, l, t) <= data.line_limit[l],
    )
    m.line_lower_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, l, t: flow(model, l, t) >= -data.line_limit[l],
    )
    m.charge_power_bound = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: model.P_charge[i, n, t]
        <= data.x_power[i, n],
    )
    m.discharge_power_bound = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: model.P_discharge[i, n, t]
        <= data.x_power[i, n],
    )
    m.soc_transition = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: model.SOC[i, n, t]
        == model.SOC[i, n, t - 1]
        + eta * model.P_charge[i, n, t]
        - model.P_discharge[i, n, t] / eta,
    )
    m.soc_capacity_bound = pyo.Constraint(
        m.IN,
        m.T_SOC,
        rule=lambda model, i, n, tau: model.SOC[i, n, tau]
        <= data.x_energy[i, n],
    )
    m.soc_periodicity = pyo.Constraint(
        m.IN,
        rule=lambda model, i, n: model.SOC[i, n, 0]
        == model.SOC[i, n, last_t],
    )
    m._market_data = data
    m._storage_pairs = tuple(storage_pairs)
    return m


def main() -> int:
    data = load_fixed_storage_case(
        Path(DATA_PATH),
        storage_id=STORAGE_ID,
        power_mw_by_node=STORAGE_POWER_MW_BY_NODE,
        duration_hours=STORAGE_DURATION_HOURS,
    )
    model = build_primal_llp(
        data,
        degradation_eur_per_mwh_by_unit={
            STORAGE_ID: DEGRADATION_EUR_PER_MWH
        },
        demand_expansion=configured_demand_expansion(),
    )
    results = solve_ipopt(
        model,
        solver_tol=SOLVER_TOL,
        max_cpu_time=MAX_CPU_TIME_SECONDS,
        tee=TEE,
    )
    termination = str(results.solver.termination_condition)
    print(f"termination={termination}")
    if termination != "optimal":
        return 1
    raw_duals = [
        float(model.dual[model.nodal_balance[n, t]])
        for n in model.N
        for t in model.T
        if model.nodal_balance[n, t] in model.dual
    ]
    print(f"primal objective={pyo.value(model.objective):,.6f} EUR/day")
    if raw_duals:
        print(
            "raw nodal-balance dual range="
            f"[{min(raw_duals):.6f}, {max(raw_duals):.6f}]"
        )
    if hasattr(model, "EB"):
        reference = sum(
            data.demand_el[n, t] for n in model.N for t in model.T
        )
        extra = sum(pyo.value(model.E_extra[key]) for key in model.EB)
        print(
            f"reference demand={reference:,.1f} MWh/day; "
            f"price-responsive expansion={extra:,.1f} MWh/day "
            f"({extra / reference:.2%}); demand curtailed below reference=0.0 MWh"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

