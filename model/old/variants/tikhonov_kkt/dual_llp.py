"""Clean dual of the fixed-demand primal lower-level problem."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pyomo.environ as pyo

try:
    from .common import (
        DEFAULT_DATA_PATH,
        generation_pairs,
        generator_nodes,
        load_fixed_storage_case,
        solve_ipopt,
        sparse_pairs,
    )
    from .primal_llp import configured_demand_expansion, demand_expansion_band
except ImportError:
    from common import (
        DEFAULT_DATA_PATH,
        generation_pairs,
        generator_nodes,
        load_fixed_storage_case,
        solve_ipopt,
        sparse_pairs,
    )
    from primal_llp import configured_demand_expansion, demand_expansion_band


# USER CONTROLS ---------------------------------------------------------------
DATA_PATH = DEFAULT_DATA_PATH
STORAGE_ID = "BESS_FIXED"
STORAGE_POWER_MW_BY_NODE = {"N6": 50.0, "N8": 50.0}
STORAGE_DURATION_HOURS = 4.0
DEGRADATION_EUR_PER_MWH = 15.0
SPARSE_STORAGE_TOL_MW = 1.0e-8
PRICE_BOUND_EUR_PER_MWH = 500.0
OTHER_DUAL_BOUND = 10_000.0
SOLVER_TOL = 1.0e-6
MAX_CPU_TIME_SECONDS = 120.0
TEE = True
# -----------------------------------------------------------------------------


def build_dual_llp(
    data,
    *,
    degradation_eur_per_mwh_by_unit: Mapping[str, float],
    price_bound_eur_per_mwh: float = PRICE_BOUND_EUR_PER_MWH,
    other_dual_bound: float = OTHER_DUAL_BOUND,
    sparse_storage_tol_mw: float = SPARSE_STORAGE_TOL_MW,
    demand_expansion=None,
) -> pyo.ConcreteModel:
    """Build the explicit LP/QP dual using the project sign convention.

    With ``demand_expansion`` supplied the lower level is a convex QP, and the
    expansion block contributes a Wolfe-dual correction: its bound multiplier
    prices the band and the primal quadratic is subtracted once.
    """

    if price_bound_eur_per_mwh <= 0.0 or other_dual_bound <= 0.0:
        raise ValueError("Dual bounds must be positive.")
    storage_pairs = sparse_pairs(data, sparse_storage_tol_mw)
    band = demand_expansion_band(data, demand_expansion)
    gen_pairs = generation_pairs(data)
    gen_nodes = generator_nodes(data)
    last_t = max(data.times)
    eta = data.eta

    m = pyo.ConcreteModel(name="Clean fixed-demand dual LLP")
    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.GT = pyo.Set(dimen=2, initialize=gen_pairs, ordered=True)
    m.IN = pyo.Set(dimen=2, initialize=storage_pairs, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)

    price_bound = float(price_bound_eur_per_mwh)
    dual_bound = float(other_dual_bound)
    m.lam = pyo.Var(
        m.N, m.T, bounds=(-price_bound, price_bound), initialize=50.0
    )
    m.lam_sys = pyo.Var(
        m.T, bounds=(-price_bound, price_bound), initialize=50.0
    )
    m.nu_gen = pyo.Var(m.GT, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_up = pyo.Var(m.L, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_dn = pyo.Var(m.L, m.T, bounds=(0.0, dual_bound), initialize=0.0)
    m.rho_ch = pyo.Var(m.IN, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.sig_dis = pyo.Var(m.IN, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.gam = pyo.Var(
        m.IN, m.T, bounds=(-dual_bound, dual_bound), initialize=0.0
    )
    m.del_soc = pyo.Var(
        m.IN, m.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0
    )
    m.rho_per = pyo.Var(
        m.IN, bounds=(-dual_bound, dual_bound), initialize=0.0
    )
    if band:
        m.EB = pyo.Set(dimen=2, initialize=sorted(band), ordered=True)
        m.E_extra = pyo.Var(m.EB, domain=pyo.NonNegativeReals, initialize=0.0)
        m.xi_extra = pyo.Var(m.EB, bounds=(-dual_bound, 0.0), initialize=0.0)
        m.demand_expansion_stationarity = pyo.Constraint(
            m.EB,
            rule=lambda model, n, t: -model.lam[n, t] + model.xi_extra[n, t]
            <= -demand_expansion.reference_price_eur_per_mwh
            + demand_expansion.slope(data.demand_el[n, t]) * model.E_extra[n, t],
        )
        expansion_dual_expr = sum(
            band[n, t] * m.xi_extra[n, t]
            - 0.5
            * demand_expansion.slope(data.demand_el[n, t])
            * m.E_extra[n, t] ** 2
            for n, t in m.EB
        )
    else:
        expansion_dual_expr = 0.0

    m.gen_stationarity = pyo.Constraint(
        m.GT,
        rule=lambda model, g, t: sum(
            model.lam[n, t] for n in gen_nodes.get(g, [])
        )
        + model.nu_gen[g, t]
        <= data.generation_cost[g],
    )
    m.charge_stationarity = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: -model.lam[n, t]
        + model.rho_ch[i, n, t]
        - eta * model.gam[i, n, t]
        <= 0.5 * degradation_eur_per_mwh_by_unit[i],
    )
    m.discharge_stationarity = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: model.lam[n, t]
        + model.sig_dis[i, n, t]
        + model.gam[i, n, t] / eta
        <= 0.5 * degradation_eur_per_mwh_by_unit[i],
    )
    m.netinjection_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, n, t: -model.lam[n, t]
        + model.lam_sys[t]
        + sum(
            data.ptdf[l, n] * (model.mu_up[l, t] + model.mu_dn[l, t])
            for l in model.L
        )
        == 0.0,
    )

    def soc_stationarity(model, i, n, tau):
        expr = model.del_soc[i, n, tau]
        if tau in model.T:
            expr += model.gam[i, n, tau]
        if (tau + 1) in model.T:
            expr -= model.gam[i, n, tau + 1]
        if tau == 0:
            expr += model.rho_per[i, n]
        if tau == last_t:
            expr -= model.rho_per[i, n]
        return expr <= 0.0

    m.soc_stationarity = pyo.Constraint(
        m.IN, m.T_SOC, rule=soc_stationarity
    )

    m.unregularized_dual_objective_expr = pyo.Expression(
        expr=sum(
            data.demand_el[n, t] * m.lam[n, t]
            for n in m.N
            for t in m.T
        )
        + sum(
            data.generation_capacity[g, t] * m.nu_gen[g, t]
            for g, t in m.GT
        )
        + sum(
            data.line_limit[l] * (m.mu_up[l, t] - m.mu_dn[l, t])
            for l in m.L
            for t in m.T
        )
        + sum(
            data.x_power[i, n]
            * (m.rho_ch[i, n, t] + m.sig_dis[i, n, t])
            for i, n in m.IN
            for t in m.T
        )
        + sum(
            data.x_energy[i, n] * m.del_soc[i, n, tau]
            for i, n in m.IN
            for tau in m.T_SOC
        )
        + expansion_dual_expr
    )
    m.objective = pyo.Objective(
        expr=m.unregularized_dual_objective_expr,
        sense=pyo.maximize,
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
    model = build_dual_llp(
        data,
        degradation_eur_per_mwh_by_unit={STORAGE_ID: DEGRADATION_EUR_PER_MWH},
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
    prices = [pyo.value(model.lam[n, t]) for n in model.N for t in model.T]
    print(
        f"dual objective={pyo.value(model.unregularized_dual_objective_expr):,.6f} EUR/day"
    )
    print(f"lambda range=[{min(prices):.6f}, {max(prices):.6f}] EUR/MWh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

