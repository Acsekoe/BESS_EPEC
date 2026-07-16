"""Single-investor MPEC with optional quadratic demand and strong duality.

This proof model represents one strategic BESS investor in the deterministic
spot market. The lower-level market clearing is embedded through primal
feasibility, dual feasibility, and a strong-duality equality. By default demand
is fixed: no load shedding, no VOLL scarcity valve, and no demand-response curve.
The quadratic demand curve can be re-enabled for feasibility experiments.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pyomo.environ as pyo

from primal_market_clearing_model import (
    DEFAULT_DATA_PATH,
    MarketData,
    build_primal_market_clearing_model,
    load_market_data,
    value,
)
from single_investor_mpec_results import export_solution, print_lambda_and_line_duals, print_solution_summary
from solver_utils import get_ipopt_solver


MODEL_NAME = "Single Investor Primal-Dual MPEC"
INVESTOR_ID = "I1"
EXISTING_ID = "E0"
# Experiment inputs (alternative capacities, stress cases, ...) live in a
# separate JSON so the baseline market_data.json stays the untouched benchmark.
EXPERIMENT_DATA_PATH = DEFAULT_DATA_PATH.with_name("market_data_euro.json")
DEFAULT_WACC = 0.08
DEFAULT_LIFETIME_YEARS = 15
DEFAULT_NODE_LIMIT_MW = 100.0
DEFAULT_RATIO_MIN = 2.0
DEFAULT_RATIO_MAX = 8.0
DEFAULT_BESS_COST_POWER_EUR_PER_MW = 6_600.0
DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH = 18_800.0
DEFAULT_DEGRADATION_EUR_PER_MWH = 15.0
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "single_investor_mpec"
USE_DEMAND_CURVE = True
DEFAULT_INITIAL_POWER_MW = DEFAULT_NODE_LIMIT_MW if not USE_DEMAND_CURVE else 10.0
DEFAULT_INITIAL_RATIO_HOURS = DEFAULT_RATIO_MAX if not USE_DEMAND_CURVE else DEFAULT_RATIO_MIN
DEFAULT_DEMAND_CURVE_ALPHA = 4000
DEFAULT_DEMAND_CURVE_BETA = 10000
DEFAULT_FIXED_DEMAND_DUAL_BOUND_EUR_PER_MWH = 30_000.0
# Below this shed level a node-hour counts as "no curtailment": the demand
# curve does not pin the price there and the QP solver dual is used instead.
SHED_INTERIOR_TOL_MW = 1e-4


def capital_recovery_factor(wacc: float, lifetime_years: int = DEFAULT_LIFETIME_YEARS) -> float:
    return wacc * (1.0 + wacc) ** lifetime_years / ((1.0 + wacc) ** lifetime_years - 1.0)


@dataclass(frozen=True)
class InvestorConfig:
    """Economic parameters of one strategic BESS investor."""

    investor_id: str = INVESTOR_ID
    wacc: float = DEFAULT_WACC
    lifetime_years: int = DEFAULT_LIFETIME_YEARS
    cost_power_eur_per_mw: float = DEFAULT_BESS_COST_POWER_EUR_PER_MW
    cost_energy_eur_per_mwh: float = DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH
    degradation_eur_per_mwh: float = DEFAULT_DEGRADATION_EUR_PER_MWH
    ratio_min: float = DEFAULT_RATIO_MIN
    ratio_max: float = DEFAULT_RATIO_MAX


@dataclass(frozen=True)
class QuadraticDemandCurve:
    """Strictly convex curtailment cost used by the embedded lower level.

    Shedding ``s`` MW at nodal demand ``D`` costs ``alpha*s + beta*s^2/(2D)``,
    so the marginal willingness-to-pay is ``alpha + beta*(s/D)``: a strictly
    decreasing inverse demand curve in the curtailed share. Wherever the
    cleared curtailment is interior the nodal price equals the curve value,
    which makes lower-level prices unique (no optimal dual face to pick from).
    """

    alpha: float  # marginal WTP at zero curtailment, EUR/MWh
    beta: float  # slope in EUR/MWh per unit curtailed share (> 0)

    def quad_coefficient(self, demand_mw: float) -> float:
        return self.beta / demand_mw if demand_mw > 0.0 else 0.0

    def marginal_wtp(self, shed_mw: float, demand_mw: float) -> float:
        return self.alpha + self.quad_coefficient(demand_mw) * shed_mw


def default_quadratic_demand_curve() -> QuadraticDemandCurve:
    """Default smooth scarcity curve, independent of stepwise demand response."""

    return QuadraticDemandCurve(alpha=DEFAULT_DEMAND_CURVE_ALPHA, beta=DEFAULT_DEMAND_CURVE_BETA)


def build_quadratic_primal_model(data: MarketData, quad: QuadraticDemandCurve) -> pyo.ConcreteModel:
    """Lower-level clearing QP: the primal LP structure plus the quadratic shed cost.

    Reuses the standalone primal builder, where ``P_shed[n,t]`` is bounded by
    full nodal demand, then swaps in the strictly convex objective.
    """

    m = build_primal_market_clearing_model(data)
    md: MarketData = m._market_data
    generation_cost = sum(md.generation_cost[g] * m.P_gen[g, t] for g in m.G for t in m.T)
    shed_linear_cost = sum(quad.alpha * m.P_shed[n, t] for n in m.N for t in m.T)
    quad_cost = sum(
        0.5 * quad.quad_coefficient(md.demand_el[n, t]) * m.P_shed[n, t] ** 2
        for n in m.N
        for t in m.T
    )
    m.objective.deactivate()
    m.quad_objective = pyo.Objective(expr=generation_cost + shed_linear_cost + quad_cost, sense=pyo.minimize)
    m._quad_demand = quad
    return m


def build_fixed_demand_primal_model(data: MarketData) -> pyo.ConcreteModel:
    """Standalone lower-level clearing LP with fixed demand and no load shedding."""

    m = build_primal_market_clearing_model(data)
    for n in m.N:
        for t in m.T:
            m.P_shed[n, t].fix(0.0)
    m._use_demand_curve = False
    return m


def quadratic_reference_lambda(
    reference: pyo.ConcreteModel, quad: QuadraticDemandCurve
) -> dict[tuple[str, int], float]:
    """Unique nodal prices for the solved reference QP.

    Where the cleared curtailment is interior the price equals the demand
    curve at that curtailment. With ``alpha > 0`` most node-hours clear with
    zero shedding and no marginal consumer, so there the price comes from the
    solver duals of the nodal balance, sign-oriented against the curve at the
    interior node-hours.
    """

    md: MarketData = reference._market_data
    curve: dict[tuple[str, int], float] = {}
    for n in reference.N:
        for t in reference.T:
            demand = md.demand_el[n, t]
            shed = value(reference.P_shed[n, t])
            if demand > 0.0 and SHED_INTERIOR_TOL_MW < shed < demand - SHED_INTERIOR_TOL_MW:
                curve[(n, t)] = quad.marginal_wtp(shed, demand)
    duals = {
        (n, t): float(reference.dual[reference.nodal_balance[n, t]])
        for n in reference.N
        for t in reference.T
    }
    if curve:
        diff_pos = sum(abs(duals[key] - curve[key]) for key in curve)
        diff_neg = sum(abs(-duals[key] - curve[key]) for key in curve)
        sign = 1.0 if diff_pos <= diff_neg else -1.0
    else:
        sign = 1.0 if sum(duals.values()) >= 0.0 else -1.0
    return {key: curve.get(key, sign * duals[key]) for key in duals}


def reference_system_price(
    reference: pyo.ConcreteModel, nodal_lambda: dict[tuple[str, int], float]
) -> dict[int, float]:
    """Uniform per-hour system price = dual of the system-balance constraint.

    Nodal price = system price + congestion rent; this returns just the system
    component, i.e. the single price a zonal / one-bidding-zone settlement pays
    at every node. The sign is aligned to ``nodal_lambda`` so it uses the same
    orientation the nodal-price recovery already picked for the solver duals.
    """

    raw_sys = {t: float(reference.dual[reference.system_balance[t]]) for t in reference.T}
    raw_nodal = {
        (n, t): float(reference.dual[reference.nodal_balance[n, t]])
        for n in reference.N
        for t in reference.T
    }
    aligned = sum(nodal_lambda[key] * raw_nodal[key] for key in raw_nodal)
    sign = -1.0 if aligned < 0.0 else 1.0
    return {t: sign * raw_sys[t] for t in reference.T}


def fixed_demand_reference_lambda(reference: pyo.ConcreteModel) -> dict[tuple[str, int], float]:
    """Nodal prices from the fixed-demand reference LP solver duals."""

    duals = {
        (n, t): float(reference.dual[reference.nodal_balance[n, t]])
        for n in reference.N
        for t in reference.T
    }
    sign = 1.0 if sum(duals.values()) >= 0.0 else -1.0
    return {key: sign * dual for key, dual in duals.items()}


def _solver_dual_cross_check(
    reference: pyo.ConcreteModel, reference_lambda: dict[tuple[str, int], float]
) -> float | None:
    """Max |solver nodal dual - curve price|, tolerant of the solver dual-sign convention."""

    duals: dict[tuple[str, int], float] = {}
    for n in reference.N:
        for t in reference.T:
            dual = reference.dual.get(reference.nodal_balance[n, t], None)
            if dual is None:
                return None
            duals[(n, t)] = float(dual)
    diff_pos = max(abs(duals[key] - reference_lambda[key]) for key in duals)
    diff_neg = max(abs(-duals[key] - reference_lambda[key]) for key in duals)
    return min(diff_pos, diff_neg)


def single_storage_data(
    data: MarketData,
    power_mw: float,
    ratio_hours: float,
    existing_power_mw: float = 0.0,
    existing_ratio_hours: float = 2.0,
    *,
    investor_id: str = INVESTOR_ID,
    rival_id: str = EXISTING_ID,
    rival_power_mw: Mapping[str, float] | None = None,
    rival_energy_mwh: Mapping[str, float] | None = None,
    power_by_node: Mapping[str, float] | None = None,
    energy_by_node: Mapping[str, float] | None = None,
) -> MarketData:
    """Return data with the active investor (and optional rival fleet) as storage units."""

    units = [investor_id]
    if power_by_node is None:
        x_power = {(investor_id, node): float(power_mw) for node in data.nodes}
        x_energy = {(investor_id, node): float(power_mw) * ratio_hours for node in data.nodes}
    else:
        x_power = {(investor_id, node): max(0.0, float(power_by_node[node])) for node in data.nodes}
        if energy_by_node is not None:
            x_energy = {(investor_id, node): max(0.0, float(energy_by_node[node])) for node in data.nodes}
        else:
            x_energy = {(investor_id, node): x_power[(investor_id, node)] * ratio_hours for node in data.nodes}
    if rival_power_mw is None:
        rival_power_mw = {node: existing_power_mw for node in data.nodes}
        rival_energy_mwh = {node: existing_power_mw * existing_ratio_hours for node in data.nodes}
    if any(v > 1e-9 for v in rival_power_mw.values()):
        units.append(rival_id)
        for node in data.nodes:
            x_power[(rival_id, node)] = float(rival_power_mw[node])
            x_energy[(rival_id, node)] = float(rival_energy_mwh[node])
    return replace(data, storage_units=units, x_power=x_power, x_energy=x_energy)


def fixed_storage_data_from_solution(model: pyo.ConcreteModel) -> MarketData:
    """Return lower-level data with storage capacities fixed at the MPEC solution."""

    data: MarketData = model._market_data
    investor_id = model._investor_id
    units = [investor_id]
    x_power = {(investor_id, node): max(0.0, value(model.X_power[node])) for node in data.nodes}
    x_energy = {(investor_id, node): max(0.0, value(model.X_energy[node])) for node in data.nodes}
    if any(v > 1e-9 for v in model._rival_power_mw.values()):
        units.append(model._rival_id)
        for node in data.nodes:
            x_power[(model._rival_id, node)] = model._rival_power_mw[node]
            x_energy[(model._rival_id, node)] = model._rival_energy_mwh[node]
    return replace(data, storage_units=units, x_power=x_power, x_energy=x_energy)


def _nodes_of_generator(data: MarketData) -> dict[str, list[str]]:
    gen_nodes: dict[str, list[str]] = {generator: [] for generator in data.generators}
    for node in data.nodes:
        for generator in data.generators_at_node.get(node, []):
            gen_nodes.setdefault(generator, []).append(node)
    return gen_nodes


def build_single_investor_mpec(
    data: MarketData,
    *,
    wacc: float = DEFAULT_WACC,
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW,
    ratio_min: float = DEFAULT_RATIO_MIN,
    ratio_max: float = DEFAULT_RATIO_MAX,
    initial_power_mw: float = DEFAULT_INITIAL_POWER_MW,
    initial_ratio_hours: float = DEFAULT_INITIAL_RATIO_HOURS,
    fixed_power_mw: float | None = None,
    dual_bound_scale: float = 10.0,
    existing_power_mw: float = 0.0,
    existing_ratio_hours: float = 2.0,
    quad_demand: QuadraticDemandCurve,
    use_demand_curve: bool = USE_DEMAND_CURVE,
    investor: InvestorConfig | None = None,
    rival_id: str = EXISTING_ID,
    rival_power_mw: Mapping[str, float] | None = None,
    rival_energy_mwh: Mapping[str, float] | None = None,
    system_price_settlement: bool = False,
) -> pyo.ConcreteModel:
    """Build the one-investor MPEC.

    The rival fleet (``rival_power_mw``/``rival_energy_mwh`` per node, or the
    legacy uniform ``existing_power_mw``) is an exogenous, non-strategic BESS
    unit inside the lower-level market clearing. It consumes part of the
    shared nodal connection limit, so the investor can only add up to
    ``node_limit_mw - rival_power_mw[n]`` at each node.

    If ``use_demand_curve`` is false, demand is fixed and ``P_shed`` is fixed
    to zero for reporting only. If true, the lower level uses one quadratic
    curtailment variable per node-hour, marginal WTP
    ``alpha + beta*(shed/demand)``, and a Wolfe-dual strong-duality equality.
    """

    if dual_bound_scale <= 0.0:
        raise ValueError("dual_bound_scale must be positive.")
    if investor is not None and (wacc, ratio_min, ratio_max) != (DEFAULT_WACC, DEFAULT_RATIO_MIN, DEFAULT_RATIO_MAX):
        raise ValueError("Pass economic parameters through `investor`, not the legacy scalar kwargs.")
    inv = investor or InvestorConfig(wacc=wacc, ratio_min=ratio_min, ratio_max=ratio_max)
    if existing_power_mw < 0.0:
        raise ValueError("existing_power_mw must be non-negative.")
    if rival_power_mw is not None and existing_power_mw > 0.0:
        raise ValueError("Pass either rival_power_mw per node or the legacy existing_power_mw scalar, not both.")
    if rival_power_mw is None:
        rival_power_mw = {node: existing_power_mw for node in data.nodes}
        rival_energy_mwh = {node: existing_power_mw * existing_ratio_hours for node in data.nodes}
    elif rival_energy_mwh is None:
        raise ValueError("rival_energy_mwh is required when rival_power_mw is given.")
    for node in data.nodes:
        if rival_power_mw[node] < 0.0 or rival_energy_mwh[node] < 0.0:
            raise ValueError(f"Negative rival capacity at node {node}.")
        if rival_power_mw[node] > node_limit_mw:
            raise ValueError(f"Rival power at node {node} exceeds the nodal connection limit.")
    invest_limit = {node: node_limit_mw - rival_power_mw[node] for node in data.nodes}
    rival_active = any(v > 1e-9 for v in rival_power_mw.values())
    storage_units = [inv.investor_id] + ([rival_id] if rival_active else [])

    gen_nodes = _nodes_of_generator(data)
    last_t = max(data.times)
    eta = data.eta
    crf_daily = capital_recovery_factor(inv.wacc, inv.lifetime_years) / 365.25
    # Congestion duals can be much larger than generator marginal costs under
    # the PTDF formulation. In fixed-demand mode this bound is purely numerical;
    # it is not a VOLL or load-shed price.
    dual_bound_base = data.voll if use_demand_curve else DEFAULT_FIXED_DEMAND_DUAL_BOUND_EUR_PER_MWH
    dual_bound = dual_bound_scale * dual_bound_base

    m = pyo.ConcreteModel(name=MODEL_NAME)

    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.G = pyo.Set(initialize=data.generators, ordered=True)
    m.I = pyo.Set(initialize=storage_units, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)

    # Upper-level investment variables. Rival capacity consumes part of the
    # shared nodal connection limit, shrinking the investor's headroom per node.
    init_ratio = max(initial_ratio_hours, inv.ratio_min)
    m.X_power = pyo.Var(
        m.N,
        bounds=lambda model, n: (0.0, invest_limit[n]),
        initialize=lambda model, n: min(max(initial_power_mw, 0.0), invest_limit[n]),
    )
    m.X_energy = pyo.Var(
        m.N,
        bounds=lambda model, n: (0.0, inv.ratio_max * invest_limit[n]),
        initialize=lambda model, n: init_ratio * min(max(initial_power_mw, 0.0), invest_limit[n]),
    )
    m.energy_ratio_min = pyo.Constraint(m.N, rule=lambda model, n: model.X_energy[n] >= inv.ratio_min * model.X_power[n])
    m.energy_ratio_max = pyo.Constraint(m.N, rule=lambda model, n: model.X_energy[n] <= inv.ratio_max * model.X_power[n])

    if fixed_power_mw is not None:
        for node in data.nodes:
            fixed_power = min(max(float(fixed_power_mw), 0.0), invest_limit[node])
            m.X_power[node].fix(fixed_power)
            m.X_energy[node].fix(init_ratio * fixed_power)

    # Lower-level primal variables.
    m.P_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_shed = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    if not use_demand_curve:
        for node in data.nodes:
            for time in data.times:
                m.P_shed[node, time].fix(0.0)
    m.P_charge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_discharge = pyo.Var(m.I, m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.SOC = pyo.Var(m.I, m.N, m.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)

    # Lower-level dual variables with broad finite bounds for Ipopt stability.
    m.lam = pyo.Var(m.N, m.T, bounds=(-dual_bound, dual_bound), initialize=80.0)
    m.lam_sys = pyo.Var(m.T, bounds=(-dual_bound, dual_bound), initialize=80.0)
    m.nu_gen = pyo.Var(m.G, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_up = pyo.Var(m.L, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_dn = pyo.Var(m.L, m.T, bounds=(0.0, dual_bound), initialize=0.0)
    m.rho_ch = pyo.Var(m.I, m.N, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.sig_dis = pyo.Var(m.I, m.N, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.gam = pyo.Var(m.I, m.N, m.T, bounds=(-dual_bound, dual_bound), initialize=0.0)
    m.del_soc = pyo.Var(m.I, m.N, m.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.rho_per = pyo.Var(m.I, m.N, bounds=(-dual_bound, dual_bound), initialize=0.0)
    m.xi_shed = pyo.Var(m.N, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)

    # Primal feasibility.
    def nodal_balance_rule(model: pyo.ConcreteModel, node: str, time: int) -> pyo.Expression:
        storage_net = sum(
            model.P_discharge[unit, node, time] - model.P_charge[unit, node, time] for unit in model.I
        )
        return (
            sum(model.P_gen[generator, time] for generator in data.generators_at_node.get(node, []))
            + storage_net
            + (model.P_shed[node, time] if use_demand_curve else 0.0)
            - data.demand_el[node, time]
            == model.NetInjection[node, time]
        )

    m.nodal_balance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)
    m.system_balance = pyo.Constraint(m.T, rule=lambda model, t: sum(model.NetInjection[n, t] for n in model.N) == 0.0)
    m.generation_capacity_bound = pyo.Constraint(
        m.G,
        m.T,
        rule=lambda model, g, t: model.P_gen[g, t] <= data.generation_capacity[g, t],
    )
    m.line_upper_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, l, t: sum(data.ptdf[l, n] * model.NetInjection[n, t] for n in model.N) <= data.line_limit[l],
    )
    m.line_lower_bound = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda model, l, t: sum(data.ptdf[l, n] * model.NetInjection[n, t] for n in model.N) >= -data.line_limit[l],
    )
    def unit_power_limit(model: pyo.ConcreteModel, i: str, n: str):
        return model.X_power[n] if i == inv.investor_id else rival_power_mw[n]

    m.charge_power_bound = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: model.P_charge[i, n, t] <= unit_power_limit(model, i, n),
    )
    m.discharge_power_bound = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: model.P_discharge[i, n, t] <= unit_power_limit(model, i, n),
    )
    m.soc_transition = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: model.SOC[i, n, t]
        == model.SOC[i, n, t - 1] + eta * model.P_charge[i, n, t] - model.P_discharge[i, n, t] / eta,
    )
    m.soc_capacity_bound = pyo.Constraint(
        m.I,
        m.N,
        m.T_SOC,
        rule=lambda model, i, n, tau: model.SOC[i, n, tau]
        <= (model.X_energy[n] if i == inv.investor_id else rival_energy_mwh[n]),
    )
    m.soc_periodicity = pyo.Constraint(
        m.I,
        m.N,
        rule=lambda model, i, n: model.SOC[i, n, 0] == model.SOC[i, n, last_t],
    )
    if use_demand_curve:
        m.load_shed_bound = pyo.Constraint(
            m.N,
            m.T,
            rule=lambda model, n, t: model.P_shed[n, t] <= data.demand_el[n, t],
        )
    else:
        m.load_shed_bound = pyo.Constraint(m.N, m.T, rule=lambda model, n, t: model.P_shed[n, t] == 0.0)

    # Dual feasibility.
    m.gen_stationarity = pyo.Constraint(
        m.G,
        m.T,
        rule=lambda model, g, t: sum(model.lam[n, t] for n in gen_nodes.get(g, [])) + model.nu_gen[g, t]
        <= data.generation_cost[g],
    )
    if use_demand_curve:
        # QP stationarity for P_shed: marginal cost is alpha + (beta/D)*shed.
        m.shed_stationarity = pyo.Constraint(
            m.N,
            m.T,
            rule=lambda model, n, t: model.lam[n, t] + model.xi_shed[n, t]
            <= quad_demand.alpha
            + quad_demand.quad_coefficient(data.demand_el[n, t]) * model.P_shed[n, t],
        )
    else:
        m.shed_stationarity = pyo.Constraint(m.N, m.T, rule=lambda model, n, t: model.xi_shed[n, t] == 0.0)
    m.charge_stationarity = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: -model.lam[n, t] + model.rho_ch[i, n, t] - eta * model.gam[i, n, t] <= 0.0,
    )
    m.discharge_stationarity = pyo.Constraint(
        m.I,
        m.N,
        m.T,
        rule=lambda model, i, n, t: model.lam[n, t] + model.sig_dis[i, n, t] + model.gam[i, n, t] / eta <= 0.0,
    )
    m.netinjection_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, n, t: -model.lam[n, t]
        + model.lam_sys[t]
        + sum(data.ptdf[l, n] * (model.mu_up[l, t] + model.mu_dn[l, t]) for l in model.L)
        == 0.0,
    )

    def soc_stationarity_rule(model: pyo.ConcreteModel, i: str, n: str, tau: int) -> pyo.Expression:
        expr = model.del_soc[i, n, tau]
        if tau in model.T:
            expr = expr + model.gam[i, n, tau]
        if (tau + 1) in model.T:
            expr = expr - model.gam[i, n, tau + 1]
        if tau == 0:
            expr = expr + model.rho_per[i, n]
        if tau == last_t:
            expr = expr - model.rho_per[i, n]
        return expr <= 0.0

    m.soc_stationarity = pyo.Constraint(m.I, m.N, m.T_SOC, rule=soc_stationarity_rule)

    # Strong duality. In fixed-demand mode the lower level is the LP without a
    # shed variable. In demand-curve mode it is a convex QP with Wolfe-dual
    # correction for the quadratic curtailment cost.
    if use_demand_curve:
        quad_cost_expr = sum(
            0.5 * quad_demand.quad_coefficient(data.demand_el[n, t]) * m.P_shed[n, t] ** 2
            for n in m.N
            for t in m.T
        )
        shed_cost_expr = sum(quad_demand.alpha * m.P_shed[n, t] for n in m.N for t in m.T) + quad_cost_expr
        demand_dual_expr = sum(
            data.demand_el[n, t] * (m.lam[n, t] + m.xi_shed[n, t]) for n in m.N for t in m.T
        )
        quad_dual_correction_expr = -quad_cost_expr
    else:
        shed_cost_expr = 0.0
        demand_dual_expr = sum(data.demand_el[n, t] * m.lam[n, t] for n in m.N for t in m.T)
        quad_dual_correction_expr = 0.0

    m.primal_objective_expr = pyo.Expression(
        expr=sum(data.generation_cost[g] * m.P_gen[g, t] for g in m.G for t in m.T)
        + shed_cost_expr
    )
    m.dual_objective_expr = pyo.Expression(
        expr=demand_dual_expr
        + quad_dual_correction_expr
        + sum(data.generation_capacity[g, t] * m.nu_gen[g, t] for g in m.G for t in m.T)
        + sum(data.line_limit[l] * (m.mu_up[l, t] - m.mu_dn[l, t]) for l in m.L for t in m.T)
        + sum(
            unit_power_limit(m, i, n) * (m.rho_ch[i, n, t] + m.sig_dis[i, n, t])
            for i in m.I
            for n in m.N
            for t in m.T
        )
        + sum(
            (m.X_energy[n] if i == inv.investor_id else rival_energy_mwh[n]) * m.del_soc[i, n, tau]
            for i in m.I
            for n in m.N
            for tau in m.T_SOC
        )
    )
    m.strong_duality = pyo.Constraint(expr=m.primal_objective_expr == m.dual_objective_expr)

    # Upper-level investor objective.
    # Price the investor is paid: nodal LMP (lam[n,t]) by default, or the
    # uniform system-wide price (lam_sys[t], the single bidding-zone / zonal
    # price that ignores congestion rent) when system_price_settlement is set.
    def settlement_price(n: str, t: int):
        return m.lam_sys[t] if system_price_settlement else m.lam[n, t]

    m.spot_revenue_expr = pyo.Expression(
        expr=sum(
            settlement_price(n, t)
            * (m.P_discharge[inv.investor_id, n, t] - m.P_charge[inv.investor_id, n, t])
            for n in m.N
            for t in m.T
        )
    )
    m.degradation_cost_expr = pyo.Expression(
        expr=0.5
        * inv.degradation_eur_per_mwh
        * sum(m.P_charge[inv.investor_id, n, t] + m.P_discharge[inv.investor_id, n, t] for n in m.N for t in m.T)
    )
    m.capex_daily_expr = pyo.Expression(
        expr=crf_daily
        * sum(
            inv.cost_power_eur_per_mw * m.X_power[n]
            + inv.cost_energy_eur_per_mwh * m.X_energy[n]
            for n in m.N
        )
    )
    m.investor_profit_expr = pyo.Expression(expr=m.spot_revenue_expr - m.degradation_cost_expr - m.capex_daily_expr)
    # Clean EPEC baseline: the MPEC selects the dual/primal optimum that
    # maximizes investor profit. Any optimistic-price effect is diagnosed ex
    # post by comparing these embedded prices to the standalone reference
    # settlement, not regularized inside the objective.
    m.objective = pyo.Objective(expr=m.investor_profit_expr, sense=pyo.maximize)

    m._market_data = data
    m._investor_id = inv.investor_id
    m._investor_config = inv
    m._wacc = inv.wacc
    m._degradation_eur_per_mwh = inv.degradation_eur_per_mwh
    m._existing_power_mw = existing_power_mw
    m._existing_ratio_hours = existing_ratio_hours
    m._node_limit_mw = node_limit_mw
    m._rival_id = rival_id
    m._rival_power_mw = dict(rival_power_mw)
    m._rival_energy_mwh = dict(rival_energy_mwh)
    m._quad_demand = quad_demand
    m._use_demand_curve = use_demand_curve
    return m


def _initialize_from_quadratic_llp(
    model: pyo.ConcreteModel, lp_data: MarketData, quad: QuadraticDemandCurve
) -> None:
    """Warm-start from the standalone lower-level clearing problem.

    Primal values come straight from the reference problem. In quadratic-demand
    mode the unique prices are read off the demand curve. In fixed-demand mode
    the solver nodal duals are used directly.
    """

    if model._use_demand_curve:
        qp = build_quadratic_primal_model(lp_data, quad)
    else:
        qp = build_fixed_demand_primal_model(lp_data)
    results = get_ipopt_solver().solve(qp, tee=False)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return

    lam_ref = quadratic_reference_lambda(qp, quad) if model._use_demand_curve else fixed_demand_reference_lambda(qp)
    gen_nodes = _nodes_of_generator(lp_data)
    node_count = max(1, len(list(model.N)))
    for g in model.G:
        for t in model.T:
            model.P_gen[g, t].set_value(max(0.0, value(qp.P_gen[g, t])))
            lam_g = sum(lam_ref[n, t] for n in gen_nodes.get(g, []))
            model.nu_gen[g, t].set_value(min(0.0, lp_data.generation_cost[g] - lam_g))
    for n in model.N:
        for t in model.T:
            model.P_shed[n, t].set_value(max(0.0, value(qp.P_shed[n, t])))
            model.NetInjection[n, t].set_value(value(qp.NetInjection[n, t]))
            model.lam[n, t].set_value(lam_ref[n, t])
            model.xi_shed[n, t].set_value(0.0)
    for t in model.T:
        model.lam_sys[t].set_value(sum(lam_ref[n, t] for n in model.N) / node_count)
    for i in model.I:
        for n in model.N:
            for t in model.T:
                model.P_charge[i, n, t].set_value(max(0.0, value(qp.P_charge[i, n, t])))
                model.P_discharge[i, n, t].set_value(max(0.0, value(qp.P_discharge[i, n, t])))
                model.gam[i, n, t].set_value(-lam_ref[n, t])
            for tau in model.T_SOC:
                model.SOC[i, n, tau].set_value(max(0.0, value(qp.SOC[i, n, tau])))


def initialize_from_reference_dispatch(model: pyo.ConcreteModel, data: MarketData, ratio_hours: float) -> None:
    """Warm-start primal and dual variables from the quadratic lower-level dispatch."""

    lp_data = single_storage_data(
        data,
        0.0,
        ratio_hours,
        investor_id=model._investor_id,
        rival_id=model._rival_id,
        rival_power_mw=model._rival_power_mw,
        rival_energy_mwh=model._rival_energy_mwh,
        power_by_node={n: value(model.X_power[n]) for n in model.N},
        energy_by_node={n: value(model.X_energy[n]) for n in model.N},
    )
    _initialize_from_quadratic_llp(model, lp_data, model._quad_demand)


def compute_reference_settlement(model: pyo.ConcreteModel) -> dict[str, object]:
    """Settle the MPEC solution against standalone lower-level reference prices.

    In demand-curve mode the QP has unique prices, read analytically off the
    demand curve at cleared curtailment. In fixed-demand mode there is no
    curtailment valve; the standalone LP fixes ``P_shed`` to zero.
    """

    fixed_data = fixed_storage_data_from_solution(model)
    quad: QuadraticDemandCurve = model._quad_demand
    if model._use_demand_curve:
        reference = build_quadratic_primal_model(fixed_data, quad)
        reference_problem = "QP"
    else:
        reference = build_fixed_demand_primal_model(fixed_data)
        reference_problem = "LP"
    solver_label = "ipopt"
    print(f"Solving reference settlement {reference_problem} with ipopt...")
    results = get_ipopt_solver().solve(reference, tee=False)
    termination = results.solver.termination_condition
    if termination != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"Reference settlement {reference_problem} did not solve optimally (termination={termination})."
        )

    reference_objective = value(reference.quad_objective if model._use_demand_curve else reference.objective)
    reference_lambda = (
        quadratic_reference_lambda(reference, quad)
        if model._use_demand_curve
        else fixed_demand_reference_lambda(reference)
    )
    dual_cross_check = _solver_dual_cross_check(reference, reference_lambda)

    mpec_lambda_max_abs_diff = max(
        abs(value(model.lam[n, t]) - reference_lambda[n, t]) for n in model.N for t in model.T
    )

    inv_id = model._investor_id
    lp_charge = sum(value(reference.P_charge[inv_id, n, t]) for n in reference.N for t in reference.T)
    lp_discharge = sum(value(reference.P_discharge[inv_id, n, t]) for n in reference.N for t in reference.T)
    lp_revenue = sum(
        reference_lambda[n, t]
        * (value(reference.P_discharge[inv_id, n, t]) - value(reference.P_charge[inv_id, n, t]))
        for n in reference.N
        for t in reference.T
    )
    lp_degradation = 0.5 * model._degradation_eur_per_mwh * (lp_charge + lp_discharge)
    capex = value(model.capex_daily_expr)
    lp_profit = lp_revenue - lp_degradation - capex

    mpec_dispatch_revenue = sum(
        reference_lambda[n, t] * (value(model.P_discharge[inv_id, n, t]) - value(model.P_charge[inv_id, n, t]))
        for n in model.N
        for t in model.T
    )
    mpec_dispatch_profit = mpec_dispatch_revenue - value(model.degradation_cost_expr) - capex

    return {
        "solver": solver_label,
        "problem": reference_problem,
        "solver_status": str(results.solver.status),
        "termination": str(termination),
        "lower_level_objective_eur_per_day": reference_objective,
        "mpec_lambda_max_abs_diff_vs_reference_eur_per_mwh": mpec_lambda_max_abs_diff,
        "reference_lambda_solver_dual_max_abs_diff": dual_cross_check,
        "spot_revenue_at_reference_prices_eur_per_day": lp_revenue,
        "degradation_cost_at_reference_dispatch_eur_per_day": lp_degradation,
        "profit_at_reference_prices_eur_per_day": lp_profit,
        "optimistic_minus_reference_profit_eur_per_day": value(model.investor_profit_expr) - lp_profit,
        "mpec_dispatch_spot_revenue_at_reference_prices_eur_per_day": mpec_dispatch_revenue,
        "mpec_dispatch_profit_at_reference_prices_eur_per_day": mpec_dispatch_profit,
        "optimistic_minus_mpec_dispatch_reference_profit_eur_per_day": value(model.investor_profit_expr)
        - mpec_dispatch_profit,
        "reference_total_charge_mwh": lp_charge,
        "reference_total_discharge_mwh": lp_discharge,
        "reference_total_storage_throughput_mwh": lp_charge + lp_discharge,
        "reference_lambda_min_eur_per_mwh": min(reference_lambda.values()),
        "reference_lambda_max_eur_per_mwh": max(reference_lambda.values()),
        "reference_lambda": reference_lambda,
        "reference_model": reference,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--data", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--tee", action="store_true", help="Show Ipopt output.")
    parser.add_argument("--initial-power-mw", type=float, default=DEFAULT_INITIAL_POWER_MW)
    parser.add_argument("--initial-ratio-hours", type=float, default=DEFAULT_INITIAL_RATIO_HOURS)
    parser.add_argument("--fixed-power-mw", type=float, default=None, help="Fix all nodal power capacities for validation.")
    parser.add_argument(
        "--existing-power-mw",
        type=float,
        default=0.0,
        help="Exogenous non-strategic BESS power at every node; reduces the investor's connection headroom.",
    )
    parser.add_argument(
        "--existing-ratio-hours",
        type=float,
        default=2.0,
        help="Energy-to-power ratio of the exogenous existing BESS fleet.",
    )
    parser.add_argument("--max-cpu-time", type=float, default=120.0)
    parser.add_argument(
        "--dual-bound-scale",
        type=float,
        default=10.0,
        help="Finite bound multiplier for lower-level dual variables relative to VOLL.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-export", action="store_true", help="Do not write detailed CSV/JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.data)
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR

    quad_demand = default_quadratic_demand_curve()
    if USE_DEMAND_CURVE:
        print(
            "Quadratic demand curve: "
            f"marginal WTP = {quad_demand.alpha:,.2f} + {quad_demand.beta:,.2f} * curtailed_share EUR/MWh"
        )
    else:
        print("Fixed demand mode: demand curve disabled; P_shed fixed to 0 MW; no VOLL/load-shed pricing.")

    model = build_single_investor_mpec(
        data,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        fixed_power_mw=args.fixed_power_mw,
        dual_bound_scale=args.dual_bound_scale,
        existing_power_mw=args.existing_power_mw,
        existing_ratio_hours=args.existing_ratio_hours,
        quad_demand=quad_demand,
        use_demand_curve=USE_DEMAND_CURVE,
    )
    initialize_from_reference_dispatch(model, data, args.initial_ratio_hours)

    solver = get_ipopt_solver({"max_cpu_time": args.max_cpu_time})
    results = solver.solve(model, tee=args.tee)
    termination = results.solver.termination_condition
    print(f"Solver status: {results.solver.status}")
    print(f"Termination: {termination}")
    if termination != pyo.TerminationCondition.optimal:
        print("MPEC solve did not terminate optimally.")
        return 1

    reference_settlement = compute_reference_settlement(model)
    if not model._use_demand_curve and reference_settlement["mpec_lambda_max_abs_diff_vs_reference_eur_per_mwh"] > 1.0:
        print(
            "WARNING: fixed-demand LP prices are non-unique here; the embedded MPEC selected different "
            "dual prices than the standalone reference LP. Treat optimistic MPEC profit with caution."
        )

    print_solution_summary(model, reference_settlement)
    print_lambda_and_line_duals(model)
    if not args.no_export:
        export_solution(
            model,
            output_dir,
            str(results.solver.status),
            str(termination),
            reference_settlement,
        )
        print(f"\nWrote detailed MPEC outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
