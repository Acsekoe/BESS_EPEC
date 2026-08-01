"""Single-investor MPEC with optional quadratic demand and strong duality.

This proof model represents one strategic BESS investor in the deterministic
spot market. The lower-level market clearing is embedded through primal
feasibility, dual feasibility, and a strong-duality equality. By default demand
is fixed: no load shedding, no VOLL scarcity valve, and no demand-response curve.
The quadratic demand curve can be re-enabled for feasibility experiments.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import pyomo.environ as pyo

_MODEL_DIR = Path(__file__).resolve().parent
_PRIMAL_DUAL_DIR = _MODEL_DIR / "Primal and dual problems"
if _PRIMAL_DUAL_DIR.is_dir() and str(_PRIMAL_DUAL_DIR) not in sys.path:
    sys.path.append(str(_PRIMAL_DUAL_DIR))

from primal_market_clearing_model import (
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
# Maintained IEEE-9 thesis input. Historical and sensitivity datasets remain
# available through the explicit --data option.
EXPERIMENT_DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "processed"
    / "market_data_IEEE_9Bus_congestion.json"
)
DEFAULT_WACC = 0.08
DEFAULT_LIFETIME_YEARS = 15
DEFAULT_NODE_LIMIT_MW = 100.0
DEFAULT_RATIO_MIN = 2.0
DEFAULT_RATIO_MAX = 8.0
DEFAULT_BESS_COST_POWER_EUR_PER_MW = 6_600.0
DEFAULT_BESS_COST_ENERGY_EUR_PER_MWH = 18_800.0
DEFAULT_DEGRADATION_EUR_PER_MWH = 15.0
# Optional strictly-convex tie-break in the lower-level dispatch objective.
# The maintained base model leaves it disabled; robustness runs can enable it
# explicitly through the CLI/configuration.
DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H = 0.0
DEFAULT_SOLVER_TOL = 1.0e-4
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "single_investor_mpec"
USE_DEMAND_CURVE = False
DEFAULT_INITIAL_POWER_MW = 10.0
DEFAULT_INITIAL_RATIO_HOURS = DEFAULT_RATIO_MIN
DEFAULT_DEMAND_CURVE_ALPHA = 4000
DEFAULT_DEMAND_CURVE_BETA = 10000
DEFAULT_PRICE_BOUND_EUR_PER_MWH = 500.0
DEFAULT_DUAL_BOUND_EUR_PER_MWH = 10_000.0
SPARSE_RIVAL_CAPACITY_TOL = 1.0e-8
SPARSE_GENERATION_CAPACITY_TOL = 1.0e-8
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
    # Fraction of each existing generator's rent this investor collects (0..1).
    # Empty => stand-alone merchant BESS. A portfolio-backed investor owns a
    # share of the exogenous wind/PV/thermal fleet already in the lower level
    # and earns its inframarginal spot rent alongside BESS arbitrage.
    owned_generation_shares: Mapping[str, float] = field(default_factory=dict)


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


def _storage_degradation_costs(
    data: MarketData,
    overrides: Mapping[str, float] | None,
) -> dict[str, float]:
    """Per-unit full-cycle degradation coefficients used by market clearing."""

    costs = {str(unit): DEFAULT_DEGRADATION_EUR_PER_MWH for unit in data.storage_units}
    if overrides is not None:
        unknown = set(overrides) - set(costs)
        if unknown:
            raise ValueError(f"Degradation costs supplied for unknown storage units: {sorted(unknown)}")
        costs.update({str(unit): float(cost) for unit, cost in overrides.items()})
    if any(cost < 0.0 for cost in costs.values()):
        raise ValueError("Storage degradation costs must be non-negative.")
    return costs


def build_quadratic_primal_model(
    data: MarketData,
    quad: QuadraticDemandCurve,
    *,
    storage_degradation_eur_per_mwh: Mapping[str, float] | None = None,
    dispatch_regularization_eur_per_mw2h: float = DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
) -> pyo.ConcreteModel:
    """Lower-level clearing QP: the primal LP structure plus the quadratic shed cost.

    Reuses the standalone primal builder, where ``P_shed[n,t]`` is bounded by
    full nodal demand, then swaps in the strictly convex objective.
    """

    if dispatch_regularization_eur_per_mw2h < 0.0:
        raise ValueError("Dispatch regularization must be non-negative.")

    m = build_primal_market_clearing_model(data)
    md: MarketData = m._market_data
    degradation = _storage_degradation_costs(md, storage_degradation_eur_per_mwh)
    reg = dispatch_regularization_eur_per_mw2h
    generation_cost = sum(md.generation_cost[g] * m.P_gen[g, t] for g in m.G for t in m.T)
    storage_degradation_cost = sum(
        0.5 * degradation[i] * (m.P_charge[i, n, t] + m.P_discharge[i, n, t])
        for i in m.I
        for n in m.N
        for t in m.T
    )
    shed_linear_cost = sum(quad.alpha * m.P_shed[n, t] for n in m.N for t in m.T)
    quad_cost = sum(
        0.5 * quad.quad_coefficient(md.demand_el[n, t]) * m.P_shed[n, t] ** 2
        for n in m.N
        for t in m.T
    )
    dispatch_regularization = 0.5 * reg * (
        sum(m.P_gen[g, t] ** 2 for g in m.G for t in m.T)
        + sum(
            m.P_charge[i, n, t] ** 2 + m.P_discharge[i, n, t] ** 2
            for i in m.I
            for n in m.N
            for t in m.T
        )
        + sum(m.SOC[i, n, tau] ** 2 for i in m.I for n in m.N for tau in m.T_SOC)
        + sum(m.NetInjection[n, t] ** 2 for n in m.N for t in m.T)
        + sum(m.P_shed[n, t] ** 2 for n in m.N for t in m.T)
    )
    m.objective.deactivate()
    m.storage_degradation_objective_expr = pyo.Expression(expr=storage_degradation_cost)
    m.dispatch_regularization_expr = pyo.Expression(expr=dispatch_regularization)
    m.quad_objective = pyo.Objective(
        expr=generation_cost
        + storage_degradation_cost
        + shed_linear_cost
        + quad_cost
        + dispatch_regularization,
        sense=pyo.minimize,
    )
    m._quad_demand = quad
    m._storage_degradation_eur_per_mwh = degradation
    m._dispatch_regularization_eur_per_mw2h = reg
    return m


def build_fixed_demand_primal_model(
    data: MarketData,
    *,
    storage_degradation_eur_per_mwh: Mapping[str, float] | None = None,
    dispatch_regularization_eur_per_mw2h: float = DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
) -> pyo.ConcreteModel:
    """Standalone regularized lower-level QP with fixed demand and no shedding."""

    if dispatch_regularization_eur_per_mw2h < 0.0:
        raise ValueError("Dispatch regularization must be non-negative.")

    m = build_primal_market_clearing_model(data, include_load_shed=False)
    degradation = _storage_degradation_costs(data, storage_degradation_eur_per_mwh)
    reg = dispatch_regularization_eur_per_mw2h
    generation_cost = sum(data.generation_cost[g] * m.P_gen[g, t] for g in m.G for t in m.T)
    storage_degradation_cost = sum(
        0.5 * degradation[i] * (m.P_charge[i, n, t] + m.P_discharge[i, n, t])
        for i in m.I
        for n in m.N
        for t in m.T
    )
    dispatch_regularization = 0.5 * reg * (
        sum(m.P_gen[g, t] ** 2 for g in m.G for t in m.T)
        + sum(
            m.P_charge[i, n, t] ** 2 + m.P_discharge[i, n, t] ** 2
            for i in m.I
            for n in m.N
            for t in m.T
        )
        + sum(m.SOC[i, n, tau] ** 2 for i in m.I for n in m.N for tau in m.T_SOC)
        + sum(m.NetInjection[n, t] ** 2 for n in m.N for t in m.T)
    )
    m.objective.set_value(generation_cost + storage_degradation_cost + dispatch_regularization)
    m.storage_degradation_objective_expr = pyo.Expression(expr=storage_degradation_cost)
    m.dispatch_regularization_expr = pyo.Expression(expr=dispatch_regularization)
    m._storage_degradation_eur_per_mwh = degradation
    m._dispatch_regularization_eur_per_mw2h = reg
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
    reg = getattr(reference, "_dispatch_regularization_eur_per_mw2h", 0.0)
    curve: dict[tuple[str, int], float] = {}
    for n in reference.N:
        for t in reference.T:
            demand = md.demand_el[n, t]
            shed = value(reference.P_shed[n, t])
            if demand > 0.0 and SHED_INTERIOR_TOL_MW < shed < demand - SHED_INTERIOR_TOL_MW:
                curve[(n, t)] = quad.marginal_wtp(shed, demand) + reg * shed
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
    rival_power_mw_by_unit: Mapping[str, Mapping[str, float]] | None = None,
    rival_energy_mwh_by_unit: Mapping[str, Mapping[str, float]] | None = None,
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
    if rival_power_mw_by_unit is None:
        if rival_power_mw is None:
            rival_power_mw = {node: existing_power_mw for node in data.nodes}
            rival_energy_mwh = {node: existing_power_mw * existing_ratio_hours for node in data.nodes}
        rival_power_mw_by_unit = (
            {rival_id: rival_power_mw} if any(v > 1e-9 for v in rival_power_mw.values()) else {}
        )
        rival_energy_mwh_by_unit = (
            {rival_id: rival_energy_mwh} if rival_power_mw_by_unit else {}
        )
    elif rival_energy_mwh_by_unit is None:
        raise ValueError("rival_energy_mwh_by_unit is required with rival_power_mw_by_unit.")
    for unit, nodal_power in rival_power_mw_by_unit.items():
        units.append(unit)
        nodal_energy = rival_energy_mwh_by_unit[unit]
        for node in data.nodes:
            x_power[(unit, node)] = float(nodal_power[node])
            x_energy[(unit, node)] = float(nodal_energy[node])
    return replace(data, storage_units=units, x_power=x_power, x_energy=x_energy)


def fixed_storage_data_from_solution(model: pyo.ConcreteModel) -> MarketData:
    """Return lower-level data with storage capacities fixed at the MPEC solution."""

    data: MarketData = model._market_data
    investor_id = model._investor_id
    units = [investor_id]
    x_power = {(investor_id, node): max(0.0, value(model.X_power[node])) for node in data.nodes}
    x_energy = {(investor_id, node): max(0.0, value(model.X_energy[node])) for node in data.nodes}
    for rival_id in model._rival_ids:
        units.append(rival_id)
        for node in data.nodes:
            x_power[(rival_id, node)] = model._rival_power_mw_by_unit[rival_id][node]
            x_energy[(rival_id, node)] = model._rival_energy_mwh_by_unit[rival_id][node]
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
    price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    price_lower_bound_eur_per_mwh: float | None = None,
    price_upper_bound_eur_per_mwh: float | None = None,
    dual_bound_eur_per_mwh: float = DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    existing_power_mw: float = 0.0,
    existing_ratio_hours: float = 2.0,
    quad_demand: QuadraticDemandCurve,
    use_demand_curve: bool = USE_DEMAND_CURVE,
    investor: InvestorConfig | None = None,
    rival_id: str = EXISTING_ID,
    rival_power_mw: Mapping[str, float] | None = None,
    rival_energy_mwh: Mapping[str, float] | None = None,
    rival_power_mw_by_unit: Mapping[str, Mapping[str, float]] | None = None,
    rival_energy_mwh_by_unit: Mapping[str, Mapping[str, float]] | None = None,
    rival_degradation_eur_per_mwh: float = DEFAULT_DEGRADATION_EUR_PER_MWH,
    rival_degradation_eur_per_mwh_by_unit: Mapping[str, float] | None = None,
    dispatch_regularization_eur_per_mw2h: float = DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
    system_price_settlement: bool = False,
    solver_tol: float = DEFAULT_SOLVER_TOL,
) -> pyo.ConcreteModel:
    """Build the one-investor MPEC.

    Rival batteries can be supplied separately through
    ``rival_power_mw_by_unit``/``rival_energy_mwh_by_unit``. The legacy
    single-rival inputs remain supported for standalone and archived callers.

    The shared nodal connection limit is enforced as an explicit private
    headroom constraint: the investor can only add up to
    ``node_limit_mw - rival_power_mw[n]`` at each node. Its imported NLP
    multiplier is the investor-specific endogenous marginal value of one more
    MW of nodal access; it is not a common market-clearing access price.

    If ``use_demand_curve`` is false, demand is fixed and no load-shedding
    primal or dual variables are created. If true, the lower level uses one quadratic
    curtailment variable per node-hour, marginal WTP
    ``alpha + beta*(shed/demand)``, and a Wolfe-dual strong-duality equality.
    """

    if price_bound_eur_per_mwh <= 0.0:
        raise ValueError("price_bound_eur_per_mwh must be positive.")
    price_lower_bound = (
        -float(price_bound_eur_per_mwh)
        if price_lower_bound_eur_per_mwh is None
        else float(price_lower_bound_eur_per_mwh)
    )
    price_upper_bound = (
        float(price_bound_eur_per_mwh)
        if price_upper_bound_eur_per_mwh is None
        else float(price_upper_bound_eur_per_mwh)
    )
    if price_lower_bound >= price_upper_bound:
        raise ValueError("The lower electricity-price bound must be below the upper bound.")
    if dual_bound_eur_per_mwh <= 0.0:
        raise ValueError("dual_bound_eur_per_mwh must be positive.")
    if investor is not None and (wacc, ratio_min, ratio_max) != (DEFAULT_WACC, DEFAULT_RATIO_MIN, DEFAULT_RATIO_MAX):
        raise ValueError("Pass economic parameters through `investor`, not the legacy scalar kwargs.")
    inv = investor or InvestorConfig(wacc=wacc, ratio_min=ratio_min, ratio_max=ratio_max)
    if rival_degradation_eur_per_mwh < 0.0:
        raise ValueError("Rival degradation cost must be non-negative.")
    if dispatch_regularization_eur_per_mw2h < 0.0:
        raise ValueError("Dispatch regularization must be non-negative.")
    if solver_tol <= 0.0:
        raise ValueError("solver_tol must be positive.")
    if existing_power_mw < 0.0:
        raise ValueError("existing_power_mw must be non-negative.")
    if rival_power_mw_by_unit is not None and (rival_power_mw is not None or existing_power_mw > 0.0):
        raise ValueError("Pass separate rival units or the legacy single-rival inputs, not both.")
    if rival_power_mw_by_unit is None:
        if rival_power_mw is None:
            rival_power_mw = {node: existing_power_mw for node in data.nodes}
            rival_energy_mwh = {node: existing_power_mw * existing_ratio_hours for node in data.nodes}
        elif rival_energy_mwh is None:
            raise ValueError("rival_energy_mwh is required when rival_power_mw is given.")
        rival_power_mw_by_unit = (
            {rival_id: rival_power_mw} if any(v > 1e-9 for v in rival_power_mw.values()) else {}
        )
        rival_energy_mwh_by_unit = (
            {rival_id: rival_energy_mwh} if rival_power_mw_by_unit else {}
        )
    elif rival_energy_mwh_by_unit is None:
        raise ValueError("rival_energy_mwh_by_unit is required with rival_power_mw_by_unit.")

    rival_ids = list(rival_power_mw_by_unit)
    if inv.investor_id in rival_ids:
        raise ValueError(f"Active investor {inv.investor_id} cannot also be a rival unit.")
    if set(rival_energy_mwh_by_unit) != set(rival_ids):
        raise ValueError("Rival power and energy unit identifiers must match.")
    if (
        rival_degradation_eur_per_mwh_by_unit is not None
        and set(rival_degradation_eur_per_mwh_by_unit) != set(rival_ids)
    ):
        raise ValueError("Rival degradation-cost unit identifiers must match rival capacities.")
    if (
        rival_degradation_eur_per_mwh_by_unit is not None
        and any(float(cost) < 0.0 for cost in rival_degradation_eur_per_mwh_by_unit.values())
    ):
        raise ValueError("Rival degradation costs must be non-negative.")
    normalized_rival_power = {
        unit: {node: float(rival_power_mw_by_unit[unit][node]) for node in data.nodes}
        for unit in rival_ids
    }
    normalized_rival_energy = {
        unit: {node: float(rival_energy_mwh_by_unit[unit][node]) for node in data.nodes}
        for unit in rival_ids
    }
    for unit in rival_ids:
        for node in data.nodes:
            if normalized_rival_power[unit][node] < 0.0 or normalized_rival_energy[unit][node] < 0.0:
                raise ValueError(f"Negative rival capacity for {unit} at node {node}.")
    aggregate_rival_power = {
        node: sum(normalized_rival_power[unit][node] for unit in rival_ids) for node in data.nodes
    }
    aggregate_rival_energy = {
        node: sum(normalized_rival_energy[unit][node] for unit in rival_ids) for node in data.nodes
    }
    for node in data.nodes:
        if aggregate_rival_power[node] > node_limit_mw + 1e-9:
            raise ValueError(f"Rival power at node {node} exceeds the nodal connection limit.")
    invest_limit = {node: max(0.0, node_limit_mw - aggregate_rival_power[node]) for node in data.nodes}
    storage_units = [inv.investor_id] + rival_ids
    storage_pairs = [(inv.investor_id, node) for node in data.nodes]
    storage_pairs.extend(
        (unit, node)
        for unit in rival_ids
        for node in data.nodes
        if normalized_rival_power[unit][node] > SPARSE_RIVAL_CAPACITY_TOL
        or normalized_rival_energy[unit][node] > SPARSE_RIVAL_CAPACITY_TOL
    )
    storage_units_at_node = {
        node: [unit for unit, pair_node in storage_pairs if pair_node == node]
        for node in data.nodes
    }
    storage_degradation = {inv.investor_id: inv.degradation_eur_per_mwh}
    for unit in rival_ids:
        storage_degradation[unit] = (
            float(rival_degradation_eur_per_mwh_by_unit[unit])
            if rival_degradation_eur_per_mwh_by_unit is not None
            else rival_degradation_eur_per_mwh
        )

    gen_nodes = _nodes_of_generator(data)
    generation_pairs = [
        (generator, time)
        for generator in data.generators
        for time in data.times
        if data.generation_capacity[generator, time] > SPARSE_GENERATION_CAPACITY_TOL
    ]
    generators_at_node_time = {
        (node, time): [
            generator
            for generator in data.generators_at_node.get(node, [])
            if (generator, time) in generation_pairs
        ]
        for node in data.nodes
        for time in data.times
    }
    last_t = max(data.times)
    eta = data.eta
    dispatch_reg = dispatch_regularization_eur_per_mw2h
    crf_daily = capital_recovery_factor(inv.wacc, inv.lifetime_years) / 365.25
    # These are explicit absolute numerical bounds, not VOLL multipliers.
    price_bound = float(price_bound_eur_per_mwh)
    dual_bound = float(dual_bound_eur_per_mwh)

    m = pyo.ConcreteModel(name=MODEL_NAME)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.G = pyo.Set(initialize=data.generators, ordered=True)
    m.GT = pyo.Set(dimen=2, initialize=generation_pairs, ordered=True)
    m.I = pyo.Set(initialize=storage_units, ordered=True)
    m.IN = pyo.Set(dimen=2, initialize=storage_pairs, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)

    # Upper-level investment variables. Rival capacity consumes part of the
    # shared nodal connection limit, shrinking the investor's headroom per node.
    init_ratio = max(initial_ratio_hours, inv.ratio_min)
    m.X_power = pyo.Var(
        m.N,
        domain=pyo.NonNegativeReals,
        initialize=lambda model, n: min(max(initial_power_mw, 0.0), invest_limit[n]),
    )
    m.X_energy = pyo.Var(
        m.N,
        domain=pyo.NonNegativeReals,
        initialize=lambda model, n: init_ratio * min(max(initial_power_mw, 0.0), invest_limit[n]),
    )
    m.investment_headroom = pyo.Constraint(
        m.N,
        rule=lambda model, n: model.X_power[n] <= invest_limit[n],
    )
    m.energy_ratio_min = pyo.Constraint(m.N, rule=lambda model, n: model.X_energy[n] >= inv.ratio_min * model.X_power[n])
    m.energy_ratio_max = pyo.Constraint(m.N, rule=lambda model, n: model.X_energy[n] <= inv.ratio_max * model.X_power[n])

    if fixed_power_mw is not None:
        for node in data.nodes:
            fixed_power = min(max(float(fixed_power_mw), 0.0), invest_limit[node])
            m.X_power[node].fix(fixed_power)
            m.X_energy[node].fix(init_ratio * fixed_power)

    # Lower-level primal variables.
    m.P_gen = pyo.Var(m.GT, domain=pyo.NonNegativeReals, initialize=0.0)
    if use_demand_curve:
        m.P_shed = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_charge = pyo.Var(m.IN, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_discharge = pyo.Var(m.IN, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.SOC = pyo.Var(m.IN, m.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)

    # Lower-level dual variables with broad finite bounds for Ipopt stability.
    price_initial = min(price_upper_bound, max(price_lower_bound, 80.0))
    m.lam = pyo.Var(
        m.N,
        m.T,
        bounds=(price_lower_bound, price_upper_bound),
        initialize=price_initial,
    )
    m.lam_sys = pyo.Var(
        m.T,
        bounds=(price_lower_bound, price_upper_bound),
        initialize=price_initial,
    )
    m.nu_gen = pyo.Var(m.GT, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_up = pyo.Var(m.L, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_dn = pyo.Var(m.L, m.T, bounds=(0.0, dual_bound), initialize=0.0)
    m.rho_ch = pyo.Var(m.IN, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.sig_dis = pyo.Var(m.IN, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.gam = pyo.Var(m.IN, m.T, bounds=(-dual_bound, dual_bound), initialize=0.0)
    m.del_soc = pyo.Var(m.IN, m.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.rho_per = pyo.Var(m.IN, bounds=(-dual_bound, dual_bound), initialize=0.0)
    if use_demand_curve:
        m.xi_shed = pyo.Var(m.N, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)

    # Primal feasibility.
    def nodal_balance_rule(model: pyo.ConcreteModel, node: str, time: int) -> pyo.Expression:
        storage_net = sum(
            model.P_discharge[unit, node, time] - model.P_charge[unit, node, time]
            for unit in storage_units_at_node[node]
        )
        return (
            sum(model.P_gen[generator, time] for generator in generators_at_node_time[node, time])
            + storage_net
            + (model.P_shed[node, time] if use_demand_curve else 0.0)
            - data.demand_el[node, time]
            == model.NetInjection[node, time]
        )

    m.nodal_balance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)
    m.system_balance = pyo.Constraint(m.T, rule=lambda model, t: sum(model.NetInjection[n, t] for n in model.N) == 0.0)
    m.generation_capacity_bound = pyo.Constraint(
        m.GT,
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
        return model.X_power[n] if i == inv.investor_id else normalized_rival_power[i][n]

    m.charge_power_bound = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: model.P_charge[i, n, t] <= unit_power_limit(model, i, n),
    )
    m.discharge_power_bound = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: model.P_discharge[i, n, t] <= unit_power_limit(model, i, n),
    )
    m.soc_transition = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: model.SOC[i, n, t]
        == model.SOC[i, n, t - 1] + eta * model.P_charge[i, n, t] - model.P_discharge[i, n, t] / eta,
    )
    m.soc_capacity_bound = pyo.Constraint(
        m.IN,
        m.T_SOC,
        rule=lambda model, i, n, tau: model.SOC[i, n, tau]
        <= (model.X_energy[n] if i == inv.investor_id else normalized_rival_energy[i][n]),
    )
    m.soc_periodicity = pyo.Constraint(
        m.IN,
        rule=lambda model, i, n: model.SOC[i, n, 0] == model.SOC[i, n, last_t],
    )
    if use_demand_curve:
        m.load_shed_bound = pyo.Constraint(
            m.N,
            m.T,
            rule=lambda model, n, t: model.P_shed[n, t] <= data.demand_el[n, t],
        )

    # Dual feasibility.
    m.gen_stationarity = pyo.Constraint(
        m.GT,
        rule=lambda model, g, t: sum(model.lam[n, t] for n in gen_nodes.get(g, [])) + model.nu_gen[g, t]
        <= data.generation_cost[g] + dispatch_reg * model.P_gen[g, t],
    )
    if use_demand_curve:
        # QP stationarity for P_shed: marginal cost is alpha + (beta/D)*shed.
        m.shed_stationarity = pyo.Constraint(
            m.N,
            m.T,
            rule=lambda model, n, t: model.lam[n, t] + model.xi_shed[n, t]
            <= quad_demand.alpha
            + (
                quad_demand.quad_coefficient(data.demand_el[n, t]) + dispatch_reg
            )
            * model.P_shed[n, t],
        )
    m.charge_stationarity = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: -model.lam[n, t]
        + model.rho_ch[i, n, t]
        - eta * model.gam[i, n, t]
        <= 0.5 * storage_degradation[i] + dispatch_reg * model.P_charge[i, n, t],
    )
    m.discharge_stationarity = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda model, i, n, t: model.lam[n, t]
        + model.sig_dis[i, n, t]
        + model.gam[i, n, t] / eta
        <= 0.5 * storage_degradation[i] + dispatch_reg * model.P_discharge[i, n, t],
    )
    m.netinjection_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda model, n, t: -model.lam[n, t]
        + model.lam_sys[t]
        + sum(data.ptdf[l, n] * (model.mu_up[l, t] + model.mu_dn[l, t]) for l in model.L)
        == dispatch_reg * model.NetInjection[n, t],
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
        return expr <= dispatch_reg * model.SOC[i, n, tau]

    m.soc_stationarity = pyo.Constraint(m.IN, m.T_SOC, rule=soc_stationarity_rule)

    # Strong duality. In fixed-demand mode the lower level is the LP without a
    # shed variable. In demand-curve mode it is a convex QP with Wolfe-dual
    # correction for the quadratic curtailment cost.
    if use_demand_curve:
        demand_quad_cost_expr = sum(
            0.5 * quad_demand.quad_coefficient(data.demand_el[n, t]) * m.P_shed[n, t] ** 2
            for n in m.N
            for t in m.T
        )
        shed_cost_expr = (
            sum(quad_demand.alpha * m.P_shed[n, t] for n in m.N for t in m.T)
            + demand_quad_cost_expr
        )
        demand_dual_expr = sum(
            data.demand_el[n, t] * (m.lam[n, t] + m.xi_shed[n, t]) for n in m.N for t in m.T
        )
    else:
        shed_cost_expr = 0.0
        demand_quad_cost_expr = 0.0
        demand_dual_expr = sum(data.demand_el[n, t] * m.lam[n, t] for n in m.N for t in m.T)

    storage_degradation_cost_expr = sum(
        0.5 * storage_degradation[i] * (m.P_charge[i, n, t] + m.P_discharge[i, n, t])
        for i, n in m.IN
        for t in m.T
    )
    dispatch_regularization_expr = 0.5 * dispatch_reg * (
        sum(m.P_gen[g, t] ** 2 for g, t in m.GT)
        + sum(
            m.P_charge[i, n, t] ** 2 + m.P_discharge[i, n, t] ** 2
            for i, n in m.IN
            for t in m.T
        )
        + sum(m.SOC[i, n, tau] ** 2 for i, n in m.IN for tau in m.T_SOC)
        + sum(m.NetInjection[n, t] ** 2 for n in m.N for t in m.T)
        + (
            sum(m.P_shed[n, t] ** 2 for n in m.N for t in m.T)
            if use_demand_curve
            else 0.0
        )
    )
    quad_dual_correction_expr = -demand_quad_cost_expr - dispatch_regularization_expr

    m.lower_level_storage_degradation_expr = pyo.Expression(expr=storage_degradation_cost_expr)
    m.dispatch_regularization_expr = pyo.Expression(expr=dispatch_regularization_expr)

    m.primal_objective_expr = pyo.Expression(
        expr=sum(data.generation_cost[g] * m.P_gen[g, t] for g, t in m.GT)
        + storage_degradation_cost_expr
        + shed_cost_expr
        + dispatch_regularization_expr
    )
    m.dual_objective_expr = pyo.Expression(
        expr=demand_dual_expr
        + quad_dual_correction_expr
        + sum(data.generation_capacity[g, t] * m.nu_gen[g, t] for g, t in m.GT)
        + sum(data.line_limit[l] * (m.mu_up[l, t] - m.mu_dn[l, t]) for l in m.L for t in m.T)
        + sum(
            unit_power_limit(m, i, n) * (m.rho_ch[i, n, t] + m.sig_dis[i, n, t])
            for i, n in m.IN
            for t in m.T
        )
        + sum(
            (m.X_energy[n] if i == inv.investor_id else normalized_rival_energy[i][n]) * m.del_soc[i, n, tau]
            for i, n in m.IN
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
    # Portfolio-backed rent: the investor's owned share of each existing
    # generator's inframarginal spot rent (price minus marginal cost times
    # cleared output), settled at the same price as its BESS. Zero for a
    # stand-alone merchant (empty owned_generation_shares).
    gen_node = {g: (gen_nodes.get(g) or [None])[0] for g in data.generators}
    m.generation_rent_expr = pyo.Expression(
        expr=sum(
            share * (settlement_price(gen_node[g], t) - data.generation_cost[g]) * m.P_gen[g, t]
            for g, share in inv.owned_generation_shares.items()
            for t in m.T
            if share != 0.0 and gen_node.get(g) is not None and (g, t) in m.GT
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
    m.investor_profit_expr = pyo.Expression(
        expr=m.spot_revenue_expr
        + m.generation_rent_expr
        - m.degradation_cost_expr
        - m.capex_daily_expr
    )
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
    m._invest_limit_mw = dict(invest_limit)
    m._rival_id = rival_id
    m._rival_ids = tuple(rival_ids)
    m._rival_power_mw = dict(aggregate_rival_power)
    m._rival_energy_mwh = dict(aggregate_rival_energy)
    m._rival_power_mw_by_unit = normalized_rival_power
    m._rival_energy_mwh_by_unit = normalized_rival_energy
    m._storage_pairs = tuple(storage_pairs)
    m._sparse_rival_capacity_tol = SPARSE_RIVAL_CAPACITY_TOL
    m._generation_pairs = tuple(generation_pairs)
    m._sparse_generation_capacity_tol = SPARSE_GENERATION_CAPACITY_TOL
    m._storage_degradation_eur_per_mwh = storage_degradation
    m._rival_degradation_eur_per_mwh = rival_degradation_eur_per_mwh
    m._dispatch_regularization_eur_per_mw2h = dispatch_reg
    m._quad_demand = quad_demand
    m._use_demand_curve = use_demand_curve
    m._price_bound_eur_per_mwh = price_bound
    m._price_lower_bound_eur_per_mwh = price_lower_bound
    m._price_upper_bound_eur_per_mwh = price_upper_bound
    m._dual_bound_eur_per_mwh = dual_bound
    m._solver_tol = solver_tol
    return m


def investment_headroom_shadow_price(model: pyo.ConcreteModel, node: str) -> float:
    """Return the endogenous ULP headroom multiplier in EUR/MW/day."""

    raw = model.dual.get(model.investment_headroom[node], float("nan"))
    value_ = float(raw)
    return 0.0 if abs(value_) < 1e-8 else value_


def _initialize_from_quadratic_llp(
    model: pyo.ConcreteModel, lp_data: MarketData, quad: QuadraticDemandCurve
) -> None:
    """Warm-start from the standalone lower-level clearing problem.

    Primal values come straight from the reference problem. In quadratic-demand
    mode the unique prices are read off the demand curve. In fixed-demand mode
    the solver nodal duals are used directly.
    """

    if model._use_demand_curve:
        qp = build_quadratic_primal_model(
            lp_data,
            quad,
            storage_degradation_eur_per_mwh=model._storage_degradation_eur_per_mwh,
            dispatch_regularization_eur_per_mw2h=model._dispatch_regularization_eur_per_mw2h,
        )
    else:
        qp = build_fixed_demand_primal_model(
            lp_data,
            storage_degradation_eur_per_mwh=model._storage_degradation_eur_per_mwh,
            dispatch_regularization_eur_per_mw2h=model._dispatch_regularization_eur_per_mw2h,
        )
    results = get_ipopt_solver(
        {"tol": model._solver_tol, "acceptable_tol": model._solver_tol}
    ).solve(qp, tee=False)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return

    lam_ref = quadratic_reference_lambda(qp, quad) if model._use_demand_curve else fixed_demand_reference_lambda(qp)
    def clipped_price(raw: float) -> float:
        return min(
            model._price_upper_bound_eur_per_mwh,
            max(model._price_lower_bound_eur_per_mwh, float(raw)),
        )

    gen_nodes = _nodes_of_generator(lp_data)
    node_count = max(1, len(list(model.N)))
    for g, t in model.GT:
        model.P_gen[g, t].set_value(max(0.0, value(qp.P_gen[g, t])))
        lam_g = sum(clipped_price(lam_ref[n, t]) for n in gen_nodes.get(g, []))
        marginal_cost = (
            lp_data.generation_cost[g]
            + model._dispatch_regularization_eur_per_mw2h * value(qp.P_gen[g, t])
        )
        model.nu_gen[g, t].set_value(min(0.0, marginal_cost - lam_g))
    for n in model.N:
        for t in model.T:
            model.NetInjection[n, t].set_value(value(qp.NetInjection[n, t]))
            model.lam[n, t].set_value(clipped_price(lam_ref[n, t]))
            if model._use_demand_curve:
                model.P_shed[n, t].set_value(max(0.0, value(qp.P_shed[n, t])))
                model.xi_shed[n, t].set_value(0.0)
    for t in model.T:
        model.lam_sys[t].set_value(
            clipped_price(sum(lam_ref[n, t] for n in model.N) / node_count)
        )
    for i, n in model.IN:
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
        rival_power_mw_by_unit=model._rival_power_mw_by_unit,
        rival_energy_mwh_by_unit=model._rival_energy_mwh_by_unit,
        power_by_node={n: value(model.X_power[n]) for n in model.N},
        energy_by_node={n: value(model.X_energy[n]) for n in model.N},
    )
    _initialize_from_quadratic_llp(model, lp_data, model._quad_demand)


def compute_reference_settlement(model: pyo.ConcreteModel) -> dict[str, object]:
    """Settle the MPEC solution against standalone lower-level reference prices.

    In demand-curve mode the QP has unique prices, read analytically off the
    demand curve at cleared curtailment. In fixed-demand mode there is no
    curtailment variable.
    """

    fixed_data = fixed_storage_data_from_solution(model)
    quad: QuadraticDemandCurve = model._quad_demand
    if model._use_demand_curve:
        reference = build_quadratic_primal_model(
            fixed_data,
            quad,
            storage_degradation_eur_per_mwh=model._storage_degradation_eur_per_mwh,
            dispatch_regularization_eur_per_mw2h=model._dispatch_regularization_eur_per_mw2h,
        )
        reference_problem = "QP"
    else:
        reference = build_fixed_demand_primal_model(
            fixed_data,
            storage_degradation_eur_per_mwh=model._storage_degradation_eur_per_mwh,
            dispatch_regularization_eur_per_mw2h=model._dispatch_regularization_eur_per_mw2h,
        )
        reference_problem = "QP"
    solver_label = "ipopt"
    print(f"Solving reference settlement {reference_problem} with ipopt...")
    results = get_ipopt_solver(
        {"tol": model._solver_tol, "acceptable_tol": model._solver_tol}
    ).solve(reference, tee=False)
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
        "--price-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_PRICE_BOUND_EUR_PER_MWH,
        help="Absolute finite bound for lambda and lambda_system (default: 500).",
    )
    parser.add_argument(
        "--dual-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_DUAL_BOUND_EUR_PER_MWH,
        help="Absolute finite bound for non-price lower-level duals (default: 10000).",
    )
    parser.add_argument(
        "--dispatch-regularization",
        type=float,
        default=DEFAULT_DISPATCH_REGULARIZATION_EUR_PER_MW2H,
        help="Neutral lower-level quadratic tie-break coefficient in EUR/(MW^2 h).",
    )
    parser.add_argument(
        "--demand-model",
        choices=["fixed", "quadratic"],
        default="fixed",
        help="Lower-level demand representation. The maintained base model uses fixed demand.",
    )
    parser.add_argument("--solver-tol", type=float, default=DEFAULT_SOLVER_TOL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-export", action="store_true", help="Do not write detailed CSV/JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dispatch_regularization < 0.0:
        raise SystemExit("--dispatch-regularization must be non-negative.")
    if args.solver_tol <= 0.0:
        raise SystemExit("--solver-tol must be positive.")
    data = load_market_data(args.data)
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR

    quad_demand = default_quadratic_demand_curve()
    use_demand_curve = args.demand_model == "quadratic"
    if use_demand_curve:
        print(
            "Quadratic demand curve: "
            f"marginal WTP = {quad_demand.alpha:,.2f} + {quad_demand.beta:,.2f} * curtailed_share EUR/MWh"
        )
    else:
        print("Fixed demand mode: no load-shedding primal or dual variables; no VOLL/load-shed pricing.")
    print(
        "Lower-level storage degradation: "
        f"{DEFAULT_DEGRADATION_EUR_PER_MWH:,.2f} EUR/MWh-cycle "
        f"({0.5 * DEFAULT_DEGRADATION_EUR_PER_MWH:,.2f} EUR/MWh on each charge/discharge leg)"
    )
    print(
        "Lower-level dispatch regularization: "
        f"{args.dispatch_regularization:.3e} EUR/(MW^2 h)"
    )

    model = build_single_investor_mpec(
        data,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        fixed_power_mw=args.fixed_power_mw,
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
        existing_power_mw=args.existing_power_mw,
        existing_ratio_hours=args.existing_ratio_hours,
        dispatch_regularization_eur_per_mw2h=args.dispatch_regularization,
        quad_demand=quad_demand,
        use_demand_curve=use_demand_curve,
        solver_tol=args.solver_tol,
    )
    initialize_from_reference_dispatch(model, data, args.initial_ratio_hours)

    solver = get_ipopt_solver(
        {
            "max_cpu_time": args.max_cpu_time,
            "tol": args.solver_tol,
            "acceptable_tol": args.solver_tol,
        }
    )
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
