"""One investor's capacity best response, as a mathematical program with
equilibrium constraints.

The investor chooses nodal power ``X_power[n]`` and energy ``X_energy[n]``
while every rival capacity is fixed.  Below it sits the ISO's convex market
QP of `iso_market`, written out here in primal and dual form so that the
investor optimises *through* the price formation rather than against a fixed
price.

Lower-level optimality is imposed in one of three ways:

``strong-duality``
    the exact condition ``primal objective == dual objective``.  Exact, but a
    single tight equality that IPOPT finds hard from a cold start.

``relaxed-strong-duality``
    the aggregate relaxation ``primal objective - dual objective <= epsilon``.
    Given primal and dual feasibility the gap equals the sum of all
    complementarity products and cannot be negative (weak duality), so this
    bounds the total complementarity slack by one number in EUR/day instead of
    bounding every product separately.  ``epsilon = 0`` is strong duality.

``relaxed-kkt`` (default)
    every complementarity product bounded by a small ``epsilon`` (Scholtes
    relaxation).  A smooth NLP, and the products are reported afterwards so
    the slack that was actually used is visible.

All are built from the same list of complementarity products, so the
formulations cannot silently drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import pyomo.environ as pyo

from investors import InvestorConfig, daily_investment_cost
from market_data import MarketData


LOWER_LEVEL_FORMULATIONS = ("relaxed-kkt", "strong-duality", "relaxed-strong-duality")
# Formulations in which ``complementarity_epsilon`` is an active relaxation.
RELAXED_FORMULATIONS = ("relaxed-kkt", "relaxed-strong-duality")
DEFAULT_COMPLEMENTARITY_EPSILON = 1.0e-6
# Assumed for a rival whose degradation the caller does not state.
DEFAULT_DEGRADATION_EUR_PER_MWH = 15.0


@dataclass(frozen=True)
class MpecContext:
    """Everything about a built MPEC that is not a Pyomo component."""

    data: MarketData
    investor: InvestorConfig
    rival_ids: tuple[str, ...]
    rival_power: dict[str, dict[str, float]]
    rival_energy: dict[str, dict[str, float]]
    degradation: dict[str, float]
    lower_level: str
    complementarity_epsilon: float
    price_bound: float
    dual_bound: float
    product_components: tuple[str, ...] = ()

    @property
    def active_id(self) -> str:
        return self.investor.investor_id


def context(model: pyo.ConcreteModel) -> MpecContext:
    return model._mpec


def build_capacity_mpec(
    data: MarketData,
    *,
    investor: InvestorConfig,
    rival_power: Mapping[str, Mapping[str, float]] | None = None,
    rival_energy: Mapping[str, Mapping[str, float]] | None = None,
    lower_level: str = "relaxed-kkt",
    complementarity_epsilon: float = DEFAULT_COMPLEMENTARITY_EPSILON,
    rival_degradation: Mapping[str, float] | None = None,
    node_connection_limit: Mapping[str, float] | None = None,
    initial_power_mw: float = 0.0,
    initial_ratio_hours: float = 2.0,
    price_bound: float = 500.0,
    dual_bound: float = 10_000.0,
    sparse_capacity_tol: float = 1e-8,
    proximal_power: Mapping[str, float] | None = None,
    proximal_energy: Mapping[str, float] | None = None,
    proximal_penalty: float = 0.0,
    proximal_energy_scale: float = 2.0,
) -> pyo.ConcreteModel:
    """Build the capacity MPEC for one investor against fixed rivals."""

    if lower_level not in LOWER_LEVEL_FORMULATIONS:
        raise ValueError(f"Unknown lower-level formulation: {lower_level}")
    if complementarity_epsilon < 0.0:
        raise ValueError("complementarity_epsilon must be non-negative.")
    if price_bound <= 0.0 or dual_bound <= 0.0:
        raise ValueError("Price and dual bounds must be positive.")
    if investor.wacc < 0.0 or investor.lifetime_years <= 0:
        raise ValueError("Invalid investor financing parameters.")
    if (
        investor.quadratic_cost_power_eur_per_mw2 < 0.0
        or investor.quadratic_cost_energy_eur_per_mwh2 < 0.0
        or investor.quadratic_cost_power_per_node_eur_per_mw2 < 0.0
        or investor.quadratic_cost_energy_per_node_eur_per_mwh2 < 0.0
    ):
        raise ValueError("Quadratic investment-cost coefficients cannot be negative.")
    nodal_costs = (
        investor.quadratic_cost_power_by_node_eur_per_mw2,
        investor.quadratic_cost_energy_by_node_eur_per_mwh2,
    )
    if set().union(*(set(values) for values in nodal_costs)) - set(data.nodes):
        raise ValueError("Node-specific quadratic costs contain an unknown node.")
    if any(value < 0.0 for values in nodal_costs for value in values.values()):
        raise ValueError("Node-specific quadratic costs cannot be negative.")
    if not 0.0 <= investor.ratio_min <= investor.ratio_max:
        raise ValueError("Invalid energy-to-power ratio bounds.")
    if proximal_penalty < 0.0 or proximal_energy_scale <= 0.0:
        raise ValueError("Invalid proximal regularizer settings.")
    if node_connection_limit is not None:
        missing_limits = set(data.nodes) - set(node_connection_limit)
        if missing_limits:
            raise ValueError(f"Missing node connection limits: {sorted(missing_limits)}")
        if not all(float(node_connection_limit[n]) > 0.0 for n in data.nodes):
            raise ValueError("Node connection limits must be positive.")
    unknown_owned = set(investor.owned_generation_shares) - set(data.generators)
    if unknown_owned:
        raise ValueError(f"Unknown owned generators: {sorted(unknown_owned)}")
    if any(not 0.0 <= share <= 1.0 for share in investor.owned_generation_shares.values()):
        raise ValueError("Generator ownership shares must be in [0, 1].")

    active = investor.investor_id
    rivals, rival_p, rival_e = _normalise_rivals(data, active, rival_power, rival_energy)

    # Rivals with no capacity at a node contribute nothing but variables, so
    # they are left out of the storage index entirely.
    storage_pairs = [(active, n) for n in data.nodes]
    storage_pairs += [
        (i, n) for i in rivals for n in data.nodes if rival_p[i][n] > sparse_capacity_tol
    ]
    storage_at_node = {n: [i for i, node in storage_pairs if node == n] for n in data.nodes}
    degradation = {active: investor.degradation_eur_per_mwh}
    degradation.update(
        {i: float((rival_degradation or {}).get(i, DEFAULT_DEGRADATION_EUR_PER_MWH)) for i in rivals}
    )

    nodes_of_generator = data.nodes_by_generator()
    generation_pairs = [
        (g, t)
        for g in data.generators
        for t in data.times
        if data.generation_capacity[g, t] > 1e-8
    ]
    generators_at_node_time = {
        (n, t): [g for g in data.generators_at_node.get(n, []) if (g, t) in generation_pairs]
        for n in data.nodes
        for t in data.times
    }
    eta = data.eta
    last_t = data.last_time

    m = pyo.ConcreteModel(name=f"Capacity MPEC [{active}, {lower_level}]")
    # Published first: the constraint rules below read rival capacities from it.
    m._mpec = MpecContext(
        data=data,
        investor=investor,
        rival_ids=tuple(rivals),
        rival_power=rival_p,
        rival_energy=rival_e,
        degradation=degradation,
        lower_level=lower_level,
        complementarity_epsilon=(
            complementarity_epsilon if lower_level in RELAXED_FORMULATIONS else 0.0
        ),
        price_bound=price_bound,
        dual_bound=dual_bound,
    )
    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.GT = pyo.Set(dimen=2, initialize=generation_pairs, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.IN = pyo.Set(dimen=2, initialize=storage_pairs, ordered=True)

    # ---------------------------------------------------------------- upper
    initial_ratio = investor.clip_duration(initial_ratio_hours)
    m.X_power = pyo.Var(m.N, domain=pyo.NonNegativeReals, initialize=max(0.0, initial_power_mw))
    m.X_energy = pyo.Var(
        m.N,
        domain=pyo.NonNegativeReals,
        initialize=initial_ratio * max(0.0, initial_power_mw),
    )
    m.energy_ratio_min = pyo.Constraint(
        m.N, rule=lambda mm, n: mm.X_energy[n] >= investor.ratio_min * mm.X_power[n]
    )
    m.energy_ratio_max = pyo.Constraint(
        m.N, rule=lambda mm, n: mm.X_energy[n] <= investor.ratio_max * mm.X_power[n]
    )
    # An infinite limit is how the shared cap is switched off: the constraint
    # is never built, so the investors' feasible regions stop being coupled and
    # the GNEP degeneracy of a shared hard constraint disappears with it.
    if node_connection_limit is not None:

        def _connection_limit(mm: pyo.ConcreteModel, n: str):
            limit = float(node_connection_limit[n])
            if not math.isfinite(limit):
                return pyo.Constraint.Skip
            return mm.X_power[n] + sum(rival_p[i][n] for i in rivals) <= limit

        m.node_connection_limit = pyo.Constraint(m.N, rule=_connection_limit)

    # ------------------------------------------------ lower level: primal
    m.P_gen = pyo.Var(m.GT, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_charge = pyo.Var(m.IN, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.P_discharge = pyo.Var(m.IN, m.T, domain=pyo.NonNegativeReals, initialize=0.0)
    m.SOC = pyo.Var(m.IN, m.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    m.NetInjection = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)
    m.DemandAdjustment = pyo.Var(m.N, m.T, domain=pyo.Reals, initialize=0.0)
    for _n in m.N:
        for _t in m.T:
            if not data.demand_is_adjustable(_n, _t):
                m.DemandAdjustment[_n, _t].fix(0.0)

    # -------------------------------------------------- lower level: duals
    # lam is the LMP; the remaining multipliers are signed by the direction of
    # the inequality they price.
    m.lam = pyo.Var(m.N, m.T, bounds=(-price_bound, price_bound), initialize=80.0)
    m.lam_sys = pyo.Var(m.T, bounds=(-price_bound, price_bound), initialize=80.0)
    m.nu_gen = pyo.Var(m.GT, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_up = pyo.Var(m.L, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.mu_dn = pyo.Var(m.L, m.T, bounds=(0.0, dual_bound), initialize=0.0)
    m.rho_ch = pyo.Var(m.IN, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.sig_dis = pyo.Var(m.IN, m.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.gam = pyo.Var(m.IN, m.T, bounds=(-dual_bound, dual_bound), initialize=0.0)
    m.del_soc = pyo.Var(m.IN, m.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0)
    m.rho_per = pyo.Var(m.IN, bounds=(-dual_bound, dual_bound), initialize=0.0)

    # ---------------------------------------------- lower level: primal feasibility
    m.nodal_balance = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda mm, n, t: sum(mm.P_gen[g, t] for g in generators_at_node_time[n, t])
        + sum(mm.P_discharge[i, n, t] - mm.P_charge[i, n, t] for i in storage_at_node[n])
        + mm.DemandAdjustment[n, t]
        - data.demand_el[n, t]
        == mm.NetInjection[n, t],
    )
    m.system_balance = pyo.Constraint(
        m.T, rule=lambda mm, t: sum(mm.NetInjection[n, t] for n in mm.N) == 0.0
    )
    m.generation_capacity_bound = pyo.Constraint(
        m.GT, rule=lambda mm, g, t: mm.P_gen[g, t] <= data.generation_capacity[g, t]
    )
    m.line_upper_bound = pyo.Constraint(
        m.L, m.T, rule=lambda mm, l, t: _flow(mm, data, l, t) <= data.line_limit[l]
    )
    m.line_lower_bound = pyo.Constraint(
        m.L, m.T, rule=lambda mm, l, t: _flow(mm, data, l, t) >= -data.line_limit[l]
    )
    m.charge_power_bound = pyo.Constraint(
        m.IN, m.T, rule=lambda mm, i, n, t: mm.P_charge[i, n, t] <= unit_power(mm, i, n)
    )
    m.discharge_power_bound = pyo.Constraint(
        m.IN, m.T, rule=lambda mm, i, n, t: mm.P_discharge[i, n, t] <= unit_power(mm, i, n)
    )
    m.soc_transition = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda mm, i, n, t: mm.SOC[i, n, t]
        == mm.SOC[i, n, t - 1] + eta * mm.P_charge[i, n, t] - mm.P_discharge[i, n, t] / eta,
    )
    m.soc_capacity_bound = pyo.Constraint(
        m.IN, m.T_SOC, rule=lambda mm, i, n, tau: mm.SOC[i, n, tau] <= unit_energy(mm, i, n)
    )
    m.soc_periodicity = pyo.Constraint(
        m.IN, rule=lambda mm, i, n: mm.SOC[i, n, 0] == mm.SOC[i, n, last_t]
    )

    # ------------------------------------------------ lower level: dual feasibility
    # Each is "reduced cost >= 0" for the matching primal variable.
    m.gen_stationarity = pyo.Constraint(
        m.GT,
        rule=lambda mm, g, t: sum(mm.lam[n, t] for n in nodes_of_generator[g])
        + mm.nu_gen[g, t]
        <= data.offer(g),
    )
    m.charge_stationarity = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda mm, i, n, t: -mm.lam[n, t] + mm.rho_ch[i, n, t] - eta * mm.gam[i, n, t]
        <= 0.5 * degradation[i],
    )
    m.discharge_stationarity = pyo.Constraint(
        m.IN,
        m.T,
        rule=lambda mm, i, n, t: mm.lam[n, t] + mm.sig_dis[i, n, t] + mm.gam[i, n, t] / eta
        <= 0.5 * degradation[i],
    )
    m.netinjection_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda mm, n, t: -mm.lam[n, t]
        + mm.lam_sys[t]
        + sum(data.ptdf[l, n] * (mm.mu_up[l, t] + mm.mu_dn[l, t]) for l in mm.L)
        == 0.0,
    )
    # Only where the term is free to move.  Where it is held at zero this
    # condition would read ``rho * 0 == lam`` and force the LMP to zero; the
    # price there is pinned by net_injection_stationarity instead.
    m.demand_adjustment_stationarity = pyo.Constraint(
        m.N,
        m.T,
        rule=lambda mm, n, t: (
            data.demand_adjustment_penalty_eur_per_mw2 * mm.DemandAdjustment[n, t]
            == mm.lam[n, t]
        )
        if data.demand_is_adjustable(n, t)
        else pyo.Constraint.Skip,
    )
    m.soc_stationarity = pyo.Constraint(
        m.IN,
        m.T_SOC,
        rule=lambda mm, i, n, tau: -_soc_reduced_cost(mm, i, n, tau, last_t) <= 0.0,
    )

    # ------------------------------------------------------ objective values
    m.lower_level_degradation = pyo.Expression(
        expr=sum(
            0.5 * degradation[i] * (m.P_charge[i, n, t] + m.P_discharge[i, n, t])
            for i, n in m.IN
            for t in m.T
        )
    )
    m.primal_objective = pyo.Expression(
        expr=sum(data.offer(g) * m.P_gen[g, t] for g, t in m.GT)
        + m.lower_level_degradation
        + 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(m.DemandAdjustment[n, t] ** 2 for n in m.N for t in m.T)
    )
    m.dual_objective = pyo.Expression(
        expr=sum(data.demand_el[n, t] * m.lam[n, t] for n in m.N for t in m.T)
        + sum(data.generation_capacity[g, t] * m.nu_gen[g, t] for g, t in m.GT)
        + sum(data.line_limit[l] * (m.mu_up[l, t] - m.mu_dn[l, t]) for l in m.L for t in m.T)
        + sum(
            unit_power(m, i, n) * (m.rho_ch[i, n, t] + m.sig_dis[i, n, t])
            for i, n in m.IN
            for t in m.T
        )
        + sum(unit_energy(m, i, n) * m.del_soc[i, n, tau] for i, n in m.IN for tau in m.T_SOC)
        - 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(m.DemandAdjustment[n, t] ** 2 for n in m.N for t in m.T)
    )

    # ------------------------------------------------- lower-level optimality
    product_components = _add_complementarity_products(m, data, degradation, nodes_of_generator)
    m.strong_duality = pyo.Constraint(expr=m.primal_objective == m.dual_objective)
    if lower_level == "relaxed-kkt":
        m.strong_duality.deactivate()
        for name in product_components:
            product = getattr(m, name)
            m.add_component(
                name.removesuffix("_product"),
                pyo.Constraint(
                    product.index_set(),
                    rule=lambda mm, *key, product=product: product[key]
                    <= complementarity_epsilon,
                ),
            )
    elif lower_level == "relaxed-strong-duality":
        # Weak duality already gives primal >= dual at every primal- and
        # dual-feasible point, so only the upper side of the gap is imposed:
        # the sum of all complementarity products is at most epsilon EUR/day.
        m.strong_duality.deactivate()
        m.relaxed_strong_duality = pyo.Constraint(
            expr=m.primal_objective - m.dual_objective <= complementarity_epsilon
        )

    # ------------------------------------------------------- investor profit
    m.spot_revenue = pyo.Expression(
        expr=sum(
            m.lam[n, t] * (m.P_discharge[active, n, t] - m.P_charge[active, n, t])
            for n in m.N
            for t in m.T
        )
    )
    m.generation_rent = pyo.Expression(
        expr=sum(
            share
            * (m.lam[nodes_of_generator[g][0], t] - data.generation_cost[g])
            * m.P_gen[g, t]
            for g, share in investor.owned_generation_shares.items()
            for t in m.T
            if share and (g, t) in m.GT
        )
    )
    m.active_degradation = pyo.Expression(
        expr=0.5
        * investor.degradation_eur_per_mwh
        * sum(m.P_charge[active, n, t] + m.P_discharge[active, n, t] for n in m.N for t in m.T)
    )
    m.daily_capex = pyo.Expression(
        expr=daily_investment_cost(
            investor,
            (m.X_power[n] for n in m.N),
            (m.X_energy[n] for n in m.N),
            data.nodes,
        )
    )
    m.unregularized_profit = pyo.Expression(
        expr=m.spot_revenue + m.generation_rent - m.active_degradation - m.daily_capex
    )

    # Moving proximal term: a purely algorithmic stabiliser for the Jacobi
    # iteration, centred on the previous sweep and always reported apart from
    # the economics.  Energy deviations are rescaled to MW-equivalents so MW
    # and MWh enter at comparable magnitudes.
    if proximal_penalty > 0.0:
        if proximal_power is None or proximal_energy is None:
            raise ValueError("A positive proximal penalty requires both capacity centres.")
        regularizer = 0.5 * proximal_penalty * sum(
            (m.X_power[n] - float(proximal_power[n])) ** 2
            + ((m.X_energy[n] - float(proximal_energy[n])) / proximal_energy_scale) ** 2
            for n in m.N
        )
    else:
        regularizer = 0.0
    m.regularizer = pyo.Expression(expr=regularizer)
    m.profit = pyo.Expression(expr=m.unregularized_profit - m.regularizer)
    m.objective = pyo.Objective(expr=m.profit, sense=pyo.maximize)

    m._mpec = replace(m._mpec, product_components=product_components)
    return m


# --------------------------------------------------------------------------
# Capacity accessors: the active investor's capacity is a variable, a rival's
# is a number, and every constraint below is written the same way regardless.


def unit_power(model: pyo.ConcreteModel, unit: str, node: str):
    ctx = context(model)
    return model.X_power[node] if unit == ctx.active_id else ctx.rival_power[unit][node]


def unit_energy(model: pyo.ConcreteModel, unit: str, node: str):
    ctx = context(model)
    return model.X_energy[node] if unit == ctx.active_id else ctx.rival_energy[unit][node]


def _flow(model: pyo.ConcreteModel, data: MarketData, line: str, time: int):
    return sum(data.ptdf[line, node] * model.NetInjection[node, time] for node in model.N)


# --------------------------------------------------------------------------
# Reduced costs.  Each is the slack in one dual-feasibility constraint, so the
# complementarity product for a primal variable is simply value * reduced cost.


def _gen_reduced_cost(model, data, nodes_of_generator, generator, time):
    return (
        data.offer(generator)
        - sum(model.lam[node, time] for node in nodes_of_generator[generator])
        - model.nu_gen[generator, time]
    )


def _charge_reduced_cost(model, eta, degradation, unit, node, time):
    return (
        0.5 * degradation[unit]
        + model.lam[node, time]
        - model.rho_ch[unit, node, time]
        + eta * model.gam[unit, node, time]
    )


def _discharge_reduced_cost(model, eta, degradation, unit, node, time):
    return (
        0.5 * degradation[unit]
        - model.lam[node, time]
        - model.sig_dis[unit, node, time]
        - model.gam[unit, node, time] / eta
    )


def _soc_reduced_cost(model, unit, node, soc_time, last_t):
    """Slack of the SOC stationarity condition, as a *cost* (so >= 0)."""

    stationarity = model.del_soc[unit, node, soc_time]
    if soc_time in model.T:
        stationarity += model.gam[unit, node, soc_time]
    if soc_time + 1 in model.T:
        stationarity -= model.gam[unit, node, soc_time + 1]
    if soc_time == 0:
        stationarity += model.rho_per[unit, node]
    if soc_time == last_t:
        stationarity -= model.rho_per[unit, node]
    return -stationarity


def _add_complementarity_products(
    model: pyo.ConcreteModel,
    data: MarketData,
    degradation: dict[str, float],
    nodes_of_generator: dict[str, list[str]],
) -> tuple[str, ...]:
    """Attach every ``slack * multiplier`` product as a named Expression.

    They are the KKT complementarity conditions of the lower level.  The
    relaxed formulation constrains them; both formulations report them.
    """

    eta = data.eta
    last_t = data.last_time
    specs = {
        "generation_lower": (
            (model.GT,),
            lambda m, g, t: m.P_gen[g, t]
            * _gen_reduced_cost(m, data, nodes_of_generator, g, t),
        ),
        "generation_upper": (
            (model.GT,),
            lambda m, g, t: (data.generation_capacity[g, t] - m.P_gen[g, t])
            * (-m.nu_gen[g, t]),
        ),
        "line_upper": (
            (model.L, model.T),
            lambda m, l, t: (data.line_limit[l] - _flow(m, data, l, t)) * (-m.mu_up[l, t]),
        ),
        "line_lower": (
            (model.L, model.T),
            lambda m, l, t: (_flow(m, data, l, t) + data.line_limit[l]) * m.mu_dn[l, t],
        ),
        "charge_lower": (
            (model.IN, model.T),
            lambda m, i, n, t: m.P_charge[i, n, t]
            * _charge_reduced_cost(m, eta, degradation, i, n, t),
        ),
        "charge_upper": (
            (model.IN, model.T),
            lambda m, i, n, t: (unit_power(m, i, n) - m.P_charge[i, n, t])
            * (-m.rho_ch[i, n, t]),
        ),
        "discharge_lower": (
            (model.IN, model.T),
            lambda m, i, n, t: m.P_discharge[i, n, t]
            * _discharge_reduced_cost(m, eta, degradation, i, n, t),
        ),
        "discharge_upper": (
            (model.IN, model.T),
            lambda m, i, n, t: (unit_power(m, i, n) - m.P_discharge[i, n, t])
            * (-m.sig_dis[i, n, t]),
        ),
        "soc_lower": (
            (model.IN, model.T_SOC),
            lambda m, i, n, tau: m.SOC[i, n, tau] * _soc_reduced_cost(m, i, n, tau, last_t),
        ),
        "soc_upper": (
            (model.IN, model.T_SOC),
            lambda m, i, n, tau: (unit_energy(m, i, n) - m.SOC[i, n, tau])
            * (-m.del_soc[i, n, tau]),
        ),
    }
    names = []
    for name, (index_sets, rule) in specs.items():
        component = f"complementarity_{name}_product"
        model.add_component(component, pyo.Expression(*index_sets, rule=rule))
        names.append(component)
    return tuple(names)


def complementarity_diagnostics(model: pyo.ConcreteModel) -> dict[str, float | int]:
    """Report how badly a solved MPEC misses exact lower-level optimality.

    ``maximum_upper_bound_violation`` is the excess over the relaxation the
    formulation actually allows (zero for strong duality), and
    ``maximum_nonnegativity_violation`` is how negative any product went.
    """

    ctx = context(model)
    products = [
        float(pyo.value(component[index]))
        for name in ctx.product_components
        for component in (getattr(model, name),)
        for index in component
    ]
    epsilon = ctx.complementarity_epsilon
    minimum = min(products, default=0.0)
    maximum = max(products, default=0.0)
    gap = float(pyo.value(model.primal_objective - model.dual_objective))
    return {
        "count": len(products),
        "epsilon": epsilon,
        "minimum_product": minimum,
        "maximum_product": maximum,
        "maximum_upper_bound_violation": max(0.0, maximum - epsilon),
        "maximum_nonnegativity_violation": max(0.0, -minimum),
        "primal_dual_gap_eur_per_day": gap,
        "absolute_primal_dual_gap_eur_per_day": abs(gap),
    }


def artificial_bound_diagnostics(model: pyo.ConcreteModel) -> dict[str, float]:
    """Measure use of finite bounds added solely to contain the NLP search.

    Natural sign restrictions on inequality multipliers are excluded. A ratio
    near one means an artificial price or dual cap may be shaping the result.
    """

    ctx = context(model)

    def largest_absolute(component) -> float:
        values = [abs(float(pyo.value(component[index]))) for index in component]
        return max(values, default=0.0)

    price_use = max(largest_absolute(model.lam), largest_absolute(model.lam_sys))
    dual_use = max(
        largest_absolute(component)
        for component in (
            model.nu_gen,
            model.mu_up,
            model.mu_dn,
            model.rho_ch,
            model.sig_dis,
            model.gam,
            model.del_soc,
            model.rho_per,
        )
    )
    price_ratio = price_use / ctx.price_bound
    dual_ratio = dual_use / ctx.dual_bound
    return {
        "maximum_price_bound_utilization": price_ratio,
        "maximum_dual_bound_utilization": dual_ratio,
        "maximum_artificial_bound_utilization": max(price_ratio, dual_ratio),
    }


def _normalise_rivals(
    data: MarketData,
    active_id: str,
    rival_power: Mapping[str, Mapping[str, float]] | None,
    rival_energy: Mapping[str, Mapping[str, float]] | None,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    power = rival_power or {}
    energy = rival_energy or {}
    if set(power) != set(energy):
        raise ValueError("Rival power and energy identifiers must match.")
    if active_id in power:
        raise ValueError(f"Active investor {active_id} cannot also be a rival.")
    ids = list(power)
    normal_power = {i: {n: float(power[i].get(n, 0.0)) for n in data.nodes} for i in ids}
    normal_energy = {i: {n: float(energy[i].get(n, 0.0)) for n in data.nodes} for i in ids}
    for i in ids:
        for n in data.nodes:
            if normal_power[i][n] < 0.0 or normal_energy[i][n] < 0.0:
                raise ValueError(f"Negative rival capacity for {i} at {n}.")
            if normal_power[i][n] == 0.0 and normal_energy[i][n] > 1e-9:
                raise ValueError(f"Rival {i} has energy but no power at {n}.")
    return ids, normal_power, normal_energy
