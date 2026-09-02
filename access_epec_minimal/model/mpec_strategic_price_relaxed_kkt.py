"""Price-strategic BESS MPEC with full availability and relaxed LLP KKT.

The active investor chooses nodal MW/MWh capacity and hourly charging and
discharging prices. There are no strategic quantity variables. For every
storage unit, the ISO may use the complete installed inverter in either
direction. The maintained default uses the physical shared-inverter constraint

    P_charge + P_discharge <= X_power.

An explicit experiment mode retains the older pair of directional bounds

    P_charge <= X_power,  P_discharge <= X_power.

SOC transition, SOC capacity, and daily periodicity remain lower-level market
constraints. Lower-level optimality is represented by primal feasibility,
dual feasibility/stationarity, and Scholtes-relaxed complementarity products.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pyomo.environ as pyo

from mpec_strategic_operation import build_model as build_price_model
from primal_market_clearing_model import (
    MarketData,
    build_primal_market_clearing_model,
    effective_generation_offer,
)


DEFAULT_COMPLEMENTARITY_EPSILON = 1.0e-3
DEFAULT_PROXIMAL_PRICE_SCALE = 1.0


def build_model(
    data: MarketData,
    *,
    complementarity_epsilon: float = DEFAULT_COMPLEMENTARITY_EPSILON,
    proximal_bid_charge: Mapping[tuple[str, int], float] | None = None,
    proximal_offer_discharge: Mapping[tuple[str, int], float] | None = None,
    proximal_bid_penalty: float | None = None,
    proximal_price_scale: float = DEFAULT_PROXIMAL_PRICE_SCALE,
    inverter_limit: str = "shared",
    **kwargs: object,
) -> pyo.ConcreteModel:
    """Build one price-only, full-availability relaxed-KKT best response."""

    epsilon = float(complementarity_epsilon)
    price_scale = float(proximal_price_scale)
    capacity_proximal_coefficient = float(kwargs.get("proximal_penalty", 0.0))
    bid_proximal_coefficient = float(
        capacity_proximal_coefficient
        if proximal_bid_penalty is None
        else proximal_bid_penalty
    )
    if epsilon < 0.0:
        raise ValueError("complementarity_epsilon must be non-negative.")
    if price_scale <= 0.0:
        raise ValueError("proximal_price_scale must be positive.")
    if bid_proximal_coefficient < 0.0:
        raise ValueError("proximal_bid_penalty must be non-negative.")
    if inverter_limit not in {"shared", "separate"}:
        raise ValueError("inverter_limit must be 'shared' or 'separate'.")

    model = build_price_model(data, **kwargs)
    model.name = f"Strategic-price relaxed-KKT MPEC [{model._active_id}]"
    model.strong_duality.deactivate()

    active = model._active_id
    eta = data.eta
    last_t = max(data.times)
    dual_bound = float(kwargs.get("dual_bound", 10_000.0))

    def unit_power(unit: str, node: str):
        return (
            model.X_power[node]
            if unit == active
            else model._rival_power[unit][node]
        )

    def unit_energy(unit: str, node: str):
        return (
            model.X_energy[node]
            if unit == active
            else model._rival_energy[unit][node]
        )

    def charge_bid(unit: str, node: str, time: int):
        return (
            model.BidCharge[node, time]
            if unit == active
            else model._rival_bid_charge[unit, node, int(time)]
        )

    def discharge_offer(unit: str, node: str, time: int):
        return (
            model.OfferDischarge[node, time]
            if unit == active
            else model._rival_offer_discharge[unit, node, int(time)]
        )

    if inverter_limit == "shared":
        # Replace the two directional limits by one physical inverter limit.
        # Either direction can use the full installed MW, but not both at full
        # power simultaneously.
        model.del_component(model.charge_power_bound)
        model.del_component(model.discharge_power_bound)
        model.shared_inverter_bound = pyo.Constraint(
            model.IN,
            model.T,
            rule=lambda m, i, n, t: m.P_charge[i, n, t]
            + m.P_discharge[i, n, t]
            <= unit_power(i, n),
        )
        model.kappa_power = pyo.Var(
            model.IN,
            model.T,
            bounds=(-dual_bound, 0.0),
            initialize=0.0,
        )

        # Directional-bound duals are not part of this nonredundant LLP.
        model.del_component(model.charge_stationarity)
        model.del_component(model.discharge_stationarity)
        model.charge_stationarity = pyo.Constraint(
            model.IN,
            model.T,
            rule=lambda m, i, n, t: -m.lam[n, t]
            + m.kappa_power[i, n, t]
            - eta * m.gam[i, n, t]
            <= -charge_bid(i, n, int(t)),
        )
        model.discharge_stationarity = pyo.Constraint(
            model.IN,
            model.T,
            rule=lambda m, i, n, t: m.lam[n, t]
            + m.kappa_power[i, n, t]
            + m.gam[i, n, t] / eta
            <= discharge_offer(i, n, int(t)),
        )
        model.dual_objective.set_value(
            sum(
                data.demand_el[n, t] * model.lam[n, t]
                for n in model.N
                for t in model.T
            )
            + sum(
                data.generation_capacity[g, t] * model.nu_gen[g, t]
                for g, t in model.GT
            )
            + sum(
                data.line_limit[l]
                * (model.mu_up[l, t] - model.mu_dn[l, t])
                for l in model.L
                for t in model.T
            )
            + sum(
                unit_power(i, n) * model.kappa_power[i, n, t]
                for i, n in model.IN
                for t in model.T
            )
            + sum(
                unit_energy(i, n) * model.del_soc[i, n, tau]
                for i, n in model.IN
                for tau in model.T_SOC
            )
            - 0.5
            * data.demand_adjustment_penalty_eur_per_mw2
            * sum(
                model.DemandAdjustment[n, t] ** 2
                for n in model.N
                for t in model.T
            )
        )
        model.del_component(model.rho_ch)
        model.del_component(model.sig_dis)

    def flow(m: pyo.ConcreteModel, line: str, time: int):
        return sum(
            data.ptdf[line, node] * m.NetInjection[node, time]
            for node in m.N
        )

    def gen_reduced_cost(m: pyo.ConcreteModel, generator: str, time: int):
        return (
            effective_generation_offer(data, generator)
            - sum(m.lam[node, time] for node in m._gen_nodes[generator])
            - m.nu_gen[generator, time]
        )

    def charge_reduced_cost(
        m: pyo.ConcreteModel, unit: str, node: str, time: int
    ):
        power_dual = (
            m.kappa_power[unit, node, time]
            if inverter_limit == "shared"
            else m.rho_ch[unit, node, time]
        )
        return (
            m.lam[node, time]
            - power_dual
            + eta * m.gam[unit, node, time]
            - charge_bid(unit, node, int(time))
        )

    def discharge_reduced_cost(
        m: pyo.ConcreteModel, unit: str, node: str, time: int
    ):
        power_dual = (
            m.kappa_power[unit, node, time]
            if inverter_limit == "shared"
            else m.sig_dis[unit, node, time]
        )
        return (
            discharge_offer(unit, node, int(time))
            - m.lam[node, time]
            - power_dual
            - m.gam[unit, node, time] / eta
        )

    def soc_reduced_cost(
        m: pyo.ConcreteModel, unit: str, node: str, soc_time: int
    ):
        stationarity_lhs = m.del_soc[unit, node, soc_time]
        if soc_time in m.T:
            stationarity_lhs += m.gam[unit, node, soc_time]
        if soc_time + 1 in m.T:
            stationarity_lhs -= m.gam[unit, node, soc_time + 1]
        if soc_time == 0:
            stationarity_lhs += m.rho_per[unit, node]
        if soc_time == last_t:
            stationarity_lhs -= m.rho_per[unit, node]
        return -stationarity_lhs

    product_names: list[str] = []

    def add_relaxed_product(
        name: str,
        index_sets: tuple[pyo.Set, ...],
        product_rule,
    ) -> None:
        product = pyo.Expression(*index_sets, rule=product_rule)
        model.add_component(f"{name}_product", product)
        model.add_component(
            name,
            pyo.Constraint(
                *index_sets,
                rule=lambda m, *key: pyo.inequality(
                    0.0, product[key], epsilon
                ),
            ),
        )
        product_names.append(f"{name}_product")

    add_relaxed_product(
        "relaxed_comp_gen_lower",
        (model.GT,),
        lambda m, g, t: m.P_gen[g, t] * gen_reduced_cost(m, g, t),
    )
    add_relaxed_product(
        "relaxed_comp_charge_lower",
        (model.IN, model.T),
        lambda m, i, n, t: m.P_charge[i, n, t]
        * charge_reduced_cost(m, i, n, t),
    )
    add_relaxed_product(
        "relaxed_comp_discharge_lower",
        (model.IN, model.T),
        lambda m, i, n, t: m.P_discharge[i, n, t]
        * discharge_reduced_cost(m, i, n, t),
    )
    add_relaxed_product(
        "relaxed_comp_soc_lower",
        (model.IN, model.T_SOC),
        lambda m, i, n, tau: m.SOC[i, n, tau]
        * soc_reduced_cost(m, i, n, tau),
    )
    add_relaxed_product(
        "relaxed_comp_gen_upper",
        (model.GT,),
        lambda m, g, t: (
            data.generation_capacity[g, t] - m.P_gen[g, t]
        )
        * (-m.nu_gen[g, t]),
    )
    add_relaxed_product(
        "relaxed_comp_line_upper",
        (model.L, model.T),
        lambda m, l, t: (data.line_limit[l] - flow(m, l, t))
        * (-m.mu_up[l, t]),
    )
    add_relaxed_product(
        "relaxed_comp_line_lower",
        (model.L, model.T),
        lambda m, l, t: (flow(m, l, t) + data.line_limit[l])
        * m.mu_dn[l, t],
    )
    if inverter_limit == "shared":
        add_relaxed_product(
            "relaxed_comp_shared_inverter_upper",
            (model.IN, model.T),
            lambda m, i, n, t: (
                unit_power(i, n)
                - m.P_charge[i, n, t]
                - m.P_discharge[i, n, t]
            )
            * (-m.kappa_power[i, n, t]),
        )
    else:
        add_relaxed_product(
            "relaxed_comp_charge_upper",
            (model.IN, model.T),
            lambda m, i, n, t: (unit_power(i, n) - m.P_charge[i, n, t])
            * (-m.rho_ch[i, n, t]),
        )
        add_relaxed_product(
            "relaxed_comp_discharge_upper",
            (model.IN, model.T),
            lambda m, i, n, t: (unit_power(i, n) - m.P_discharge[i, n, t])
            * (-m.sig_dis[i, n, t]),
        )
    add_relaxed_product(
        "relaxed_comp_soc_upper",
        (model.IN, model.T_SOC),
        lambda m, i, n, tau: (unit_energy(i, n) - m.SOC[i, n, tau])
        * (-m.del_soc[i, n, tau]),
    )

    # Extend the moving quadratic capacity proximal term to both hourly price
    # profiles.  There is no quantity regularizer because accepted quantities
    # are lower-level ISO decisions, not strategic variables.
    if bid_proximal_coefficient > 0.0:
        if proximal_bid_charge is None or proximal_offer_discharge is None:
            raise ValueError(
                "A positive price-strategy proximal penalty requires both price centres."
            )
        model.price_strategy_regularizer = pyo.Expression(
            expr=0.5
            * bid_proximal_coefficient
            * sum(
                (
                    (model.BidCharge[n, t] - float(proximal_bid_charge[n, int(t)]))
                    / price_scale
                )
                ** 2
                + (
                    (
                        model.OfferDischarge[n, t]
                        - float(proximal_offer_discharge[n, int(t)])
                    )
                    / price_scale
                )
                ** 2
                for n in model.N
                for t in model.T
            )
        )
        model.regularizer.set_value(
            model.regularizer.expr + model.price_strategy_regularizer
        )
        model.profit.set_value(model.unregularized_profit - model.regularizer)
        model.objective.set_value(model.profit)
        model._proximal_bid_charge = dict(proximal_bid_charge)
        model._proximal_offer_discharge = dict(proximal_offer_discharge)

    model._lower_level_optimality = "relaxed-kkt"
    model._strategic_price_relaxed_kkt = True
    model._full_quantity_availability = True
    model._complementarity_epsilon = epsilon
    model._proximal_price_scale = price_scale
    model._proximal_bid_penalty = bid_proximal_coefficient
    model._inverter_limit = inverter_limit
    model._relaxed_kkt_product_components = tuple(product_names)
    return model


def initialise_lower_level(
    model: pyo.ConcreteModel,
    data: MarketData,
    generation_offers: Mapping[tuple[str, int], float] | None = None,
) -> None:
    """Seed the MPEC from the exact fixed-capacity, fixed-price ISO LP.

    ``generation_offers`` optionally overrides the per-generator static offer
    with hour-varying submitted offers keyed by ``(generator, hour)``; missing
    keys fall back to the static effective offer.
    """

    active = model._active_id
    units = [active, *model._rival_ids]
    x_power: dict[tuple[str, str], float] = {}
    x_energy: dict[tuple[str, str], float] = {}
    for unit in units:
        for node in data.nodes:
            if unit == active:
                x_power[unit, node] = float(pyo.value(model.X_power[node]))
                x_energy[unit, node] = float(pyo.value(model.X_energy[node]))
            else:
                x_power[unit, node] = model._rival_power[unit][node]
                x_energy[unit, node] = model._rival_energy[unit][node]

    fixed_data = replace(
        data,
        storage_units=units,
        x_power=x_power,
        x_energy=x_energy,
    )
    lower = build_primal_market_clearing_model(
        fixed_data, include_load_shed=False
    )

    def charge_bid(unit: str, node: str, time: int) -> float:
        return (
            float(pyo.value(model.BidCharge[node, time]))
            if unit == active
            else model._rival_bid_charge.get((unit, node, int(time)), 0.0)
        )

    def discharge_offer(unit: str, node: str, time: int) -> float:
        return (
            float(pyo.value(model.OfferDischarge[node, time]))
            if unit == active
            else model._rival_offer_discharge.get(
                (unit, node, int(time)), 0.0
            )
        )

    if model._inverter_limit == "shared":
        lower.del_component(lower.charge_power_bound)
        lower.del_component(lower.discharge_power_bound)
        lower.shared_inverter_bound = pyo.Constraint(
            lower.I,
            lower.N,
            lower.T,
            rule=lambda m, i, n, t: m.P_charge[i, n, t]
            + m.P_discharge[i, n, t]
            <= fixed_data.x_power[i, n],
        )
    offers = generation_offers or {}

    def submitted_offer(generator: str, time: int) -> float:
        return float(
            offers.get(
                (generator, int(time)),
                effective_generation_offer(data, generator),
            )
        )

    lower.objective.set_value(
        sum(
            submitted_offer(g, int(t)) * lower.P_gen[g, t]
            for g in lower.G
            for t in lower.T
        )
        + sum(
            discharge_offer(i, n, int(t)) * lower.P_discharge[i, n, t]
            - charge_bid(i, n, int(t)) * lower.P_charge[i, n, t]
            for i in lower.I
            for n in lower.N
            for t in lower.T
        )
        + 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(
            lower.DemandAdjustment[n, t] ** 2
            for n in lower.N
            for t in lower.T
        )
    )

    candidates = []
    if os.environ.get("IPOPT_EXECUTABLE"):
        candidates.append(Path(os.environ["IPOPT_EXECUTABLE"]))
    if shutil.which("ipopt"):
        candidates.append(Path(shutil.which("ipopt") or ""))
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "idaes" / "bin" / "ipopt.exe")
    executable = next((candidate for candidate in candidates if candidate.is_file()), None)
    solver_kwargs = {"solver_io": "nl"}
    if executable is not None:
        solver_kwargs["executable"] = str(executable)
    solver = pyo.SolverFactory("ipopt", **solver_kwargs)
    solver.options.update(
        {
            "linear_solver": "ma57",
            "tol": 1.0e-8,
            "acceptable_tol": 1.0e-7,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "print_level": 0,
        }
    )
    result = solver.solve(lower, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return

    def set_seed(variable: pyo.Var, raw_value: float) -> None:
        value = float(raw_value)
        if variable.lb is not None:
            value = max(value, float(pyo.value(variable.lb)))
        if variable.ub is not None:
            value = min(value, float(pyo.value(variable.ub)))
        variable.set_value(value)

    for generator, time in model.GT:
        set_seed(model.P_gen[generator, time], lower.P_gen[generator, time].value)
        set_seed(
            model.nu_gen[generator, time],
            lower.dual[lower.generation_capacity_bound[generator, time]],
        )
    for node in model.N:
        for time in model.T:
            set_seed(model.NetInjection[node, time], lower.NetInjection[node, time].value)
            set_seed(
                model.DemandAdjustment[node, time],
                lower.DemandAdjustment[node, time].value,
            )
            set_seed(
                model.lam[node, time],
                lower.dual[lower.nodal_balance[node, time]],
            )
    for time in model.T:
        set_seed(model.lam_sys[time], lower.dual[lower.system_balance[time]])
    for line in model.L:
        for time in model.T:
            set_seed(
                model.mu_up[line, time],
                lower.dual[lower.line_upper_bound[line, time]],
            )
            set_seed(
                model.mu_dn[line, time],
                lower.dual[lower.line_lower_bound[line, time]],
            )
    for unit, node in model.IN:
        for time in model.T:
            key = unit, node, time
            set_seed(model.P_charge[key], lower.P_charge[key].value)
            set_seed(model.P_discharge[key], lower.P_discharge[key].value)
            if model._inverter_limit == "shared":
                set_seed(
                    model.kappa_power[key],
                    lower.dual[lower.shared_inverter_bound[key]],
                )
            else:
                set_seed(
                    model.rho_ch[key],
                    lower.dual[lower.charge_power_bound[key]],
                )
                set_seed(
                    model.sig_dis[key],
                    lower.dual[lower.discharge_power_bound[key]],
                )
            set_seed(model.gam[key], lower.dual[lower.soc_transition[key]])
        for soc_time in model.T_SOC:
            key = unit, node, soc_time
            set_seed(model.SOC[key], lower.SOC[key].value)
            set_seed(
                model.del_soc[key],
                lower.dual[lower.soc_capacity_bound[key]],
            )
        set_seed(
            model.rho_per[unit, node],
            lower.dual[lower.soc_periodicity[unit, node]],
        )
