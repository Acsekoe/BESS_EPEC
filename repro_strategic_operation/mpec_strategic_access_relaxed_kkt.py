"""Co-optimised nodal BESS access bidding with relaxed lower-level KKT.

The active investor chooses, at every node, a requested MW quantity, a
non-negative pay-as-bid willingness to pay in EUR/MW-day, and independent MWh
capacity.  A regulator-fixed slope can turn the flat access bid into a linear
downward-sloping marginal bid.  The ISO lower level jointly awards connection
MW and clears the physical 24-hour market.  Awarded MW, dispatch, SOC, and all
network variables therefore belong to the lower level; energy investment
remains upper-level.

For fixed strategies the ISO problem is a convex quadratic program when the
access slope is positive and a linear program when it is zero.  Its optimality
is embedded with primal feasibility, dual feasibility/stationarity, and one
Scholtes-relaxed product for every complementarity pair.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import pyomo.environ as pyo

from mpec_strong_duality import (
    InvestorConfig,
    _generator_nodes,
    capital_recovery_factor,
)
from primal_market_clearing_model import MarketData


DEFAULT_COMPLEMENTARITY_EPSILON = 1.0e-3
DEFAULT_ACCESS_BID_BOUND = 500.0
DEFAULT_INVESTOR_REQUEST_LIMIT_MW = 200.0


AccessProfile = Mapping[str, Mapping[str, float]]


def _ipopt_executable() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("IPOPT_EXECUTABLE"):
        candidates.append(Path(os.environ["IPOPT_EXECUTABLE"]))
    if shutil.which("ipopt"):
        candidates.append(Path(shutil.which("ipopt") or ""))
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "idaes" / "bin" / "ipopt.exe")
    return next((path for path in candidates if path.is_file()), None)


def _normalise_rival_profile(
    data: MarketData,
    active: str,
    quantity: AccessProfile | None,
    bid: AccessProfile | None,
    energy_capacity: AccessProfile | None,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    raw_quantity = quantity or {}
    raw_bid = bid or {}
    raw_energy = energy_capacity or {}
    if set(raw_quantity) != set(raw_bid) or set(raw_quantity) != set(raw_energy):
        raise ValueError("Rival access quantity, bid, and energy identifiers must match.")
    if active in raw_quantity:
        raise ValueError(f"Active investor {active} cannot also be a rival.")
    rivals = list(raw_quantity)
    normal_quantity = {
        unit: {node: float(raw_quantity[unit].get(node, 0.0)) for node in data.nodes}
        for unit in rivals
    }
    normal_bid = {
        unit: {node: float(raw_bid[unit].get(node, 0.0)) for node in data.nodes}
        for unit in rivals
    }
    normal_energy = {
        unit: {node: float(raw_energy[unit].get(node, 0.0)) for node in data.nodes}
        for unit in rivals
    }
    if any(
        normal_quantity[unit][node] < 0.0
        or normal_bid[unit][node] < 0.0
        or normal_energy[unit][node] < 0.0
        for unit in rivals
        for node in data.nodes
    ):
        raise ValueError("Rival access quantities, bids, and energy must be non-negative.")
    return rivals, normal_quantity, normal_bid, normal_energy


def build_fixed_access_market(
    data: MarketData,
    *,
    access_quantity: AccessProfile,
    access_bid: AccessProfile,
    energy_capacity: AccessProfile,
    degradation: Mapping[str, float],
    node_limit_mw: float,
    access_bid_slope_eur_per_mw2_day: float = 0.0,
) -> pyo.ConcreteModel:
    """Build the exact ISO QP for one fixed access-strategy profile."""

    units = list(access_quantity)
    if not units or set(units) != set(access_bid) or set(units) != set(energy_capacity):
        raise ValueError("Fixed access profiles must contain the same nonempty investor set.")
    if set(units) != set(degradation):
        raise ValueError("Every fixed access investor requires a degradation cost.")
    access_slope = float(access_bid_slope_eur_per_mw2_day)
    if access_slope < 0.0:
        raise ValueError("The access-bid slope cannot be negative.")
    if any(
        float(access_bid[i][n]) + 1.0e-6
        < access_slope * float(access_quantity[i][n])
        for i in units
        for n in data.nodes
    ):
        raise ValueError(
            "Every access bid must cover the declining curve through its "
            "requested quantity: bid >= slope * quantity."
        )

    generation_pairs = [
        (generator, time)
        for generator in data.generators
        for time in data.times
        if data.generation_capacity[generator, time] > 1.0e-8
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
    last_time = max(data.times)

    model = pyo.ConcreteModel(name="Fixed-strategy co-optimised access market")
    model.I = pyo.Set(initialize=units, ordered=True)
    model.N = pyo.Set(initialize=data.nodes, ordered=True)
    model.GT = pyo.Set(dimen=2, initialize=generation_pairs, ordered=True)
    model.L = pyo.Set(initialize=data.lines, ordered=True)
    model.T = pyo.Set(initialize=data.times, ordered=True)
    model.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    model.IN = pyo.Set(
        dimen=2,
        initialize=[(unit, node) for unit in units for node in data.nodes],
        ordered=True,
    )

    model.X_awarded = pyo.Var(model.IN, domain=pyo.NonNegativeReals)
    model.P_gen = pyo.Var(model.GT, domain=pyo.NonNegativeReals)
    model.P_charge = pyo.Var(model.IN, model.T, domain=pyo.NonNegativeReals)
    model.P_discharge = pyo.Var(model.IN, model.T, domain=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.IN, model.T_SOC, domain=pyo.NonNegativeReals)
    model.NetInjection = pyo.Var(model.N, model.T, domain=pyo.Reals)

    model.access_request_bound = pyo.Constraint(
        model.IN,
        rule=lambda m, i, n: m.X_awarded[i, n] <= float(access_quantity[i][n]),
    )
    model.nodal_access_bound = pyo.Constraint(
        model.N,
        rule=lambda m, n: sum(m.X_awarded[i, n] for i in m.I) <= node_limit_mw,
    )
    model.nodal_balance = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: sum(
            m.P_gen[g, t] for g in generators_at_node_time[n, t]
        )
        + sum(m.P_discharge[i, n, t] - m.P_charge[i, n, t] for i in m.I)
        - data.demand_el[n, t]
        == m.NetInjection[n, t],
    )
    model.system_balance = pyo.Constraint(
        model.T,
        rule=lambda m, t: sum(m.NetInjection[n, t] for n in m.N) == 0.0,
    )
    model.generation_capacity_bound = pyo.Constraint(
        model.GT,
        rule=lambda m, g, t: m.P_gen[g, t] <= data.generation_capacity[g, t],
    )
    model.line_upper_bound = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, l, t: sum(
            data.ptdf[l, n] * m.NetInjection[n, t] for n in m.N
        )
        <= data.line_limit[l],
    )
    model.line_lower_bound = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, l, t: sum(
            data.ptdf[l, n] * m.NetInjection[n, t] for n in m.N
        )
        >= -data.line_limit[l],
    )
    model.shared_inverter_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.P_charge[i, n, t]
        + m.P_discharge[i, n, t]
        <= m.X_awarded[i, n],
    )
    model.soc_transition = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.SOC[i, n, t]
        == m.SOC[i, n, t - 1]
        + data.eta * m.P_charge[i, n, t]
        - m.P_discharge[i, n, t] / data.eta,
    )
    model.soc_capacity_bound = pyo.Constraint(
        model.IN,
        model.T_SOC,
        rule=lambda m, i, n, tau: m.SOC[i, n, tau]
        <= float(energy_capacity[i][n]),
    )
    model.soc_periodicity = pyo.Constraint(
        model.IN,
        rule=lambda m, i, n: m.SOC[i, n, 0] == m.SOC[i, n, last_time],
    )
    model.objective = pyo.Objective(
        expr=sum(
            data.generation_cost[g] * model.P_gen[g, t]
            for g, t in model.GT
        )
        + sum(
            0.5
            * float(degradation[i])
            * (model.P_charge[i, n, t] + model.P_discharge[i, n, t])
            for i, n in model.IN
            for t in model.T
        )
        - sum(
            float(access_bid[i][n]) * model.X_awarded[i, n]
            for i, n in model.IN
        )
        + 0.5
        * access_slope
        * sum(model.X_awarded[i, n] ** 2 for i, n in model.IN),
        sense=pyo.minimize,
    )
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model._market_data = data
    model._access_quantity = {
        (i, n): float(access_quantity[i][n]) for i, n in model.IN
    }
    model._access_bid = {(i, n): float(access_bid[i][n]) for i, n in model.IN}
    model._energy_capacity = {
        (i, n): float(energy_capacity[i][n]) for i, n in model.IN
    }
    model._degradation = dict(degradation)
    model._node_limit_mw = float(node_limit_mw)
    model._access_bid_slope_eur_per_mw2_day = access_slope
    return model


def solve_fixed_access_market(
    data: MarketData,
    *,
    access_quantity: AccessProfile,
    access_bid: AccessProfile,
    energy_capacity: AccessProfile,
    degradation: Mapping[str, float],
    node_limit_mw: float,
    access_bid_slope_eur_per_mw2_day: float = 0.0,
) -> pyo.ConcreteModel:
    """Solve and return the exact fixed-strategy access/dispatch QP."""

    model = build_fixed_access_market(
        data,
        access_quantity=access_quantity,
        access_bid=access_bid,
        energy_capacity=energy_capacity,
        degradation=degradation,
        node_limit_mw=node_limit_mw,
        access_bid_slope_eur_per_mw2_day=access_bid_slope_eur_per_mw2_day,
    )
    if float(access_bid_slope_eur_per_mw2_day) > 0.0:
        solver_kwargs: dict[str, object] = {"solver_io": "nl"}
        executable = _ipopt_executable()
        if executable is not None:
            solver_kwargs["executable"] = str(executable)
        solver = pyo.SolverFactory("ipopt", **solver_kwargs)
        if not solver.available(exception_flag=False):
            raise RuntimeError(
                "Ipopt is required for a positive access-bid slope."
            )
        solver.options.update(
            {
                "linear_solver": "ma57",
                "tol": 1.0e-9,
                "constr_viol_tol": 1.0e-9,
                "dual_inf_tol": 1.0e-9,
                "compl_inf_tol": 1.0e-9,
                "bound_relax_factor": 0.0,
                "honor_original_bounds": "yes",
                "max_iter": 1000,
                "max_cpu_time": 60.0,
                "print_level": 0,
            }
        )
    else:
        solver = pyo.SolverFactory("highs")
    result = solver.solve(model, tee=False)
    if result.solver.termination_condition not in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
    }:
        raise RuntimeError(
            "Fixed-strategy access market did not solve optimally: "
            f"{result.solver.termination_condition}"
        )
    return model


def build_model(
    data: MarketData,
    *,
    investor: InvestorConfig,
    rival_access_quantity: AccessProfile | None = None,
    rival_access_bid: AccessProfile | None = None,
    rival_energy_capacity: AccessProfile | None = None,
    rival_degradation: Mapping[str, float] | None = None,
    node_limit_mw: float = 40.0,
    investor_request_limit_mw: float = DEFAULT_INVESTOR_REQUEST_LIMIT_MW,
    access_bid_bound: float = DEFAULT_ACCESS_BID_BOUND,
    access_bid_slope_eur_per_mw2_day: float = 0.0,
    initial_access_quantity: Mapping[str, float] | None = None,
    initial_access_bid: Mapping[str, float] | None = None,
    initial_energy_capacity: Mapping[str, float] | None = None,
    price_bound: float = 500.0,
    dual_bound: float = 10_000.0,
    complementarity_epsilon: float = DEFAULT_COMPLEMENTARITY_EPSILON,
    proximal_access_quantity: Mapping[str, float] | None = None,
    proximal_access_bid: Mapping[str, float] | None = None,
    proximal_energy_capacity: Mapping[str, float] | None = None,
    proximal_penalty: float = 0.0,
    proximal_quantity_scale: float = 1.0,
    proximal_price_scale: float = 10.0,
    proximal_energy_scale: float = 2.0,
    **_: object,
) -> pyo.ConcreteModel:
    """Build one investor's co-optimised strategic-access best response."""

    epsilon = float(complementarity_epsilon)
    access_slope = float(access_bid_slope_eur_per_mw2_day)
    if min(
        node_limit_mw,
        investor_request_limit_mw,
        access_bid_bound,
        price_bound,
        dual_bound,
        proximal_quantity_scale,
        proximal_price_scale,
        proximal_energy_scale,
    ) <= 0.0:
        raise ValueError("Access, price, dual, request, and scaling bounds must be positive.")
    if epsilon < 0.0 or proximal_penalty < 0.0 or access_slope < 0.0:
        raise ValueError(
            "Complementarity epsilon, proximal penalty, and access slope cannot be negative."
        )
    if (
        access_slope * min(node_limit_mw, investor_request_limit_mw)
        > access_bid_bound
    ):
        raise ValueError(
            "The bid bound is too small to keep the marginal access bid "
            "non-negative at the maximum nodal request."
        )
    if not 0.0 < data.eta <= 1.0:
        raise ValueError("Storage efficiency must be in (0, 1].")
    if not 0.0 <= investor.ratio_min <= investor.ratio_max:
        raise ValueError("Invalid investor duration bounds.")
    unknown_owned = set(investor.owned_generation_shares) - set(data.generators)
    if unknown_owned:
        raise ValueError(f"Unknown owned generators: {sorted(unknown_owned)}")

    active = investor.investor_id
    rivals, rival_quantity, rival_bid, rival_energy = _normalise_rival_profile(
        data,
        active,
        rival_access_quantity,
        rival_access_bid,
        rival_energy_capacity,
    )
    if any(
        rival_bid[unit][node] + 1.0e-6
        < access_slope * rival_quantity[unit][node]
        for unit in rivals
        for node in data.nodes
    ):
        raise ValueError(
            "Every rival access bid must satisfy bid >= slope * quantity."
        )
    degradation = {active: float(investor.degradation_eur_per_mwh)}
    degradation.update(
        {
            unit: float((rival_degradation or {}).get(unit, 15.0))
            for unit in rivals
        }
    )
    units = [active, *rivals]
    storage_pairs = [(unit, node) for unit in units for node in data.nodes]
    gen_nodes = _generator_nodes(data)
    generation_pairs = [
        (generator, time)
        for generator in data.generators
        for time in data.times
        if data.generation_capacity[generator, time] > 1.0e-8
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
    initial_quantity = {
        node: min(
            node_limit_mw,
            max(0.0, float((initial_access_quantity or {}).get(node, 0.0))),
        )
        for node in data.nodes
    }
    initial_quantity_total = sum(initial_quantity.values())
    if initial_quantity_total > investor_request_limit_mw:
        initial_quantity_scale = investor_request_limit_mw / initial_quantity_total
        initial_quantity = {
            node: value * initial_quantity_scale
            for node, value in initial_quantity.items()
        }
    initial_bid_values = {
        node: min(
            access_bid_bound,
            max(
                access_slope * initial_quantity[node],
                float((initial_access_bid or {}).get(node, 0.0)),
                0.0,
            ),
        )
        for node in data.nodes
    }
    initial_energy = {
        node: min(
            investor.ratio_max * min(node_limit_mw, investor_request_limit_mw),
            max(
                0.0,
                float((initial_energy_capacity or {}).get(node, 0.0)),
            ),
        )
        for node in data.nodes
    }
    eta = data.eta
    last_time = max(data.times)
    crf_daily = capital_recovery_factor(investor.wacc, investor.lifetime_years) / 365.25

    model = pyo.ConcreteModel(name=f"Strategic-access relaxed-KKT MPEC [{active}]")
    model.I = pyo.Set(initialize=units, ordered=True)
    model.N = pyo.Set(initialize=data.nodes, ordered=True)
    model.GT = pyo.Set(dimen=2, initialize=generation_pairs, ordered=True)
    model.L = pyo.Set(initialize=data.lines, ordered=True)
    model.T = pyo.Set(initialize=data.times, ordered=True)
    model.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    model.IN = pyo.Set(dimen=2, initialize=storage_pairs, ordered=True)

    model.AccessQuantity = pyo.Var(
        model.N,
        bounds=(0.0, min(node_limit_mw, investor_request_limit_mw)),
        initialize=lambda _, n: initial_quantity[n],
    )
    model.AccessBid = pyo.Var(
        model.N,
        bounds=(0.0, access_bid_bound),
        initialize=lambda _, n: initial_bid_values[n],
    )
    model.EnergyCapacity = pyo.Var(
        model.N,
        bounds=(0.0, investor.ratio_max * min(node_limit_mw, investor_request_limit_mw)),
        initialize=lambda _, n: initial_energy[n],
    )
    model.total_access_request_bound = pyo.Constraint(
        expr=sum(model.AccessQuantity[n] for n in model.N)
        <= investor_request_limit_mw
    )
    model.nonnegative_marginal_access_bid = pyo.Constraint(
        model.N,
        rule=lambda m, n: m.AccessBid[n] >= access_slope * m.AccessQuantity[n],
    )

    def requested_quantity(unit: str, node: str):
        return (
            model.AccessQuantity[node]
            if unit == active
            else rival_quantity[unit][node]
        )

    def access_bid(unit: str, node: str):
        return model.AccessBid[node] if unit == active else rival_bid[unit][node]

    def energy_capacity(unit: str, node: str):
        return (
            model.EnergyCapacity[node]
            if unit == active
            else rival_energy[unit][node]
        )

    model.X_awarded = pyo.Var(model.IN, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_gen = pyo.Var(model.GT, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_charge = pyo.Var(model.IN, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_discharge = pyo.Var(model.IN, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.SOC = pyo.Var(model.IN, model.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    model.NetInjection = pyo.Var(model.N, model.T, domain=pyo.Reals, initialize=0.0)

    model.lam = pyo.Var(model.N, model.T, bounds=(-price_bound, price_bound), initialize=60.0)
    model.lam_sys = pyo.Var(model.T, bounds=(-price_bound, price_bound), initialize=60.0)
    model.nu_gen = pyo.Var(model.GT, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.mu_up = pyo.Var(model.L, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.mu_dn = pyo.Var(model.L, model.T, bounds=(0.0, dual_bound), initialize=0.0)
    model.kappa_power = pyo.Var(model.IN, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.gam = pyo.Var(model.IN, model.T, bounds=(-dual_bound, dual_bound), initialize=0.0)
    model.del_soc = pyo.Var(model.IN, model.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.rho_per = pyo.Var(model.IN, bounds=(-dual_bound, dual_bound), initialize=0.0)
    model.alpha_request = pyo.Var(model.IN, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.alpha_node = pyo.Var(model.N, bounds=(-dual_bound, 0.0), initialize=0.0)

    model.access_request_bound = pyo.Constraint(
        model.IN,
        rule=lambda m, i, n: m.X_awarded[i, n] <= requested_quantity(i, n),
    )
    model.nodal_access_bound = pyo.Constraint(
        model.N,
        rule=lambda m, n: sum(m.X_awarded[i, n] for i in m.I) <= node_limit_mw,
    )
    model.minimum_duration = pyo.Constraint(
        model.N,
        rule=lambda m, n: m.EnergyCapacity[n]
        >= investor.ratio_min * m.X_awarded[active, n],
    )
    model.maximum_duration = pyo.Constraint(
        model.N,
        rule=lambda m, n: m.EnergyCapacity[n]
        <= investor.ratio_max * m.X_awarded[active, n],
    )
    model.nodal_balance = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: sum(
            m.P_gen[g, t] for g in generators_at_node_time[n, t]
        )
        + sum(m.P_discharge[i, n, t] - m.P_charge[i, n, t] for i in m.I)
        - data.demand_el[n, t]
        == m.NetInjection[n, t],
    )
    model.system_balance = pyo.Constraint(
        model.T,
        rule=lambda m, t: sum(m.NetInjection[n, t] for n in m.N) == 0.0,
    )
    model.generation_capacity_bound = pyo.Constraint(
        model.GT,
        rule=lambda m, g, t: m.P_gen[g, t] <= data.generation_capacity[g, t],
    )
    model.line_upper_bound = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, l, t: sum(data.ptdf[l, n] * m.NetInjection[n, t] for n in m.N)
        <= data.line_limit[l],
    )
    model.line_lower_bound = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, l, t: sum(data.ptdf[l, n] * m.NetInjection[n, t] for n in m.N)
        >= -data.line_limit[l],
    )
    model.shared_inverter_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.P_charge[i, n, t]
        + m.P_discharge[i, n, t]
        <= m.X_awarded[i, n],
    )
    model.soc_transition = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.SOC[i, n, t]
        == m.SOC[i, n, t - 1]
        + eta * m.P_charge[i, n, t]
        - m.P_discharge[i, n, t] / eta,
    )
    model.soc_capacity_bound = pyo.Constraint(
        model.IN,
        model.T_SOC,
        rule=lambda m, i, n, tau: m.SOC[i, n, tau]
        <= energy_capacity(i, n),
    )
    model.soc_periodicity = pyo.Constraint(
        model.IN,
        rule=lambda m, i, n: m.SOC[i, n, 0] == m.SOC[i, n, last_time],
    )

    def gen_reduced_cost(m: pyo.ConcreteModel, generator: str, time: int):
        return (
            data.generation_cost[generator]
            - sum(m.lam[node, time] for node in gen_nodes[generator])
            - m.nu_gen[generator, time]
        )

    def charge_reduced_cost(m: pyo.ConcreteModel, unit: str, node: str, time: int):
        return (
            0.5 * degradation[unit]
            + m.lam[node, time]
            - m.kappa_power[unit, node, time]
            + eta * m.gam[unit, node, time]
        )

    def discharge_reduced_cost(m: pyo.ConcreteModel, unit: str, node: str, time: int):
        return (
            0.5 * degradation[unit]
            - m.lam[node, time]
            - m.kappa_power[unit, node, time]
            - m.gam[unit, node, time] / eta
        )

    def soc_reduced_cost(m: pyo.ConcreteModel, unit: str, node: str, soc_time: int):
        stationarity_lhs = m.del_soc[unit, node, soc_time]
        if soc_time in m.T:
            stationarity_lhs += m.gam[unit, node, soc_time]
        if soc_time + 1 in m.T:
            stationarity_lhs -= m.gam[unit, node, soc_time + 1]
        if soc_time == 0:
            stationarity_lhs += m.rho_per[unit, node]
        if soc_time == last_time:
            stationarity_lhs -= m.rho_per[unit, node]
        return -stationarity_lhs

    def awarded_reduced_cost(m: pyo.ConcreteModel, unit: str, node: str):
        return (
            -access_bid(unit, node)
            + access_slope * m.X_awarded[unit, node]
            + sum(m.kappa_power[unit, node, time] for time in m.T)
            - m.alpha_request[unit, node]
            - m.alpha_node[node]
        )

    model.gen_stationarity = pyo.Constraint(
        model.GT,
        rule=lambda m, g, t: gen_reduced_cost(m, g, t) >= 0.0,
    )
    model.charge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: charge_reduced_cost(m, i, n, t) >= 0.0,
    )
    model.discharge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: discharge_reduced_cost(m, i, n, t) >= 0.0,
    )
    model.netinjection_stationarity = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: -m.lam[n, t]
        + m.lam_sys[t]
        + sum(data.ptdf[l, n] * (m.mu_up[l, t] + m.mu_dn[l, t]) for l in m.L)
        == 0.0,
    )
    model.soc_stationarity = pyo.Constraint(
        model.IN,
        model.T_SOC,
        rule=lambda m, i, n, tau: soc_reduced_cost(m, i, n, tau) >= 0.0,
    )
    model.awarded_stationarity = pyo.Constraint(
        model.IN,
        rule=lambda m, i, n: awarded_reduced_cost(m, i, n) >= 0.0,
    )

    def flow(m: pyo.ConcreteModel, line: str, time: int):
        return sum(data.ptdf[line, node] * m.NetInjection[node, time] for node in m.N)

    product_names: list[str] = []

    def add_relaxed_product(name: str, index_sets: tuple[pyo.Set, ...], rule) -> None:
        product = pyo.Expression(*index_sets, rule=rule)
        model.add_component(f"{name}_product", product)
        model.add_component(
            name,
            pyo.Constraint(
                *index_sets,
                rule=lambda m, *key: pyo.inequality(0.0, product[key], epsilon),
            ),
        )
        product_names.append(f"{name}_product")

    add_relaxed_product(
        "relaxed_comp_awarded_lower",
        (model.IN,),
        lambda m, i, n: m.X_awarded[i, n] * awarded_reduced_cost(m, i, n),
    )
    add_relaxed_product(
        "relaxed_comp_request_upper",
        (model.IN,),
        lambda m, i, n: (requested_quantity(i, n) - m.X_awarded[i, n])
        * (-m.alpha_request[i, n]),
    )
    add_relaxed_product(
        "relaxed_comp_nodal_access_upper",
        (model.N,),
        lambda m, n: (
            node_limit_mw - sum(m.X_awarded[i, n] for i in m.I)
        )
        * (-m.alpha_node[n]),
    )
    add_relaxed_product(
        "relaxed_comp_gen_lower",
        (model.GT,),
        lambda m, g, t: m.P_gen[g, t] * gen_reduced_cost(m, g, t),
    )
    add_relaxed_product(
        "relaxed_comp_charge_lower",
        (model.IN, model.T),
        lambda m, i, n, t: m.P_charge[i, n, t] * charge_reduced_cost(m, i, n, t),
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
        lambda m, i, n, tau: m.SOC[i, n, tau] * soc_reduced_cost(m, i, n, tau),
    )
    add_relaxed_product(
        "relaxed_comp_gen_upper",
        (model.GT,),
        lambda m, g, t: (data.generation_capacity[g, t] - m.P_gen[g, t])
        * (-m.nu_gen[g, t]),
    )
    add_relaxed_product(
        "relaxed_comp_line_upper",
        (model.L, model.T),
        lambda m, l, t: (data.line_limit[l] - flow(m, l, t)) * (-m.mu_up[l, t]),
    )
    add_relaxed_product(
        "relaxed_comp_line_lower",
        (model.L, model.T),
        lambda m, l, t: (flow(m, l, t) + data.line_limit[l]) * m.mu_dn[l, t],
    )
    add_relaxed_product(
        "relaxed_comp_shared_inverter_upper",
        (model.IN, model.T),
        lambda m, i, n, t: (
            m.X_awarded[i, n] - m.P_charge[i, n, t] - m.P_discharge[i, n, t]
        )
        * (-m.kappa_power[i, n, t]),
    )
    add_relaxed_product(
        "relaxed_comp_soc_upper",
        (model.IN, model.T_SOC),
        lambda m, i, n, tau: (
            energy_capacity(i, n) - m.SOC[i, n, tau]
        )
        * (-m.del_soc[i, n, tau]),
    )

    model.lower_level_degradation = pyo.Expression(
        expr=sum(
            0.5
            * degradation[i]
            * (model.P_charge[i, n, t] + model.P_discharge[i, n, t])
            for i, n in model.IN
            for t in model.T
        )
    )
    model.primal_objective = pyo.Expression(
        expr=sum(
            data.generation_cost[g] * model.P_gen[g, t]
            for g, t in model.GT
        )
        + model.lower_level_degradation
        - sum(access_bid(i, n) * model.X_awarded[i, n] for i, n in model.IN)
        + 0.5
        * access_slope
        * sum(model.X_awarded[i, n] ** 2 for i, n in model.IN)
    )
    model.dual_objective = pyo.Expression(
        expr=sum(data.demand_el[n, t] * model.lam[n, t] for n in model.N for t in model.T)
        + sum(data.generation_capacity[g, t] * model.nu_gen[g, t] for g, t in model.GT)
        + sum(
            data.line_limit[l] * (model.mu_up[l, t] - model.mu_dn[l, t])
            for l in model.L
            for t in model.T
        )
        + sum(
            requested_quantity(i, n) * model.alpha_request[i, n]
            for i, n in model.IN
        )
        + node_limit_mw * sum(model.alpha_node[n] for n in model.N)
        + sum(
            energy_capacity(i, n) * model.del_soc[i, n, tau]
            for i, n in model.IN
            for tau in model.T_SOC
        )
        - 0.5
        * access_slope
        * sum(model.X_awarded[i, n] ** 2 for i, n in model.IN)
    )

    model.spot_revenue = pyo.Expression(
        expr=sum(
            model.lam[n, t]
            * (model.P_discharge[active, n, t] - model.P_charge[active, n, t])
            for n in model.N
            for t in model.T
        )
    )
    model.generation_rent = pyo.Expression(
        expr=sum(
            share
            * (model.lam[gen_nodes[g][0], t] - data.generation_cost[g])
            * model.P_gen[g, t]
            for g, share in investor.owned_generation_shares.items()
            for t in model.T
            if share and (g, t) in model.GT
        )
    )
    model.active_degradation = pyo.Expression(
        expr=0.5
        * investor.degradation_eur_per_mwh
        * sum(
            model.P_charge[active, n, t] + model.P_discharge[active, n, t]
            for n in model.N
            for t in model.T
        )
    )
    model.daily_capex = pyo.Expression(
        expr=crf_daily
        * sum(
            investor.cost_power_eur_per_mw * model.X_awarded[active, n]
            + investor.cost_energy_eur_per_mwh * model.EnergyCapacity[n]
            for n in model.N
        )
    )
    model.access_payment = pyo.Expression(
        expr=sum(
            model.AccessBid[n] * model.X_awarded[active, n]
            - 0.5 * access_slope * model.X_awarded[active, n] ** 2
            for n in model.N
        )
    )
    model.unregularized_profit = pyo.Expression(
        expr=model.spot_revenue
        + model.generation_rent
        - model.active_degradation
        - model.daily_capex
        - model.access_payment
    )

    if proximal_penalty > 0.0:
        if (
            proximal_access_quantity is None
            or proximal_access_bid is None
            or proximal_energy_capacity is None
        ):
            raise ValueError(
                "A positive access proximal penalty requires quantity, bid, and energy centres."
            )
        # Scale each strategic dimension before squaring.  Unlike the previous
        # L1 term, this regularizer has zero gradient at the proximal centre.
        regularizer = 0.5 * proximal_penalty * sum(
            (
                (model.AccessQuantity[n] - float(proximal_access_quantity[n]))
                / proximal_quantity_scale
            )
            ** 2
            + (
                (model.EnergyCapacity[n] - float(proximal_energy_capacity[n]))
                / proximal_energy_scale
            )
            ** 2
            + (
                (model.AccessBid[n] - float(proximal_access_bid[n]))
                / proximal_price_scale
            )
            ** 2
            for n in model.N
        )
    else:
        regularizer = 0.0
    model.regularizer = pyo.Expression(expr=regularizer)
    model.profit = pyo.Expression(expr=model.unregularized_profit - model.regularizer)
    model.objective = pyo.Objective(expr=model.profit, sense=pyo.maximize)

    model._market_data = data
    model._investor = investor
    model._active_id = active
    model._rival_ids = tuple(rivals)
    model._rival_access_quantity = rival_quantity
    model._rival_access_bid = rival_bid
    model._rival_energy_capacity = rival_energy
    model._unit_degradation = degradation
    model._gen_nodes = gen_nodes
    model._node_limit_mw = float(node_limit_mw)
    model._investor_request_limit_mw = float(investor_request_limit_mw)
    model._access_bid_bound = float(access_bid_bound)
    model._access_bid_slope_eur_per_mw2_day = access_slope
    model._lower_level_optimality = "relaxed-kkt"
    model._strategic_access = True
    model._access_settlement = "integrated-linear-pay-as-bid"
    model._complementarity_epsilon = epsilon
    model._relaxed_kkt_product_components = tuple(product_names)
    model._proximal_access_quantity = dict(proximal_access_quantity or {})
    model._proximal_access_bid = dict(proximal_access_bid or {})
    model._proximal_energy_capacity = dict(proximal_energy_capacity or {})
    model._proximal_regularizer = "scaled-quadratic"
    return model


def _fixed_profiles_from_model(
    model: pyo.ConcreteModel,
    data: MarketData,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    active = model._active_id
    units = [active, *model._rival_ids]
    quantity: dict[str, dict[str, float]] = {}
    bid: dict[str, dict[str, float]] = {}
    energy: dict[str, dict[str, float]] = {}
    for unit in units:
        quantity[unit] = {}
        bid[unit] = {}
        energy[unit] = {}
        for node in data.nodes:
            if unit == active:
                quantity[unit][node] = float(pyo.value(model.AccessQuantity[node]))
                bid[unit][node] = float(pyo.value(model.AccessBid[node]))
                energy[unit][node] = float(pyo.value(model.EnergyCapacity[node]))
            else:
                quantity[unit][node] = model._rival_access_quantity[unit][node]
                bid[unit][node] = model._rival_access_bid[unit][node]
                energy[unit][node] = model._rival_energy_capacity[unit][node]
    return quantity, bid, energy


def initialise_lower_level(model: pyo.ConcreteModel, data: MarketData) -> None:
    """Seed all lower primal/dual variables from the exact fixed-strategy LP."""

    quantity, bid, energy = _fixed_profiles_from_model(model, data)
    lower = solve_fixed_access_market(
        data,
        access_quantity=quantity,
        access_bid=bid,
        energy_capacity=energy,
        degradation=model._unit_degradation,
        node_limit_mw=model._node_limit_mw,
        access_bid_slope_eur_per_mw2_day=(
            model._access_bid_slope_eur_per_mw2_day
        ),
    )

    def seed(variable: pyo.Var, raw_value: float) -> None:
        value = float(raw_value)
        if abs(value) < 1.0e-9:
            value = 0.0
        if variable.lb is not None:
            value = max(value, float(pyo.value(variable.lb)))
        if variable.ub is not None:
            value = min(value, float(pyo.value(variable.ub)))
        variable.set_value(value)

    for unit, node in model.IN:
        seed(model.X_awarded[unit, node], lower.X_awarded[unit, node].value)
        seed(
            model.alpha_request[unit, node],
            lower.dual[lower.access_request_bound[unit, node]],
        )
        for time in model.T:
            key = unit, node, time
            seed(model.P_charge[key], lower.P_charge[key].value)
            seed(model.P_discharge[key], lower.P_discharge[key].value)
            seed(model.kappa_power[key], lower.dual[lower.shared_inverter_bound[key]])
            seed(model.gam[key], lower.dual[lower.soc_transition[key]])
        for soc_time in model.T_SOC:
            key = unit, node, soc_time
            seed(model.SOC[key], lower.SOC[key].value)
            seed(model.del_soc[key], lower.dual[lower.soc_capacity_bound[key]])
        seed(
            model.rho_per[unit, node],
            lower.dual[lower.soc_periodicity[unit, node]],
        )
    for node in model.N:
        seed(model.alpha_node[node], lower.dual[lower.nodal_access_bound[node]])
        for time in model.T:
            seed(model.NetInjection[node, time], lower.NetInjection[node, time].value)
            seed(model.lam[node, time], lower.dual[lower.nodal_balance[node, time]])
    for generator, time in model.GT:
        seed(model.P_gen[generator, time], lower.P_gen[generator, time].value)
        seed(
            model.nu_gen[generator, time],
            lower.dual[lower.generation_capacity_bound[generator, time]],
        )
    for time in model.T:
        seed(model.lam_sys[time], lower.dual[lower.system_balance[time]])
    for line in model.L:
        for time in model.T:
            seed(model.mu_up[line, time], lower.dual[lower.line_upper_bound[line, time]])
            seed(model.mu_dn[line, time], lower.dual[lower.line_lower_bound[line, time]])

def diagnostics(model: pyo.ConcreteModel) -> dict[str, float | int]:
    """Evaluate relaxed products and the fixed-strategy primal-dual gap."""

    products = [
        float(pyo.value(component[index]))
        for name in model._relaxed_kkt_product_components
        for component in (getattr(model, name),)
        for index in component
    ]
    epsilon = float(model._complementarity_epsilon)
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
