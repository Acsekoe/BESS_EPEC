"""Three-bus strategic BESS availability toy model.

The ISO clears a linear dispatch problem for fixed hourly charge/discharge
availability offers.  One investor's best response is represented as a
Scholtes-relaxed KKT MPEC and solved with IPOPT.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pyomo.environ as pyo


Profile = Mapping[tuple[str, int], float]
CapacityProfile = Mapping[str, float]


@dataclass(frozen=True)
class Generator:
    generator_id: str
    node: str
    marginal_cost: float
    capacity_mw: Mapping[int, float]
    quadratic_cost_eur_per_mw2: float = 0.0


@dataclass(frozen=True)
class StorageInvestor:
    investor_id: str
    node: str
    power_mw: float
    energy_mwh: float
    degradation_eur_per_mwh: float = 15.0
    dispatch_quadratic_eur_per_mw2: float = 0.02
    wacc: float = 0.08
    lifetime_years: int = 15
    cost_power_eur_per_mw: float = 6_600.0
    cost_energy_eur_per_mwh: float = 18_800.0
    power_upper_mw: float = 30.0
    duration_min_hours: float = 2.0
    duration_max_hours: float = 8.0
    owned_generation_shares: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ToyData:
    nodes: tuple[str, ...]
    times: tuple[int, ...]
    soc_times: tuple[int, ...]
    lines: tuple[str, ...]
    generators: tuple[Generator, ...]
    investors: tuple[StorageInvestor, ...]
    demand_mw: Mapping[tuple[str, int], float]
    ptdf: Mapping[tuple[str, str], float]
    line_limit_mw: Mapping[str, float]
    efficiency: float
    demand_adjustment_penalty_eur_per_mw2: float


@dataclass(frozen=True)
class BestResponse:
    investor_id: str
    optimal: bool
    termination: str
    solve_seconds: float
    charge_offer_mw: dict[int, float]
    discharge_offer_mw: dict[int, float]
    profit_eur_per_day: float
    recleared_profit_eur_per_day: float
    embedded_reclear_profit_gap_eur_per_day: float
    maximum_lmp_reclear_gap_eur_per_mwh: float
    maximum_complementarity_product: float
    maximum_complementarity_violation: float
    absolute_primal_dual_gap_eur_per_day: float
    power_mw: float | None = None
    energy_mwh: float | None = None


def capital_recovery_factor(wacc: float, lifetime_years: int) -> float:
    if wacc == 0.0:
        return 1.0 / lifetime_years
    growth = (1.0 + wacc) ** lifetime_years
    return wacc * growth / (growth - 1.0)


def default_data() -> ToyData:
    """Return the intentionally small, congestible six-hour test system."""

    times = (1, 2, 3, 4, 5, 6)
    generators = (
        Generator("G1_BASE", "N1", 25.0, {t: 100.0 for t in times}, 0.04),
        Generator("G2_MID", "N2", 55.0, {t: 65.0 for t in times}, 0.06),
        Generator("G3_PEAK", "N3", 110.0, {t: 100.0 for t in times}, 0.08),
        Generator(
            "RES_WIND_N1",
            "N1",
            0.0,
            dict(zip(times, (18.0, 15.0, 12.0, 10.0, 14.0, 20.0), strict=True)),
            0.001,
        ),
        Generator(
            "RES_PV_N3",
            "N3",
            0.0,
            dict(zip(times, (0.0, 15.0, 45.0, 20.0, 0.0, 0.0), strict=True)),
            0.001,
        ),
    )
    investors = (
        # Match the maintained four-investor population in the IEEE-9 model.
        # Splitting the original two batteries into four equal batteries keeps
        # aggregate toy storage fixed at 40 MW / 100 MWh.
        StorageInvestor("I1", "N3", power_mw=10.0, energy_mwh=25.0, wacc=0.08),
        StorageInvestor("I2", "N3", power_mw=10.0, energy_mwh=25.0, wacc=0.12),
        StorageInvestor(
            "I3",
            "N3",
            power_mw=10.0,
            energy_mwh=25.0,
            wacc=0.08,
            owned_generation_shares={"RES_WIND_N1": 0.8, "RES_PV_N3": 0.2},
        ),
        StorageInvestor(
            "I4",
            "N3",
            power_mw=10.0,
            energy_mwh=25.0,
            wacc=0.08,
            owned_generation_shares={"RES_WIND_N1": 0.2, "RES_PV_N3": 0.8},
        ),
    )
    demand_profiles = {
        "N1": (45.0, 45.0, 45.0, 55.0, 65.0, 55.0),
        "N2": (35.0, 40.0, 50.0, 55.0, 45.0, 35.0),
        "N3": (25.0, 30.0, 45.0, 60.0, 90.0, 40.0),
    }
    demand = {
        (node, time): values[position]
        for node, values in demand_profiles.items()
        for position, time in enumerate(times)
    }

    # Triangle network with N1 as the PTDF reference bus. Orientations are
    # N1->N2, N2->N3, and N1->N3 respectively.
    ptdf = {
        ("L12", "N1"): 0.0,
        ("L12", "N2"): -2.0 / 3.0,
        ("L12", "N3"): -1.0 / 3.0,
        ("L23", "N1"): 0.0,
        ("L23", "N2"): 1.0 / 3.0,
        ("L23", "N3"): -1.0 / 3.0,
        ("L13", "N1"): 0.0,
        ("L13", "N2"): -1.0 / 3.0,
        ("L13", "N3"): -2.0 / 3.0,
    }
    return ToyData(
        nodes=("N1", "N2", "N3"),
        times=times,
        soc_times=(0, *times),
        lines=("L12", "L23", "L13"),
        generators=generators,
        investors=investors,
        demand_mw=demand,
        ptdf=ptdf,
        line_limit_mw={"L12": 40.0, "L23": 25.0, "L13": 30.0},
        efficiency=0.92,
        demand_adjustment_penalty_eur_per_mw2=500.0,
    )


def full_availability(data: ToyData) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], float]]:
    charge = {
        (investor.investor_id, time): investor.power_mw
        for investor in data.investors
        for time in data.times
    }
    return charge, dict(charge)


def installed_capacities(data: ToyData) -> tuple[dict[str, float], dict[str, float]]:
    """Return the power and energy profiles stored in the toy data."""

    power = {investor.investor_id: investor.power_mw for investor in data.investors}
    energy = {investor.investor_id: investor.energy_mwh for investor in data.investors}
    return power, energy


def _capacity_profiles(
    data: ToyData,
    power_capacity_mw: CapacityProfile | None,
    energy_capacity_mwh: CapacityProfile | None,
) -> tuple[dict[str, float], dict[str, float]]:
    default_power, default_energy = installed_capacities(data)
    power = default_power if power_capacity_mw is None else {
        key: float(value) for key, value in power_capacity_mw.items()
    }
    energy = default_energy if energy_capacity_mwh is None else {
        key: float(value) for key, value in energy_capacity_mwh.items()
    }
    expected = set(default_power)
    if set(power) != expected or set(energy) != expected:
        raise ValueError("Capacity profiles must cover every investor exactly once.")
    investor_by_id = {investor.investor_id: investor for investor in data.investors}
    for investor_id in expected:
        investor = investor_by_id[investor_id]
        if not 0.0 <= power[investor_id] <= investor.power_upper_mw + 1.0e-9:
            raise ValueError(f"Power capacity outside its bounds for {investor_id}.")
        if energy[investor_id] < -1.0e-9:
            raise ValueError(f"Negative energy capacity for {investor_id}.")
        if energy[investor_id] < investor.duration_min_hours * power[investor_id] - 1.0e-8:
            raise ValueError(f"Energy capacity is below minimum duration for {investor_id}.")
        if energy[investor_id] > investor.duration_max_hours * power[investor_id] + 1.0e-8:
            raise ValueError(f"Energy capacity is above maximum duration for {investor_id}.")
    return power, energy


def _validate_profiles(
    data: ToyData,
    charge: Profile,
    discharge: Profile,
    power_capacity_mw: CapacityProfile | None = None,
) -> None:
    investor_by_id = {investor.investor_id: investor for investor in data.investors}
    power = (
        {investor_id: investor.power_mw for investor_id, investor in investor_by_id.items()}
        if power_capacity_mw is None
        else {key: float(value) for key, value in power_capacity_mw.items()}
    )
    expected = {(investor_id, time) for investor_id in investor_by_id for time in data.times}
    if set(power) != set(investor_by_id):
        raise ValueError("Power-capacity profile must cover every investor exactly once.")
    if set(charge) != expected or set(discharge) != expected:
        raise ValueError("Availability profiles must cover every investor-hour pair.")
    for key in expected:
        limit = power[key[0]]
        if not 0.0 <= float(charge[key]) <= limit + 1.0e-9:
            raise ValueError(f"Charging availability outside [0, {limit}] for {key}.")
        if not 0.0 <= float(discharge[key]) <= limit + 1.0e-9:
            raise ValueError(f"Discharging availability outside [0, {limit}] for {key}.")


def build_market(
    data: ToyData,
    charge: Profile,
    discharge: Profile,
    *,
    power_capacity_mw: CapacityProfile | None = None,
    energy_capacity_mwh: CapacityProfile | None = None,
) -> pyo.ConcreteModel:
    """Build the exact ISO LP for one fixed availability profile."""

    power, energy = _capacity_profiles(data, power_capacity_mw, energy_capacity_mwh)
    _validate_profiles(data, charge, discharge, power)
    generator_by_id = {generator.generator_id: generator for generator in data.generators}
    investor_by_id = {investor.investor_id: investor for investor in data.investors}
    generators_at_node = {
        node: [g.generator_id for g in data.generators if g.node == node]
        for node in data.nodes
    }
    investors_at_node = {
        node: [i.investor_id for i in data.investors if i.node == node]
        for node in data.nodes
    }
    last_time = max(data.times)

    model = pyo.ConcreteModel(name="Three-bus fixed-availability ISO clearing")
    model.N = pyo.Set(initialize=data.nodes, ordered=True)
    model.T = pyo.Set(initialize=data.times, ordered=True)
    model.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    model.L = pyo.Set(initialize=data.lines, ordered=True)
    model.G = pyo.Set(initialize=tuple(generator_by_id), ordered=True)
    model.I = pyo.Set(initialize=tuple(investor_by_id), ordered=True)

    model.PGen = pyo.Var(model.G, model.T, domain=pyo.NonNegativeReals)
    model.PCharge = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals)
    model.PDischarge = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.I, model.T_SOC, domain=pyo.NonNegativeReals)
    model.NetInjection = pyo.Var(model.N, model.T, domain=pyo.Reals)
    model.DemandAdjustment = pyo.Var(model.N, model.T, domain=pyo.Reals)

    model.nodal_balance = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: sum(m.PGen[g, t] for g in generators_at_node[n])
        + sum(m.PDischarge[i, t] - m.PCharge[i, t] for i in investors_at_node[n])
        + m.DemandAdjustment[n, t]
        - data.demand_mw[n, t]
        == m.NetInjection[n, t],
    )
    model.system_balance = pyo.Constraint(
        model.T, rule=lambda m, t: sum(m.NetInjection[n, t] for n in m.N) == 0.0
    )
    model.generation_upper = pyo.Constraint(
        model.G,
        model.T,
        rule=lambda m, g, t: m.PGen[g, t] <= generator_by_id[g].capacity_mw[t],
    )

    def flow(m: pyo.ConcreteModel, line: str, time_: int):
        return sum(data.ptdf[line, node] * m.NetInjection[node, time_] for node in m.N)

    model.line_upper = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, line, t: flow(m, line, t) <= data.line_limit_mw[line],
    )
    model.line_lower = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, line, t: flow(m, line, t) >= -data.line_limit_mw[line],
    )
    model.charge_offer_upper = pyo.Constraint(
        model.I,
        model.T,
        rule=lambda m, i, t: m.PCharge[i, t] <= float(charge[i, int(t)]),
    )
    model.discharge_offer_upper = pyo.Constraint(
        model.I,
        model.T,
        rule=lambda m, i, t: m.PDischarge[i, t] <= float(discharge[i, int(t)]),
    )
    model.shared_inverter_upper = pyo.Constraint(
        model.I,
        model.T,
        rule=lambda m, i, t: m.PCharge[i, t] + m.PDischarge[i, t]
        <= power[i],
    )
    model.soc_transition = pyo.Constraint(
        model.I,
        model.T,
        rule=lambda m, i, t: m.SOC[i, t]
        == m.SOC[i, t - 1]
        + data.efficiency * m.PCharge[i, t]
        - m.PDischarge[i, t] / data.efficiency,
    )
    model.soc_upper = pyo.Constraint(
        model.I,
        model.T_SOC,
        rule=lambda m, i, tau: m.SOC[i, tau] <= energy[i],
    )
    model.soc_periodicity = pyo.Constraint(
        model.I, rule=lambda m, i: m.SOC[i, 0] == m.SOC[i, last_time]
    )
    model.objective = pyo.Objective(
        expr=sum(
            generator_by_id[g].marginal_cost * model.PGen[g, t]
            + 0.5
            * generator_by_id[g].quadratic_cost_eur_per_mw2
            * model.PGen[g, t] ** 2
            for g in model.G
            for t in model.T
        )
        + sum(
            0.5
            * investor_by_id[i].degradation_eur_per_mwh
            * (model.PCharge[i, t] + model.PDischarge[i, t])
            + 0.5
            * investor_by_id[i].dispatch_quadratic_eur_per_mw2
            * (model.PCharge[i, t] ** 2 + model.PDischarge[i, t] ** 2)
            for i in model.I
            for t in model.T
        )
        + 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(model.DemandAdjustment[n, t] ** 2 for n in model.N for t in model.T),
        sense=pyo.minimize,
    )
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model._toy_data = data
    model._charge_profile = dict(charge)
    model._discharge_profile = dict(discharge)
    model._power_capacity_mw = dict(power)
    model._energy_capacity_mwh = dict(energy)
    return model


def solve_market(
    data: ToyData,
    charge: Profile,
    discharge: Profile,
    *,
    power_capacity_mw: CapacityProfile | None = None,
    energy_capacity_mwh: CapacityProfile | None = None,
) -> pyo.ConcreteModel:
    model = build_market(
        data,
        charge,
        discharge,
        power_capacity_mw=power_capacity_mw,
        energy_capacity_mwh=energy_capacity_mwh,
    )
    solver = pyo.SolverFactory("highs")
    if not solver.available(exception_flag=False):
        solver = pyo.SolverFactory("appsi_highs")
    result = solver.solve(model, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"ISO clearing failed: {result.solver.termination_condition}")
    return model


def _ipopt_executable() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("IPOPT_EXECUTABLE"):
        candidates.append(Path(os.environ["IPOPT_EXECUTABLE"]))
    if shutil.which("ipopt"):
        candidates.append(Path(shutil.which("ipopt") or ""))
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "idaes" / "bin" / "ipopt.exe")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def build_best_response_mpec(
    data: ToyData,
    active_id: str,
    charge_profile: Profile,
    discharge_profile: Profile,
    *,
    power_capacity_mw: CapacityProfile | None = None,
    energy_capacity_mwh: CapacityProfile | None = None,
    endogenous_investment: bool = False,
    complementarity_epsilon: float = 1.0e-5,
    availability_tie_breaker_eur_per_mw: float = 1.0e-3,
    strategic_charge: bool = False,
    price_bound: float = 500.0,
    dual_bound: float = 10_000.0,
) -> pyo.ConcreteModel:
    """Build one investor's availability or joint investment best-response MPEC."""

    power, energy = _capacity_profiles(data, power_capacity_mw, energy_capacity_mwh)
    _validate_profiles(data, charge_profile, discharge_profile, power)
    epsilon = float(complementarity_epsilon)
    tie_breaker = float(availability_tie_breaker_eur_per_mw)
    if epsilon < 0.0 or tie_breaker < 0.0:
        raise ValueError("Complementarity epsilon and availability tie-breaker cannot be negative.")
    investor_by_id = {investor.investor_id: investor for investor in data.investors}
    generator_by_id = {generator.generator_id: generator for generator in data.generators}
    if active_id not in investor_by_id:
        raise ValueError(f"Unknown active investor: {active_id}")
    active = investor_by_id[active_id]
    generators_at_node = {
        node: [g.generator_id for g in data.generators if g.node == node]
        for node in data.nodes
    }
    investors_at_node = {
        node: [i.investor_id for i in data.investors if i.node == node]
        for node in data.nodes
    }
    last_time = max(data.times)

    formulation = "joint investment/availability" if endogenous_investment else "availability"
    model = pyo.ConcreteModel(name=f"Three-bus strategic {formulation} MPEC [{active_id}]")
    model.N = pyo.Set(initialize=data.nodes, ordered=True)
    model.T = pyo.Set(initialize=data.times, ordered=True)
    model.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    model.L = pyo.Set(initialize=data.lines, ordered=True)
    model.G = pyo.Set(initialize=tuple(generator_by_id), ordered=True)
    model.I = pyo.Set(initialize=tuple(investor_by_id), ordered=True)

    if endogenous_investment:
        model.PowerCapacity = pyo.Var(
            bounds=(0.0, active.power_upper_mw), initialize=power[active_id]
        )
        model.EnergyCapacity = pyo.Var(
            bounds=(0.0, active.duration_max_hours * active.power_upper_mw),
            initialize=energy[active_id],
        )
        model.minimum_duration = pyo.Constraint(
            expr=model.EnergyCapacity >= active.duration_min_hours * model.PowerCapacity
        )
        model.maximum_duration = pyo.Constraint(
            expr=model.EnergyCapacity <= active.duration_max_hours * model.PowerCapacity
        )
        offer_upper = active.power_upper_mw
    else:
        model.PowerCapacity = pyo.Param(initialize=power[active_id])
        model.EnergyCapacity = pyo.Param(initialize=energy[active_id])
        offer_upper = power[active_id]

    model.ChargeOffer = pyo.Var(
        model.T,
        bounds=(0.0, offer_upper),
        initialize=lambda _, t: float(charge_profile[active_id, int(t)]),
    )
    model.DischargeOffer = pyo.Var(
        model.T,
        bounds=(0.0, offer_upper),
        initialize=lambda _, t: float(discharge_profile[active_id, int(t)]),
    )
    model.charge_offer_capacity = pyo.Constraint(
        model.T, rule=lambda m, t: m.ChargeOffer[t] <= m.PowerCapacity
    )
    model.discharge_offer_capacity = pyo.Constraint(
        model.T, rule=lambda m, t: m.DischargeOffer[t] <= m.PowerCapacity
    )
    if endogenous_investment and not strategic_charge:
        model.full_charge_availability = pyo.Constraint(
            model.T, rule=lambda m, t: m.ChargeOffer[t] == m.PowerCapacity
        )
    elif not strategic_charge:
        for time_ in model.T:
            model.ChargeOffer[time_].fix(power[active_id])

    def installed_power(unit: str):
        return model.PowerCapacity if unit == active_id else power[unit]

    def installed_energy(unit: str):
        return model.EnergyCapacity if unit == active_id else energy[unit]

    def available_charge(unit: str, time_: int):
        return model.ChargeOffer[time_] if unit == active_id else float(charge_profile[unit, time_])

    def available_discharge(unit: str, time_: int):
        return model.DischargeOffer[time_] if unit == active_id else float(discharge_profile[unit, time_])

    model.PGen = pyo.Var(model.G, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.PCharge = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.PDischarge = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.SOC = pyo.Var(model.I, model.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    model.NetInjection = pyo.Var(model.N, model.T, domain=pyo.Reals, initialize=0.0)
    model.DemandAdjustment = pyo.Var(model.N, model.T, domain=pyo.Reals, initialize=0.0)

    model.LMP = pyo.Var(model.N, model.T, bounds=(-price_bound, price_bound), initialize=55.0)
    model.SystemPrice = pyo.Var(model.T, bounds=(-price_bound, price_bound), initialize=55.0)
    model.GenUpperDual = pyo.Var(model.G, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.LineUpperDual = pyo.Var(model.L, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.LineLowerDual = pyo.Var(model.L, model.T, bounds=(0.0, dual_bound), initialize=0.0)
    model.ChargeOfferDual = pyo.Var(model.I, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.DischargeOfferDual = pyo.Var(model.I, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.InverterDual = pyo.Var(model.I, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.SOCDynamicDual = pyo.Var(model.I, model.T, bounds=(-dual_bound, dual_bound), initialize=0.0)
    model.SOCUpperDual = pyo.Var(model.I, model.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.SOCPeriodDual = pyo.Var(model.I, bounds=(-dual_bound, dual_bound), initialize=0.0)

    model.nodal_balance = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: sum(m.PGen[g, t] for g in generators_at_node[n])
        + sum(m.PDischarge[i, t] - m.PCharge[i, t] for i in investors_at_node[n])
        + m.DemandAdjustment[n, t]
        - data.demand_mw[n, t]
        == m.NetInjection[n, t],
    )
    model.system_balance = pyo.Constraint(
        model.T, rule=lambda m, t: sum(m.NetInjection[n, t] for n in m.N) == 0.0
    )
    model.generation_upper = pyo.Constraint(
        model.G,
        model.T,
        rule=lambda m, g, t: m.PGen[g, t] <= generator_by_id[g].capacity_mw[t],
    )

    def flow(m: pyo.ConcreteModel, line: str, time_: int):
        return sum(data.ptdf[line, node] * m.NetInjection[node, time_] for node in m.N)

    model.line_upper = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, line, t: flow(m, line, t) <= data.line_limit_mw[line],
    )
    model.line_lower = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, line, t: flow(m, line, t) >= -data.line_limit_mw[line],
    )
    model.charge_offer_upper = pyo.Constraint(
        model.I,
        model.T,
        rule=lambda m, i, t: m.PCharge[i, t] <= available_charge(i, int(t)),
    )
    model.discharge_offer_upper = pyo.Constraint(
        model.I,
        model.T,
        rule=lambda m, i, t: m.PDischarge[i, t] <= available_discharge(i, int(t)),
    )
    model.shared_inverter_upper = pyo.Constraint(
        model.I,
        model.T,
        rule=lambda m, i, t: m.PCharge[i, t] + m.PDischarge[i, t]
        <= installed_power(i),
    )
    model.soc_transition = pyo.Constraint(
        model.I,
        model.T,
        rule=lambda m, i, t: m.SOC[i, t]
        == m.SOC[i, t - 1]
        + data.efficiency * m.PCharge[i, t]
        - m.PDischarge[i, t] / data.efficiency,
    )
    model.soc_upper = pyo.Constraint(
        model.I,
        model.T_SOC,
        rule=lambda m, i, tau: m.SOC[i, tau] <= installed_energy(i),
    )
    model.soc_periodicity = pyo.Constraint(
        model.I, rule=lambda m, i: m.SOC[i, 0] == m.SOC[i, last_time]
    )

    def gen_reduced_cost(m: pyo.ConcreteModel, generator: str, time_: int):
        return (
            generator_by_id[generator].marginal_cost
            + generator_by_id[generator].quadratic_cost_eur_per_mw2
            * m.PGen[generator, time_]
            - m.LMP[generator_by_id[generator].node, time_]
            - m.GenUpperDual[generator, time_]
        )

    def charge_reduced_cost(m: pyo.ConcreteModel, unit: str, time_: int):
        investor = investor_by_id[unit]
        return (
            0.5 * investor.degradation_eur_per_mwh
            + investor.dispatch_quadratic_eur_per_mw2 * m.PCharge[unit, time_]
            + m.LMP[investor.node, time_]
            - m.ChargeOfferDual[unit, time_]
            - m.InverterDual[unit, time_]
            + data.efficiency * m.SOCDynamicDual[unit, time_]
        )

    def discharge_reduced_cost(m: pyo.ConcreteModel, unit: str, time_: int):
        investor = investor_by_id[unit]
        return (
            0.5 * investor.degradation_eur_per_mwh
            + investor.dispatch_quadratic_eur_per_mw2 * m.PDischarge[unit, time_]
            - m.LMP[investor.node, time_]
            - m.DischargeOfferDual[unit, time_]
            - m.InverterDual[unit, time_]
            - m.SOCDynamicDual[unit, time_] / data.efficiency
        )

    def soc_reduced_cost(m: pyo.ConcreteModel, unit: str, tau: int):
        stationarity_lhs = m.SOCUpperDual[unit, tau]
        if tau in m.T:
            stationarity_lhs += m.SOCDynamicDual[unit, tau]
        if tau + 1 in m.T:
            stationarity_lhs -= m.SOCDynamicDual[unit, tau + 1]
        if tau == 0:
            stationarity_lhs += m.SOCPeriodDual[unit]
        if tau == last_time:
            stationarity_lhs -= m.SOCPeriodDual[unit]
        return -stationarity_lhs

    model.gen_stationarity = pyo.Constraint(
        model.G, model.T, rule=lambda m, g, t: gen_reduced_cost(m, g, int(t)) >= 0.0
    )
    model.charge_stationarity = pyo.Constraint(
        model.I, model.T, rule=lambda m, i, t: charge_reduced_cost(m, i, int(t)) >= 0.0
    )
    model.discharge_stationarity = pyo.Constraint(
        model.I, model.T, rule=lambda m, i, t: discharge_reduced_cost(m, i, int(t)) >= 0.0
    )
    model.soc_stationarity = pyo.Constraint(
        model.I, model.T_SOC, rule=lambda m, i, tau: soc_reduced_cost(m, i, int(tau)) >= 0.0
    )
    model.net_injection_stationarity = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: -m.LMP[n, t]
        + m.SystemPrice[t]
        + sum(
            data.ptdf[line, n]
            * (m.LineUpperDual[line, t] + m.LineLowerDual[line, t])
            for line in m.L
        )
        == 0.0,
    )
    model.demand_adjustment_stationarity = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: data.demand_adjustment_penalty_eur_per_mw2
        * m.DemandAdjustment[n, t]
        - m.LMP[n, t]
        == 0.0,
    )

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
        "relaxed_gen_lower",
        (model.G, model.T),
        lambda m, g, t: m.PGen[g, t] * gen_reduced_cost(m, g, int(t)),
    )
    add_relaxed_product(
        "relaxed_charge_lower",
        (model.I, model.T),
        lambda m, i, t: m.PCharge[i, t] * charge_reduced_cost(m, i, int(t)),
    )
    add_relaxed_product(
        "relaxed_discharge_lower",
        (model.I, model.T),
        lambda m, i, t: m.PDischarge[i, t] * discharge_reduced_cost(m, i, int(t)),
    )
    add_relaxed_product(
        "relaxed_soc_lower",
        (model.I, model.T_SOC),
        lambda m, i, tau: m.SOC[i, tau] * soc_reduced_cost(m, i, int(tau)),
    )
    add_relaxed_product(
        "relaxed_gen_upper",
        (model.G, model.T),
        lambda m, g, t: (generator_by_id[g].capacity_mw[int(t)] - m.PGen[g, t])
        * (-m.GenUpperDual[g, t]),
    )
    add_relaxed_product(
        "relaxed_line_upper",
        (model.L, model.T),
        lambda m, line, t: (data.line_limit_mw[line] - flow(m, line, int(t)))
        * (-m.LineUpperDual[line, t]),
    )
    add_relaxed_product(
        "relaxed_line_lower",
        (model.L, model.T),
        lambda m, line, t: (flow(m, line, int(t)) + data.line_limit_mw[line])
        * m.LineLowerDual[line, t],
    )
    add_relaxed_product(
        "relaxed_charge_offer_upper",
        (model.I, model.T),
        lambda m, i, t: (available_charge(i, int(t)) - m.PCharge[i, t])
        * (-m.ChargeOfferDual[i, t]),
    )
    add_relaxed_product(
        "relaxed_discharge_offer_upper",
        (model.I, model.T),
        lambda m, i, t: (available_discharge(i, int(t)) - m.PDischarge[i, t])
        * (-m.DischargeOfferDual[i, t]),
    )
    add_relaxed_product(
        "relaxed_inverter_upper",
        (model.I, model.T),
        lambda m, i, t: (
            installed_power(i) - m.PCharge[i, t] - m.PDischarge[i, t]
        )
        * (-m.InverterDual[i, t]),
    )
    add_relaxed_product(
        "relaxed_soc_upper",
        (model.I, model.T_SOC),
        lambda m, i, tau: (installed_energy(i) - m.SOC[i, tau])
        * (-m.SOCUpperDual[i, tau]),
    )

    model.primal_objective = pyo.Expression(
        expr=sum(
            generator_by_id[g].marginal_cost * model.PGen[g, t]
            + 0.5
            * generator_by_id[g].quadratic_cost_eur_per_mw2
            * model.PGen[g, t] ** 2
            for g in model.G
            for t in model.T
        )
        + sum(
            0.5
            * investor_by_id[i].degradation_eur_per_mwh
            * (model.PCharge[i, t] + model.PDischarge[i, t])
            + 0.5
            * investor_by_id[i].dispatch_quadratic_eur_per_mw2
            * (model.PCharge[i, t] ** 2 + model.PDischarge[i, t] ** 2)
            for i in model.I
            for t in model.T
        )
        + 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(model.DemandAdjustment[n, t] ** 2 for n in model.N for t in model.T)
    )
    model.dual_objective = pyo.Expression(
        expr=sum(data.demand_mw[n, t] * model.LMP[n, t] for n in model.N for t in model.T)
        + sum(
            generator_by_id[g].capacity_mw[int(t)] * model.GenUpperDual[g, t]
            for g in model.G
            for t in model.T
        )
        + sum(
            data.line_limit_mw[line]
            * (model.LineUpperDual[line, t] - model.LineLowerDual[line, t])
            for line in model.L
            for t in model.T
        )
        + sum(
            available_charge(i, int(t)) * model.ChargeOfferDual[i, t]
            + available_discharge(i, int(t)) * model.DischargeOfferDual[i, t]
            + installed_power(i) * model.InverterDual[i, t]
            for i in model.I
            for t in model.T
        )
        + sum(
            installed_energy(i) * model.SOCUpperDual[i, tau]
            for i in model.I
            for tau in model.T_SOC
        )
        - 0.5
        * sum(
            generator_by_id[g].quadratic_cost_eur_per_mw2 * model.PGen[g, t] ** 2
            for g in model.G
            for t in model.T
        )
        - 0.5
        * sum(
            investor_by_id[i].dispatch_quadratic_eur_per_mw2
            * (model.PCharge[i, t] ** 2 + model.PDischarge[i, t] ** 2)
            for i in model.I
            for t in model.T
        )
        - 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(model.DemandAdjustment[n, t] ** 2 for n in model.N for t in model.T)
    )

    model.storage_revenue = pyo.Expression(
        expr=sum(
            model.LMP[active.node, t]
            * (model.PDischarge[active_id, t] - model.PCharge[active_id, t])
            for t in model.T
        )
    )
    model.generation_rent = pyo.Expression(
        expr=sum(
            share
            * (model.LMP[generator_by_id[g].node, t] - generator_by_id[g].marginal_cost)
            * model.PGen[g, t]
            for g, share in active.owned_generation_shares.items()
            for t in model.T
        )
    )
    model.degradation_cost = pyo.Expression(
        expr=0.5
        * active.degradation_eur_per_mwh
        * sum(
            model.PCharge[active_id, t] + model.PDischarge[active_id, t]
            for t in model.T
        )
        + 0.5
        * active.dispatch_quadratic_eur_per_mw2
        * sum(
            model.PCharge[active_id, t] ** 2 + model.PDischarge[active_id, t] ** 2
            for t in model.T
        )
    )
    daily_crf = capital_recovery_factor(active.wacc, active.lifetime_years) / 365.25
    model.daily_capex = pyo.Expression(
        expr=daily_crf
        * (
            active.cost_power_eur_per_mw * model.PowerCapacity
            + active.cost_energy_eur_per_mwh * model.EnergyCapacity
        )
    )
    model.economic_profit = pyo.Expression(
        expr=model.storage_revenue
        + model.generation_rent
        - model.degradation_cost
        - model.daily_capex
    )
    # Availability is economically irrelevant whenever its bound is slack.
    # This tiny lexicographic reward selects full availability among otherwise
    # equivalent strategies and makes the Nash residual identifiable.
    model.availability_tie_breaker = pyo.Expression(
        expr=tie_breaker
        * sum(model.ChargeOffer[t] + model.DischargeOffer[t] for t in model.T)
    )
    model.objective = pyo.Objective(
        expr=model.economic_profit + model.availability_tie_breaker,
        sense=pyo.maximize,
    )
    model._toy_data = data
    model._active_id = active_id
    model._complementarity_epsilon = epsilon
    model._availability_tie_breaker = tie_breaker
    model._strategic_charge = bool(strategic_charge)
    model._endogenous_investment = bool(endogenous_investment)
    model._power_capacity_mw = dict(power)
    model._energy_capacity_mwh = dict(energy)
    model._relaxed_products = tuple(product_names)
    return model


def _seed(variable: pyo.Var, raw_value: float | None) -> None:
    value = 0.0 if raw_value is None else float(raw_value)
    if abs(value) < 1.0e-9:
        value = 0.0
    if variable.lb is not None:
        value = max(value, float(pyo.value(variable.lb)))
    if variable.ub is not None:
        value = min(value, float(pyo.value(variable.ub)))
    variable.set_value(value)


def initialise_best_response(
    model: pyo.ConcreteModel,
    data: ToyData,
    charge_profile: Profile,
    discharge_profile: Profile,
) -> pyo.ConcreteModel:
    """Seed lower-level primal and dual variables from the exact ISO LP."""

    active_id = model._active_id
    charge = dict(charge_profile)
    discharge = dict(discharge_profile)
    for time_ in data.times:
        charge[active_id, time_] = float(pyo.value(model.ChargeOffer[time_]))
        discharge[active_id, time_] = float(pyo.value(model.DischargeOffer[time_]))
    power = dict(model._power_capacity_mw)
    energy = dict(model._energy_capacity_mwh)
    power[active_id] = float(pyo.value(model.PowerCapacity))
    energy[active_id] = float(pyo.value(model.EnergyCapacity))
    lower = solve_market(
        data,
        charge,
        discharge,
        power_capacity_mw=power,
        energy_capacity_mwh=energy,
    )

    for generator in model.G:
        for time_ in model.T:
            key = generator, time_
            _seed(model.PGen[key], lower.PGen[key].value)
            _seed(model.GenUpperDual[key], lower.dual.get(lower.generation_upper[key], 0.0))
    for investor in model.I:
        for time_ in model.T:
            key = investor, time_
            _seed(model.PCharge[key], lower.PCharge[key].value)
            _seed(model.PDischarge[key], lower.PDischarge[key].value)
            _seed(model.ChargeOfferDual[key], lower.dual.get(lower.charge_offer_upper[key], 0.0))
            _seed(model.DischargeOfferDual[key], lower.dual.get(lower.discharge_offer_upper[key], 0.0))
            _seed(model.InverterDual[key], lower.dual.get(lower.shared_inverter_upper[key], 0.0))
            _seed(model.SOCDynamicDual[key], lower.dual.get(lower.soc_transition[key], 0.0))
        for tau in model.T_SOC:
            key = investor, tau
            _seed(model.SOC[key], lower.SOC[key].value)
            _seed(model.SOCUpperDual[key], lower.dual.get(lower.soc_upper[key], 0.0))
        _seed(model.SOCPeriodDual[investor], lower.dual.get(lower.soc_periodicity[investor], 0.0))
    for node in model.N:
        for time_ in model.T:
            key = node, time_
            _seed(model.NetInjection[key], lower.NetInjection[key].value)
            _seed(model.DemandAdjustment[key], lower.DemandAdjustment[key].value)
            _seed(model.LMP[key], lower.dual.get(lower.nodal_balance[key], 0.0))
    for time_ in model.T:
        _seed(model.SystemPrice[time_], lower.dual.get(lower.system_balance[time_], 0.0))
    for line in model.L:
        for time_ in model.T:
            key = line, time_
            _seed(model.LineUpperDual[key], lower.dual.get(lower.line_upper[key], 0.0))
            _seed(model.LineLowerDual[key], lower.dual.get(lower.line_lower[key], 0.0))
    return lower


def mpec_diagnostics(model: pyo.ConcreteModel) -> dict[str, float]:
    products = [
        float(pyo.value(getattr(model, name)[index]))
        for name in model._relaxed_products
        for index in getattr(model, name)
    ]
    epsilon = float(model._complementarity_epsilon)
    maximum = max(products, default=0.0)
    minimum = min(products, default=0.0)
    gap = float(pyo.value(model.primal_objective - model.dual_objective))
    return {
        "minimum_product": minimum,
        "maximum_product": maximum,
        "maximum_violation": max(0.0, maximum - epsilon, -minimum),
        "absolute_primal_dual_gap": abs(gap),
    }


def solve_best_response(
    data: ToyData,
    active_id: str,
    charge_profile: Profile,
    discharge_profile: Profile,
    *,
    power_capacity_mw: CapacityProfile | None = None,
    energy_capacity_mwh: CapacityProfile | None = None,
    endogenous_investment: bool = False,
    complementarity_epsilon: float = 1.0e-5,
    availability_tie_breaker_eur_per_mw: float = 1.0e-3,
    strategic_charge: bool = False,
    max_solve_seconds: float = 60.0,
    tee: bool = False,
) -> BestResponse:
    model = build_best_response_mpec(
        data,
        active_id,
        charge_profile,
        discharge_profile,
        power_capacity_mw=power_capacity_mw,
        energy_capacity_mwh=energy_capacity_mwh,
        endogenous_investment=endogenous_investment,
        complementarity_epsilon=complementarity_epsilon,
        availability_tie_breaker_eur_per_mw=availability_tie_breaker_eur_per_mw,
        strategic_charge=strategic_charge,
    )
    initialise_best_response(model, data, charge_profile, discharge_profile)
    executable = _ipopt_executable()
    if executable is None:
        raise RuntimeError(
            "IPOPT was not found. Set IPOPT_EXECUTABLE or install it under LOCALAPPDATA/idaes/bin."
        )
    solver = pyo.SolverFactory("ipopt", solver_io="nl", executable=str(executable))
    solver.options.update(
        {
            "linear_solver": "ma57",
            "tol": 1.0e-7,
            "constr_viol_tol": 1.0e-7,
            "acceptable_tol": 1.0e-6,
            "max_iter": 3000,
            "max_cpu_time": float(max_solve_seconds),
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "print_level": 5 if tee else 0,
        }
    )
    started = time.perf_counter()
    try:
        result = solver.solve(model, tee=tee)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        power, energy = _capacity_profiles(data, power_capacity_mw, energy_capacity_mwh)
        current_market = solve_market(
            data,
            charge_profile,
            discharge_profile,
            power_capacity_mw=power,
            energy_capacity_mwh=energy,
        )
        return BestResponse(
            investor_id=active_id,
            optimal=False,
            termination=f"error: {exc}",
            solve_seconds=elapsed,
            charge_offer_mw={t: float(charge_profile[active_id, t]) for t in data.times},
            discharge_offer_mw={t: float(discharge_profile[active_id, t]) for t in data.times},
            profit_eur_per_day=market_profit(current_market, active_id),
            recleared_profit_eur_per_day=market_profit(current_market, active_id),
            embedded_reclear_profit_gap_eur_per_day=float("nan"),
            maximum_lmp_reclear_gap_eur_per_mwh=float("nan"),
            maximum_complementarity_product=float("nan"),
            maximum_complementarity_violation=float("nan"),
            absolute_primal_dual_gap_eur_per_day=float("nan"),
            power_mw=power[active_id],
            energy_mwh=energy[active_id],
        )
    elapsed = time.perf_counter() - started
    termination = str(result.solver.termination_condition)
    optimal = result.solver.termination_condition in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
    }
    diagnostics = mpec_diagnostics(model)
    candidate_charge = dict(charge_profile)
    candidate_discharge = dict(discharge_profile)
    candidate_power, candidate_energy = _capacity_profiles(
        data, power_capacity_mw, energy_capacity_mwh
    )
    active = next(investor for investor in data.investors if investor.investor_id == active_id)
    candidate_power[active_id] = max(0.0, float(pyo.value(model.PowerCapacity)))
    candidate_energy[active_id] = min(
        active.duration_max_hours * candidate_power[active_id],
        max(
            active.duration_min_hours * candidate_power[active_id],
            float(pyo.value(model.EnergyCapacity)),
        ),
    )
    for time_ in data.times:
        candidate_charge[active_id, time_] = min(
            candidate_power[active_id],
            max(0.0, float(pyo.value(model.ChargeOffer[time_]))),
        )
        candidate_discharge[active_id, time_] = min(
            candidate_power[active_id],
            max(0.0, float(pyo.value(model.DischargeOffer[time_]))),
        )
    recleared = solve_market(
        data,
        candidate_charge,
        candidate_discharge,
        power_capacity_mw=candidate_power,
        energy_capacity_mwh=candidate_energy,
    )
    embedded_profit = float(pyo.value(model.economic_profit))
    recleared_profit = market_profit(recleared, active_id)
    maximum_lmp_gap = max(
        abs(
            float(pyo.value(model.LMP[node, time_]))
            - float(recleared.dual[recleared.nodal_balance[node, time_]])
        )
        for node in data.nodes
        for time_ in data.times
    )
    return BestResponse(
        investor_id=active_id,
        optimal=optimal,
        termination=termination,
        solve_seconds=elapsed,
        charge_offer_mw={t: candidate_charge[active_id, t] for t in data.times},
        discharge_offer_mw={t: candidate_discharge[active_id, t] for t in data.times},
        profit_eur_per_day=embedded_profit,
        recleared_profit_eur_per_day=recleared_profit,
        embedded_reclear_profit_gap_eur_per_day=abs(embedded_profit - recleared_profit),
        maximum_lmp_reclear_gap_eur_per_mwh=maximum_lmp_gap,
        maximum_complementarity_product=diagnostics["maximum_product"],
        maximum_complementarity_violation=diagnostics["maximum_violation"],
        absolute_primal_dual_gap_eur_per_day=diagnostics["absolute_primal_dual_gap"],
        power_mw=candidate_power[active_id],
        energy_mwh=candidate_energy[active_id],
    )


def market_profit(model: pyo.ConcreteModel, investor_id: str) -> float:
    data: ToyData = model._toy_data
    investor_by_id = {investor.investor_id: investor for investor in data.investors}
    generator_by_id = {generator.generator_id: generator for generator in data.generators}
    investor = investor_by_id[investor_id]
    power = getattr(
        model,
        "_power_capacity_mw",
        {item.investor_id: item.power_mw for item in data.investors},
    )
    energy = getattr(
        model,
        "_energy_capacity_mwh",
        {item.investor_id: item.energy_mwh for item in data.investors},
    )
    storage_revenue = sum(
        float(model.dual[model.nodal_balance[investor.node, t]])
        * (float(pyo.value(model.PDischarge[investor_id, t])) - float(pyo.value(model.PCharge[investor_id, t])))
        for t in data.times
    )
    generation_rent = sum(
        share
        * (
            float(model.dual[model.nodal_balance[generator_by_id[g].node, t]])
            - generator_by_id[g].marginal_cost
        )
        * float(pyo.value(model.PGen[g, t]))
        for g, share in investor.owned_generation_shares.items()
        for t in data.times
    )
    degradation = 0.5 * investor.degradation_eur_per_mwh * sum(
        float(pyo.value(model.PCharge[investor_id, t]))
        + float(pyo.value(model.PDischarge[investor_id, t]))
        for t in data.times
    )
    degradation += 0.5 * investor.dispatch_quadratic_eur_per_mw2 * sum(
        float(pyo.value(model.PCharge[investor_id, t])) ** 2
        + float(pyo.value(model.PDischarge[investor_id, t])) ** 2
        for t in data.times
    )
    daily_crf = capital_recovery_factor(investor.wacc, investor.lifetime_years) / 365.25
    capex = daily_crf * (
        investor.cost_power_eur_per_mw * power[investor_id]
        + investor.cost_energy_eur_per_mwh * energy[investor_id]
    )
    return storage_revenue + generation_rent - degradation - capex
