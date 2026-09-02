"""Exact competitive market reclear for one frozen strategy profile.

Clears the ISO problem (Ipopt, high accuracy) for fixed capacities, storage
bids/offers, and submitted generation offers, then reports dispatch, prices,
congestion, curtailment, and per-investor profit decomposition using TRUE
economic costs. This is the ground truth against which MPEC-claimed profits
must be validated; large MPEC profits without matching exact-reclear effects
are numerical or price-selection artefacts.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pyomo.environ as pyo

from investors import InvestorConfig, capital_recovery_factor
from primal_market_clearing_model import (
    MarketData,
    build_primal_market_clearing_model,
    effective_generation_offer,
)


def _ipopt(tolerance: float) -> pyo.SolverFactory:
    candidates = []
    if os.environ.get("IPOPT_EXECUTABLE"):
        candidates.append(Path(os.environ["IPOPT_EXECUTABLE"]))
    if shutil.which("ipopt"):
        candidates.append(Path(shutil.which("ipopt") or ""))
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "idaes" / "bin" / "ipopt.exe")
    executable = next((c for c in candidates if c.is_file()), None)
    kwargs = {"solver_io": "nl"}
    if executable is not None:
        kwargs["executable"] = str(executable)
    solver = pyo.SolverFactory("ipopt", **kwargs)
    solver.options.update(
        {
            "linear_solver": "ma57",
            "tol": tolerance,
            "acceptable_tol": 10.0 * tolerance,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "print_level": 0,
        }
    )
    return solver


@dataclass(frozen=True)
class ReclearResult:
    model: pyo.ConcreteModel
    data: MarketData
    generation_offer: Mapping[tuple[str, int], float]
    charge_bid: Mapping[tuple[str, str, int], float]
    discharge_offer: Mapping[tuple[str, str, int], float]
    power: Mapping[tuple[str, str], float]
    energy: Mapping[tuple[str, str], float]

    def lmp(self, node: str, time: int) -> float:
        return float(self.model.dual[self.model.nodal_balance[node, time]])


def clear(
    data: MarketData,
    *,
    investor_ids: Sequence[str],
    power: Mapping[tuple[str, str], float],
    energy: Mapping[tuple[str, str], float],
    charge_bid: Mapping[tuple[str, str, int], float],
    discharge_offer: Mapping[tuple[str, str, int], float],
    generation_offer: Mapping[tuple[str, int], float] | None = None,
    inverter_limit: str = "shared",
    tolerance: float = 1.0e-9,
) -> ReclearResult:
    """Solve the exact ISO clearing for one frozen strategy profile."""

    units = list(investor_ids)
    fixed = replace(
        data,
        storage_units=units,
        x_power={(i, n): float(power[i, n]) for i in units for n in data.nodes},
        x_energy={(i, n): float(energy[i, n]) for i in units for n in data.nodes},
    )
    model = build_primal_market_clearing_model(fixed, include_load_shed=False)

    if inverter_limit == "shared":
        model.del_component(model.charge_power_bound)
        model.del_component(model.discharge_power_bound)
        model.shared_inverter_bound = pyo.Constraint(
            model.I,
            model.N,
            model.T,
            rule=lambda m, i, n, t: m.P_charge[i, n, t] + m.P_discharge[i, n, t]
            <= fixed.x_power[i, n],
        )
    elif inverter_limit != "separate":
        raise ValueError("inverter_limit must be 'shared' or 'separate'.")

    offers = dict(generation_offer or {})

    def submitted(generator: str, time: int) -> float:
        return float(
            offers.get((generator, int(time)), effective_generation_offer(data, generator))
        )

    model.objective.set_value(
        sum(
            submitted(g, int(t)) * model.P_gen[g, t]
            for g in model.G
            for t in model.T
        )
        + sum(
            float(discharge_offer[i, n, int(t)]) * model.P_discharge[i, n, t]
            - float(charge_bid[i, n, int(t)]) * model.P_charge[i, n, t]
            for i in model.I
            for n in model.N
            for t in model.T
        )
        + 0.5
        * data.demand_adjustment_penalty_eur_per_mw2
        * sum(model.DemandAdjustment[n, t] ** 2 for n in model.N for t in model.T)
    )

    result = _ipopt(tolerance).solve(model, tee=False)
    termination = result.solver.termination_condition
    if termination != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Exact reclear failed: {termination}")
    return ReclearResult(
        model=model,
        data=data,
        generation_offer=offers,
        charge_bid=dict(charge_bid),
        discharge_offer=dict(discharge_offer),
        power=dict(power),
        energy=dict(energy),
    )


def _generator_node(data: MarketData, generator: str) -> str:
    for node, units in data.generators_at_node.items():
        if generator in units:
            return node
    raise KeyError(generator)


def _owner(investors: Sequence[InvestorConfig], generator: str) -> str:
    for investor in investors:
        if investor.owned_generation_shares.get(generator, 0.0) > 0.0:
            return investor.investor_id
    return ""


def generation_rows(
    result: ReclearResult, investors: Sequence[InvestorConfig]
) -> list[dict[str, object]]:
    data, model = result.data, result.model
    rows = []
    for g in data.generators:
        node = _generator_node(data, g)
        for t in data.times:
            capacity = float(data.generation_capacity[g, int(t)])
            dispatch = float(pyo.value(model.P_gen[g, t]))
            rows.append(
                {
                    "generator": g,
                    "owner": _owner(investors, g),
                    "node": node,
                    "time": int(t),
                    "true_cost_eur_per_mwh": float(data.generation_cost[g]),
                    "submitted_offer_eur_per_mwh": float(
                        result.generation_offer.get(
                            (g, int(t)), effective_generation_offer(data, g)
                        )
                    ),
                    "capacity_mw": capacity,
                    "dispatch_mw": dispatch,
                    "curtailment_mw": max(0.0, capacity - dispatch),
                    "lmp_eur_per_mwh": result.lmp(node, int(t)),
                }
            )
    return rows


def storage_rows(result: ReclearResult) -> list[dict[str, object]]:
    model = result.model
    rows = []
    for i in model.I:
        for n in model.N:
            for t in model.T:
                rows.append(
                    {
                        "investor": str(i),
                        "node": str(n),
                        "time": int(t),
                        "power_mw": float(result.power[str(i), str(n)]),
                        "charge_mw": float(pyo.value(model.P_charge[i, n, t])),
                        "discharge_mw": float(pyo.value(model.P_discharge[i, n, t])),
                        "soc_mwh": float(pyo.value(model.SOC[i, n, t])),
                        "charge_bid_eur_per_mwh": float(result.charge_bid[str(i), str(n), int(t)]),
                        "discharge_offer_eur_per_mwh": float(
                            result.discharge_offer[str(i), str(n), int(t)]
                        ),
                        "lmp_eur_per_mwh": result.lmp(str(n), int(t)),
                    }
                )
    return rows


def price_rows(result: ReclearResult) -> list[dict[str, object]]:
    data, model = result.data, result.model
    rows = []
    for n in data.nodes:
        for t in data.times:
            rows.append(
                {
                    "node": n,
                    "time": int(t),
                    "lmp_eur_per_mwh": result.lmp(n, int(t)),
                    "demand_adjustment_mw": float(
                        pyo.value(model.DemandAdjustment[n, t])
                    ),
                }
            )
    return rows


def line_rows(result: ReclearResult) -> list[dict[str, object]]:
    data, model = result.data, result.model
    rows = []
    for l in data.lines:
        for t in data.times:
            flow = sum(
                data.ptdf[l, n] * float(pyo.value(model.NetInjection[n, t]))
                for n in data.nodes
            )
            rows.append(
                {
                    "line": l,
                    "time": int(t),
                    "flow_mw": flow,
                    "limit_mw": float(data.line_limit[l]),
                    "utilisation": flow / float(data.line_limit[l]),
                    "congestion_dual_eur_per_mwh": float(
                        model.dual[model.line_upper_bound[l, t]]
                        + model.dual[model.line_lower_bound[l, t]]
                    ),
                }
            )
    return rows


def profit_decomposition(
    result: ReclearResult,
    investors: Sequence[InvestorConfig],
) -> list[dict[str, object]]:
    """True-cost profit decomposition per investor at the recleared dispatch."""

    data, model = result.data, result.model
    rows = []
    for investor in investors:
        active = investor.investor_id
        spot = sum(
            result.lmp(n, int(t))
            * (
                float(pyo.value(model.P_discharge[active, n, t]))
                - float(pyo.value(model.P_charge[active, n, t]))
            )
            for n in data.nodes
            for t in data.times
        )
        degradation = 0.5 * investor.degradation_eur_per_mwh * sum(
            float(pyo.value(model.P_charge[active, n, t]))
            + float(pyo.value(model.P_discharge[active, n, t]))
            for n in data.nodes
            for t in data.times
        )
        rent_wind = rent_pv = rent_other = 0.0
        for g, share in investor.owned_generation_shares.items():
            if share <= 0.0:
                continue
            node = _generator_node(data, g)
            rent = share * sum(
                (result.lmp(node, int(t)) - data.generation_cost[g])
                * float(pyo.value(model.P_gen[g, t]))
                for t in data.times
                if data.generation_capacity[g, int(t)] > 1e-8
            )
            if "Wind" in g:
                rent_wind += rent
            elif "PV" in g:
                rent_pv += rent
            else:
                rent_other += rent
        crf_daily = (
            capital_recovery_factor(investor.wacc, investor.lifetime_years) / 365.25
        )
        capex = crf_daily * sum(
            investor.cost_power_eur_per_mw * float(result.power[active, n])
            + investor.cost_energy_eur_per_mwh * float(result.energy[active, n])
            for n in data.nodes
        )
        renewable_curtailment = sum(
            share
            * max(
                0.0,
                float(data.generation_capacity[g, int(t)])
                - float(pyo.value(model.P_gen[g, t])),
            )
            for g, share in investor.owned_generation_shares.items()
            if share > 0.0 and ("Wind" in g or "PV" in g)
            for t in data.times
            if data.generation_capacity[g, int(t)] > 1e-8
        )
        rows.append(
            {
                "investor": active,
                "storage_spot_revenue_eur_per_day": spot,
                "generation_rent_wind_eur_per_day": rent_wind,
                "generation_rent_pv_eur_per_day": rent_pv,
                "generation_rent_other_eur_per_day": rent_other,
                "degradation_cost_eur_per_day": degradation,
                "daily_capex_eur_per_day": capex,
                "renewable_curtailment_mwh": renewable_curtailment,
                "profit_eur_per_day": spot
                + rent_wind
                + rent_pv
                + rent_other
                - degradation
                - capex,
            }
        )
    return rows
