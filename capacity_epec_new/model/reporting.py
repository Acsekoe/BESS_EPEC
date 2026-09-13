"""Turning a run into files on disk.

Nothing here decides anything: it only serialises what `capacity_game`
produced.  The one rule it enforces is that a non-finite number is written as
JSON ``null`` rather than the non-standard ``NaN``, so an invalid result stays
visibly invalid instead of being silently parsed as a value.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyomo.environ as pyo

import iso_market
from capacity_game import Capacities, GameConfig, GameState
from investors import InvestorConfig
from market_data import MarketData


def json_dumps(value: object, **kwargs: object) -> str:
    """Standards-compliant JSON in which invalid numbers become null."""

    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        return item

    return json.dumps(clean(value), allow_nan=False, **kwargs)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json_dumps(payload, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _duration(power_mw: float, energy_mwh: float) -> float | None:
    return energy_mwh / power_mw if power_mw > 1.0e-9 else None


def capacity_rows(
    data: MarketData,
    investors: Iterable[InvestorConfig],
    power: Capacities,
    energy: Capacities,
    sweep: int | None = None,
) -> list[dict[str, object]]:
    """One row per investor and node, with its share of that node."""

    investors = tuple(investors)
    nodal_total = {
        node: sum(power[i.investor_id, node] for i in investors) for node in data.nodes
    }
    rows = []
    for investor in investors:
        for node in data.nodes:
            key = investor.investor_id, node
            row: dict[str, object] = {} if sweep is None else {"sweep": sweep}
            row.update(
                investor=investor.investor_id,
                node=node,
                power_mw=power[key],
                energy_mwh=energy[key],
                duration_hours=_duration(power[key], energy[key]),
                nodal_total_power_mw=nodal_total[node],
                investor_nodal_power_share=(
                    power[key] / nodal_total[node] if nodal_total[node] > 1.0e-9 else None
                ),
            )
            rows.append(row)
    return rows


def capacity_total_rows(
    data: MarketData,
    investors: Iterable[InvestorConfig],
    power: Capacities,
    energy: Capacities,
    sweep: int | None = None,
) -> list[dict[str, object]]:
    """One row per investor: the whole portfolio and its share of the system."""

    investors = tuple(investors)
    system_power = sum(power.values())
    rows = []
    for investor in investors:
        unit = investor.investor_id
        total_power = sum(power[unit, node] for node in data.nodes)
        total_energy = sum(energy[unit, node] for node in data.nodes)
        row: dict[str, object] = {} if sweep is None else {"sweep": sweep}
        row.update(
            investor=unit,
            total_power_mw=total_power,
            total_energy_mwh=total_energy,
            portfolio_duration_hours=_duration(total_power, total_energy),
            investor_system_power_share=(
                total_power / system_power if system_power > 1.0e-9 else None
            ),
        )
        rows.append(row)
    return rows


def market_rows(
    market: pyo.ConcreteModel, data: MarketData
) -> list[dict[str, object]]:
    """The cleared market hour by hour and node by node."""

    return [
        {
            "time": time,
            "node": node,
            "demand_mw": data.demand_el[node, time],
            "lmp_eur_per_mwh": iso_market.lmp(market, node, time),
            "generation_mw": sum(
                float(pyo.value(market.P_gen[generator, time]))
                for generator in data.generators_at_node.get(node, [])
            ),
            "charge_mw": sum(
                float(pyo.value(market.P_charge[unit, node, time])) for unit in market.I
            ),
            "discharge_mw": sum(
                float(pyo.value(market.P_discharge[unit, node, time])) for unit in market.I
            ),
            "net_injection_mw": float(pyo.value(market.NetInjection[node, time])),
            "demand_adjustment_mw": float(pyo.value(market.DemandAdjustment[node, time])),
        }
        for time in data.times
        for node in data.nodes
    ]


def checkpoint(state: GameState, config: GameConfig) -> dict[str, object]:
    """Full-precision state, enough to reconstruct any sweep exactly."""

    return {
        "format_version": 3,
        "market_design": config.market_design,
        "formulation": f"capacity-only-{config.lower_level}",
        "sweep": state.sweep,
        "iteration_converged": state.converged,
        "stable_sweeps": state.stable_sweeps,
        "power": [
            {"investor": investor, "node": node, "mw": value}
            for (investor, node), value in sorted(state.power.items())
        ],
        "energy": [
            {"investor": investor, "node": node, "mwh": value}
            for (investor, node), value in sorted(state.energy.items())
        ],
    }


def run_config(
    config: GameConfig,
    data_path: Path,
    data_sha256: str,
    *,
    demand_adjustment_penalty_eur_per_mw2: float | None = None,
) -> dict[str, object]:
    """Everything needed to reproduce the run, including the input hash.

    ``demand_adjustment_penalty_eur_per_mw2`` is the *effective* value after any
    CLI override.  It has to be stated separately because it lives in the market
    input rather than in GameConfig, so the file hash alone no longer identifies
    the market once the value can be overridden.
    """

    settings = asdict(config)
    settings.pop("investors")
    settings["solver"].pop("executable", None)
    return {
        "formulation": f"capacity-only-{config.lower_level}",
        "strategic_variables": ["nodal_power_capacity_mw", "nodal_energy_capacity_mwh"],
        "operational_bidding": False,
        "data": str(data_path.resolve()),
        "data_sha256": data_sha256,
        "demand_adjustment_penalty_eur_per_mw2": demand_adjustment_penalty_eur_per_mw2,
        "shared_node_cap_enforced": config.shared_node_cap_enforced,
        **settings,
        "investors": [
            {**asdict(investor), "owned_generation_shares": dict(investor.owned_generation_shares)}
            for investor in config.investors
        ],
    }

