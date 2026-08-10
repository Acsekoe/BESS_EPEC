"""Shared data and solver helpers for the isolated Tikhonov experiment."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import pyomo.environ as pyo


MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODEL_DIR.parent
PRIMAL_DUAL_DIR = MODEL_DIR / "Primal and dual problems"
for import_dir in (MODEL_DIR, PRIMAL_DUAL_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from ieee9_strategic_operation_mpec import apply_generator_calibration
from primal_market_clearing_model import MarketData, load_market_data
from solver_utils import get_ipopt_solver


DEFAULT_DATA_PATH = (
    MODEL_DIR
    / "data"
    / "processed"
    / "market_data_IEEE_9Bus_distributed_congestion.json"
)


def load_fixed_storage_case(
    data_path: Path,
    *,
    storage_id: str,
    power_mw_by_node: Mapping[str, float],
    duration_hours: float,
) -> MarketData:
    """Load the active calibration and install one fixed-capacity storage unit."""

    if duration_hours <= 0.0:
        raise ValueError("duration_hours must be positive.")
    data = load_calibrated_case(data_path)
    unknown = set(power_mw_by_node) - set(data.nodes)
    if unknown:
        raise ValueError(f"Unknown storage nodes: {sorted(unknown)}")
    nodal_power = {
        node: max(0.0, float(power_mw_by_node.get(node, 0.0)))
        for node in data.nodes
    }
    return replace(
        data,
        storage_units=[storage_id],
        x_power={(storage_id, node): nodal_power[node] for node in data.nodes},
        x_energy={
            (storage_id, node): duration_hours * nodal_power[node]
            for node in data.nodes
        },
    )


def load_calibrated_case(data_path: Path) -> MarketData:
    """Load the active three-machine calibration without an artificial peaker."""

    base = load_market_data(Path(data_path))
    data, _ = apply_generator_calibration(
        base,
        conventional_capacity_adder_mw=0.0,
        peaker_node=None,
        peaker_capacity_mw=0.0,
        peaker_cost_eur_per_mwh=95.0,
    )
    return data


def solve_ipopt(
    model: pyo.ConcreteModel,
    *,
    solver_tol: float,
    max_cpu_time: float,
    tee: bool,
) -> pyo.SolverResults:
    if solver_tol <= 0.0:
        raise ValueError("solver_tol must be positive.")
    if max_cpu_time <= 0.0:
        raise ValueError("max_cpu_time must be positive.")
    return get_ipopt_solver(
        {
            "tol": solver_tol,
            "acceptable_tol": solver_tol,
            "max_cpu_time": max_cpu_time,
        }
    ).solve(model, tee=tee)


def sparse_pairs(data: MarketData, tolerance_mw: float) -> list[tuple[str, str]]:
    if tolerance_mw < 0.0:
        raise ValueError("tolerance_mw must be non-negative.")
    return [
        (unit, node)
        for unit in data.storage_units
        for node in data.nodes
        if data.x_power[unit, node] > tolerance_mw
    ]


def generation_pairs(data: MarketData) -> list[tuple[str, int]]:
    return [
        (generator, int(time))
        for generator in data.generators
        for time in data.times
        if data.generation_capacity[generator, time] > 1.0e-8
    ]


def generator_nodes(data: MarketData) -> dict[str, list[str]]:
    result = {generator: [] for generator in data.generators}
    for node in data.nodes:
        for generator in data.generators_at_node.get(node, []):
            result.setdefault(generator, []).append(node)
    return result
