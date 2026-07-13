"""Prepare benchmark input data for the primal spot-market clearing model.

The Excel workbook is the human-editable assumption file. This script validates
it, computes derived network data such as PTDF values, and writes a compact JSON
file consumed by ``primal_market_clearing_model.py``.

Default run:
    python prepare_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "input" / "bess_epec_inputs.xlsx"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "market_data.json"
DEFAULT_PTDF_OUTPUT = SCRIPT_DIR / "data" / "processed" / "ptdf.csv"
TOL = 1e-9

# Tiered price-elastic demand (stepwise demand bid curve) is an experiment
# feature and is NOT written to the baseline market_data.json. It is only
# emitted when the workbook has an explicit optional "demand_tiers" sheet
# (columns: tier, share, wtp_eur_per_mwh). Experiment tier schedules live in
# data/processed/market_data_experiment.json, which is maintained by hand.


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}.") from exc


def _require_columns(df: pd.DataFrame, sheet: str, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{sheet}' is missing columns: {missing}")


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")


def read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    required_sheets = [
        "settings",
        "nodes",
        "transmission",
        "generators",
        "demand_el",
        "res_generation",
        "storage",
    ]
    workbook = pd.read_excel(path, sheet_name=None)
    missing = [sheet for sheet in required_sheets if sheet not in workbook]
    if missing:
        raise ValueError(f"Workbook is missing required sheets: {missing}")
    return {name: df for name, df in workbook.items()}


def settings_dict(settings: pd.DataFrame) -> dict[str, float]:
    _require_columns(settings, "settings", ["parameter", "value"])
    result: dict[str, float] = {}
    for row in _records(settings):
        result[str(row["parameter"])] = _as_float(row["value"], str(row["parameter"]))
    return result


def demand_tier_records(tables: dict[str, pd.DataFrame], voll: float) -> list[dict[str, Any]] | None:
    """Build the tiered demand schedule from the optional "demand_tiers" sheet.

    Returns None when the sheet is absent, so the baseline output carries no
    demand_tiers key and downstream models fall back to a single VOLL tier.
    """

    if "demand_tiers" not in tables:
        return None
    df = tables["demand_tiers"]
    _require_columns(df, "demand_tiers", ["tier", "share", "wtp_eur_per_mwh"])
    tiers = [
        {
            "tier": str(row["tier"]),
            "share": _as_float(row["share"], f"share for tier {row['tier']}"),
            "wtp_eur_per_mwh": _as_float(row["wtp_eur_per_mwh"], f"wtp_eur_per_mwh for tier {row['tier']}"),
        }
        for row in _records(df)
    ]

    names = [tier["tier"] for tier in tiers]
    if len(set(names)) != len(names):
        raise ValueError("Demand tier names must be unique.")
    for tier in tiers:
        if tier["share"] <= 0.0:
            raise ValueError(f"Demand tier {tier['tier']!r} must have positive share.")
        if tier["wtp_eur_per_mwh"] < 0.0:
            raise ValueError(f"Demand tier {tier['tier']!r} must have non-negative willingness-to-pay.")
        if tier["wtp_eur_per_mwh"] > voll + TOL:
            raise ValueError(f"Demand tier {tier['tier']!r} willingness-to-pay exceeds VOLL.")

    total_share = sum(tier["share"] for tier in tiers)
    if total_share > 1.0 + 1e-6:
        raise ValueError(f"Demand tier shares sum to {total_share}, which exceeds 1.")
    if total_share < 1.0 - 1e-6:
        tiers.append({"tier": "T_VOLL", "share": 1.0 - total_share, "wtp_eur_per_mwh": voll})
    return tiers


def calculate_ptdf(nodes: list[str], transmission: pd.DataFrame, slack_node: str) -> dict[tuple[str, str], float]:
    if slack_node not in nodes:
        raise ValueError(f"Slack node {slack_node!r} is not in nodes.")

    node_map = {node: idx for idx, node in enumerate(nodes)}
    slack_idx = node_map[slack_node]
    b_matrix = np.zeros((len(nodes), len(nodes)))

    for row in _records(transmission):
        from_node = str(row["from_node"])
        to_node = str(row["to_node"])
        if from_node not in node_map or to_node not in node_map:
            raise ValueError(f"Line {row['line']!r} references an unknown node.")
        reactance = _as_float(row["reactance_x"], f"reactance_x for line {row['line']}")
        if reactance <= 0:
            raise ValueError(f"Line {row['line']!r} must have positive reactance_x.")

        i = node_map[from_node]
        j = node_map[to_node]
        susceptance = 1.0 / reactance
        b_matrix[i, j] -= susceptance
        b_matrix[j, i] -= susceptance
        b_matrix[i, i] += susceptance
        b_matrix[j, j] += susceptance

    non_slack = [idx for idx in range(len(nodes)) if idx != slack_idx]
    reduced = b_matrix[np.ix_(non_slack, non_slack)]
    try:
        inverse_reduced = np.linalg.inv(reduced)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Transmission network B matrix is singular.") from exc

    x_bus = np.zeros((len(nodes), len(nodes)))
    for row_idx, node_i in enumerate(non_slack):
        for col_idx, node_j in enumerate(non_slack):
            x_bus[node_i, node_j] = inverse_reduced[row_idx, col_idx]

    ptdf: dict[tuple[str, str], float] = {}
    for row in _records(transmission):
        line = str(row["line"])
        i = node_map[str(row["from_node"])]
        j = node_map[str(row["to_node"])]
        susceptance = 1.0 / _as_float(row["reactance_x"], f"reactance_x for line {line}")
        for node in nodes:
            n = node_map[node]
            ptdf[(line, node)] = float(susceptance * (x_bus[i, n] - x_bus[j, n]))
    return ptdf


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    _require_columns(tables["nodes"], "nodes", ["node", "load_share", "shared_bess_limit_mw"])
    _require_columns(tables["transmission"], "transmission", ["line", "from_node", "to_node", "reactance_x", "limit_mw"])
    _require_columns(tables["generators"], "generators", ["generator", "node", "mc_eur_per_mwh", "pmax_mw"])
    _require_columns(tables["demand_el"], "demand_el", ["node", "hour", "demand_mw"])
    _require_columns(tables["res_generation"], "res_generation", ["node", "hour", "wind_mw", "pv_mw"])
    _require_columns(tables["storage"], "storage", ["storage_unit", "node", "x_power_mw", "x_energy_mwh"])


def build_processed_data(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    validate_tables(tables)
    settings = settings_dict(tables["settings"])

    nodes = [str(node) for node in tables["nodes"]["node"].tolist()]
    times = list(range(1, 25))
    soc_times = list(range(0, 25))
    lines = [str(line) for line in tables["transmission"]["line"].tolist()]

    if len(set(nodes)) != len(nodes):
        raise ValueError("Node names must be unique.")

    demand_records = tables["demand_el"].copy()
    demand_records["hour"] = demand_records["hour"].astype(int)
    demand_index = {(str(row["node"]), int(row["hour"])): _as_float(row["demand_mw"], "demand_mw") for row in _records(demand_records)}
    missing_demand = [(node, hour) for node in nodes for hour in times if (node, hour) not in demand_index]
    if missing_demand:
        raise ValueError(f"Demand data is incomplete. First missing pairs: {missing_demand[:10]}")

    res_records = tables["res_generation"].copy()
    res_records["hour"] = res_records["hour"].astype(int)
    res_index = {
        (str(row["node"]), int(row["hour"])): (
            _as_float(row["wind_mw"], "wind_mw"),
            _as_float(row["pv_mw"], "pv_mw"),
        )
        for row in _records(res_records)
    }
    missing_res = [(node, hour) for node in nodes for hour in times if (node, hour) not in res_index]
    if missing_res:
        raise ValueError(f"RES data is incomplete. First missing pairs: {missing_res[:10]}")

    generators: list[str] = []
    generation_cost: dict[str, float] = {}
    generation_capacity_records: list[dict[str, Any]] = []
    generators_at_node: dict[str, list[str]] = {node: [] for node in nodes}

    for row in _records(tables["generators"]):
        generator = str(row["generator"])
        node = str(row["node"])
        if node not in nodes:
            raise ValueError(f"Generator {generator!r} references unknown node {node!r}.")
        generators.append(generator)
        generators_at_node[node].append(generator)
        generation_cost[generator] = _as_float(row["mc_eur_per_mwh"], f"mc_eur_per_mwh for {generator}")
        pmax = _as_float(row["pmax_mw"], f"pmax_mw for {generator}")
        for hour in times:
            generation_capacity_records.append({"generator": generator, "hour": hour, "capacity_mw": pmax})

    for node in nodes:
        wind_total = sum(res_index[node, hour][0] for hour in times)
        pv_total = sum(res_index[node, hour][1] for hour in times)
        if wind_total > TOL:
            generator = f"RES_Wind_{node}"
            generators.append(generator)
            generators_at_node[node].append(generator)
            generation_cost[generator] = 0.0
            for hour in times:
                generation_capacity_records.append({"generator": generator, "hour": hour, "capacity_mw": res_index[node, hour][0]})
        if pv_total > TOL:
            generator = f"RES_PV_{node}"
            generators.append(generator)
            generators_at_node[node].append(generator)
            generation_cost[generator] = 0.0
            for hour in times:
                generation_capacity_records.append({"generator": generator, "hour": hour, "capacity_mw": res_index[node, hour][1]})

    if len(set(generators)) != len(generators):
        raise ValueError("Generator names must be unique, including generated RES names.")

    storage_units = sorted(str(unit) for unit in tables["storage"]["storage_unit"].unique())
    storage_index = {(str(row["storage_unit"]), str(row["node"])): row for row in _records(tables["storage"])}
    missing_storage = [(unit, node) for unit in storage_units for node in nodes if (unit, node) not in storage_index]
    if missing_storage:
        raise ValueError(f"Storage data is incomplete. First missing pairs: {missing_storage[:10]}")

    x_power_records: list[dict[str, Any]] = []
    x_energy_records: list[dict[str, Any]] = []
    for unit in storage_units:
        for node in nodes:
            row = storage_index[unit, node]
            power = _as_float(row["x_power_mw"], f"x_power_mw for {unit}/{node}")
            energy = _as_float(row["x_energy_mwh"], f"x_energy_mwh for {unit}/{node}")
            if min(power, energy) < -TOL:
                raise ValueError(f"Storage values must be non-negative for {unit}/{node}.")
            x_power_records.append({"storage_unit": unit, "node": node, "power_mw": power})
            x_energy_records.append({"storage_unit": unit, "node": node, "energy_mwh": energy})

    slack_node = str(settings.get("slack_node", "N3"))
    ptdf = calculate_ptdf(nodes, tables["transmission"], slack_node)
    ptdf_records = [{"line": line, "node": node, "ptdf": ptdf[line, node]} for line in lines for node in nodes]

    line_limit = {
        str(row["line"]): _as_float(row["limit_mw"], f"limit_mw for {row['line']}")
        for row in _records(tables["transmission"])
    }

    result = {
        "metadata": {
            "source_workbook": str(DEFAULT_INPUT.relative_to(SCRIPT_DIR)),
            "description": "Prepared deterministic spot-market BESS EPEC benchmark data.",
            "slack_node": slack_node,
        },
        "nodes": nodes,
        "generators": generators,
        "storage_units": storage_units,
        "times": times,
        "soc_times": soc_times,
        "lines": lines,
        "generators_at_node": generators_at_node,
        "generation_cost": generation_cost,
        "generation_capacity": generation_capacity_records,
        "demand_el": [{"node": node, "hour": hour, "demand_mw": demand_index[node, hour]} for node in nodes for hour in times],
        "voll": _as_float(settings.get("voll", 10000.0), "voll"),
        "x_power": x_power_records,
        "x_energy": x_energy_records,
        "line_limit": line_limit,
        "ptdf": ptdf_records,
        "eta": _as_float(settings.get("eta", 0.936), "eta"),
    }
    tiers = demand_tier_records(tables, result["voll"])
    if tiers is not None:
        result["demand_tiers"] = tiers
    return result


def write_processed_data(data: dict[str, Any], output_path: Path, ptdf_output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ptdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    pd.DataFrame(data["ptdf"]).to_csv(ptdf_output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic spot-market BESS EPEC benchmark data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the Excel assumption workbook.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path for processed model JSON.")
    parser.add_argument("--ptdf-output", type=Path, default=DEFAULT_PTDF_OUTPUT, help="Optional PTDF audit CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tables = read_workbook(args.input)
    data = build_processed_data(tables)
    try:
        data["metadata"]["source_workbook"] = str(Path(args.input).resolve().relative_to(SCRIPT_DIR))
    except ValueError:
        data["metadata"]["source_workbook"] = str(args.input)
    write_processed_data(data, args.output, args.ptdf_output)
    print(f"Wrote processed data to {args.output}")
    print(f"Wrote PTDF audit table to {args.ptdf_output}")
    print(
        "Prepared "
        f"{len(data['nodes'])} nodes, {len(data['generators'])} generators, "
        f"{len(data['storage_units'])} storage units."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
