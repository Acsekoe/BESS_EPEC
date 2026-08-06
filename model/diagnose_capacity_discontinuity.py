"""Fixed-demand diagnostics for the N6/N8 BESS capacity discontinuity.

The script never changes the maintained market data or EPEC. It aggregates the
saved four-investor fleet into one physically equivalent storage unit, clears
the exact hard market repeatedly, and exports:

1. a joint N6/N8 capacity grid under the base network; and
2. a compact PV-availability and export-corridor sensitivity screen along the
   saved fleet's N6/N8 capacity ray.

These are post-hoc market diagnostics, not equilibrium computations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pyomo.environ as pyo


MODEL_DIR = Path(__file__).resolve().parent
TIKHONOV_DIR = MODEL_DIR / "tikhonov_kkt"
for candidate in (MODEL_DIR, TIKHONOV_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from single_investor_mpec import (  # noqa: E402
    build_fixed_demand_primal_model,
    capital_recovery_factor,
    fixed_demand_reference_lambda,
)
from tikhonov_kkt.common import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_calibrated_case,
)


DEFAULT_FLEET_PATH = (
    MODEL_DIR
    / "output"
    / "jacobi_tikhonov_runs"
    / "tikhonov_capacity_gamma1e-4_20iters_d025"
    / "final_capacities.csv"
)
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / "capacity_discontinuity_diagnostics_2026-08-05"
STORAGE_ID = "BESS_AGG"
DEGRADATION_EUR_PER_MWH = 15.0
POWER_COST_EUR_PER_MW = 6_600.0
ENERGY_COST_EUR_PER_MWH = 18_800.0
WACC = 0.08
LIFETIME_YEARS = 15
N6_LINES = ("L46", "L69")
N8_LINES = ("L98", "L78")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--fleet", type=Path, default=DEFAULT_FLEET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_aggregate_fleet(path: Path, nodes: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    power = {node: 0.0 for node in nodes}
    energy = {node: 0.0 for node in nodes}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            node = row["node"]
            power[node] += float(row["x_power_mw"])
            energy[node] += float(row["x_energy_mwh"])
    return power, energy


def scaled_case(
    base,
    base_power: dict[str, float],
    base_energy: dict[str, float],
    *,
    n6_power_mw: float,
    n8_power_mw: float,
    pv_scale: float = 1.0,
    n6_line_scale: float = 1.0,
    n8_line_scale: float = 1.0,
):
    power = dict(base_power)
    energy = dict(base_energy)
    for node, candidate_power in (("N6", n6_power_mw), ("N8", n8_power_mw)):
        duration = base_energy[node] / base_power[node]
        power[node] = max(0.0, float(candidate_power))
        energy[node] = power[node] * duration

    generation_capacity = {
        (generator, hour): capacity
        * (pv_scale if "PV" in generator.upper() else 1.0)
        for (generator, hour), capacity in base.generation_capacity.items()
    }
    line_limit = dict(base.line_limit)
    for line in N6_LINES:
        line_limit[line] = base.line_limit[line] * n6_line_scale
    for line in N8_LINES:
        line_limit[line] = base.line_limit[line] * n8_line_scale

    return replace(
        base,
        storage_units=[STORAGE_ID],
        x_power={(STORAGE_ID, node): power[node] for node in base.nodes},
        x_energy={(STORAGE_ID, node): energy[node] for node in base.nodes},
        generation_capacity=generation_capacity,
        line_limit=line_limit,
    )


def solve_case(data) -> dict[str, object]:
    model = build_fixed_demand_primal_model(
        data,
        storage_degradation_eur_per_mwh={STORAGE_ID: DEGRADATION_EUR_PER_MWH},
    )
    solver = pyo.SolverFactory("highs")
    if not solver.available(False):
        raise RuntimeError("The HiGHS LP solver is required for this diagnostic.")
    results = solver.solve(model, tee=False)
    termination = str(results.solver.termination_condition)
    if termination != "optimal":
        return {"termination": termination}

    prices = fixed_demand_reference_lambda(model)
    pv_generators = {
        node: [
            generator
            for generator in data.generators_at_node[node]
            if "PV" in generator.upper()
        ]
        for node in data.nodes
    }
    pv_hours = sorted(
        {
            hour
            for generators in pv_generators.values()
            for generator in generators
            for hour in data.times
            if data.generation_capacity[generator, hour] > 1.0e-6
        }
    )

    curtailment = {}
    curtailed_hours = {}
    for node in ("N6", "N8"):
        hourly = [
            max(
                0.0,
                sum(
                    data.generation_capacity[generator, hour]
                    - pyo.value(model.P_gen[generator, hour])
                    for generator in pv_generators[node]
                ),
            )
            for hour in data.times
        ]
        curtailment[node] = sum(hourly)
        curtailed_hours[node] = sum(value > 1.0e-4 for value in hourly)

    crf_daily = capital_recovery_factor(WACC, LIFETIME_YEARS) / 365.25
    node_economics = {}
    for node in ("N6", "N8"):
        power = data.x_power[STORAGE_ID, node]
        energy = data.x_energy[STORAGE_ID, node]
        operating_margin = sum(
            prices[node, hour]
            * (
                pyo.value(model.P_discharge[STORAGE_ID, node, hour])
                - pyo.value(model.P_charge[STORAGE_ID, node, hour])
            )
            - 0.5
            * DEGRADATION_EUR_PER_MWH
            * (
                pyo.value(model.P_charge[STORAGE_ID, node, hour])
                + pyo.value(model.P_discharge[STORAGE_ID, node, hour])
            )
            for hour in data.times
        )
        if power > 1.0e-8:
            duration = energy / power
            unit_capex = crf_daily * (
                POWER_COST_EUR_PER_MW + ENERGY_COST_EUR_PER_MWH * duration
            )
            unit_operating_margin = operating_margin / power
            unit_net_margin = unit_operating_margin - unit_capex
        else:
            duration = 0.0
            unit_capex = None
            unit_operating_margin = None
            unit_net_margin = None
        node_economics[node] = {
            "duration_hours": duration,
            "operating_margin_eur_per_day": operating_margin,
            "unit_operating_margin_eur_per_mw_day": unit_operating_margin,
            "unit_capex_eur_per_mw_day": unit_capex,
            "unit_net_margin_eur_per_mw_day": unit_net_margin,
        }

    return {
        "termination": termination,
        "objective_eur_per_day": pyo.value(model.objective),
        "pv_curtailment_n6_mwh": curtailment["N6"],
        "pv_curtailment_n8_mwh": curtailment["N8"],
        "pv_curtailment_total_mwh": curtailment["N6"] + curtailment["N8"],
        "curtailed_hours_n6": curtailed_hours["N6"],
        "curtailed_hours_n8": curtailed_hours["N8"],
        "lambda_n6_pv_min": min(prices["N6", hour] for hour in pv_hours),
        "lambda_n6_pv_max": max(prices["N6", hour] for hour in pv_hours),
        "lambda_n8_pv_min": min(prices["N8", hour] for hour in pv_hours),
        "lambda_n8_pv_max": max(prices["N8", hour] for hour in pv_hours),
        "lambda_n6_pv_regimes": "|".join(
            f"{value:.3f}" for value in sorted({round(prices["N6", hour], 3) for hour in pv_hours})
        ),
        "lambda_n8_pv_regimes": "|".join(
            f"{value:.3f}" for value in sorted({round(prices["N8", hour], 3) for hour in pv_hours})
        ),
        **{f"n6_{key}": value for key, value in node_economics["N6"].items()},
        **{f"n8_{key}": value for key, value in node_economics["N8"].items()},
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_calibrated_case(args.data)
    base_power, base_energy = load_aggregate_fleet(args.fleet, list(base.nodes))

    joint_values = [float(value) for value in range(0, 201, 25)]
    joint_rows = []
    started = time.perf_counter()
    for index, (n6_power, n8_power) in enumerate(
        ((n6, n8) for n6 in joint_values for n8 in joint_values), start=1
    ):
        case = scaled_case(
            base,
            base_power,
            base_energy,
            n6_power_mw=n6_power,
            n8_power_mw=n8_power,
        )
        row = {
            "n6_power_mw": n6_power,
            "n8_power_mw": n8_power,
            **solve_case(case),
        }
        joint_rows.append(row)
        if index % 10 == 0:
            print(f"joint grid: {index}/{len(joint_values) ** 2}", flush=True)
    write_csv(args.output_dir / "joint_n6_n8_grid.csv", joint_rows)

    configurations = [
        ("base", 1.0, 1.0, 1.0),
        ("pv_0p8", 0.8, 1.0, 1.0),
        ("pv_0p9", 0.9, 1.0, 1.0),
        ("pv_1p1", 1.1, 1.0, 1.0),
        ("pv_1p2", 1.2, 1.0, 1.0),
        ("corridors_0p8", 1.0, 0.8, 0.8),
        ("corridors_0p9", 1.0, 0.9, 0.9),
        ("corridors_1p1", 1.0, 1.1, 1.1),
        ("corridors_1p2", 1.0, 1.2, 1.2),
        ("n8_lines_0p9", 1.0, 1.0, 0.9),
        ("n8_lines_1p1", 1.0, 1.0, 1.1),
        ("n8_lines_1p2", 1.0, 1.0, 1.2),
        ("n6_lines_0p9", 1.0, 0.9, 1.0),
        ("n6_lines_1p1", 1.0, 1.1, 1.0),
    ]
    ray_scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    screen_rows = []
    total_screen = len(configurations) * len(ray_scales)
    for index, ((name, pv_scale, n6_line_scale, n8_line_scale), ray_scale) in enumerate(
        (
            (configuration, ray_scale)
            for configuration in configurations
            for ray_scale in ray_scales
        ),
        start=1,
    ):
        case = scaled_case(
            base,
            base_power,
            base_energy,
            n6_power_mw=ray_scale * base_power["N6"],
            n8_power_mw=ray_scale * base_power["N8"],
            pv_scale=pv_scale,
            n6_line_scale=n6_line_scale,
            n8_line_scale=n8_line_scale,
        )
        screen_rows.append(
            {
                "configuration": name,
                "pv_scale": pv_scale,
                "n6_line_scale": n6_line_scale,
                "n8_line_scale": n8_line_scale,
                "capacity_ray_scale": ray_scale,
                "n6_power_mw": case.x_power[STORAGE_ID, "N6"],
                "n8_power_mw": case.x_power[STORAGE_ID, "N8"],
                **solve_case(case),
            }
        )
        if index % 10 == 0:
            print(f"sensitivity screen: {index}/{total_screen}", flush=True)
    write_csv(args.output_dir / "pv_corridor_screen.csv", screen_rows)

    summary = {
        "data_path": str(args.data.resolve()),
        "fleet_path": str(args.fleet.resolve()),
        "output_is_post_hoc_equilibrium_diagnostic": True,
        "base_aggregate_power_mw": base_power,
        "base_aggregate_energy_mwh": base_energy,
        "joint_grid_values_mw": joint_values,
        "capacity_ray_scales": ray_scales,
        "configurations": [name for name, *_ in configurations],
        "solves": len(joint_rows) + len(screen_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
