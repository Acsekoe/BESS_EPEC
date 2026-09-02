"""Run the four-investor joint investment and hourly storage price-bid game.

Each investor submits a charging buy bid and discharging sell offer for every
node and hour while also choosing nodal MW and continuous MWh capacity. The
ISO chooses all accepted quantities, so there is no quantity withholding.
Fresh runs start from a symmetric 5 MW, three-hour fleet at every node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pyomo.environ as pyo

import mpec_strategic_price_relaxed_kkt
from investors import InvestorConfig
from jacobi_diagonalization import four_investors
from primal_market_clearing_model import MarketData, load_market_data


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = MODEL_DIR / "input" / "market_data.json"
DEFAULT_CAPACITIES = (
    MODEL_DIR
    / "output"
    / "capacity_only_high_limits_30_sweeps"
    / "final_capacities.csv"
)
DEFAULT_OUTPUT = MODEL_DIR / "output" / "joint_investment_hourly_bid_game"


@dataclass(frozen=True)
class BidGameConfig:
    investors: tuple[InvestorConfig, ...]
    capacity_fixed: bool = False
    node_limit_mw: float = 1_000.0
    max_sweeps: int = 1
    damping: float = 0.25
    tolerance_eur_per_mwh: float = 0.5
    consecutive_sweeps: int = 2
    bid_price_bound: float = 500.0
    inverter_limit: str = "shared"
    minimum_strategic_capacity_mw: float = 0.1
    initial_power_mw: float = 5.0
    initial_ratio_hours: float = 3.0
    tolerance_mw: float = 0.5
    tolerance_mwh: float = 1.0
    proximal_capacity_penalty: float = 0.01
    proximal_energy_scale: float = 2.0
    proximal_bid_penalty: float = 0.1
    complementarity_epsilon: float = 1.0e-3
    solver_tolerance: float = 1.0e-4
    max_solver_iterations: int = 3_000
    max_solve_seconds: float = 600.0
    parallel_workers: int = 4
    ipopt_linear_solver: str = "ma57"
    ipopt_executable: str | None = None
    tee: bool = False


@dataclass(frozen=True)
class BidResponse:
    investor_id: str
    termination: str
    optimal: bool
    seconds: float
    proposed_power: dict[str, float]
    proposed_energy: dict[str, float]
    charge_bid: dict[tuple[str, int], float]
    discharge_offer: dict[tuple[str, int], float]
    profit_eur_per_day: float
    maximum_product: float
    maximum_complementarity_violation: float
    primal_dual_gap_eur_per_day: float


def fresh_capacities(
    data: MarketData,
    config: BidGameConfig,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    if config.initial_power_mw < 0.0:
        raise ValueError("initial_power_mw must be non-negative.")
    if config.initial_power_mw * len(config.investors) > config.node_limit_mw:
        raise ValueError("Initial aggregate power exceeds the nodal limit.")
    power = {
        (investor.investor_id, node): config.initial_power_mw
        for investor in config.investors
        for node in data.nodes
    }
    energy = {
        (investor.investor_id, node): config.initial_power_mw
        * min(
            investor.ratio_max,
            max(investor.ratio_min, config.initial_ratio_hours),
        )
        for investor in config.investors
        for node in data.nodes
    }
    return power, energy


def load_capacities(
    path: Path,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    power: dict[tuple[str, str], float] = {}
    energy: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(row["investor"]), str(row["node"])
            power[key] = float(row["power_mw"])
            energy[key] = float(row["energy_mwh"])
    expected = {
        (investor.investor_id, node)
        for investor in investors
        for node in data.nodes
    }
    if set(power) != expected or set(energy) != expected:
        missing = sorted(expected - set(power))
        extra = sorted(set(power) - expected)
        raise ValueError(f"Capacity profile mismatch; missing={missing}, extra={extra}")
    return power, energy


def initial_prices(
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
) -> tuple[
    dict[tuple[str, str, int], float],
    dict[tuple[str, str, int], float],
]:
    charge: dict[tuple[str, str, int], float] = {}
    discharge: dict[tuple[str, str, int], float] = {}
    for investor in investors:
        half_cost = 0.5 * investor.degradation_eur_per_mwh
        for node in data.nodes:
            for time_ in data.times:
                key = investor.investor_id, node, int(time_)
                charge[key] = -half_cost
                discharge[key] = half_cost
    return charge, discharge


def load_prices(
    path: Path,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
) -> tuple[
    dict[tuple[str, str, int], float],
    dict[tuple[str, str, int], float],
]:
    charge: dict[tuple[str, str, int], float] = {}
    discharge: dict[tuple[str, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(row["investor"]), str(row["node"]), int(row["time"])
            charge[key] = float(row["charge_bid_eur_per_mwh"])
            discharge[key] = float(row["discharge_offer_eur_per_mwh"])
    expected = {
        (investor.investor_id, node, int(time_))
        for investor in investors
        for node in data.nodes
        for time_ in data.times
    }
    if set(charge) != expected or set(discharge) != expected:
        missing = sorted(expected - set(charge))
        extra = sorted(set(charge) - expected)
        raise ValueError(f"Price profile mismatch; missing={missing}, extra={extra}")
    return charge, discharge


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_bool(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _ipopt_path(config: BidGameConfig) -> Path | None:
    candidates: list[Path] = []
    if config.ipopt_executable:
        candidates.append(Path(config.ipopt_executable))
    if os.environ.get("IPOPT_EXECUTABLE"):
        candidates.append(Path(os.environ["IPOPT_EXECUTABLE"]))
    if shutil.which("ipopt"):
        candidates.append(Path(shutil.which("ipopt") or ""))
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "idaes" / "bin" / "ipopt.exe")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _diagnostics(model: pyo.ConcreteModel) -> tuple[float, float, float]:
    products = [
        float(pyo.value(component[index]))
        for name in model._relaxed_kkt_product_components
        for component in (getattr(model, name),)
        for index in component
    ]
    minimum = min(products, default=0.0)
    maximum = max(products, default=0.0)
    epsilon = float(model._complementarity_epsilon)
    violation = max(0.0, maximum - epsilon, -minimum)
    gap = float(pyo.value(model.primal_objective - model.dual_objective))
    return maximum, violation, gap


def solve_bid_response(
    data: MarketData,
    config: BidGameConfig,
    investor: InvestorConfig,
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
    snapshot_charge: dict[tuple[str, str, int], float],
    snapshot_discharge: dict[tuple[str, str, int], float],
) -> BidResponse:
    active = investor.investor_id
    rivals = [item for item in config.investors if item.investor_id != active]
    model = mpec_strategic_price_relaxed_kkt.build_model(
        data,
        investor=investor,
        rival_power={
            rival.investor_id: {
                node: power[rival.investor_id, node] for node in data.nodes
            }
            for rival in rivals
        },
        rival_energy={
            rival.investor_id: {
                node: energy[rival.investor_id, node] for node in data.nodes
            }
            for rival in rivals
        },
        rival_bid_charge={
            rival.investor_id: {
                (node, int(time_)): snapshot_charge[
                    rival.investor_id, node, int(time_)
                ]
                for node in data.nodes
                for time_ in data.times
            }
            for rival in rivals
        },
        rival_offer_discharge={
            rival.investor_id: {
                (node, int(time_)): snapshot_discharge[
                    rival.investor_id, node, int(time_)
                ]
                for node in data.nodes
                for time_ in data.times
            }
            for rival in rivals
        },
        initial_bid_charge={
            (node, int(time_)): snapshot_charge[active, node, int(time_)]
            for node in data.nodes
            for time_ in data.times
        },
        initial_offer_discharge={
            (node, int(time_)): snapshot_discharge[active, node, int(time_)]
            for node in data.nodes
            for time_ in data.times
        },
        node_limit_mw=config.node_limit_mw,
        bid_price_bound=config.bid_price_bound,
        price_bound=config.bid_price_bound,
        inverter_limit=config.inverter_limit,
        complementarity_epsilon=config.complementarity_epsilon,
        proximal_bid_charge={
            (node, int(time_)): snapshot_charge[active, node, int(time_)]
            for node in data.nodes
            for time_ in data.times
        },
        proximal_offer_discharge={
            (node, int(time_)): snapshot_discharge[active, node, int(time_)]
            for node in data.nodes
            for time_ in data.times
        },
        proximal_bid_penalty=config.proximal_bid_penalty,
        proximal_price_scale=1.0,
        proximal_power={node: power[active, node] for node in data.nodes},
        proximal_energy={node: energy[active, node] for node in data.nodes},
        proximal_penalty=(
            0.0 if config.capacity_fixed else config.proximal_capacity_penalty
        ),
        proximal_energy_scale=config.proximal_energy_scale,
    )
    half_cost = 0.5 * investor.degradation_eur_per_mwh
    for node in data.nodes:
        model.X_power[node].set_value(power[active, node])
        model.X_energy[node].set_value(energy[active, node])
        if config.capacity_fixed:
            model.X_power[node].fix()
            model.X_energy[node].fix()
        if (
            config.capacity_fixed
            and power[active, node] < config.minimum_strategic_capacity_mw
        ):
            for time_ in data.times:
                model.BidCharge[node, time_].fix(-half_cost)
                model.OfferDischarge[node, time_].fix(half_cost)

    mpec_strategic_price_relaxed_kkt.initialise_lower_level(model, data)
    executable = _ipopt_path(config)
    solver_kwargs = {"solver_io": "nl"}
    if executable is not None:
        solver_kwargs["executable"] = str(executable)
    solver = pyo.SolverFactory("ipopt", **solver_kwargs)
    if not solver.available(exception_flag=False):
        raise RuntimeError("IPOPT is unavailable.")
    solver.options.update(
        {
            "linear_solver": config.ipopt_linear_solver,
            "max_iter": config.max_solver_iterations,
            "max_cpu_time": config.max_solve_seconds,
            "tol": config.solver_tolerance,
            "acceptable_tol": config.solver_tolerance,
            "constr_viol_tol": config.solver_tolerance,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "print_level": 5 if config.tee else 0,
        }
    )
    started = time.perf_counter()
    try:
        result = solver.solve(model, tee=config.tee)
        termination = result.solver.termination_condition
        optimal = termination in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
        }
        termination_text = str(termination)
    except Exception as exc:
        optimal = False
        termination_text = f"error: {exc}"
    seconds = time.perf_counter() - started
    if not optimal:
        return BidResponse(
            active,
            termination_text,
            False,
            seconds,
            {node: power[active, node] for node in data.nodes},
            {node: energy[active, node] for node in data.nodes},
            {
                (node, int(time_)): snapshot_charge[active, node, int(time_)]
                for node in data.nodes
                for time_ in data.times
            },
            {
                (node, int(time_)): snapshot_discharge[active, node, int(time_)]
                for node in data.nodes
                for time_ in data.times
            },
            math.nan,
            math.nan,
            math.nan,
            math.nan,
        )
    maximum, violation, gap = _diagnostics(model)
    return BidResponse(
        active,
        termination_text,
        True,
        seconds,
        {node: float(pyo.value(model.X_power[node])) for node in data.nodes},
        {node: float(pyo.value(model.X_energy[node])) for node in data.nodes},
        {
            (node, int(time_)): float(pyo.value(model.BidCharge[node, time_]))
            for node in data.nodes
            for time_ in data.times
        },
        {
            (node, int(time_)): float(pyo.value(model.OfferDischarge[node, time_]))
            for node in data.nodes
            for time_ in data.times
        },
        float(pyo.value(model.unregularized_profit)),
        maximum,
        violation,
        gap,
    )


def _solve_all(
    data: MarketData,
    config: BidGameConfig,
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
    charge: dict[tuple[str, str, int], float],
    discharge: dict[tuple[str, str, int], float],
) -> dict[str, BidResponse]:
    if config.parallel_workers == 1:
        return {
            investor.investor_id: solve_bid_response(
                data, config, investor, power, energy, charge, discharge
            )
            for investor in config.investors
        }
    responses: dict[str, BidResponse] = {}
    with ProcessPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = {
            executor.submit(
                solve_bid_response,
                data,
                config,
                investor,
                power,
                energy,
                charge,
                discharge,
            ): investor.investor_id
            for investor in config.investors
        }
        for future in as_completed(futures):
            investor_id = futures[future]
            try:
                responses[investor_id] = future.result()
            except Exception as exc:
                responses[investor_id] = BidResponse(
                    investor_id,
                    f"error: {exc}",
                    False,
                    0.0,
                    {},
                    {},
                    {},
                    {},
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                )
    return responses


def _write_bids(
    path: Path,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    charge: dict[tuple[str, str, int], float],
    discharge: dict[tuple[str, str, int], float],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "investor",
                "node",
                "time",
                "power_mw",
                "charge_bid_eur_per_mwh",
                "discharge_offer_eur_per_mwh",
                "same_hour_spread_eur_per_mwh",
            ),
        )
        writer.writeheader()
        for investor in investors:
            for node in data.nodes:
                for time_ in data.times:
                    key = investor.investor_id, node, int(time_)
                    writer.writerow(
                        {
                            "investor": investor.investor_id,
                            "node": node,
                            "time": time_,
                            "power_mw": power[investor.investor_id, node],
                            "charge_bid_eur_per_mwh": charge[key],
                            "discharge_offer_eur_per_mwh": discharge[key],
                            "same_hour_spread_eur_per_mwh": discharge[key] - charge[key],
                        }
                    )


def _capacity_trajectory_rows(
    sweep: int,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    nodal_totals = {
        node: sum(power[investor.investor_id, node] for investor in investors)
        for node in data.nodes
    }
    for investor in investors:
        investor_id = investor.investor_id
        for node in data.nodes:
            power_mw = power[investor_id, node]
            energy_mwh = energy[investor_id, node]
            nodal_total = nodal_totals[node]
            rows.append(
                {
                    "sweep": sweep,
                    "investor": investor_id,
                    "node": node,
                    "power_mw": power_mw,
                    "energy_mwh": energy_mwh,
                    "duration_hours": (
                        energy_mwh / power_mw if power_mw > 1.0e-9 else None
                    ),
                    "nodal_total_power_mw": nodal_total,
                    "investor_nodal_power_share": (
                        power_mw / nodal_total if nodal_total > 1.0e-9 else None
                    ),
                }
            )
    return rows


def _bid_trajectory_rows(
    sweep: int,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
    charge: dict[tuple[str, str, int], float],
    discharge: dict[tuple[str, str, int], float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for investor in investors:
        investor_id = investor.investor_id
        for node in data.nodes:
            power_mw = power[investor_id, node]
            energy_mwh = energy[investor_id, node]
            for time_ in data.times:
                key = investor_id, node, int(time_)
                rows.append(
                    {
                        "sweep": sweep,
                        "investor": investor_id,
                        "node": node,
                        "time": int(time_),
                        "power_mw": power_mw,
                        "energy_mwh": energy_mwh,
                        "duration_hours": (
                            energy_mwh / power_mw if power_mw > 1.0e-9 else None
                        ),
                        "charge_bid_eur_per_mwh": charge[key],
                        "discharge_offer_eur_per_mwh": discharge[key],
                        "same_hour_spread_eur_per_mwh": discharge[key] - charge[key],
                    }
                )
    return rows


def _capacity_total_rows(
    sweep: int,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    system_power = sum(power.values())
    for investor in investors:
        investor_id = investor.investor_id
        total_power = sum(power[investor_id, node] for node in data.nodes)
        total_energy = sum(energy[investor_id, node] for node in data.nodes)
        rows.append(
            {
                "sweep": sweep,
                "investor": investor_id,
                "total_power_mw": total_power,
                "total_energy_mwh": total_energy,
                "portfolio_duration_hours": (
                    total_energy / total_power if total_power > 1.0e-9 else None
                ),
                "investor_system_power_share": (
                    total_power / system_power if system_power > 1.0e-9 else None
                ),
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_capacity_snapshot(
    path: Path,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
) -> None:
    rows = []
    for investor in investors:
        investor_id = investor.investor_id
        for node in data.nodes:
            power_mw = power[investor_id, node]
            energy_mwh = energy[investor_id, node]
            rows.append(
                {
                    "investor": investor_id,
                    "node": node,
                    "power_mw": power_mw,
                    "energy_mwh": energy_mwh,
                    "duration_hours": (
                        energy_mwh / power_mw if power_mw > 1.0e-9 else None
                    ),
                }
            )
    _write_rows(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--capacities", type=Path, default=DEFAULT_CAPACITIES)
    parser.add_argument("--fixed-capacity", action="store_true")
    parser.add_argument(
        "--exclude-investor",
        action="append",
        choices=("I1", "I2", "I3", "I4"),
        default=[],
        help="Exclude an investor from a fresh run; may be repeated.",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--additional-sweeps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sweeps", type=int, default=1)
    parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--node-limit-mw", type=float, default=1_000.0)
    parser.add_argument("--initial-power-mw", type=float, default=5.0)
    parser.add_argument("--initial-ratio-hours", type=float, default=3.0)
    parser.add_argument("--tolerance-mw", type=float, default=0.5)
    parser.add_argument("--tolerance-mwh", type=float, default=1.0)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--complementarity-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--bid-price-bound", type=float, default=500.0)
    parser.add_argument(
        "--wind-offer-cost",
        type=float,
        help="Override the RES_Wind_N1 offer cost in EUR/MWh for a fresh run.",
    )
    parser.add_argument(
        "--pv-offer-cost",
        type=float,
        help="Override both PV generator offer costs in EUR/MWh for a fresh run.",
    )
    parser.add_argument(
        "--inverter-limit",
        choices=("shared", "separate"),
        default="shared",
        help=(
            "shared: P_charge + P_discharge <= X_power (maintained default); "
            "separate: independent directional bounds matching the reproduction"
        ),
    )
    parser.add_argument("--minimum-strategic-capacity-mw", type=float, default=0.1)
    parser.add_argument("--proximal-capacity-penalty", type=float, default=0.01)
    parser.add_argument("--proximal-bid-penalty", type=float, default=0.1)
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.data)
    resume_dir = args.resume_from.resolve() if args.resume_from is not None else None
    previous_config: dict[str, object] = {}
    if resume_dir is not None:
        required = (
            "run_config.json",
            "history.csv",
            "final_capacities.csv",
            "final_hourly_bids.csv",
            "capacity_by_investor_node_by_sweep.csv",
            "capacity_totals_by_investor_by_sweep.csv",
            "bids_by_investor_node_hour_by_sweep.csv",
        )
        missing = [name for name in required if not (resume_dir / name).is_file()]
        if missing:
            raise ValueError(f"Resume directory is missing required files: {missing}")
        previous_config = json.loads(
            (resume_dir / "run_config.json").read_text(encoding="utf-8")
        )
        if previous_config.get("formulation") != (
            "joint-investment-hourly-price-bid-relaxed-kkt"
        ):
            raise ValueError("Only the joint investment-and-bid game can be resumed.")
        if args.exclude_investor:
            raise ValueError(
                "Do not use --exclude-investor when resuming; the saved investor "
                "population is restored automatically."
            )

    requested_cost_overrides: dict[str, float] = {}
    if args.wind_offer_cost is not None:
        requested_cost_overrides["RES_Wind_N1"] = float(args.wind_offer_cost)
    if args.pv_offer_cost is not None:
        requested_cost_overrides.update(
            {
                "RES_PV_N6": float(args.pv_offer_cost),
                "RES_PV_N8": float(args.pv_offer_cost),
            }
        )
    if resume_dir is not None:
        if requested_cost_overrides:
            raise ValueError(
                "Do not provide renewable offer-cost overrides when resuming; "
                "the saved overrides are restored automatically."
            )
        saved_cost_overrides = previous_config.get(
            "generation_cost_overrides_eur_per_mwh", {}
        )
        if not isinstance(saved_cost_overrides, dict):
            raise ValueError("Saved generation-cost overrides must be a mapping.")
        generation_cost_overrides = {
            str(generator): float(cost)
            for generator, cost in saved_cost_overrides.items()
        }
    else:
        generation_cost_overrides = requested_cost_overrides
    unknown_generators = set(generation_cost_overrides) - set(data.generators)
    if unknown_generators:
        raise ValueError(
            f"Generation-cost overrides contain unknown generators: {unknown_generators}"
        )
    if generation_cost_overrides:
        generation_cost = dict(data.generation_cost)
        generation_cost.update(generation_cost_overrides)
        data = replace(data, generation_cost=generation_cost)

    all_investors = four_investors(data)

    if resume_dir is not None:
        saved_investors = previous_config.get("investors")
        if not isinstance(saved_investors, list) or not saved_investors:
            raise ValueError("Resume configuration has no saved investor population.")
        saved_ids = tuple(str(row["investor_id"]) for row in saved_investors)
        investors_by_id = {
            investor.investor_id: investor for investor in all_investors
        }
        unknown_ids = set(saved_ids) - set(investors_by_id)
        if unknown_ids:
            raise ValueError(f"Resume configuration has unknown investors: {unknown_ids}")
        investors = tuple(investors_by_id[investor_id] for investor_id in saved_ids)
    else:
        excluded = set(args.exclude_investor)
        investors = tuple(
            investor
            for investor in all_investors
            if investor.investor_id not in excluded
        )
        if len(investors) < 2:
            raise ValueError("The strategic game requires at least two investors.")

    def setting(name: str, fresh_value: object) -> object:
        return previous_config.get(name, fresh_value) if resume_dir is not None else fresh_value

    sweeps_to_run = args.additional_sweeps if resume_dir is not None else args.max_sweeps
    config = BidGameConfig(
        investors=investors,
        capacity_fixed=bool(setting("capacity_fixed", args.fixed_capacity)),
        node_limit_mw=float(setting("node_limit_mw", args.node_limit_mw)),
        max_sweeps=sweeps_to_run,
        damping=float(setting("damping", args.damping)),
        initial_power_mw=float(setting("initial_power_mw", args.initial_power_mw)),
        initial_ratio_hours=float(
            setting("initial_ratio_hours", args.initial_ratio_hours)
        ),
        tolerance_mw=float(setting("tolerance_mw", args.tolerance_mw)),
        tolerance_mwh=float(setting("tolerance_mwh", args.tolerance_mwh)),
        tolerance_eur_per_mwh=float(setting("tolerance_eur_per_mwh", 0.5)),
        consecutive_sweeps=int(setting("consecutive_sweeps", 2)),
        parallel_workers=args.parallel_workers,
        solver_tolerance=float(
            setting("solver_tolerance", args.solver_tolerance)
        ),
        complementarity_epsilon=float(
            setting("complementarity_epsilon", args.complementarity_epsilon)
        ),
        bid_price_bound=float(setting("bid_price_bound", args.bid_price_bound)),
        inverter_limit=str(setting("inverter_limit", args.inverter_limit)),
        minimum_strategic_capacity_mw=float(
            setting(
                "minimum_strategic_capacity_mw",
                args.minimum_strategic_capacity_mw,
            )
        ),
        proximal_capacity_penalty=float(
            setting("proximal_capacity_penalty", args.proximal_capacity_penalty)
        ),
        proximal_energy_scale=float(setting("proximal_energy_scale", 2.0)),
        proximal_bid_penalty=float(
            setting("proximal_bid_penalty", args.proximal_bid_penalty)
        ),
        max_solver_iterations=int(setting("max_solver_iterations", 3_000)),
        max_solve_seconds=float(setting("max_solve_seconds", 600.0)),
        ipopt_linear_solver=str(setting("ipopt_linear_solver", "ma57")),
        tee=args.tee,
    )
    if not 0.0 < config.damping <= 1.0:
        raise ValueError("Damping must be in (0, 1].")
    if config.max_sweeps <= 0 or config.parallel_workers <= 0:
        raise ValueError("Sweep and worker counts must be positive.")
    if min(
        config.tolerance_mw,
        config.tolerance_mwh,
        config.tolerance_eur_per_mwh,
        config.proximal_capacity_penalty,
        config.proximal_bid_penalty,
    ) < 0.0:
        raise ValueError("Tolerances and proximal penalties must be non-negative.")
    if resume_dir is not None:
        power, energy = load_capacities(
            resume_dir / "final_capacities.csv", data, investors
        )
        charge, discharge = load_prices(
            resume_dir / "final_hourly_bids.csv", data, investors
        )
        capacity_source = f"resume:{resume_dir}"
    elif config.capacity_fixed:
        power, energy = load_capacities(args.capacities, data, investors)
        charge, discharge = initial_prices(data, investors)
        capacity_source = str(args.capacities.resolve())
    else:
        power, energy = fresh_capacities(data, config)
        charge, discharge = initial_prices(data, investors)
        capacity_source = "fresh_symmetric_initialisation"
    if resume_dir is not None and args.output_dir.resolve() == resume_dir:
        raise ValueError("Resume into a new output directory to preserve the source run.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "formulation": (
            "fixed-capacity-hourly-price-bid-relaxed-kkt"
            if config.capacity_fixed
            else "joint-investment-hourly-price-bid-relaxed-kkt"
        ),
        "strategic_variables": [
            *(
                []
                if config.capacity_fixed
                else ["nodal_power_capacity_mw", "nodal_energy_capacity_mwh"]
            ),
            "node_hour_charge_bid_eur_per_mwh",
            "node_hour_discharge_offer_eur_per_mwh",
        ],
        "strategic_quantities": False,
        "capacity_fixed": config.capacity_fixed,
        "capacity_source": capacity_source,
        "resume_from": str(resume_dir) if resume_dir is not None else None,
        "additional_sweeps": sweeps_to_run if resume_dir is not None else None,
        "generation_cost_overrides_eur_per_mwh": generation_cost_overrides,
        "effective_renewable_offer_costs_eur_per_mwh": {
            generator: data.generation_cost[generator]
            for generator in ("RES_Wind_N1", "RES_PV_N6", "RES_PV_N8")
        },
        **{
            key: value
            for key, value in asdict(config).items()
            if key not in {"investors", "ipopt_executable"}
        },
        "investors": [asdict(investor) for investor in investors],
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    if resume_dir is not None:
        history = _read_csv_rows(resume_dir / "history.csv")
        if not history:
            raise ValueError("Resume history is empty.")
        start_sweep = max(int(row["sweep"]) for row in history)
        capacity_trajectory = _read_csv_rows(
            resume_dir / "capacity_by_investor_node_by_sweep.csv"
        )
        capacity_totals_trajectory = _read_csv_rows(
            resume_dir / "capacity_totals_by_investor_by_sweep.csv"
        )
        bid_trajectory = _read_csv_rows(
            resume_dir / "bids_by_investor_node_hour_by_sweep.csv"
        )
        stable_sweeps = int(float(history[-1]["stable_sweeps"]))
    else:
        history: list[dict[str, object]] = []
        start_sweep = 0
        capacity_trajectory = _capacity_trajectory_rows(
            0, data, investors, power, energy
        )
        capacity_totals_trajectory = _capacity_total_rows(
            0, data, investors, power, energy
        )
        bid_trajectory = _bid_trajectory_rows(
            0, data, investors, power, energy, charge, discharge
        )
        stable_sweeps = 0
    _write_rows(
        args.output_dir / "capacity_by_investor_node_by_sweep.csv",
        capacity_trajectory,
    )
    _write_rows(
        args.output_dir / "capacity_totals_by_investor_by_sweep.csv",
        capacity_totals_trajectory,
    )
    _write_rows(
        args.output_dir / "bids_by_investor_node_hour_by_sweep.csv",
        bid_trajectory,
    )
    _write_capacity_snapshot(
        args.output_dir / "current_capacities.csv",
        data,
        investors,
        power,
        energy,
    )
    converged = False
    for sweep in range(
        start_sweep + 1, start_sweep + config.max_sweeps + 1
    ):
        old_power = dict(power)
        old_energy = dict(energy)
        old_charge = dict(charge)
        old_discharge = dict(discharge)
        responses = _solve_all(
            data, config, old_power, old_energy, old_charge, old_discharge
        )
        all_optimal = all(response.optimal for response in responses.values())
        maximum_raw_power_deviation = 0.0
        maximum_raw_energy_deviation = 0.0
        maximum_raw_bid_deviation = 0.0
        for investor in investors:
            investor_id = investor.investor_id
            response = responses[investor_id]
            for node in data.nodes:
                capacity_key = investor_id, node
                if response.optimal and not config.capacity_fixed:
                    proposed_power = response.proposed_power[node]
                    proposed_energy = response.proposed_energy[node]
                    maximum_raw_power_deviation = max(
                        maximum_raw_power_deviation,
                        abs(proposed_power - old_power[capacity_key]),
                    )
                    maximum_raw_energy_deviation = max(
                        maximum_raw_energy_deviation,
                        abs(proposed_energy - old_energy[capacity_key]),
                    )
                    power[capacity_key] = (
                        (1.0 - config.damping) * old_power[capacity_key]
                        + config.damping * proposed_power
                    )
                    energy[capacity_key] = (
                        (1.0 - config.damping) * old_energy[capacity_key]
                        + config.damping * proposed_energy
                    )
                strategic = (
                    not config.capacity_fixed
                    or old_power[capacity_key] >= config.minimum_strategic_capacity_mw
                )
                for time_ in data.times:
                    key = investor_id, node, int(time_)
                    if response.optimal and strategic:
                        proposed_charge = response.charge_bid[node, int(time_)]
                        proposed_discharge = response.discharge_offer[node, int(time_)]
                        maximum_raw_bid_deviation = max(
                            maximum_raw_bid_deviation,
                            abs(proposed_charge - old_charge[key]),
                            abs(proposed_discharge - old_discharge[key]),
                        )
                        charge[key] = (
                            (1.0 - config.damping) * old_charge[key]
                            + config.damping * proposed_charge
                        )
                        discharge[key] = (
                            (1.0 - config.damping) * old_discharge[key]
                            + config.damping * proposed_discharge
                        )
        if not config.capacity_fixed:
            for node in data.nodes:
                nodal_power = sum(
                    power[investor.investor_id, node] for investor in investors
                )
                if nodal_power > config.node_limit_mw:
                    scale = config.node_limit_mw / nodal_power
                    for investor in investors:
                        key = investor.investor_id, node
                        power[key] *= scale
                        energy[key] *= scale

        stable = (
            all_optimal
            and maximum_raw_bid_deviation <= config.tolerance_eur_per_mwh
            and (
                config.capacity_fixed
                or (
                    maximum_raw_power_deviation <= config.tolerance_mw
                    and maximum_raw_energy_deviation <= config.tolerance_mwh
                )
            )
        )
        stable_sweeps = stable_sweeps + 1 if stable else 0
        row: dict[str, object] = {
            "sweep": sweep,
            "all_best_responses_optimal": all_optimal,
            "max_raw_power_deviation_mw": maximum_raw_power_deviation,
            "max_raw_energy_deviation_mwh": maximum_raw_energy_deviation,
            "max_raw_bid_deviation_eur_per_mwh": maximum_raw_bid_deviation,
            "max_complementarity_product": max(
                (
                    response.maximum_product
                    for response in responses.values()
                    if response.optimal
                ),
                default=math.nan,
            ),
            "max_complementarity_violation": max(
                (
                    response.maximum_complementarity_violation
                    for response in responses.values()
                    if response.optimal
                ),
                default=math.nan,
            ),
            "max_absolute_primal_dual_gap_eur_per_day": max(
                (
                    abs(response.primal_dual_gap_eur_per_day)
                    for response in responses.values()
                    if response.optimal
                ),
                default=math.nan,
            ),
            "stable_sweeps": stable_sweeps,
            "solve_seconds": sum(response.seconds for response in responses.values()),
        }
        for investor in investors:
            investor_id = investor.investor_id
            response = responses[investor_id]
            row[f"termination_{investor_id}"] = response.termination
            row[f"profit_{investor_id}_eur_per_day"] = response.profit_eur_per_day
            row[f"total_power_{investor_id}_mw"] = sum(
                power[investor_id, node] for node in data.nodes
            )
            row[f"total_energy_{investor_id}_mwh"] = sum(
                energy[investor_id, node] for node in data.nodes
            )
        history.append(row)
        with (args.output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        _write_bids(
            args.output_dir / "current_hourly_bids.csv",
            data,
            investors,
            power,
            charge,
            discharge,
        )
        _write_capacity_snapshot(
            args.output_dir / "current_capacities.csv",
            data,
            investors,
            power,
            energy,
        )
        capacity_trajectory.extend(
            _capacity_trajectory_rows(sweep, data, investors, power, energy)
        )
        capacity_totals_trajectory.extend(
            _capacity_total_rows(sweep, data, investors, power, energy)
        )
        bid_trajectory.extend(
            _bid_trajectory_rows(
                sweep, data, investors, power, energy, charge, discharge
            )
        )
        _write_rows(
            args.output_dir / "capacity_by_investor_node_by_sweep.csv",
            capacity_trajectory,
        )
        _write_rows(
            args.output_dir / "capacity_totals_by_investor_by_sweep.csv",
            capacity_totals_trajectory,
        )
        _write_rows(
            args.output_dir / "bids_by_investor_node_hour_by_sweep.csv",
            bid_trajectory,
        )
        print(
            f"sweep={sweep:03d} "
            f"power_residual={maximum_raw_power_deviation:.4f} MW "
            f"energy_residual={maximum_raw_energy_deviation:.4f} MWh "
            f"bid_residual={maximum_raw_bid_deviation:.4f} EUR/MWh "
            f"optimal={all_optimal}",
            flush=True,
        )
        if stable_sweeps >= config.consecutive_sweeps:
            converged = True
            break

    _write_bids(
        args.output_dir / "final_hourly_bids.csv",
        data,
        investors,
        power,
        charge,
        discharge,
    )
    _write_capacity_snapshot(
        args.output_dir / "final_capacities.csv",
        data,
        investors,
        power,
        energy,
    )
    investor_power_totals = {
        investor.investor_id: sum(
            power[investor.investor_id, node] for node in data.nodes
        )
        for investor in investors
    }
    investor_energy_totals = {
        investor.investor_id: sum(
            energy[investor.investor_id, node] for node in data.nodes
        )
        for investor in investors
    }
    summary = {
        "formulation": (
            "fixed-capacity-hourly-price-bid-relaxed-kkt"
            if config.capacity_fixed
            else "joint-investment-hourly-price-bid-relaxed-kkt"
        ),
        "capacity_fixed": config.capacity_fixed,
        "strategic_quantities": False,
        "inverter_limit": config.inverter_limit,
        "converged": converged,
        "sweeps": len(history),
        "final_raw_power_residual_mw": history[-1][
            "max_raw_power_deviation_mw"
        ],
        "final_raw_energy_residual_mwh": history[-1][
            "max_raw_energy_deviation_mwh"
        ],
        "final_raw_bid_residual_eur_per_mwh": history[-1][
            "max_raw_bid_deviation_eur_per_mwh"
        ],
        "all_best_responses_optimal": all(
            _row_bool(row["all_best_responses_optimal"]) for row in history
        ),
        "maximum_complementarity_violation": max(
            float(row["max_complementarity_violation"]) for row in history
        ),
        "investor_total_power_mw": investor_power_totals,
        "investor_total_energy_mwh": investor_energy_totals,
        "investor_total_power_range_mw": (
            max(investor_power_totals.values()) - min(investor_power_totals.values())
        ),
        "capacity_trajectory_file": "capacity_by_investor_node_by_sweep.csv",
        "investor_totals_trajectory_file": "capacity_totals_by_investor_by_sweep.csv",
        "bid_trajectory_file": "bids_by_investor_node_hour_by_sweep.csv",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if all(
        _row_bool(row["all_best_responses_optimal"]) for row in history
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
