"""Run the three-investor portfolio bidding game on the owner-split input.

I1 (merchant) strategically bids storage only; I3 and I4 additionally choose
hourly submitted offers for the wind/PV units they own outright. The ISO
clears against submitted offers while investor profit uses true economic
cost. Truthful renewable offers equal true cost (0 EUR/MWh for wind and PV in
the maintained input) and strategic offers may lie above or below truthful.

Experiment layers (fixed-point iteration is damped Jacobi throughout):

- default: fixed capacities, storage prices fixed truthful, strategic
  generation offers only (Phase 2);
- ``--strategic-storage on``: hourly storage prices become strategic too;
- ``--strategic-generation off``: storage-only comparison case;
- ``--endogenous-capacity``: restores nodal MW/MWh investment (Phase 4).

Every run finishes with an exact market reclear of the final profile plus
truthful-generation-offer counterfactuals, so MPEC-claimed profits are always
validated against ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pyomo.environ as pyo

import exact_reclear
import mpec_strategic_portfolio_relaxed_kkt as portfolio_mpec
from investors import InvestorConfig, split_portfolio_investors
from primal_market_clearing_model import MarketData, load_market_data
from run_hourly_bid_game import (
    _diagnostics,
    _ipopt_path,
    _read_csv_rows,
    _row_bool,
    _write_rows,
    fresh_capacities,
    initial_prices,
    load_capacities,
    load_prices,
)


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = MODEL_DIR / "input" / "market_data_strategic_generation.json"
DEFAULT_CAPACITIES = (
    MODEL_DIR
    / "output"
    / "three_investors_without_i2_bid_penalty_0p001_20_sweeps"
    / "final_capacities.csv"
)
DEFAULT_OUTPUT = MODEL_DIR / "output" / "portfolio_bid_game"

RESUME_REQUIRED_FILES = (
    "run_config.json",
    "history.csv",
    "final_capacities.csv",
    "final_hourly_bids.csv",
    "final_generation_offers.csv",
)


@dataclass(frozen=True)
class PortfolioGameConfig:
    investors: tuple[InvestorConfig, ...]
    capacity_fixed: bool = True
    strategic_generation: bool = True
    strategic_storage: bool = False
    node_limit_mw: float = 1_000.0
    max_sweeps: int = 1
    damping: float = 0.25
    tolerance_mw: float = 0.5
    tolerance_mwh: float = 1.0
    tolerance_eur_per_mwh: float = 0.5
    consecutive_sweeps: int = 2
    bid_price_bound: float = 500.0
    generation_offer_bound: float = 500.0
    inverter_limit: str = "shared"
    minimum_strategic_capacity_mw: float = 0.1
    initial_power_mw: float = 5.0
    initial_ratio_hours: float = 3.0
    proximal_capacity_penalty: float = 0.01
    proximal_energy_scale: float = 2.0
    proximal_bid_penalty: float = 0.001
    proximal_generation_penalty: float = 0.01
    complementarity_epsilon: float = 1.0e-3
    solver_tolerance: float = 1.0e-4
    max_solver_iterations: int = 3_000
    max_solve_seconds: float = 600.0
    parallel_workers: int = 3
    ipopt_linear_solver: str = "ma57"
    ipopt_executable: str | None = None
    tee: bool = False


@dataclass(frozen=True)
class PortfolioResponse:
    investor_id: str
    termination: str
    optimal: bool
    seconds: float
    proposed_power: dict[str, float]
    proposed_energy: dict[str, float]
    charge_bid: dict[tuple[str, int], float]
    discharge_offer: dict[tuple[str, int], float]
    generation_offer: dict[tuple[str, int], float]
    profit_eur_per_day: float
    maximum_product: float
    maximum_complementarity_violation: float
    primal_dual_gap_eur_per_day: float


def strategic_generation_pairs(
    data: MarketData, investor: InvestorConfig
) -> list[tuple[str, int]]:
    """Owned (generator, hour) pairs with positive availability."""

    owned = portfolio_mpec.owned_generators(investor)
    return [
        (g, int(t))
        for g in owned
        for t in data.times
        if data.generation_capacity[g, int(t)] > 1e-8
    ]


def truthful_generation_offers(
    data: MarketData, investors: tuple[InvestorConfig, ...]
) -> dict[tuple[str, str, int], float]:
    return {
        (investor.investor_id, g, t): float(data.generation_cost[g])
        for investor in investors
        for g, t in strategic_generation_pairs(data, investor)
    }


def load_generation_offers(
    path: Path, data: MarketData, investors: tuple[InvestorConfig, ...]
) -> dict[tuple[str, str, int], float]:
    offers: dict[tuple[str, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            offers[str(row["investor"]), str(row["generator"]), int(row["time"])] = float(
                row["offer_eur_per_mwh"]
            )
    expected = set(truthful_generation_offers(data, investors))
    if set(offers) != expected:
        missing = sorted(expected - set(offers))
        extra = sorted(set(offers) - expected)
        raise ValueError(
            f"Generation-offer profile mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    return offers


def solve_portfolio_response(
    data: MarketData,
    config: PortfolioGameConfig,
    investor: InvestorConfig,
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
    snapshot_charge: dict[tuple[str, str, int], float],
    snapshot_discharge: dict[tuple[str, str, int], float],
    snapshot_generation: dict[tuple[str, str, int], float],
) -> PortfolioResponse:
    active = investor.investor_id
    rivals = [item for item in config.investors if item.investor_id != active]
    own_pairs = strategic_generation_pairs(data, investor)
    own_offers = {
        (g, t): snapshot_generation[active, g, t] for g, t in own_pairs
    }
    rival_offers = {
        (g, t): value
        for (owner, g, t), value in snapshot_generation.items()
        if owner != active
    }
    model = portfolio_mpec.build_model(
        data,
        investor=investor,
        rival_power={
            rival.investor_id: {n: power[rival.investor_id, n] for n in data.nodes}
            for rival in rivals
        },
        rival_energy={
            rival.investor_id: {n: energy[rival.investor_id, n] for n in data.nodes}
            for rival in rivals
        },
        rival_bid_charge={
            rival.investor_id: {
                (n, int(t)): snapshot_charge[rival.investor_id, n, int(t)]
                for n in data.nodes
                for t in data.times
            }
            for rival in rivals
        },
        rival_offer_discharge={
            rival.investor_id: {
                (n, int(t)): snapshot_discharge[rival.investor_id, n, int(t)]
                for n in data.nodes
                for t in data.times
            }
            for rival in rivals
        },
        initial_bid_charge={
            (n, int(t)): snapshot_charge[active, n, int(t)]
            for n in data.nodes
            for t in data.times
        },
        initial_offer_discharge={
            (n, int(t)): snapshot_discharge[active, n, int(t)]
            for n in data.nodes
            for t in data.times
        },
        rival_generation_offer=rival_offers,
        initial_generation_offer=own_offers,
        proximal_generation_offer=own_offers,
        proximal_generation_penalty=(
            config.proximal_generation_penalty if config.strategic_generation else 0.0
        ),
        generation_offer_bound=config.generation_offer_bound,
        node_limit_mw=config.node_limit_mw,
        bid_price_bound=config.bid_price_bound,
        price_bound=config.bid_price_bound,
        inverter_limit=config.inverter_limit,
        complementarity_epsilon=config.complementarity_epsilon,
        proximal_bid_charge={
            (n, int(t)): snapshot_charge[active, n, int(t)]
            for n in data.nodes
            for t in data.times
        },
        proximal_offer_discharge={
            (n, int(t)): snapshot_discharge[active, n, int(t)]
            for n in data.nodes
            for t in data.times
        },
        proximal_bid_penalty=(
            config.proximal_bid_penalty if config.strategic_storage else 0.0
        ),
        proximal_price_scale=1.0,
        proximal_power={n: power[active, n] for n in data.nodes},
        proximal_energy={n: energy[active, n] for n in data.nodes},
        proximal_penalty=(
            0.0 if config.capacity_fixed else config.proximal_capacity_penalty
        ),
        proximal_energy_scale=config.proximal_energy_scale,
    )

    half_cost = 0.5 * investor.degradation_eur_per_mwh
    for n in data.nodes:
        model.X_power[n].set_value(power[active, n])
        model.X_energy[n].set_value(energy[active, n])
        if config.capacity_fixed:
            model.X_power[n].fix()
            model.X_energy[n].fix()
        small = (
            config.capacity_fixed
            and power[active, n] < config.minimum_strategic_capacity_mw
        )
        for t in data.times:
            if not config.strategic_storage:
                model.BidCharge[n, t].fix(snapshot_charge[active, n, int(t)])
                model.OfferDischarge[n, t].fix(snapshot_discharge[active, n, int(t)])
            elif small:
                model.BidCharge[n, t].fix(-half_cost)
                model.OfferDischarge[n, t].fix(half_cost)
    if not config.strategic_generation:
        for g, t in model.GT_OWNED:
            model.OfferGeneration[g, t].fix(own_offers[g, int(t)])

    portfolio_mpec.initialise_lower_level(model, data)
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
        return PortfolioResponse(
            active,
            termination_text,
            False,
            seconds,
            {n: power[active, n] for n in data.nodes},
            {n: energy[active, n] for n in data.nodes},
            {
                (n, int(t)): snapshot_charge[active, n, int(t)]
                for n in data.nodes
                for t in data.times
            },
            {
                (n, int(t)): snapshot_discharge[active, n, int(t)]
                for n in data.nodes
                for t in data.times
            },
            dict(own_offers),
            math.nan,
            math.nan,
            math.nan,
            math.nan,
        )
    maximum, violation, gap = _diagnostics(model)
    return PortfolioResponse(
        active,
        termination_text,
        True,
        seconds,
        {n: float(pyo.value(model.X_power[n])) for n in data.nodes},
        {n: float(pyo.value(model.X_energy[n])) for n in data.nodes},
        {
            (n, int(t)): float(pyo.value(model.BidCharge[n, t]))
            for n in data.nodes
            for t in data.times
        },
        {
            (n, int(t)): float(pyo.value(model.OfferDischarge[n, t]))
            for n in data.nodes
            for t in data.times
        },
        {
            (g, int(t)): float(pyo.value(model.OfferGeneration[g, t]))
            for g, t in model.GT_OWNED
        },
        float(pyo.value(model.unregularized_profit)),
        maximum,
        violation,
        gap,
    )


def _solve_all(
    data: MarketData,
    config: PortfolioGameConfig,
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
    charge: dict[tuple[str, str, int], float],
    discharge: dict[tuple[str, str, int], float],
    generation: dict[tuple[str, str, int], float],
) -> dict[str, PortfolioResponse]:
    if config.parallel_workers == 1:
        return {
            investor.investor_id: solve_portfolio_response(
                data, config, investor, power, energy, charge, discharge, generation
            )
            for investor in config.investors
        }
    responses: dict[str, PortfolioResponse] = {}
    with ProcessPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = {
            executor.submit(
                solve_portfolio_response,
                data,
                config,
                investor,
                power,
                energy,
                charge,
                discharge,
                generation,
            ): investor.investor_id
            for investor in config.investors
        }
        for future in as_completed(futures):
            investor_id = futures[future]
            try:
                responses[investor_id] = future.result()
            except Exception as exc:
                responses[investor_id] = PortfolioResponse(
                    investor_id,
                    f"error: {exc}",
                    False,
                    0.0,
                    {},
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


def _write_hourly_bids(
    path: Path,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    charge: dict[tuple[str, str, int], float],
    discharge: dict[tuple[str, str, int], float],
) -> None:
    rows = [
        {
            "investor": investor.investor_id,
            "node": n,
            "time": int(t),
            "power_mw": power[investor.investor_id, n],
            "charge_bid_eur_per_mwh": charge[investor.investor_id, n, int(t)],
            "discharge_offer_eur_per_mwh": discharge[investor.investor_id, n, int(t)],
            "same_hour_spread_eur_per_mwh": discharge[investor.investor_id, n, int(t)]
            - charge[investor.investor_id, n, int(t)],
        }
        for investor in investors
        for n in data.nodes
        for t in data.times
    ]
    _write_rows(path, rows)


def _generation_offer_rows(
    sweep: int | None,
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    generation: dict[tuple[str, str, int], float],
) -> list[dict[str, object]]:
    rows = []
    for investor in investors:
        for g, t in strategic_generation_pairs(data, investor):
            row = {
                "investor": investor.investor_id,
                "generator": g,
                "time": int(t),
                "true_cost_eur_per_mwh": float(data.generation_cost[g]),
                "offer_eur_per_mwh": generation[investor.investor_id, g, int(t)],
                "markup_eur_per_mwh": generation[investor.investor_id, g, int(t)]
                - float(data.generation_cost[g]),
                "availability_mw": float(data.generation_capacity[g, int(t)]),
            }
            if sweep is not None:
                row = {"sweep": sweep, **row}
            rows.append(row)
    return rows


def _capacity_rows(
    data: MarketData,
    investors: tuple[InvestorConfig, ...],
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    return [
        {
            "investor": investor.investor_id,
            "node": n,
            "power_mw": power[investor.investor_id, n],
            "energy_mwh": energy[investor.investor_id, n],
            "duration_hours": (
                energy[investor.investor_id, n] / power[investor.investor_id, n]
                if power[investor.investor_id, n] > 1e-9
                else None
            ),
        }
        for investor in investors
        for n in data.nodes
    ]


def _write_exact_outputs(
    output_dir: Path,
    data: MarketData,
    config: PortfolioGameConfig,
    power: dict[tuple[str, str], float],
    energy: dict[tuple[str, str], float],
    charge: dict[tuple[str, str, int], float],
    discharge: dict[tuple[str, str, int], float],
    generation: dict[tuple[str, str, int], float],
    claimed_profits: dict[str, float],
) -> dict[str, object]:
    """Exact reclear of the final profile plus truthful counterfactuals."""

    investors = config.investors
    ids = [investor.investor_id for investor in investors]
    truthful = truthful_generation_offers(data, investors)

    def flatten(offers: dict[tuple[str, str, int], float]) -> dict[tuple[str, int], float]:
        return {(g, int(t)): value for (_, g, t), value in offers.items()}

    def reclear(offers: dict[tuple[str, str, int], float]) -> exact_reclear.ReclearResult:
        return exact_reclear.clear(
            data,
            investor_ids=ids,
            power=power,
            energy=energy,
            charge_bid=charge,
            discharge_offer=discharge,
            generation_offer=flatten(offers),
            inverter_limit=config.inverter_limit,
        )

    final = reclear(generation)
    _write_rows(
        output_dir / "exact_final_generation_dispatch.csv",
        exact_reclear.generation_rows(final, investors),
    )
    _write_rows(
        output_dir / "exact_final_storage_dispatch.csv",
        exact_reclear.storage_rows(final),
    )
    _write_rows(output_dir / "exact_final_prices.csv", exact_reclear.price_rows(final))
    _write_rows(output_dir / "exact_final_line_flows.csv", exact_reclear.line_rows(final))

    final_decomposition = exact_reclear.profit_decomposition(final, investors)
    for row in final_decomposition:
        claimed = claimed_profits.get(str(row["investor"]), math.nan)
        row["mpec_claimed_profit_eur_per_day"] = claimed
        row["claimed_minus_recleared_eur_per_day"] = (
            claimed - float(row["profit_eur_per_day"])
            if not math.isnan(claimed)
            else math.nan
        )
    _write_rows(
        output_dir / "exact_final_profit_decomposition.csv", final_decomposition
    )
    final_profit = {
        str(row["investor"]): float(row["profit_eur_per_day"])
        for row in final_decomposition
    }

    scenarios: list[tuple[str, dict[tuple[str, str, int], float]]] = []
    for investor in investors:
        own = [key for key in generation if key[0] == investor.investor_id]
        if own:
            reverted = dict(generation)
            for key in own:
                reverted[key] = truthful[key]
            scenarios.append((f"truthful_{investor.investor_id}", reverted))
    scenarios.append(("all_truthful_generation", dict(truthful)))

    counterfactual_rows: list[dict[str, object]] = []
    for scenario, offers in scenarios:
        recleared = exact_reclear.profit_decomposition(reclear(offers), investors)
        for row in recleared:
            row = {"scenario": scenario, **row}
            row["profit_delta_vs_final_eur_per_day"] = float(
                row["profit_eur_per_day"]
            ) - final_profit[str(row["investor"])]
            counterfactual_rows.append(row)
    _write_rows(
        output_dir / "truthful_offer_counterfactuals.csv", counterfactual_rows
    )
    return {
        "recleared_profit_eur_per_day": final_profit,
        "claimed_minus_recleared_eur_per_day": {
            str(row["investor"]): row["claimed_minus_recleared_eur_per_day"]
            for row in final_decomposition
        },
        "counterfactual_scenarios": [scenario for scenario, _ in scenarios],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--capacities", type=Path, default=DEFAULT_CAPACITIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strategic-generation", choices=("on", "off"), default="on"
    )
    parser.add_argument("--strategic-storage", choices=("on", "off"), default="off")
    parser.add_argument(
        "--endogenous-capacity",
        action="store_true",
        help="Restore nodal MW/MWh investment (fresh symmetric start).",
    )
    parser.add_argument(
        "--include-i2",
        action="store_true",
        help="Restore I2 for a robustness run (excluded by default).",
    )
    parser.add_argument(
        "--wind-true-cost",
        type=float,
        help="Override the true cost of every wind unit (EUR/MWh).",
    )
    parser.add_argument(
        "--pv-true-cost",
        type=float,
        help="Override the true cost of every PV unit (EUR/MWh).",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--additional-sweeps", type=int, default=10)
    parser.add_argument("--max-sweeps", type=int, default=1)
    parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--node-limit-mw", type=float, default=1_000.0)
    parser.add_argument("--initial-power-mw", type=float, default=5.0)
    parser.add_argument("--initial-ratio-hours", type=float, default=3.0)
    parser.add_argument("--tolerance-mw", type=float, default=0.5)
    parser.add_argument("--tolerance-mwh", type=float, default=1.0)
    parser.add_argument("--tolerance-eur-per-mwh", type=float, default=0.5)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--complementarity-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--bid-price-bound", type=float, default=500.0)
    parser.add_argument("--generation-offer-bound", type=float, default=500.0)
    parser.add_argument(
        "--inverter-limit", choices=("shared", "separate"), default="shared"
    )
    parser.add_argument("--minimum-strategic-capacity-mw", type=float, default=0.1)
    parser.add_argument("--proximal-capacity-penalty", type=float, default=0.01)
    parser.add_argument("--proximal-bid-penalty", type=float, default=0.001)
    parser.add_argument("--proximal-generation-penalty", type=float, default=0.01)
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.data)
    resume_dir = args.resume_from.resolve() if args.resume_from is not None else None
    previous_config: dict[str, object] = {}
    if resume_dir is not None:
        missing = [
            name for name in RESUME_REQUIRED_FILES if not (resume_dir / name).is_file()
        ]
        if missing:
            raise ValueError(f"Resume directory is missing required files: {missing}")
        previous_config = json.loads(
            (resume_dir / "run_config.json").read_text(encoding="utf-8")
        )
        if previous_config.get("game") != "portfolio-bid-relaxed-kkt":
            raise ValueError("Only the portfolio bid game can be resumed.")

    def setting(name: str, fresh_value: object) -> object:
        return (
            previous_config.get(name, fresh_value)
            if resume_dir is not None
            else fresh_value
        )

    if resume_dir is not None and (args.wind_true_cost is not None or args.pv_true_cost is not None):
        raise ValueError(
            "Do not provide true-cost overrides when resuming; the saved "
            "overrides are restored automatically."
        )
    true_cost_overrides: dict[str, float] = {}
    if resume_dir is not None:
        saved = previous_config.get("true_cost_overrides_eur_per_mwh", {})
        true_cost_overrides = {str(g): float(c) for g, c in dict(saved).items()}
    else:
        if args.wind_true_cost is not None:
            true_cost_overrides.update(
                {g: float(args.wind_true_cost) for g in data.generators if "RES_Wind" in g}
            )
        if args.pv_true_cost is not None:
            true_cost_overrides.update(
                {g: float(args.pv_true_cost) for g in data.generators if "RES_PV" in g}
            )
    if true_cost_overrides:
        costs = dict(data.generation_cost)
        costs.update(true_cost_overrides)
        data = replace(data, generation_cost=costs)

    include_i2 = bool(setting("include_i2", args.include_i2))
    investors = split_portfolio_investors(data, include_i2=include_i2)
    if resume_dir is not None:
        saved_ids = tuple(
            str(row["investor_id"]) for row in previous_config.get("investors", [])
        )
        current_ids = tuple(investor.investor_id for investor in investors)
        if saved_ids != current_ids:
            raise ValueError(
                f"Saved population {saved_ids} does not match {current_ids}."
            )

    sweeps_to_run = args.additional_sweeps if resume_dir is not None else args.max_sweeps
    config = PortfolioGameConfig(
        investors=investors,
        capacity_fixed=bool(setting("capacity_fixed", not args.endogenous_capacity)),
        strategic_generation=bool(
            setting("strategic_generation", args.strategic_generation == "on")
        ),
        strategic_storage=bool(
            setting("strategic_storage", args.strategic_storage == "on")
        ),
        node_limit_mw=float(setting("node_limit_mw", args.node_limit_mw)),
        max_sweeps=sweeps_to_run,
        damping=float(setting("damping", args.damping)),
        tolerance_mw=float(setting("tolerance_mw", args.tolerance_mw)),
        tolerance_mwh=float(setting("tolerance_mwh", args.tolerance_mwh)),
        tolerance_eur_per_mwh=float(
            setting("tolerance_eur_per_mwh", args.tolerance_eur_per_mwh)
        ),
        consecutive_sweeps=int(setting("consecutive_sweeps", 2)),
        bid_price_bound=float(setting("bid_price_bound", args.bid_price_bound)),
        generation_offer_bound=float(
            setting("generation_offer_bound", args.generation_offer_bound)
        ),
        inverter_limit=str(setting("inverter_limit", args.inverter_limit)),
        minimum_strategic_capacity_mw=float(
            setting("minimum_strategic_capacity_mw", args.minimum_strategic_capacity_mw)
        ),
        initial_power_mw=float(setting("initial_power_mw", args.initial_power_mw)),
        initial_ratio_hours=float(
            setting("initial_ratio_hours", args.initial_ratio_hours)
        ),
        proximal_capacity_penalty=float(
            setting("proximal_capacity_penalty", args.proximal_capacity_penalty)
        ),
        proximal_energy_scale=float(setting("proximal_energy_scale", 2.0)),
        proximal_bid_penalty=float(
            setting("proximal_bid_penalty", args.proximal_bid_penalty)
        ),
        proximal_generation_penalty=float(
            setting("proximal_generation_penalty", args.proximal_generation_penalty)
        ),
        complementarity_epsilon=float(
            setting("complementarity_epsilon", args.complementarity_epsilon)
        ),
        solver_tolerance=float(setting("solver_tolerance", args.solver_tolerance)),
        max_solver_iterations=int(setting("max_solver_iterations", 3_000)),
        max_solve_seconds=float(setting("max_solve_seconds", 600.0)),
        parallel_workers=args.parallel_workers,
        ipopt_linear_solver=str(setting("ipopt_linear_solver", "ma57")),
        tee=args.tee,
    )
    if not 0.0 < config.damping <= 1.0:
        raise ValueError("Damping must be in (0, 1].")
    if config.max_sweeps <= 0 or config.parallel_workers <= 0:
        raise ValueError("Sweep and worker counts must be positive.")
    if not config.strategic_generation and not config.strategic_storage and config.capacity_fixed:
        raise ValueError(
            "Nothing is strategic: enable generation offers, storage prices, "
            "or endogenous capacity."
        )

    if resume_dir is not None:
        power, energy = load_capacities(
            resume_dir / "final_capacities.csv", data, investors
        )
        charge, discharge = load_prices(
            resume_dir / "final_hourly_bids.csv", data, investors
        )
        generation = load_generation_offers(
            resume_dir / "final_generation_offers.csv", data, investors
        )
        capacity_source = f"resume:{resume_dir}"
    else:
        if config.capacity_fixed:
            power, energy = load_capacities(args.capacities, data, investors)
            capacity_source = str(args.capacities.resolve())
        else:
            power, energy = fresh_capacities(data, config)
            capacity_source = "fresh_symmetric_initialisation"
        charge, discharge = initial_prices(data, investors)
        generation = truthful_generation_offers(data, investors)
    if resume_dir is not None and args.output_dir.resolve() == resume_dir:
        raise ValueError("Resume into a new output directory to preserve the source run.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "game": "portfolio-bid-relaxed-kkt",
        "formulation": (
            ("fixed-capacity" if config.capacity_fixed else "joint-investment")
            + "-portfolio-bid-relaxed-kkt"
        ),
        "strategic_variables": [
            *(
                []
                if config.capacity_fixed
                else ["nodal_power_capacity_mw", "nodal_energy_capacity_mwh"]
            ),
            *(
                ["node_hour_charge_bid_eur_per_mwh", "node_hour_discharge_offer_eur_per_mwh"]
                if config.strategic_storage
                else []
            ),
            *(
                ["generator_hour_offer_eur_per_mwh"]
                if config.strategic_generation
                else []
            ),
        ],
        "include_i2": include_i2,
        "capacity_source": capacity_source,
        "resume_from": str(resume_dir) if resume_dir is not None else None,
        "additional_sweeps": sweeps_to_run if resume_dir is not None else None,
        "true_cost_overrides_eur_per_mwh": true_cost_overrides,
        "renewable_true_costs_eur_per_mwh": {
            g: data.generation_cost[g] for g in data.generators if "RES_" in g
        },
        "data": str(Path(args.data).resolve()),
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
        offer_trajectory = _read_csv_rows(
            resume_dir / "generation_offers_by_investor_generator_hour_by_sweep.csv"
        )
        bid_trajectory = _read_csv_rows(
            resume_dir / "bids_by_investor_node_hour_by_sweep.csv"
        )
        stable_sweeps = int(float(history[-1]["stable_sweeps"]))
    else:
        history = []
        start_sweep = 0
        offer_trajectory = _generation_offer_rows(0, data, investors, generation)
        bid_trajectory = [
            {
                "sweep": 0,
                "investor": investor.investor_id,
                "node": n,
                "time": int(t),
                "charge_bid_eur_per_mwh": charge[investor.investor_id, n, int(t)],
                "discharge_offer_eur_per_mwh": discharge[investor.investor_id, n, int(t)],
            }
            for investor in investors
            for n in data.nodes
            for t in data.times
        ]
        stable_sweeps = 0

    converged = False
    claimed_profits: dict[str, float] = {}
    for sweep in range(start_sweep + 1, start_sweep + config.max_sweeps + 1):
        old_power = dict(power)
        old_energy = dict(energy)
        old_charge = dict(charge)
        old_discharge = dict(discharge)
        old_generation = dict(generation)
        responses = _solve_all(
            data, config, old_power, old_energy, old_charge, old_discharge, old_generation
        )
        all_optimal = all(response.optimal for response in responses.values())
        raw_power_dev = raw_energy_dev = raw_bid_dev = raw_offer_dev = 0.0
        damped_offer_dev = 0.0
        for investor in investors:
            investor_id = investor.investor_id
            response = responses[investor_id]
            if not response.optimal:
                continue
            for n in data.nodes:
                capacity_key = investor_id, n
                if not config.capacity_fixed:
                    raw_power_dev = max(
                        raw_power_dev,
                        abs(response.proposed_power[n] - old_power[capacity_key]),
                    )
                    raw_energy_dev = max(
                        raw_energy_dev,
                        abs(response.proposed_energy[n] - old_energy[capacity_key]),
                    )
                    power[capacity_key] = (
                        (1.0 - config.damping) * old_power[capacity_key]
                        + config.damping * response.proposed_power[n]
                    )
                    energy[capacity_key] = (
                        (1.0 - config.damping) * old_energy[capacity_key]
                        + config.damping * response.proposed_energy[n]
                    )
                if config.strategic_storage:
                    strategic = (
                        not config.capacity_fixed
                        or old_power[capacity_key]
                        >= config.minimum_strategic_capacity_mw
                    )
                    for t in data.times:
                        key = investor_id, n, int(t)
                        if not strategic:
                            continue
                        proposed_charge = response.charge_bid[n, int(t)]
                        proposed_discharge = response.discharge_offer[n, int(t)]
                        raw_bid_dev = max(
                            raw_bid_dev,
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
            if config.strategic_generation:
                for g, t in strategic_generation_pairs(data, investor):
                    key = investor_id, g, int(t)
                    proposed = response.generation_offer[g, int(t)]
                    raw_offer_dev = max(
                        raw_offer_dev, abs(proposed - old_generation[key])
                    )
                    generation[key] = (
                        (1.0 - config.damping) * old_generation[key]
                        + config.damping * proposed
                    )
                    damped_offer_dev = max(
                        damped_offer_dev,
                        abs(generation[key] - old_generation[key]),
                    )
        if not config.capacity_fixed:
            for n in data.nodes:
                nodal_power = sum(
                    power[investor.investor_id, n] for investor in investors
                )
                if nodal_power > config.node_limit_mw:
                    scale = config.node_limit_mw / nodal_power
                    for investor in investors:
                        key = investor.investor_id, n
                        power[key] *= scale
                        energy[key] *= scale

        stable = (
            all_optimal
            and (not config.strategic_storage or raw_bid_dev <= config.tolerance_eur_per_mwh)
            and (
                not config.strategic_generation
                or raw_offer_dev <= config.tolerance_eur_per_mwh
            )
            and (
                config.capacity_fixed
                or (
                    raw_power_dev <= config.tolerance_mw
                    and raw_energy_dev <= config.tolerance_mwh
                )
            )
        )
        stable_sweeps = stable_sweeps + 1 if stable else 0
        row: dict[str, object] = {
            "sweep": sweep,
            "all_best_responses_optimal": all_optimal,
            "max_raw_power_deviation_mw": raw_power_dev,
            "max_raw_energy_deviation_mwh": raw_energy_dev,
            "max_raw_bid_deviation_eur_per_mwh": raw_bid_dev,
            "max_raw_generation_offer_deviation_eur_per_mwh": raw_offer_dev,
            "max_damped_generation_offer_deviation_eur_per_mwh": damped_offer_dev,
            "max_complementarity_product": max(
                (r.maximum_product for r in responses.values() if r.optimal),
                default=math.nan,
            ),
            "max_complementarity_violation": max(
                (
                    r.maximum_complementarity_violation
                    for r in responses.values()
                    if r.optimal
                ),
                default=math.nan,
            ),
            "max_absolute_primal_dual_gap_eur_per_day": max(
                (
                    abs(r.primal_dual_gap_eur_per_day)
                    for r in responses.values()
                    if r.optimal
                ),
                default=math.nan,
            ),
            "stable_sweeps": stable_sweeps,
            "solve_seconds": sum(r.seconds for r in responses.values()),
        }
        for investor in investors:
            response = responses[investor.investor_id]
            row[f"termination_{investor.investor_id}"] = response.termination
            row[f"profit_{investor.investor_id}_eur_per_day"] = response.profit_eur_per_day
            if response.optimal:
                claimed_profits[investor.investor_id] = response.profit_eur_per_day
        history.append(row)
        with (args.output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        offer_trajectory.extend(
            _generation_offer_rows(sweep, data, investors, generation)
        )
        _write_rows(
            args.output_dir
            / "generation_offers_by_investor_generator_hour_by_sweep.csv",
            offer_trajectory,
        )
        bid_trajectory.extend(
            {
                "sweep": sweep,
                "investor": investor.investor_id,
                "node": n,
                "time": int(t),
                "charge_bid_eur_per_mwh": charge[investor.investor_id, n, int(t)],
                "discharge_offer_eur_per_mwh": discharge[
                    investor.investor_id, n, int(t)
                ],
            }
            for investor in investors
            for n in data.nodes
            for t in data.times
        )
        _write_rows(
            args.output_dir / "bids_by_investor_node_hour_by_sweep.csv",
            bid_trajectory,
        )
        _write_rows(
            args.output_dir / "current_generation_offers.csv",
            _generation_offer_rows(None, data, investors, generation),
        )
        _write_rows(
            args.output_dir / "current_capacities.csv",
            _capacity_rows(data, investors, power, energy),
        )
        print(
            f"sweep={sweep:03d} "
            f"power_residual={raw_power_dev:.4f} MW "
            f"energy_residual={raw_energy_dev:.4f} MWh "
            f"bid_residual={raw_bid_dev:.4f} EUR/MWh "
            f"gen_offer_residual={raw_offer_dev:.4f} EUR/MWh "
            f"optimal={all_optimal}",
            flush=True,
        )
        if stable_sweeps >= config.consecutive_sweeps:
            converged = True
            break

    _write_rows(
        args.output_dir / "final_generation_offers.csv",
        _generation_offer_rows(None, data, investors, generation),
    )
    _write_hourly_bids(
        args.output_dir / "final_hourly_bids.csv",
        data,
        investors,
        power,
        charge,
        discharge,
    )
    _write_rows(
        args.output_dir / "final_capacities.csv",
        _capacity_rows(data, investors, power, energy),
    )

    print("Exact reclear of the final profile...", flush=True)
    exact_summary = _write_exact_outputs(
        args.output_dir,
        data,
        config,
        power,
        energy,
        charge,
        discharge,
        generation,
        claimed_profits,
    )

    summary = {
        "formulation": run_config["formulation"],
        "strategic_variables": run_config["strategic_variables"],
        "converged": converged,
        "sweeps": len(history),
        "final_raw_power_residual_mw": history[-1]["max_raw_power_deviation_mw"],
        "final_raw_energy_residual_mwh": history[-1]["max_raw_energy_deviation_mwh"],
        "final_raw_bid_residual_eur_per_mwh": history[-1][
            "max_raw_bid_deviation_eur_per_mwh"
        ],
        "final_raw_generation_offer_residual_eur_per_mwh": history[-1][
            "max_raw_generation_offer_deviation_eur_per_mwh"
        ],
        "all_best_responses_optimal": all(
            _row_bool(row["all_best_responses_optimal"]) for row in history
        ),
        "maximum_complementarity_violation": max(
            float(row["max_complementarity_violation"]) for row in history
        ),
        "mpec_claimed_profit_eur_per_day": claimed_profits,
        **exact_summary,
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
