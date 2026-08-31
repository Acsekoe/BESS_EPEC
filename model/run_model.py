"""Executable entry point for the maintained multi-investor BESS games."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pyomo.environ as pyo

import mpec_strategic_access_relaxed_kkt

from jacobi_diagonalization import (
    BestResponseResult,
    JacobiConfig,
    JacobiResult,
    SolveOutcome,
    build_best_response,
    collect_best_response,
    four_investors,
    run_jacobi,
)
from mpec_strong_duality import InvestorConfig
from primal_market_clearing_model import MarketData, load_market_data


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = MODEL_DIR / "input" / "market_data.json"


def _load_investors(path: Path | None, data: MarketData) -> tuple[InvestorConfig, ...]:
    """Load an optional economic investor population; retain the baseline by default."""

    if path is None:
        return four_investors(data)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Investor configuration does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read investor configuration {path}: {exc}") from exc
    rows = payload.get("investors") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Investor configuration must contain a nonempty 'investors' list.")

    allowed = {
        "investor_id",
        "wacc",
        "lifetime_years",
        "cost_power_eur_per_mw",
        "cost_energy_eur_per_mwh",
        "degradation_eur_per_mwh",
        "ratio_min",
        "ratio_max",
        "owned_generation_shares",
    }
    investors: list[InvestorConfig] = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Investor entry {position} must be an object.")
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(
                f"Investor entry {position} has unknown fields: {sorted(unknown)}"
            )
        if not isinstance(row.get("investor_id"), str) or not row["investor_id"].strip():
            raise ValueError(f"Investor entry {position} requires a nonempty investor_id.")
        shares = row.get("owned_generation_shares", {})
        if not isinstance(shares, dict):
            raise ValueError(
                f"Investor {row['investor_id']} owned_generation_shares must be an object."
            )
        unknown_generators = set(shares) - set(data.generators)
        if unknown_generators:
            raise ValueError(
                f"Investor {row['investor_id']} owns unknown generators: "
                f"{sorted(unknown_generators)}"
            )
        try:
            investors.append(
                InvestorConfig(
                    investor_id=row["investor_id"].strip(),
                    wacc=float(row.get("wacc", 0.08)),
                    lifetime_years=int(row.get("lifetime_years", 15)),
                    cost_power_eur_per_mw=float(
                        row.get("cost_power_eur_per_mw", 6_600.0)
                    ),
                    cost_energy_eur_per_mwh=float(
                        row.get("cost_energy_eur_per_mwh", 18_800.0)
                    ),
                    degradation_eur_per_mwh=float(
                        row.get("degradation_eur_per_mwh", 15.0)
                    ),
                    ratio_min=float(row.get("ratio_min", 2.0)),
                    ratio_max=float(row.get("ratio_max", 8.0)),
                    owned_generation_shares={
                        str(generator): float(share)
                        for generator, share in shares.items()
                    },
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Investor {row['investor_id']} contains a nonnumeric parameter."
            ) from exc

    ids = [investor.investor_id for investor in investors]
    if len(set(ids)) != len(ids):
        raise ValueError("Investor IDs must be unique.")
    for investor in investors:
        if investor.wacc < 0.0 or investor.lifetime_years <= 0:
            raise ValueError(f"Investor {investor.investor_id} has invalid financing data.")
        if min(
            investor.cost_power_eur_per_mw,
            investor.cost_energy_eur_per_mwh,
            investor.degradation_eur_per_mwh,
            investor.ratio_min,
        ) < 0.0 or investor.ratio_max < investor.ratio_min:
            raise ValueError(f"Investor {investor.investor_id} has invalid storage data.")
        if any(not 0.0 <= share <= 1.0 for share in investor.owned_generation_shares.values()):
            raise ValueError(
                f"Investor {investor.investor_id} ownership shares must lie in [0, 1]."
            )
    for generator in data.generators:
        total_share = sum(
            investor.owned_generation_shares.get(generator, 0.0)
            for investor in investors
        )
        if total_share > 1.0 + 1e-9:
            raise ValueError(
                f"Aggregate ownership of {generator} exceeds 100%: {total_share:g}."
            )
    return tuple(investors)


def _ipopt_executable() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("IPOPT_EXECUTABLE"):
        candidates.append(Path(os.environ["IPOPT_EXECUTABLE"]))
    if shutil.which("ipopt"):
        candidates.append(Path(shutil.which("ipopt") or ""))
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "idaes" / "bin" / "ipopt.exe")
    if os.environ.get("USERPROFILE"):
        base = Path(os.environ["USERPROFILE"]) / "miniconda3"
        candidates += [
            base / "envs" / "bilevel-ipopt" / "Library" / "bin" / "ipopt.exe",
            base / "Library" / "bin" / "ipopt.exe",
        ]
    return next((path for path in candidates if path.is_file()), None)


def _strong_duality_solver(args: argparse.Namespace):
    executable = _ipopt_executable()
    kwargs = {"solver_io": "nl"}
    if executable is not None:
        kwargs["executable"] = str(executable)
    solver = pyo.SolverFactory("ipopt", **kwargs)
    if not solver.available(exception_flag=False):
        raise RuntimeError("Ipopt is unavailable. Set IPOPT_EXECUTABLE or install Ipopt.")
    options = {
        "linear_solver": args.ipopt_linear_solver,
        "max_iter": args.max_solver_iterations,
        "max_cpu_time": args.max_solve_seconds,
        "tol": args.solver_tolerance,
        "acceptable_tol": max(args.solver_tolerance, 1e-5),
        "print_level": 5 if args.tee else 0,
    }
    if args.formulation in {
        "relaxed-kkt",
        "strategic-price-relaxed-kkt",
        "strategic-quantity",
        "strategic-price-quantity",
        "strategic-access",
    }:
        options.update(
            {
                "acceptable_tol": args.solver_tolerance,
                "constr_viol_tol": args.solver_tolerance,
                "acceptable_constr_viol_tol": args.solver_tolerance,
            }
        )
    solver.options.update(options)

    def solve(model: pyo.ConcreteModel) -> SolveOutcome:
        started = time.perf_counter()
        result = solver.solve(model, tee=args.tee)
        termination = result.solver.termination_condition
        accepted = {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
        }
        is_optimal = termination in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
        }
        return SolveOutcome(
            str(termination), termination in accepted, is_optimal, time.perf_counter() - started
        )

    return solve


def _kkt_solver(args: argparse.Namespace):
    from pyomo.contrib.appsi.base import TerminationCondition as AppsiTerminationCondition
    from pyomo.contrib.appsi.solvers import Highs

    solver = Highs()
    if not solver.available():
        raise RuntimeError("HiGHS is unavailable. Install the highspy package.")
    solver.config.time_limit = args.max_solve_seconds
    solver.config.mip_gap = args.mip_gap
    solver.config.load_solution = False
    solver.config.stream_solver = args.tee
    solver.config.warmstart = not args.no_warm_start
    if args.parallel_workers > 1:
        solver.highs_options["threads"] = 1

    def solve(model: pyo.ConcreteModel) -> SolveOutcome:
        started = time.perf_counter()
        result = solver.solve(model)
        best = result.best_feasible_objective
        has_solution = best is not None and math.isfinite(float(best))
        if has_solution:
            result.solution_loader.load_vars()
        return SolveOutcome(
            str(result.termination_condition),
            has_solution,
            result.termination_condition == AppsiTerminationCondition.optimal,
            time.perf_counter() - started,
            float(best) if best is not None else None,
            (
                float(result.best_objective_bound)
                if result.best_objective_bound is not None
                else None
            ),
        )

    return solve


def _parallel_best_response(
    data: MarketData,
    config: JacobiConfig,
    investor: InvestorConfig,
    snapshot_power: dict[tuple[str, str], float],
    snapshot_energy: dict[tuple[str, str], float],
    snapshot_bid_charge: dict[tuple[str, str, int], float],
    snapshot_offer_discharge: dict[tuple[str, str, int], float],
    snapshot_bid_charge_price: dict[tuple[str, str, int], float],
    snapshot_offer_discharge_price: dict[tuple[str, str, int], float],
    snapshot_access_quantity: dict[tuple[str, str], float],
    snapshot_access_bid: dict[tuple[str, str], float],
    solver_arguments: dict[str, object],
) -> BestResponseResult:
    """Process-pool worker: build and solve one Jacobi best response."""

    args = argparse.Namespace(**solver_arguments)
    started = time.perf_counter()
    try:
        model = build_best_response(
            data,
            config,
            investor,
            snapshot_power,
            snapshot_energy,
            snapshot_bid_charge,
            snapshot_offer_discharge,
            snapshot_bid_charge_price,
            snapshot_offer_discharge_price,
            snapshot_access_quantity,
            snapshot_access_bid,
        )
        solver = (
            _strong_duality_solver(args)
            if config.formulation
            in {
                "strong-duality",
                "relaxed-kkt",
                "strategic-operation",
                "strategic-price-relaxed-kkt",
                "strategic-quantity",
                "strategic-price-quantity",
                "strategic-access",
            }
            else _kkt_solver(args)
        )
        outcome = solver(model)
    except Exception as exc:
        outcome = SolveOutcome(
            f"error: {exc}", False, False, time.perf_counter() - started
        )
        investor_id = investor.investor_id
        return BestResponseResult(
            investor_id,
            outcome,
            {n: snapshot_power[investor_id, n] for n in data.nodes},
            {n: snapshot_energy[investor_id, n] for n in data.nodes},
            math.nan,
            {
                (n, int(t)): snapshot_bid_charge[investor_id, n, int(t)]
                for n in data.nodes
                for t in data.times
            }
            if snapshot_bid_charge
            else {},
            {
                (n, int(t)): snapshot_offer_discharge[investor_id, n, int(t)]
                for n in data.nodes
                for t in data.times
            }
            if snapshot_offer_discharge
            else {},
            proposed_access_quantity=(
                {
                    n: snapshot_access_quantity[investor_id, n]
                    for n in data.nodes
                }
                if snapshot_access_quantity
                else {}
            ),
            proposed_access_bid=(
                {n: snapshot_access_bid[investor_id, n] for n in data.nodes}
                if snapshot_access_bid
                else {}
            ),
        )
    return collect_best_response(model, outcome, data)


def _serialisable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _data_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resume_signature(config: JacobiConfig, data_sha256: str) -> dict[str, object]:
    signature: dict[str, object] = {
        "data_sha256": data_sha256,
        "formulation": config.formulation,
        "node_limit_mw": config.node_limit_mw,
        "damping": config.damping,
        "tolerance_mw": config.tolerance_mw,
        "tolerance_mwh": config.tolerance_mwh,
        "consecutive_sweeps": config.consecutive_sweeps,
        "initial_ratio_hours": config.initial_ratio_hours,
        "numerical_initial_power_mw": config.numerical_initial_power_mw,
        "cleanup_tolerance": config.cleanup_tolerance,
        "proximal_penalty": config.proximal_penalty,
        "proximal_energy_scale": config.proximal_energy_scale,
        "price_bound": config.price_bound,
        "dual_bound": config.dual_bound,
        "big_m_dual": config.big_m_dual,
        "sparse_capacity_tol": config.sparse_capacity_tol,
        "warm_start_lower_level": config.warm_start_lower_level,
        "investors": [
            {
                "investor_id": investor.investor_id,
                "wacc": investor.wacc,
                "lifetime_years": investor.lifetime_years,
                "cost_power_eur_per_mw": investor.cost_power_eur_per_mw,
                "cost_energy_eur_per_mwh": investor.cost_energy_eur_per_mwh,
                "degradation_eur_per_mwh": investor.degradation_eur_per_mwh,
                "ratio_min": investor.ratio_min,
                "ratio_max": investor.ratio_max,
                "owned_generation_shares": dict(
                    sorted(investor.owned_generation_shares.items())
                ),
            }
            for investor in config.investors
        ],
    }
    if config.formulation in {
        "strategic-operation",
        "strategic-price-relaxed-kkt",
    }:
        signature.update(
            {
                "bid_price_bound": config.bid_price_bound,
                "initial_bid_charge_eur_per_mwh": config.initial_bid_charge_eur_per_mwh,
                "initial_offer_discharge_eur_per_mwh": config.initial_offer_discharge_eur_per_mwh,
                "tolerance_bid_eur_per_mwh": config.tolerance_bid_eur_per_mwh,
                "strategic_quantity_withholding": False,
            }
        )
    if config.formulation in {
        "strategic-quantity",
        "strategic-price-quantity",
    }:
        signature.update(
            {
                "initial_charge_bid_mw": config.initial_charge_bid_mw,
                "initial_discharge_bid_mw": config.initial_discharge_bid_mw,
                "tolerance_quantity_bid_mw": config.tolerance_quantity_bid_mw,
                "strategic_quantity_withholding": True,
            }
        )
    if config.formulation == "strategic-price-quantity":
        signature.update(
            {
                "bid_price_bound": config.bid_price_bound,
                "initial_bid_charge_eur_per_mwh": config.initial_bid_charge_eur_per_mwh,
                "initial_offer_discharge_eur_per_mwh": config.initial_offer_discharge_eur_per_mwh,
                "tolerance_bid_eur_per_mwh": config.tolerance_bid_eur_per_mwh,
                "proximal_price_scale": config.proximal_price_scale,
                "strategic_price_bidding": True,
            }
        )
    if config.formulation == "strategic-price-relaxed-kkt":
        signature["proximal_price_scale"] = config.proximal_price_scale
    if config.formulation == "strategic-access":
        signature.update(
            {
                "access_request_limit_mw": config.access_request_limit_mw,
                "access_bid_bound_eur_per_mw_day": config.access_bid_bound,
                "initial_access_bid_eur_per_mw_day": (
                    config.initial_access_bid_eur_per_mw_day
                ),
                "access_undamped_sweeps": config.access_undamped_sweeps,
                "tolerance_access_bid_eur_per_mw_day": (
                    config.tolerance_access_bid_eur_per_mw_day
                ),
                "proximal_price_scale": config.proximal_price_scale,
                "access_settlement": "pay-as-bid",
            }
        )
    if config.formulation in {
        "relaxed-kkt",
        "strategic-price-relaxed-kkt",
        "strategic-quantity",
        "strategic-price-quantity",
        "strategic-access",
    }:
        signature["complementarity_epsilon"] = config.complementarity_epsilon
    return signature


def _checkpoint(
    path: Path,
    state: JacobiResult,
    config: JacobiConfig,
    data_sha256: str,
) -> None:
    price_strategic = config.formulation in {
        "strategic-operation",
        "strategic-price-relaxed-kkt",
    }
    quantity_strategic = config.formulation == "strategic-quantity"
    combined_strategic = config.formulation == "strategic-price-quantity"
    access_strategic = config.formulation == "strategic-access"
    payload = {
        "format_version": (
            7
            if access_strategic
            else 5
            if combined_strategic
            else 4
            if quantity_strategic
            else 3
            if price_strategic
            else 2
        ),
        "sweep": state.sweep,
        "converged": state.converged,
        "stop_reason": state.stop_reason,
        "formulation": config.formulation,
        "projection_count": state.projection_count,
        "stable_sweeps": state.stable_sweeps,
        "power_mw": {f"{i}|{n}": value for (i, n), value in state.power.items()},
        "energy_mwh": {f"{i}|{n}": value for (i, n), value in state.energy.items()},
        "history": state.history,
        "resume_signature": _resume_signature(config, data_sha256),
    }
    if price_strategic:
        payload["bid_charge_eur_per_mwh"] = {
            f"{i}|{n}|{t}": value
            for (i, n, t), value in state.bid_charge.items()
        }
        payload["offer_discharge_eur_per_mwh"] = {
            f"{i}|{n}|{t}": value
            for (i, n, t), value in state.offer_discharge.items()
        }
    elif quantity_strategic or combined_strategic:
        payload["charge_bid_mw"] = {
            f"{i}|{n}|{t}": value
            for (i, n, t), value in state.bid_charge.items()
        }
        payload["discharge_bid_mw"] = {
            f"{i}|{n}|{t}": value
            for (i, n, t), value in state.offer_discharge.items()
        }
        if combined_strategic:
            payload["bid_charge_eur_per_mwh"] = {
                f"{i}|{n}|{t}": value
                for (i, n, t), value in state.bid_charge_price.items()
            }
            payload["offer_discharge_eur_per_mwh"] = {
                f"{i}|{n}|{t}": value
                for (i, n, t), value in state.offer_discharge_price.items()
            }
    if access_strategic:
        payload["access_quantity_mw"] = {
            f"{i}|{n}": value
            for (i, n), value in state.access_quantity.items()
        }
        payload["access_bid_eur_per_mw_day"] = {
            f"{i}|{n}": value for (i, n), value in state.access_bid.items()
        }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    data: MarketData,
    config: JacobiConfig,
    data_sha256: str,
    allow_proximal_penalty_change: bool = False,
) -> JacobiResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Resume checkpoint does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read resume checkpoint {path}: {exc}") from exc

    price_strategic = config.formulation in {
        "strategic-operation",
        "strategic-price-relaxed-kkt",
    }
    quantity_strategic = config.formulation == "strategic-quantity"
    combined_strategic = config.formulation == "strategic-price-quantity"
    access_strategic = config.formulation == "strategic-access"
    expected_version = (
        7
        if access_strategic
        else 5
        if combined_strategic
        else 4
        if quantity_strategic
        else 3
        if price_strategic
        else 2
    )
    if payload.get("format_version") != expected_version:
        raise ValueError(
            f"This formulation requires checkpoint format version {expected_version}."
        )
    expected_signature = _resume_signature(config, data_sha256)
    actual_signature = payload.get("resume_signature")
    if (
        access_strategic
        and isinstance(actual_signature, dict)
        and "access_undamped_sweeps" not in actual_signature
    ):
        # Format-v7 checkpoints written before this schedule existed used the
        # configured damping immediately, which is equivalent to a zero cutoff.
        actual_signature = {**actual_signature, "access_undamped_sweeps": 0}
    proximal_penalty_changed = False
    if actual_signature != expected_signature:
        if not isinstance(actual_signature, dict):
            raise ValueError("Resume checkpoint has no valid configuration signature.")
        mismatches = [
            key
            for key, expected in expected_signature.items()
            if actual_signature.get(key) != expected
        ]
        if allow_proximal_penalty_change and set(mismatches) == {"proximal_penalty"}:
            proximal_penalty_changed = True
        else:
            details = ", ".join(mismatches) if mismatches else "unknown fields"
            raise ValueError(
                "Resume checkpoint does not match the requested game configuration: "
                f"{details}. Use the original settings and data, or explicitly allow "
                "a proximal-penalty restart."
            )

    investor_ids = [investor.investor_id for investor in config.investors]
    expected_keys = {
        f"{investor_id}|{node}"
        for investor_id in investor_ids
        for node in data.nodes
    }

    def capacities(field: str) -> dict[tuple[str, str], float]:
        raw = payload.get(field)
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError(
                f"Checkpoint field {field!r} does not contain exactly the current "
                "investor-node keys."
            )
        parsed: dict[tuple[str, str], float] = {}
        for key, raw_value in raw.items():
            investor_id, node = key.split("|", 1)
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {field} value for {key}: {raw_value!r}") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid {field} value for {key}: {raw_value!r}")
            parsed[investor_id, node] = value
        return parsed

    power = capacities("power_mw")
    energy = capacities("energy_mwh")
    investors_by_id = {
        investor.investor_id: investor for investor in config.investors
    }
    tolerance = max(config.cleanup_tolerance, 1e-7)
    if not access_strategic:
        for investor_id in investor_ids:
            investor = investors_by_id[investor_id]
            for node in data.nodes:
                key = investor_id, node
                if energy[key] < investor.ratio_min * power[key] - tolerance:
                    raise ValueError(
                        f"Checkpoint violates the minimum energy-to-power ratio at {key}."
                    )
                if energy[key] > investor.ratio_max * power[key] + tolerance:
                    raise ValueError(
                        f"Checkpoint violates the maximum energy-to-power ratio at {key}."
                    )
    for node in data.nodes:
        total = sum(power[investor_id, node] for investor_id in investor_ids)
        if total > config.node_limit_mw + tolerance:
            raise ValueError(
                f"Checkpoint exceeds the shared connection limit at {node}: "
                f"{total:g} MW."
            )

    access_quantity: dict[tuple[str, str], float] = {}
    access_bid: dict[tuple[str, str], float] = {}
    if access_strategic:
        access_quantity = capacities("access_quantity_mw")
        access_bid = capacities("access_bid_eur_per_mw_day")
        for investor_id in investor_ids:
            total_request = sum(
                access_quantity[investor_id, node] for node in data.nodes
            )
            if total_request > config.access_request_limit_mw + tolerance:
                raise ValueError(
                    f"Checkpoint exceeds the access request limit for {investor_id}."
                )
            for node in data.nodes:
                key = investor_id, node
                if access_quantity[key] > config.node_limit_mw + tolerance:
                    raise ValueError(f"Checkpoint access request exceeds the nodal limit at {key}.")
                if access_bid[key] > config.access_bid_bound + tolerance:
                    raise ValueError(f"Checkpoint access bid exceeds its bound at {key}.")

    bid_charge: dict[tuple[str, str, int], float] = {}
    offer_discharge: dict[tuple[str, str, int], float] = {}
    bid_charge_price: dict[tuple[str, str, int], float] = {}
    offer_discharge_price: dict[tuple[str, str, int], float] = {}
    if price_strategic or quantity_strategic or combined_strategic:
        expected_hour_keys = {
            f"{investor_id}|{node}|{int(time)}"
            for investor_id in investor_ids
            for node in data.nodes
            for time in data.times
        }

    def prices(field: str) -> dict[tuple[str, str, int], float]:
        raw = payload.get(field)
        if not isinstance(raw, dict) or set(raw) != expected_hour_keys:
            raise ValueError(
                f"Checkpoint field {field!r} does not contain exactly the "
                "current investor-node-hour keys."
            )
        parsed: dict[tuple[str, str, int], float] = {}
        for key, raw_value in raw.items():
            investor_id, node, raw_time = key.split("|", 2)
            try:
                value = float(raw_value)
                time = int(raw_time)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {field} value for {key}: {raw_value!r}"
                ) from exc
            if not math.isfinite(value) or abs(value) > config.bid_price_bound + 1e-7:
                raise ValueError(f"Invalid {field} value for {key}: {raw_value!r}")
            parsed[investor_id, node, time] = value
        return parsed

    def quantities(field: str) -> dict[tuple[str, str, int], float]:
        raw = payload.get(field)
        if not isinstance(raw, dict) or set(raw) != expected_hour_keys:
            raise ValueError(
                f"Checkpoint field {field!r} does not contain exactly the "
                "current investor-node-hour keys."
            )
        parsed: dict[tuple[str, str, int], float] = {}
        for key, raw_value in raw.items():
            investor_id, node, raw_time = key.split("|", 2)
            try:
                value = float(raw_value)
                time = int(raw_time)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {field} value for {key}: {raw_value!r}"
                ) from exc
            if (
                not math.isfinite(value)
                or value < 0.0
                or value > power[investor_id, node] + tolerance
            ):
                raise ValueError(f"Invalid {field} value for {key}: {raw_value!r}")
            parsed[investor_id, node, time] = value
        return parsed

    if price_strategic:
        bid_charge = prices("bid_charge_eur_per_mwh")
        offer_discharge = prices("offer_discharge_eur_per_mwh")
    elif quantity_strategic or combined_strategic:
        bid_charge = quantities("charge_bid_mw")
        offer_discharge = quantities("discharge_bid_mw")
        if combined_strategic:
            bid_charge_price = prices("bid_charge_eur_per_mwh")
            offer_discharge_price = prices("offer_discharge_eur_per_mwh")

    checked_charge_prices = bid_charge if price_strategic else bid_charge_price
    checked_discharge_prices = (
        offer_discharge if price_strategic else offer_discharge_price
    )
    for key in checked_charge_prices:
        if checked_discharge_prices[key] + 1e-8 < checked_charge_prices[key] / (data.eta**2):
            raise ValueError(
                f"Checkpoint prices permit a same-hour storage loop at {key}."
            )

    try:
        sweep = int(payload["sweep"])
        projection_count = int(payload.get("projection_count", 0))
        stable_sweeps = int(payload.get("stable_sweeps", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint has invalid sweep counters.") from exc
    if sweep < 0 or projection_count < 0 or not 0 <= stable_sweeps <= sweep:
        raise ValueError("Checkpoint has out-of-range sweep counters.")
    history = payload.get("history", [])
    if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise ValueError("Checkpoint history must be a list of records.")
    for row in history:
        row.setdefault("max_raw_bid_deviation_eur_per_mwh", 0.0)
        row.setdefault("max_iterate_bid_change_eur_per_mwh", 0.0)
        row.setdefault("max_raw_quantity_bid_deviation_mw", 0.0)
        row.setdefault("max_iterate_quantity_bid_change_mw", 0.0)
        row.setdefault("max_raw_access_quantity_deviation_mw", 0.0)
        row.setdefault("max_iterate_access_quantity_change_mw", 0.0)
        row.setdefault("max_raw_access_bid_deviation_eur_per_mw_day", 0.0)
        row.setdefault("max_iterate_access_bid_change_eur_per_mw_day", 0.0)
        row.setdefault("effective_damping", None)
        row.setdefault("old_access_request_mw", None)
        row.setdefault("best_response_access_request_mw", None)
        row.setdefault("new_access_request_mw", None)

    return JacobiResult(
        power=power,
        energy=energy,
        history=history,
        sweep=sweep,
        converged=(
            False
            if proximal_penalty_changed
            else bool(payload.get("converged", False))
        ),
        stop_reason=(
            "restarted with changed proximal penalty"
            if proximal_penalty_changed
            else str(payload.get("stop_reason", ""))
        ),
        projection_count=projection_count,
        stable_sweeps=0 if proximal_penalty_changed else stable_sweeps,
        bid_charge=bid_charge,
        offer_discharge=offer_discharge,
        bid_charge_price=bid_charge_price,
        offer_discharge_price=offer_discharge_price,
        access_quantity=access_quantity,
        access_bid=access_bid,
    )


def _write_outputs(
    output_dir: Path,
    data: MarketData,
    state: JacobiResult,
    config: JacobiConfig,
    args: argparse.Namespace,
    data_sha256: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(_serialisable_args(args), indent=2), encoding="utf-8"
    )
    if state.history:
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(state.history[0]))
            writer.writeheader()
            writer.writerows(state.history)
    with (output_dir / "final_capacities.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["investor", "node", "power_mw", "energy_mwh"])
        for investor in config.investors:
            for node in data.nodes:
                key = investor.investor_id, node
                writer.writerow([*key, state.power[key], state.energy[key]])
    access_lower = None
    if config.formulation == "strategic-access":
        access_lower = mpec_strategic_access_relaxed_kkt.solve_fixed_access_market(
            data,
            access_quantity={
                investor.investor_id: {
                    node: state.access_quantity[investor.investor_id, node]
                    for node in data.nodes
                }
                for investor in config.investors
            },
            access_bid={
                investor.investor_id: {
                    node: state.access_bid[investor.investor_id, node]
                    for node in data.nodes
                }
                for investor in config.investors
            },
            energy_capacity={
                investor.investor_id: {
                    node: state.energy[investor.investor_id, node]
                    for node in data.nodes
                }
                for investor in config.investors
            },
            degradation={
                investor.investor_id: investor.degradation_eur_per_mwh
                for investor in config.investors
            },
            node_limit_mw=config.node_limit_mw,
        )
        with (output_dir / "final_access_bids.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "investor",
                    "node",
                    "requested_power_mw",
                    "access_bid_eur_per_mw_day",
                    "awarded_power_mw",
                    "energy_capacity_mwh",
                    "implied_duration_hours",
                    "pay_as_bid_payment_eur_per_day",
                ]
            )
            for investor in config.investors:
                for node in data.nodes:
                    key = investor.investor_id, node
                    writer.writerow(
                        [
                            *key,
                            state.access_quantity[key],
                            state.access_bid[key],
                            state.power[key],
                            state.energy[key],
                            (
                                state.energy[key] / state.power[key]
                                if state.power[key] > config.cleanup_tolerance
                                else 0.0
                            ),
                            state.access_bid[key] * state.power[key],
                        ]
                    )
        with (output_dir / "final_nodal_access.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "node",
                    "limit_mw",
                    "total_requested_mw",
                    "total_awarded_mw",
                    "scarcity_value_eur_per_mw_day",
                ]
            )
            for node in data.nodes:
                writer.writerow(
                    [
                        node,
                        config.node_limit_mw,
                        sum(
                            state.access_quantity[investor.investor_id, node]
                            for investor in config.investors
                        ),
                        sum(
                            state.power[investor.investor_id, node]
                            for investor in config.investors
                        ),
                        -float(
                            access_lower.dual[access_lower.nodal_access_bound[node]]
                        ),
                    ]
                )
    if config.formulation in {
        "strategic-operation",
        "strategic-price-relaxed-kkt",
    }:
        with (output_dir / "final_strategic_bids.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "investor",
                    "node",
                    "hour",
                    "full_power_available_mw",
                    "bid_charge_eur_per_mwh",
                    "offer_discharge_eur_per_mwh",
                ]
            )
            for investor in config.investors:
                for node in data.nodes:
                    for time in data.times:
                        key = investor.investor_id, node, int(time)
                        writer.writerow(
                            [
                                *key,
                                state.power[investor.investor_id, node],
                                state.bid_charge[key],
                                state.offer_discharge[key],
                            ]
                        )
    elif config.formulation in {
        "strategic-quantity",
        "strategic-price-quantity",
    }:
        combined = config.formulation == "strategic-price-quantity"
        with (output_dir / "final_strategic_quantities.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            columns = [
                "investor",
                "node",
                "hour",
                "installed_power_mw",
                "charge_bid_mw",
                "discharge_bid_mw",
            ]
            if combined:
                columns += [
                    "bid_charge_eur_per_mwh",
                    "offer_discharge_eur_per_mwh",
                ]
            writer.writerow(columns)
            for investor in config.investors:
                for node in data.nodes:
                    for time in data.times:
                        key = investor.investor_id, node, int(time)
                        row = [
                            *key,
                            state.power[investor.investor_id, node],
                            state.bid_charge[key],
                            state.offer_discharge[key],
                        ]
                        if combined:
                            row += [
                                state.bid_charge_price[key],
                                state.offer_discharge_price[key],
                            ]
                        writer.writerow(row)
    summary = {
        "formulation": config.formulation,
        "parallel_workers": args.parallel_workers,
        "sweeps": state.sweep,
        "converged": state.converged,
        "stop_reason": state.stop_reason,
        "projection_count": state.projection_count,
        "stable_sweeps": state.stable_sweeps,
        "total_power_mw": sum(state.power.values()),
        "total_energy_mwh": sum(state.energy.values()),
        "investor_totals": {
            investor.investor_id: {
                "power_mw": sum(
                    state.power[investor.investor_id, node]
                    for node in data.nodes
                ),
                "energy_mwh": sum(
                    state.energy[investor.investor_id, node]
                    for node in data.nodes
                ),
            }
            for investor in config.investors
        },
    }
    if config.formulation in {
        "relaxed-kkt",
        "strategic-price-relaxed-kkt",
        "strategic-quantity",
        "strategic-price-quantity",
        "strategic-access",
    }:
        latest = state.history[-len(config.investors) :] if state.history else []
        summary["relaxed_kkt"] = {
            "complementarity_epsilon": config.complementarity_epsilon,
            "latest_maximum_product": max(
                (
                    float(row["complementarity_max_product"])
                    for row in latest
                    if row.get("complementarity_max_product") is not None
                ),
                default=None,
            ),
            "latest_maximum_violation": max(
                (
                    float(row["complementarity_max_violation"])
                    for row in latest
                    if row.get("complementarity_max_violation") is not None
                ),
                default=None,
            ),
            "latest_maximum_absolute_primal_dual_gap_eur_per_day": max(
                (
                    abs(float(row["primal_dual_gap_eur_per_day"]))
                    for row in latest
                    if row.get("primal_dual_gap_eur_per_day") is not None
                ),
                default=None,
            ),
        }
    if config.formulation == "strategic-access":
        assert access_lower is not None
        summary["strategic_access"] = {
            "settlement": "pay-as-bid",
            "access_bid_unit": "EUR/MW-day",
            "access_bid_bound_eur_per_mw_day": config.access_bid_bound,
            "investor_request_limit_mw": config.access_request_limit_mw,
            "initial_access_bid_eur_per_mw_day": (
                config.initial_access_bid_eur_per_mw_day
            ),
            "undamped_sweeps": config.access_undamped_sweeps,
            "later_damping": config.damping,
            "access_bid_tolerance_eur_per_mw_day": (
                config.tolerance_access_bid_eur_per_mw_day
            ),
            "total_requested_power_mw": sum(state.access_quantity.values()),
            "maximum_minimum_ratio_violation_mwh": max(
                max(
                    0.0,
                    investor.ratio_min
                    * state.power[investor.investor_id, node]
                    - state.energy[investor.investor_id, node],
                )
                for investor in config.investors
                for node in data.nodes
            ),
            "maximum_maximum_ratio_violation_mwh": max(
                max(
                    0.0,
                    state.energy[investor.investor_id, node]
                    - investor.ratio_max
                    * state.power[investor.investor_id, node],
                )
                for investor in config.investors
                for node in data.nodes
            ),
            "total_access_payment_eur_per_day": sum(
                state.access_bid[key] * state.power[key]
                for key in state.access_bid
            ),
            "common_lower_objective_eur_per_day": float(
                pyo.value(access_lower.objective)
            ),
            "common_generation_cost_eur_per_day": sum(
                data.generation_cost[g]
                * float(pyo.value(access_lower.P_gen[g, t]))
                for g, t in access_lower.GT
            ),
            "investors": {
                investor.investor_id: {
                    "requested_power_mw": sum(
                        state.access_quantity[investor.investor_id, node]
                        for node in data.nodes
                    ),
                    "awarded_power_mw": sum(
                        state.power[investor.investor_id, node]
                        for node in data.nodes
                    ),
                    "energy_capacity_mwh": sum(
                        state.energy[investor.investor_id, node]
                        for node in data.nodes
                    ),
                    "access_payment_eur_per_day": sum(
                        state.access_bid[investor.investor_id, node]
                        * state.power[investor.investor_id, node]
                        for node in data.nodes
                    ),
                    "minimum_access_bid_eur_per_mw_day": min(
                        state.access_bid[investor.investor_id, node]
                        for node in data.nodes
                    ),
                    "maximum_access_bid_eur_per_mw_day": max(
                        state.access_bid[investor.investor_id, node]
                        for node in data.nodes
                    ),
                }
                for investor in config.investors
            },
            "nodal_access": {
                node: {
                    "limit_mw": config.node_limit_mw,
                    "requested_mw": sum(
                        state.access_quantity[investor.investor_id, node]
                        for investor in config.investors
                    ),
                    "awarded_mw": sum(
                        state.power[investor.investor_id, node]
                        for investor in config.investors
                    ),
                    "scarcity_value_eur_per_mw_day": -float(
                        access_lower.dual[access_lower.nodal_access_bound[node]]
                    ),
                }
                for node in data.nodes
            },
        }
    if config.formulation in {
        "strategic-operation",
        "strategic-price-relaxed-kkt",
    }:
        price_relaxed_kkt = config.formulation == "strategic-price-relaxed-kkt"
        price_summary = {
            "quantity_withholding": False,
            "full_quantity_availability": True,
            "bid_price_bound_eur_per_mwh": config.bid_price_bound,
            "bid_tolerance_eur_per_mwh": config.tolerance_bid_eur_per_mwh,
            "lower_level_optimality": (
                "relaxed-kkt" if price_relaxed_kkt else "strong-duality"
            ),
            "active_capacity_bid_ranges": {
                investor.investor_id: {
                    "charge_bid_min": min(
                        state.bid_charge[investor.investor_id, node, int(time)]
                        for node in data.nodes
                        for time in data.times
                        if state.power[investor.investor_id, node]
                        > config.sparse_capacity_tol
                    ),
                    "charge_bid_max": max(
                        state.bid_charge[investor.investor_id, node, int(time)]
                        for node in data.nodes
                        for time in data.times
                        if state.power[investor.investor_id, node]
                        > config.sparse_capacity_tol
                    ),
                    "discharge_offer_min": min(
                        state.offer_discharge[
                            investor.investor_id, node, int(time)
                        ]
                        for node in data.nodes
                        for time in data.times
                        if state.power[investor.investor_id, node]
                        > config.sparse_capacity_tol
                    ),
                    "discharge_offer_max": max(
                        state.offer_discharge[
                            investor.investor_id, node, int(time)
                        ]
                        for node in data.nodes
                        for time in data.times
                        if state.power[investor.investor_id, node]
                        > config.sparse_capacity_tol
                    ),
                }
                for investor in config.investors
                if any(
                    state.power[investor.investor_id, node]
                    > config.sparse_capacity_tol
                    for node in data.nodes
                )
            },
        }
        if price_relaxed_kkt:
            price_summary.update(
                {
                    "proximal_penalty_scope": "capacities and prices",
                    "proximal_price_scale_eur_per_mwh": config.proximal_price_scale,
                }
            )
        summary[
            "strategic_price_relaxed_kkt"
            if price_relaxed_kkt
            else "strategic_operation"
        ] = price_summary
    elif config.formulation in {
        "strategic-quantity",
        "strategic-price-quantity",
    }:
        combined = config.formulation == "strategic-price-quantity"
        strategic_summary = {
            "quantity_withholding": True,
            "quantity_tolerance_mw": config.tolerance_quantity_bid_mw,
            "settlement": "realised dispatch at nodal LMP",
            "active_capacity_bid_ranges_mw": {
                investor.investor_id: {
                    "charge_bid_min": min(
                        state.bid_charge[investor.investor_id, node, int(time)]
                        for node in data.nodes
                        for time in data.times
                        if state.power[investor.investor_id, node]
                        > config.sparse_capacity_tol
                    ),
                    "charge_bid_max": max(
                        state.bid_charge[investor.investor_id, node, int(time)]
                        for node in data.nodes
                        for time in data.times
                        if state.power[investor.investor_id, node]
                        > config.sparse_capacity_tol
                    ),
                    "discharge_bid_min": min(
                        state.offer_discharge[
                            investor.investor_id, node, int(time)
                        ]
                        for node in data.nodes
                        for time in data.times
                        if state.power[investor.investor_id, node]
                        > config.sparse_capacity_tol
                    ),
                    "discharge_bid_max": max(
                        state.offer_discharge[
                            investor.investor_id, node, int(time)
                        ]
                        for node in data.nodes
                        for time in data.times
                        if state.power[investor.investor_id, node]
                        > config.sparse_capacity_tol
                    ),
                }
                for investor in config.investors
                if any(
                    state.power[investor.investor_id, node]
                    > config.sparse_capacity_tol
                    for node in data.nodes
                )
            },
        }
        if combined:
            strategic_summary.update(
                {
                    "strategic_prices": True,
                    "bid_price_bound_eur_per_mwh": config.bid_price_bound,
                    "bid_tolerance_eur_per_mwh": config.tolerance_bid_eur_per_mwh,
                    "proximal_penalty_scope": "capacities, quantities, and prices",
                    "proximal_price_scale_eur_per_mwh": config.proximal_price_scale,
                    "active_capacity_price_ranges_eur_per_mwh": {
                        investor.investor_id: {
                            "charge_bid_min": min(
                                state.bid_charge_price[
                                    investor.investor_id, node, int(time)
                                ]
                                for node in data.nodes
                                for time in data.times
                                if state.power[investor.investor_id, node]
                                > config.sparse_capacity_tol
                            ),
                            "charge_bid_max": max(
                                state.bid_charge_price[
                                    investor.investor_id, node, int(time)
                                ]
                                for node in data.nodes
                                for time in data.times
                                if state.power[investor.investor_id, node]
                                > config.sparse_capacity_tol
                            ),
                            "discharge_offer_min": min(
                                state.offer_discharge_price[
                                    investor.investor_id, node, int(time)
                                ]
                                for node in data.nodes
                                for time in data.times
                                if state.power[investor.investor_id, node]
                                > config.sparse_capacity_tol
                            ),
                            "discharge_offer_max": max(
                                state.offer_discharge_price[
                                    investor.investor_id, node, int(time)
                                ]
                                for node in data.nodes
                                for time in data.times
                                if state.power[investor.investor_id, node]
                                > config.sparse_capacity_tol
                            ),
                        }
                        for investor in config.investors
                        if any(
                            state.power[investor.investor_id, node]
                            > config.sparse_capacity_tol
                            for node in data.nodes
                        )
                    },
                }
            )
        summary[
            "strategic_price_quantity" if combined else "strategic_quantity"
        ] = strategic_summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _checkpoint(output_dir / "checkpoint.json", state, config, data_sha256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a capacity, price-strategic, or quantity-strategic EPEC."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--investor-config",
        type=Path,
        metavar="JSON",
        help="Optional investor population; omit for the maintained four-player baseline.",
    )
    parser.add_argument(
        "--formulation",
        choices=(
            "strong-duality",
            "relaxed-kkt",
            "kkt-bigm",
            "strategic-operation",
            "strategic-price-relaxed-kkt",
            "strategic-quantity",
            "strategic-price-quantity",
            "strategic-access",
        ),
        default="strong-duality",
    )
    parser.add_argument("--node-limit-mw", type=float, default=100.0)
    parser.add_argument("--max-sweeps", type=int, default=60)
    parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument(
        "--access-undamped-sweeps",
        type=int,
        default=10,
        help=(
            "For strategic-access only, apply full best responses through this "
            "total sweep number, then use --damping."
        ),
    )
    parser.add_argument("--tolerance-mw", type=float, default=0.5)
    parser.add_argument("--tolerance-mwh", type=float, default=1.0)
    parser.add_argument("--consecutive-sweeps", type=int, default=2)
    parser.add_argument(
        "--run-to-max-sweeps",
        action="store_true",
        help="Continue through --max-sweeps even after satisfying convergence.",
    )
    parser.add_argument(
        "--initial-power-mw",
        type=float,
        default=0.0,
        help=(
            "Initial installed MW per investor-node; for strategic-access, "
            "the initial requested MW per investor-node before enforcing the "
            "portfolio request cap."
        ),
    )
    parser.add_argument("--initial-ratio-hours", type=float, default=2.0)
    parser.add_argument(
        "--numerical-initial-power-mw",
        type=float,
        default=10.0,
        help="NLP starting guess only; it does not change the Jacobi capacity state.",
    )
    parser.add_argument("--cleanup-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--proximal-penalty",
        type=float,
        default=0.0,
        help="L1 best-response regularizer in EUR/(MW day); zero is the baseline.",
    )
    parser.add_argument("--proximal-energy-scale", type=float, default=2.0)
    parser.add_argument(
        "--proximal-price-scale",
        type=float,
        default=10.0,
        help="EUR/MWh normalization for the combined strategy's L1 price changes.",
    )
    parser.add_argument("--price-bound", type=float, default=500.0)
    parser.add_argument("--dual-bound", type=float, default=10_000.0)
    parser.add_argument("--big-m-dual", type=float, default=800.0)
    parser.add_argument(
        "--complementarity-epsilon",
        type=float,
        default=1.0e-3,
        help=(
            "Upper bound x*y <= epsilon for every relaxed-KKT "
            "complementarity pair."
        ),
    )
    parser.add_argument(
        "--bid-price-bound",
        type=float,
        default=500.0,
        help="Absolute EUR/MWh bound for strategic charging bids and discharge offers.",
    )
    parser.add_argument(
        "--initial-bid-charge",
        type=float,
        default=0.0,
        help="Initial charging bid in EUR/MWh for every investor, node, and hour.",
    )
    parser.add_argument(
        "--initial-offer-discharge",
        type=float,
        default=0.0,
        help="Initial discharge offer in EUR/MWh for every investor, node, and hour.",
    )
    parser.add_argument(
        "--bid-tolerance",
        type=float,
        default=0.5,
        help="Raw EUR/MWh convergence tolerance for strategic bid prices.",
    )
    parser.add_argument(
        "--initial-charge-quantity-mw",
        type=float,
        default=0.0,
        help="Initial maximum charging bid in MW for every node-hour.",
    )
    parser.add_argument(
        "--initial-discharge-quantity-mw",
        type=float,
        default=0.0,
        help="Initial maximum discharging bid in MW for every node-hour.",
    )
    parser.add_argument(
        "--quantity-bid-tolerance-mw",
        type=float,
        default=0.5,
        help="Raw MW convergence tolerance for strategic quantity bids.",
    )
    parser.add_argument(
        "--access-request-limit-mw",
        type=float,
        default=200.0,
        help="Maximum total nodal MW requested by one access bidder.",
    )
    parser.add_argument(
        "--access-bid-bound",
        type=float,
        default=500.0,
        help="Non-negative upper bound on access bids in EUR/MW-day.",
    )
    parser.add_argument(
        "--initial-access-bid",
        type=float,
        default=1.0,
        help="Initial access bid in EUR/MW-day at every node.",
    )
    parser.add_argument(
        "--access-bid-tolerance",
        type=float,
        default=0.5,
        help="Raw EUR/MW-day convergence tolerance for access bids.",
    )
    parser.add_argument("--sparse-capacity-tolerance", type=float, default=1e-8)
    parser.add_argument("--solver-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--ipopt-linear-solver",
        choices=("ma57", "mumps"),
        default="ma57",
        help="Sparse linear solver used by Ipopt; MA57 is the default.",
    )
    parser.add_argument("--max-solver-iterations", type=int, default=1500)
    parser.add_argument("--max-solve-seconds", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        choices=range(1, 5),
        default=4,
        metavar="{1,2,3,4}",
        help="Separate investor-solve processes; use one worker per investor when possible.",
    )
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-from",
        type=Path,
        metavar="CHECKPOINT",
        help=(
            "Continue from a compatible checkpoint. --max-sweeps is the total "
            "target sweep number, not the number of additional sweeps."
        ),
    )
    parser.add_argument(
        "--allow-proximal-penalty-change",
        action="store_true",
        help=(
            "Allow --resume-from when only --proximal-penalty differs; reset "
            "the convergence streak. Use a new --output-dir to preserve stage 1."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.parallel_workers > 1 and args.tee:
        raise SystemExit("--tee is only available with --parallel-workers 1.")
    data = load_market_data(args.data)
    data_sha256 = _data_sha256(args.data)
    try:
        investors = _load_investors(args.investor_config, data)
    except ValueError as exc:
        raise SystemExit(f"Cannot load investors: {exc}") from exc
    config = JacobiConfig(
        investors=investors,
        formulation=args.formulation,
        node_limit_mw=args.node_limit_mw,
        max_sweeps=args.max_sweeps,
        damping=args.damping,
        access_undamped_sweeps=args.access_undamped_sweeps,
        tolerance_mw=args.tolerance_mw,
        tolerance_mwh=args.tolerance_mwh,
        consecutive_sweeps=args.consecutive_sweeps,
        stop_at_convergence=not args.run_to_max_sweeps,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        numerical_initial_power_mw=args.numerical_initial_power_mw,
        cleanup_tolerance=args.cleanup_tolerance,
        proximal_penalty=args.proximal_penalty,
        proximal_energy_scale=args.proximal_energy_scale,
        proximal_price_scale=args.proximal_price_scale,
        price_bound=args.price_bound,
        dual_bound=args.dual_bound,
        big_m_dual=args.big_m_dual,
        complementarity_epsilon=args.complementarity_epsilon,
        sparse_capacity_tol=args.sparse_capacity_tolerance,
        warm_start_lower_level=not args.no_warm_start,
        bid_price_bound=args.bid_price_bound,
        initial_bid_charge_eur_per_mwh=args.initial_bid_charge,
        initial_offer_discharge_eur_per_mwh=args.initial_offer_discharge,
        tolerance_bid_eur_per_mwh=args.bid_tolerance,
        initial_charge_bid_mw=args.initial_charge_quantity_mw,
        initial_discharge_bid_mw=args.initial_discharge_quantity_mw,
        tolerance_quantity_bid_mw=args.quantity_bid_tolerance_mw,
        access_request_limit_mw=args.access_request_limit_mw,
        access_bid_bound=args.access_bid_bound,
        initial_access_bid_eur_per_mw_day=args.initial_access_bid,
        tolerance_access_bid_eur_per_mw_day=args.access_bid_tolerance,
    )
    initial_state = None
    if args.resume_from is not None:
        try:
            initial_state = _load_checkpoint(
                args.resume_from,
                data,
                config,
                data_sha256,
                allow_proximal_penalty_change=args.allow_proximal_penalty_change,
            )
        except ValueError as exc:
            raise SystemExit(f"Cannot resume: {exc}") from exc
        if not initial_state.converged and args.max_sweeps <= initial_state.sweep:
            raise SystemExit(
                "Cannot resume: --max-sweeps must be greater than the checkpoint "
                f"sweep ({initial_state.sweep})."
            )
    output_dir = args.output_dir or (
        args.resume_from.resolve().parent
        if args.resume_from is not None
        else MODEL_DIR / "output" / datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    )

    def after_sweep(state: JacobiResult) -> None:
        latest = state.history[-len(investors):]
        max_mw = max(float(row["max_raw_deviation_mw"]) for row in latest)
        max_mwh = max(float(row["max_raw_deviation_mwh"]) for row in latest)
        status = ", ".join(
            f"{row['investor']}={row['termination']}" for row in latest
        )
        bid_status = (
            (
                " / "
                f"{max(float(row['max_raw_access_quantity_deviation_mw']) for row in latest):.3f} MW requests / "
                f"{max(float(row['max_raw_access_bid_deviation_eur_per_mw_day']) for row in latest):.3f} EUR/MW-day"
            )
            if args.formulation == "strategic-access"
            else " / "
            f"{max(float(row['max_raw_bid_deviation_eur_per_mwh']) for row in latest):.3f} EUR/MWh"
            if args.formulation
            in {"strategic-operation", "strategic-price-relaxed-kkt"}
            else (
                " / "
                f"{max(float(row['max_raw_quantity_bid_deviation_mw']) for row in latest):.3f} MW bids / "
                f"{max(float(row['max_raw_bid_deviation_eur_per_mwh']) for row in latest):.3f} EUR/MWh"
                if args.formulation == "strategic-price-quantity"
                else (
                    " / "
                    f"{max(float(row['max_raw_quantity_bid_deviation_mw']) for row in latest):.3f} MW bids"
                    if args.formulation == "strategic-quantity"
                    else ""
                )
            )
        )
        complementarity_status = (
            "; max KKT product "
            f"{max((float(row['complementarity_max_product']) for row in latest if row.get('complementarity_max_product') is not None), default=math.nan):.3e}, "
            "violation "
            f"{max((float(row['complementarity_max_violation']) for row in latest if row.get('complementarity_max_violation') is not None), default=math.nan):.1e}"
            if args.formulation
            in {
                "relaxed-kkt",
                "strategic-price-relaxed-kkt",
                "strategic-quantity",
                "strategic-price-quantity",
                "strategic-access",
            }
            else ""
        )
        print(
            f"sweep {state.sweep:>3}: max raw deviation {max_mw:.3f} MW / "
            f"{max_mwh:.3f} MWh{bid_status}{complementarity_status}; {status}"
        )
        if not args.no_output:
            output_dir.mkdir(parents=True, exist_ok=True)
            _checkpoint(
                output_dir / "checkpoint.json", state, config, data_sha256
            )

    damping_status = f"damping={args.damping:g}"
    if args.formulation == "strategic-access" and args.access_undamped_sweeps > 0:
        damping_status = (
            f"undamped through sweep {args.access_undamped_sweeps}, "
            f"then damping={args.damping:g}"
        )
    print(
        f"Running {len(investors)}-investor Jacobi with {args.formulation}; "
        f"{damping_status}, max_sweeps={args.max_sweeps}, "
        f"parallel_workers={args.parallel_workers}."
    )
    if initial_state is not None:
        print(
            f"Resuming from {args.resume_from} after sweep {initial_state.sweep}; "
            f"retained convergence streak={initial_state.stable_sweeps}."
        )
    if args.parallel_workers == 1:
        solver = (
            _strong_duality_solver(args)
            if args.formulation
            in {
                "strong-duality",
                "relaxed-kkt",
                "strategic-operation",
                "strategic-price-relaxed-kkt",
                "strategic-quantity",
                "strategic-price-quantity",
                "strategic-access",
            }
            else _kkt_solver(args)
        )
        state = run_jacobi(
            data, config, solver, after_sweep, initial_state=initial_state
        )
    else:
        solver_arguments = vars(args).copy()
        with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:

            def solve_batch(
                batch_data: MarketData,
                batch_config: JacobiConfig,
                snapshot_power: dict[tuple[str, str], float],
                snapshot_energy: dict[tuple[str, str], float],
                snapshot_bid_charge: dict[tuple[str, str, int], float],
                snapshot_offer_discharge: dict[tuple[str, str, int], float],
                snapshot_bid_charge_price: dict[tuple[str, str, int], float],
                snapshot_offer_discharge_price: dict[tuple[str, str, int], float],
                snapshot_access_quantity: dict[tuple[str, str], float],
                snapshot_access_bid: dict[tuple[str, str], float],
            ) -> dict[str, BestResponseResult]:
                futures = {
                    executor.submit(
                        _parallel_best_response,
                        batch_data,
                        batch_config,
                        investor,
                        snapshot_power,
                        snapshot_energy,
                        snapshot_bid_charge,
                        snapshot_offer_discharge,
                        snapshot_bid_charge_price,
                        snapshot_offer_discharge_price,
                        snapshot_access_quantity,
                        snapshot_access_bid,
                        solver_arguments,
                    ): investor.investor_id
                    for investor in batch_config.investors
                }
                solved: dict[str, BestResponseResult] = {}
                for future in as_completed(futures):
                    investor_id = futures[future]
                    try:
                        result = future.result()
                        solved[result.investor_id] = result
                    except Exception as exc:
                        solved[investor_id] = BestResponseResult(
                            investor_id,
                            SolveOutcome(f"worker error: {exc}", False, False, 0.0),
                            {n: snapshot_power[investor_id, n] for n in batch_data.nodes},
                            {n: snapshot_energy[investor_id, n] for n in batch_data.nodes},
                            math.nan,
                            {
                                (n, int(t)): snapshot_bid_charge[
                                    investor_id, n, int(t)
                                ]
                                for n in batch_data.nodes
                                for t in batch_data.times
                            }
                            if snapshot_bid_charge
                            else {},
                            {
                                (n, int(t)): snapshot_offer_discharge[
                                    investor_id, n, int(t)
                                ]
                                for n in batch_data.nodes
                                for t in batch_data.times
                            }
                            if snapshot_offer_discharge
                            else {},
                            proposed_access_quantity=(
                                {
                                    n: snapshot_access_quantity[investor_id, n]
                                    for n in batch_data.nodes
                                }
                                if snapshot_access_quantity
                                else {}
                            ),
                            proposed_access_bid=(
                                {
                                    n: snapshot_access_bid[investor_id, n]
                                    for n in batch_data.nodes
                                }
                                if snapshot_access_bid
                                else {}
                            ),
                        )
                return solved

            state = run_jacobi(
                data,
                config,
                None,
                after_sweep,
                solve_batch=solve_batch,
                initial_state=initial_state,
            )
    if not args.no_output:
        _write_outputs(output_dir, data, state, config, args, data_sha256)
        print(f"Outputs: {output_dir}")
    print(
        f"Finished after {state.sweep} sweep(s): {state.stop_reason}. "
        f"Total capacity={sum(state.power.values()):.3f} MW / "
        f"{sum(state.energy.values()):.3f} MWh."
    )
    return 0 if state.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
