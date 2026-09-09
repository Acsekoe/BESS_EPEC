"""Oracle-efficient, memory-augmented Nikaido--Isoda search.

The expensive operation in this game is a fresh multistart best-response
oracle.  This driver therefore separates two loops:

* every iteration refreshes one local response per investor, in parallel, and
  scores damped Jacobi/one-player/two-player steps with exact market re-clears
  of a small archive of previously discovered deviations;
* targeted and all-player multistart oracles are called only on a declared
  schedule, after stagnation, or near the requested regret tolerance.

Archived regret is a lower bound and is never reported as equilibrium
certification.  Successful termination requires a fresh all-player multistart
audit at one frozen profile.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pyomo

import capacity_game as game
import regret_guided_search as legacy
import reporting
from investors import three_investors
from iso_market import settle
from market_data import DEFAULT_DATA_PATH, MarketData, load_market_data, validate_afrr
from solvers import SolverSettings


Capacities = game.Capacities
Branch = legacy.Branch
Trial = legacy.Trial
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "oracle_efficient_regret"


@dataclass(frozen=True)
class OracleSearchConfig:
    """Algorithmic controls, separate from the economic game definition."""

    epsilon_eur_per_day: float = 20.0
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
    joint_alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    maximum_iterations: int = 40
    archive_limit_per_investor: int = 3
    screen_branches_per_investor: int = 1
    archive_validation_finalists: int = 1
    targeted_multistart_interval: int = 8
    targeted_refinement_starts: int = 1
    targeted_max_solve_seconds: float = 120.0
    full_audit_interval: int = 0
    stagnation_iterations_before_targeted: int = 1
    near_target_multiplier: float = 5.0
    minimum_reduction_eur_per_day: float = 1.0
    minimum_relative_reduction: float = 1.0e-4
    final_confirmation_audits: int = 2

    def validate(self) -> None:
        legacy._validate_grid("alpha_grid", self.alpha_grid)
        legacy._validate_grid("joint_alpha_grid", self.joint_alpha_grid)
        integer_fields = (
            self.archive_limit_per_investor,
            self.screen_branches_per_investor,
            self.archive_validation_finalists,
            self.targeted_refinement_starts,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("Archive, branch and finalist counts must be positive.")
        if self.maximum_iterations < 0:
            raise ValueError("Maximum iterations cannot be negative.")
        optional_counts = (
            self.targeted_multistart_interval,
            self.full_audit_interval,
            self.stagnation_iterations_before_targeted,
            self.final_confirmation_audits,
        )
        if any(value < 0 for value in optional_counts):
            raise ValueError("Oracle intervals and audit counts cannot be negative.")
        if self.epsilon_eur_per_day < 0.0 or self.near_target_multiplier < 1.0:
            raise ValueError("Invalid regret target or near-target multiplier.")
        if self.targeted_max_solve_seconds <= 0.0:
            raise ValueError("Targeted solve time limit must be positive.")
        if self.minimum_reduction_eur_per_day < 0.0:
            raise ValueError("Minimum reduction cannot be negative.")
        if not 0.0 <= self.minimum_relative_reduction < 1.0:
            raise ValueError("Minimum relative reduction must lie in [0, 1).")


@dataclass
class CostLedger:
    """Observable solve counts and wall time by algorithmic tier."""

    local_response_batches: int = 0
    local_best_response_calls: int = 0
    targeted_multistart_calls: int = 0
    full_multistart_audits: int = 0
    best_response_refinements: int = 0
    best_response_screen_clears: int = 0
    best_response_reclear_calls: int = 0
    explicit_archive_market_clears: int = 0
    screened_trial_profiles: int = 0
    fully_scored_trial_profiles: int = 0
    local_response_seconds: float = 0.0
    targeted_multistart_seconds: float = 0.0
    full_audit_seconds: float = 0.0
    archive_scoring_seconds: float = 0.0

    @property
    def estimated_total_market_clears(self) -> int:
        return (
            self.best_response_screen_clears
            + self.best_response_reclear_calls
            + self.explicit_archive_market_clears
            + self.full_multistart_audits
        )


@dataclass
class ArchiveEvaluation:
    """Exact re-clear score over a restricted archive of deviations."""

    power: Capacities
    energy: Capacities
    current_profit: dict[str, float]
    regret: dict[str, float]
    candidate_profit: dict[str, float]
    best_branch: dict[str, Branch | None]
    market: object | None
    market_signature: str
    seconds: float
    valid: bool
    error: str = ""

    @property
    def maximum_regret(self) -> float:
        values = tuple(self.regret.values())
        return max(values) if values and all(math.isfinite(v) for v in values) else math.inf

    @property
    def sum_regret(self) -> float:
        values = tuple(self.regret.values())
        return sum(values) if values and all(math.isfinite(v) for v in values) else math.inf


@dataclass
class OracleSearchResult:
    power: Capacities
    energy: Capacities
    archive: dict[str, list[Branch]]
    history: list[dict[str, object]] = field(default_factory=list)
    trial_history: list[dict[str, object]] = field(default_factory=list)
    audit_history: list[legacy.ProfileEvaluation] = field(default_factory=list)
    ledger: CostLedger = field(default_factory=CostLedger)
    iterations: int = 0
    converged: bool = False
    stop_reason: str = ""
    seconds: float = 0.0
    final_restricted_evaluation: ArchiveEvaluation | None = None
    final_audit: legacy.ProfileEvaluation | None = None


ProgressCallback = Callable[[Mapping[str, object]], None]
IterationCallback = Callable[[OracleSearchResult], None]


def _branch_key(data: MarketData, branch: Branch) -> tuple[float, ...]:
    return tuple(
        round(value, 8)
        for node in data.nodes
        for value in (branch.power[node], branch.energy[node])
    )


def merge_response_archive(
    data: MarketData,
    archive: dict[str, list[Branch]],
    responses: Mapping[str, game.BestResponse],
    limit: int,
    *,
    include_screened: bool = True,
) -> None:
    """Keep the newest distinct response branches in a bounded archive."""

    for unit, response in responses.items():
        newest = [
            branch
            for branch in legacy._all_response_branches(data, response)
            if include_screened or branch.source_phase != "screen"
        ]
        combined = newest + archive.get(unit, [])
        unique: list[Branch] = []
        seen: set[tuple[float, ...]] = set()
        for branch in combined:
            key = _branch_key(data, branch)
            if key in seen:
                continue
            seen.add(key)
            unique.append(branch)
            if len(unique) >= limit:
                break
        archive[unit] = unique


def _account_response(ledger: CostLedger, response: game.BestResponse) -> None:
    records = tuple(response.start_records)
    screens = [row for row in records if row.get("phase") == "screen"]
    refinements = [row for row in records if row.get("phase") == "refine"]
    ledger.best_response_screen_clears += len(screens)
    ledger.best_response_refinements += len(refinements)
    ledger.best_response_reclear_calls += sum(
        bool(row.get("has_solution"))
        and isinstance(row.get("exact_profit"), (int, float))
        and math.isfinite(float(row["exact_profit"]))
        for row in refinements
    )


def _account_responses(
    ledger: CostLedger, responses: Mapping[str, game.BestResponse]
) -> None:
    for response in responses.values():
        _account_response(ledger, response)


def evaluate_archive(
    data: MarketData,
    config: game.GameConfig,
    power: Capacities,
    energy: Capacities,
    archive: Mapping[str, Iterable[Branch]],
    ledger: CostLedger,
) -> ArchiveEvaluation:
    """Reclear a profile and a bounded set of known unilateral deviations."""

    started = time.perf_counter()
    try:
        ledger.explicit_archive_market_clears += 1
        market = game.clear(data, config, power, energy)
        current = {
            investor.investor_id: game.recleared_profit(
                market, data, config, investor, power, energy
            )
            for investor in config.investors
        }
        signature = legacy._market_active_set_signature(market, 1.0e-5)
    except Exception as exc:
        return ArchiveEvaluation(
            dict(power), dict(energy), {}, {}, {}, {}, None, "",
            time.perf_counter() - started, False, str(exc)
        )

    regret: dict[str, float] = {}
    candidate_profit: dict[str, float] = {}
    best_branch: dict[str, Branch | None] = {}
    for investor in config.investors:
        unit = investor.investor_id
        best = current[unit]
        winner: Branch | None = None
        for branch in archive.get(unit, ()):
            candidate_power, candidate_energy = game._with_candidate(
                data, power, energy, unit, branch.power, branch.energy
            )
            try:
                ledger.explicit_archive_market_clears += 1
                candidate_market = game.clear(
                    data, config, candidate_power, candidate_energy
                )
                profit = game.recleared_profit(
                    candidate_market,
                    data,
                    config,
                    investor,
                    candidate_power,
                    candidate_energy,
                )
            except Exception:
                continue
            if math.isfinite(profit) and profit > best:
                best, winner = profit, branch
        candidate_profit[unit] = best
        best_branch[unit] = winner
        regret[unit] = max(0.0, best - current[unit])

    elapsed = time.perf_counter() - started
    ledger.archive_scoring_seconds += elapsed
    return ArchiveEvaluation(
        dict(power), dict(energy), current, regret, candidate_profit,
        best_branch, market, signature, elapsed, True
    )


def _restricted_pool(
    config: game.GameConfig,
    archive: Mapping[str, list[Branch]],
    base: ArchiveEvaluation,
    limit: int,
) -> dict[str, list[Branch]]:
    result: dict[str, list[Branch]] = {}
    for unit in config.investor_ids:
        ordered = ([base.best_branch[unit]] if base.best_branch.get(unit) else []) + list(
            archive.get(unit, ())
        )
        seen: set[int] = set()
        result[unit] = []
        for branch in ordered:
            identity = id(branch)
            if identity in seen:
                continue
            seen.add(identity)
            result[unit].append(branch)
            if len(result[unit]) >= limit:
                break
    return result


def _directions(base: ArchiveEvaluation) -> dict[str, Branch]:
    return {
        unit: branch
        for unit, branch in base.best_branch.items()
        if branch is not None and base.regret.get(unit, 0.0) > 0.0
    }


def _trials(
    data: MarketData,
    config: game.GameConfig,
    base: ArchiveEvaluation,
    kind: str,
    grid: Iterable[float],
) -> list[Trial]:
    directions = _directions(base)
    ranked = sorted(
        directions, key=lambda unit: (-base.regret[unit], unit)
    )
    if kind == "simultaneous":
        units = ranked
    elif kind == "line":
        units = ranked[:1]
    elif kind == "joint":
        units = ranked[:2]
    else:
        raise ValueError(f"Unknown trial kind: {kind}")
    if not units:
        return []
    return [
        legacy._interpolated_trial(
            data,
            config,
            base.power,
            base.energy,
            {unit: (directions[unit], alpha) for unit in units},
            kind=kind,
        )
        for alpha in grid
        if alpha > 0.0
    ]


def _evaluation_key(
    evaluation: ArchiveEvaluation,
    base: ArchiveEvaluation,
    data: MarketData,
    config: game.GameConfig,
    label: str,
) -> tuple[float, float, float, str]:
    return (
        evaluation.sum_regret,
        evaluation.maximum_regret,
        legacy._profile_distance(
            data,
            config,
            evaluation.power,
            evaluation.energy,
            base.power,
            base.energy,
        ),
        label,
    )


def _sufficient_reduction(
    before: ArchiveEvaluation,
    after: ArchiveEvaluation,
    search: OracleSearchConfig,
) -> bool:
    required = max(
        search.minimum_reduction_eur_per_day,
        search.minimum_relative_reduction * before.sum_regret,
    )
    return after.valid and after.sum_regret <= before.sum_regret - required


def choose_trial(
    data: MarketData,
    config: game.GameConfig,
    search: OracleSearchConfig,
    base: ArchiveEvaluation,
    archive: Mapping[str, list[Branch]],
    ledger: CostLedger,
    kind: str,
) -> tuple[Trial, ArchiveEvaluation] | None:
    """Two-stage reclear screen; no MPEC is solved here."""

    grid = search.joint_alpha_grid if kind == "joint" else search.alpha_grid
    trials = _trials(data, config, base, kind, grid)
    if not trials:
        return None
    screen_pool = _restricted_pool(
        config, archive, base, search.screen_branches_per_investor
    )
    screened: list[tuple[Trial, ArchiveEvaluation]] = []
    for trial in trials:
        ledger.screened_trial_profiles += 1
        evaluation = evaluate_archive(
            data, config, trial.power, trial.energy, screen_pool, ledger
        )
        if evaluation.valid:
            screened.append((trial, evaluation))
    screened.sort(
        key=lambda item: _evaluation_key(item[1], base, data, config, item[0].label)
    )
    finalists = screened[: search.archive_validation_finalists]
    screen_is_full_archive = all(
        {_branch_key(data, branch) for branch in screen_pool[unit]}
        == {_branch_key(data, branch) for branch in archive.get(unit, ())}
        for unit in config.investor_ids
    )
    fully_scored: list[tuple[Trial, ArchiveEvaluation]] = []
    for trial, screened_evaluation in finalists:
        ledger.fully_scored_trial_profiles += 1
        fully_scored.append(
            (trial, screened_evaluation)
            if screen_is_full_archive
            else (
                trial,
                evaluate_archive(
                    data, config, trial.power, trial.energy, archive, ledger
                ),
            )
        )
    if not fully_scored:
        return None
    return min(
        fully_scored,
        key=lambda item: _evaluation_key(item[1], base, data, config, item[0].label),
    )


def _evaluation_record(
    iteration: int,
    evaluation: ArchiveEvaluation,
    archive: Mapping[str, list[Branch]],
    *,
    selected_step: str,
    selection_reason: str,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "maximum_restricted_regret_eur_per_day": evaluation.maximum_regret,
        "sum_restricted_regret_eur_per_day": evaluation.sum_regret,
        "restricted_regret_eur_per_day": dict(evaluation.regret),
        "current_profit_eur_per_day": dict(evaluation.current_profit),
        "archive_size": {unit: len(items) for unit, items in archive.items()},
        "market_signature": evaluation.market_signature,
        "selected_step": selected_step,
        "selection_reason": selection_reason,
        "valid": evaluation.valid,
    }


def _trial_record(
    iteration: int, kind: str, selected: tuple[Trial, ArchiveEvaluation] | None
) -> dict[str, object]:
    if selected is None:
        return {"iteration": iteration, "kind": kind, "found": False}
    trial, evaluation = selected
    return {
        "iteration": iteration,
        "kind": kind,
        "found": True,
        "label": trial.label,
        "maximum_restricted_regret_eur_per_day": evaluation.maximum_regret,
        "sum_restricted_regret_eur_per_day": evaluation.sum_regret,
        "directions": dict(trial.directions),
    }


def _full_audit(
    data: MarketData,
    config: game.GameConfig,
    search: OracleSearchConfig,
    power: Capacities,
    energy: Capacities,
    ledger: CostLedger,
) -> legacy.ProfileEvaluation:
    started = time.perf_counter()
    audit_search = legacy.SearchConfig(
        epsilon_eur_per_day=search.epsilon_eur_per_day,
        maximum_branches_per_investor=search.archive_limit_per_investor,
        cycle_diagnostic_steps=0,
        resolution_probe_mw=0.0,
    )
    evaluation = legacy.evaluate_profile(
        data, config, audit_search, power, energy
    )
    ledger.full_multistart_audits += 1
    ledger.full_audit_seconds += time.perf_counter() - started
    _account_responses(ledger, evaluation.responses)
    return evaluation


def _merge_audit_archive(
    data: MarketData,
    archive: dict[str, list[Branch]],
    audit: legacy.ProfileEvaluation,
    limit: int,
) -> None:
    synthetic = {
        unit: audit.responses[unit]
        for unit in audit.responses
    }
    merge_response_archive(data, archive, synthetic, limit)


def run_oracle_efficient_search(
    data: MarketData,
    config: game.GameConfig,
    search: OracleSearchConfig,
    *,
    initial: game.GameState | None = None,
    on_iteration: IterationCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> OracleSearchResult:
    """Run the two-loop search from the supplied or symmetric profile."""

    game.validate(data, config)
    search.validate()
    started = time.perf_counter()
    state = initial if initial is not None else game.initial_state(data, config)
    archive = {unit: [] for unit in config.investor_ids}
    result = OracleSearchResult(dict(state.power), dict(state.energy), archive)
    stagnation = 0
    last_full_audit_key: tuple[float, ...] | None = None

    for iteration in range(search.maximum_iterations + 1):
        if on_progress is not None:
            on_progress({"iteration": iteration, "phase": "local_parallel_refresh"})
        tick = time.perf_counter()
        responses = game.solve_all(
            data, config, result.power, result.energy, use_multistart=False
        )
        result.ledger.local_response_batches += 1
        result.ledger.local_best_response_calls += len(config.investors)
        result.ledger.local_response_seconds += time.perf_counter() - tick
        _account_responses(result.ledger, responses)
        merge_response_archive(
            data,
            result.archive,
            responses,
            search.archive_limit_per_investor,
            include_screened=False,
        )
        evaluation = evaluate_archive(
            data,
            config,
            result.power,
            result.energy,
            result.archive,
            result.ledger,
        )
        if not evaluation.valid:
            result.stop_reason = f"archive evaluation failed: {evaluation.error}"
            break

        scheduled_targeted = (
            iteration > 0
            and iteration < search.maximum_iterations
            and search.targeted_multistart_interval > 0
            and iteration % search.targeted_multistart_interval == 0
        )
        stalled_targeted = (
            iteration < search.maximum_iterations
            and search.stagnation_iterations_before_targeted > 0
            and stagnation >= search.stagnation_iterations_before_targeted
        )
        if scheduled_targeted or stalled_targeted:
            unit = max(
                config.investor_ids,
                key=lambda item: (evaluation.regret[item], item),
            )
            investor = next(i for i in config.investors if i.investor_id == unit)
            if on_progress is not None:
                on_progress(
                    {
                        "iteration": iteration,
                        "phase": "targeted_multistart",
                        "investor": unit,
                    }
                )
            tick = time.perf_counter()
            targeted_config = replace(
                config,
                refinement_starts=search.targeted_refinement_starts,
                always_refine_incumbent=False,
                solver=replace(
                    config.solver,
                    max_seconds=min(
                        config.solver.max_seconds,
                        search.targeted_max_solve_seconds,
                    ),
                ),
            )
            response = game.best_response(
                data,
                targeted_config,
                investor,
                result.power,
                result.energy,
                use_multistart=True,
            )
            result.ledger.targeted_multistart_calls += 1
            result.ledger.targeted_multistart_seconds += time.perf_counter() - tick
            _account_response(result.ledger, response)
            merge_response_archive(
                data,
                result.archive,
                {unit: response},
                search.archive_limit_per_investor,
            )
            evaluation = evaluate_archive(
                data,
                config,
                result.power,
                result.energy,
                result.archive,
                result.ledger,
            )
            stagnation = 0

        should_full_audit = (
            iteration > 0
            and (
                (
                    search.full_audit_interval > 0
                    and iteration % search.full_audit_interval == 0
                )
                or evaluation.maximum_regret
                <= search.near_target_multiplier * search.epsilon_eur_per_day
            )
        )
        if should_full_audit:
            key = legacy._profile_key(data, config, result.power, result.energy)
            if key != last_full_audit_key:
                if on_progress is not None:
                    on_progress({"iteration": iteration, "phase": "full_multistart_audit"})
                audit = _full_audit(
                    data,
                    config,
                    search,
                    result.power,
                    result.energy,
                    result.ledger,
                )
                last_full_audit_key = key
                result.audit_history.append(audit)
                result.final_audit = audit
                _merge_audit_archive(
                    data,
                    result.archive,
                    audit,
                    search.archive_limit_per_investor,
                )
                evaluation = evaluate_archive(
                    data,
                    config,
                    result.power,
                    result.energy,
                    result.archive,
                    result.ledger,
                )
                if audit.certifiable and audit.maximum_regret <= search.epsilon_eur_per_day:
                    confirmations = [audit]
                    for _ in range(max(0, search.final_confirmation_audits - 1)):
                        confirmations.append(
                            _full_audit(
                                data,
                                config,
                                search,
                                result.power,
                                result.energy,
                                result.ledger,
                            )
                        )
                    result.audit_history.extend(confirmations[1:])
                    result.final_audit = max(
                        confirmations, key=lambda item: item.maximum_regret
                    )
                    if all(
                        item.certifiable
                        and item.maximum_regret <= search.epsilon_eur_per_day
                        for item in confirmations
                    ):
                        result.converged = True
                        result.stop_reason = (
                            f"maximum regret <= {search.epsilon_eur_per_day:g} EUR/day "
                            f"in {len(confirmations)} full audits"
                        )

        result.final_restricted_evaluation = evaluation
        result.history.append(
            _evaluation_record(
                iteration,
                evaluation,
                result.archive,
                selected_step="current_profile",
                selection_reason="parallel local refresh and archived deviation re-clear",
            )
        )
        result.iterations = iteration
        result.seconds = time.perf_counter() - started
        if on_iteration is not None:
            on_iteration(result)
        if result.converged:
            break
        if iteration == search.maximum_iterations:
            result.stop_reason = f"maximum {search.maximum_iterations} iterations reached"
            break

        selected: tuple[Trial, ArchiveEvaluation] | None = None
        selected_kind = ""
        for kind in ("simultaneous", "line", "joint"):
            candidate = choose_trial(
                data,
                config,
                search,
                evaluation,
                result.archive,
                result.ledger,
                kind,
            )
            result.trial_history.append(_trial_record(iteration + 1, kind, candidate))
            if candidate is not None and _sufficient_reduction(
                evaluation, candidate[1], search
            ):
                selected, selected_kind = candidate, kind
                break
        if selected is None:
            stagnation += 1
            if stagnation > search.stagnation_iterations_before_targeted:
                result.stop_reason = (
                    "stagnation: no archived-regret-reducing simultaneous, "
                    "one-player or joint step"
                )
                break
            continue

        trial, accepted = selected
        result.power, result.energy = dict(trial.power), dict(trial.energy)
        result.final_restricted_evaluation = accepted
        result.history[-1]["selected_step"] = trial.label
        result.history[-1]["selection_reason"] = (
            f"{selected_kind} step reduced archived Nikaido-Isoda sum"
        )
        stagnation = 0

    if not result.converged and search.final_confirmation_audits > 0:
        key = legacy._profile_key(data, config, result.power, result.energy)
        if key != last_full_audit_key:
            if on_progress is not None:
                on_progress({"iteration": result.iterations, "phase": "final_full_audit"})
            audit = _full_audit(
                data,
                config,
                search,
                result.power,
                result.energy,
                result.ledger,
            )
            result.audit_history.append(audit)
            result.final_audit = audit
            if audit.certifiable and audit.maximum_regret <= search.epsilon_eur_per_day:
                confirmations = [audit]
                for _ in range(search.final_confirmation_audits - 1):
                    confirmations.append(
                        _full_audit(
                            data,
                            config,
                            search,
                            result.power,
                            result.energy,
                            result.ledger,
                        )
                    )
                result.audit_history.extend(confirmations[1:])
                result.final_audit = max(
                    confirmations, key=lambda item: item.maximum_regret
                )
                if all(
                    item.certifiable
                    and item.maximum_regret <= search.epsilon_eur_per_day
                    for item in confirmations
                ):
                    result.converged = True
                    result.stop_reason = (
                        f"maximum regret <= {search.epsilon_eur_per_day:g} EUR/day "
                        f"in {len(confirmations)} final audits"
                    )
            elif search.maximum_iterations == 0:
                result.stop_reason = (
                    "audit-only evaluation completed; "
                    f"maximum regret={audit.maximum_regret:.6g} EUR/day"
                )
    result.seconds = time.perf_counter() - started
    return result


def _parse_grid(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Grid must be comma-separated numbers.") from exc
    try:
        legacy._validate_grid("grid", values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return values


def _read_capacities(path: Path | None):
    if path is None:
        return None
    power: Capacities = {}
    energy: Capacities = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row["investor"], row["node"]
            power[key] = float(row["power_mw"])
            energy[key] = float(row["energy_mwh"])
    if not power:
        raise ValueError(f"No capacity rows in {path}.")
    return power, energy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-capacities", type=Path, default=None)
    parser.add_argument("--initial-power-mw", type=float, default=5.0)
    parser.add_argument("--initial-ratio-hours", type=float, default=3.0)
    parser.add_argument("--epsilon-eur-per-day", type=float, default=20.0)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--alpha-grid", type=_parse_grid, default=(0.0, 0.1, 0.25, 0.5, 1.0))
    parser.add_argument("--joint-alpha-grid", type=_parse_grid, default=(0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--archive-limit", type=int, default=3)
    parser.add_argument("--screen-branches", type=int, default=1)
    parser.add_argument("--archive-finalists", type=int, default=1)
    parser.add_argument("--targeted-multistart-interval", type=int, default=8)
    parser.add_argument("--targeted-refinement-starts", type=int, default=1)
    parser.add_argument("--targeted-max-solve-seconds", type=float, default=120.0)
    parser.add_argument("--full-audit-interval", type=int, default=0)
    parser.add_argument("--stagnation-trigger", type=int, default=1)
    parser.add_argument("--near-target-multiplier", type=float, default=5.0)
    parser.add_argument("--minimum-reduction-eur-per-day", type=float, default=1.0)
    parser.add_argument("--minimum-relative-reduction", type=float, default=1.0e-4)
    parser.add_argument("--final-confirmation-audits", type=int, default=2)
    parser.add_argument("--refinement-starts", type=int, default=3)
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--node-limit-mw", type=float, default=None)
    parser.add_argument("--formulation", choices=("relaxed-kkt", "strong-duality"), default="relaxed-kkt")
    parser.add_argument("--complementarity-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--audit-reclear-gap-tolerance-eur-per-day", type=float, default=2.0)
    parser.add_argument("--ipopt-linear-solver", default="ma57")
    parser.add_argument("--max-solver-iterations", type=int, default=3_000)
    parser.add_argument("--max-solve-seconds", type=float, default=600.0)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def _build_configs(args: argparse.Namespace, data: MarketData):
    config = game.GameConfig(
        investors=three_investors(data),
        lower_level=args.formulation,
        complementarity_epsilon=args.complementarity_epsilon,
        refinement_starts=args.refinement_starts,
        proximal_penalty=0.0,
        uniform_node_connection_limit_mw=args.node_limit_mw,
        symmetric_initialization=True,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        audit_profit_tolerance_eur_per_day=args.epsilon_eur_per_day,
        audit_reclear_gap_tolerance_eur_per_day=args.audit_reclear_gap_tolerance_eur_per_day,
        parallel_workers=args.parallel_workers,
        solver=SolverSettings(
            linear_solver=args.ipopt_linear_solver,
            max_iterations=args.max_solver_iterations,
            max_seconds=args.max_solve_seconds,
            tolerance=args.solver_tolerance,
            tee=args.tee,
        ),
    )
    search = OracleSearchConfig(
        epsilon_eur_per_day=args.epsilon_eur_per_day,
        alpha_grid=args.alpha_grid,
        joint_alpha_grid=args.joint_alpha_grid,
        maximum_iterations=args.max_iterations,
        archive_limit_per_investor=args.archive_limit,
        screen_branches_per_investor=args.screen_branches,
        archive_validation_finalists=args.archive_finalists,
        targeted_multistart_interval=args.targeted_multistart_interval,
        targeted_refinement_starts=args.targeted_refinement_starts,
        targeted_max_solve_seconds=args.targeted_max_solve_seconds,
        full_audit_interval=args.full_audit_interval,
        stagnation_iterations_before_targeted=args.stagnation_trigger,
        near_target_multiplier=args.near_target_multiplier,
        minimum_reduction_eur_per_day=args.minimum_reduction_eur_per_day,
        minimum_relative_reduction=args.minimum_relative_reduction,
        final_confirmation_audits=args.final_confirmation_audits,
    )
    return config, search


def _flat_history(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        row = {key: value for key, value in record.items() if not isinstance(value, dict)}
        for group in ("restricted_regret_eur_per_day", "current_profit_eur_per_day"):
            for unit, value in record[group].items():
                row[f"{group}_{unit}"] = value
        for unit, value in record["archive_size"].items():
            row[f"archive_size_{unit}"] = value
        rows.append(row)
    return rows


def _audit_payload(audit: legacy.ProfileEvaluation) -> dict[str, object]:
    return legacy._serializable_evaluation(audit)


def write_outputs(
    out: Path,
    data: MarketData,
    data_path: Path,
    config: game.GameConfig,
    search: OracleSearchConfig,
    result: OracleSearchResult,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = reporting.run_config(
        config, data_path, hashlib.sha256(data_path.read_bytes()).hexdigest()
    )
    payload.update(
        algorithm="oracle-efficient-memory-augmented-nikaido-isoda",
        search=asdict(search),
        python_version=platform.python_version(),
        pyomo_version=pyomo.__version__,
        code_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(game.__file__), Path(legacy.__file__))
        },
    )
    reporting.write_json(out / "run_config.json", payload)
    reporting.write_rows(
        out / "current_capacities.csv",
        reporting.capacity_rows(data, config.investors, result.power, result.energy),
    )
    reporting.write_rows(out / "history.csv", _flat_history(result.history))
    reporting.write_json(out / "trial_history.json", result.trial_history)
    reporting.write_json(
        out / "archive.json",
        {
            unit: [asdict(branch) for branch in branches]
            for unit, branches in result.archive.items()
        },
    )
    reporting.write_json(
        out / "audit_history.json",
        [_audit_payload(audit) for audit in result.audit_history],
    )
    reporting.write_json(out / "cost_ledger.json", asdict(result.ledger) | {
        "estimated_total_market_clears": result.ledger.estimated_total_market_clears
    })
    restricted = result.final_restricted_evaluation
    audit = result.final_audit
    summary = {
        "algorithm": "oracle-efficient-memory-augmented-nikaido-isoda",
        "converged": result.converged,
        "numerically_validated_epsilon_equilibrium": result.converged,
        "global_best_response_certified": False,
        "epsilon_eur_per_day": search.epsilon_eur_per_day,
        "iterations": result.iterations,
        "stop_reason": result.stop_reason,
        "runtime_seconds": result.seconds,
        "restricted_regret_is_lower_bound": True,
        "final_restricted_maximum_regret_eur_per_day": (
            restricted.maximum_regret if restricted is not None else None
        ),
        "final_restricted_sum_regret_eur_per_day": (
            restricted.sum_regret if restricted is not None else None
        ),
        "final_audited_maximum_regret_eur_per_day": (
            audit.maximum_regret if audit is not None else None
        ),
        "final_audited_sum_regret_eur_per_day": (
            audit.sum_regret if audit is not None else None
        ),
        "final_audit_certifiable": audit.certifiable if audit is not None else False,
        "cost": asdict(result.ledger) | {
            "estimated_total_market_clears": result.ledger.estimated_total_market_clears
        },
    }
    reporting.write_json(out / "summary.json", summary)
    if restricted is not None and restricted.market is not None:
        reporting.write_rows(
            out / "final_market.csv", reporting.market_rows(restricted.market, data)
        )
        reporting.write_rows(
            out / "profit_decomposition.csv",
            [
                asdict(settle(restricted.market, data, investor, result.power, result.energy))
                for investor in config.investors
            ],
        )


def main() -> int:
    args = _parse_args()
    data_path = args.data.resolve()
    data = load_market_data(data_path)
    validate_afrr(data)
    config, search = _build_configs(args, data)
    initial = game.initial_state(data, config, _read_capacities(args.initial_capacities))
    out = args.output_dir.resolve()

    def progress(event: Mapping[str, object]) -> None:
        print(" ".join(f"{key}={value}" for key, value in event.items()), flush=True)

    def checkpoint(partial: OracleSearchResult) -> None:
        evaluation = partial.final_restricted_evaluation
        if evaluation is not None:
            print(
                f"iteration={partial.iterations:03d} "
                f"restricted_max={evaluation.maximum_regret:.6g} "
                f"restricted_sum={evaluation.sum_regret:.6g}",
                flush=True,
            )
        write_outputs(out, data, data_path, config, search, partial)

    print(
        "Oracle-efficient NI search: "
        f"symmetric={args.initial_capacities is None}, "
        f"epsilon={search.epsilon_eur_per_day:g} EUR/day",
        flush=True,
    )
    result = run_oracle_efficient_search(
        data,
        config,
        search,
        initial=initial,
        on_iteration=checkpoint,
        on_progress=progress,
    )
    write_outputs(out, data, data_path, config, search, result)
    print(
        reporting.json_dumps(
            {
                "converged": result.converged,
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
                "runtime_seconds": result.seconds,
                "output_dir": str(out),
            },
            indent=2,
        )
    )
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
