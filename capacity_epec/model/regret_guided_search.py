"""Regret-guided, branch-aware equilibrium search for the capacity EPEC.

This is the reproducible search motivated by
``workflow/summary_2026-09-08_14-05.md``.  It deliberately lives beside the
legacy Jacobi and Gauss--Seidel runners.  The Nikaido--Isoda sum of exact-
recleared regrets guides the search; maximum regret is reserved for the
epsilon-equilibrium acceptance test.

Run the maintained three-investor case from its declared symmetric start with

    python model/regret_guided_search.py

The search is deterministic.  IPOPT still supplies local NLP solutions, so a
successful run is a numerically validated epsilon-equilibrium under the
documented multistart search, not a global-optimality certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pyomo
import pyomo.environ as pyo

import capacity_game as game
import reporting
from investors import three_investors
from iso_market import settle
from market_data import DEFAULT_DATA_PATH, MarketData, load_market_data, validate_afrr
from solvers import SolverSettings, ipopt_path


Capacities = game.Capacities
NodalMap = game.NodalMap
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "regret_guided"


@dataclass(frozen=True)
class SearchConfig:
    """Frozen thresholds and grids that define the outer search."""

    epsilon_eur_per_day: float = 20.0
    branch_profit_tolerance_eur_per_day: float = 1.0
    alpha_grid: tuple[float, ...] = tuple(i / 10 for i in range(11))
    joint_alpha_grid: tuple[float, ...] = tuple(i / 4 for i in range(5))
    full_validation_finalists: int = 3
    maximum_branches_per_investor: int = 3
    maximum_outer_iterations: int = 50
    maximum_restarts: int = 1
    fallback_mode: str = "all"
    minimum_reduction_eur_per_day: float = 1.0
    minimum_relative_reduction: float = 1.0e-4
    relative_regret_floor_eur_per_day: float = 1.0
    confirmation_audits: int = 2
    breakpoint_bisection_rounds: int = 4
    active_set_tolerance: float = 1.0e-5
    cycle_diagnostic_steps: int = 6
    cycle_capacity_tolerance: float = 0.1
    resolution_probe_mw: float = 1.0e-3

    def validate(self) -> None:
        if self.epsilon_eur_per_day < 0.0:
            raise ValueError("epsilon_eur_per_day cannot be negative.")
        if self.branch_profit_tolerance_eur_per_day < 0.0:
            raise ValueError("branch_profit_tolerance_eur_per_day cannot be negative.")
        if self.full_validation_finalists <= 0:
            raise ValueError("full_validation_finalists must be positive.")
        if self.maximum_branches_per_investor <= 0:
            raise ValueError("maximum_branches_per_investor must be positive.")
        if self.maximum_outer_iterations <= 0:
            raise ValueError("maximum_outer_iterations must be positive.")
        if self.maximum_restarts < 0:
            raise ValueError("maximum_restarts cannot be negative.")
        if self.fallback_mode not in {"all", "none"}:
            raise ValueError("fallback_mode must be 'all' or 'none'.")
        if self.minimum_reduction_eur_per_day < 0.0:
            raise ValueError("minimum_reduction_eur_per_day cannot be negative.")
        if not 0.0 <= self.minimum_relative_reduction < 1.0:
            raise ValueError("minimum_relative_reduction must lie in [0, 1).")
        if self.relative_regret_floor_eur_per_day <= 0.0:
            raise ValueError("relative_regret_floor_eur_per_day must be positive.")
        if self.confirmation_audits < 2:
            raise ValueError("At least two independent final audits are required.")
        if self.breakpoint_bisection_rounds < 0:
            raise ValueError("breakpoint_bisection_rounds cannot be negative.")
        if self.active_set_tolerance <= 0.0:
            raise ValueError("active_set_tolerance must be positive.")
        if self.cycle_diagnostic_steps < 0 or self.cycle_capacity_tolerance < 0.0:
            raise ValueError("Cycle-diagnostic settings cannot be negative.")
        if self.resolution_probe_mw < 0.0:
            raise ValueError("resolution_probe_mw cannot be negative.")
        _validate_grid("alpha_grid", self.alpha_grid)
        _validate_grid("joint_alpha_grid", self.joint_alpha_grid)


@dataclass(frozen=True)
class Branch:
    """One exact-priced best-response branch retained near the best found."""

    investor_id: str
    label: str
    power: NodalMap
    energy: NodalMap
    exact_profit_eur_per_day: float
    local_optimal: bool
    source_phase: str


@dataclass
class ProfileEvaluation:
    """A fresh zero-proximal, multistart evaluation of one frozen profile."""

    power: Capacities
    energy: Capacities
    current_profit: dict[str, float]
    regret: dict[str, float]
    relative_regret: dict[str, float]
    branches: dict[str, tuple[Branch, ...]]
    all_candidates: dict[str, tuple[Branch, ...]]
    responses: dict[str, game.BestResponse]
    valid: bool
    diagnostics_valid: bool
    market: object | None
    seconds: float
    error: str = ""

    @property
    def maximum_regret(self) -> float:
        values = tuple(self.regret.values())
        return max(values) if values and all(math.isfinite(v) for v in values) else math.inf

    @property
    def sum_regret(self) -> float:
        values = tuple(self.regret.values())
        return sum(values) if values and all(math.isfinite(v) for v in values) else math.inf

    @property
    def certifiable(self) -> bool:
        return self.valid and self.diagnostics_valid


@dataclass(frozen=True)
class Trial:
    """A line- or two-direction profile to screen."""

    label: str
    kind: str
    power: Capacities
    energy: Capacities
    directions: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class ScreenResult:
    trial: Trial
    valid: bool
    maximum_regret: float
    sum_regret: float
    regret: Mapping[str, float]
    current_profit: Mapping[str, float]
    candidate_profit: Mapping[str, float]
    market_signature: str
    seconds: float
    error: str = ""


@dataclass
class SearchResult:
    """Complete result and audit trail of one outer search."""

    power: Capacities
    energy: Capacities
    evaluations: list[ProfileEvaluation]
    iteration_records: list[dict[str, object]]
    trial_records: list[dict[str, object]]
    confirmation_evaluations: list[ProfileEvaluation]
    cycle_records: list[dict[str, object]]
    resolution_records: list[dict[str, object]]
    regret_normalization_eur_per_day: dict[str, float]
    empirical_regret_resolution_eur_per_day: float | None
    converged: bool
    stop_reason: str
    outer_iterations: int
    restarts_used: int
    seconds: float

    @property
    def final_evaluation(self) -> ProfileEvaluation:
        return self.evaluations[-1]


IterationCallback = Callable[[SearchResult], None]
ProgressCallback = Callable[[Mapping[str, object]], None]


def _validate_grid(name: str, grid: Iterable[float]) -> None:
    values = tuple(grid)
    if not values or any(not math.isfinite(a) or a < 0.0 or a > 1.0 for a in values):
        raise ValueError(f"{name} must contain finite values in [0, 1].")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} must be strictly increasing and duplicate-free.")
    if values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError(f"{name} must include both 0 and 1.")


def _profile_key(data: MarketData, config: game.GameConfig, power, energy) -> tuple[float, ...]:
    return tuple(
        round(float(side[unit, node]), 10)
        for unit in config.investor_ids
        for node in data.nodes
        for side in (power, energy)
    )


def _profile_distance(
    data: MarketData,
    config: game.GameConfig,
    left_power: Capacities,
    left_energy: Capacities,
    right_power: Capacities,
    right_energy: Capacities,
) -> float:
    return math.sqrt(
        sum(
            (left_power[unit, node] - right_power[unit, node]) ** 2
            + (left_energy[unit, node] - right_energy[unit, node]) ** 2
            for unit in config.investor_ids
            for node in data.nodes
        )
    )


def _aggregate_profile_distance(
    data: MarketData,
    config: game.GameConfig,
    left_power: Capacities,
    left_energy: Capacities,
    right_power: Capacities,
    right_energy: Capacities,
) -> float:
    return math.sqrt(
        sum(
            (
                sum(left_power[unit, node] for unit in config.investor_ids)
                - sum(right_power[unit, node] for unit in config.investor_ids)
            )
            ** 2
            + (
                sum(left_energy[unit, node] for unit in config.investor_ids)
                - sum(right_energy[unit, node] for unit in config.investor_ids)
            )
            ** 2
            for node in data.nodes
        )
    )


def _all_response_branches(
    data: MarketData, response: game.BestResponse
) -> tuple[Branch, ...]:
    """Return every unique exact-priced candidate generated by one BR audit."""

    raw: list[Branch] = []
    for record in response.start_records:
        exact = record.get("exact_profit")
        power = record.get("power")
        energy = record.get("energy")
        if (
            record.get("has_solution")
            and isinstance(exact, (int, float))
            and math.isfinite(float(exact))
            and isinstance(power, dict)
            and isinstance(energy, dict)
            and all(node in power and node in energy for node in data.nodes)
        ):
            raw.append(
                Branch(
                    investor_id=response.investor_id,
                    label=str(record.get("start", "candidate")),
                    power={node: float(power[node]) for node in data.nodes},
                    energy={node: float(energy[node]) for node in data.nodes},
                    exact_profit_eur_per_day=float(exact),
                    local_optimal=bool(record.get("local_optimal", False)),
                    source_phase=str(record.get("phase", "unknown")),
                )
            )
    if response.outcome.has_solution and math.isfinite(response.recleared_profit_eur_per_day):
        raw.append(
            Branch(
                investor_id=response.investor_id,
                label=response.start_label,
                power=dict(response.power),
                energy=dict(response.energy),
                exact_profit_eur_per_day=float(response.recleared_profit_eur_per_day),
                local_optimal=response.outcome.optimal,
                source_phase="selected",
            )
        )
    if not raw:
        return ()

    unique: dict[tuple[float, ...], Branch] = {}
    for branch in raw:
        key = tuple(
            round(value, 9)
            for node in data.nodes
            for value in (branch.power[node], branch.energy[node])
        )
        incumbent = unique.get(key)
        if incumbent is None or (
            branch.exact_profit_eur_per_day,
            branch.local_optimal,
            branch.source_phase == "selected",
            branch.label,
        ) > (
            incumbent.exact_profit_eur_per_day,
            incumbent.local_optimal,
            incumbent.source_phase == "selected",
            incumbent.label,
        ):
            unique[key] = branch

    candidates = list(unique.values())
    candidates.sort(
        key=lambda branch: (
            -branch.exact_profit_eur_per_day,
            -int(branch.local_optimal),
            branch.label,
            branch.source_phase,
        )
    )
    return tuple(candidates)


def _retain_near_best(
    candidates: tuple[Branch, ...], search: SearchConfig
) -> tuple[Branch, ...]:
    if not candidates:
        return ()
    best = candidates[0].exact_profit_eur_per_day
    return tuple(
        branch
        for branch in candidates
        if branch.exact_profit_eur_per_day
        >= best - search.branch_profit_tolerance_eur_per_day
    )[: search.maximum_branches_per_investor]


def _merge_candidate_pool(
    pool: dict[str, list[Branch]], evaluations: Iterable[ProfileEvaluation]
) -> None:
    """Append every retained BR branch ever found; never discard a stale one."""

    for evaluation in evaluations:
        for unit, candidates in evaluation.branches.items():
            known = {
                tuple(
                    round(value, 9)
                    for node in sorted(branch.power)
                    for value in (branch.power[node], branch.energy[node])
                )
                for branch in pool.setdefault(unit, [])
            }
            for branch in candidates:
                key = tuple(
                    round(value, 9)
                    for node in sorted(branch.power)
                    for value in (branch.power[node], branch.energy[node])
                )
                if key not in known:
                    pool[unit].append(branch)
                    known.add(key)


def evaluate_profile(
    data: MarketData,
    config: game.GameConfig,
    search: SearchConfig,
    power: Capacities,
    energy: Capacities,
    regret_normalization: Mapping[str, float] | None = None,
) -> ProfileEvaluation:
    """Perform a fresh zero-proximal, exact-recleared multistart audit."""

    started = time.perf_counter()
    audit_config = replace(config, proximal_penalty=0.0)
    try:
        market = game.clear(data, audit_config, power, energy)
        current_profit = {
            investor.investor_id: game.recleared_profit(
                market, data, audit_config, investor, power, energy
            )
            for investor in audit_config.investors
        }
    except Exception as exc:
        return ProfileEvaluation(
            power=dict(power),
            energy=dict(energy),
            current_profit={},
            regret={},
            relative_regret={},
            branches={},
            all_candidates={},
            responses={},
            valid=False,
            diagnostics_valid=False,
            market=None,
            seconds=time.perf_counter() - started,
            error=f"incumbent reclear failed: {exc}",
        )

    responses = game.solve_all(data, audit_config, dict(power), dict(energy), True)
    all_candidates = {
        unit: _all_response_branches(data, responses[unit])
        for unit in audit_config.investor_ids
    }
    branches = {
        unit: _retain_near_best(all_candidates[unit], search)
        for unit in audit_config.investor_ids
    }
    valid = all(
        unit in responses
        and responses[unit].outcome.has_solution
        and branches[unit]
        and math.isfinite(current_profit[unit])
        for unit in audit_config.investor_ids
    )
    regret: dict[str, float] = {}
    relative: dict[str, float] = {}
    if valid:
        for unit in audit_config.investor_ids:
            best = max(branch.exact_profit_eur_per_day for branch in branches[unit])
            regret[unit] = max(0.0, best - current_profit[unit])
            scale = (
                regret_normalization[unit]
                if regret_normalization is not None
                else max(
                    abs(current_profit[unit]),
                    search.relative_regret_floor_eur_per_day,
                )
            )
            relative[unit] = regret[unit] / scale
    else:
        regret = {unit: math.inf for unit in audit_config.investor_ids}
        relative = {unit: math.inf for unit in audit_config.investor_ids}

    diagnostics_valid = valid and all(
        _response_diagnostics_valid(responses[unit], audit_config)
        for unit in audit_config.investor_ids
    )
    return ProfileEvaluation(
        power=dict(power),
        energy=dict(energy),
        current_profit=current_profit,
        regret=regret,
        relative_regret=relative,
        branches=branches,
        all_candidates=all_candidates,
        responses=responses,
        valid=valid,
        diagnostics_valid=diagnostics_valid,
        market=market,
        seconds=time.perf_counter() - started,
    )


def _response_diagnostics_valid(
    response: game.BestResponse, config: game.GameConfig
) -> bool:
    numbers = (
        response.embedded_profit_eur_per_day,
        response.recleared_profit_eur_per_day,
        response.max_complementarity_violation,
        response.max_bound_violation,
        response.primal_dual_gap_eur_per_day,
    )
    return (
        response.outcome.has_solution
        and response.outcome.optimal
        and all(math.isfinite(value) for value in numbers)
        and abs(
            response.embedded_profit_eur_per_day
            - response.recleared_profit_eur_per_day
        )
        <= config.audit_reclear_gap_tolerance_eur_per_day
        and response.max_complementarity_violation <= config.solver.tolerance
        and response.max_bound_violation <= config.solver.tolerance
    )


def _interpolated_trial(
    data: MarketData,
    config: game.GameConfig,
    power: Capacities,
    energy: Capacities,
    choices: Mapping[str, tuple[Branch, float]],
    *,
    kind: str,
) -> Trial:
    trial_power, trial_energy = dict(power), dict(energy)
    directions: dict[str, dict[str, object]] = {}
    labels = []
    for unit in config.investor_ids:
        if unit not in choices:
            continue
        branch, alpha = choices[unit]
        for node in data.nodes:
            key = unit, node
            trial_power[key] = (1.0 - alpha) * power[key] + alpha * branch.power[node]
            trial_energy[key] = (1.0 - alpha) * energy[key] + alpha * branch.energy[node]
            if trial_power[key] < config.cleanup_tolerance:
                trial_power[key] = 0.0
                trial_energy[key] = 0.0
        directions[unit] = {"branch": branch.label, "alpha": alpha}
        labels.append(f"{unit}:{branch.label}@{alpha:g}")
    return Trial(
        label="|".join(labels),
        kind=kind,
        power=trial_power,
        energy=trial_energy,
        directions=directions,
    )


def _market_active_set_signature(market: object, tolerance: float) -> str:
    """Hash the lower-level active set used to locate price-support cliffs."""

    tokens: list[str] = []
    names = (
        "generation_capacity_bound",
        "line_upper_bound",
        "line_lower_bound",
        "charge_power_bound",
        "discharge_power_bound",
        "soc_capacity_bound",
    )
    for name in names:
        component = getattr(market, name, None)
        if component is None:
            continue
        for index, constraint in component.items():
            body = float(pyo.value(constraint.body))
            lower = None if constraint.lower is None else float(pyo.value(constraint.lower))
            upper = None if constraint.upper is None else float(pyo.value(constraint.upper))
            if lower is not None and abs(body - lower) <= tolerance:
                status = "L"
            elif upper is not None and abs(upper - body) <= tolerance:
                status = "U"
            else:
                status = "I"
            tokens.append(f"{name}:{index}:{status}")
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()[:16]


def _deduplicate_trials(
    data: MarketData, config: game.GameConfig, trials: Iterable[Trial]
) -> list[Trial]:
    unique: dict[tuple[float, ...], Trial] = {}
    for trial in trials:
        key = _profile_key(data, config, trial.power, trial.energy)
        if key not in unique or trial.label < unique[key].label:
            unique[key] = trial
    return list(unique.values())


def _line_trials(
    data: MarketData,
    config: game.GameConfig,
    search: SearchConfig,
    evaluation: ProfileEvaluation,
    investor_id: str,
) -> list[Trial]:
    return _deduplicate_trials(
        data,
        config,
        (
            _interpolated_trial(
                data,
                config,
                evaluation.power,
                evaluation.energy,
                {investor_id: (branch, alpha)},
                kind="line",
            )
            for branch in evaluation.branches[investor_id]
            for alpha in search.alpha_grid
            if alpha > 0.0
        ),
    )


def _simultaneous_branch_sets(
    config: game.GameConfig, evaluation: ProfileEvaluation
) -> list[dict[str, Branch]]:
    """Primary BR vector plus deterministic one-branch-at-a-time variants."""

    primary = {
        unit: evaluation.branches[unit][0] for unit in config.investor_ids
    }
    variants = [primary]
    for unit in config.investor_ids:
        for alternative in evaluation.branches[unit][1:]:
            candidate = dict(primary)
            candidate[unit] = alternative
            variants.append(candidate)
    return variants


def _simultaneous_trials(
    data: MarketData,
    config: game.GameConfig,
    search: SearchConfig,
    evaluation: ProfileEvaluation,
) -> list[Trial]:
    return _deduplicate_trials(
        data,
        config,
        (
            _interpolated_trial(
                data,
                config,
                evaluation.power,
                evaluation.energy,
                {unit: (branch, alpha) for unit, branch in branches.items()},
                kind="simultaneous",
            )
            for branches in _simultaneous_branch_sets(config, evaluation)
            for alpha in search.alpha_grid
            if alpha > 0.0
        ),
    )


def _joint_trials(
    data: MarketData,
    config: game.GameConfig,
    search: SearchConfig,
    evaluation: ProfileEvaluation,
    first: str,
    second: str,
) -> list[Trial]:
    return _deduplicate_trials(
        data,
        config,
        (
            _interpolated_trial(
                data,
                config,
                evaluation.power,
                evaluation.energy,
                {first: (left, alpha), second: (right, beta)},
                kind="joint",
            )
            for left in evaluation.branches[first]
            for right in evaluation.branches[second]
            for alpha in search.joint_alpha_grid
            for beta in search.joint_alpha_grid
            if alpha > 0.0 and beta > 0.0
        ),
    )


def screen_trial(
    data: MarketData,
    config: game.GameConfig,
    base: ProfileEvaluation,
    trial: Trial,
    candidate_pool: Mapping[str, Iterable[Branch]],
    active_set_tolerance: float,
) -> ScreenResult:
    """Cheap lower-bound estimate using every response accumulated so far."""

    started = time.perf_counter()
    try:
        market = game.clear(data, config, trial.power, trial.energy)
        signature = _market_active_set_signature(market, active_set_tolerance)
        current = {
            investor.investor_id: game.recleared_profit(
                market, data, config, investor, trial.power, trial.energy
            )
            for investor in config.investors
        }
    except Exception as exc:
        return ScreenResult(
            trial, False, math.inf, math.inf, {}, {}, {},
            "", time.perf_counter() - started, str(exc)
        )

    candidate_profit: dict[str, float] = {}
    regret: dict[str, float] = {}
    for investor in config.investors:
        unit = investor.investor_id
        feasible_profits = [current[unit]]  # standing still is always feasible
        for branch in candidate_pool[unit]:
            candidate_power, candidate_energy = game._with_candidate(
                data, trial.power, trial.energy, unit, branch.power, branch.energy
            )
            try:
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
                if math.isfinite(profit):
                    feasible_profits.append(profit)
            except Exception:
                # A base-profile response can violate remaining nodal headroom
                # after another investor moves.  It is then simply not a
                # feasible conditional deviation at this trial profile.
                continue
        candidate_profit[unit] = max(feasible_profits)
        regret[unit] = max(0.0, candidate_profit[unit] - current[unit])
    return ScreenResult(
        trial=trial,
        valid=True,
        maximum_regret=max(regret.values()),
        sum_regret=sum(regret.values()),
        regret=regret,
        current_profit=current,
        candidate_profit=candidate_profit,
        market_signature=signature,
        seconds=time.perf_counter() - started,
    )


def _screen_sort_key(
    result: ScreenResult,
    base: ProfileEvaluation,
    data: MarketData,
    config: game.GameConfig,
) -> tuple[float, float, float, str]:
    return (
        result.sum_regret,
        result.maximum_regret,
        _profile_distance(
            data,
            config,
            result.trial.power,
            result.trial.energy,
            base.power,
            base.energy,
        ),
        result.trial.label,
    )


def _evaluation_sort_key(
    item: tuple[Trial, ProfileEvaluation],
    base: ProfileEvaluation,
    data: MarketData,
    config: game.GameConfig,
) -> tuple[float, float, float, str]:
    trial, evaluation = item
    return (
        evaluation.sum_regret,
        evaluation.maximum_regret,
        _profile_distance(
            data,
            config,
            evaluation.power,
            evaluation.energy,
            base.power,
            base.energy,
        ),
        trial.label,
    )


def _direction_family(trial: Trial) -> tuple[tuple[str, str], ...] | None:
    alphas = {float(item["alpha"]) for item in trial.directions.values()}
    if len(alphas) != 1:
        return None
    return tuple(
        (unit, str(details["branch"]))
        for unit, details in sorted(trial.directions.items())
    )


def _trial_at_alpha(
    data: MarketData,
    config: game.GameConfig,
    base: ProfileEvaluation,
    family: tuple[tuple[str, str], ...],
    alpha: float,
    kind: str,
) -> Trial:
    choices = {}
    for unit, label in family:
        branch = next(item for item in base.branches[unit] if item.label == label)
        choices[unit] = branch, alpha
    return _interpolated_trial(
        data,
        config,
        base.power,
        base.energy,
        choices,
        kind=f"{kind}_breakpoint",
    )


def _bisect_active_set_changes(
    data: MarketData,
    config: game.GameConfig,
    search: SearchConfig,
    base: ProfileEvaluation,
    screened: list[ScreenResult],
    candidate_pool: Mapping[str, Iterable[Branch]],
) -> list[ScreenResult]:
    """Bisect alpha intervals whose market active-set signatures differ."""

    if search.breakpoint_bisection_rounds == 0 or base.market is None:
        return []
    base_signature = _market_active_set_signature(
        base.market, search.active_set_tolerance
    )
    groups: dict[tuple[tuple[str, str], ...], dict[float, ScreenResult]] = {}
    for item in screened:
        family = _direction_family(item.trial)
        if family is not None and item.valid:
            alpha = float(next(iter(item.trial.directions.values()))["alpha"])
            groups.setdefault(family, {})[alpha] = item

    added: list[ScreenResult] = []
    for _ in range(search.breakpoint_bisection_rounds):
        pending: list[tuple[tuple[tuple[str, str], ...], float, str]] = []
        for family, observations in groups.items():
            points = [(0.0, base_signature)] + sorted(
                (alpha, item.market_signature) for alpha, item in observations.items()
            )
            for (left, left_signature), (right, right_signature) in zip(
                points, points[1:]
            ):
                midpoint = round((left + right) / 2.0, 12)
                if (
                    left_signature != right_signature
                    and midpoint not in observations
                    and midpoint > left
                    and midpoint < right
                ):
                    pending.append((family, midpoint, observations[right].trial.kind))
        if not pending:
            break
        for family, alpha, kind in pending:
            trial = _trial_at_alpha(data, config, base, family, alpha, kind)
            item = screen_trial(
                data,
                config,
                base,
                trial,
                candidate_pool,
                search.active_set_tolerance,
            )
            groups[family][alpha] = item
            added.append(item)
    return added


def _trial_record(
    iteration: int,
    result: ScreenResult,
    validation: ProfileEvaluation | None = None,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "kind": result.trial.kind,
        "label": result.trial.label,
        "directions": dict(result.trial.directions),
        "screen_valid": result.valid,
        "screen_maximum_regret_eur_per_day": result.maximum_regret,
        "screen_sum_regret_eur_per_day": result.sum_regret,
        "screen_regret_eur_per_day": dict(result.regret),
        "market_active_set_signature": result.market_signature,
        "screen_seconds": result.seconds,
        "screen_error": result.error,
        "fully_validated": validation is not None,
        "validated_maximum_regret_eur_per_day": (
            validation.maximum_regret if validation is not None else None
        ),
        "validated_sum_regret_eur_per_day": (
            validation.sum_regret if validation is not None else None
        ),
        "validated_regret_eur_per_day": (
            dict(validation.regret) if validation is not None else None
        ),
        "validation_seconds": validation.seconds if validation is not None else None,
        "validation_error": validation.error if validation is not None else "",
        "validated_audit": (
            _serializable_evaluation(validation) if validation is not None else None
        ),
    }


def _try_trials(
    data: MarketData,
    config: game.GameConfig,
    search: SearchConfig,
    base: ProfileEvaluation,
    trials: list[Trial],
    iteration: int,
    trial_records: list[dict[str, object]],
    candidate_pool: dict[str, list[Branch]],
    regret_normalization: Mapping[str, float],
    on_progress: ProgressCallback | None = None,
) -> tuple[Trial, ProfileEvaluation] | None:
    screened = [
        screen_trial(
            data,
            config,
            base,
            trial,
            candidate_pool,
            search.active_set_tolerance,
        )
        for trial in trials
    ]
    screened.extend(
        _bisect_active_set_changes(
            data, config, search, base, screened, candidate_pool
        )
    )
    screened.sort(key=lambda item: _screen_sort_key(item, base, data, config))
    if on_progress is not None:
        on_progress(
            {
                "iteration": iteration,
                "phase": "screen_complete",
                "kind": trials[0].kind if trials else "empty",
                "trials": len(screened),
                "best_screened_sum_regret_eur_per_day": (
                    screened[0].sum_regret if screened else None
                ),
            }
        )
    finalists = [item for item in screened if item.valid][
        : search.full_validation_finalists
    ]
    validations: dict[str, ProfileEvaluation] = {}
    for index, finalist in enumerate(finalists, start=1):
        if on_progress is not None:
            on_progress(
                {
                    "iteration": iteration,
                    "phase": "full_validation_started",
                    "kind": finalist.trial.kind,
                    "finalist": index,
                    "finalists": len(finalists),
                    "label": finalist.trial.label,
                }
            )
        validations[finalist.trial.label] = evaluate_profile(
            data,
            config,
            search,
            finalist.trial.power,
            finalist.trial.energy,
            regret_normalization,
        )
        if on_progress is not None:
            on_progress(
                {
                    "iteration": iteration,
                    "phase": "full_validation_complete",
                    "kind": finalist.trial.kind,
                    "finalist": index,
                    "finalists": len(finalists),
                    "label": finalist.trial.label,
                    "sum_regret_eur_per_day": validations[
                        finalist.trial.label
                    ].sum_regret,
                    "maximum_regret_eur_per_day": validations[
                        finalist.trial.label
                    ].maximum_regret,
                }
            )
    _merge_candidate_pool(candidate_pool, validations.values())
    for result in screened:
        trial_records.append(
            _trial_record(iteration, result, validations.get(result.trial.label))
        )
    candidates = [
        (result.trial, validations[result.trial.label])
        for result in finalists
        if validations[result.trial.label].valid
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: _evaluation_sort_key(item, base, data, config),
    )


def _sufficient_reduction(
    before: ProfileEvaluation,
    after: ProfileEvaluation,
    search: SearchConfig,
) -> bool:
    required = max(
        search.minimum_reduction_eur_per_day,
        search.minimum_relative_reduction * before.sum_regret,
    )
    return after.sum_regret <= before.sum_regret - required


def _evaluation_record(
    iteration: int,
    evaluation: ProfileEvaluation,
    *,
    selected_step: str,
    selection_reason: str,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "maximum_regret_eur_per_day": evaluation.maximum_regret,
        "sum_regret_eur_per_day": evaluation.sum_regret,
        "regret_eur_per_day": dict(evaluation.regret),
        "relative_regret": dict(evaluation.relative_regret),
        "current_profit_eur_per_day": dict(evaluation.current_profit),
        "valid": evaluation.valid,
        "diagnostics_valid": evaluation.diagnostics_valid,
        "selected_step": selected_step,
        "selection_reason": selection_reason,
        "audit_seconds": evaluation.seconds,
        "total_power_mw": sum(evaluation.power.values()),
        "total_energy_mwh": sum(evaluation.energy.values()),
        "error": evaluation.error,
    }


def _diagnose_best_response_cycle(
    data: MarketData,
    config: game.GameConfig,
    search: SearchConfig,
    start: ProfileEvaluation,
    regret_normalization: Mapping[str, float],
    candidate_pool: dict[str, list[Branch]],
) -> list[dict[str, object]]:
    """Follow full best replies briefly and record returns to prior profiles."""

    if search.cycle_diagnostic_steps == 0 or not start.valid:
        return []
    records: list[dict[str, object]] = []
    seen = [(start.power, start.energy)]
    current = start
    for step in range(1, search.cycle_diagnostic_steps + 1):
        unit = min(
            config.investor_ids,
            key=lambda item: (-current.regret[item], item),
        )
        branch = current.branches[unit][0]
        trial = _interpolated_trial(
            data,
            config,
            current.power,
            current.energy,
            {unit: (branch, 1.0)},
            kind="cycle_diagnostic",
        )
        following = evaluate_profile(
            data,
            config,
            search,
            trial.power,
            trial.energy,
            regret_normalization,
        )
        _merge_candidate_pool(candidate_pool, [following])
        distances = [
            _profile_distance(
                data, config, following.power, following.energy, old_power, old_energy
            )
            for old_power, old_energy in seen
        ]
        aggregate_distances = [
            _aggregate_profile_distance(
                data, config, following.power, following.energy, old_power, old_energy
            )
            for old_power, old_energy in seen
        ]
        return_index = (
            min(range(len(distances)), key=distances.__getitem__) if distances else None
        )
        aggregate_return_index = (
            min(range(len(aggregate_distances)), key=aggregate_distances.__getitem__)
            if aggregate_distances
            else None
        )
        returned_full = return_index is not None and (
            distances[return_index] <= search.cycle_capacity_tolerance
        )
        returned_aggregate = aggregate_return_index is not None and (
            aggregate_distances[aggregate_return_index]
            <= search.cycle_capacity_tolerance
        )
        returned = returned_full or returned_aggregate
        records.append(
            {
                "step": step,
                "investor": unit,
                "branch": branch.label,
                "maximum_regret_eur_per_day": following.maximum_regret,
                "sum_regret_eur_per_day": following.sum_regret,
                "regret_eur_per_day": following.regret,
                "distance_to_nearest_prior_profile": (
                    distances[return_index] if return_index is not None else None
                ),
                "aggregate_distance_to_nearest_prior_profile": (
                    aggregate_distances[aggregate_return_index]
                    if aggregate_return_index is not None
                    else None
                ),
                "returns_to_profile_index": return_index if returned_full else None,
                "aggregate_returns_to_profile_index": (
                    aggregate_return_index if returned_aggregate else None
                ),
                "cycle_detected": returned,
                "valid": following.valid,
                "error": following.error,
            }
        )
        if returned or not following.valid:
            break
        seen.append((following.power, following.energy))
        current = following
    return records


def _measure_regret_resolution(
    data: MarketData,
    config: game.GameConfig,
    search: SearchConfig,
    base: ProfileEvaluation,
    regret_normalization: Mapping[str, float],
    candidate_pool: dict[str, list[Branch]],
) -> tuple[list[dict[str, object]], float | None]:
    """Re-audit tiny feasible moves to measure the local numerical/price floor."""

    if search.resolution_probe_mw == 0.0 or not base.valid:
        return [], None
    records: list[dict[str, object]] = []
    changes: list[float] = []
    base_signature = (
        _market_active_set_signature(base.market, search.active_set_tolerance)
        if base.market is not None
        else ""
    )
    for unit in config.investor_ids:
        branch = base.branches[unit][0]
        scale = max(
            [
                abs(branch.power[node] - base.power[unit, node])
                for node in data.nodes
            ]
            + [
                abs(branch.energy[node] - base.energy[unit, node])
                / max(config.initial_ratio_hours, 1.0)
                for node in data.nodes
            ]
        )
        if scale <= 1.0e-12:
            continue
        alpha = min(1.0, search.resolution_probe_mw / scale)
        trial = _interpolated_trial(
            data,
            config,
            base.power,
            base.energy,
            {unit: (branch, alpha)},
            kind="resolution_probe",
        )
        probe = evaluate_profile(
            data,
            config,
            search,
            trial.power,
            trial.energy,
            regret_normalization,
        )
        _merge_candidate_pool(candidate_pool, [probe])
        change = (
            max(abs(probe.regret[item] - base.regret[item]) for item in config.investor_ids)
            if probe.valid
            else math.inf
        )
        if math.isfinite(change):
            changes.append(change)
        signature = (
            _market_active_set_signature(probe.market, search.active_set_tolerance)
            if probe.market is not None
            else ""
        )
        records.append(
            {
                "investor": unit,
                "branch": branch.label,
                "alpha": alpha,
                "maximum_power_move_mw": alpha
                * max(
                    abs(branch.power[node] - base.power[unit, node])
                    for node in data.nodes
                ),
                "base_regret_eur_per_day": base.regret,
                "probe_regret_eur_per_day": probe.regret,
                "maximum_regret_change_eur_per_day": change,
                "active_set_changed": signature != base_signature,
                "valid": probe.valid,
                "error": probe.error,
            }
        )
    return records, max(changes) if changes else None


def run_regret_guided_search(
    data: MarketData,
    config: game.GameConfig,
    search: SearchConfig,
    *,
    initial: game.GameState | None = None,
    on_iteration: IterationCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> SearchResult:
    """Run regret-guided diagonalisation from a frozen initial profile."""

    game.validate(data, config)
    search.validate()
    started = time.perf_counter()
    state = initial if initial is not None else game.initial_state(data, config)
    evaluation = evaluate_profile(data, config, search, state.power, state.energy)
    regret_normalization = {
        unit: max(
            abs(evaluation.current_profit.get(unit, 0.0)),
            search.relative_regret_floor_eur_per_day,
        )
        for unit in config.investor_ids
    }
    if evaluation.valid:
        evaluation.relative_regret = {
            unit: evaluation.regret[unit] / regret_normalization[unit]
            for unit in config.investor_ids
        }
    candidate_pool: dict[str, list[Branch]] = {
        unit: [] for unit in config.investor_ids
    }
    _merge_candidate_pool(candidate_pool, [evaluation])
    result = SearchResult(
        power=dict(state.power),
        energy=dict(state.energy),
        evaluations=[evaluation],
        iteration_records=[
            _evaluation_record(
                0,
                evaluation,
                selected_step="initial_profile",
                selection_reason="declared starting profile",
            )
        ],
        trial_records=[],
        confirmation_evaluations=[],
        cycle_records=[],
        resolution_records=[],
        regret_normalization_eur_per_day=regret_normalization,
        empirical_regret_resolution_eur_per_day=None,
        converged=False,
        stop_reason="",
        outer_iterations=0,
        restarts_used=0,
        seconds=0.0,
    )
    visited_profiles = {_profile_key(data, config, evaluation.power, evaluation.energy)}
    if on_iteration is not None:
        on_iteration(result)

    for iteration in range(1, search.maximum_outer_iterations + 1):
        if not evaluation.valid:
            result.stop_reason = f"invalid full audit: {evaluation.error or 'best response failed'}"
            break

        if evaluation.maximum_regret <= search.epsilon_eur_per_day:
            if not evaluation.certifiable:
                result.stop_reason = (
                    "regret target reached but solver/reclear diagnostics failed"
                )
                break
            confirmations = [
                evaluate_profile(
                    data,
                    config,
                    search,
                    evaluation.power,
                    evaluation.energy,
                    regret_normalization,
                )
                for _ in range(search.confirmation_audits)
            ]
            _merge_candidate_pool(candidate_pool, confirmations)
            result.confirmation_evaluations = confirmations
            if all(
                audit.certifiable
                and audit.maximum_regret <= search.epsilon_eur_per_day
                for audit in confirmations
            ):
                result.converged = True
                result.stop_reason = (
                    f"maximum regret <= {search.epsilon_eur_per_day:g} EUR/day in "
                    f"{search.confirmation_audits} fresh deterministic audits"
                )
                break
            # Continue from the most conservative fresh audit if confirmation
            # uncovers a different local branch.
            evaluation = max(confirmations, key=lambda item: item.maximum_regret)
            result.evaluations[-1] = evaluation

        ranked = sorted(
            config.investor_ids,
            key=lambda unit: (-evaluation.relative_regret[unit], unit),
        )
        active = ranked[0]
        simultaneous = _try_trials(
            data,
            config,
            search,
            evaluation,
            _simultaneous_trials(data, config, search, evaluation),
            iteration,
            result.trial_records,
            candidate_pool,
            regret_normalization,
            on_progress,
        )
        selected = (
            simultaneous
            if simultaneous is not None
            and _sufficient_reduction(evaluation, simultaneous[1], search)
            else None
        )
        reason = "simultaneous best-response line search on sum of regrets"
        attempted = [simultaneous] if simultaneous is not None else []

        if selected is None and search.fallback_mode == "all":
            line = _try_trials(
                data,
                config,
                search,
                evaluation,
            _line_trials(data, config, search, evaluation, active),
            iteration,
            result.trial_records,
                candidate_pool,
                regret_normalization,
                on_progress,
            )
            if line is not None and _sufficient_reduction(evaluation, line[1], search):
                selected = line
                reason = (
                    "fallback: largest stable-normalized-regret one-direction line search"
                )
            if line is not None:
                attempted.append(line)

        if (
            selected is None
            and search.fallback_mode == "all"
            and len(ranked) >= 2
        ):
            joint = _try_trials(
                data,
                config,
                search,
                evaluation,
                _joint_trials(data, config, search, evaluation, ranked[0], ranked[1]),
                iteration,
                result.trial_records,
                candidate_pool,
                regret_normalization,
                on_progress,
            )
            if joint is not None and _sufficient_reduction(evaluation, joint[1], search):
                selected = joint
                reason = "two-largest-regret joint fallback"
            if joint is not None:
                attempted.append(joint)

        if selected is None:
            restart_candidates = [
                item
                for item in attempted
                if _profile_key(data, config, item[1].power, item[1].energy)
                not in visited_profiles
            ]
            if result.restarts_used < search.maximum_restarts and restart_candidates:
                selected = min(
                    restart_candidates,
                    key=lambda item: _evaluation_sort_key(item, evaluation, data, config),
                )
                result.restarts_used += 1
                reason = (
                    "deterministic restart from the best fully validated rejected trial"
                )
            else:
                if search.fallback_mode == "none":
                    result.stop_reason = (
                        "simultaneous-only diagnostic: no fully validated finalist "
                        "produced sufficient sum-regret reduction"
                    )
                else:
                    result.stop_reason = (
                        "stagnation: no fully validated simultaneous, one-investor, "
                        "or two-investor trial produced sufficient sum-regret "
                        "reduction, and the deterministic restart budget is exhausted"
                    )
                break

        trial, evaluation = selected
        result.power = dict(evaluation.power)
        result.energy = dict(evaluation.energy)
        visited_profiles.add(_profile_key(data, config, evaluation.power, evaluation.energy))
        result.evaluations.append(evaluation)
        result.outer_iterations = iteration
        result.iteration_records.append(
            _evaluation_record(
                iteration,
                evaluation,
                selected_step=trial.label,
                selection_reason=reason,
            )
        )
        result.seconds = time.perf_counter() - started
        if on_iteration is not None:
            on_iteration(result)
    else:
        result.stop_reason = "maximum outer iterations reached"

    final = result.final_evaluation
    result.resolution_records, result.empirical_regret_resolution_eur_per_day = (
        _measure_regret_resolution(
            data,
            config,
            search,
            final,
            regret_normalization,
            candidate_pool,
        )
    )
    if not result.converged:
        result.cycle_records = _diagnose_best_response_cycle(
            data,
            config,
            search,
            final,
            regret_normalization,
            candidate_pool,
        )
        if any(record["cycle_detected"] for record in result.cycle_records):
            result.stop_reason += "; best-response cycle detected"
    result.seconds = time.perf_counter() - started
    return result


def _parse_grid(step: float) -> tuple[float, ...]:
    if not math.isfinite(step) or step <= 0.0 or step > 1.0:
        raise argparse.ArgumentTypeError("grid step must lie in (0, 1]")
    count = int(math.floor(1.0 / step + 1.0e-12))
    values = [round(index * step, 12) for index in range(count + 1)]
    if not math.isclose(values[-1], 1.0):
        values.append(1.0)
    else:
        values[-1] = 1.0
    return tuple(values)


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
    parser.add_argument("--market", choices=("energy-only", "afrr"), default="energy-only")
    parser.add_argument("--afrr-demand-up-mw", type=float, default=None)
    parser.add_argument("--afrr-demand-down-mw", type=float, default=None)
    parser.add_argument(
        "--formulation", choices=("relaxed-kkt", "strong-duality"), default="relaxed-kkt"
    )
    parser.add_argument("--complementarity-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--quadratic-cost-power-eur-per-mw2", type=float, default=0.0)
    parser.add_argument("--quadratic-cost-energy-eur-per-mwh2", type=float, default=0.0)
    parser.add_argument("--quadratic-cost-power-per-node-eur-per-mw2", type=float, default=0.0)
    parser.add_argument("--quadratic-cost-energy-per-node-eur-per-mwh2", type=float, default=0.0)
    parser.add_argument("--node-limit-mw", type=float, default=None)
    parser.add_argument("--initial-power-mw", type=float, default=5.0)
    parser.add_argument("--initial-ratio-hours", type=float, default=3.0)
    parser.add_argument("--initial-capacities", type=Path, default=None)

    parser.add_argument("--epsilon-eur-per-day", type=float, default=20.0)
    parser.add_argument("--branch-profit-tolerance-eur-per-day", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--joint-alpha-step", type=float, default=0.25)
    parser.add_argument("--full-validation-finalists", type=int, default=3)
    parser.add_argument("--maximum-branches-per-investor", type=int, default=3)
    parser.add_argument("--max-outer-iterations", type=int, default=50)
    parser.add_argument("--maximum-restarts", type=int, default=1)
    parser.add_argument(
        "--fallback-mode",
        choices=("all", "none"),
        default="all",
        help="Use all fallbacks, or isolate the simultaneous line-search diagnostic.",
    )
    parser.add_argument("--minimum-reduction-eur-per-day", type=float, default=1.0)
    parser.add_argument("--minimum-relative-reduction", type=float, default=1.0e-4)
    parser.add_argument("--confirmation-audits", type=int, default=2)
    parser.add_argument("--breakpoint-bisection-rounds", type=int, default=4)
    parser.add_argument("--active-set-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--cycle-diagnostic-steps", type=int, default=6)
    parser.add_argument("--cycle-capacity-tolerance", type=float, default=0.1)
    parser.add_argument("--resolution-probe-mw", type=float, default=1.0e-3)

    parser.add_argument("--refinement-starts", type=int, default=3)
    parser.add_argument("--audit-reclear-gap-tolerance-eur-per-day", type=float, default=2.0)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--ipopt-linear-solver", default="ma57")
    parser.add_argument("--max-solver-iterations", type=int, default=3_000)
    parser.add_argument("--max-solve-seconds", type=float, default=600.0)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def _build_configs(args: argparse.Namespace, data: MarketData):
    cost_values = (
        args.quadratic_cost_power_eur_per_mw2,
        args.quadratic_cost_energy_eur_per_mwh2,
        args.quadratic_cost_power_per_node_eur_per_mw2,
        args.quadratic_cost_energy_per_node_eur_per_mwh2,
    )
    if any(value < 0.0 for value in cost_values):
        raise ValueError("Quadratic investment-cost coefficients cannot be negative.")
    if args.node_limit_mw is not None and not args.node_limit_mw > 0.0:
        raise ValueError("The node limit must be positive.")
    config = game.GameConfig(
        investors=three_investors(
            data,
            quadratic_cost_power_eur_per_mw2=args.quadratic_cost_power_eur_per_mw2,
            quadratic_cost_energy_eur_per_mwh2=args.quadratic_cost_energy_eur_per_mwh2,
            quadratic_cost_power_per_node_eur_per_mw2=(
                args.quadratic_cost_power_per_node_eur_per_mw2
            ),
            quadratic_cost_energy_per_node_eur_per_mwh2=(
                args.quadratic_cost_energy_per_node_eur_per_mwh2
            ),
        ),
        lower_level=args.formulation,
        market_design=args.market,
        complementarity_epsilon=args.complementarity_epsilon,
        refinement_starts=args.refinement_starts,
        proximal_penalty=0.0,
        uniform_node_connection_limit_mw=args.node_limit_mw,
        symmetric_initialization=True,
        initial_power_mw=args.initial_power_mw,
        initial_ratio_hours=args.initial_ratio_hours,
        audit_profit_tolerance_eur_per_day=args.epsilon_eur_per_day,
        audit_reclear_gap_tolerance_eur_per_day=(
            args.audit_reclear_gap_tolerance_eur_per_day
        ),
        parallel_workers=args.parallel_workers,
        solver=SolverSettings(
            linear_solver=args.ipopt_linear_solver,
            max_iterations=args.max_solver_iterations,
            max_seconds=args.max_solve_seconds,
            tolerance=args.solver_tolerance,
            tee=args.tee,
        ),
    )
    search = SearchConfig(
        epsilon_eur_per_day=args.epsilon_eur_per_day,
        branch_profit_tolerance_eur_per_day=(
            args.branch_profit_tolerance_eur_per_day
        ),
        alpha_grid=_parse_grid(args.alpha_step),
        joint_alpha_grid=_parse_grid(args.joint_alpha_step),
        full_validation_finalists=args.full_validation_finalists,
        maximum_branches_per_investor=args.maximum_branches_per_investor,
        maximum_outer_iterations=args.max_outer_iterations,
        maximum_restarts=args.maximum_restarts,
        fallback_mode=args.fallback_mode,
        minimum_reduction_eur_per_day=args.minimum_reduction_eur_per_day,
        minimum_relative_reduction=args.minimum_relative_reduction,
        confirmation_audits=args.confirmation_audits,
        breakpoint_bisection_rounds=args.breakpoint_bisection_rounds,
        active_set_tolerance=args.active_set_tolerance,
        cycle_diagnostic_steps=args.cycle_diagnostic_steps,
        cycle_capacity_tolerance=args.cycle_capacity_tolerance,
        resolution_probe_mw=args.resolution_probe_mw,
    )
    return config, search


def _serializable_evaluation(evaluation: ProfileEvaluation) -> dict[str, object]:
    return {
        "maximum_regret_eur_per_day": evaluation.maximum_regret,
        "sum_regret_eur_per_day": evaluation.sum_regret,
        "current_profit_eur_per_day": evaluation.current_profit,
        "regret_eur_per_day": evaluation.regret,
        "relative_regret": evaluation.relative_regret,
        "valid": evaluation.valid,
        "diagnostics_valid": evaluation.diagnostics_valid,
        "seconds": evaluation.seconds,
        "error": evaluation.error,
        "branches": {
            unit: [asdict(branch) for branch in candidates]
            for unit, candidates in evaluation.branches.items()
        },
        "all_candidate_count": {
            unit: len(candidates)
            for unit, candidates in evaluation.all_candidates.items()
        },
        "best_response_starts": {
            unit: list(response.start_records)
            for unit, response in evaluation.responses.items()
        },
    }


def _flat_iteration_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if not isinstance(value, dict)}
        for group in ("regret_eur_per_day", "relative_regret", "current_profit_eur_per_day"):
            for unit, value in record[group].items():
                row[f"{group}_{unit}"] = value
        rows.append(row)
    return rows


def _solver_runtime(config: game.GameConfig) -> dict[str, str | None]:
    executable = ipopt_path(config.solver)
    version = None
    if executable is not None:
        try:
            completed = subprocess.run(
                [str(executable), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = (completed.stdout or completed.stderr).strip() or None
        except (OSError, subprocess.SubprocessError):
            version = None
    return {
        "name": "ipopt",
        "executable": str(executable) if executable is not None else None,
        "version": version,
    }


def _write_outputs(
    out: Path,
    data: MarketData,
    data_path: Path,
    config: game.GameConfig,
    search: SearchConfig,
    result: SearchResult,
    warm_start: dict[str, str] | None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    state = game.GameState(power=result.power, energy=result.energy)
    state.sweep = result.outer_iterations
    state.converged = result.converged
    state.stop_reason = result.stop_reason
    run_payload = reporting.run_config(
        config, data_path, hashlib.sha256(data_path.read_bytes()).hexdigest()
    )
    run_payload.update(
        algorithm="regret-guided-branch-aware-diagonalisation",
        search=asdict(search),
        warm_start=warm_start,
        deterministic_random_seed=None,
        python_version=platform.python_version(),
        pyomo_version=pyomo.__version__,
        solver_runtime=_solver_runtime(config),
        code_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(game.__file__),
                Path(__file__).with_name("iso_market.py"),
                Path(__file__).with_name("mpec.py"),
            )
        },
    )
    reporting.write_json(out / "run_config.json", run_payload)
    reporting.write_json(
        out / "checkpoint.json",
        {
            **reporting.checkpoint(state, config),
            "algorithm": "regret-guided-branch-aware-diagonalisation",
            "maximum_regret_eur_per_day": result.final_evaluation.maximum_regret,
            "stop_reason": result.stop_reason,
        },
    )
    reporting.write_rows(
        out / "final_capacities.csv",
        reporting.capacity_rows(data, config.investors, result.power, result.energy),
    )
    reporting.write_rows(out / "history.csv", _flat_iteration_rows(result.iteration_records))
    reporting.write_json(out / "trial_history.json", result.trial_records)
    reporting.write_json(out / "cycle_diagnostic.json", result.cycle_records)
    reporting.write_json(out / "regret_resolution.json", result.resolution_records)
    for index, evaluation in enumerate(result.evaluations):
        reporting.write_json(
            out / f"audit_iteration_{index:03d}.json",
            _serializable_evaluation(evaluation),
        )
    reporting.write_json(
        out / "confirmation_audits.json",
        [_serializable_evaluation(item) for item in result.confirmation_evaluations],
    )
    final = result.final_evaluation
    reporting.write_rows(
        out / "final_regrets.csv",
        [
            {
                "investor": unit,
                "current_profit_eur_per_day": final.current_profit.get(unit),
                "regret_eur_per_day": final.regret.get(unit),
                "relative_regret": final.relative_regret.get(unit),
                "retained_branches": len(final.branches.get(unit, ())),
                "best_response_optimal": (
                    final.responses[unit].outcome.optimal if unit in final.responses else False
                ),
            }
            for unit in config.investor_ids
        ],
    )
    if final.market is not None:
        reporting.write_rows(out / "final_market.csv", reporting.market_rows(final.market, data))
        reporting.write_rows(
            out / "profit_decomposition.csv",
            [
                asdict(settle(final.market, data, investor, result.power, result.energy))
                for investor in config.investors
            ],
        )
    maximum = final.maximum_regret
    reporting.write_json(
        out / "summary.json",
        {
            "algorithm": "regret-guided-branch-aware-diagonalisation",
            "converged": result.converged,
            "numerically_validated_epsilon_equilibrium": result.converged,
            "global_best_response_certified": False,
            "epsilon_eur_per_day": search.epsilon_eur_per_day,
            "maximum_regret_eur_per_day": maximum,
            "sum_regret_eur_per_day": final.sum_regret,
            "regret_eur_per_day": final.regret,
            "relative_regret": final.relative_regret,
            "regret_normalization_eur_per_day": (
                result.regret_normalization_eur_per_day
            ),
            "primary_merit": "nikaido_isoda_sum_of_exact_recleared_regrets",
            "primary_merit_eur_per_day": final.sum_regret,
            "threshold_sensitivity": {
                str(value): math.isfinite(maximum) and maximum <= value
                for value in (10, 20, 50, 100)
            },
            "diagnostics_valid": final.diagnostics_valid,
            "confirmation_audits": len(result.confirmation_evaluations),
            "empirical_regret_resolution_eur_per_day": (
                result.empirical_regret_resolution_eur_per_day
            ),
            "epsilon_below_empirical_resolution": (
                result.empirical_regret_resolution_eur_per_day is not None
                and search.epsilon_eur_per_day
                < result.empirical_regret_resolution_eur_per_day
            ),
            "cycle_detected": any(
                row["cycle_detected"] for row in result.cycle_records
            ),
            "outer_iterations": result.outer_iterations,
            "restarts_used": result.restarts_used,
            "stop_reason": result.stop_reason,
            "runtime_seconds": result.seconds,
            "full_multistart_audits": (
                len(result.evaluations) + len(result.confirmation_evaluations)
                + sum(bool(row["fully_validated"]) for row in result.trial_records)
            ),
            "screened_trial_profiles": len(result.trial_records),
        },
    )


def main() -> int:
    args = _parse_args()
    data_path = args.data.resolve()
    data = load_market_data(data_path)
    data = replace(
        data,
        afrr_enabled=args.market == "afrr",
        afrr_demand_up_mw=(
            data.afrr_demand_up_mw
            if args.afrr_demand_up_mw is None
            else args.afrr_demand_up_mw
        ),
        afrr_demand_down_mw=(
            data.afrr_demand_down_mw
            if args.afrr_demand_down_mw is None
            else args.afrr_demand_down_mw
        ),
    )
    validate_afrr(data)
    config, search = _build_configs(args, data)
    capacities = _read_capacities(args.initial_capacities)
    initial = game.initial_state(data, config, capacities)
    out = args.output_dir.resolve()
    warm_start = (
        {
            "file": str(args.initial_capacities.resolve()),
            "sha256": hashlib.sha256(args.initial_capacities.read_bytes()).hexdigest(),
        }
        if args.initial_capacities is not None
        else None
    )

    def checkpoint(partial: SearchResult) -> None:
        latest = partial.final_evaluation
        print(
            f"iteration={partial.outer_iterations:03d} "
            f"max_regret={latest.maximum_regret:.6g} EUR/day "
            f"sum_regret={latest.sum_regret:.6g} "
            f"valid={latest.valid} diagnostics={latest.diagnostics_valid}",
            flush=True,
        )
        _write_outputs(out, data, data_path, config, search, partial, warm_start)

    def progress(event: Mapping[str, object]) -> None:
        details = " ".join(f"{key}={value}" for key, value in event.items())
        print(details, flush=True)

    print(
        "Regret-guided capacity EPEC: "
        f"market={config.market_design}, investors={len(config.investors)}, "
        f"epsilon={search.epsilon_eur_per_day:g} EUR/day",
        flush=True,
    )
    result = run_regret_guided_search(
        data,
        config,
        search,
        initial=initial,
        on_iteration=checkpoint,
        on_progress=progress,
    )
    _write_outputs(out, data, data_path, config, search, result, warm_start)
    summary = {
        "converged": result.converged,
        "maximum_regret_eur_per_day": result.final_evaluation.maximum_regret,
        "outer_iterations": result.outer_iterations,
        "restarts_used": result.restarts_used,
        "stop_reason": result.stop_reason,
        "output_dir": str(out),
    }
    print(reporting.json_dumps(summary, indent=2), flush=True)
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
