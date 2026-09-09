"""Tests for the regret-guided outer driver without expensive MPEC solves."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

import capacity_game as game
import regret_guided_search as regret_search
from investors import three_investors
from market_data import load_market_data


@pytest.fixture(scope="module")
def data():
    return load_market_data()


def _response(data, investor, power, energy, target_power):
    unit = investor.investor_id
    candidate_power = {node: power[unit, node] for node in data.nodes}
    candidate_energy = {node: energy[unit, node] for node in data.nodes}
    candidate_power["N1"] = target_power[unit]
    candidate_energy["N1"] = 3.0 * target_power[unit]
    exact = 100.0
    record = {
        "start": "toy_best",
        "phase": "refine",
        "has_solution": True,
        "local_optimal": True,
        "power": candidate_power,
        "energy": candidate_energy,
        "exact_profit": exact,
    }
    return game.BestResponse(
        investor_id=unit,
        outcome=game.SolveOutcome("optimal", True, True, 0.0),
        power=candidate_power,
        energy=candidate_energy,
        embedded_profit_eur_per_day=exact,
        max_complementarity_product=0.0,
        max_complementarity_violation=0.0,
        primal_dual_gap_eur_per_day=0.0,
        recleared_profit_eur_per_day=exact,
        start_label="toy_best",
        start_records=(record,),
        max_bound_violation=0.0,
    )


def test_simultaneous_line_search_reduces_sum_and_confirms(data, monkeypatch):
    """The default step is the simultaneous BR vector, not a player update."""

    investors = three_investors(data)
    config = game.GameConfig(investors=investors, parallel_workers=1)
    target_power = {"I1": 6.0, "I2": 5.0, "I3": 5.0}

    def fake_clear(_data, _config, power, energy):
        return SimpleNamespace(power=dict(power), energy=dict(energy))

    def fake_profit(market, _data, _config, investor, power, energy):
        error = power[investor.investor_id, "N1"] - target_power[investor.investor_id]
        return 100.0 - 10.0 * error * error

    def fake_solve_all(_data, _config, power, energy, use_multistart):
        assert use_multistart
        return {
            investor.investor_id: _response(
                data, investor, power, energy, target_power
            )
            for investor in investors
        }

    monkeypatch.setattr(game, "clear", fake_clear)
    monkeypatch.setattr(game, "recleared_profit", fake_profit)
    monkeypatch.setattr(game, "solve_all", fake_solve_all)

    search = regret_search.SearchConfig(
        epsilon_eur_per_day=0.01,
        alpha_grid=(0.0, 0.5, 1.0),
        joint_alpha_grid=(0.0, 0.5, 1.0),
        full_validation_finalists=1,
        maximum_branches_per_investor=1,
        maximum_outer_iterations=2,
        minimum_reduction_eur_per_day=0.01,
        confirmation_audits=2,
        breakpoint_bisection_rounds=0,
        cycle_diagnostic_steps=0,
        resolution_probe_mw=0.0,
    )
    result = regret_search.run_regret_guided_search(data, config, search)

    assert result.converged
    assert result.outer_iterations == 1
    assert result.evaluations[0].sum_regret == pytest.approx(10.0)
    assert result.final_evaluation.maximum_regret == pytest.approx(0.0)
    assert len(result.confirmation_evaluations) == 2
    assert result.iteration_records[1]["selection_reason"].startswith("simultaneous")
    assert set(result.iteration_records[1]["selected_step"].split("|")) == {
        "I1:toy_best@1",
        "I2:toy_best@1",
        "I3:toy_best@1",
    }


def test_active_set_change_is_bisected(data, monkeypatch):
    """A market-signature flip triggers sub-grid alpha refinement."""

    investor = three_investors(data)[0]
    config = game.GameConfig(investors=(investor,), parallel_workers=1)
    state = game.initial_state(data, config)
    unit = investor.investor_id
    branch = regret_search.Branch(
        investor_id=unit,
        label="branch",
        power={node: state.power[unit, node] + 1.0 for node in data.nodes},
        energy={node: state.energy[unit, node] + 3.0 for node in data.nodes},
        exact_profit_eur_per_day=1.0,
        local_optimal=True,
        source_phase="test",
    )
    base = regret_search.ProfileEvaluation(
        power=state.power,
        energy=state.energy,
        current_profit={unit: 0.0},
        regret={unit: 1.0},
        relative_regret={unit: 1.0},
        branches={unit: (branch,)},
        all_candidates={unit: (branch,)},
        responses={},
        valid=True,
        diagnostics_valid=True,
        market=object(),
        seconds=0.0,
    )
    search = regret_search.SearchConfig(
        alpha_grid=(0.0, 0.2, 0.3, 1.0),
        joint_alpha_grid=(0.0, 1.0),
        breakpoint_bisection_rounds=5,
        cycle_diagnostic_steps=0,
        resolution_probe_mw=0.0,
    )

    monkeypatch.setattr(
        regret_search, "_market_active_set_signature", lambda market, tolerance: "low"
    )

    def fake_screen(data, config, base, trial, candidate_pool, tolerance):
        alpha = float(next(iter(trial.directions.values()))["alpha"])
        signature = "low" if alpha < 0.23 else "high"
        return regret_search.ScreenResult(
            trial=trial,
            valid=True,
            maximum_regret=1.0,
            sum_regret=1.0,
            regret={unit: 1.0},
            current_profit={unit: 0.0},
            candidate_profit={unit: 1.0},
            market_signature=signature,
            seconds=0.0,
        )

    monkeypatch.setattr(regret_search, "screen_trial", fake_screen)
    initial = [
        fake_screen(data, config, base, trial, {unit: [branch]}, 1.0e-5)
        for trial in regret_search._line_trials(data, config, search, base, unit)
    ]
    added = regret_search._bisect_active_set_changes(
        data, config, search, base, initial, {unit: [branch]}
    )
    alphas = [float(next(iter(item.trial.directions.values()))["alpha"]) for item in added]

    assert alphas
    assert min(abs(alpha - 0.23) for alpha in alphas) < 0.01


def test_candidate_pool_never_discards_distinct_responses(data):
    investor = three_investors(data)[0]
    config = game.GameConfig(investors=(investor,), parallel_workers=1)
    state = game.initial_state(data, config)
    first = regret_search.Branch(
        investor.investor_id,
        "first",
        {node: state.power[investor.investor_id, node] for node in data.nodes},
        {node: state.energy[investor.investor_id, node] for node in data.nodes},
        1.0,
        True,
        "test",
    )
    second = regret_search.Branch(
        investor.investor_id,
        "second",
        {node: state.power[investor.investor_id, node] + 1.0 for node in data.nodes},
        {node: state.energy[investor.investor_id, node] + 3.0 for node in data.nodes},
        2.0,
        True,
        "test",
    )
    empty = dict(
        power=state.power,
        energy=state.energy,
        current_profit={investor.investor_id: 0.0},
        regret={investor.investor_id: 0.0},
        relative_regret={investor.investor_id: 0.0},
        responses={},
        valid=True,
        diagnostics_valid=True,
        market=None,
        seconds=0.0,
    )
    evaluations = [
        regret_search.ProfileEvaluation(
            **empty,
            branches={investor.investor_id: (candidate,)},
            all_candidates={investor.investor_id: (candidate,)},
        )
        for candidate in (first, second, first)
    ]
    pool = {investor.investor_id: []}
    regret_search._merge_candidate_pool(pool, evaluations)

    assert [item.label for item in pool[investor.investor_id]] == ["first", "second"]


def test_sufficient_decrease_uses_sum_not_maximum():
    before = SimpleNamespace(sum_regret=20.0, maximum_regret=10.0)
    # The max becomes worse, but the Nikaido--Isoda sum improves materially.
    after = SimpleNamespace(sum_regret=11.0, maximum_regret=11.0)
    search = regret_search.SearchConfig(
        minimum_reduction_eur_per_day=1.0,
        minimum_relative_reduction=0.0,
    )

    assert regret_search._sufficient_reduction(before, after, search)
