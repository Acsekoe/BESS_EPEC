"""Unit tests for the oracle-efficient search without expensive MPEC solves."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

import capacity_game as game
import oracle_efficient_regret_search as oracle
import regret_guided_search as legacy
from investors import three_investors
from market_data import load_market_data


@pytest.fixture(scope="module")
def data():
    return load_market_data()


def _response(data, investor, power, energy, targets):
    unit = investor.investor_id
    candidate_power = {node: power[unit, node] for node in data.nodes}
    candidate_energy = {node: energy[unit, node] for node in data.nodes}
    candidate_power["N1"] = targets[unit]
    candidate_energy["N1"] = 3.0 * targets[unit]
    record = {
        "start": "local",
        "phase": "refine",
        "has_solution": True,
        "local_optimal": True,
        "power": candidate_power,
        "energy": candidate_energy,
        "exact_profit": 100.0,
    }
    return game.BestResponse(
        investor_id=unit,
        outcome=game.SolveOutcome("optimal", True, True, 0.0),
        power=candidate_power,
        energy=candidate_energy,
        embedded_profit_eur_per_day=100.0,
        max_complementarity_product=0.0,
        max_complementarity_violation=0.0,
        primal_dual_gap_eur_per_day=0.0,
        recleared_profit_eur_per_day=100.0,
        start_label="local",
        start_records=(record,),
        max_bound_violation=0.0,
    )


def test_memory_augmented_jacobi_reaches_toy_target(data, monkeypatch):
    investors = three_investors(data)
    config = game.GameConfig(investors=investors, parallel_workers=1)
    targets = {"I1": 6.0, "I2": 5.0, "I3": 5.0}

    monkeypatch.setattr(
        game,
        "clear",
        lambda _data, _config, power, energy: SimpleNamespace(
            power=dict(power), energy=dict(energy)
        ),
    )

    def fake_profit(_market, _data, _config, investor, power, _energy):
        error = power[investor.investor_id, "N1"] - targets[investor.investor_id]
        return 100.0 - 10.0 * error * error

    def fake_solve_all(_data, _config, power, energy, use_multistart):
        assert not use_multistart
        return {
            investor.investor_id: _response(
                data, investor, power, energy, targets
            )
            for investor in investors
        }

    def fake_audit(_data, _config, _search, power, energy, ledger):
        responses = fake_solve_all(data, config, power, energy, False)
        return legacy.ProfileEvaluation(
            power=dict(power),
            energy=dict(energy),
            current_profit={unit: 100.0 for unit in targets},
            regret={unit: 0.0 for unit in targets},
            relative_regret={unit: 0.0 for unit in targets},
            branches={unit: () for unit in targets},
            all_candidates={unit: () for unit in targets},
            responses=responses,
            valid=True,
            diagnostics_valid=True,
            market=object(),
            seconds=0.0,
        )

    monkeypatch.setattr(game, "recleared_profit", fake_profit)
    monkeypatch.setattr(game, "solve_all", fake_solve_all)
    monkeypatch.setattr(oracle, "_full_audit", fake_audit)
    search = oracle.OracleSearchConfig(
        epsilon_eur_per_day=0.01,
        alpha_grid=(0.0, 0.5, 1.0),
        joint_alpha_grid=(0.0, 0.5, 1.0),
        maximum_iterations=2,
        archive_limit_per_investor=2,
        screen_branches_per_investor=1,
        archive_validation_finalists=1,
        targeted_multistart_interval=0,
        full_audit_interval=0,
        stagnation_iterations_before_targeted=0,
        minimum_reduction_eur_per_day=0.01,
        final_confirmation_audits=1,
    )
    result = oracle.run_oracle_efficient_search(data, config, search)

    assert result.converged
    assert result.power["I1", "N1"] == pytest.approx(6.0)
    assert result.final_restricted_evaluation.maximum_regret == pytest.approx(0.0)
    assert result.history[0]["selection_reason"].startswith("simultaneous")
    assert result.ledger.local_response_batches == 2


def test_response_archive_is_bounded_and_deduplicated(data):
    investor = three_investors(data)[0]
    config = game.GameConfig(investors=(investor,), parallel_workers=1)
    state = game.initial_state(data, config)
    archive = {investor.investor_id: []}
    for target in (6.0, 7.0, 7.0, 8.0):
        response = _response(
            data,
            investor,
            state.power,
            state.energy,
            {investor.investor_id: target},
        )
        oracle.merge_response_archive(data, archive, {investor.investor_id: response}, 2)

    assert len(archive[investor.investor_id]) == 2
    assert [branch.power["N1"] for branch in archive[investor.investor_id]] == [8.0, 7.0]


def test_zero_iterations_is_valid_for_audit_only_mode():
    oracle.OracleSearchConfig(maximum_iterations=0).validate()


def test_restricted_regret_is_computed_by_reclearing_archived_deviation(
    data, monkeypatch
):
    investor = three_investors(data)[0]
    config = game.GameConfig(investors=(investor,), parallel_workers=1)
    state = game.initial_state(data, config)
    response = _response(
        data,
        investor,
        state.power,
        state.energy,
        {investor.investor_id: 6.0},
    )
    archive = {investor.investor_id: list(legacy._all_response_branches(data, response))}
    monkeypatch.setattr(game, "clear", lambda *args: object())
    monkeypatch.setattr(
        game,
        "recleared_profit",
        lambda _m, _d, _c, _i, power, _energy: power[investor.investor_id, "N1"],
    )
    ledger = oracle.CostLedger()
    evaluation = oracle.evaluate_archive(
        data, config, state.power, state.energy, archive, ledger
    )

    assert evaluation.maximum_regret == pytest.approx(1.0)
    assert ledger.explicit_archive_market_clears == 2
