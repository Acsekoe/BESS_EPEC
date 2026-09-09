"""Regression tests for the parts of the capacity game that can fail quietly.

These target economically material failures: an audit that passes on missing
numbers, a search that silently skips part of the capacity space, prices with
the wrong sign, or a candidate deviation lost because a local solve crashed.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import pyomo.environ as pyo

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

import capacity_game as game
import mpec
import solvers
from investors import daily_investment_cost, three_investors
from iso_market import lmp as settle_lmp, settle
from market_data import load_market_data


@pytest.fixture(scope="module")
def data():
    return load_market_data()


def one_investor_config(data, **overrides) -> game.GameConfig:
    return game.GameConfig(
        investors=(three_investors(data)[0],), parallel_workers=1, **overrides
    )


def test_gauss_seidel_later_investor_sees_fresh_update(data, monkeypatch):
    """Sequential responses must observe earlier updates in the same sweep."""

    investors = three_investors(data)[:2]
    config = game.GameConfig(
        investors=investors,
        max_sweeps=1,
        damping=1.0,
        multistart_every_sweeps=0,
        parallel_workers=1,
    )
    state = game.initial_state(data, config)
    observed = {}

    def fake_best_response(data, config, investor, power, energy, use_multistart=False):
        unit = investor.investor_id
        proposed_power = {node: power[unit, node] for node in data.nodes}
        proposed_energy = {node: energy[unit, node] for node in data.nodes}
        if unit == "I1":
            proposed_power["N1"] += 1.0
            proposed_energy["N1"] += 3.0
        else:
            observed["i1_n1"] = power["I1", "N1"]
        return game.BestResponse(
            investor_id=unit,
            outcome=game.SolveOutcome("optimal", True, True, 0.0),
            power=proposed_power,
            energy=proposed_energy,
            embedded_profit_eur_per_day=0.0,
            max_complementarity_product=0.0,
            max_complementarity_violation=0.0,
            primal_dual_gap_eur_per_day=0.0,
            recleared_profit_eur_per_day=0.0,
        )

    monkeypatch.setattr(game, "best_response", fake_best_response)
    result = game.run_gauss_seidel(data, config, initial=state)

    assert observed["i1_n1"] == pytest.approx(6.0)
    assert result.power["I1", "N1"] == pytest.approx(6.0)


# ---------------------------------------------------------------- population


def test_maintained_population_is_merchant_wind_and_solar(data):
    investors = three_investors(data)
    assert [investor.investor_id for investor in investors] == ["I1", "I2", "I3"]
    assert all(investor.wacc == pytest.approx(0.08) for investor in investors)

    wind = {generator for generator in data.generators if "Wind" in generator}
    solar = {generator for generator in data.generators if "PV" in generator}
    assert dict(investors[0].owned_generation_shares) == {}
    assert dict(investors[1].owned_generation_shares) == {
        generator: 1.0 for generator in wind
    }
    assert dict(investors[2].owned_generation_shares) == {
        generator: 1.0 for generator in solar
    }


def test_quadratic_investment_cost_is_portfolio_wide(data):
    investor = three_investors(
        data,
        quadratic_cost_power_eur_per_mw2=20_000.0,
        quadratic_cost_energy_eur_per_mwh2=5_000.0,
    )[0]
    concentrated = daily_investment_cost(investor, [10.0, 0.0], [30.0, 0.0])
    distributed = daily_investment_cost(investor, [5.0, 5.0], [15.0, 15.0])
    assert concentrated == pytest.approx(distributed)


# --------------------------------------------------------------------- search


def test_multistart_probes_every_node(data):
    config = game.GameConfig(investors=three_investors(data))
    state = game.initial_state(data, config)
    labels = {
        label
        for label, _, _ in game.capacity_starts(
            data, config, config.investors[0], state.power, state.energy
        )
    }
    for node in data.nodes:
        assert f"replace_{node}_40mw_4h" in labels
        assert f"relocate_{node}" in labels
    assert {"incumbent", "zero", "scaled_up", "shrunken"} <= labels


def test_feasible_candidate_survives_a_failed_refinement(data, monkeypatch):
    config = one_investor_config(data, proximal_penalty=0.0, refinement_starts=1)
    state = game.initial_state(data, config)
    monkeypatch.setattr(game, "clear", lambda *a: object())
    monkeypatch.setattr(
        game,
        "recleared_profit",
        lambda m, d, c, i, p, e: -((sum(p.values()) - 40.0) ** 2),
    )
    monkeypatch.setattr(
        game,
        "solve_from_start",
        lambda *a: (_ for _ in ()).throw(RuntimeError("deliberate local failure")),
    )
    response = game.best_response(
        data, config, config.investors[0], state.power, state.energy, use_multistart=True
    )
    # Screening found a feasible, profitable capacity even though every local
    # refinement crashed: that candidate must not be thrown away.
    assert response.outcome.has_solution and not response.outcome.optimal
    assert math.isfinite(response.recleared_profit_eur_per_day)
    assert any(
        row["phase"] == "refine" and "deliberate" in row["error"]
        for row in response.start_records
    )
    assert sum(bool(row["selected"]) for row in response.start_records) == 1


def test_numerical_profit_tie_prefers_locally_optimal_response(data, monkeypatch):
    """Micro-EUR re-clear noise must not make a zero response unauditable."""

    config = one_investor_config(data, proximal_penalty=0.0, refinement_starts=1)
    state = game.initial_state(data, config)
    investor = config.investors[0]
    unit = investor.investor_id
    monkeypatch.setattr(game, "clear", lambda *a: object())

    def fake_profit(_market, _data, _config, _investor, power, _energy):
        return 9.99995 if power[unit, "N1"] > 5.0 else 10.0

    def fake_refinement(_data, _config, _investor, _power, _energy, label, p, e):
        refined_power = dict(p)
        refined_energy = dict(e)
        refined_power["N1"] += 1.0e-8
        refined_energy["N1"] += 3.0e-8
        return game.BestResponse(
            investor_id=unit,
            outcome=game.SolveOutcome("optimal", True, True, 0.0),
            power=refined_power,
            energy=refined_energy,
            embedded_profit_eur_per_day=9.99995,
            max_complementarity_product=0.0,
            max_complementarity_violation=0.0,
            primal_dual_gap_eur_per_day=0.0,
            start_label=label,
            max_bound_violation=0.0,
        )

    monkeypatch.setattr(game, "recleared_profit", fake_profit)
    monkeypatch.setattr(game, "solve_from_start", fake_refinement)
    response = game.best_response(
        data, config, investor, state.power, state.energy, use_multistart=False
    )

    assert response.outcome.optimal
    assert response.recleared_profit_eur_per_day == pytest.approx(9.99995)


# ----------------------------------------------------------------- formulation


def test_relaxed_products_are_bounded_above_only(data):
    investor = three_investors(data)[0]
    rivals = {unit: {node: 0.0 for node in data.nodes} for unit in ("I2", "I3")}
    model = mpec.build_capacity_mpec(
        data,
        investor=investor,
        rival_power=rivals,
        rival_energy=rivals,
        lower_level="relaxed-kkt",
        complementarity_epsilon=1.0e-4,
    )
    constraint = next(iter(model.complementarity_generation_lower.values()))
    assert constraint.lower is None
    assert pyo.value(constraint.upper) == 1.0e-4
    assert not model.strong_duality.active


def test_strong_duality_uses_the_exact_equality_and_no_relaxation(data):
    investor = three_investors(data)[0]
    rivals = {unit: {node: 0.0 for node in data.nodes} for unit in ("I2", "I3")}
    model = mpec.build_capacity_mpec(
        data,
        investor=investor,
        rival_power=rivals,
        rival_energy=rivals,
        lower_level="strong-duality",
    )
    assert model.strong_duality.active
    assert not hasattr(model, "complementarity_generation_lower")
    # Both formulations still report the same products, at epsilon zero.
    assert mpec.context(model).complementarity_epsilon == 0.0
    assert len(mpec.context(model).product_components) == 10


def test_shared_nodal_connection_limit_bounds_the_active_investor(data):
    investor = three_investors(data)[0]
    rivals = {
        "I2": {node: 10.0 for node in data.nodes},
        "I3": {node: 12.0 for node in data.nodes},
    }
    model = mpec.build_capacity_mpec(
        data,
        investor=investor,
        rival_power=rivals,
        rival_energy={
            unit: {node: 3.0 * value for node, value in values.items()}
            for unit, values in rivals.items()
        },
        node_connection_limit={node: 40.0 for node in data.nodes},
    )
    for node in data.nodes:
        model.X_power[node].set_value(18.0)
        assert pyo.value(model.node_connection_limit[node].body) == pytest.approx(40.0)
        assert pyo.value(model.node_connection_limit[node].upper) == pytest.approx(40.0)


def test_an_infinite_node_limit_decouples_the_investors(data):
    """An infinite limit must switch the shared cap off everywhere at once.

    Deactivating only the MPEC constraint would leave the exact screen and
    reclear still rejecting every oversubscribed profile, so a best response
    that the MPEC is now free to propose could never be priced.
    """

    investor = three_investors(data)[0]
    rivals = {
        "I2": {node: 10.0 for node in data.nodes},
        "I3": {node: 12.0 for node in data.nodes},
    }
    model = mpec.build_capacity_mpec(
        data,
        investor=investor,
        rival_power=rivals,
        rival_energy={
            unit: {node: 3.0 * value for node, value in values.items()}
            for unit, values in rivals.items()
        },
        node_connection_limit={node: math.inf for node in data.nodes},
    )
    assert len(model.node_connection_limit) == 0

    config = game.GameConfig(
        investors=three_investors(data),
        parallel_workers=1,
        uniform_node_connection_limit_mw=math.inf,
    )
    oversubscribed = {
        (unit, node): 500.0 for unit in config.investor_ids for node in data.nodes
    }
    game._check_profile_node_limits(data, config, oversubscribed)

    capped = replace(config, uniform_node_connection_limit_mw=40.0)
    with pytest.raises(ValueError, match="connection limit exceeded"):
        game._check_profile_node_limits(data, capped, oversubscribed)


# ---------------------------------------------------------------------- market


def test_ipopt_matches_the_expected_primal_and_dual_signs():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.NonNegativeReals)
    model.y = pyo.Var()
    model.balance = pyo.Constraint(expr=model.x + model.y == 3)
    model.cap = pyo.Constraint(expr=model.x <= 1)
    model.objective = pyo.Objective(expr=model.y**2)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solvers.solve_market_qp(model, solvers.SolverSettings())
    assert pyo.value(model.x) == pytest.approx(1.0, abs=1e-7)
    assert model.dual[model.balance] == pytest.approx(4.0, abs=1e-7)
    assert model.dual[model.cap] == pytest.approx(-4.0, abs=1e-7)
    assert solvers.maximum_bound_violation(model) < 1e-7


def test_settlement_decomposition_adds_up(data):
    config = game.GameConfig(investors=three_investors(data), parallel_workers=1)
    state = game.initial_state(data, config)
    market = game.clear(data, config, state.power, state.energy)
    for investor in config.investors:
        row = settle(market, data, investor, state.power, state.energy)
        assert row.storage_operating_surplus == pytest.approx(
            row.storage_settlement - row.degradation
        )
        assert row.profit == pytest.approx(
            row.storage_settlement
            + row.owned_generation_rent
            - row.degradation
            - row.capex
        )
        assert row.capex > 0.0


# ----------------------------------------------------------------------- audit


def test_audit_switches_the_proximal_penalty_off(data, monkeypatch):
    investor = three_investors(data)[0]
    config = one_investor_config(data, proximal_penalty=0.25)
    state = game.GameState(
        power={(investor.investor_id, node): 1.0 for node in data.nodes},
        energy={(investor.investor_id, node): 3.0 for node in data.nodes},
    )
    response = game.BestResponse(
        investor_id=investor.investor_id,
        outcome=solvers.SolveOutcome("optimal", True, True, 0.0),
        power={node: 1.0 for node in data.nodes},
        energy={node: 3.0 for node in data.nodes},
        embedded_profit_eur_per_day=10.0,
        max_complementarity_product=0.0,
        max_complementarity_violation=0.0,
        primal_dual_gap_eur_per_day=0.0,
        max_bound_violation=0.0,
    )
    seen = {}

    def fake_audit_state(_data, audit_config, _state):
        seen["penalty"] = audit_config.proximal_penalty
        return {investor.investor_id: response}

    monkeypatch.setattr(game, "clear", lambda *a: object())
    monkeypatch.setattr(game, "recleared_profit", lambda *a: 10.0)
    monkeypatch.setattr(game, "audit_state", fake_audit_state)

    report = game.audit_equilibrium(data, config, state)
    assert seen["penalty"] == 0.0
    assert report.valid and report.rows[0].profitable_deviation_eur_per_day == 0.0
    assert report.passed


def test_a_failed_common_reclear_can_never_pass(data, monkeypatch):
    config = game.GameConfig(investors=three_investors(data))
    state = game.initial_state(data, config)
    monkeypatch.setattr(
        game, "clear", lambda *a: (_ for _ in ()).throw(RuntimeError("reclear failed"))
    )
    report = game.audit_equilibrium(data, config, state)
    assert report.market is None and len(report.rows) == 3
    assert not report.valid and not report.passed
    assert all(math.isnan(row.profitable_deviation_eur_per_day) for row in report.rows)


def test_a_missing_number_is_not_a_passing_number(data, monkeypatch):
    investor = three_investors(data)[0]
    config = one_investor_config(data)
    state = game.GameState(
        power={(investor.investor_id, node): 1.0 for node in data.nodes},
        energy={(investor.investor_id, node): 3.0 for node in data.nodes},
    )
    response = game.BestResponse(
        investor_id=investor.investor_id,
        outcome=solvers.SolveOutcome("error", False, False, 0.0),
        power={node: 1.0 for node in data.nodes},
        energy={node: 3.0 for node in data.nodes},
        embedded_profit_eur_per_day=math.nan,
        max_complementarity_product=math.nan,
        max_complementarity_violation=math.nan,
        primal_dual_gap_eur_per_day=math.nan,
    )
    monkeypatch.setattr(game, "clear", lambda *a: object())
    monkeypatch.setattr(game, "recleared_profit", lambda *a: 10.0)
    monkeypatch.setattr(game, "audit_state", lambda *a: {investor.investor_id: response})
    report = game.audit_equilibrium(data, config, state)
    assert not report.rows[0].audit_valid
    assert math.isnan(report.rows[0].profitable_deviation_eur_per_day)
    assert report.worst("profitable_deviation_eur_per_day") is None
    assert not report.passed


# ------------------------------------------------------------------ regression


def test_no_access_market_survives_anywhere():
    """The access market and its nodal tariff were removed.

    Both are gone for the same reason: they were symmetric shared costs that
    every investor faced identically, so they could only ever compress the
    capacity split towards equal shares. Nothing may quietly reintroduce
    either one.
    """

    for source in MODEL_DIR.glob("*.py"):
        text = source.read_text(encoding="utf-8").lower()
        assert "access_price" not in text, source.name
        assert "access market" not in text, source.name
        assert "system_capacity_limit" not in text, source.name
        assert "access_cost" not in text, source.name
        assert "access_alpha" not in text, source.name
        assert "access_beta" not in text, source.name


def test_solver_settings_reject_invalid_tolerance():
    with pytest.raises(ValueError):
        replace(solvers.SolverSettings(), tolerance=0.0)


# --------------------------------------------------------------------------
# Convergence metric
#
# The capacity metric asks whether the iterate stopped moving.  In a game whose
# equilibria form a set that question has no good answer: the profile slides
# along the set at essentially constant payoff.  These tests pin the two
# metrics that ask about payoffs instead, and the guard that stops either of
# them from certifying the dead zone between best-response branches.


def _response(unit, data, power, energy, *, exact, optimal=True):
    """A best response that proposes the incumbent and is worth ``exact``."""

    return game.BestResponse(
        investor_id=unit,
        outcome=solvers.SolveOutcome("optimal" if optimal else "maxIterations", True, optimal, 0.0),
        power={node: power[unit, node] for node in data.nodes},
        energy={node: energy[unit, node] for node in data.nodes},
        embedded_profit_eur_per_day=exact,
        max_complementarity_product=0.0,
        max_complementarity_violation=0.0,
        primal_dual_gap_eur_per_day=0.0,
        recleared_profit_eur_per_day=exact,
    )


def _fixed_payoff_game(data, monkeypatch, profits, deviations, **overrides):
    """Two investors whose exact profits and deviation values are dictated."""

    defaults = {
        "max_sweeps": 4,
        "consecutive_sweeps": 1,
        "multistart_every_sweeps": 0,
    }
    config = game.GameConfig(
        investors=three_investors(data)[:2],
        damping=1.0,
        proximal_penalty=0.0,
        parallel_workers=1,
        **{**defaults, **overrides},
    )

    def fake_solve_all(data_, config_, power, energy, use_multistart):
        return {
            unit: _response(unit, data_, power, energy, exact=deviations[unit])
            for unit in config_.investor_ids
        }

    monkeypatch.setattr(game, "solve_all", fake_solve_all)
    monkeypatch.setattr(game, "clear", lambda *a, **k: object())
    monkeypatch.setattr(
        game, "recleared_profit", lambda market, d, c, investor, p, e: profits[investor.investor_id]
    )
    return config


def test_regret_is_measured_the_way_the_audit_measures_it(data, monkeypatch):
    """Per-sweep regret must be the audit's profitable deviation, clamped at zero.

    The point of the metric is that the monitor and the certificate are one
    number, so a convergence plot can be read on the same axis as the audit.
    """

    profits = {"I1": 100.0, "I2": 1_000.0}
    deviations = {"I1": 130.0, "I2": 990.0}
    config = _fixed_payoff_game(data, monkeypatch, profits, deviations)
    state = game.initial_state(data, config)
    responses = game.solve_all(data, config, state.power, state.energy, False)

    gap = game.sweep_regret(data, config, state.power, state.energy, responses)

    assert gap.measured
    assert gap.regret["I1"] == pytest.approx(30.0)
    # Signed per investor, so a best response worse than the incumbent stays
    # visible as the solver report it is.
    assert gap.regret["I2"] == pytest.approx(-10.0)
    # ...but the metric itself is clamped, exactly as audit_equilibrium clamps.
    assert gap.max_regret_eur_per_day == pytest.approx(30.0)
    assert gap.relative_regret == pytest.approx(30.0 / 1_000.0)


def test_a_loss_making_investor_blocks_a_flat_payoff_stop(data, monkeypatch):
    """Flat payoffs are not an equilibrium if someone is losing money.

    This is the dead zone between best-response branches: the damped symmetric
    runs parked there at ~250 MW with I1 at -1819 EUR/day.  Nothing moves, so
    the profit metric is satisfied and every deviation is zero, yet no investor
    would accept the outcome.
    """

    profits = {"I1": -1_819.0, "I2": 70_000.0}
    config = _fixed_payoff_game(
        data, monkeypatch, profits, profits, convergence_metric="profit"
    )
    state = game.run_jacobi(data, config)

    assert not state.converged
    assert state.stop_reason == "maximum sweeps reached"
    assert state.history[-1]["max_incumbent_profit_change_eur_per_day"] == pytest.approx(0.0)
    assert state.history[-1]["individually_rational"] is False

    # The guard is the only thing holding it back; without it the run stops.
    relaxed = game.run_jacobi(
        data, replace(config, require_individual_rationality=False)
    )
    assert relaxed.converged
    assert "profit moved at most" in relaxed.stop_reason


def test_regret_stop_fires_on_the_nash_gap_not_on_capacity(data, monkeypatch):
    """A profile nobody wants to leave converges even though capacity moved."""

    profits = {"I1": 100.0, "I2": 1_000.0}
    config = _fixed_payoff_game(
        data,
        monkeypatch,
        profits,
        profits,
        convergence_metric="regret",
        tolerance_relative_regret=1.0e-3,
        multistart_every_sweeps=1,
        # Capacity would never call this converged.
        tolerance_mw=0.0,
        tolerance_mwh=0.0,
    )
    state = game.run_jacobi(data, config)

    assert state.converged
    assert state.history[-1]["relative_regret"] == pytest.approx(0.0)
    assert "relative regret" in state.stop_reason


def test_an_unsolved_best_response_is_not_zero_regret(data, monkeypatch):
    """A deviation that was never valued must not read as nothing to gain."""

    profits = {"I1": 100.0, "I2": 1_000.0}
    config = _fixed_payoff_game(
        data, monkeypatch, profits, profits,
        convergence_metric="regret", multistart_every_sweeps=1,
    )

    def fake_solve_all(data_, config_, power, energy, use_multistart):
        responses = {
            unit: _response(unit, data_, power, energy, exact=profits[unit])
            for unit in config_.investor_ids
        }
        responses["I1"] = replace(
            responses["I1"], recleared_profit_eur_per_day=math.nan
        )
        return responses

    monkeypatch.setattr(game, "solve_all", fake_solve_all)
    state = game.run_jacobi(data, config)

    assert not state.converged
    assert state.history[-1]["regret_error"]


def test_regret_metric_refuses_a_regularized_best_response(data):
    """A proximal term biases deviations downward; stopping on it certifies it."""

    config = game.GameConfig(
        investors=three_investors(data)[:1],
        convergence_metric="regret",
        proximal_penalty=0.01,
        multistart_every_sweeps=1,
    )
    with pytest.raises(ValueError, match="proximal_penalty=0.0"):
        game.validate(data, config)


def test_gauss_seidel_refuses_a_payoff_metric(data):
    """No single clear values responses that answered different profiles."""

    config = game.GameConfig(
        investors=three_investors(data)[:1],
        convergence_metric="regret",
        proximal_penalty=0.0,
        multistart_every_sweeps=1,
    )
    with pytest.raises(ValueError, match="only defined for run_jacobi"):
        game.run_gauss_seidel(data, config)


def test_only_a_multistart_sweep_can_certify_a_small_regret(data, monkeypatch):
    """A quiet single-start sweep neither stops the run nor resets the count.

    Without multistart a sweep solves one local NLP from the incumbent, so a
    zero deviation only says it did not leave the branch it started on. It is
    not evidence against convergence either, so it holds the counter rather
    than clearing it.
    """

    profits = {"I1": 100.0, "I2": 1_000.0}
    config = _fixed_payoff_game(
        data,
        monkeypatch,
        profits,
        profits,
        convergence_metric="regret",
        multistart_every_sweeps=3,
        consecutive_sweeps=2,
        max_sweeps=6,
    )
    state = game.run_jacobi(data, config)

    # Multistart sweeps are 1 and 3.  Sweep 2 is quiet but cannot count, so the
    # second qualifying sweep — and the stop — is sweep 3, not sweep 2.
    assert state.converged
    assert state.sweep == 3
    counts = [row["stable_sweeps"] for row in state.history]
    assert counts == [1, 1, 2]


def test_regret_without_multistart_monitors_but_never_certifies(data, monkeypatch):
    """Measuring regret and stopping on it are different jobs.

    With no multistart every sweep still records its regret -- that is the
    convergence curve, and it is nearly free -- but no sweep searched widely
    enough to certify a stop, so the run goes the distance and the final audit
    stays the certificate. Refusing this configuration would block the very
    workflow the regret column exists for.
    """

    profits = {"I1": 100.0, "I2": 1_000.0}
    config = _fixed_payoff_game(
        data, monkeypatch, profits, profits,
        convergence_metric="regret", multistart_every_sweeps=0, max_sweeps=3,
    )
    state = game.run_jacobi(data, config)

    assert not state.converged
    assert "none could certify a stop" in state.stop_reason
    # ...but every sweep still measured it.
    assert all(row["relative_regret"] == 0.0 for row in state.history)


# --------------------------------------------------------------------------
# The demand-adjustment penalty rho
#
# rho exists to pin a unique LMP. It is not free: at the ISO's optimum the
# stationarity condition is exactly DemandAdjustment = LMP/rho, so a small rho
# is a large virtual generator at every node -- including the six with no
# demand, where "adjusting demand" is not a thing that can happen.


def test_demand_adjustment_is_price_over_rho_where_it_may_act(data):
    """Where demand exists the term settles at LMP/rho -- it is not a rounding
    term but an inverse elasticity, so halving rho doubles the energy involved.
    """

    config = game.GameConfig(investors=three_investors(data), parallel_workers=1)
    state = game.initial_state(data, config)
    for rho in (25.0, 500.0):
        scaled = replace(data, demand_adjustment_penalty_eur_per_mw2=rho)
        market = game.clear(scaled, config, state.power, state.energy)
        live = [
            (n, t)
            for n in scaled.nodes
            for t in scaled.times
            if scaled.demand_is_adjustable(n, t)
        ]
        assert live, "the input is expected to have some adjustable demand"
        worst = max(
            abs(
                float(pyo.value(market.DemandAdjustment[n, t]))
                - settle_lmp(market, n, t) / rho
            )
            for n, t in live
        )
        assert worst < 1.0e-8, f"rho={rho}: adjustment departs from LMP/rho by {worst}"


def test_no_energy_is_conjured_where_there_is_no_demand(data):
    """Six of nine nodes carry no load, and the term must not act there.

    Left free it is not a demand adjustment at all but a virtual generator: at
    rho = 25 those six nodes were supplying 213.6 MWh/day between them, 1.6% of
    system load, from nothing.
    """

    config = game.GameConfig(investors=three_investors(data), parallel_workers=1)
    state = game.initial_state(data, config)
    market = game.clear(data, config, state.power, state.energy)
    empty = [n for n in data.nodes if all(data.demand_el[n, t] == 0.0 for t in data.times)]
    assert empty, "this input is expected to have nodes with no demand at all"
    conjured = sum(
        abs(float(pyo.value(market.DemandAdjustment[n, t])))
        for n in empty
        for t in data.times
    )
    assert conjured == 0.0


def test_holding_the_adjustment_at_zero_does_not_zero_the_price(data):
    """The LMP at a node with no demand is a price, not a hole.

    Holding the term at zero must not be done by way of its stationarity
    condition: ``rho * 0 == lam`` would pin the LMP there to zero. The price is
    set by net_injection_stationarity against the system price and line duals.
    """

    config = game.GameConfig(investors=three_investors(data), parallel_workers=1)
    state = game.initial_state(data, config)
    market = game.clear(data, config, state.power, state.energy)
    empty = [n for n in data.nodes if all(data.demand_el[n, t] == 0.0 for t in data.times)]
    priced = [settle_lmp(market, n, 20) for n in empty]
    assert all(p > 1.0 for p in priced), f"LMPs collapsed at unloaded nodes: {priced}"


# --------------------------------------------------------------------------
# Generation ownership
#
# Who owns which generator is not bookkeeping. With one investor owning both
# PV nodes, congestion relief at N6 and N8 is a single public good: whoever
# builds the storage, that owner captures the price support, so nothing pins
# which investor builds and the equilibrium split is a continuum. Splitting the
# two PV nodes between two owners makes each a private good.


def test_default_population_is_unchanged(data):
    """Omitting ownership must preserve the maintained population exactly."""

    default = three_investors(data)
    assert [i.investor_id for i in default] == ["I1", "I2", "I3"]
    assert dict(default[0].owned_generation_shares) == {}
    assert set(default[1].owned_generation_shares) == {"RES_Wind_N1"}
    assert set(default[2].owned_generation_shares) == {"RES_PV_N6", "RES_PV_N8"}


def test_ownership_override_gives_each_owner_one_pv_node(data):
    investors = three_investors(
        data,
        ownership={
            "I1": {"RES_Wind_N1": 1.0},
            "I2": {"RES_PV_N6": 1.0},
            "I3": {"RES_PV_N8": 1.0},
        },
    )
    assert dict(investors[0].owned_generation_shares) == {"RES_Wind_N1": 1.0}
    assert dict(investors[1].owned_generation_shares) == {"RES_PV_N6": 1.0}
    assert dict(investors[2].owned_generation_shares) == {"RES_PV_N8": 1.0}


def test_generation_cannot_be_owned_twice(data):
    """Rent is paid once. Two owners at full share would invent it."""

    with pytest.raises(ValueError, match="more than once"):
        three_investors(
            data,
            ownership={"I2": {"RES_PV_N6": 1.0}, "I3": {"RES_PV_N6": 0.5}},
        )
    # Splitting one generator between owners is legitimate.
    split = three_investors(
        data, ownership={"I2": {"RES_PV_N6": 0.4}, "I3": {"RES_PV_N6": 0.6}}
    )
    assert split[1].owned_generation_shares["RES_PV_N6"] == pytest.approx(0.4)


def test_ownership_rejects_an_unknown_generator(data):
    with pytest.raises(ValueError, match="Unknown generators"):
        three_investors(data, ownership={"I2": {"RES_PV_N7": 1.0}})
