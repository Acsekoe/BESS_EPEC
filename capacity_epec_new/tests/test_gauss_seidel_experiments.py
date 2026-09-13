from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

from capacity_game import (  # noqa: E402
    AuditReport,
    BestResponse,
    GameConfig,
    GameState,
    InvestorAudit,
    gauss_seidel_sweep_order,
    run_gauss_seidel,
)
from investors import InvestorConfig  # noqa: E402
from solvers import SolveOutcome, SolverSettings  # noqa: E402


def config(**overrides) -> GameConfig:
    base = GameConfig(
        investors=tuple(InvestorConfig(unit) for unit in ("I1", "I2", "I3")),
        solver=SolverSettings(tolerance=1e-4),
        complementarity_epsilon=1e-3,
    )
    return replace(base, **overrides)


def audit_row(investor: str, profit: float, deviation: float) -> InvestorAudit:
    return InvestorAudit(
        investor=investor,
        audit_valid=True,
        optimal=True,
        termination="optimal",
        # Deliberately outside the strict equilibrium distance threshold: the
        # epsilon-stationary diagnostic must not silently depend on distance.
        max_power_deviation_mw=100.0,
        max_energy_deviation_mwh=100.0,
        embedded_profit_eur_per_day=profit + deviation,
        recleared_profit_eur_per_day=profit + deviation,
        embedded_reclear_profit_gap_eur_per_day=0.0,
        current_recleared_profit_eur_per_day=profit,
        profitable_deviation_eur_per_day=deviation,
        complementarity_max_product=1e-3,
        complementarity_max_violation=0.0,
        max_bound_violation=0.0,
        absolute_primal_dual_gap_eur_per_day=0.0,
        maximum_artificial_bound_utilization=0.5,
    )


def test_fixed_and_rotating_orders() -> None:
    fixed = config(gauss_seidel_order=("I2", "I3", "I1"))
    assert gauss_seidel_sweep_order(fixed, 1) == ("I2", "I3", "I1")
    assert gauss_seidel_sweep_order(fixed, 9) == ("I2", "I3", "I1")

    rotating = config(gauss_seidel_rotate_first_mover=True)
    assert [gauss_seidel_sweep_order(rotating, sweep) for sweep in range(1, 5)] == [
        ("I1", "I2", "I3"),
        ("I2", "I3", "I1"),
        ("I3", "I1", "I2"),
        ("I1", "I2", "I3"),
    ]


def test_rotating_run_uses_fresh_updates_and_restarts_cycle_at_first_pass(
    monkeypatch,
) -> None:
    cfg = config(
        gauss_seidel_rotate_first_mover=True,
        max_sweeps=4,
        consecutive_sweeps=4,
        damping=1.0,
        convergence_metric="capacity",
        multistart_every_sweeps=0,
    )
    data = SimpleNamespace(
        nodes=("N1",), generators=(), node_connection_limit={"N1": 200.0}
    )
    state = GameState(
        power={(unit, "N1"): 5.0 for unit in cfg.investor_ids},
        energy={(unit, "N1"): 15.0 for unit in cfg.investor_ids},
        # A staged run may already have a Jacobi sweep number.  Rotation still
        # has to start from the configured base order on its first GS pass.
        sweep=1,
    )
    calls: list[tuple[str, float]] = []

    def fake_best_response(data_, config_, investor, power, energy, use_multistart):
        calls.append((investor.investor_id, power["I1", "N1"]))
        return BestResponse(
            investor_id=investor.investor_id,
            outcome=SolveOutcome("optimal", True, True, 0.0),
            power={"N1": power[investor.investor_id, "N1"] + 1.0},
            energy={"N1": energy[investor.investor_id, "N1"] + 2.0},
            embedded_profit_eur_per_day=0.0,
            max_complementarity_product=0.0,
            max_complementarity_violation=0.0,
            primal_dual_gap_eur_per_day=0.0,
        )

    monkeypatch.setattr("capacity_game.best_response", fake_best_response)
    result = run_gauss_seidel(data, cfg, initial=state)

    assert [unit for unit, _ in calls] == [
        "I1", "I2", "I3",
        "I2", "I3", "I1",
        "I3", "I1", "I2",
    ]
    # I2 sees I1's just-applied update in the first sequential pass.
    assert calls[1][1] == pytest.approx(6.0)
    assert [row["update_order"] for row in result.history] == [
        "I1->I2->I3",
        "I2->I3->I1",
        "I3->I1->I2",
    ]


def test_relative_candidate_uses_each_investors_own_payoff() -> None:
    report = AuditReport(
        config=config(audit_relative_regret_tolerance=0.01),
        rows=(
            audit_row("I1", profit=100.0, deviation=1.0),
            audit_row("I2", profit=1_000.0, deviation=9.0),
            audit_row("I3", profit=10_000.0, deviation=50.0),
        ),
        market=None,
    )

    assert report.max_relative_regret == pytest.approx(0.01)
    assert report.passes_relative_regret()
    assert not report.passed

    failing = replace(
        report,
        rows=(audit_row("I1", profit=100.0, deviation=1.01), *report.rows[1:]),
    )
    assert not failing.passes_relative_regret()


def test_relative_candidate_requires_valid_numerics() -> None:
    bad = replace(
        audit_row("I1", profit=100.0, deviation=0.0),
        embedded_reclear_profit_gap_eur_per_day=10.01,
    )
    report = AuditReport(config=config(), rows=(bad,), market=None)

    assert not report.numerically_valid
    assert not report.passes_relative_regret()
