"""Unit tests for the lightweight audited alternating GS driver."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

import run_audited_alternating_gauss_seidel as alternating_gs


class FakeAudit:
    def __init__(self, **overrides: float) -> None:
        self.valid = True
        self.rows = (SimpleNamespace(optimal=True),) * 3
        self.config = SimpleNamespace(
            audit_reclear_gap_tolerance_eur_per_day=2.0,
            solver=SimpleNamespace(tolerance=1.0e-6),
        )
        self.values = {
            "profitable_deviation_eur_per_day": 19.0,
            "embedded_reclear_profit_gap_eur_per_day": 1.5,
            "complementarity_max_violation": 5.0e-7,
            "max_bound_violation": 5.0e-7,
            **overrides,
        }

    def worst(self, field: str) -> float:
        return self.values[field]


def test_rotating_order_moves_each_investor_to_the_front() -> None:
    investors = ("I1", "I2", "I3")

    assert alternating_gs.rotating_order(investors, 1) == ("I1", "I2", "I3")
    assert alternating_gs.rotating_order(investors, 2) == ("I2", "I3", "I1")
    assert alternating_gs.rotating_order(investors, 3) == ("I3", "I1", "I2")
    assert alternating_gs.rotating_order(investors, 4) == ("I1", "I2", "I3")


def test_audit_acceptance_uses_regret_and_numerics_not_capacity_distance() -> None:
    assert alternating_gs.audit_numerically_valid(FakeAudit())
    assert alternating_gs.audit_passes(FakeAudit(), epsilon=20.0)
    assert not alternating_gs.audit_passes(
        FakeAudit(profitable_deviation_eur_per_day=20.01), epsilon=20.0
    )
    assert not alternating_gs.audit_passes(
        FakeAudit(embedded_reclear_profit_gap_eur_per_day=2.01), epsilon=20.0
    )
    assert not alternating_gs.audit_passes(
        FakeAudit(profitable_deviation_eur_per_day=math.nan), epsilon=20.0
    )


def test_nonoptimal_response_is_not_a_valid_best_profile() -> None:
    audit = FakeAudit()
    audit.rows = (SimpleNamespace(optimal=True), SimpleNamespace(optimal=False))

    assert not alternating_gs.audit_numerically_valid(audit)
    assert not alternating_gs.audit_passes(audit, epsilon=20.0)
