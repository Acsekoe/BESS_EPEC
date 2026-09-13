"""Everything this project needs from a solver.

Two very different solves happen here:

* the ISO's fixed-capacity market, a convex QP that must be solved tightly
  because its duals *are* the LMPs (IPOPT at a tight tolerance); and
* an investor's capacity MPEC, a non-convex NLP for which IPOPT returns a
  local solution that always has to be audited afterwards.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pyomo.environ as pyo




@dataclass(frozen=True)
class SolverSettings:
    """Solver choices shared by the market QP and the capacity MPEC."""

    linear_solver: str = "ma57"
    max_iterations: int = 3_000
    max_seconds: float = 45
    tolerance: float = 1.0e-6
    tee: bool = False
    executable: str | None = None

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("The solver tolerance must be positive.")
        if self.max_iterations <= 0 or self.max_seconds <= 0.0:
            raise ValueError("Solver iteration and time limits must be positive.")


@dataclass(frozen=True)
class SolveOutcome:
    """What one MPEC solve returned, before any economic interpretation."""

    termination: str
    has_solution: bool
    optimal: bool
    seconds: float


def ipopt_path(settings: SolverSettings) -> Path | None:
    """First IPOPT binary that actually exists, or None to let Pyomo look."""

    candidates = [
        settings.executable,
        os.environ.get("IPOPT_EXECUTABLE"),
        shutil.which("ipopt"),
        (
            str(Path(os.environ["LOCALAPPDATA"]) / "idaes" / "bin" / "ipopt.exe")
            if os.environ.get("LOCALAPPDATA")
            else None
        ),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def _ipopt(settings: SolverSettings, options: dict[str, object]):
    kwargs: dict[str, object] = {"solver_io": "nl"}
    executable = ipopt_path(settings)
    if executable is not None:
        kwargs["executable"] = str(executable)
    solver = pyo.SolverFactory("ipopt", **kwargs)
    if not solver.available(exception_flag=False):
        raise RuntimeError("IPOPT is unavailable.")
    solver.options.update(options)
    return solver


def solve_market_qp(model: pyo.ConcreteModel, settings: SolverSettings) -> None:
    """Solve a fixed-capacity market to optimality and import its duals.

    Raises if the QP does not solve: an unsolved market has no LMPs, and a
    silently degraded one would corrupt every profit downstream.
    """

    solver = _ipopt(
        settings,
        {
            "linear_solver": settings.linear_solver,
            "max_iter": settings.max_iterations,
            "max_cpu_time": settings.max_seconds,
            # The duals are prices, so this solve is deliberately tighter than
            # the strategic NLP above it.
            "tol": min(settings.tolerance, 1.0e-11),
            # Reserve-shortage coefficients are 12000 EUR/MW/block. Gradient
            # scaling can otherwise accept economically large unscaled KKT
            # products despite a tiny scaled tolerance. Prices need accuracy
            # in original EUR units for both warm starts and settlement.
            "nlp_scaling_method": "none",
            "acceptable_tol": 1.0e-10,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "warm_start_init_point": "no",
            "print_level": 0,
        },
    )
    result = solver.solve(model, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"Market clearing failed: {result.solver.termination_condition}"
        )


def solve_mpec(model: pyo.ConcreteModel, settings: SolverSettings) -> SolveOutcome:
    """Solve one investor's capacity MPEC; never raises on a bad termination."""

    solver = _ipopt(
        settings,
        {
            "linear_solver": settings.linear_solver,
            "max_iter": settings.max_iterations,
            "max_cpu_time": settings.max_seconds,
            "tol": settings.tolerance,
            "acceptable_tol": max(settings.tolerance, 1.0e-4),
            "constr_viol_tol": settings.tolerance,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            # Pyomo still exports every Var.value as the primal start; "no"
            # only stops IPOPT importing stale bound/constraint multipliers.
            "warm_start_init_point": "no",
            "print_level": 5 if settings.tee else 0,
        },
    )
    started = time.perf_counter()
    try:
        result = solver.solve(model, tee=settings.tee)
        termination = result.solver.termination_condition
        optimal = termination in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
        }
        has_solution = optimal or termination == pyo.TerminationCondition.feasible
        return SolveOutcome(
            str(termination), has_solution, optimal, time.perf_counter() - started
        )
    except Exception as exc:  # a crashed solve is a failed start, not a run failure
        return SolveOutcome(
            f"error: {exc}", False, False, time.perf_counter() - started
        )


def maximum_bound_violation(model: pyo.ConcreteModel) -> float:
    """Worst violation of any variable or constraint bound at the current point.

    Returns infinity when anything is unevaluable, so a broken point can never
    be mistaken for a feasible one.
    """

    worst = 0.0
    components = list(model.component_data_objects(pyo.Var)) + list(
        model.component_data_objects(pyo.Constraint, active=True)
    )
    for component in components:
        body = component if component.ctype is pyo.Var else component.body
        value = pyo.value(body, exception=False)
        if value is None or not math.isfinite(value):
            return math.inf
        for bound, sign in ((component.lb, -1), (component.ub, 1)):
            if bound is not None:
                worst = max(worst, sign * (value - pyo.value(bound)))
    return worst
