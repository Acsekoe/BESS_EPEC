"""Solve a tiny nonlinear Pyomo model with the project Ipopt setup."""

from __future__ import annotations

import argparse

import pyomo.environ as pyo

from solver_utils import find_ipopt_executable, get_ipopt_solver


def build_model() -> pyo.ConcreteModel:
    model = pyo.ConcreteModel(name="Ipopt smoke test")
    model.x = pyo.Var(bounds=(-10.0, 10.0), initialize=0.0)
    model.objective = pyo.Objective(expr=(model.x - 1.0) ** 2)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that Pyomo can solve an NLP with Ipopt.")
    parser.add_argument("--tee", action="store_true", help="Show Ipopt output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    executable = find_ipopt_executable()
    print(f"Ipopt executable: {executable if executable is not None else 'not found explicitly'}")

    model = build_model()
    solver = get_ipopt_solver()
    results = solver.solve(model, tee=args.tee)
    termination = results.solver.termination_condition
    status = results.solver.status

    print(f"Solver status: {status}")
    print(f"Termination: {termination}")
    print(f"x: {pyo.value(model.x):.8f}")
    print(f"objective: {pyo.value(model.objective):.8e}")

    return 0 if termination == pyo.TerminationCondition.optimal else 1


if __name__ == "__main__":
    raise SystemExit(main())
