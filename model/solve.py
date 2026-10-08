"""Gurobi solve and numerical feasibility checks shared by the model runners.

Every LP, MILP, and bilinear MPEC in model/ is solved by Gurobi.
"""
import pyomo.environ as pyo


def make_solver():
    return pyo.SolverFactory("gurobi_direct")


def solver_version():
    return ".".join(map(str, make_solver().version()))


def solve(m, *, seconds=60, mip_gap=1e-6, solver=None):
    solver = solver or make_solver()
    # NonConvex=2: spatial branch-and-bound proves global optimality for price*quantity.
    # IntFeasTol=1e-9: a binary within 1e-5 of integral times dual_m=1e5 would
    # otherwise leave complementarity products of about 1e-3.
    solver.options.update({"TimeLimit": seconds, "MIPGap": mip_gap, "NonConvex": 2,
                           "IntFeasTol": 1e-9})
    result = solver.solve(m, load_solutions=False)
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        m.solutions.load_from(result)
    return result


def require_optimal(m, **kwargs):
    result = solve(m, **kwargs)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solve did not prove optimality: {result.solver.termination_condition}")
    return result


def residual(m):
    errors = [0.0]
    for c in m.component_data_objects(pyo.Constraint, active=True):
        body = pyo.value(c.body)
        if c.has_lb():
            errors.append(pyo.value(c.lower) - body)
        if c.has_ub():
            errors.append(body - pyo.value(c.upper))
    for v in m.component_data_objects(pyo.Var):
        if v.lb is not None:
            errors.append(v.lb - pyo.value(v))
        if v.ub is not None:
            errors.append(pyo.value(v) - v.ub)
    return max(errors)
