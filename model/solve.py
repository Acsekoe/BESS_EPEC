"""HiGHS solve and numerical feasibility checks shared by the model runners."""
import pyomo.environ as pyo


def solve(m, *, seconds=60, mip_gap=1e-6, solver=None):
    solver = solver or pyo.SolverFactory("highs")
    solver.options.update({"time_limit": seconds, "mip_rel_gap": mip_gap})
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
