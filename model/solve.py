"""Solver interfaces and numerical feasibility checks shared by the model runners.

Gurobi solves every LP, MILP, and bilinear MPEC in model/. Ipopt solves only the
relaxed-complementarity MPEC (mpec_relaxed.py); ipopt.exe must be on PATH.
"""
import pyomo.environ as pyo


def make_solver():
    return pyo.SolverFactory("gurobi_direct")


def solver_version():
    return ".".join(map(str, make_solver().version()))


def make_nlp_solver():
    return pyo.SolverFactory("ipopt")


def nlp_solver_version():
    return ".".join(map(str, make_nlp_solver().version()))


def solve_nlp(m, *, seconds=60, solver=None, log_file=None):
    """Local NLP solve; status "optimal" means a locally optimal point only."""
    solver = solver or make_nlp_solver()
    # Ipopt's default constr_viol_tol=1e-4 would allow products of 2*epsilon at
    # epsilon=1e-4, so constraint violations are held well below epsilon.
    options = {"max_wall_time": seconds, "tol": 1e-8, "constr_viol_tol": 1e-8}
    options.update(solver.options)  # options set on a passed solver take precedence
    if log_file is not None:
        options["output_file"] = str(log_file)
    result = solver.solve(m, options=options, load_solutions=False)
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        m.solutions.load_from(result)
    return result


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
