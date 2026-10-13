"""Solver interfaces and numerical feasibility checks shared by the model runners.

Gurobi solves every market LP/QP, range problem, and SOS1 MPEC in model/. Ipopt
solves only the unique-price reclear with balancing (solution_space.py);
ipopt.exe must be on PATH.
"""
import math

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
    # max_wall_time exists from Ipopt 3.14; older builds (IDAES 3.13) only limit CPU time.
    time_limit = "max_wall_time" if solver.version() >= (3, 14) else "max_cpu_time"
    options = {time_limit: seconds, "tol": 1e-8, "constr_viol_tol": 1e-8}
    options.update(solver.options)  # options set on a passed solver take precedence
    if log_file is not None:
        options["output_file"] = str(log_file)
    result = solver.solve(m, options=options, load_solutions=False)
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        m.solutions.load_from(result)
    return result


def solve(m, *, seconds=60, mip_gap=1e-6, solver=None):
    solver = solver or make_solver()
    # PreSOS1BigM=0: presolve may otherwise rewrite the MPEC's SOS1 sets as Big-M
    # binaries again. IntFeasTol=1e-9: an SOS1 member below IntFeasTol counts as
    # zero; on I1/N9 1e-9 also cuts the search from 95 583 to 22 230 nodes.
    # NonConvex=2: ranges may maximize a convex quadratic profit over a dual face
    # (dispatch cost slope > 0).
    solver.options.update({"TimeLimit": seconds, "MIPGap": mip_gap, "NonConvex": 2, "PreSOS1BigM": 0,
                           "IntFeasTol": 1e-9})
    result = solver.solve(m, load_solutions=False)
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        m.solutions.load_from(result)
    return result


def objective_bounds(solver):
    """Gurobi's incumbent and best bound, read directly: Pyomo reports the incumbent
    as both bounds for models without integer variables, including SOS1 MPECs."""
    model = solver._solver_model
    incumbent = model.ObjVal if model.SolCount > 0 else None
    try:
        bound = model.ObjBound if model.IsMIP else incumbent
    except AttributeError:  # gurobipy: attribute not available (e.g. no bound computed)
        bound = None
    return tuple(value if value is not None and math.isfinite(value) else None for value in (incumbent, bound))


def load_incumbent(solver):
    """Load Gurobi's best solution after a time limit; solve() loads only proven
    optima. Returns False if Gurobi has no solution."""
    if solver._solver_model.SolCount == 0:
        return False
    for variable, gurobi_variable in solver._pyomo_var_to_solver_var_map.items():
        variable.set_value(gurobi_variable.X, skip_validation=True)
    return True


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
