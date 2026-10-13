"""Root relaxation of the SOS1 MPEC versus the social-planner value of storage.

Dropping the SOS1 sets is the only thing that separates the MPEC root from the
MPEC. Without complementarity the active investor's own multipliers
(mu_power, mu_soc_upper, ...) are free, so its stationarity rows no longer tie
prices to its capacity X: the duals can keep the no-storage prices (dual value
C(0)) while the primal dispatch uses the storage (cost C(X)). profit_linear then
becomes the system cost saving of the storage:

    root relaxation = planner value = C(0) - min_X [C(X) + capex(X)]

C(X) is the market cost with the active investor's capacity X and fixed rivals.
This holds exactly for the pure LP (eps = 0, no curtailment or dispatch slope).
With the quadratic terms the root picks the same capacity and a nearby value.
No complementarity formulation can therefore change the root; only branching
or decomposition can.

Both problems are convex QPs (LPs with --lp-variant).
--solver gurobi (default) uses model/solve.py and needs a full licence for
IEEE-9/5-bus. --solver clarabel writes an LP file, reads its matrices with HiGHS
and solves with Clarabel (pip install clarabel highspy scipy); with eps = 1e-4
it agrees with Gurobi's root to about 0.1 EUR (I1/N9: 7742.81 vs 7742.93).

python scripts/diagnostics/root_vs_planner.py . --investor I1 --nodes N9 --profile model/input/capacities_capex120_curtail_d3_slope1e-4.json --balancing-eps 1e-4
python scripts/diagnostics/root_vs_planner.py . --data model/input/ieee5_market_data.json --profile model/input/ieee5_capacities.json --investor I1 --balancing-eps 1e-4
Add --lp-variant to drop curtailment and dispatch slopes and set eps to 0.
"""
import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--data", type=Path, default=Path("model/input/market_data.json"))
parser.add_argument("--profile", type=Path, default=Path("model/input/capacities_capex120_curtail_d3_slope1e-4.json"))
parser.add_argument("--investor", default="I1")
parser.add_argument("--nodes", nargs="+", help="Candidate nodes (default: all)")
parser.add_argument("--balancing-eps", type=float, default=1e-4)
parser.add_argument("--lp-variant", action="store_true",
                    help="Remove curtailment and dispatch slopes and use eps = 0 (pure LP)")
parser.add_argument("--solver", choices=("gurobi", "clarabel"), default="gurobi")
args = parser.parse_args()
sys.path.insert(0, str(args.root / "model"))
import pyomo.environ as pyo
import mpec
from prepare_input import load_case, prepare_input
from primal_llp import add_primal


def lp_variant(profile):
    """Same investors without curtailment or dispatch cost slopes."""
    result = copy.deepcopy(profile)
    for config in result.values():
        config.pop("curtailment_slope_eur_per_mwh_per_mw", None)
        config["dispatch_cost_slope_eur_per_mwh_per_mw"] = 0.0
    return result


def solve_gurobi(m):
    from solve import require_optimal
    require_optimal(m, seconds=600)


def solve_clarabel(m):
    """Write an LP file, read its matrices with HiGHS, solve with Clarabel, load the values."""
    import clarabel
    import highspy
    import numpy as np
    import scipy.sparse as sp

    fd, path = tempfile.mkstemp(suffix=".lp"); os.close(fd)
    _, symbol_map_id = m.write(path, io_options={"symbolic_solver_labels": True})
    symbol_map = m.solutions.symbol_map[symbol_map_id]
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.readModel(path); os.remove(path)
    lp, hessian = h.getModel().lp_, h.getModel().hessian_
    n = lp.num_col_
    sense = 1.0 if lp.sense_ == highspy.ObjSense.kMinimize else -1.0  # Clarabel minimizes
    q = sense * np.array(lp.col_cost_)
    P = sp.csc_matrix((n, n))
    if hessian.dim_ > 0:
        H = sp.csc_matrix((hessian.value_, hessian.index_, hessian.start_), shape=(n, n))
        if hessian.format_ == highspy.HessianFormat.kTriangular:
            H = H + sp.triu(H, 1).T
        P = sp.triu(sense * H).tocsc()
    a = lp.a_matrix_
    A = sp.csc_matrix((a.value_, a.index_, a.start_), shape=(lp.num_row_, n)).tocsr()
    I = sp.identity(n, format="csr")
    zero_rows, zero_rhs, ineq_rows, ineq_rhs = [], [], [], []
    for M, lower, upper in ((A, np.array(lp.row_lower_), np.array(lp.row_upper_)),
                            (I, np.array(lp.col_lower_), np.array(lp.col_upper_))):
        equal = np.isfinite(lower) & np.isfinite(upper) & (np.abs(upper - lower) < 1e-12)
        has_upper, has_lower = np.isfinite(upper) & ~equal, np.isfinite(lower) & ~equal
        zero_rows.append(M[equal]); zero_rhs.append(lower[equal])
        ineq_rows += [M[has_upper], -M[has_lower]]; ineq_rhs += [upper[has_upper], -lower[has_lower]]
    Az, An = sp.vstack(zero_rows), sp.vstack(ineq_rows)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
    solution = clarabel.DefaultSolver(
        P, q, sp.vstack([Az, An]).tocsc(), np.concatenate(zero_rhs + ineq_rhs),
        [clarabel.ZeroConeT(Az.shape[0]), clarabel.NonnegativeConeT(An.shape[0])], settings).solve()
    if str(solution.status) != "Solved":
        raise RuntimeError(f"Clarabel: {solution.status}")
    for name, value in zip(lp.col_names_, solution.x):
        component = symbol_map.bySymbol.get(name)
        if component is not None:
            component.set_value(float(value), skip_validation=True)


solve = solve_gurobi if args.solver == "gurobi" else solve_clarabel


def capacities(m):
    return {n: (round(pyo.value(m.X_power[n]), 2), round(pyo.value(m.X_energy[n]), 2))
            for n in m.investment_nodes if pyo.value(m.X_power[n]) > 1e-3}


def root_relaxation(data, profile, eps):
    """The MPEC of mpec.py with every SOS1 set deactivated."""
    m = mpec.build_mpec(data, profile, args.investor, nodes=nodes, balancing_eps=eps)
    for sos in m.component_objects(pyo.SOSConstraint):
        sos.deactivate()
    solve(m)
    return pyo.value(m.profit_linear), capacities(m)


def planner(data, profile, eps):
    """min market_cost + capex over the active investor's capacity (same rules as mpec.py)."""
    m = prepare_input(data, profile, active_investor=args.investor, investment_nodes=nodes, balancing_eps=eps)
    mpec.add_investment(m)
    add_primal(m)
    m.capex = pyo.Expression(rule=mpec.investment_cost_rule)
    m.objective = pyo.Objective(expr=m.market_cost + m.capex, sense=pyo.minimize)
    for n in m.investment_nodes:  # C(0): no active storage
        m.X_power[n].fix(0.0)
        m.X_energy[n].fix(0.0)
    solve(m)
    no_storage_cost = pyo.value(m.market_cost)
    for n in m.investment_nodes:
        m.X_power[n].unfix()
        m.X_energy[n].unfix()
    solve(m)
    return no_storage_cost - pyo.value(m.market_cost + m.capex), capacities(m)


data, profile = load_case(args.root / args.data, args.root / args.profile)
nodes = args.nodes or data["nodes"]
eps = 0.0 if args.lp_variant else args.balancing_eps
if args.lp_variant:
    profile = lp_variant(profile)
root_value, root_capacity = root_relaxation(data, profile, eps)
planner_value, planner_capacity = planner(data, profile, eps)
print(f"{args.investor} at {' '.join(nodes)}, eps {eps}{', LP variant' if args.lp_variant else ''}, {args.solver}")
print(f"  MPEC root relaxation {root_value:12.2f} EUR/day  MW/MWh {root_capacity}")
print(f"  planner value        {planner_value:12.2f} EUR/day  MW/MWh {planner_capacity}")
print(f"  difference           {planner_value - root_value:12.2f}")
