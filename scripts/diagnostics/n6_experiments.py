"""N6 experiments: Gurobi integrality tolerance and the identity cut for the bilinear MPEC."""
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "model"))
import pyomo.environ as pyo

from prepare_input import load_case
from mpec import build_mpec, complementarity_pairs
from solve import solve, residual


def report(label, m, result, seconds):
    status = str(result.solver.termination_condition)
    line = f"{label}: {status} in {seconds:.0f}s, bounds {result.problem.lower_bound} / {result.problem.upper_bound}"
    if status == "optimal":
        pairs = list(complementarity_pairs(m))
        line += (f"\n    profit {pyo.value(m.profit):.6f} linear {pyo.value(m.profit_linear):.6f}"
                 f" identity err {abs(pyo.value(m.profit - m.profit_linear)):.2e}"
                 f" max comp {max(abs(pyo.value(s * u)) for s, u in pairs):.2e} viol {residual(m):.2e}"
                 f"\n    X_power {pyo.value(m.X_power['N6']):.6f} X_energy {pyo.value(m.X_energy['N6']):.6f}")
        saturated = [u.name for _, u in pairs if pyo.value(u) >= 0.999 * pyo.value(m.dual_m)]
        families = sorted({name.split("[")[0] for name in saturated})
        line += f"\n    saturated multipliers: {len(saturated)} in {families}"
    print(line, flush=True)


def run(label, objective, options, cut=False, seconds=300, dual_m=100000.0):
    data, profile = load_case()
    m = build_mpec(data, profile, "I1", nodes=["N6"], dual_m=dual_m, objective=objective)
    if cut:
        # Valid at every KKT point: direct settlement equals the strong-duality form.
        m.identity_cut = pyo.Constraint(expr=m.profit == m.profit_linear)
    solver = pyo.SolverFactory("gurobi_direct")
    solver.options.update(options)
    start = time.time()
    result = solve(m, seconds=seconds, solver=solver)
    report(label, m, result, time.time() - start)


run("A linear, gurobi defaults", "linear", {})
run("B linear, IntFeasTol=1e-9", "linear", {"IntFeasTol": 1e-9})
run("C bilinear + identity cut, IntFeasTol=1e-9", "bilinear", {"IntFeasTol": 1e-9}, cut=True)
run("D bilinear, IntFeasTol=1e-9, dual_m=2e4 (no cut)", "bilinear", {"IntFeasTol": 1e-9}, dual_m=20000.0)
