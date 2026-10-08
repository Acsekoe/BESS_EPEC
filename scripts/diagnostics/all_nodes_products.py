"""All-nodes I1 bilinear MPEC: which complementarity products exceed 1e-5, and does FeasibilityTol fix them?"""
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "model"))
import pyomo.environ as pyo

from prepare_input import load_case
from mpec import build_mpec, complementarity_pairs
from solve import solve, residual

for extra in ({}, {"FeasibilityTol": 1e-9}):
    data, profile = load_case()
    m = build_mpec(data, profile, "I1")
    solver = pyo.SolverFactory("gurobi_direct")
    solver.options.update(extra)
    start = time.time()
    result = solve(m, seconds=600, solver=solver)
    print(f"options {extra}: {result.solver.termination_condition} in {time.time() - start:.0f}s,"
          f" profit {pyo.value(m.profit):.6f}, gap {pyo.value(m.market_cost - m.dual_value):.2e},"
          f" violation {residual(m):.2e}", flush=True)
    rows = sorted(((abs(pyo.value(s * u)), pyo.value(s), pyo.value(u), u.name) for s, u in complementarity_pairs(m)),
                  reverse=True)
    for product, slack, mu, name in rows[:6]:
        print(f"    {name:40s} slack {slack: .3e} mu {mu: .3e} product {product:.2e}")
    print(f"    products > 1e-4: {sum(r[0] > 1e-4 for r in rows)}", flush=True)
