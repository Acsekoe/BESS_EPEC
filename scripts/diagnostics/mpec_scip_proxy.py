"""Solve the SOS1 MPEC of model/mpec.py with SCIP (open source) instead of Gurobi.

A proxy for machines without a full Gurobi licence. It writes the Pyomo MPEC
as an LP file (SOS1 sets included) and solves it with SCIP. SCIP handles the
quadratic objective with nonlinear machinery and is very slow on it, so use
--lp-variant (eps = 0, no curtailment or dispatch slopes: an MILP with SOS1).
That variant reintroduces optimistic price selection, so its optima are not
results of the working model; they show how the branch-and-bound bound moves
with the number of candidate nodes. pip install pyscipopt (SCIP 10).

python scripts/diagnostics/mpec_scip_proxy.py . --data model/input/ieee5_market_data.json --profile model/input/ieee5_capacities.json --investor I1 --lp-variant --node-sets N5 N2 N2,N5 N1,N2,N3,N4,N5 --seconds 240
"""
import argparse
import copy
import os
import sys
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--data", type=Path, default=Path("model/input/market_data.json"))
parser.add_argument("--profile", type=Path, default=Path("model/input/capacities_capex120_curtail_d3_slope1e-4.json"))
parser.add_argument("--investor", default="I1")
parser.add_argument("--node-sets", nargs="+", required=True, help="Comma-separated candidate node sets, e.g. N9 N3,N6,N8,N9")
parser.add_argument("--balancing-eps", type=float, default=1e-4)
parser.add_argument("--lp-variant", action="store_true",
                    help="Remove curtailment and dispatch slopes and use eps = 0 (MILP)")
parser.add_argument("--seconds", type=float, default=240)
args = parser.parse_args()
sys.path.insert(0, str(args.root / "model"))
import pyomo.environ as pyo
from pyscipopt import Model
from mpec import build_mpec
from prepare_input import load_case


def lp_variant(profile):
    result = copy.deepcopy(profile)
    for config in result.values():
        config.pop("curtailment_slope_eur_per_mwh_per_mw", None)
        config["dispatch_cost_slope_eur_per_mwh_per_mw"] = 0.0
    return result


def solve_scip(m, seconds):
    fd, path = tempfile.mkstemp(suffix=".lp"); os.close(fd)
    _, symbol_map_id = m.write(path, io_options={"symbolic_solver_labels": True})
    symbol_map = m.solutions.symbol_map[symbol_map_id]
    text = Path(path).read_text().replace(" ^ 2", "^2")  # SCIP's LP reader wants x^2
    Path(path).write_text(text)
    scip = Model()
    scip.hideOutput(True)
    scip.readProblem(path)
    os.remove(path)
    scip.setParam("limits/time", seconds)
    start = time.time()
    scip.optimize()
    seconds_used = time.time() - start
    if scip.getNSols():
        best = scip.getBestSol()
        for v in scip.getVars():
            component = symbol_map.bySymbol.get(v.name)
            if component is not None:
                component.set_value(scip.getSolVal(best, v), skip_validation=True)
    incumbent = scip.getObjVal() if scip.getNSols() else None
    return scip.getStatus(), incumbent, scip.getDualbound(), seconds_used, scip.getNNodes()


data, profile = load_case(args.root / args.data, args.root / args.profile)
eps = 0.0 if args.lp_variant else args.balancing_eps
if args.lp_variant:
    profile = lp_variant(profile)
print(f"{args.investor}, eps {eps}{', LP variant' if args.lp_variant else ''}, SCIP {Model().version()}, "
      f"{args.seconds:.0f} s per node set", flush=True)
for node_set in args.node_sets:
    nodes = node_set.split(",")
    m = build_mpec(data, profile, args.investor, nodes=nodes, balancing_eps=eps)
    status, incumbent, bound, seconds_used, tree_nodes = solve_scip(m, args.seconds)
    capacity = {n: (round(pyo.value(m.X_power[n]), 1), round(pyo.value(m.X_energy[n]), 1))
                for n in nodes if incumbent is not None and pyo.value(m.X_power[n]) > 1e-3}
    incumbent_text = f"{incumbent:10.2f}" if incumbent is not None else "      none"
    print(f"{node_set:16s} {status:9s} incumbent {incumbent_text} bound {bound:10.2f} "
          f"{seconds_used:6.1f} s {tree_nodes:8d} nodes  MW/MWh {capacity}", flush=True)
