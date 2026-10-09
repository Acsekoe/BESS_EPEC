"""The PV-pocket step (1 day, eps = 1e-4 unique prices): hourly curtailment,
flows, and prices around I1's step capacity, and each investor's profit
decomposition just below, at, and just above the step.

Run with the repo root as first argument. Inputs are the relaxed-MPEC
capacity profiles in workflow/step_analysis_2026-10-09/.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
PROFILES = ROOT / "workflow" / "step_analysis_2026-10-09"
sys.path.insert(0, str(ROOT / "model"))
import pyomo.environ as pyo

from prepare_input import load_case
from solution_space import unique_price_reclear

EPS = 1e-4
POCKET = ("N3", "N6", "N8", "N9")
PV = ("RES_PV_N6", "RES_PV_N8")
data, _ = load_case()


def scaled(profile, investor, factor):
    q = copy.deepcopy(profile)
    for key in ("power_mw", "energy_mwh"):
        q[investor][key] = {n: v * factor for n, v in q[investor][key].items()}
    return q


def hourly_table(profile, investor, label):
    p, price = unique_price_reclear(data, profile, EPS)
    print(f"--- {label}: {investor} pocket {sum(profile[investor]['power_mw'].values()):.3f} MW")
    print(" h  PVavail  curtail  charge  disch    L46    L78 | N6price N8price N5price")
    for t in data["times"]:
        available = sum(pyo.value(p.generation_capacity[g, t]) for g in PV)
        used = sum(pyo.value(p.generation[g, t]) for g in PV)
        charge = sum(pyo.value(p.charge[i, n, t]) for i, n in p.storage_pairs if n in POCKET)
        discharge = sum(pyo.value(p.discharge[i, n, t]) for i, n in p.storage_pairs if n in POCKET)
        print(f"{t:2d} {available:8.2f} {available - used:8.2f} {charge:7.2f} {discharge:6.2f} "
              f"{pyo.value(p.line_flow['L46', t]):6.1f} {pyo.value(p.line_flow['L78', t]):6.1f} | "
              f"{price['N6', t]:7.2f} {price['N8', t]:7.2f} {price['N5', t]:7.2f}")


def decomposition(profile, investor, label):
    print(f"{label}: {investor} {sum(profile[investor]['power_mw'].values()):.3f} MW, "
          f"{sum(profile[investor]['energy_mwh'].values()):.3f} MWh")
    for factor in (0.0, 0.999, 1.0, 1.001):
        q = scaled(profile, investor, factor)
        p, price = unique_price_reclear(data, q, EPS)
        pairs = [(i, n) for i, n in p.storage_pairs if i == investor]
        buy = sum(price[n, t] * pyo.value(p.charge[i, n, t]) for i, n in pairs for t in p.timesteps)
        sell = sum(price[n, t] * pyo.value(p.discharge[i, n, t]) for i, n in pairs for t in p.timesteps)
        degradation = sum(0.5 * pyo.value(p.degradation[i]) * pyo.value(p.charge[i, n, t] + p.discharge[i, n, t])
                          for i, n in pairs for t in p.timesteps)
        generation = sum(pyo.value(p.owned_share[investor, g])
                         * (price[pyo.value(p.generator_node[g]), t] - pyo.value(p.generation_cost[g]))
                         * pyo.value(p.generation[g, t]) for g in p.generators for t in p.timesteps)
        capex = sum(pyo.value(p.cost_power_daily[investor]) * q[investor]["power_mw"][n]
                    + pyo.value(p.cost_energy_daily[investor]) * q[investor]["energy_mwh"][n]
                    for n in q[investor]["power_mw"])
        midday = [round(price["N6", t], 2) for t in (12, 13, 14, 15)]
        print(f"  x{factor:<6} sell {sell:9.0f} buy {buy:8.0f} degradation {degradation:6.0f} "
              f"own generation {generation:9.0f} capex {capex:6.0f} -> profit "
              f"{sell - buy - degradation + generation - capex:9.0f} | N6 h12-15 {midday}", flush=True)


i1 = json.loads((PROFILES / "I1_pocket_eps1e-4.json").read_text())
hourly_table(scaled(i1, "I1", 0.0), "I1", "no storage")
for factor in (0.999, 1.0, 1.001):
    hourly_table(scaled(i1, "I1", factor), "I1", f"x{factor}")
decomposition(i1, "I1", "I1 relaxed MPEC, pocket nodes, eps=1e-4")
decomposition(json.loads((PROFILES / "I4_eps1e-4.json").read_text()), "I4", "I4 relaxed MPEC, all nodes, eps=1e-4")
decomposition(json.loads((PROFILES / "I4_eps0.json").read_text()), "I4", "I4 relaxed MPEC, all nodes, eps=0")
