"""Build the modified IEEE 5-bus case: model/input/ieee5_market_data.json and
model/input/ieee5_capacities.json. The JSON files carry every number; this
script only documents how they were made.

Topology (modified IEEE 5-bus figure): lines 1-2, 1-4, 2-3, 2-4, 3-5, 4-5;
generators at buses 1, 2, 3, 5; loads at buses 1, 3, 4, 5. Reactances of 1-2,
2-3, 2-4, 4-5 are the per-unit values of the IEEE 5-bus line table (Stagg &
El-Abiad; Bhandakkar & Mathew 2018, 230 kV, 100 MVA, 60 Hz); 1-4 and 3-5 are
assumed. Loads, generators, costs, and line limits are designed for two
pockets: PV at bus 5 is export-constrained at midday (L35, L45), wind at bus 2
at night (L12, L23). Hourly shapes are the IEEE-9 day
(model/input/market_data.json), normalized to peak 1.

python scripts/data/build_ieee5.py .
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ieee9 = json.loads((root / "model/input/market_data.json").read_text(encoding="utf-8"))
times = ieee9["times"]

# Hourly shapes from the IEEE-9 day, peak 1.
demand9 = {t: sum(r["demand_mw"] for r in ieee9["demand_el"] if r["hour"] == t) for t in times}
cap9 = {(r["generator"], r["hour"]): r["capacity_mw"] for r in ieee9["generation_capacity"]}
demand_shape = {t: demand9[t] / max(demand9.values()) for t in times}
wind_shape = {t: cap9["RES_Wind_N1", t] / max(cap9["RES_Wind_N1", s] for s in times) for t in times}
pv_shape = {t: cap9["RES_PV_N8", t] / max(cap9["RES_PV_N8", s] for s in times) for t in times}

nodes = ["N1", "N2", "N3", "N4", "N5"]
lines = {  # name: (from, to, reactance p.u., limit MW)
    "L12": ("N1", "N2", 0.06, 80.0),   # wind export corridor
    "L14": ("N1", "N4", 0.24, 300.0),  # reactance assumed
    "L23": ("N2", "N3", 0.18, 60.0),   # wind export corridor
    "L24": ("N2", "N4", 0.18, 300.0),
    "L35": ("N3", "N5", 0.12, 120.0),  # reactance assumed; PV export corridor
    "L45": ("N4", "N5", 0.24, 100.0),  # PV export corridor
}
generators = {  # name: (node, cost EUR/MWh, peak MW, shape)
    "G1": ("N1", 52.0, 250.0, None),
    "RES_Wind_N2": ("N2", 0.0, 250.0, wind_shape),
    "G3": ("N3", 67.0, 250.0, None),
    "RES_PV_N5": ("N5", 0.0, 350.0, pv_shape),
}
peak_load = {"N1": 96.0, "N2": 0.0, "N3": 144.0, "N4": 120.0, "N5": 40.0}
slack = "N1"


def ptdf_matrix():
    """DC shift factors: flow on line l per MW injected at n and withdrawn at the slack."""
    index = [n for n in nodes if n != slack]
    size = len(index)
    b = [[0.0] * size for _ in range(size)]
    for frm, to, x, _ in lines.values():
        for a, c in ((frm, to), (to, frm)):
            if a in index:
                b[index.index(a)][index.index(a)] += 1 / x
                if c in index:
                    b[index.index(a)][index.index(c)] -= 1 / x
    result = {}
    for n in nodes:
        rhs = [1.0 if m == n else 0.0 for m in index]
        theta = solve_linear(b, rhs) if n != slack else [0.0] * size
        angle = {m: theta[k] for k, m in enumerate(index)} | {slack: 0.0}
        for line, (frm, to, x, _) in lines.items():
            result[line, n] = (angle[frm] - angle[to]) / x
    return result


def solve_linear(a, rhs):
    """Gaussian elimination with partial pivoting (no numpy here)."""
    size = len(rhs)
    m = [row[:] + [rhs[k]] for k, row in enumerate(a)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(size):
            if r != col:
                factor = m[r][col] / m[col][col]
                m[r] = [v - factor * p for v, p in zip(m[r], m[col])]
    return [m[k][size] / m[k][k] for k in range(size)]


ptdf = ptdf_matrix()
data = {
    "metadata": {
        "description": ("Modified IEEE 5-bus case for BESS siting with two pockets. Lines 1-2, 1-4, 2-3, 2-4, 3-5, 4-5; "
                        "G1 thermal 250 MW at 52 EUR/MWh (N1), wind 250 MW (N2), G3 thermal 250 MW at 67 EUR/MWh (N3), "
                        "PV 350 MW (N5); peak loads N1 96, N3 144, N4 120, N5 40 MW. PV export corridor "
                        "L35 120 MW and L45 100 MW (N5 export-constrained at midday), wind export corridor "
                        "L12 80 MW and L23 60 MW (N2 export-constrained at night), L14 and L24 300 MW. "
                        "Reactances of L12, L23, L24, L45 from the IEEE 5-bus table (Stagg & El-Abiad; "
                        "Bhandakkar & Mathew 2018); L14 and L35 assumed. Hourly shapes: IEEE-9 day, peak 1. "
                        "Loads, generators, costs, and limits are design choices. Built by scripts/data/build_ieee5.py."),
        "slack_node": slack,
        "line_reactance_pu": {line: x for line, (_, _, x, _) in lines.items()},
        "line_ends": {line: [frm, to] for line, (frm, to, _, _) in lines.items()},
    },
    "nodes": nodes,
    "generators": list(generators),
    "times": times,
    "soc_times": ieee9["soc_times"],
    "lines": list(lines),
    "generators_at_node": {n: [g for g, spec in generators.items() if spec[0] == n] for n in nodes},
    "generation_cost": {g: spec[1] for g, spec in generators.items()},
    "generation_capacity": [dict(generator=g, hour=t, capacity_mw=round(spec[2] * (spec[3][t] if spec[3] else 1.0), 6))
                            for g, spec in generators.items() for t in times],
    "demand_el": [dict(node=n, hour=t, demand_mw=round(peak_load[n] * demand_shape[t], 6)) for n in nodes for t in times],
    "voll": ieee9["voll"],
    "line_limit": {line: spec[3] for line, spec in lines.items()},
    "ptdf": [dict(line=line, node=n, ptdf=round(ptdf[line, n], 12)) for line in lines for n in nodes],
    "eta": ieee9["eta"],
}
(root / "model/input/ieee5_market_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

# Investors as in capacities_capex120_curtail_d3_slope1e-4.json; ownership mapped to the
# 5-bus plants: I3 80 % wind / 20 % PV, I4 20 % wind / 80 % PV; curtailment slope 0.1 on both.
profile = json.loads((root / "model/input/capacities_capex120_curtail_d3_slope1e-4.json").read_text(encoding="utf-8"))
for config in profile.values():
    config["power_mw"], config["energy_mwh"] = {}, {}
profile["I3"]["owned_generation_shares"] = {"RES_Wind_N2": 0.8, "RES_PV_N5": 0.2}
profile["I4"]["owned_generation_shares"] = {"RES_Wind_N2": 0.2, "RES_PV_N5": 0.8}
for i in ("I3", "I4"):
    profile[i]["curtailment_slope_eur_per_mwh_per_mw"] = {"RES_Wind_N2": 0.1, "RES_PV_N5": 0.1}
(root / "model/input/ieee5_capacities.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
print("PTDF (bus 1 reference):")
for line in lines:
    print(f"  {line}: " + "  ".join(f"{n} {ptdf[line, n]:+.3f}" for n in nodes))
