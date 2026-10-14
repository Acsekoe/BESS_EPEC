"""Build the 5-bus test system, the final market data (since 2026-10-14):
model/input/five_bus_market_data.json and model/input/five_bus_capacities.json.
The JSON files carry every number; this script only documents how they were made.

Modified IEEE 5-bus case with two radial pockets. Lines 1-2, 1-3, 1-4, 3-4,
4-5 are the IEEE 5-bus table lines 1-2, 2-3, 2-4, 3-4, 4-5 with table buses 1
and 2 relabeled N2 and N1; every reactance is from the table (Stagg &
El-Abiad; Bhandakkar & Mathew 2018, 230 kV, 100 MVA, 60 Hz). Wind corridor L12
140 MW, PV corridor L45 120 MW, L13/L14/L34 300 MW. Wind 200 MW (N2), PV 300 MW
(N5); peak loads N1 120, N3 180, N4 150, N5 40 MW. The pocket lines (260 MW)
stay below the minimum load outside the pockets, so the thermal unit always
sets the price at N1/N3/N4.

Thermal: one unit G1 at N1, 600 MW, marginal cost 41 + 0.075 * g EUR/MWh,
standing for a thermal merit order. Calibrated so that, without storage, the
hub price spans 52 EUR/MWh at the lowest residual load (147.5 MW, hour 16) to
66.7 at the highest (342 MW, hour 20): the range of the earlier G1 52 / G3 67
pair, but continuous, so storage moves the hub price smoothly instead of across
a step. Loads, generators, costs, and limits are design choices.

Flexible demand (added 2026-10-14): a price-responsive load, e.g. power-to-heat,
at each pocket node, N2 10 MW and N5 25 MW (about a sixth of each pocket's
peak surplus over its export line: N2 60 MW, N5 146.8 MW). Willingness to pay
falls linearly from 60 EUR/MWh at 0 MW to 0 at full size (stylized: roughly the
value of replacing gas-fired heat, gas plus CO2; not sourced). Without it, a
pocket's price jumps from the curtailment price (about 0) to the storage value
(about 47 EUR/MWh at N5) at one storage size; with it, storage first displaces
the load and the price rises along its curve. 60 lies above the storage value,
so no vertical piece remains. Size chosen 2026-10-14 from a scan of scales 0.1
to 2 (workflow/flexible_demand_2026-10-14/): ramps of about 16 MW (N2) and 35 MW
(N5) of battery power, while a merchant keeps 82-89 % of its arbitrage (twice
the size: 56-80 %). run.py --flexible-demand-scale scales it; 0 removes it.

Hourly shapes are the IEEE-9 day of the earlier IEEE-9 case (total demand, wind
at N1, PV at N8), normalized to peak 1. That case and the earlier 5-bus variants
(current; V5 with G1 250 MW and G3 300 MW at 67; V6 with G1 600 MW) are at
commit 82690b2 (model/input/, scripts/data/build_ieee5.py;
workflow/summary_2026-10-12_18-47.md). V7 (G1 300 MW at 52, G3 300 MW at 67)
was the 5-bus test system on 2026-10-14 until the quadratic G1 replaced it.

python scripts/data/build_five_bus.py .
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
root = parser.parse_args().root
times = list(range(1, 25))

# IEEE-9 day: total demand [MW], wind at N1 [MW], PV at N8 [MW].
demand9 = [446.49, 426.195, 412.665, 405.9, 419.43, 460.02, 541.2, 608.85, 629.145, 608.85, 595.32, 581.79,
           568.26, 561.495, 561.495, 575.025, 602.085, 635.91, 676.5, 669.735, 635.91, 575.025, 514.14, 473.55]
wind9 = [94.5, 98.4375, 102.375, 105.0, 102.375, 94.5, 81.375, 68.25, 55.125, 45.9375, 39.375, 36.75,
         39.375, 44.625, 52.5, 60.375, 68.25, 76.125, 81.375, 86.625, 89.25, 91.875, 94.5, 95.8125]
pv9 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.906625, 44.533125, 124.69275, 213.759, 302.82525, 378.5315625,
       423.0646875, 436.424625, 400.798125, 320.6385, 222.665625, 124.69275, 35.6265, 4.4533125, 0.0, 0.0, 0.0, 0.0]
demand_shape = {t: demand9[t - 1] / max(demand9) for t in times}
wind_shape = {t: wind9[t - 1] / max(wind9) for t in times}
pv_shape = {t: pv9[t - 1] / max(pv9) for t in times}

nodes = ["N1", "N2", "N3", "N4", "N5"]
slack = "N1"
lines = {  # name: (from, to, reactance p.u., limit MW)
    "L12": ("N1", "N2", 0.06, 140.0),  # wind export corridor (radial pocket)
    "L13": ("N1", "N3", 0.18, 300.0),
    "L14": ("N1", "N4", 0.18, 300.0),
    "L34": ("N3", "N4", 0.03, 300.0),
    "L45": ("N4", "N5", 0.24, 120.0),  # PV export corridor (radial pocket)
}
generators = {  # name: (node, cost EUR/MWh, peak MW, shape)
    "G1": ("N1", 41.0, 600.0, None),
    "RES_Wind_N2": ("N2", 0.0, 200.0, wind_shape),
    "RES_PV_N5": ("N5", 0.0, 300.0, pv_shape),
}
generation_cost_slope = {"G1": 0.075}  # EUR/MWh per MW: marginal cost 41 + 0.075 * g
peak_load = {"N1": 120.0, "N2": 0.0, "N3": 180.0, "N4": 150.0, "N5": 40.0}
# Pocket flexible demand: willingness to pay 60 * (1 - q / max_mw) EUR/MWh.
flexible_demand = {"N2": {"max_mw": 10.0, "max_price_eur_per_mwh": 60.0},
                   "N5": {"max_mw": 25.0, "max_price_eur_per_mwh": 60.0}}
description = ("5-bus test system (final market data since 2026-10-14): modified IEEE 5-bus case for BESS siting "
               "with two radial pockets (workflow/summary_2026-10-12_18-47.md). Lines 1-2, 1-3, 1-4, 3-4, 4-5 = IEEE "
               "5-bus table lines 1-2, 2-3, 2-4, 3-4, 4-5 with table buses 1 and 2 relabeled N2 and N1; all "
               "reactances from the table (Stagg & El-Abiad; Bhandakkar & Mathew 2018). Wind corridor L12 140 MW, "
               "PV corridor L45 120 MW, L13/L14/L34 300 MW. One thermal unit G1 (N1), 600 MW, marginal cost "
               "41 + 0.075 * g EUR/MWh (hub price 52-67 without storage); wind 200 MW (N2), PV 300 MW (N5); peak "
               "loads N1 120, N3 180, N4 150, N5 40 MW. Flexible demand (e.g. power-to-heat) N2 10 MW and N5 25 MW, "
               "willingness to pay falling linearly from 60 EUR/MWh at 0 MW to 0 at full size (stylized). Hourly "
               "shapes: IEEE-9 day, peak 1. Loads, generators, costs, and limits are design choices. Built by "
               "scripts/data/build_five_bus.py.")


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
        "description": description,
        "slack_node": slack,
        "line_reactance_pu": {line: x for line, (_, _, x, _) in lines.items()},
        "line_ends": {line: [frm, to] for line, (frm, to, _, _) in lines.items()},
    },
    "nodes": nodes,
    "generators": list(generators),
    "times": times,
    "soc_times": [0] + times,
    "lines": list(lines),
    "generators_at_node": {n: [g for g, spec in generators.items() if spec[0] == n] for n in nodes},
    "generation_cost": {g: spec[1] for g, spec in generators.items()},
    "generation_cost_slope": generation_cost_slope,
    "generation_capacity": [dict(generator=g, hour=t, capacity_mw=round(spec[2] * (spec[3][t] if spec[3] else 1.0), 6))
                            for g, spec in generators.items() for t in times],
    "demand_el": [dict(node=n, hour=t, demand_mw=round(peak_load[n] * demand_shape[t], 6)) for n in nodes for t in times],
    "flexible_demand": flexible_demand,
    "voll": 10000.0,
    "line_limit": {line: spec[3] for line, spec in lines.items()},
    "ptdf": [dict(line=line, node=n, ptdf=round(ptdf[line, n], 12)) for line in lines for n in nodes],
    "eta": 0.936,
}
(root / "model/input/five_bus_market_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

# Investors start at zero capacity. I1 merchant (WACC 8 %; the second merchant
# I2, the same at 12 %, was dropped 2026-10-14: it never entered); I3 owns
# 80 % wind / 20 % PV, I4 the rest, both with curtailment slope 0.1 EUR/MWh per
# MW. Energy cost 120 000 EUR/MWh (BESS system, IRENA 2026 at the ECB 2025 rate),
# power cost 1 EUR/MW (grid-access placeholder), degradation 3 EUR/MWh, dispatch
# slope 1e-4 (project_context.md). Lifetime and WACC are not sourced.
ownership = {"I1": {}, "I3": {"RES_Wind_N2": 0.8, "RES_PV_N5": 0.2},
             "I4": {"RES_Wind_N2": 0.2, "RES_PV_N5": 0.8}}
profile = {}
for i, wacc in {"I1": 0.08, "I3": 0.08, "I4": 0.08}.items():
    profile[i] = dict(wacc=wacc, lifetime_years=15, cost_power_eur_per_mw=1.0, cost_energy_eur_per_mwh=120000.0,
                      degradation_eur_per_mwh=3.0, ratio_min=2.0, ratio_max=8.0, power_mw={}, energy_mwh={},
                      owned_generation_shares=ownership[i])
    if ownership[i]:
        profile[i]["curtailment_slope_eur_per_mwh_per_mw"] = {"RES_Wind_N2": 0.1, "RES_PV_N5": 0.1}
    profile[i]["dispatch_cost_slope_eur_per_mwh_per_mw"] = 0.0001
(root / "model/input/five_bus_capacities.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
print("PTDF (bus 1 reference):")
for line in lines:
    print(f"  {line}: " + "  ".join(f"{n} {ptdf[line, n]:+.3f}" for n in nodes))
