"""Read, validate, and prepare input sets and parameters; no model equations."""
import json
import math
from pathlib import Path

import pyomo.environ as pyo

INPUT = Path(__file__).parent / "input"


def load_case(data_path=INPUT / "market_data.json", profile_path=INPUT / "capacities.json"):
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    return data, profile


def validate(data, profile):
    if data["times"] != list(range(1, len(data["times"]) + 1)) or not data["times"]:
        raise ValueError("The baseline requires consecutive one-hour periods starting at 1.")
    if data["soc_times"] != list(range(len(data["times"]) + 1)):
        raise ValueError("SOC times must include 0 and every market hour.")
    if not 0 < data["eta"] <= 1 or data["voll"] <= 0:
        raise ValueError("Invalid efficiency or value of lost load.")
    assigned = [g for n in data["nodes"] for g in data["generators_at_node"][n]]
    if sorted(assigned) != sorted(data["generators"]):
        raise ValueError("Each generator must belong to exactly one node.")
    for investor, config in profile.items():
        if not 0 <= config["ratio_min"] <= config["ratio_max"] or config["ratio_max"] <= 0:
            raise ValueError(f"Invalid duration limits for {investor}.")
        for field in ("wacc", "cost_power_eur_per_mw", "cost_energy_eur_per_mwh", "degradation_eur_per_mwh"):
            if not math.isfinite(config[field]) or config[field] < 0:
                raise ValueError(f"Invalid {field} for {investor}.")
        if config["lifetime_years"] <= 0:
            raise ValueError("Lifetime must be positive.")
        for field in ("power_mw", "energy_mwh"):
            if set(config[field]) - set(data["nodes"]):
                raise ValueError(f"Unknown capacity node for {investor}.")
        for n in data["nodes"]:
            power = config["power_mw"].get(n, 0.0)
            energy = config["energy_mwh"].get(n, 0.0)
            if not all(math.isfinite(v) and v >= 0 for v in (power, energy)):
                raise ValueError("Capacities must be finite and nonnegative.")
            if not config["ratio_min"] * power - 1e-9 <= energy <= config["ratio_max"] * power + 1e-9:
                raise ValueError(f"Capacity violates duration bounds: {investor}, {n}.")
        for g, share in config.get("owned_generation_shares", {}).items():
            if g not in data["generators"] or not 0 <= share <= 1:
                raise ValueError("Invalid generation ownership.")
    for g in data["generators"]:
        if sum(c.get("owned_generation_shares", {}).get(g, 0) for c in profile.values()) > 1 + 1e-9:
            raise ValueError(f"Aggregate ownership exceeds 100% for {g}.")


def capital_recovery_factor(wacc, years):
    return wacc / (1 - (1 + wacc) ** -years) if wacc else 1 / years


def prepare_input(data, profile, *, active_investor=None, investment_nodes=None,
                  node_limit=1000.0, balancing_eps=0.0):
    """Return a model containing only input sets and numerical parameters.

    For a fixed-capacity market, omit active_investor. For an MPEC, include all
    permitted investment locations even when their starting capacity is zero.
    Zero-capacity rival storage is omitted, as in the original formulation.
    balancing_eps > 0 adds price-responsive nodal balancing (see primal_llp.py).
    """
    validate(data, profile)
    if not math.isfinite(balancing_eps) or balancing_eps < 0:
        raise ValueError("The balancing slope must be finite and nonnegative.")
    nodes = data["nodes"]
    investment_nodes = nodes if investment_nodes is None else investment_nodes
    if active_investor is not None:
        if active_investor not in profile:
            raise ValueError(f"Unknown investor {active_investor}.")
        if not investment_nodes or set(investment_nodes) - set(nodes):
            raise ValueError("Choose valid investment nodes.")
        if len(set(investment_nodes)) != len(investment_nodes):
            raise ValueError("Investment nodes must be unique.")
        if not math.isfinite(node_limit) or node_limit <= 0:
            raise ValueError("The nodal investment limit must be positive and finite.")

    generation_capacity = {(r["generator"], r["hour"]): r["capacity_mw"]
                           for r in data["generation_capacity"]}
    demand = {(r["node"], r["hour"]): r["demand_mw"] for r in data["demand_el"]}
    ptdf = {(r["line"], r["node"]): r["ptdf"] for r in data["ptdf"]}
    for g in data["generators"]:
        for t in data["times"]:
            if not math.isfinite(generation_capacity[g, t]) or generation_capacity[g, t] < 0:
                raise ValueError("Generation capacity must be finite and nonnegative.")
    for n in nodes:
        for t in data["times"]:
            if not math.isfinite(demand[n, t]) or demand[n, t] < 0:
                raise ValueError("Demand must be finite and nonnegative.")
    for line in data["lines"]:
        if not math.isfinite(data["line_limit"][line]) or data["line_limit"][line] < 0:
            raise ValueError("Line limits must be finite and nonnegative.")

    power = {(i, n): c["power_mw"].get(n, 0.0) for i, c in profile.items() for n in nodes}
    energy = {(i, n): c["energy_mwh"].get(n, 0.0) for i, c in profile.items() for n in nodes}
    power_max, energy_max = dict(power), dict(energy)
    headroom = {}
    if active_investor is not None:
        for n in nodes:
            headroom[n] = node_limit - sum(power[i, n] for i in profile if i != active_investor)
            if headroom[n] < 0:
                raise ValueError("Rival power exceeds the shared nodal investment limit.")
            power_max[active_investor, n] = headroom[n] if n in investment_nodes else 0.0
            energy_max[active_investor, n] = profile[active_investor]["ratio_max"] * power_max[active_investor, n]
    storage_pairs = [pair for pair in power if power_max[pair] > 0]

    m = pyo.ConcreteModel()
    m._profile = profile
    m.active_investor = active_investor

    # Sets: generators, nodes, investors, lines, and one-hour periods.
    m.nodes = pyo.Set(initialize=nodes, ordered=True, doc="Network nodes")
    m.generators = pyo.Set(initialize=data["generators"], ordered=True)
    m.investors = pyo.Set(initialize=list(profile), ordered=True)
    m.lines = pyo.Set(initialize=data["lines"], ordered=True)
    m.timesteps = pyo.Set(initialize=data["times"], ordered=True, doc="Hours 1,...,T")
    m.soc_timesteps = pyo.Set(initialize=data["soc_times"], ordered=True, doc="States 0,...,T")
    m.storage_pairs = pyo.Set(dimen=2, initialize=storage_pairs, ordered=True,
                              doc="Investor-node pairs with existing or possible storage")
    m.generators_at_node = pyo.Set(m.nodes, initialize=data["generators_at_node"])
    m.storage_at_node = pyo.Set(m.nodes, initialize={n: [i for i, site in storage_pairs if site == n] for n in nodes})

    # Market parameters: EUR/MWh, MW, MWh, and dimensionless PTDF/efficiency.
    m.generation_cost = pyo.Param(m.generators, initialize=data["generation_cost"])
    m.generation_capacity = pyo.Param(m.generators, m.timesteps, initialize=generation_capacity)
    m.demand = pyo.Param(m.nodes, m.timesteps, initialize=demand)
    m.line_limit = pyo.Param(m.lines, initialize=data["line_limit"])
    m.ptdf = pyo.Param(m.lines, m.nodes, initialize=ptdf)
    m.eta = pyo.Param(initialize=data["eta"], doc="Efficiency in each direction")
    m.voll = pyo.Param(initialize=data["voll"], doc="Lost-load cost [EUR/MWh]")
    m.balancing_eps = pyo.Param(initialize=balancing_eps,
                                doc="Nodal balancing slope; 0 is the pure LP [MW per EUR/MWh]")
    generator_node = {g: n for n in nodes for g in data["generators_at_node"][n]}
    m.generator_node = pyo.Param(m.generators, within=pyo.Any, initialize=generator_node)
    m.installed_power = pyo.Param(m.investors, m.nodes, initialize=power)
    m.installed_energy = pyo.Param(m.investors, m.nodes, initialize=energy)
    m.power_max = pyo.Param(m.storage_pairs, initialize={pair: power_max[pair] for pair in storage_pairs},
                          doc="Physical upper bound used for primal Big-M [MW]")
    m.energy_max = pyo.Param(m.storage_pairs, initialize={pair: energy_max[pair] for pair in storage_pairs},
                           doc="Physical upper bound used for primal Big-M [MWh]")
    m.degradation = pyo.Param(m.investors, initialize={i: c["degradation_eur_per_mwh"] for i, c in profile.items()})
    m.ratio_min = pyo.Param(m.investors, initialize={i: c["ratio_min"] for i, c in profile.items()})
    m.ratio_max = pyo.Param(m.investors, initialize={i: c["ratio_max"] for i, c in profile.items()})
    m.owned_share = pyo.Param(m.investors, m.generators, initialize={
        (i, g): c.get("owned_generation_shares", {}).get(g, 0.0)
        for i, c in profile.items() for g in data["generators"]})
    daily_crf = {i: capital_recovery_factor(c["wacc"], c["lifetime_years"]) / 365.25 for i, c in profile.items()}
    m.cost_power_daily = pyo.Param(m.investors, initialize={i: daily_crf[i] * c["cost_power_eur_per_mw"] for i, c in profile.items()})
    m.cost_energy_daily = pyo.Param(m.investors, initialize={i: daily_crf[i] * c["cost_energy_eur_per_mwh"] for i, c in profile.items()})
    if active_investor is not None:
        m.investment_nodes = pyo.Set(initialize=investment_nodes, ordered=True)
        m.headroom = pyo.Param(m.nodes, initialize=headroom)
    return m
