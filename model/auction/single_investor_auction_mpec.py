"""One strategic BESS investor with embedded access-auction and spot followers.

The active investor chooses one access-bid quantity and pay-as-bid price at
every node. Rival bid quantities, prices, and storage durations are exogenous
input vectors. The model embeds two convex lower-level problems:

1. the nodal access-auction LP, using primal feasibility, dual feasibility,
   and strong duality; and
2. the fixed-demand spot-market LP, using primal feasibility, dual
   feasibility/stationarity, and strong duality.

Auction awards are the installed BESS power capacities. The active investor
chooses its awarded energy capacity inside the 2-8 hour envelope. Rival energy
capacity is ``fixed duration * endogenous auction award``, so a change in the
active bid cannot leave a rival with a physically inconsistent MW/MWh pair.

This is an optimistic MPEC whenever the auction allocation or spot-market
prices are non-unique. Ipopt provides a local solution candidate, not a proof
of a global Stackelberg optimum.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import pyomo.environ as pyo

# The auction formulation lives in its own experimental folder while the spot
# formulation and common solver helpers remain in the maintained model tree.
_AUCTION_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _AUCTION_DIR.parent
_PRIMAL_DUAL_DIR = _MODEL_DIR / "Primal and dual problems"
for module_dir in (_MODEL_DIR, _PRIMAL_DUAL_DIR):
    if module_dir.is_dir() and str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))

from nodal_access_auction_dual import build_dual as build_auction_dual
from nodal_access_auction_primal import Bid, awarded_mw, build_primal as build_auction_primal
from config import (
    DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY,
    parse_single_mpec_cli,
)
from primal_market_clearing_model import MarketData, load_market_data, value
from single_investor_mpec import (
    DEFAULT_DEGRADATION_EUR_PER_MWH,
    DEFAULT_FIXED_DEMAND_DUAL_BOUND_EUR_PER_MWH,
    DEFAULT_NODE_LIMIT_MW,
    InvestorConfig,
    build_fixed_demand_primal_model,
    capital_recovery_factor,
    fixed_demand_reference_lambda,
)
from solver_utils import get_ipopt_solver


MODEL_NAME = "One-Leader Two-Follower Access and Spot MPEC"
ACTIVE_INVESTOR_ID = "I1"


@dataclass(frozen=True)
class RivalBid:
    """One fixed rival bid block at one node."""

    investor: str
    node: str
    quantity_mw: float
    price_eur_per_mw_day: float
    duration_hours: float


def demo_rival_bid_vector(
    nodes: Iterable[str], active_id: str = ACTIVE_INVESTOR_ID
) -> list[RivalBid]:
    """Made-up sparse rival vectors for a runnable IEEE-9 demonstration."""

    node_set = set(nodes)
    offers = {
        ("I1", "N3"): (35.0, 13.0, 4.0),
        ("I1", "N8"): (55.0, 22.0, 4.0),
        ("I1", "N9"): (25.0, 9.0, 4.0),
        ("I2", "N3"): (30.0, 15.0, 4.0),
        ("I2", "N8"): (60.0, 25.0, 4.0),
        ("I2", "N9"): (20.0, 8.0, 4.0),
        ("I3", "N3"): (40.0, 12.0, 6.0),
        ("I3", "N8"): (50.0, 18.0, 6.0),
        ("I3", "N9"): (20.0, 6.0, 6.0),
        ("I4", "N3"): (25.0, 10.0, 4.0),
        ("I4", "N8"): (45.0, 20.0, 4.0),
        ("I4", "N9"): (35.0, 14.0, 4.0),
    }
    return [
        RivalBid(investor, node, quantity, price, duration)
        for (investor, node), (quantity, price, duration) in offers.items()
        if node in node_set and investor != active_id
    ]


def load_rival_bid_vector(path: Path) -> list[RivalBid]:
    """Read a record list or compact all-investor nodal-vector JSON object."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "investors" in raw:
        records = []
        for investor, profile in raw["investors"].items():
            quantities = profile["quantity_mw_by_node"]
            prices = profile["price_eur_per_mw_day_by_node"]
            if set(quantities) != set(prices):
                raise ValueError(f"Quantity and price nodes differ for {investor}.")
            for node in quantities:
                records.append(
                    {
                        "investor": investor,
                        "node": node,
                        "quantity_mw": quantities[node],
                        "price_eur_per_mw_day": prices[node],
                        "duration_hours": profile["duration_hours"],
                    }
                )
    else:
        records = raw["rival_bids"] if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError("Rival-bid JSON must be a list or contain a 'rival_bids' list.")
    return [RivalBid(**record) for record in records]


def _generator_nodes(data: MarketData) -> dict[str, list[str]]:
    result = {generator: [] for generator in data.generators}
    for node in data.nodes:
        for generator in data.generators_at_node.get(node, []):
            result.setdefault(generator, []).append(node)
    return result


def _normalize_rival_bids(
    data: MarketData,
    rival_bids: Iterable[RivalBid],
    active_id: str,
    ratio_min: float,
    ratio_max: float,
) -> tuple[list[str], dict[tuple[str, str], float], dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    records = list(rival_bids)
    if not records:
        raise ValueError("At least one rival bid is required.")
    node_set = set(data.nodes)
    seen: set[tuple[str, str]] = set()
    rival_ids: list[str] = []
    for bid in records:
        key = (bid.investor, bid.node)
        if bid.investor == active_id:
            raise ValueError(f"Rival input cannot use active investor id {active_id!r}.")
        if bid.node not in node_set:
            raise ValueError(f"Unknown rival-bid node {bid.node!r}.")
        if key in seen:
            raise ValueError(f"Duplicate rival bid {key}.")
        if bid.quantity_mw < 0.0 or bid.price_eur_per_mw_day < 0.0:
            raise ValueError(f"Negative rival bid quantity or price for {key}.")
        if not ratio_min <= bid.duration_hours <= ratio_max:
            raise ValueError(f"Rival duration for {key} must be in [{ratio_min}, {ratio_max}].")
        seen.add(key)
        if bid.investor not in rival_ids:
            rival_ids.append(bid.investor)

    quantity = {(investor, node): 0.0 for investor in rival_ids for node in data.nodes}
    price = {(investor, node): 0.0 for investor in rival_ids for node in data.nodes}
    duration = {(investor, node): ratio_min for investor in rival_ids for node in data.nodes}
    for bid in records:
        key = (bid.investor, bid.node)
        quantity[key] = float(bid.quantity_mw)
        price[key] = float(bid.price_eur_per_mw_day)
        duration[key] = float(bid.duration_hours)
    return rival_ids, quantity, price, duration


def build_one_leader_two_follower_mpec(
    data: MarketData,
    rival_bids: Iterable[RivalBid],
    *,
    active_investor: InvestorConfig | None = None,
    active_nodes: Iterable[str] | None = None,
    node_limit_mw: float | Mapping[str, float] = DEFAULT_NODE_LIMIT_MW,
    min_bid_price_eur_per_mw_day: float = 0.0,
    max_bid_price_eur_per_mw_day: float = DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY,
    initial_bid_quantity_mw: float | Mapping[str, float] = 10.0,
    initial_bid_price_eur_per_mw_day: float | Mapping[str, float] = 10.0,
    initial_duration_hours: float | Mapping[str, float] = 4.0,
    rival_degradation_eur_per_mwh: float = DEFAULT_DEGRADATION_EUR_PER_MWH,
    dual_bound_scale: float = 10.0,
) -> pyo.ConcreteModel:
    """Build one investor's MPEC against fixed rival bid vectors."""

    inv = active_investor or InvestorConfig(investor_id=ACTIVE_INVESTOR_ID)
    rival_bids = list(rival_bids)
    if not 0.0 <= min_bid_price_eur_per_mw_day <= max_bid_price_eur_per_mw_day:
        raise ValueError("Bid-price bounds must satisfy 0 <= minimum <= maximum.")
    if dual_bound_scale <= 0.0:
        raise ValueError("dual_bound_scale must be positive.")
    if rival_degradation_eur_per_mwh < 0.0:
        raise ValueError("Rival degradation cost must be nonnegative.")
    if isinstance(node_limit_mw, Mapping):
        limits = {node: float(node_limit_mw[node]) for node in data.nodes}
    else:
        limits = {node: float(node_limit_mw) for node in data.nodes}
    if any(limit <= 0.0 for limit in limits.values()):
        raise ValueError("Every nodal auction limit must be positive.")

    def nodal_initial(value_or_map: float | Mapping[str, float], node: str) -> float:
        return float(value_or_map[node] if isinstance(value_or_map, Mapping) else value_or_map)

    initial_quantity = {node: nodal_initial(initial_bid_quantity_mw, node) for node in data.nodes}
    initial_price = {node: nodal_initial(initial_bid_price_eur_per_mw_day, node) for node in data.nodes}
    initial_duration = {node: nodal_initial(initial_duration_hours, node) for node in data.nodes}
    if any(not inv.ratio_min <= duration <= inv.ratio_max for duration in initial_duration.values()):
        raise ValueError("Initial duration is outside the active investor's duration envelope.")

    rivals, rival_quantity, rival_price, rival_duration = _normalize_rival_bids(
        data, rival_bids, inv.investor_id, inv.ratio_min, inv.ratio_max
    )
    permitted_active_nodes = set(data.nodes if active_nodes is None else active_nodes)
    unknown_active_nodes = permitted_active_nodes - set(data.nodes)
    if unknown_active_nodes:
        raise ValueError(f"Unknown active-investor nodes: {sorted(unknown_active_nodes)}")
    if not permitted_active_nodes:
        raise ValueError("At least one active-investor node must be enabled.")
    investors = [inv.investor_id, *rivals]
    degradation = {inv.investor_id: inv.degradation_eur_per_mwh}
    degradation.update({rival: rival_degradation_eur_per_mwh for rival in rivals})
    gen_nodes = _generator_nodes(data)
    last_t = max(data.times)
    eta = data.eta
    dual_bound = dual_bound_scale * max(
        DEFAULT_FIXED_DEMAND_DUAL_BOUND_EUR_PER_MWH,
        float(data.voll),
        max(data.generation_cost.values(), default=0.0),
    )

    model = pyo.ConcreteModel(name=MODEL_NAME)
    model.N = pyo.Set(initialize=data.nodes, ordered=True)
    model.G = pyo.Set(initialize=data.generators, ordered=True)
    model.I = pyo.Set(initialize=investors, ordered=True)
    model.R = pyo.Set(initialize=rivals, ordered=True)
    model.T = pyo.Set(initialize=data.times, ordered=True)
    model.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    model.L = pyo.Set(initialize=data.lines, ordered=True)

    # Upper-level decisions of the active investor.
    model.active_bid_quantity = pyo.Var(
        model.N,
        bounds=lambda m, n: (0.0, limits[n]),
        initialize=lambda m, n: min(max(initial_quantity[n], 0.0), limits[n]),
    )
    model.active_bid_price = pyo.Var(
        model.N,
        bounds=(min_bid_price_eur_per_mw_day, max_bid_price_eur_per_mw_day),
        initialize=lambda m, n: min(
            max(initial_price[n], min_bid_price_eur_per_mw_day), max_bid_price_eur_per_mw_day
        ),
    )
    model.active_energy = pyo.Var(
        model.N,
        bounds=lambda m, n: (0.0, inv.ratio_max * limits[n]),
        initialize=0.0,
    )
    for node in data.nodes:
        if node not in permitted_active_nodes:
            model.active_bid_quantity[node].fix(0.0)
            model.active_bid_price[node].fix(min_bid_price_eur_per_mw_day)
            model.active_energy[node].fix(0.0)

    def bid_quantity(m: pyo.ConcreteModel, investor: str, node: str):
        return m.active_bid_quantity[node] if investor == inv.investor_id else rival_quantity[investor, node]

    def bid_price(m: pyo.ConcreteModel, investor: str, node: str):
        return m.active_bid_price[node] if investor == inv.investor_id else rival_price[investor, node]

    # ------------------------------------------------------------------
    # Follower 1: access auction primal, dual, and strong duality.
    # ------------------------------------------------------------------
    model.award = pyo.Var(model.I, model.N, domain=pyo.NonNegativeReals, initialize=0.0)
    model.auction_capacity = pyo.Constraint(
        model.N,
        rule=lambda m, n: sum(m.award[i, n] for i in m.I) <= limits[n],
    )
    model.auction_bid_limit = pyo.Constraint(
        model.I,
        model.N,
        rule=lambda m, i, n: m.award[i, n] <= bid_quantity(m, i, n),
    )
    model.auction_capacity_dual = pyo.Var(model.N, domain=pyo.NonNegativeReals, initialize=0.0)
    model.auction_quantity_dual = pyo.Var(model.I, model.N, domain=pyo.NonNegativeReals, initialize=0.0)
    model.auction_dual_feasibility = pyo.Constraint(
        model.I,
        model.N,
        rule=lambda m, i, n: m.auction_capacity_dual[n] + m.auction_quantity_dual[i, n]
        >= bid_price(m, i, n),
    )
    model.auction_primal_value = pyo.Expression(
        expr=sum(bid_price(model, i, n) * model.award[i, n] for i in model.I for n in model.N)
    )
    model.auction_dual_value = pyo.Expression(
        expr=sum(limits[n] * model.auction_capacity_dual[n] for n in model.N)
        + sum(
            bid_quantity(model, i, n) * model.auction_quantity_dual[i, n]
            for i in model.I
            for n in model.N
        )
    )
    model.auction_strong_duality = pyo.Constraint(
        expr=model.auction_primal_value == model.auction_dual_value
    )
    # Eliminate structurally absent rival blocks from the NLP. If both their
    # quantity and price are zero, award=0 and quantity-dual=0 always admit an
    # optimal auction primal-dual pair.
    for rival in rivals:
        for node in data.nodes:
            if rival_quantity[rival, node] == 0.0 and rival_price[rival, node] == 0.0:
                model.award[rival, node].fix(0.0)
                model.auction_quantity_dual[rival, node].fix(0.0)

    model.active_energy_min = pyo.Constraint(
        model.N,
        rule=lambda m, n: m.active_energy[n] >= inv.ratio_min * m.award[inv.investor_id, n],
    )
    model.active_energy_max = pyo.Constraint(
        model.N,
        rule=lambda m, n: m.active_energy[n] <= inv.ratio_max * m.award[inv.investor_id, n],
    )

    def power_capacity(m: pyo.ConcreteModel, investor: str, node: str):
        return m.award[investor, node]

    def energy_capacity(m: pyo.ConcreteModel, investor: str, node: str):
        if investor == inv.investor_id:
            return m.active_energy[node]
        return rival_duration[investor, node] * m.award[investor, node]

    # ------------------------------------------------------------------
    # Follower 2: fixed-demand spot-market primal feasibility.
    # ------------------------------------------------------------------
    model.P_gen = pyo.Var(model.G, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_charge = pyo.Var(model.I, model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_discharge = pyo.Var(model.I, model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.SOC = pyo.Var(model.I, model.N, model.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    model.NetInjection = pyo.Var(model.N, model.T, domain=pyo.Reals, initialize=0.0)
    for rival in rivals:
        for node in data.nodes:
            if rival_quantity[rival, node] == 0.0:
                for time in data.times:
                    model.P_charge[rival, node, time].fix(0.0)
                    model.P_discharge[rival, node, time].fix(0.0)
                for tau in data.soc_times:
                    model.SOC[rival, node, tau].fix(0.0)

    model.nodal_balance = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: sum(m.P_gen[g, t] for g in data.generators_at_node.get(n, []))
        + sum(m.P_discharge[i, n, t] - m.P_charge[i, n, t] for i in m.I)
        - data.demand_el[n, t]
        == m.NetInjection[n, t],
    )
    model.system_balance = pyo.Constraint(
        model.T,
        rule=lambda m, t: sum(m.NetInjection[n, t] for n in m.N) == 0.0,
    )
    model.generation_capacity_bound = pyo.Constraint(
        model.G,
        model.T,
        rule=lambda m, g, t: m.P_gen[g, t] <= data.generation_capacity[g, t],
    )
    model.line_upper_bound = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, line, t: sum(data.ptdf[line, n] * m.NetInjection[n, t] for n in m.N)
        <= data.line_limit[line],
    )
    model.line_lower_bound = pyo.Constraint(
        model.L,
        model.T,
        rule=lambda m, line, t: sum(data.ptdf[line, n] * m.NetInjection[n, t] for n in m.N)
        >= -data.line_limit[line],
    )
    model.charge_power_bound = pyo.Constraint(
        model.I,
        model.N,
        model.T,
        rule=lambda m, i, n, t: m.P_charge[i, n, t] <= power_capacity(m, i, n),
    )
    model.discharge_power_bound = pyo.Constraint(
        model.I,
        model.N,
        model.T,
        rule=lambda m, i, n, t: m.P_discharge[i, n, t] <= power_capacity(m, i, n),
    )
    model.soc_transition = pyo.Constraint(
        model.I,
        model.N,
        model.T,
        rule=lambda m, i, n, t: m.SOC[i, n, t]
        == m.SOC[i, n, t - 1] + eta * m.P_charge[i, n, t] - m.P_discharge[i, n, t] / eta,
    )
    model.soc_capacity_bound = pyo.Constraint(
        model.I,
        model.N,
        model.T_SOC,
        rule=lambda m, i, n, tau: m.SOC[i, n, tau] <= energy_capacity(m, i, n),
    )
    model.soc_periodicity = pyo.Constraint(
        model.I,
        model.N,
        rule=lambda m, i, n: m.SOC[i, n, 0] == m.SOC[i, n, last_t],
    )

    # Spot-market dual variables. Bounds are numerical safeguards and are
    # checked after solve; a binding bound invalidates the economic result.
    model.lam = pyo.Var(model.N, model.T, bounds=(-dual_bound, dual_bound), initialize=80.0)
    model.lam_sys = pyo.Var(model.T, bounds=(-dual_bound, dual_bound), initialize=80.0)
    model.nu_gen = pyo.Var(model.G, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.mu_up = pyo.Var(model.L, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.mu_dn = pyo.Var(model.L, model.T, bounds=(0.0, dual_bound), initialize=0.0)
    model.rho_ch = pyo.Var(model.I, model.N, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.sig_dis = pyo.Var(model.I, model.N, model.T, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.gam = pyo.Var(model.I, model.N, model.T, bounds=(-dual_bound, dual_bound), initialize=0.0)
    model.del_soc = pyo.Var(model.I, model.N, model.T_SOC, bounds=(-dual_bound, 0.0), initialize=0.0)
    model.rho_per = pyo.Var(model.I, model.N, bounds=(-dual_bound, dual_bound), initialize=0.0)

    # Spot-market dual feasibility/stationarity.
    model.gen_stationarity = pyo.Constraint(
        model.G,
        model.T,
        rule=lambda m, g, t: sum(m.lam[n, t] for n in gen_nodes.get(g, [])) + m.nu_gen[g, t]
        <= data.generation_cost[g],
    )
    model.charge_stationarity = pyo.Constraint(
        model.I,
        model.N,
        model.T,
        rule=lambda m, i, n, t: -m.lam[n, t] + m.rho_ch[i, n, t] - eta * m.gam[i, n, t]
        <= 0.5 * degradation[i],
    )
    model.discharge_stationarity = pyo.Constraint(
        model.I,
        model.N,
        model.T,
        rule=lambda m, i, n, t: m.lam[n, t] + m.sig_dis[i, n, t] + m.gam[i, n, t] / eta
        <= 0.5 * degradation[i],
    )
    model.netinjection_stationarity = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: -m.lam[n, t]
        + m.lam_sys[t]
        + sum(data.ptdf[line, n] * (m.mu_up[line, t] + m.mu_dn[line, t]) for line in m.L)
        == 0.0,
    )

    def soc_stationarity(m: pyo.ConcreteModel, investor: str, node: str, tau: int):
        expression = m.del_soc[investor, node, tau]
        if tau in m.T:
            expression += m.gam[investor, node, tau]
        if (tau + 1) in m.T:
            expression -= m.gam[investor, node, tau + 1]
        if tau == 0:
            expression += m.rho_per[investor, node]
        if tau == last_t:
            expression -= m.rho_per[investor, node]
        return expression <= 0.0

    model.soc_stationarity = pyo.Constraint(model.I, model.N, model.T_SOC, rule=soc_stationarity)

    model.spot_primal_value = pyo.Expression(
        expr=sum(data.generation_cost[g] * model.P_gen[g, t] for g in model.G for t in model.T)
        + sum(
            0.5 * degradation[i] * (model.P_charge[i, n, t] + model.P_discharge[i, n, t])
            for i in model.I
            for n in model.N
            for t in model.T
        )
    )
    model.spot_dual_value = pyo.Expression(
        expr=sum(data.demand_el[n, t] * model.lam[n, t] for n in model.N for t in model.T)
        + sum(data.generation_capacity[g, t] * model.nu_gen[g, t] for g in model.G for t in model.T)
        + sum(
            data.line_limit[line] * (model.mu_up[line, t] - model.mu_dn[line, t])
            for line in model.L
            for t in model.T
        )
        + sum(
            power_capacity(model, i, n) * (model.rho_ch[i, n, t] + model.sig_dis[i, n, t])
            for i in model.I
            for n in model.N
            for t in model.T
        )
        + sum(
            energy_capacity(model, i, n) * model.del_soc[i, n, tau]
            for i in model.I
            for n in model.N
            for tau in model.T_SOC
        )
    )
    model.spot_strong_duality = pyo.Constraint(
        expr=model.spot_primal_value == model.spot_dual_value
    )

    # ------------------------------------------------------------------
    # Active investor objective.
    # ------------------------------------------------------------------
    crf_daily = capital_recovery_factor(inv.wacc, inv.lifetime_years) / 365.0
    gen_node = {g: (gen_nodes.get(g) or [None])[0] for g in data.generators}
    model.active_spot_revenue = pyo.Expression(
        expr=sum(
            model.lam[n, t]
            * (model.P_discharge[inv.investor_id, n, t] - model.P_charge[inv.investor_id, n, t])
            for n in model.N
            for t in model.T
        )
    )
    model.active_generation_rent = pyo.Expression(
        expr=sum(
            share * (model.lam[gen_node[g], t] - data.generation_cost[g]) * model.P_gen[g, t]
            for g, share in inv.owned_generation_shares.items()
            for t in model.T
            if share != 0.0 and gen_node.get(g) is not None
        )
    )
    model.active_degradation_cost = pyo.Expression(
        expr=0.5
        * inv.degradation_eur_per_mwh
        * sum(
            model.P_charge[inv.investor_id, n, t] + model.P_discharge[inv.investor_id, n, t]
            for n in model.N
            for t in model.T
        )
    )
    model.active_capex_daily = pyo.Expression(
        expr=crf_daily
        * sum(
            inv.cost_power_eur_per_mw * model.award[inv.investor_id, n]
            + inv.cost_energy_eur_per_mwh * model.active_energy[n]
            for n in model.N
        )
    )
    model.active_access_payment = pyo.Expression(
        expr=sum(model.active_bid_price[n] * model.award[inv.investor_id, n] for n in model.N)
    )
    model.active_profit = pyo.Expression(
        expr=model.active_spot_revenue
        + model.active_generation_rent
        - model.active_degradation_cost
        - model.active_capex_daily
        - model.active_access_payment
    )
    model.objective = pyo.Objective(expr=model.active_profit, sense=pyo.maximize)

    model._market_data = data
    model._active_investor = inv
    model._active_id = inv.investor_id
    model._rival_bids = rival_bids
    model._rival_quantity = rival_quantity
    model._rival_price = rival_price
    model._rival_duration = rival_duration
    model._node_limits = limits
    model._degradation = degradation
    model._dual_bound = dual_bound
    model._initial_duration_hours = initial_duration
    model._permitted_active_nodes = sorted(permitted_active_nodes)
    return model


def _current_fixed_bids(model: pyo.ConcreteModel) -> list[Bid]:
    bids: list[Bid] = []
    active_id = model._active_id
    for node in model.N:
        bids.append(
            Bid(
                active_id,
                node,
                max(0.0, value(model.active_bid_quantity[node])),
                max(0.0, value(model.active_bid_price[node])),
            )
        )
    for rival in model.R:
        for node in model.N:
            bids.append(
                Bid(
                    rival,
                    node,
                    model._rival_quantity[rival, node],
                    model._rival_price[rival, node],
                )
            )
    return bids


def initialize_from_independent_followers(model: pyo.ConcreteModel) -> None:
    """Warm-start awards and spot dispatch from separate follower solves."""

    bids = _current_fixed_bids(model)
    primal = build_auction_primal(bids, model._node_limits)
    dual = build_auction_dual(bids, model._node_limits)
    solver = get_ipopt_solver({"max_cpu_time": 60.0})
    primal_result = solver.solve(primal, tee=False)
    dual_result = solver.solve(dual, tee=False)
    if primal_result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return

    awards = awarded_mw(primal)
    for investor in model.I:
        for node in model.N:
            model.award[investor, node].set_value(awards.get((investor, node), 0.0))
    if dual_result.solver.termination_condition == pyo.TerminationCondition.optimal:
        for node in model.N:
            model.auction_capacity_dual[node].set_value(max(0.0, value(dual.capacity_dual[node])))
        bid_index = {(dual.bid[k].investor, dual.bid[k].node): k for k in dual.K}
        for investor in model.I:
            for node in model.N:
                model.auction_quantity_dual[investor, node].set_value(
                    max(0.0, value(dual.quantity_dual[bid_index[investor, node]]))
                )

    active_id = model._active_id
    for node in model.N:
        model.active_energy[node].set_value(
            model._initial_duration_hours[node] * value(model.award[active_id, node])
        )

    x_power = {(i, n): value(model.award[i, n]) for i in model.I for n in model.N}
    x_energy = {
        (i, n): (
            value(model.active_energy[n])
            if i == active_id
            else model._rival_duration[i, n] * value(model.award[i, n])
        )
        for i in model.I
        for n in model.N
    }
    spot_data = replace(
        model._market_data,
        storage_units=list(model.I),
        x_power=x_power,
        x_energy=x_energy,
    )
    reference = build_fixed_demand_primal_model(
        spot_data,
        storage_degradation_eur_per_mwh=model._degradation,
        dispatch_regularization_eur_per_mw2h=0.0,
    )
    result = solver.solve(reference, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return
    prices = fixed_demand_reference_lambda(reference)
    gen_nodes = _generator_nodes(spot_data)
    for g in model.G:
        for t in model.T:
            model.P_gen[g, t].set_value(max(0.0, value(reference.P_gen[g, t])))
            marginal_price = sum(prices[n, t] for n in gen_nodes.get(g, []))
            model.nu_gen[g, t].set_value(min(0.0, spot_data.generation_cost[g] - marginal_price))
    for n in model.N:
        for t in model.T:
            model.NetInjection[n, t].set_value(value(reference.NetInjection[n, t]))
            model.lam[n, t].set_value(prices[n, t])
    for t in model.T:
        model.lam_sys[t].set_value(sum(prices[n, t] for n in model.N) / max(1, len(list(model.N))))
    for i in model.I:
        for n in model.N:
            for t in model.T:
                model.P_charge[i, n, t].set_value(max(0.0, value(reference.P_charge[i, n, t])))
                model.P_discharge[i, n, t].set_value(max(0.0, value(reference.P_discharge[i, n, t])))
                model.gam[i, n, t].set_value(-prices[n, t])
            for tau in model.T_SOC:
                model.SOC[i, n, tau].set_value(max(0.0, value(reference.SOC[i, n, tau])))


def solve_mpec(model: pyo.ConcreteModel, max_cpu_time: float, tee: bool = False) -> str:
    result = get_ipopt_solver({"max_cpu_time": max_cpu_time}).solve(model, tee=tee)
    return str(result.solver.termination_condition)


def summarize(model: pyo.ConcreteModel, termination: str) -> dict[str, object]:
    active_id = model._active_id
    auction_gap = value(model.auction_primal_value) - value(model.auction_dual_value)
    spot_gap = value(model.spot_primal_value) - value(model.spot_dual_value)
    nodes: dict[str, dict[str, object]] = {}
    for node in model.N:
        award = max(0.0, value(model.award[active_id, node]))
        energy = max(0.0, value(model.active_energy[node]))
        nodes[node] = {
            "active_bid_quantity_mw": value(model.active_bid_quantity[node]),
            "active_bid_price_eur_per_mw_day": value(model.active_bid_price[node]),
            "active_award_mw": award,
            "active_energy_mwh": energy,
            "active_duration_hours": energy / award if award > 1e-7 else None,
            "total_award_mw": sum(max(0.0, value(model.award[i, node])) for i in model.I),
            "rival_awards_mw": {
                rival: max(0.0, value(model.award[rival, node])) for rival in model.R
            },
        }

    # Independent reclear at the final bids diagnoses optimistic/tied auction
    # allocation. It does not alter the MPEC result.
    independent = build_auction_primal(_current_fixed_bids(model), model._node_limits)
    independent_term = str(
        get_ipopt_solver({"max_cpu_time": 60.0}).solve(independent, tee=False).solver.termination_condition
    )
    reclear_awards = awarded_mw(independent) if independent_term == "optimal" else {}
    max_award_difference = max(
        (
            abs(value(model.award[i, n]) - reclear_awards.get((i, n), 0.0))
            for i in model.I
            for n in model.N
        ),
        default=0.0,
    )

    dual_bound = model._dual_bound
    dual_values = [abs(value(model.lam[n, t])) for n in model.N for t in model.T]
    dual_bound_fraction = max(dual_values, default=0.0) / dual_bound
    return {
        "termination": termination,
        "interpretation": "local optimistic MPEC candidate",
        "active_investor": active_id,
        "permitted_active_nodes": model._permitted_active_nodes,
        "profit_eur_per_day": value(model.active_profit),
        "spot_revenue_eur_per_day": value(model.active_spot_revenue),
        "generation_rent_eur_per_day": value(model.active_generation_rent),
        "degradation_cost_eur_per_day": value(model.active_degradation_cost),
        "capex_eur_per_day": value(model.active_capex_daily),
        "access_payment_eur_per_day": value(model.active_access_payment),
        "auction_primal_value_eur_per_day": value(model.auction_primal_value),
        "auction_dual_value_eur_per_day": value(model.auction_dual_value),
        "auction_strong_duality_gap": auction_gap,
        "spot_primal_value_eur_per_day": value(model.spot_primal_value),
        "spot_dual_value_eur_per_day": value(model.spot_dual_value),
        "spot_strong_duality_gap": spot_gap,
        "independent_auction_termination": independent_term,
        "max_embedded_vs_reclear_award_difference_mw": max_award_difference,
        "max_lmp_fraction_of_numerical_bound": dual_bound_fraction,
        "nodes": nodes,
    }


def print_summary(summary: Mapping[str, object]) -> None:
    print(f"Termination: {summary['termination']}")
    print(f"Interpretation: {summary['interpretation']}")
    print(f"Active profit: {summary['profit_eur_per_day']:,.2f} EUR/day")
    print(
        "  revenue {0:,.2f} - degradation {1:,.2f} - CAPEX {2:,.2f} - access {3:,.2f}".format(
            summary["spot_revenue_eur_per_day"],
            summary["degradation_cost_eur_per_day"],
            summary["capex_eur_per_day"],
            summary["access_payment_eur_per_day"],
        )
    )
    print(f"Auction strong-duality gap: {summary['auction_strong_duality_gap']:.3e}")
    print(f"Spot strong-duality gap: {summary['spot_strong_duality_gap']:.3e}")
    print(
        "Embedded vs independent auction max award difference: "
        f"{summary['max_embedded_vs_reclear_award_difference_mw']:.3e} MW"
    )
    print("Active bids and awards:")
    for node, row in summary["nodes"].items():
        if row["active_bid_quantity_mw"] > 1e-5 or row["active_award_mw"] > 1e-5:
            duration = row["active_duration_hours"]
            duration_text = "-" if duration is None else f"{duration:.2f} h"
            print(
                f"  {node}: bid {row['active_bid_quantity_mw']:.3f} MW @ "
                f"{row['active_bid_price_eur_per_mw_day']:.3f} EUR/MW/day; "
                f"award {row['active_award_mw']:.3f} MW, "
                f"energy {row['active_energy_mwh']:.3f} MWh ({duration_text})"
            )


def main() -> int:
    cfg = parse_single_mpec_cli()
    data = load_market_data(cfg.data_path)
    if cfg.active_node is not None and cfg.active_node not in data.nodes:
        raise SystemExit(f"Unknown active node {cfg.active_node!r}; choose from {list(data.nodes)}")
    if cfg.rival_bids_path:
        all_input_bids = load_rival_bid_vector(cfg.rival_bids_path)
        rivals = [bid for bid in all_input_bids if bid.investor != cfg.active_investor]
    else:
        rivals = demo_rival_bid_vector(data.nodes, cfg.active_investor)
    # Import locally so this best-response module remains safe to import from a
    # future Gauss-Seidel driver without creating a module-import cycle.
    from epec_diagonalization import four_investor_portfolio_profiles

    profiles = {profile.investor_id: profile for profile in four_investor_portfolio_profiles(data)}
    active = profiles[cfg.active_investor]
    model = build_one_leader_two_follower_mpec(
        data,
        rivals,
        active_investor=active,
        active_nodes=None if cfg.active_node is None else [cfg.active_node],
        node_limit_mw=cfg.node_limit_mw,
        min_bid_price_eur_per_mw_day=cfg.min_bid_price_eur_per_mw_day,
        max_bid_price_eur_per_mw_day=cfg.max_bid_price_eur_per_mw_day,
        initial_bid_quantity_mw=cfg.initial_bid_quantity_mw,
        initial_bid_price_eur_per_mw_day=cfg.initial_bid_price_eur_per_mw_day,
        initial_duration_hours=cfg.initial_duration_hours,
        dual_bound_scale=cfg.dual_bound_scale,
    )
    initialize_from_independent_followers(model)
    termination = solve_mpec(model, cfg.max_cpu_time, tee=cfg.tee)
    summary = summarize(model, termination)
    print_summary(summary)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        **summary,
        "data_path": str(cfg.data_path),
        "active_node_argument": cfg.active_node,
        "rival_bids": [asdict(bid) for bid in rivals],
    }
    cfg.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {cfg.output_path}")
    return 0 if termination == "optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
