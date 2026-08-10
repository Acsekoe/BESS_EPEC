"""One-leader, two-follower MPEC for strategic nodal BESS access bids.

The leader chooses continuous access bid quantities and prices plus its energy
capacity. Follower 1 is the nodal access auction. Follower 2 is the
fixed-demand spot-market LP. Each follower is embedded only through primal
feasibility, dual feasibility/stationarity, and strong duality.

Under the maintained ``uniform`` payment rule the auction objective carries a
small strictly concave quadratic regularization, so the allocation is unique,
exact price ties split pro rata, and every award pays the nodal clearing price
(the auction capacity dual) instead of its own bid. The legacy ``pay_as_bid``
rule keeps the linear auction; an optional external bid-price enumeration then
fixes the active raw bid to exact grid ticks with a smaller deterministic merit
offset ordering equal raw bids. There is no dispatch regularizer, load
shedding, objective penalty, or auxiliary outside-option problem.
Electricity-price nonuniqueness retains the optimistic MPEC convention.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import pyomo.environ as pyo

_AUCTION_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _AUCTION_DIR.parent
_PRIMAL_DUAL_DIR = _MODEL_DIR / "Primal and dual problems"
for module_dir in (_MODEL_DIR, _PRIMAL_DUAL_DIR):
    if module_dir.is_dir() and str(module_dir) not in sys.path:
        sys.path.append(str(module_dir))

from config import (
    DEFAULT_AUCTION_QUADRATIC_EPSILON_EUR_PER_MW2_DAY,
    DEFAULT_CAPACITY_CLEANUP_TOL_MW,
    DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY,
    DEFAULT_NODE_LIMIT_MW,
    DEFAULT_PAYMENT_RULE,
    DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    DEFAULT_SOLVER_TOL,
    parse_single_mpec_cli,
)
from nodal_access_auction_dual import build_dual as build_auction_dual
from nodal_access_auction_primal import Bid, awarded_mw, build_primal as build_auction_primal
from nodal_access_auction_primal import priority_offsets, uniform_clearing_prices
from primal_market_clearing_model import MarketData, load_market_data, value
from single_investor_mpec import (
    DEFAULT_DEGRADATION_EUR_PER_MWH,
    InvestorConfig,
    build_fixed_demand_primal_model,
    capital_recovery_factor,
    fixed_demand_reference_lambda,
)
from solver_utils import get_ipopt_solver


MODEL_NAME = "Clean One-Leader Two-Follower Access Auction MPEC"
ACTIVE_INVESTOR_ID = "I1"
SPARSE_GENERATION_CAPACITY_TOL = 1.0e-8
TICK_AWARD_RECLEAR_TOLERANCE_MW = 1.0e-3
TICK_AUCTION_GAP_TOLERANCE = 1.0e-5
TICK_SPOT_GAP_TOLERANCE = 1.0e-2


@dataclass(frozen=True)
class RivalBid:
    investor: str
    node: str
    quantity_mw: float
    price_eur_per_mw_day: float
    duration_hours: float


def demo_rival_bid_vector(
    nodes: Iterable[str], active_id: str = ACTIVE_INVESTOR_ID
) -> list[RivalBid]:
    """Sparse runnable demonstration used only when no bid file is supplied."""
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
    """Read either record-list or compact all-investor bid JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "investors" in raw:
        records: list[dict[str, object]] = []
        for investor, profile in raw["investors"].items():
            quantities = profile["quantity_mw_by_node"]
            prices = profile["price_eur_per_mw_day_by_node"]
            if set(quantities) != set(prices):
                raise ValueError(f"Quantity and price nodes differ for {investor}.")
            duration_input = profile["duration_hours"]
            for node in quantities:
                duration = duration_input[node] if isinstance(duration_input, dict) else duration_input
                records.append(
                    {
                        "investor": investor,
                        "node": node,
                        "quantity_mw": quantities[node],
                        "price_eur_per_mw_day": prices[node],
                        "duration_hours": duration,
                    }
                )
    else:
        records = raw["rival_bids"] if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError("Bid JSON must be a list or contain an 'investors'/'rival_bids' block.")
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
    cleanup_tol_mw: float,
) -> tuple[
    list[str],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
]:
    records = list(rival_bids)
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

    quantity = {(i, n): 0.0 for i in rival_ids for n in data.nodes}
    price = {(i, n): 0.0 for i in rival_ids for n in data.nodes}
    duration = {(i, n): ratio_min for i in rival_ids for n in data.nodes}
    for bid in records:
        key = (bid.investor, bid.node)
        clean_quantity = 0.0 if bid.quantity_mw <= cleanup_tol_mw else float(bid.quantity_mw)
        quantity[key] = clean_quantity
        price[key] = 0.0 if clean_quantity == 0.0 else float(bid.price_eur_per_mw_day)
        duration[key] = float(bid.duration_hours)
    return rival_ids, quantity, price, duration


def build_one_leader_two_follower_mpec(
    data: MarketData,
    rival_bids: Iterable[RivalBid],
    *,
    active_investor: InvestorConfig | None = None,
    active_nodes: Iterable[str] | None = None,
    node_limit_mw: float | Mapping[str, float] = DEFAULT_NODE_LIMIT_MW,
    max_bid_price_eur_per_mw_day: float = DEFAULT_MAX_BID_PRICE_EUR_PER_MW_DAY,
    initial_bid_quantity_mw: float | Mapping[str, float] = 10.0,
    initial_bid_price_eur_per_mw_day: float | Mapping[str, float] = 10.0,
    initial_duration_hours: float | Mapping[str, float] = 4.0,
    rival_degradation_eur_per_mwh: float = DEFAULT_DEGRADATION_EUR_PER_MWH,
    price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    dual_bound_eur_per_mwh: float = DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    capacity_cleanup_tol_mw: float = DEFAULT_CAPACITY_CLEANUP_TOL_MW,
    tie_break_epsilon_eur_per_mw_day: float = 0.0,
    payment_rule: str = DEFAULT_PAYMENT_RULE,
    auction_quadratic_epsilon_eur_per_mw2_day: float = (
        DEFAULT_AUCTION_QUADRATIC_EPSILON_EUR_PER_MW2_DAY
    ),
) -> pyo.ConcreteModel:
    """Build the clean continuous best-response MPEC against fixed rival bids."""
    inv = active_investor or InvestorConfig(investor_id=ACTIVE_INVESTOR_ID)
    if max_bid_price_eur_per_mw_day <= 0.0:
        raise ValueError("Maximum bid price must be positive.")
    if price_bound_eur_per_mwh <= 0.0 or dual_bound_eur_per_mwh <= 0.0:
        raise ValueError("Price and internal-dual bounds must be positive.")
    if capacity_cleanup_tol_mw < 0.0:
        raise ValueError("Capacity cleanup tolerance must be nonnegative.")
    if tie_break_epsilon_eur_per_mw_day < 0.0:
        raise ValueError("Tie-break epsilon must be nonnegative.")
    if rival_degradation_eur_per_mwh < 0.0:
        raise ValueError("Rival degradation cost must be nonnegative.")
    if payment_rule not in {"uniform", "pay_as_bid"}:
        raise ValueError("Payment rule must be 'uniform' or 'pay_as_bid'.")
    auction_epsilon = float(auction_quadratic_epsilon_eur_per_mw2_day)
    if payment_rule == "uniform":
        if auction_epsilon <= 0.0:
            raise ValueError(
                "The uniform payment rule requires a positive quadratic auction epsilon."
            )
        if tie_break_epsilon_eur_per_mw_day != 0.0:
            raise ValueError(
                "The uniform payment rule uses pro-rata quadratic tie splitting; "
                "merit offsets are a pay-as-bid tick device."
            )
    elif auction_epsilon != 0.0:
        raise ValueError("The pay-as-bid rule keeps the legacy linear auction.")

    limits = (
        {node: float(node_limit_mw[node]) for node in data.nodes}
        if isinstance(node_limit_mw, Mapping)
        else {node: float(node_limit_mw) for node in data.nodes}
    )
    if any(limit <= 0.0 for limit in limits.values()):
        raise ValueError("Every nodal auction limit must be positive.")

    def nodal_initial(value_or_map: float | Mapping[str, float], node: str) -> float:
        return float(value_or_map[node] if isinstance(value_or_map, Mapping) else value_or_map)

    initial_quantity = {n: nodal_initial(initial_bid_quantity_mw, n) for n in data.nodes}
    initial_price = {n: nodal_initial(initial_bid_price_eur_per_mw_day, n) for n in data.nodes}
    initial_duration = {n: nodal_initial(initial_duration_hours, n) for n in data.nodes}
    if any(not inv.ratio_min <= hours <= inv.ratio_max for hours in initial_duration.values()):
        raise ValueError("Initial duration is outside the active investor duration envelope.")

    rivals, rival_quantity, rival_price, rival_duration = _normalize_rival_bids(
        data,
        rival_bids,
        inv.investor_id,
        inv.ratio_min,
        inv.ratio_max,
        capacity_cleanup_tol_mw,
    )
    permitted_active_nodes = set(data.nodes if active_nodes is None else active_nodes)
    unknown_active_nodes = permitted_active_nodes - set(data.nodes)
    if unknown_active_nodes:
        raise ValueError(f"Unknown active-investor nodes: {sorted(unknown_active_nodes)}")
    if not permitted_active_nodes:
        raise ValueError("At least one active-investor node must be enabled.")

    active_pairs = [(inv.investor_id, node) for node in data.nodes]
    rival_pairs = [
        (rival, node)
        for rival in rivals
        for node in data.nodes
        if rival_quantity[rival, node] > capacity_cleanup_tol_mw
    ]
    storage_pairs = active_pairs + rival_pairs
    storage_pairs_at_node = {
        node: [(investor, n) for investor, n in storage_pairs if n == node]
        for node in data.nodes
    }
    positive_generator_hours = [
        (generator, time)
        for generator in data.generators
        for time in data.times
        if data.generation_capacity[generator, time] > SPARSE_GENERATION_CAPACITY_TOL
    ]
    generator_hours_at_node_time = {
        (node, time): [
            generator
            for generator in data.generators_at_node.get(node, [])
            if (generator, time) in positive_generator_hours
        ]
        for node in data.nodes
        for time in data.times
    }
    gen_nodes = _generator_nodes(data)
    degradation = {inv.investor_id: inv.degradation_eur_per_mwh}
    degradation.update({rival: rival_degradation_eur_per_mwh for rival in rivals})
    priority_offset = priority_offsets(
        [inv.investor_id, *rivals], tie_break_epsilon_eur_per_mw_day
    )
    last_t = max(data.times)
    eta = data.eta

    model = pyo.ConcreteModel(name=MODEL_NAME)
    model.N = pyo.Set(initialize=data.nodes, ordered=True)
    model.R = pyo.Set(initialize=rivals, ordered=True)
    model.A = pyo.Set(dimen=2, initialize=storage_pairs, ordered=True)
    model.IN = pyo.Set(dimen=2, initialize=storage_pairs, ordered=True)
    model.GT = pyo.Set(dimen=2, initialize=positive_generator_hours, ordered=True)
    model.T = pyo.Set(initialize=data.times, ordered=True)
    model.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    model.L = pyo.Set(initialize=data.lines, ordered=True)

    model.active_bid_quantity = pyo.Var(
        model.N,
        bounds=lambda m, n: (0.0, limits[n]),
        initialize=lambda m, n: min(max(initial_quantity[n], 0.0), limits[n]),
    )
    model.active_bid_price = pyo.Var(
        model.N,
        bounds=(0.0, max_bid_price_eur_per_mw_day),
        initialize=lambda m, n: min(max(initial_price[n], 0.0), max_bid_price_eur_per_mw_day),
    )
    model.active_energy = pyo.Var(
        model.N,
        bounds=lambda m, n: (0.0, inv.ratio_max * limits[n]),
        initialize=lambda m, n: min(
            inv.ratio_max * limits[n],
            max(0.0, initial_duration[n] * initial_quantity[n]),
        ),
    )
    for node in data.nodes:
        if node not in permitted_active_nodes:
            model.active_bid_quantity[node].fix(0.0)
            model.active_bid_price[node].fix(0.0)
            model.active_energy[node].fix(0.0)

    def bid_quantity(m: pyo.ConcreteModel, investor: str, node: str):
        return m.active_bid_quantity[node] if investor == inv.investor_id else rival_quantity[investor, node]

    def bid_price(m: pyo.ConcreteModel, investor: str, node: str):
        return m.active_bid_price[node] if investor == inv.investor_id else rival_price[investor, node]

    def effective_bid_price(m: pyo.ConcreteModel, investor: str, node: str):
        return bid_price(m, investor, node) + priority_offset[investor]

    # Follower 1: auction primal, dual, and strong duality. With a positive
    # quadratic epsilon this is the exact KKT system of the concave auction QP:
    # equality of the primal objective and the Lagrangian dual value enforces
    # every complementarity condition. Under the uniform payment rule bids are
    # capped at max_bid_price, so the auction duals never need a wider bound.
    model.award = pyo.Var(model.A, domain=pyo.NonNegativeReals, initialize=0.0)
    model.auction_capacity = pyo.Constraint(
        model.N,
        rule=lambda m, n: sum(m.award[i, node] for i, node in storage_pairs_at_node[n]) <= limits[n],
    )
    model.auction_bid_limit = pyo.Constraint(
        model.A,
        rule=lambda m, i, n: m.award[i, n] <= bid_quantity(m, i, n),
    )
    auction_dual_bound = (
        max_bid_price_eur_per_mw_day if payment_rule == "uniform" else dual_bound_eur_per_mwh
    )
    model.auction_capacity_dual = pyo.Var(
        model.N, bounds=(0.0, auction_dual_bound), initialize=0.0
    )
    model.auction_quantity_dual = pyo.Var(
        model.A, bounds=(0.0, auction_dual_bound), initialize=0.0
    )
    model.auction_dual_feasibility = pyo.Constraint(
        model.A,
        rule=lambda m, i, n: m.auction_capacity_dual[n] + m.auction_quantity_dual[i, n]
        >= effective_bid_price(m, i, n) - auction_epsilon * m.award[i, n],
    )
    model.auction_regularization = pyo.Expression(
        expr=0.5 * auction_epsilon * sum(model.award[i, n] ** 2 for i, n in model.A)
    )
    model.auction_primal_value = pyo.Expression(
        expr=sum(effective_bid_price(model, i, n) * model.award[i, n] for i, n in model.A)
        - model.auction_regularization
    )
    model.auction_dual_value = pyo.Expression(
        expr=sum(limits[n] * model.auction_capacity_dual[n] for n in model.N)
        + sum(
            bid_quantity(model, i, n) * model.auction_quantity_dual[i, n]
            for i, n in model.A
        )
        + model.auction_regularization
    )
    model.auction_strong_duality = pyo.Constraint(
        expr=model.auction_primal_value == model.auction_dual_value
    )

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
        return (
            m.active_energy[node]
            if investor == inv.investor_id
            else rival_duration[investor, node] * m.award[investor, node]
        )

    # Follower 2: fixed-demand spot-market primal feasibility.
    model.P_gen = pyo.Var(model.GT, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_charge = pyo.Var(model.IN, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_discharge = pyo.Var(model.IN, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.SOC = pyo.Var(model.IN, model.T_SOC, domain=pyo.NonNegativeReals, initialize=0.0)
    model.NetInjection = pyo.Var(model.N, model.T, domain=pyo.Reals, initialize=0.0)
    model.nodal_balance = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, n, t: sum(m.P_gen[g, t] for g in generator_hours_at_node_time[n, t])
        + sum(
            m.P_discharge[i, node, t] - m.P_charge[i, node, t]
            for i, node in storage_pairs_at_node[n]
        )
        - data.demand_el[n, t]
        == m.NetInjection[n, t],
    )
    model.system_balance = pyo.Constraint(
        model.T, rule=lambda m, t: sum(m.NetInjection[n, t] for n in m.N) == 0.0
    )
    model.generation_capacity_bound = pyo.Constraint(
        model.GT,
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
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.P_charge[i, n, t] <= power_capacity(m, i, n),
    )
    model.discharge_power_bound = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.P_discharge[i, n, t] <= power_capacity(m, i, n),
    )
    model.soc_transition = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: m.SOC[i, n, t]
        == m.SOC[i, n, t - 1] + eta * m.P_charge[i, n, t] - m.P_discharge[i, n, t] / eta,
    )
    model.soc_capacity_bound = pyo.Constraint(
        model.IN,
        model.T_SOC,
        rule=lambda m, i, n, tau: m.SOC[i, n, tau] <= energy_capacity(m, i, n),
    )
    model.soc_periodicity = pyo.Constraint(
        model.IN,
        rule=lambda m, i, n: m.SOC[i, n, 0] == m.SOC[i, n, last_t],
    )

    # Spot duals: exact maintained EPEC bounds.
    model.lam = pyo.Var(
        model.N, model.T, bounds=(-price_bound_eur_per_mwh, price_bound_eur_per_mwh), initialize=80.0
    )
    model.lam_sys = pyo.Var(
        model.T, bounds=(-price_bound_eur_per_mwh, price_bound_eur_per_mwh), initialize=80.0
    )
    model.nu_gen = pyo.Var(model.GT, bounds=(-dual_bound_eur_per_mwh, 0.0), initialize=0.0)
    model.mu_up = pyo.Var(model.L, model.T, bounds=(-dual_bound_eur_per_mwh, 0.0), initialize=0.0)
    model.mu_dn = pyo.Var(model.L, model.T, bounds=(0.0, dual_bound_eur_per_mwh), initialize=0.0)
    model.rho_ch = pyo.Var(model.IN, model.T, bounds=(-dual_bound_eur_per_mwh, 0.0), initialize=0.0)
    model.sig_dis = pyo.Var(model.IN, model.T, bounds=(-dual_bound_eur_per_mwh, 0.0), initialize=0.0)
    model.gam = pyo.Var(
        model.IN, model.T, bounds=(-dual_bound_eur_per_mwh, dual_bound_eur_per_mwh), initialize=0.0
    )
    model.del_soc = pyo.Var(model.IN, model.T_SOC, bounds=(-dual_bound_eur_per_mwh, 0.0), initialize=0.0)
    model.rho_per = pyo.Var(
        model.IN, bounds=(-dual_bound_eur_per_mwh, dual_bound_eur_per_mwh), initialize=0.0
    )

    model.gen_stationarity = pyo.Constraint(
        model.GT,
        rule=lambda m, g, t: sum(m.lam[n, t] for n in gen_nodes.get(g, [])) + m.nu_gen[g, t]
        <= data.generation_cost[g],
    )
    model.charge_stationarity = pyo.Constraint(
        model.IN,
        model.T,
        rule=lambda m, i, n, t: -m.lam[n, t] + m.rho_ch[i, n, t] - eta * m.gam[i, n, t]
        <= 0.5 * degradation[i],
    )
    model.discharge_stationarity = pyo.Constraint(
        model.IN,
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

    model.soc_stationarity = pyo.Constraint(model.IN, model.T_SOC, rule=soc_stationarity)
    model.spot_primal_value = pyo.Expression(
        expr=sum(data.generation_cost[g] * model.P_gen[g, t] for g, t in model.GT)
        + sum(
            0.5 * degradation[i] * (model.P_charge[i, n, t] + model.P_discharge[i, n, t])
            for i, n in model.IN
            for t in model.T
        )
    )
    model.spot_dual_value = pyo.Expression(
        expr=sum(data.demand_el[n, t] * model.lam[n, t] for n in model.N for t in model.T)
        + sum(data.generation_capacity[g, t] * model.nu_gen[g, t] for g, t in model.GT)
        + sum(
            data.line_limit[line] * (model.mu_up[line, t] - model.mu_dn[line, t])
            for line in model.L
            for t in model.T
        )
        + sum(
            power_capacity(model, i, n) * (model.rho_ch[i, n, t] + model.sig_dis[i, n, t])
            for i, n in model.IN
            for t in model.T
        )
        + sum(
            energy_capacity(model, i, n) * model.del_soc[i, n, tau]
            for i, n in model.IN
            for tau in model.T_SOC
        )
    )
    model.spot_strong_duality = pyo.Constraint(
        expr=model.spot_primal_value == model.spot_dual_value
    )

    # Leader objective.
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
            if share != 0.0 and gen_node.get(g) is not None and (g, t) in model.GT
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
    if payment_rule == "uniform":
        model.active_access_payment = pyo.Expression(
            expr=sum(
                model.auction_capacity_dual[n] * model.award[inv.investor_id, n]
                for n in model.N
            )
        )
    else:
        model.active_access_payment = pyo.Expression(
            expr=sum(
                model.active_bid_price[n] * model.award[inv.investor_id, n]
                for n in model.N
            )
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
    model._max_bid_price_eur_per_mw_day = float(max_bid_price_eur_per_mw_day)
    model._investors = [inv.investor_id, *rivals]
    model._rival_quantity = rival_quantity
    model._rival_price = rival_price
    model._rival_duration = rival_duration
    model._node_limits = limits
    model._degradation = degradation
    model._initial_duration_hours = initial_duration
    model._permitted_active_nodes = sorted(permitted_active_nodes)
    model._price_bound_eur_per_mwh = float(price_bound_eur_per_mwh)
    model._dual_bound_eur_per_mwh = float(dual_bound_eur_per_mwh)
    model._capacity_cleanup_tol_mw = float(capacity_cleanup_tol_mw)
    model._tie_break_epsilon_eur_per_mw_day = float(tie_break_epsilon_eur_per_mw_day)
    model._payment_rule = payment_rule
    model._auction_quadratic_epsilon_eur_per_mw2_day = auction_epsilon
    model._priority_offset_eur_per_mw_day = priority_offset
    model._storage_pairs = tuple(storage_pairs)
    model._positive_generator_hours = tuple(positive_generator_hours)
    return model


def _current_fixed_bids(model: pyo.ConcreteModel) -> list[Bid]:
    bids: list[Bid] = []
    for investor, node in model.A:
        if investor == model._active_id:
            quantity = max(0.0, value(model.active_bid_quantity[node]))
            price = max(0.0, value(model.active_bid_price[node]))
        else:
            quantity = model._rival_quantity[investor, node]
            price = model._rival_price[investor, node]
        bids.append(Bid(investor, node, quantity, price))
    return bids


def initialize_from_independent_followers(model: pyo.ConcreteModel) -> None:
    """Warm-start the embedded variables without imposing a secondary selection rule."""
    bids = _current_fixed_bids(model)
    solver = get_ipopt_solver({"max_cpu_time": 60.0})
    primal = build_auction_primal(
        bids,
        model._node_limits,
        model._tie_break_epsilon_eur_per_mw_day,
        model._auction_quadratic_epsilon_eur_per_mw2_day,
    )
    primal_result = solver.solve(primal, tee=False)
    if primal_result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return

    awards = awarded_mw(primal)
    for investor, node in model.A:
        model.award[investor, node].set_value(awards.get((investor, node), 0.0))
    if model._payment_rule == "uniform":
        epsilon = model._auction_quadratic_epsilon_eur_per_mw2_day
        dual_cap = model._max_bid_price_eur_per_mw_day
        clearing = uniform_clearing_prices(primal)
        raw_price = {(bid.investor, bid.node): bid.price_eur_per_mw for bid in bids}
        for node in model.N:
            model.auction_capacity_dual[node].set_value(
                min(dual_cap, max(0.0, clearing[node]))
            )
        for investor, node in model.A:
            eta = (
                raw_price[investor, node]
                - epsilon * awards.get((investor, node), 0.0)
                - clearing[node]
            )
            model.auction_quantity_dual[investor, node].set_value(
                min(dual_cap, max(0.0, eta))
            )
    else:
        dual = build_auction_dual(
            bids,
            model._node_limits,
            dual_bound_eur_per_mw_day=model._dual_bound_eur_per_mwh,
            tie_break_epsilon_eur_per_mw_day=model._tie_break_epsilon_eur_per_mw_day,
        )
        dual_result = solver.solve(dual, tee=False)
        if dual_result.solver.termination_condition == pyo.TerminationCondition.optimal:
            for node in model.N:
                model.auction_capacity_dual[node].set_value(max(0.0, value(dual.capacity_dual[node])))
            bid_index = {(dual.bid[k].investor, dual.bid[k].node): k for k in dual.K}
            for investor, node in model.A:
                model.auction_quantity_dual[investor, node].set_value(
                    max(0.0, value(dual.quantity_dual[bid_index[investor, node]]))
                )

    active_id = model._active_id
    for node in model.N:
        model.active_energy[node].set_value(
            model._initial_duration_hours[node] * value(model.award[active_id, node])
        )

    x_power = {(i, n): 0.0 for i in model._investors for n in model.N}
    x_energy = {(i, n): 0.0 for i in model._investors for n in model.N}
    for investor, node in model.IN:
        award = value(model.award[investor, node])
        x_power[investor, node] = award
        x_energy[investor, node] = (
            value(model.active_energy[node])
            if investor == active_id
            else model._rival_duration[investor, node] * award
        )
    spot_data = replace(
        model._market_data,
        storage_units=list(model._investors),
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
    for g, t in model.GT:
        model.P_gen[g, t].set_value(max(0.0, value(reference.P_gen[g, t])))
        marginal_price = sum(prices[n, t] for n in gen_nodes.get(g, []))
        model.nu_gen[g, t].set_value(
            max(-model._dual_bound_eur_per_mwh, min(0.0, spot_data.generation_cost[g] - marginal_price))
        )
    for n in model.N:
        for t in model.T:
            model.NetInjection[n, t].set_value(value(reference.NetInjection[n, t]))
            model.lam[n, t].set_value(
                min(model._price_bound_eur_per_mwh, max(-model._price_bound_eur_per_mwh, prices[n, t]))
            )
    for t in model.T:
        average_price = sum(prices[n, t] for n in model.N) / max(1, len(list(model.N)))
        model.lam_sys[t].set_value(
            min(model._price_bound_eur_per_mwh, max(-model._price_bound_eur_per_mwh, average_price))
        )
    for investor, node in model.IN:
        for t in model.T:
            model.P_charge[investor, node, t].set_value(
                max(0.0, value(reference.P_charge[investor, node, t]))
            )
            model.P_discharge[investor, node, t].set_value(
                max(0.0, value(reference.P_discharge[investor, node, t]))
            )
            model.gam[investor, node, t].set_value(-prices[node, t])
        for tau in model.T_SOC:
            model.SOC[investor, node, tau].set_value(
                max(0.0, value(reference.SOC[investor, node, tau]))
            )


def solve_mpec(
    model: pyo.ConcreteModel,
    max_cpu_time: float,
    solver_tol: float = DEFAULT_SOLVER_TOL,
    tee: bool = False,
) -> str:
    solver = get_ipopt_solver(
        {
            "max_cpu_time": max_cpu_time,
            "tol": solver_tol,
            "acceptable_tol": solver_tol,
        }
    )
    result = solver.solve(model, tee=tee, load_solutions=False)
    status = str(result.solver.status)
    termination = str(result.solver.termination_condition)
    message = str(result.solver.message or "")
    model._solver_status = status
    model._solver_message = message
    if len(result.solution) > 0:
        try:
            model.solutions.load_from(result)
        except ValueError:
            # A failed tick candidate must not abort the remaining price search.
            # Its initialized values are retained only for diagnostic export.
            pass
    return termination


def _max_abs_component(component: pyo.Var) -> tuple[float, str | None]:
    maximum = 0.0
    name: str | None = None
    for index in component:
        magnitude = abs(value(component[index]))
        if magnitude > maximum:
            maximum = magnitude
            name = f"{component.name}[{index!r}]"
    return maximum, name


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
            "active_duration_hours": energy / award if award > model._capacity_cleanup_tol_mw else None,
            "total_award_mw": sum(
                max(0.0, value(model.award[i, n]))
                for i, n in model.A
                if n == node
            ),
            "auction_capacity_dual_eur_per_mw_day": value(model.auction_capacity_dual[node]),
            "rival_awards_mw": {
                rival: max(0.0, value(model.award[rival, node]))
                for rival in model.R
                if (rival, node) in model.A
            },
        }

    fixed_bids = _current_fixed_bids(model)
    independent = build_auction_primal(
        fixed_bids,
        model._node_limits,
        model._tie_break_epsilon_eur_per_mw_day,
        model._auction_quadratic_epsilon_eur_per_mw2_day,
    )
    independent_solver = get_ipopt_solver(
        {"max_cpu_time": 60.0, "tol": 1.0e-9, "acceptable_tol": 1.0e-8}
    )
    independent_term = str(
        independent_solver.solve(independent, tee=False).solver.termination_condition
    )
    reclear_awards = awarded_mw(independent) if independent_term == "optimal" else {}
    max_award_difference = max(
        (
            abs(value(model.award[i, n]) - reclear_awards.get((i, n), 0.0))
            for i, n in model.A
        ),
        default=0.0,
    )
    max_clearing_price_difference = 0.0
    if model._payment_rule == "uniform" and independent_term == "optimal":
        reclear_clearing = uniform_clearing_prices(independent)
        for node in model.N:
            nodes[node]["independent_clearing_price_eur_per_mw_day"] = reclear_clearing[node]
            payment_relevant = (
                nodes[node]["active_award_mw"] > model._capacity_cleanup_tol_mw
                or reclear_awards.get((active_id, node), 0.0) > model._capacity_cleanup_tol_mw
            )
            if payment_relevant:
                max_clearing_price_difference = max(
                    max_clearing_price_difference,
                    abs(
                        nodes[node]["auction_capacity_dual_eur_per_mw_day"]
                        - reclear_clearing[node]
                    ),
                )

    price_components = (model.lam, model.lam_sys)
    internal_components = (
        model.auction_capacity_dual,
        model.auction_quantity_dual,
        model.nu_gen,
        model.mu_up,
        model.mu_dn,
        model.rho_ch,
        model.sig_dis,
        model.gam,
        model.del_soc,
        model.rho_per,
    )
    price_candidates = [_max_abs_component(component) for component in price_components]
    internal_candidates = [_max_abs_component(component) for component in internal_components]
    max_price, max_price_name = max(
        price_candidates, key=lambda candidate: candidate[0], default=(0.0, None)
    )
    max_internal, max_internal_name = max(
        internal_candidates, key=lambda candidate: candidate[0], default=(0.0, None)
    )
    return {
        "termination": termination,
        "solver_status": getattr(model, "_solver_status", None),
        "solver_message": getattr(model, "_solver_message", None),
        "interpretation": "local optimistic continuous-bid MPEC candidate",
        "formulation": "leader + auction primal/dual/strong-duality + spot primal/dual/strong-duality",
        "payment_rule": model._payment_rule,
        "auction_quadratic_epsilon_eur_per_mw2_day": (
            model._auction_quadratic_epsilon_eur_per_mw2_day
        ),
        "bid_merit_tie_break_epsilon_eur_per_mw_day": model._tie_break_epsilon_eur_per_mw_day,
        "priority_offset_eur_per_mw_day": model._priority_offset_eur_per_mw_day,
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
        "max_embedded_vs_reclear_clearing_price_difference_eur_per_mw_day": (
            max_clearing_price_difference
        ),
        "price_bound_eur_per_mwh": model._price_bound_eur_per_mwh,
        "dual_bound_eur_per_mwh": model._dual_bound_eur_per_mwh,
        "max_abs_price_dual": max_price,
        "max_abs_price_dual_name": max_price_name,
        "max_abs_price_dual_fraction_of_bound": max_price / model._price_bound_eur_per_mwh,
        "max_abs_internal_dual": max_internal,
        "max_abs_internal_dual_name": max_internal_name,
        "max_abs_internal_dual_fraction_of_bound": max_internal / model._dual_bound_eur_per_mwh,
        "embedded_storage_pair_count": len(model._storage_pairs),
        "positive_generator_hour_count": len(model._positive_generator_hours),
        "variable_count": model.nvariables(),
        "constraint_count": model.nconstraints(),
        "nodes": nodes,
    }


def print_summary(summary: Mapping[str, object]) -> None:
    print(f"Termination: {summary['termination']}")
    print(f"Interpretation: {summary['interpretation']}")
    print(f"Active profit: {summary['profit_eur_per_day']:,.2f} EUR/day")
    print(
        "  spot {0:,.2f} + generation {1:,.2f} - degradation {2:,.2f} "
        "- CAPEX {3:,.2f} - access {4:,.2f}".format(
            summary["spot_revenue_eur_per_day"],
            summary["generation_rent_eur_per_day"],
            summary["degradation_cost_eur_per_day"],
            summary["capex_eur_per_day"],
            summary["access_payment_eur_per_day"],
        )
    )
    print(f"Auction strong-duality gap: {summary['auction_strong_duality_gap']:.3e}")
    print(f"Spot strong-duality gap: {summary['spot_strong_duality_gap']:.3e}")
    print(
        f"Model size: {summary['variable_count']} variables / "
        f"{summary['constraint_count']} constraints"
    )
    print("Active bids and awards:")
    for node, row in summary["nodes"].items():
        if row["active_bid_quantity_mw"] > 1.0e-5 or row["active_award_mw"] > 1.0e-5:
            duration = row["active_duration_hours"]
            duration_text = "-" if duration is None else f"{duration:.2f} h"
            print(
                f"  {node}: bid {row['active_bid_quantity_mw']:.3f} MW @ "
                f"{row['active_bid_price_eur_per_mw_day']:.3f} EUR/MW/day; "
                f"award {row['active_award_mw']:.3f} MW; "
                f"energy {row['active_energy_mwh']:.3f} MWh ({duration_text})"
            )


def _price_is_on_tick(price: float, tick: float) -> bool:
    nearest = round(price / tick) * tick
    return abs(price - nearest) <= 1.0e-9 * max(1.0, abs(price))


def pay_as_bid_tick_candidates(
    *,
    active_investor: str,
    active_node: str,
    rivals: Iterable[RivalBid],
    max_bid_price_eur_per_mw_day: float,
    bid_price_tick_eur_per_mw_day: float,
    tie_break_epsilon_eur_per_mw_day: float,
) -> list[float]:
    """Return the economically distinct raw bid prices on the exact grid."""
    tick = bid_price_tick_eur_per_mw_day
    if tick <= 0.0:
        raise ValueError("Bid-price tick must be positive in grid mode.")
    if tie_break_epsilon_eur_per_mw_day <= 0.0:
        raise ValueError("Grid mode requires a positive deterministic merit offset.")
    rival_list = list(rivals)
    investors = {active_investor, *(bid.investor for bid in rival_list)}
    maximum_priority_gap = tie_break_epsilon_eur_per_mw_day * max(0, len(investors) - 1)
    if maximum_priority_gap >= tick:
        raise ValueError(
            "The maximum deterministic priority gap must be smaller than one bid tick."
        )

    candidate_ticks = {0}
    for rival in rival_list:
        if rival.node != active_node or rival.quantity_mw <= 1.0e-9:
            continue
        if not _price_is_on_tick(rival.price_eur_per_mw_day, tick):
            raise ValueError(
                f"Rival bid {rival.investor}/{rival.node} at "
                f"{rival.price_eur_per_mw_day} is not on the {tick} price grid."
            )
        rival_tick = round(rival.price_eur_per_mw_day / tick)
        candidate_ticks.update({rival_tick, rival_tick + 1})
    return sorted(
        round(index * tick, 12)
        for index in candidate_ticks
        if 0.0 <= index * tick <= max_bid_price_eur_per_mw_day + 1.0e-12
    )


def tick_candidate_diagnostic(
    summary: Mapping[str, object],
    elapsed_seconds: float,
    fixed_bid_price: float,
) -> dict[str, object]:
    payment_check = sum(
        row["active_bid_price_eur_per_mw_day"] * row["active_award_mw"]
        for row in summary["nodes"].values()
    )
    payment_residual = summary["access_payment_eur_per_day"] - payment_check
    rejection_reasons: list[str] = []
    if summary["termination"] != "optimal":
        rejection_reasons.append(f"termination_{summary['termination']}")
    if summary["max_embedded_vs_reclear_award_difference_mw"] > TICK_AWARD_RECLEAR_TOLERANCE_MW:
        rejection_reasons.append("auction_award_reclear_mismatch")
    if abs(summary["auction_strong_duality_gap"]) > TICK_AUCTION_GAP_TOLERANCE:
        rejection_reasons.append("auction_strong_duality_gap")
    if abs(summary["spot_strong_duality_gap"]) > TICK_SPOT_GAP_TOLERANCE:
        rejection_reasons.append("spot_strong_duality_gap")
    if abs(payment_residual) > 1.0e-6:
        rejection_reasons.append("pay_as_bid_payment_mismatch")
    return {
        "fixed_bid_price_eur_per_mw_day": fixed_bid_price,
        "termination": summary["termination"],
        "solver_status": summary.get("solver_status"),
        "solver_message": summary.get("solver_message"),
        "valid": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "elapsed_seconds": elapsed_seconds,
        "profit_eur_per_day": summary["profit_eur_per_day"],
        "access_payment_eur_per_day": summary["access_payment_eur_per_day"],
        "payment_reconciliation_residual_eur_per_day": payment_residual,
        "max_embedded_vs_reclear_award_difference_mw": summary[
            "max_embedded_vs_reclear_award_difference_mw"
        ],
        "auction_strong_duality_gap": summary["auction_strong_duality_gap"],
        "spot_strong_duality_gap": summary["spot_strong_duality_gap"],
    }


def solve_fixed_price_response(
    data: MarketData,
    rivals: list[RivalBid],
    active: InvestorConfig,
    cfg,
    fixed_bid_price: float | None = None,
) -> tuple[pyo.ConcreteModel, dict[str, object], float]:
    active_nodes = None if cfg.active_node is None else [cfg.active_node]
    model = build_one_leader_two_follower_mpec(
        data,
        rivals,
        active_investor=active,
        active_nodes=active_nodes,
        node_limit_mw=cfg.node_limit_mw,
        max_bid_price_eur_per_mw_day=cfg.max_bid_price_eur_per_mw_day,
        initial_bid_quantity_mw=cfg.initial_bid_quantity_mw,
        initial_bid_price_eur_per_mw_day=cfg.initial_bid_price_eur_per_mw_day,
        initial_duration_hours=cfg.initial_duration_hours,
        price_bound_eur_per_mwh=cfg.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=cfg.dual_bound_eur_per_mwh,
        capacity_cleanup_tol_mw=cfg.capacity_cleanup_tol_mw,
        tie_break_epsilon_eur_per_mw_day=(
            cfg.tie_break_epsilon_eur_per_mw_day
            if cfg.bid_price_tick_eur_per_mw_day > 0.0
            else 0.0
        ),
        payment_rule=cfg.payment_rule,
        auction_quadratic_epsilon_eur_per_mw2_day=(
            cfg.auction_quadratic_epsilon_eur_per_mw2_day
            if cfg.payment_rule == "uniform"
            else 0.0
        ),
    )
    if fixed_bid_price is not None:
        model.active_bid_price[cfg.active_node].fix(fixed_bid_price)
    initialize_from_independent_followers(model)
    started = time.perf_counter()
    termination = solve_mpec(model, cfg.max_cpu_time, cfg.solver_tol, tee=cfg.tee)
    elapsed = time.perf_counter() - started
    summary = summarize(model, termination)
    summary["solve_seconds"] = elapsed
    return model, summary, elapsed


def run_tick_search(
    data: MarketData,
    rivals: list[RivalBid],
    active: InvestorConfig,
    cfg,
) -> tuple[dict[str, object], bool]:
    if cfg.active_node is None:
        raise ValueError("Exact bid-price enumeration requires exactly one active node.")
    prices = pay_as_bid_tick_candidates(
        active_investor=cfg.active_investor,
        active_node=cfg.active_node,
        rivals=rivals,
        max_bid_price_eur_per_mw_day=cfg.max_bid_price_eur_per_mw_day,
        bid_price_tick_eur_per_mw_day=cfg.bid_price_tick_eur_per_mw_day,
        tie_break_epsilon_eur_per_mw_day=cfg.tie_break_epsilon_eur_per_mw_day,
    )
    records: list[tuple[dict[str, object], dict[str, object]]] = []
    for price in prices:
        print(f"Pay-as-bid grid candidate: {price:.12g} EUR/MW/day")
        _, summary, elapsed = solve_fixed_price_response(
            data, rivals, active, cfg, fixed_bid_price=price
        )
        diagnostic = tick_candidate_diagnostic(summary, elapsed, price)
        print(
            f"  termination={summary['termination']}, valid={diagnostic['valid']}, "
            f"profit={summary['profit_eur_per_day']:,.2f} EUR/day"
        )
        records.append((summary, diagnostic))
    valid_records = [record for record in records if record[1]["valid"]]
    if not valid_records:
        return {
            "termination": "no_valid_tick_candidate",
            "interpretation": "pay-as-bid tick search failed",
            "bid_price_tick_eur_per_mw_day": cfg.bid_price_tick_eur_per_mw_day,
            "candidate_prices_eur_per_mw_day": prices,
            "tick_candidate_diagnostics": [record[1] for record in records],
        }, False
    selected_summary, selected_diagnostic = max(
        valid_records, key=lambda record: record[0]["profit_eur_per_day"]
    )
    return {
        **selected_summary,
        "interpretation": "local MPEC candidate selected on exact pay-as-bid grid",
        "bid_price_selection_mode": "pay_as_bid_tick_enumeration",
        "bid_price_tick_eur_per_mw_day": cfg.bid_price_tick_eur_per_mw_day,
        "candidate_prices_eur_per_mw_day": prices,
        "selected_tick_price_eur_per_mw_day": selected_diagnostic[
            "fixed_bid_price_eur_per_mw_day"
        ],
        "tick_candidate_diagnostics": [record[1] for record in records],
    }, True


def main() -> int:
    cfg = parse_single_mpec_cli()
    if cfg.bid_price_tick_eur_per_mw_day > 0.0 and cfg.payment_rule != "pay_as_bid":
        raise ValueError("Exact tick enumeration is a pay-as-bid device; use --payment-rule pay_as_bid.")
    data = load_market_data(cfg.data_path)
    from epec_diagonalization import four_investor_portfolio_profiles

    profiles = {profile.investor_id: profile for profile in four_investor_portfolio_profiles(data)}
    active = profiles[cfg.active_investor]
    if cfg.rival_bids_path:
        all_input_bids = load_rival_bid_vector(cfg.rival_bids_path)
        rivals = [bid for bid in all_input_bids if bid.investor != cfg.active_investor]
    else:
        rivals = demo_rival_bid_vector(data.nodes, cfg.active_investor)
    if cfg.bid_price_tick_eur_per_mw_day > 0.0:
        summary, succeeded = run_tick_search(data, rivals, active, cfg)
    else:
        _, summary, _ = solve_fixed_price_response(data, rivals, active, cfg)
        succeeded = summary["termination"] == "optimal"
    summary["configuration"] = {
        **asdict(cfg),
        "data_path": str(cfg.data_path),
        "rival_bids_path": None if cfg.rival_bids_path is None else str(cfg.rival_bids_path),
        "output_path": str(cfg.output_path),
    }
    summary["rival_bids"] = [asdict(bid) for bid in rivals]
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if succeeded:
        print_summary(summary)
    else:
        print(f"Termination: {summary['termination']}")
    print(f"Wrote {cfg.output_path}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
