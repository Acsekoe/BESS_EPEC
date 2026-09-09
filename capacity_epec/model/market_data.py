"""The IEEE-9 market input.

`MarketData` is the immutable description of the network, demand and
generation fleet that every other module reads.  Storage capacity is *not*
part of it: capacities are the strategic variables of the game and are always
passed in explicitly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "input" / "market_data.json"
AFRR_DEMAND_UP = 50.0
AFRR_DEMAND_DOWN = 50.0
AFRR_PENALTY_PRICE = 3000.0  # EUR/MW/h, NOT EUR/MW/block


@dataclass(frozen=True)
class MarketData:
    """One deterministic market day on a fixed network."""

    nodes: Sequence[str]
    generators: Sequence[str]
    times: Sequence[int]
    soc_times: Sequence[int]
    lines: Sequence[str]
    generators_at_node: Mapping[str, Sequence[str]]
    generation_cost: Mapping[str, float]
    generation_capacity: Mapping[tuple[str, int], float]
    demand_el: Mapping[tuple[str, int], float]
    line_limit: Mapping[str, float]
    ptdf: Mapping[tuple[str, str], float]
    eta: float
    demand_adjustment_penalty_eur_per_mw2: float
    node_connection_limit: Mapping[str, float]
    # Submitted ISO offers.  When a generator is absent it offers its true
    # economic cost, so an input without this block is a truthful benchmark.
    generation_offer: Mapping[str, float] | None = None
    afrr_enabled: bool = False
    afrr_block_hours: Mapping[int, Sequence[int]] = field(
        default_factory=lambda: {b: tuple(range(4 * b - 3, 4 * b + 1)) for b in range(1, 7)}
    )
    afrr_demand_up_mw: float = AFRR_DEMAND_UP
    afrr_demand_down_mw: float = AFRR_DEMAND_DOWN
    afrr_penalty_eur_per_mw_hour: float = AFRR_PENALTY_PRICE
    afrr_response_hours: float = 0.5
    afrr_activation_up: float = 0.10
    afrr_activation_down: float = 0.10
    afrr_generator_offer_eur_per_mw_hour: Mapping[str, float] = field(default_factory=dict)
    afrr_eligible_generators: Sequence[str] = ()

    def afrr_block(self, time: int) -> int:
        return next(b for b, hours in self.afrr_block_hours.items() if time in hours)

    def afrr_duration(self, block: int) -> float:
        return float(len(self.afrr_block_hours[block]))

    @property
    def last_time(self) -> int:
        return max(self.times)

    def demand_is_adjustable(self, node: str, time: int) -> bool:
        """Whether the ISO's demand-adjustment term may act here.

        The term exists to pin a unique LMP, and at the ISO's optimum it settles
        at ``DemandAdjustment = LMP / rho``.  That makes it a virtual generator
        with marginal cost ``rho * x``, so where there is no demand it is not
        adjusting anything -- it is energy conjured from nothing.  Six of the
        nine IEEE-9 nodes carry no load at any hour, and at rho = 25 they were
        supplying 213.6 MWh/day between them, 1.6% of system load.

        Holding it at zero there costs nothing: the LMP at such a node is
        already pinned by ``net_injection_stationarity`` against the system
        price and the line duals, and the cleared prices are unchanged.
        """

        return self.demand_el[node, time] > 0.0

    def offer(self, generator: str) -> float:
        """What the ISO dispatch objective pays for one generator."""

        if self.generation_offer is not None and generator in self.generation_offer:
            return float(self.generation_offer[generator])
        return float(self.generation_cost[generator])

    def nodes_by_generator(self) -> dict[str, list[str]]:
        """Invert ``generators_at_node``."""

        located: dict[str, list[str]] = {g: [] for g in self.generators}
        for node in self.nodes:
            for generator in self.generators_at_node.get(node, []):
                located[generator].append(node)
        return located


def load_market_data(path: Path = DEFAULT_DATA_PATH) -> MarketData:
    """Read and validate the canonical benchmark input."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    afrr = raw.get("afrr", {})

    def keyed(records: Sequence[Mapping[str, Any]], key_fields, value_field):
        return {
            tuple(record[field] for field in key_fields): float(record[value_field])
            for record in records
        }

    data = MarketData(
        nodes=[str(node) for node in raw["nodes"]],
        generators=[str(generator) for generator in raw["generators"]],
        times=[int(hour) for hour in raw["times"]],
        soc_times=[int(hour) for hour in raw["soc_times"]],
        lines=[str(line) for line in raw["lines"]],
        generators_at_node={
            str(node): [str(generator) for generator in generators]
            for node, generators in raw["generators_at_node"].items()
        },
        generation_cost={
            str(generator): float(cost)
            for generator, cost in raw["generation_cost"].items()
        },
        generation_capacity=keyed(raw["generation_capacity"], ["generator", "hour"], "capacity_mw"),
        demand_el=keyed(raw["demand_el"], ["node", "hour"], "demand_mw"),
        line_limit={str(line): float(limit) for line, limit in raw["line_limit"].items()},
        ptdf=keyed(raw["ptdf"], ["line", "node"], "ptdf"),
        eta=float(raw["eta"]),
        demand_adjustment_penalty_eur_per_mw2=float(
            raw.get("demand_adjustment_penalty_eur_per_mw2", 500.0)
        ),
        node_connection_limit={
            str(node): float(limit) for node, limit in raw["node_connection_limit"].items()
        },
        generation_offer=(
            {str(generator): float(offer) for generator, offer in raw["generation_offer"].items()}
            if "generation_offer" in raw
            else None
        ),
        afrr_block_hours={int(b): tuple(hours) for b, hours in afrr.get(
            "block_hours", {b: list(range(4*b-3, 4*b+1)) for b in range(1, 7)}
        ).items()},
        afrr_demand_up_mw=float(afrr.get("demand_up_mw", AFRR_DEMAND_UP)),
        afrr_demand_down_mw=float(afrr.get("demand_down_mw", AFRR_DEMAND_DOWN)),
        afrr_penalty_eur_per_mw_hour=float(afrr.get("penalty_eur_per_mw_hour", AFRR_PENALTY_PRICE)),
        afrr_response_hours=float(afrr.get("response_hours", 0.5)),
        afrr_activation_up=float(afrr.get("activation_up", 0.10)),
        afrr_activation_down=float(afrr.get("activation_down", 0.10)),
        afrr_generator_offer_eur_per_mw_hour={str(g): float(c) for g, c in afrr.get(
            "generator_offer_eur_per_mw_hour", {}
        ).items()},
        afrr_eligible_generators=tuple(afrr.get("eligible_generators", [])),
    )
    _validate(data)
    return data


def _validate(data: MarketData) -> None:
    validate_afrr(data)
    if not 0.0 < data.eta <= 1.0:
        raise ValueError("Storage efficiency eta must lie in (0, 1].")
    if data.demand_adjustment_penalty_eur_per_mw2 <= 0.0:
        raise ValueError("The demand-adjustment penalty must be positive.")
    missing_connection_limits = [n for n in data.nodes if n not in data.node_connection_limit]
    if missing_connection_limits:
        raise ValueError(f"Missing node connection limits for {missing_connection_limits}.")
    if any(data.node_connection_limit[n] <= 0.0 for n in data.nodes):
        raise ValueError("Node connection limits must be positive.")
    if data.generation_offer is not None:
        unknown = set(data.generation_offer) - set(data.generators)
        if unknown:
            raise ValueError(f"generation_offer names unknown generators: {sorted(unknown)}")
    missing_nodes = [n for n in data.nodes if n not in data.generators_at_node]
    if missing_nodes:
        raise ValueError(f"Missing generators_at_node entries for {missing_nodes}.")
    for time_ in data.times:
        for node in data.nodes:
            if (node, time_) not in data.demand_el:
                raise ValueError(f"Missing demand_el for node {node}, hour {time_}.")
        for generator in data.generators:
            if (generator, time_) not in data.generation_capacity:
                raise ValueError(
                    f"Missing generation_capacity for {generator}, hour {time_}."
                )
    for line in data.lines:
        if line not in data.line_limit:
            raise ValueError(f"Missing line limit for line {line}.")
        for node in data.nodes:
            if (line, node) not in data.ptdf:
                raise ValueError(f"Missing PTDF for line {line}, node {node}.")


def validate_afrr(data: MarketData) -> None:
    """Validate the explicit 24-hour/six-block deterministic benchmark."""
    expected = {b: tuple(range(4*b-3, 4*b+1)) for b in range(1, 7)}
    if {b: tuple(h) for b, h in data.afrr_block_hours.items()} != expected:
        raise ValueError("aFRR requires six consecutive four-hour blocks numbered 1..6.")
    if data.afrr_enabled and (list(data.times) != list(range(1, 25))
                             or list(data.soc_times) != list(range(25))):
        raise ValueError("aFRR requires hourly times 1..24 and SOC endpoints 0..24.")
    values = (data.afrr_demand_up_mw, data.afrr_demand_down_mw,
              data.afrr_penalty_eur_per_mw_hour, data.afrr_response_hours,
              data.afrr_activation_up, data.afrr_activation_down)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("aFRR parameters must be finite and nonnegative.")
    if data.afrr_penalty_eur_per_mw_hour <= 0 or not 0 < data.afrr_response_hours <= 4:
        raise ValueError("aFRR penalty must be positive and response duration in (0, 4].")
    if max(data.afrr_activation_up, data.afrr_activation_down) > 1:
        raise ValueError("Directional expected activation fractions must be in [0, 1].")
    if len(set(data.afrr_eligible_generators)) != len(data.afrr_eligible_generators):
        raise ValueError("Duplicate aFRR eligible generators.")
    if not set(data.afrr_eligible_generators) <= set(data.generators):
        raise ValueError("Unknown aFRR eligible generator.")
    offers = data.afrr_generator_offer_eur_per_mw_hour
    if not set(offers) <= set(data.generators) or not set(data.afrr_eligible_generators) <= set(offers):
        raise ValueError("aFRR offers must cover all eligible generators and name known generators.")
    if not all(math.isfinite(c) and c >= 0 for c in offers.values()):
        raise ValueError("aFRR offers must be finite and nonnegative.")
