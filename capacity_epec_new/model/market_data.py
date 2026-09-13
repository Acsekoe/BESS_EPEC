"""The IEEE-9 market input.

`MarketData` is the immutable description of the network, demand and
generation fleet that every other module reads.  Storage capacity is *not*
part of it: capacities are the strategic variables of the game and are always
passed in explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "input" / "market_data_smoothed.json"


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
    @property
    def last_time(self) -> int:
        return max(self.times)

    def demand_is_adjustable(self, node: str, time: int) -> bool:
        """Whether the ISO's demand-adjustment term may act here.

        With positive rho its optimum satisfies ``DemandAdjustment = LMP/rho``.
        That makes it a virtual generator with marginal cost ``rho*x``, not a
        neutral price tie-break. Where there is no demand it is not adjusting
        anything -- it is energy conjured from nothing. Six of the
        nine IEEE-9 nodes carry no load at any hour, and at rho = 25 they were
        supplying 213.6 MWh/day between them, 1.6% of system load.

        Holding it at zero at unloaded nodes removes that bug without imposing
        ``LMP=0``. With rho zero it is instead held at zero everywhere to model
        inelastic demand; dual-price uniqueness then has to be checked.
        """

        # rho == 0 is the explicit inelastic-demand benchmark.  In that case
        # DemandAdjustment is fixed at zero everywhere and its KKT stationarity
        # equation must be skipped (otherwise 0 == LMP would destroy prices).
        return (
            self.demand_adjustment_penalty_eur_per_mw2 > 0.0
            and self.demand_el[node, time] > 0.0
        )

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
    )
    _validate(data)
    return data


def _validate(data: MarketData) -> None:
    if not 0.0 < data.eta <= 1.0:
        raise ValueError("Storage efficiency eta must lie in (0, 1].")
    if data.demand_adjustment_penalty_eur_per_mw2 < 0.0:
        raise ValueError(
            "The demand-adjustment penalty cannot be negative; zero selects "
            "the inelastic-demand benchmark."
        )
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
