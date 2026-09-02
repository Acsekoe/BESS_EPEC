"""Investor economic parameters and capital-recovery helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from primal_market_clearing_model import MarketData


@dataclass(frozen=True)
class InvestorConfig:
    investor_id: str
    wacc: float = 0.08
    lifetime_years: int = 15
    cost_power_eur_per_mw: float = 6_600.0
    cost_energy_eur_per_mwh: float = 18_800.0
    degradation_eur_per_mwh: float = 15.0
    ratio_min: float = 2.0
    ratio_max: float = 8.0
    owned_generation_shares: Mapping[str, float] = field(default_factory=dict)


def capital_recovery_factor(wacc: float, lifetime_years: int) -> float:
    if wacc == 0.0:
        return 1.0 / lifetime_years
    growth = (1.0 + wacc) ** lifetime_years
    return wacc * growth / (growth - 1.0)


def _generator_nodes(data: MarketData) -> dict[str, list[str]]:
    result = {g: [] for g in data.generators}
    for n in data.nodes:
        for g in data.generators_at_node.get(n, []):
            result[g].append(n)
    return result


def split_portfolio_investors(
    data: MarketData, *, include_i2: bool = False
) -> tuple[InvestorConfig, ...]:
    """Population for the owner-split strategic-generation experiments.

    Requires the owner-split input (market_data_strategic_generation.json):
    each renewable unit carries its owner id in its name (``_I3_``/``_I4_``)
    and is owned 100% by that investor, replacing the fractional shares of
    one aggregate unit. I1 remains the merchant benchmark; I2 is excluded by
    default but can be restored for robustness tests.
    """

    owned: dict[str, dict[str, float]] = {"I3": {}, "I4": {}}
    for generator in data.generators:
        for investor_id in owned:
            if f"_{investor_id}_" in generator:
                owned[investor_id][generator] = 1.0
    if not owned["I3"] or not owned["I4"]:
        raise ValueError(
            "The split-portfolio population requires owner-specific renewable "
            "units (use market_data_strategic_generation.json)."
        )
    population = [
        InvestorConfig("I1", wacc=0.08),
        InvestorConfig("I3", wacc=0.08, owned_generation_shares=owned["I3"]),
        InvestorConfig("I4", wacc=0.08, owned_generation_shares=owned["I4"]),
    ]
    if include_i2:
        population.insert(1, InvestorConfig("I2", wacc=0.12))
    return tuple(population)
