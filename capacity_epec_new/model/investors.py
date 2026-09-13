"""Investor economics: financing, investment cost, and the maintained population."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from market_data import MarketData


@dataclass(frozen=True)
class InvestorConfig:
    """One strategic investor.

    ``owned_generation_shares`` maps a generator to the fraction of its market
    rent this investor earns; it distinguishes renewable portfolio owners from
    pure merchants.
    """

    investor_id: str
    wacc: float = 0.08
    lifetime_years: int = 15
    cost_power_eur_per_mw: float = 6_600.0
    cost_energy_eur_per_mwh: float = 18_800.0
    degradation_eur_per_mwh: float = 15.0
    ratio_min: float = 2.0
    ratio_max: float = 8.0
    owned_generation_shares: Mapping[str, float] = field(default_factory=dict)
    quadratic_cost_power_eur_per_mw2: float = 0.0
    quadratic_cost_energy_eur_per_mwh2: float = 0.0
    # Per-node curvature: 0.5*coef*sum_n(X[n]**2), as opposed to the two fields
    # above which curve on the PORTFOLIO TOTAL (0.5*coef*(sum_n X[n])**2). The
    # portfolio form has an identical marginal cost at every node (it only
    # taxes aggregate size), so it cannot break ties between nodes. This form
    # makes a node's own marginal cost rise with that node's own capacity,
    # which is what actually discourages piling everything onto a few nodes.
    quadratic_cost_power_per_node_eur_per_mw2: float = 0.0
    quadratic_cost_energy_per_node_eur_per_mwh2: float = 0.0
    # Optional node-specific overrides of the two common per-node coefficients.
    # A missing node uses the scalar coefficient above.
    quadratic_cost_power_by_node_eur_per_mw2: Mapping[str, float] = field(
        default_factory=dict
    )
    quadratic_cost_energy_by_node_eur_per_mwh2: Mapping[str, float] = field(
        default_factory=dict
    )

    def clip_duration(self, hours: float) -> float:
        """Project a duration onto this investor's admissible range."""

        return min(self.ratio_max, max(self.ratio_min, hours))


def capital_recovery_factor(wacc: float, lifetime_years: int) -> float:
    if wacc == 0.0:
        return 1.0 / lifetime_years
    growth = (1.0 + wacc) ** lifetime_years
    return wacc * growth / (growth - 1.0)


def daily_investment_cost(
    investor: InvestorConfig,
    power_values: Iterable[object],
    energy_values: Iterable[object],
    nodes: Iterable[str] | None = None,
) -> object:
    """Annualised daily capital cost of one nodal capacity portfolio.

    Accepts plain floats or Pyomo variables, so the same expression serves the
    MPEC objective and the post-solve accounting.
    """

    power_values = tuple(power_values)
    energy_values = tuple(energy_values)
    if len(power_values) != len(energy_values):
        raise ValueError("Power and energy capacity vectors must have the same length.")
    if nodes is None:
        if (
            investor.quadratic_cost_power_by_node_eur_per_mw2
            or investor.quadratic_cost_energy_by_node_eur_per_mwh2
        ):
            raise ValueError("Node labels are required for node-specific quadratic costs.")
        node_labels = tuple("" for _ in power_values)
    else:
        node_labels = tuple(nodes)
        if len(node_labels) != len(power_values):
            raise ValueError("Node labels and capacity vectors must have the same length.")
    nodal_power_quadratic = sum(
        investor.quadratic_cost_power_by_node_eur_per_mw2.get(
            node, investor.quadratic_cost_power_per_node_eur_per_mw2
        )
        * value**2
        for node, value in zip(node_labels, power_values)
    )
    nodal_energy_quadratic = sum(
        investor.quadratic_cost_energy_by_node_eur_per_mwh2.get(
            node, investor.quadratic_cost_energy_per_node_eur_per_mwh2
        )
        * value**2
        for node, value in zip(node_labels, energy_values)
    )
    overnight = (
        investor.cost_power_eur_per_mw * sum(power_values)
        + investor.cost_energy_eur_per_mwh * sum(energy_values)
        + 0.5
        * investor.quadratic_cost_power_eur_per_mw2
        * (sum(power_values)) ** 2
        + 0.5
        * investor.quadratic_cost_energy_eur_per_mwh2
        * (sum(energy_values)) ** 2
        + 0.5
        * nodal_power_quadratic
        + 0.5
        * nodal_energy_quadratic
    )
    return capital_recovery_factor(investor.wacc, investor.lifetime_years) / 365.25 * overnight


def validate_ownership(
    data: MarketData, ownership: Mapping[str, Mapping[str, float]]
) -> None:
    """Reject a generation split that would invent or double-count rent."""

    unknown = {
        generator
        for shares in ownership.values()
        for generator in shares
        if generator not in data.generators
    }
    if unknown:
        raise ValueError(f"Unknown generators in ownership: {sorted(unknown)}")
    for unit, shares in ownership.items():
        if any(not 0.0 <= share <= 1.0 for share in shares.values()):
            raise ValueError(f"{unit}: ownership shares must lie in [0, 1].")
    totals: dict[str, float] = {}
    for shares in ownership.values():
        for generator, share in shares.items():
            totals[generator] = totals.get(generator, 0.0) + share
    oversold = {g: t for g, t in totals.items() if t > 1.0 + 1.0e-9}
    if oversold:
        raise ValueError(
            "Generation cannot be owned more than once; "
            f"over-allocated: { {g: round(t, 6) for g, t in sorted(oversold.items())} }"
        )


def three_investors(
    data: MarketData,
    *,
    ownership: Mapping[str, Mapping[str, float]] | None = None,
    quadratic_cost_power_eur_per_mw2: float = 0.0,
    quadratic_cost_energy_eur_per_mwh2: float = 0.0,
    quadratic_cost_power_per_node_eur_per_mw2: float = 0.0,
    quadratic_cost_energy_per_node_eur_per_mwh2: float = 0.0,
    quadratic_cost_power_by_node_eur_per_mw2: Mapping[str, float] | None = None,
    quadratic_cost_energy_by_node_eur_per_mwh2: Mapping[str, float] | None = None,
) -> tuple[InvestorConfig, ...]:
    """The maintained population: merchant, wind owner, and solar owner.

    All three investors use an 8% WACC, so their only economic asymmetry is the
    generation portfolio.  The default gives I2 all wind and I3 all PV.

    ``ownership`` overrides that, mapping investor id to {generator: share}.
    Ownership changes who captures price support from congestion relief, but it
    does not by itself prove either a unique split or a continuum.  Those are
    equilibrium results that must be established by unrestricted best responses
    under the selected market data.
    """

    nodal_cost_maps = (
        quadratic_cost_power_by_node_eur_per_mw2 or {},
        quadratic_cost_energy_by_node_eur_per_mwh2 or {},
    )
    unknown_cost_nodes = set().union(*(set(values) for values in nodal_cost_maps)) - set(
        data.nodes
    )
    if unknown_cost_nodes:
        raise ValueError(f"Unknown nodes in quadratic cost maps: {sorted(unknown_cost_nodes)}")
    if any(value < 0.0 for values in nodal_cost_maps for value in values.values()):
        raise ValueError("Node-specific quadratic cost coefficients cannot be negative.")

    if ownership is not None:
        validate_ownership(data, ownership)
        common = {
            "quadratic_cost_power_eur_per_mw2": quadratic_cost_power_eur_per_mw2,
            "quadratic_cost_energy_eur_per_mwh2": quadratic_cost_energy_eur_per_mwh2,
            "quadratic_cost_power_per_node_eur_per_mw2": quadratic_cost_power_per_node_eur_per_mw2,
            "quadratic_cost_energy_per_node_eur_per_mwh2": quadratic_cost_energy_per_node_eur_per_mwh2,
            "quadratic_cost_power_by_node_eur_per_mw2": dict(
                quadratic_cost_power_by_node_eur_per_mw2 or {}
            ),
            "quadratic_cost_energy_by_node_eur_per_mwh2": dict(
                quadratic_cost_energy_by_node_eur_per_mwh2 or {}
            ),
        }
        return tuple(
            InvestorConfig(
                unit,
                wacc=0.08,
                owned_generation_shares=dict(ownership.get(unit, {})),
                **common,
            )
            for unit in ("I1", "I2", "I3")
        )

    wind = [generator for generator in data.generators if "Wind" in generator]
    solar = [generator for generator in data.generators if "PV" in generator]
    if not wind or not solar:
        raise ValueError("The three-investor population requires wind and PV generation.")

    common = {
        "quadratic_cost_power_eur_per_mw2": quadratic_cost_power_eur_per_mw2,
        "quadratic_cost_energy_eur_per_mwh2": quadratic_cost_energy_eur_per_mwh2,
        "quadratic_cost_power_per_node_eur_per_mw2": quadratic_cost_power_per_node_eur_per_mw2,
        "quadratic_cost_energy_per_node_eur_per_mwh2": quadratic_cost_energy_per_node_eur_per_mwh2,
        "quadratic_cost_power_by_node_eur_per_mw2": dict(
            quadratic_cost_power_by_node_eur_per_mw2 or {}
        ),
        "quadratic_cost_energy_by_node_eur_per_mwh2": dict(
            quadratic_cost_energy_by_node_eur_per_mwh2 or {}
        ),
    }
    return (
        InvestorConfig("I1", wacc=0.08, **common),
        InvestorConfig(
            "I2",
            wacc=0.08,
            owned_generation_shares={generator: 1.0 for generator in wind},
            **common,
        ),
        InvestorConfig(
            "I3",
            wacc=0.08,
            owned_generation_shares={generator: 1.0 for generator in solar},
            **common,
        ),
    )
