"""Portfolio-strategic MPEC: storage prices plus owned-generation offers.

Extends the maintained relaxed-KKT hourly-price MPEC so the active investor
also chooses submitted offers ``OfferGeneration[g, t]`` for every generator it
owns outright. The ISO lower level clears against submitted offers while the
investor's generation profit keeps using true economic cost, separating the
offer role of ``generation_cost`` from its economic role:

- true cost (``data.generation_cost``): owner rent ``(LMP - cost) * dispatch``;
- submitted offer: ISO dispatch objective, generator stationarity, and the
  relaxed complementarity products.

Rival generation offers are fixed at the Jacobi snapshot. Strategic offers are
two-sided, ``-offer bound <= OfferGeneration[g, t] <= offer bound``: an owner
may offer above truthful (withholding, price support) or below truthful
(price suppression). The truthful renewable offer equals the true cost, which
is 0 EUR/MWh for wind and PV in the maintained experiment input. Offers exist
only for owned (g, t) pairs with positive availability; zero-availability
hours have no offer variable.
"""

from __future__ import annotations

from collections.abc import Mapping

import pyomo.environ as pyo

import mpec_strategic_price_relaxed_kkt as price_mpec
from investors import InvestorConfig
from primal_market_clearing_model import MarketData, effective_generation_offer


DEFAULT_GENERATION_OFFER_BOUND = 500.0


def owned_generators(investor: InvestorConfig) -> list[str]:
    """Generators the investor bids strategically: 100% owned units only."""

    owned = [g for g, share in investor.owned_generation_shares.items() if share > 0.0]
    partial = [
        g
        for g in owned
        if abs(investor.owned_generation_shares[g] - 1.0) > 1e-12
    ]
    if partial:
        raise ValueError(
            "Strategic generation offers require 100% unit ownership; "
            f"fractional shares found for {sorted(partial)} (use the owner-"
            "split input)."
        )
    return owned


def build_model(
    data: MarketData,
    *,
    investor: InvestorConfig,
    rival_generation_offer: Mapping[tuple[str, int], float] | None = None,
    initial_generation_offer: Mapping[tuple[str, int], float] | None = None,
    proximal_generation_offer: Mapping[tuple[str, int], float] | None = None,
    proximal_generation_penalty: float = 0.0,
    generation_offer_bound: float = DEFAULT_GENERATION_OFFER_BOUND,
    **kwargs: object,
) -> pyo.ConcreteModel:
    """Build one portfolio best response (storage prices + generation offers)."""

    bound = float(generation_offer_bound)
    generation_proximal = float(proximal_generation_penalty)
    if bound <= 0.0:
        raise ValueError("generation_offer_bound must be positive.")
    if generation_proximal < 0.0:
        raise ValueError("proximal_generation_penalty must be non-negative.")

    owned = owned_generators(investor)
    if any(abs(data.generation_cost[g]) > bound for g in owned):
        raise ValueError("A true cost lies outside the generation offer bounds.")

    model = price_mpec.build_model(data, investor=investor, **kwargs)
    model.name = f"Strategic-portfolio relaxed-KKT MPEC [{investor.investor_id}]"

    owned_pairs = [(g, t) for (g, t) in model.GT if g in owned]
    rival_offers_input = dict(rival_generation_offer or {})
    unknown_rival = {g for g, _ in rival_offers_input} & set(owned)
    if unknown_rival:
        raise ValueError(
            f"Rival generation offers include the active investor's units: {sorted(unknown_rival)}"
        )
    # Zero-availability hours carry no offer variable; ignore such keys so a
    # full 24-hour truthful profile is a valid initial/proximal centre.
    initial_offers = dict(initial_generation_offer or {})
    foreign_initial = {g for g, _ in initial_offers if g not in owned}
    if foreign_initial:
        raise ValueError(
            f"Initial generation offers for units not owned by the active investor: {sorted(foreign_initial)}"
        )

    model.GT_OWNED = pyo.Set(dimen=2, initialize=owned_pairs, ordered=True)
    model.OfferGeneration = pyo.Var(
        model.GT_OWNED,
        bounds=(-bound, bound),
        initialize=lambda _, g, t: min(
            bound,
            max(
                -bound,
                float(initial_offers.get((g, int(t)), data.generation_cost[g])),
            ),
        ),
    )

    fixed_offers: dict[tuple[str, int], float] = {}
    for g, t in model.GT:
        if g in owned:
            continue
        offer = float(
            rival_offers_input.get((g, int(t)), effective_generation_offer(data, g))
        )
        if not -bound <= offer <= bound:
            raise ValueError(f"Rival generation offer outside bounds for {g}, {t}.")
        fixed_offers[g, int(t)] = offer

    def submitted_offer(m: pyo.ConcreteModel, g: str, t: int):
        return (
            m.OfferGeneration[g, t] if g in owned else fixed_offers[g, int(t)]
        )

    # 1. Generator dual stationarity clears against submitted offers.
    model.del_component(model.gen_stationarity)
    model.gen_stationarity = pyo.Constraint(
        model.GT,
        rule=lambda m, g, t: sum(m.lam[n, t] for n in m._gen_nodes[g])
        + m.nu_gen[g, t]
        <= submitted_offer(m, g, int(t)),
    )

    # 2. Rebuild the generation complementarity product on submitted offers.
    epsilon = float(model._complementarity_epsilon)
    model.del_component(model.relaxed_comp_gen_lower)
    model.del_component(model.relaxed_comp_gen_lower_product)
    model.relaxed_comp_gen_lower_product = pyo.Expression(
        model.GT,
        rule=lambda m, g, t: m.P_gen[g, t]
        * (
            submitted_offer(m, g, int(t))
            - sum(m.lam[n, t] for n in m._gen_nodes[g])
            - m.nu_gen[g, t]
        ),
    )
    model.relaxed_comp_gen_lower = pyo.Constraint(
        model.GT,
        rule=lambda m, g, t: pyo.inequality(
            0.0, m.relaxed_comp_gen_lower_product[g, t], epsilon
        ),
    )

    # 3. The ISO primal objective values generation at submitted offers.
    model.primal_objective.set_value(
        model.primal_objective.expr
        + sum(
            (submitted_offer(model, g, int(t)) - effective_generation_offer(data, g))
            * model.P_gen[g, t]
            for g, t in model.GT
        )
    )

    # 4. Moving proximal term on the strategic generation offers.
    if generation_proximal > 0.0:
        if proximal_generation_offer is None:
            raise ValueError(
                "A positive generation proximal penalty requires offer centres."
            )
        model.generation_offer_regularizer = pyo.Expression(
            expr=0.5
            * generation_proximal
            * sum(
                (
                    model.OfferGeneration[g, t]
                    - float(proximal_generation_offer[g, int(t)])
                )
                ** 2
                for g, t in model.GT_OWNED
            )
        )
        model.regularizer.set_value(
            model.regularizer.expr + model.generation_offer_regularizer
        )
        model.profit.set_value(model.unregularized_profit - model.regularizer)
        model.objective.set_value(model.profit)
        model._proximal_generation_offer = dict(proximal_generation_offer)

    model._strategic_portfolio = True
    model._owned_generators = tuple(owned)
    model._generation_offer_bound = bound
    model._proximal_generation_penalty = generation_proximal
    model._fixed_generation_offers = fixed_offers
    return model


def submitted_offer_values(model: pyo.ConcreteModel) -> dict[tuple[str, int], float]:
    """Current submitted offers for every (g, t) in the lower level."""

    offers = dict(model._fixed_generation_offers)
    for g, t in model.GT_OWNED:
        offers[g, int(t)] = float(pyo.value(model.OfferGeneration[g, t]))
    return offers


def initialise_lower_level(model: pyo.ConcreteModel, data: MarketData) -> None:
    """Seed the MPEC from the exact ISO clearing at current submitted offers."""

    price_mpec.initialise_lower_level(
        model, data, generation_offers=submitted_offer_values(model)
    )
