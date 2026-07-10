"""Sealed pay-as-bid nodal BESS connection-capacity auction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Bid:
    investor: str
    node: str
    quantity_mw: float
    price_eur_per_mw: float


@dataclass(frozen=True)
class AcceptedBid:
    investor: str
    node: str
    accepted_mw: float
    price_eur_per_mw: float

    @property
    def payment_eur(self) -> float:
        return self.accepted_mw * self.price_eur_per_mw


@dataclass(frozen=True)
class AuctionResult:
    accepted_bids: tuple[AcceptedBid, ...]
    allocations_mw: Mapping[tuple[str, str], float]
    payments_eur: Mapping[str, float]
    node_utilization_mw: Mapping[str, float]


def clear_pay_as_bid_auction(
    bids: Iterable[Bid],
    node_limits_mw: Mapping[str, float],
) -> AuctionResult:
    """Clear independent nodal pay-as-bid auctions.

    Bids are ranked by descending price at each node. The final accepted bid may
    be partially filled if less capacity remains than the requested quantity.
    """

    bids_by_node: dict[str, list[Bid]] = {node: [] for node in node_limits_mw}
    for bid in bids:
        if bid.node not in node_limits_mw:
            raise ValueError(f"Bid references unknown node {bid.node!r}.")
        if bid.quantity_mw < 0.0:
            raise ValueError("Bid quantity must be nonnegative.")
        if bid.price_eur_per_mw < 0.0:
            raise ValueError("Bid price must be nonnegative.")
        if bid.quantity_mw > 0.0 and bid.price_eur_per_mw > 0.0:
            bids_by_node[bid.node].append(bid)

    accepted: list[AcceptedBid] = []
    allocations: dict[tuple[str, str], float] = {}
    payments: dict[str, float] = {}
    utilization: dict[str, float] = {}

    for node, limit in node_limits_mw.items():
        remaining = float(limit)
        ranked_bids = sorted(
            bids_by_node[node],
            key=lambda b: (-b.price_eur_per_mw, b.investor),
        )

        price_groups: list[list[Bid]] = []
        for bid in ranked_bids:
            if not price_groups or abs(price_groups[-1][0].price_eur_per_mw - bid.price_eur_per_mw) > 1e-9:
                price_groups.append([bid])
            else:
                price_groups[-1].append(bid)

        for group in price_groups:
            if remaining <= 1e-9:
                break
            group_quantity = sum(bid.quantity_mw for bid in group)
            if group_quantity <= remaining + 1e-9:
                acceptance_ratio = 1.0
            else:
                acceptance_ratio = remaining / group_quantity

            for bid in group:
                accepted_mw = bid.quantity_mw * acceptance_ratio
                if accepted_mw <= 0.0:
                    continue

                accepted_bid = AcceptedBid(
                    investor=bid.investor,
                    node=node,
                    accepted_mw=accepted_mw,
                    price_eur_per_mw=bid.price_eur_per_mw,
                )
                accepted.append(accepted_bid)
                allocations[(bid.investor, node)] = allocations.get((bid.investor, node), 0.0) + accepted_mw
                payments[bid.investor] = payments.get(bid.investor, 0.0) + accepted_bid.payment_eur

            remaining -= min(group_quantity, remaining)

        utilization[node] = float(limit) - max(remaining, 0.0)

    return AuctionResult(
        accepted_bids=tuple(accepted),
        allocations_mw=allocations,
        payments_eur=payments,
        node_utilization_mw=utilization,
    )
