# Project context: clean capacity and hourly-bid games

Last updated: 2026-09-01

## Objective

Study strategic BESS investment and hourly market bidding in the IEEE-9 market
without an access auction or strategic quantity withholding. The baseline
strategic variables are nodal power capacity `X_power` and energy capacity
`X_energy`. The joint game chooses those investments together with a charging
buy bid and discharge sell offer for every investor, node, and hour.

## Timing and market

Four investors simultaneously choose their capacities. Conditional on the
capacity profile, the ISO competitively dispatches generation and storage over
24 hours subject to nodal balance, PTDF transmission limits, generator limits,
storage power limits, SOC dynamics, energy limits, and cyclic SOC.

Investor profit contains storage LMP settlement, owned renewable-generation
rent, degradation cost, and annualised power/energy investment cost.
Renewable curtailment is reported as a physical diagnostic but carries no
additional owner-level penalty beyond the market revenue lost on energy that
is not dispatched.

## Capacity constraints

Duration remains continuously endogenous between two and eight hours:

```text
2 * X_power[i,n] <= X_energy[i,n] <= 8 * X_power[i,n]
```

The model retains `sum_i X_power[i,n] <= node_limit_mw` only as a simple
physical upper bound. The baseline value is 1000 MW at every node so it should
not influence the result. Nodal utilisation is reported in `summary.json`.

## Solution and verification

Each capacity best response defaults to a Scholtes-relaxed KKT MPEC solved by
IPOPT; exact strong duality is retained as a cross-check. A damped Jacobi
iteration applies simultaneous best responses against a frozen common profile.
Convergence is judged from raw power and energy best-response deviations for
consecutive sweeps. A quadratic ISO demand-adjustment penalty selects a unique
market price; it is not a strategic variable and is reported in the market
output.

The final common profile is independently cleared with the exact competitive
market. Every investor is then solved once more against that same frozen
profile, and each deviation is independently recleared. Embedded/recleared
profit gaps and exact profitable deviations are exported.

IPOPT reports local NLP solutions; a converged Jacobi fixed point is not a
proof of global best-response optimality. Multistart checks remain necessary
before making a strong equilibrium claim.

## Joint investment and hourly bid/offer game

The ISO can dispatch up to the full installed inverter capacity. Investors do
not choose offered MW. Each price is bounded by default to +/-500 EUR/MWh and
the pair satisfies

```text
OfferDischarge[i,n,t] >= BidCharge[i,n,t] / eta^2
```

to rule out a negative-cost same-hour storage loop. In the joint game all
node-hour prices remain strategic even at nodes with little current capacity,
so an investor can enter a new node and choose its prices in the same best
response. The default fresh state is 5 MW and 15 MWh per investor and node,
with truthful initial prices. The old fixed-capacity diagnostic remains
available only through the explicit `--fixed-capacity` option.

Every run writes two capacity trajectories. The long nodal file stores MW,
MWh, duration, nodal total MW, and each investor's nodal share for every sweep.
The investor-total file stores portfolio MW, MWh, duration, and system power
share. Sweep 0 records the starting profile and later sweeps record the damped
state actually carried forward. The joint game also writes every hourly price
profile to `bids_by_investor_node_hour_by_sweep.csv`.

## Moving quadratic regularization

Every strategic variable has a squared-deviation term centred on its value in
the preceding Jacobi sweep. The capacity term covers power MW and continuously
chosen energy MWh; energy deviations are divided by the configured duration
scale before squaring. The operational term covers both hourly price profiles.
This is an algorithmic proximal term and must be reported separately from the
unregularized economic profit. A final equilibrium claim should be checked as
the penalty is reduced.

## Maintained files

- `model/mpec_strong_duality.py`: clean capacity MPEC core.
- `model/mpec_relaxed_kkt.py`: default relaxed-complementarity embedding.
- `model/jacobi_diagonalization.py`: four-player capacity iteration.
- `model/run_model.py`: capacity-game executable and audit/output driver.
- `model/mpec_strategic_operation.py`: hourly bid/offer market embedding.
- `model/mpec_strategic_price_relaxed_kkt.py`: relaxed-KKT hourly-price MPEC.
- `model/run_hourly_bid_game.py`: joint investment and hourly bid/offer iteration.
- `model/primal_market_clearing_model.py`: exact fixed-capacity ISO model.
- `model/input/market_data.json`: IEEE-9 data.
