# Clean strategic nodal-access auction EPEC

This folder contains the experimental two-follower auction EPEC. For each best
response, one BESS investor is the leader and chooses continuous nodal access
bid quantities, pay-as-bid prices, and awarded energy capacity. The model embeds
exactly two followers:

1. the nodal access-auction LP through primal feasibility, dual feasibility,
   and strong duality; and
2. the fixed-demand electricity spot-market LP through primal feasibility,
   dual feasibility/stationarity, and strong duality.

The leader maximizes storage spot revenue plus any owned-generator rent minus
degradation, annualized BESS CAPEX, and its pay-as-bid access payment.

The multi-investor driver uses the same four thesis investors as the normal
EPEC: merchant investors I1/I2 and the wind-heavy/solar-heavy portfolio
investors I3/I4. It applies their best responses immediately in Gauss--Seidel
order. After the final sweep it independently re-clears the auction once and
clears one common spot market containing all four awarded fleets. Consequently,
the consolidated output distinguishes each investor's last optimistic
best-response profit from profit under the final common settlement.

The core MPEC has no dispatch regularization, load shedding, objective penalty,
damping, or outside-option solve. Bid prices can either remain continuous or be
selected on an exact external grid. With `--bid-price-tick 0.01`, the driver
enumerates zero, each rival price, and one tick above each rival, solves one MPEC
per candidate, rejects candidates that fail the independent auction reclear or
strong-duality checks, and selects the highest-profit valid response. A solver
error for one price is stored in that candidate's diagnostics and does not abort
the remaining price candidates.

Equal raw grid bids use a deterministic `0.001` EUR/MW/day merit offset ordered
by investor ID. The offset affects auction ranking only; access payment remains
the raw submitted bid. Since the largest four-investor priority gap is `0.003`,
it is smaller than one `0.01` tick and can never reverse two different raw bids.

## Maintained numerical formulation

- electricity-price variables `lam` and `lam_sys`: `[-500, 500]` EUR/MWh;
- every other follower dual, including auction duals: absolute bound 10,000
  with the appropriate sign restriction;
- Ipopt `tol=acceptable_tol=1e-4` and sparse MUMPS;
- fixed demand and zero dispatch regularization;
- active investor represented at every enabled node;
- rival auction/storage blocks created only for bid quantities above `1e-4` MW;
- generator-hour blocks created only for strictly positive available capacity;
- bid quantities at or below `1e-4` MW normalized to exact zero between
  Gauss--Seidel responses.
- optional exact bid tick `0.01` EUR/MW/day with a `0.001` sub-tick merit
  priority for unique ranking.
- convergence requires bid quantity, raw bid price, duration, and auction award
  changes all to satisfy their respective tolerances;
- the default raw-price tolerance is `0.001`, below the `0.01` grid step, so a
  one-tick strategic price movement cannot be reported as convergence.

The active and rival batteries remain separate storage units in the embedded
spot market. Rivals are never collapsed into a virtual aggregate battery.

## Important interpretation

In continuous mode (`--bid-price-tick 0`), equal bids can still leave auction
awards non-unique and the MPEC remains optimistic. In tick mode, the sub-tick
priority gives every raw-price tie a unique ranking. For the two-investor N8
case, I1 wins at the tied raw price of 30.00 because it has higher deterministic
priority; I2 must bid 30.01 to outrank I1.

Continuous pay-as-bid competition can also have a best-response discontinuity:
an investor may prefer an arbitrarily small increment above a rival bid without
an attained optimum. Treat an Ipopt result as a local optimistic candidate, not
as proof of a pure-strategy equilibrium.

## Single best response

From the repository root:

```powershell
python model\auction\single_investor_auction_mpec.py `
  --active-investor I1 `
  --active-node N8 `
  --rival-bids model\auction\data\auction_mpec_cases\two_investor_n8_uniform.json `
  --bid-price-tick 0.01 `
  --tie-break-epsilon 0.001 `
  --initial-bid-quantity 70 `
  --initial-bid-price 30 `
  --initial-duration 4 `
  --max-cpu-time 120 `
  --output model\auction\output\single_investor_auction_mpec\clean_I1_N8.json
```

Omit `--active-node` to allow the active investor to bid at all IEEE-9 nodes.

## Four-investor auction EPEC

`--update-rule jacobi` represents simultaneous sealed-bid proposals in the
diagonalization algorithm: all four MPECs see the same pre-sweep rival-bid
snapshot, their proposed bids are applied together, and one common auction
assigns awards. `--update-rule seidel` instead applies each response immediately.
Only a converged fixed point is interpreted as an EPEC solution; intermediate
iteration snapshots are numerical conjectures, not public bid disclosure.

Run the continuous-pay-as-bid Jacobi experiment from zero bids:

```powershell
python model\auction\gauss_seidel.py `
  --initial-bids model\auction\data\auction_mpec_cases\zero_competition.json `
  --investor-order I1 I2 I3 I4 `
  --update-rule jacobi `
  --active-nodes N8 `
  --bid-price-tick 0 `
  --zero-bid-numerical-quantity 70 `
  --zero-bid-numerical-price 0.01 `
  --price-tol 0.001 `
  --quantity-tol 0.05 `
  --award-tol 0.05 `
  --duration-tol 0.01 `
  --max-iterations 3 `
  --max-cpu-time 120 `
  --output-dir model\auction\output\gauss_seidel\portfolio4_N8_zero_continuous_jacobi_3iters
```

Run the four-investor tick-grid EPEC at N8 from the balanced bid profile:

```powershell
python model\auction\gauss_seidel.py `
  --initial-bids model\auction\data\auction_mpec_cases\balanced_competition.json `
  --active-nodes N8 `
  --bid-price-tick 0.01 `
  --tie-break-epsilon 0.001 `
  --price-tol 0.001 `
  --award-tol 0.05 `
  --max-iterations 20 `
  --max-cpu-time 120 `
  --output-dir model\auction\output\gauss_seidel\portfolio4_N8_tick001
```

For a neutral start in which every investor submits `(0 MW, 0 EUR/MW/day)` at
every node, replace the initial-bid path with
`model\auction\data\auction_mpec_cases\zero_competition.json`. The stored
4-hour duration is only a valid numerical energy-ratio seed; initial awarded MW
and MWh remain zero. When an economic bid quantity is zero, the NLP variable is
initialized at 70 MW by default (`--zero-bid-numerical-quantity 70`). This is a
warm start only: it neither changes the zero bid state nor imposes a positive
lower bound. The analogous numerical price guess is `0.01` EUR/MW/day. Both
avoid starting the auxiliary auction/spot warm start from an infeasible
zero-storage dispatch while preserving the economic `(0,0)` state.

Continue a checkpoint by setting `--resume` and increasing `--max-iterations`
to the desired total iteration number. The output contains `run_config.json`,
`iteration_history.csv`, a checkpoint after every completed sweep, one JSON
summary per investor response, `final_bids_and_awards.csv`,
`joint_node_hour_prices.csv`, `joint_storage_hour_operation.csv`,
`joint_settlement.json`, and a consolidated `summary.json`.

A run that stops at `max_iterations` is a feasible diagnostic iterate, not a
computed equilibrium. The final common re-clear is still useful for checking
auction feasibility, common electricity prices, access payments, and settled
profits, but it does not turn a nonconverged bidding path into an equilibrium.

Exact enumeration currently requires exactly one active node. Joint enumeration
across nine nodal prices would require the Cartesian product of all candidate
ticks and is intentionally not approximated by rounding independent continuous
solutions.
