# Project Context: Strategic BESS Investment in Nodal Spot Markets

Last updated: 2026-07-22

## Current objective

The project models strategic investment in battery energy storage systems (BESS)
in a deterministic nodal spot market. It compares decentralized investment by
competing investors with a central-planner benchmark.

The main research question is:

> How do strategic ownership, nodal grid-access limits, and network topology
> change BESS capacity, siting, ownership, prices, and investor rents relative to
> a non-strategic system optimum?

The active model is **spot-market only**. aFRR, reserve capacity, activation,
reserve prices, and stochastic reserve scenarios are outside the project scope.

## Active model

The model covers one representative 24-hour day with hourly resolution. Its main
components are:

- deterministic market clearing through a PTDF-based DC-OPF;
- conventional and renewable generation, curtailment, and nodal demand;
- nodal BESS charging, discharging, and periodic state-of-charge dynamics;
- endogenous BESS power (MW) and energy (MWh) investment;
- a 2-8 hour energy-to-power envelope;
- shared nodal BESS connection limits, configurable from a 100 MW default;
- annualized power and energy CAPEX using investor-specific WACC;
- physical degradation cost of 15 EUR/MWh of throughput.

The active empirical scope is the calibrated IEEE-9 congestion system. It is
designed to retain manageable evening prices, midday solar-export congestion at
N8, and a 100 MW shared BESS connection limit per node. The older 5-bus euro
system and earlier IEEE-9 calibrations remain in the repository as historical
cases but are not the current thesis benchmark.

## Models and solution methods

### Central planner

`model/central_planner_benchmark.py` jointly selects BESS capacities and market
dispatch by minimizing:

> generation cost + curtailment cost + storage CAPEX + degradation cost.

It is a convex QP and represents the first-best efficiency benchmark. Ownership,
market transfers, and generator rents do not enter its objective.

### Single-investor MPEC

`model/single_investor_mpec.py` lets one strategic investor choose nodal BESS MW
and MWh while anticipating the spot-market response through primal feasibility,
dual feasibility, stationarity, and Wolfe strong duality. This is equivalent to
the lower-level KKT system for the convex market-clearing problem, with strong
duality replacing explicit complementarity equations. The investor earns storage
spot revenue and, if portfolio-backed, its share of existing generator rent, net
of degradation and CAPEX.

The lower-level objective and the independent reference market use the same
storage-degradation cost. No artificial dispatch regularizer is included.

### Multi-investor EPEC

`model/epec_diagonalization.py` couples investor MPECs through shared nodal
connection limits and market outcomes. It supports Gauss-Jacobi and Gauss-Seidel
updates with damping and feasibility projection.

The maintained algorithm uses private rival-headroom bounds and a final
shared-limit projection safeguard. Checkpoint/resume support persists MW and MWh
strategies after every completed iteration. Experimental access-price code was
removed after its projected price iterations failed to converge.

The later two-follower access-auction MPEC experiment, its Gauss-Seidel driver,
bid profiles, and diagnostic outputs are archived under `model/auction/`. They
are not part of the maintained spot-market EPEC workflow.

The main four-investor specification is:

- I1: stand-alone merchant BESS, 8% WACC;
- I2: stand-alone merchant BESS, 12% WACC;
- I3: wind-heavy renewable portfolio, 8% WACC;
- I4: solar-heavy renewable portfolio, 8% WACC.

## Current empirical result

For the active IEEE-9 congestion case with a 100 MW nodal connection limit:

- the central planner installs **238.924 MW / 826.862 MWh**;
- the original four-investor projection EPEC installs **239.004 MW /
  827.044 MWh** and converges in seven Gauss-Seidel iterations;
- a cross-machine verification run installs **238.939 MW / 826.9 MWh** and
  converges in eight iterations;
- both the planner and EPEC place 100 MW at N8, while the remaining small
  planner-EPEC siting difference is mainly a shift between N3 and N9;
- approximate EPEC social cost, revalued at the planner's 8% WACC, is about
  0.134% above the planner cost.

The close aggregate match is an observed result for this calibration, not a
mathematical proof that strategic behavior or dual selection cannot change
aggregate investment. Strategic ownership still materially changes rent
allocation, and fine siting remains weakly identified where nodes have similar
price profiles.

The previously reported approximately 326.6 MW equality belongs to earlier
network calibrations. It is a historical sensitivity result, not the headline
quantity for the active IEEE-9 congestion case.

## Dual-price nonuniqueness

The current lower-level formulation can have a unique or effectively consistent
primal dispatch while admitting multiple valid dual price vectors. The MPEC uses
an optimistic convention and can select prices favorable to the investor. Those
prices can differ materially from the prices returned by the independent joint
market reclear.

In the active IEEE-9 four-investor projection result, the maximum MPEC-versus-
joint-settlement LMP difference is about 44.57 EUR/MWh. Merchant investors I1
and I2 have last-iteration optimistic MPEC profits of about 13.5 and 5.7
kEUR/day, respectively, but joint-settlement profits of about -0.04 and -0.30
kEUR/day. Consequently, the current EPEC result should be presented as an
**optimistic-equilibrium convention or candidate equilibrium**, not as a
unique-price or fully verified equilibrium. Fine nodal allocation and investor
rents are less firmly identified than aggregate capacity.

## Discarded nodal access-price experiment

Both sweep-level and investor-level projected access-price updates were tested.
Neither eliminated the discontinuous multi-node capacity cycle. The final
investor-level run was stopped after iteration 19 with N8 at 132.06 MW, an N8
price of 74.49 EUR/MW/day, and a 32.06 MW overload/residual; all completed MPEC
solves in that run were optimal. The failure was therefore algorithmic rather
than attributable to solver termination.

These saved runs are diagnostic and must not be reported as equilibria, clearing
access prices, or settled investor profits. The maintained code no longer
contains access-price variables, payments, price updates, or related CLI flags.
If nodal access allocation is revisited, use a separate explicit merit-order
auction design rather than extending the discarded tâtonnement code.

## Chosen thesis direction: expose the dual face

The project will not make implementation of a unique price-selection rule a
prerequisite for using the present EPEC results. Instead, it will disclose and
quantify the price ambiguity.

For fixed final capacities and primal dispatch, characterize the feasible dual
face using lower-level stationarity, dual feasibility, and strong duality. For
each of the four investors, calculate:

- minimum feasible revenue/profit over the dual face;
- maximum feasible revenue/profit over the dual face;
- the optimistic MPEC value;
- the joint-settlement/reference-market value.

The main figure should be an investor-wise interval plot showing the feasible
revenue or profit range, with markers for optimistic MPEC and joint settlement.
Because separately optimized minima and maxima need not occur under the same
price vector, the figure and text must label them as **investor-wise bounds**.
If practical, sample common feasible dual vectors to illustrate the joint revenue
distribution without implying that all investor-wise extremes are simultaneous.

The intended thesis conclusion is:

> Aggregate BESS investment is approximately identical in the tested planner and
> four-investor EPEC cases, whereas strategic behavior changes siting and
> ownership. Because market-clearing prices lie on a non-singleton dual face, the
> allocation of rents and potentially fine nodal siting are not uniquely
> identified. The reported EPEC adopts the optimistic dual-selection convention.

This framing permits use of the result while being explicit about what the model
does and does not identify.

## Validation and interpretation rules

- Do not call the 326.6 MW result a proof; call it an observed robust result for
  the tested cases.
- Distinguish physical dispatch consistency from price and rent uniqueness.
- Report both optimistic MPEC profit and joint-settlement profit.
- Treat small relocation among nodes with nearly identical LMP profiles as
  economic indifference, not necessarily failed aggregate convergence.
- State when an EPEC run reaches its iteration limit without formal convergence.
- Do not interpret a dual-face revenue interval as an equilibrium range unless
  investment optimality is also established for the corresponding price rule.
- If feasible, check whether profitability and marginal investment incentives
  retain their signs across adverse dual-face prices. This would strengthen the
  claim that dual ambiguity reallocates rents more than aggregate capacity.

## Current priorities

1. Implement the dual-face revenue/profit bounds for the converged IEEE-9
   projection EPEC, using a standalone/reference-market check first if useful.
2. Produce the four-investor interval graph with optimistic and joint-settlement
   markers.
3. Prepare a like-for-like 9-bus planner-versus-EPEC comparison at the same nodal
   limit, emphasizing aggregate quantity, siting, prices, and system cost.
4. Add the benchmark and dual-nonuniqueness interpretation to
   `Overleaf_Alex/model_extension.tex`.
5. If nodal access allocation is revisited, formulate it as a separate explicit
   quantity-and-price bid auction with deterministic merit-order clearing.
6. Keep a unique neutral electricity-pricing rule, such as minimum-norm dual
   pricing, as a possible robustness extension if the dual-face ranges undermine
   the intended interpretation; it is not the immediate primary task.
7. Return to broader nodal-access auction design only after the benchmark and
   dual-face analysis are documented.

## Important current caveats

- The 5-bus results and the earlier approximately 326.6 MW result are historical
  sensitivities, not the active empirical benchmark.
- The saved IEEE-9 four-investor projection run converged with zero projection
  events; its small cross-machine siting variation was mainly relocation across
  near-price-equivalent nodes.
- The discarded access-price runs are infeasible and nonconverged, with
  persistent multi-node cycling; their output folders are diagnostic history.
- Loose nodal limits can produce overinvestment and negative settled merchant
  profit even when the optimistic MPEC reports positive incentives.
- The central planner and projection EPEC approximately agree on aggregate MW in
  the active IEEE-9 case but
  differ in siting, ownership, prices, and rent allocation.
