# Project Context: Strategic BESS Investment in Nodal Spot Markets

Last updated: 2026-07-20

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
- physical degradation cost of 15 EUR/MWh of throughput;
- a small common dispatch regularizer, currently `1e-4 EUR/MW2h`, used as a
  neutral primal-dispatch tie-break.

The processed datasets include a 5-bus euro system and a 9-bus system. They use
the same broad system-wide demand and generation quantities but different network
topologies and nodal placement.

## Models and solution methods

### Central planner

`model/central_planner_benchmark.py` jointly selects BESS capacities and market
dispatch by minimizing:

> generation cost + curtailment cost + storage CAPEX + degradation cost.

It is a convex QP and represents the first-best efficiency benchmark. Ownership,
market transfers, and generator rents do not enter its objective.

### Single-investor MPEC

`model/single_investor_mpec.py` lets one strategic investor choose nodal BESS MW
and MWh while anticipating the spot-market response through the lower-level KKT
conditions and Wolfe strong duality. The investor earns storage spot revenue and,
if portfolio-backed, its share of existing generator rent, net of degradation and
CAPEX.

The lower-level objective and the independent reference-market QP both include
the same storage degradation and primal dispatch regularization.

### Multi-investor EPEC

`model/epec_diagonalization.py` couples investor MPECs through shared nodal
connection limits and market outcomes. It supports Gauss-Jacobi and Gauss-Seidel
updates with damping and feasibility projection.

The main four-investor specification is:

- I1: stand-alone merchant BESS, 8% WACC;
- I2: stand-alone merchant BESS, 12% WACC;
- I3: wind-heavy renewable portfolio, 8% WACC;
- I4: solar-heavy renewable portfolio, 8% WACC.

## Current empirical result

Total installed BESS power is approximately **326.6 MW** in the tested cases:

- 9-bus central planner at 100 MW and 200 MW nodal limits;
- 9-bus four-investor EPEC at the 100 MW nodal limit;
- 5-bus four-investor EPEC at the 200 MW nodal limit.

The current interpretation is that aggregate storage quantity is mainly pinned by
system-wide arbitrage economics. Strategic ownership, topology, and nodal limits
primarily change **where capacity is installed, who owns it, and how rents are
distributed**. In the planner runs, relaxing the nodal limit changes siting but
leaves total capacity and total cost unchanged.

This equality is an important empirical finding, but it is **not a mathematical
proof** that strategic behavior or dual selection can never change aggregate
capacity. Results must be described as applying to the tested systems and
parameters.

## Dual-price nonuniqueness

The current lower-level formulation can have a unique or effectively consistent
primal dispatch while admitting multiple valid dual price vectors. The MPEC uses
an optimistic convention and can select prices favorable to the investor. Those
prices can differ materially from the prices returned by the independent joint
market reclear.

In the standalone test with degradation and regularization:

- investment was 49.3 MW;
- MPEC and reference primal-dispatch profits agreed when evaluated at the same
  reference prices;
- the maximum MPEC-reference LMP difference remained about 16.25 EUR/MWh;
- the optimistic-reference profit gap remained about 1.36 kEUR/day.

The four-investor results also contain material optimistic-versus-settlement
profit differences. Consequently, the current EPEC result should be presented as
an **optimistic-equilibrium convention or candidate equilibrium**, not as a
unique-price or fully verified equilibrium. Fine nodal allocation and investor
rents are less firmly identified than aggregate capacity.

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

1. Implement the dual-face revenue/profit bounds first in the standalone MPEC or
   reference-market workflow, then apply them to the four-investor result.
2. Produce the four-investor interval graph with optimistic and joint-settlement
   markers.
3. Prepare a like-for-like 9-bus planner-versus-EPEC comparison at the same nodal
   limit, emphasizing aggregate quantity, siting, prices, and system cost.
4. Add the benchmark and dual-nonuniqueness interpretation to
   `Overleaf_Alex/model_extension.tex`.
5. Keep a unique neutral pricing rule, such as minimum-norm dual pricing, as a
   possible robustness extension if the dual-face ranges undermine the intended
   interpretation; it is not the immediate primary task.
6. Return to nodal-access auction design only after the benchmark and dual-face
   analysis are documented.

## Important current caveats

- The 5-bus 200 MW EPEC run reached its 45-iteration limit without formal
  convergence, although aggregate MW and MWh were stable.
- The saved 9-bus four-investor run had zero projection events; its remaining
  energy movement was mainly relocation across near-price-equivalent nodes.
- Loose nodal limits can produce overinvestment and negative settled merchant
  profit even when the optimistic MPEC reports positive incentives.
- The central planner and EPEC agree on aggregate MW in the current cases but
  differ in siting, ownership, prices, and rent allocation.
