# Project Context: Strategic BESS Investment in Nodal Spot Markets

Last updated: 2026-08-03

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
- conventional and renewable generation, renewable curtailment, and fixed nodal demand;
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

The current candidate distributed-congestion calibration restores the three
IEEE-9 synchronous machines at N1, N2, and N3. The restored N2 unit has 300 MW
capacity and a 52.20 EUR/MWh marginal cost; new runs do not add the former
artificial N5 peaker by default. The N6 export lines L46/L69 are limited to
180 MW and the N8 export lines L98/L78 to 210 MW. With a 200 MW nodal BESS
limit, a planner smoke test places 53.39 MW at N6 and 65.63 MW at N8, confirming two distinct renewable
congestion-relief locations. This is a candidate calibration pending a
converged EPEC rerun; older saved results retain their historical input and
peaker calibration as documented in their run configurations.

## Models and solution methods

### Central planner

`model/central_planner_benchmark.py` jointly selects BESS capacities and market
dispatch. In the maintained fixed-demand, zero-regularization base it minimizes:

> generation cost + storage CAPEX + degradation cost.

It is a convex LP in the base configuration and represents the first-best
efficiency benchmark. Quadratic load curtailment and dispatch regularization are
explicit robustness options. Ownership, market transfers, and generator rents
do not enter its objective.

### Single-investor MPEC

`model/single_investor_mpec.py` lets one strategic investor choose nodal BESS MW
and MWh while anticipating the spot-market response through primal feasibility,
dual feasibility, and Wolfe strong duality. For the convex market-clearing
problem these primal-dual optimality conditions characterize a lower-level
optimum without adding explicit complementarity equations. The investor earns storage
spot revenue and, if portfolio-backed, its share of existing generator rent, net
of degradation and CAPEX.

The maintained base configuration uses fixed demand, the same storage-
degradation cost in the embedded and reference markets, and zero artificial
dispatch regularization. Quadratic load curtailment and a nonzero quadratic
dispatch tie-break remain explicit robustness options rather than hidden
defaults. Ipopt uses `tol=1e-4` and `acceptable_tol=1e-4` in the normal EPEC.
The maintained electricity-price bounds for `lam` and `lam_sys` are the
explicit interval `[-500, 500]` EUR/MWh. All other embedded lower-level duals
use the wider absolute interval `[-10,000, 10,000]`. These are explicit
numerical/price-selection bounds rather than multiples of VOLL. Fixed-
demand shedding variables and their dual conditions are omitted. The embedded
primal-dual model also omits zero-capacity rival node blocks and zero-capacity
generator-hours; the active investor remains represented at every node. Ipopt
uses its sparse MUMPS linear solver through the common solver configuration.
Capacity pairs at or below `1e-4` in both MW and MWh are normalized to exact
zero between diagonalization updates so economically absent batteries remain
sparse.

An experimental alternative in `model/single_investor_mpec_relaxed_kkt.py`
deactivates strong duality and imposes individual Scholtes-style relaxed KKT
products `slack * dual <= epsilon_comp`. The capacity diagonalization driver
selects it explicitly with `--lower-level-optimality relaxed-kkt`; it is not the
maintained default. Each run records the maximum and summed complementarity
products because a small per-pair epsilon can accumulate across thousands of
conditions.

The experimental formulation in `model/tikhonov_kkt/` is the exact finite-gamma
strong-duality approach. It pairs the directly regularized dual with the matching primal objective
`C(x) + sum(h^2)/(2*gamma)` and enforces primal feasibility, dual feasibility,
`h + gamma*lambda = 0`, and strong duality without complementarity-product
constraints or a Scholtes epsilon. At `gamma=1e-3`, the single-investor model
solved optimally at 187.182153 MW / 547.213300 MWh and 18,819.57 EUR/day; an
independent same-fleet soft-market audit matched prices within 0.019 EUR/MWh.
This removes finite-epsilon price freedom but not the physical imbalance of a
finite-gamma soft market. The superseded relaxed-KKT/Scholtes scripts are
archived under `model/tikhonov_kkt/old/` and are not used by new runs.

`model/tikhonov_kkt/jacobi_epec.py` embeds that exact strong-duality MPEC in an
isolated four-investor driver. It implements explicit simultaneous
Gauss--Jacobi sweeps:
all best responses use one frozen capacity snapshot, successful proposals are
damped together, and only then is the shared nodal limit projected. Top-level
controls include parallel workers and an optional sequence of
`(gamma, max sweeps)` continuation stages. The default damping is 0.25, chosen
after a smoke test showed a sharp I1 best-response change between quarter- and
half-applied first-sweep fleets.
Convergence requires both damped iterate changes and raw undamped best-response
residuals below tolerance, preventing a small damping factor from creating a
false convergence declaration. Per-investor history records matched
strong-duality gaps and original nodal/system balance residuals; final
capacities are settled once in the exact unregularized market. This
experimental driver does not make a finite-gamma market physically exact.

`model/tikhonov_kkt/epec.py` is the unified finite-gamma entry point. Its
`capacity` mode runs the capacity-only Jacobi EPEC, while
`strategic-operation` mode adds hourly charge/discharge quantity offers and,
by default, two-sided bid prices using the matched strategic soft-market MPEC.
Both modes support independent Jacobi best responses with multiple worker
processes; the maintained default is two workers. Both apply the same delayed
upper-level proximal continuation: iterations 1-10 are unpenalized, followed
by 1 EUR/MW^2/day increments every five iterations. Capacity mode measures
MW/MWh movement; strategic-operation mode additionally measures changes in
withheld quantities and normalized bid prices.
Fresh Tikhonov strategic-operation runs now enter iteration 1 from zero
economic MW/MWh and zero quantity offers at every node. The separate 10 MW/node
four-hour MPEC numerical guess remains positive and does not enter the economic
state. The older projected Jacobi initializer is available only through an
explicit runner flag.

The separate experimental builder
`model/single_investor_mpec_min_norm_prices.py` retains exact primary strong
duality and adds an ISO-owned secondary convex QP that minimizes
`0.5 * sum(lambda[n,t]^2)` over the primary dual-optimal set. Its embedded KKT
conditions are selected with `--lower-level-optimality iso-min-norm-dual`.
Unlike a leader-objective lambda penalty, this makes the common price-selection
rule part of the market design rather than an investor preference.
The normal capacity EPEC now exposes this experimental rule explicitly through
`--price-selection iso-min-norm`; the default remains the clean optimistic
strong-duality baseline. Fresh Jacobi tests can also separate a zero economic
fleet from the positive Ipopt numerical guess with
`--economic-seed-power-mw 0`. On the distributed-congestion IEEE-9 case, one
exact single-investor best response solved at 187.11 MW / 546.95 MWh with zero
hard-balance residual, a 0.306 EUR/MWh maximum embedded-versus-joint LMP gap,
and an 85.02 EUR/day profit gap. A sparse four-investor one-sweep smoke test
solved all four minimum-norm MPECs at `tol=1e-4`; it remains a non-equilibrium
diagnostic because the saved fleet applies 0.25 damping to simultaneous raw
best responses.
The optional `--iso-min-norm-complementarity-epsilon` relaxes only the
nonnegative aggregate of the secondary ISO KKT complementarity products; zero
remains the exact default, while primary hard-market feasibility, primary
strong duality, and secondary stationarity stay exact. A `1e-3` three-sweep
trial completed the first sweep but was aborted during the second after the
active-rival best responses again approached the 300-second limits. Its
iteration-1 checkpoint is a diagnostic, not an equilibrium. The relaxation
also changed I3's raw first response from about 203.2 MW to 187.0 MW, indicating
material local-solution sensitivity despite the small aggregate tolerance.

### Multi-investor EPEC

`model/epec_diagonalization.py` couples investor MPECs through shared nodal
connection limits and market outcomes. Every active investor sees each rival as
a separate lower-level battery with that rival's own nodal MW and MWh; rivals
are not collapsed into one virtual storage unit. It supports Gauss-Jacobi and
Gauss-Seidel updates with damping and feasibility projection.

The normal command-line entry point is the clean baseline: fixed demand, exact
hard nodal and system balances, exact lower-level strong duality, zero dispatch
regularization, zero leader-side price penalty, and no Tikhonov gamma or demand-
expansion block. Its output is labeled
`clean-fixed-demand-exact-strong-duality`. Alternative lower-level price rules
remain experimental and are invoked from their dedicated drivers; in
particular, the finite-gamma and elastic-demand work stays under
`model/tikhonov_kkt/` rather than defining the maintained outer EPEC.

A 221-case exact fixed-demand post-hoc screen in
`model/output/capacity_discontinuity_diagnostics_2026-08-05/` confirms that the
capacity discontinuity is a coupled N6/N8 curtailment-elimination frontier, not
an artifact of holding N6 fixed. Along the saved fleet's N6/N8 capacity ray,
moving from 0.75 to 1.00 of the fleet eliminates about 121 MWh/day of remaining
PV curtailment and changes the aggregate N8 average net margin from about
+88.95 to -17.71 EUR/MW/day. The joint grid shifts this frontier as N6 and N8
move together but does not remove the jump. Base no-storage PV curtailment is
about 584.6 MWh/day, 12.5% of PV availability. PV scaling moves the frontier
sharply: 0.8x largely removes the opportunity, while 1.2x keeps curtailment and
positive storage returns beyond 2.25 times the saved N6/N8 fleet. Therefore the
problem is not simply excessive PV; the base PV/network calibration lies near
a synchronized regime transition. Raising only the N8 corridor limits by 20%
reduced the largest screened N8 margin step from about 106.7 to 30.2
EUR/MW/day, but did not make the payoff continuous. These screens hold
aggregate E/P ratios fixed and are diagnostics, not equilibrium results.

`model/epec_jacobi_initializer.py` is the maintained initialization workflow.
The same workflow is now called automatically at the start of every fresh
Gauss-Seidel diagonalization run: it solves one Jacobi sweep from a common
economic capacity snapshot while using a separately configurable positive
numerical MPEC guess, then proportionally projects only overloaded nodes while
preserving each E/P ratio. The projected fleet is iteration 0 of the Seidel
loop. Checkpoint-resumed runs do not repeat initialization, and the standalone
initializer remains available for diagnostics. The projected allocation is an
initialization heuristic, not an equilibrium or an access-allocation mechanism.

The maintained algorithm uses private rival-headroom bounds and a final
shared-limit projection safeguard. Checkpoint/resume support persists MW and MWh
strategies after every completed iteration. Each investor's private nodal
headroom is represented as an explicit upper-level constraint, and its endogenous
NLP multiplier is exported as that investor's local marginal value of another MW
of access. These investor-specific shadow values are willingness-to-pay
diagnostics, not a common market-clearing access price. The earlier projected
common access-price iteration was removed after it failed to converge.

Capacity-only Gauss--Jacobi sweeps can run independent investor best responses
in separate worker processes. Parallel workers return lightweight price and
access diagnostics rather than serializing full Pyomo models. Fixed rival
node-hour storage blocks are omitted below a configurable power threshold
(default 0.01 MW), while the small capacity continues to consume shared nodal
headroom; 0.01/0.05/0.1 MW sensitivities should be checked before treating this
numerical sparsification as immaterial. An MPEC that exhausts its CPU-time,
iteration, or evaluation allowance is now left unchanged for that sweep rather
than immediately repeating the same expensive solve. Other numerical failures
retain one retry from a materially different 50% capacity start.

The separate two-follower access-auction experiment is under `model/auction/`.
Each leader chooses continuous nodal bid quantity, bid price, and energy
capacity while embedding both the access auction and the spot-market LP by
primal feasibility, dual feasibility/stationarity, and strong duality. The
maintained payment rule is a uniform nodal clearing price: the auction
objective carries a small strictly concave quadratic regularization (default
`1e-3` EUR/MW^2/day) that makes the allocation unique, continuous in the bids,
and symmetric among uncapped exact-price ties subject to bid-quantity limits,
and every awarded MW pays the auction capacity dual rather than its own bid.
When the nodal limit is slack the clearing price is zero. Independently
re-cleared auctions resolve any
degenerate multiplier face with a deterministic highest-rejected-bid
convention. This mechanism replaces continuous pay-as-bid bidding, whose
documented zero-price collapse, epsilon-overbidding race, and optimistic
embedded tie selection prevented any meaningful price/quantity equilibrium;
pay-as-bid remains available only as a legacy diagnostic mode, and the exact
`0.01` EUR/MW/day tick enumeration with `0.001` sub-tick merit priority
belongs to that legacy mode. Its diagonalization driver supports Gauss-Seidel
and simultaneous Gauss-Jacobi over the four thesis investors with no
outside-option solve, damping, or dispatch regularizer. In Jacobi mode every
investor responds to the same frozen bid snapshot, all sealed-bid proposals
are applied simultaneously, and one common auction assigns awards.
Convergence requires bid quantities, raw prices, durations, common awards,
unilateral embedded-award deviations, and, under the uniform rule, each
investor's embedded nodal clearing price against the common re-clear price to
stabilize. After the last sweep, one common auction re-clear and one common
spot market settlement report final awards, clearing prices, access payments,
and settled profits for all investors; the last optimistic best-response
profits remain separate diagnostics. It uses the maintained spot-price bounds
of `[-500,500]` and the `10,000` absolute bound for spot follower duals;
auction duals are bounded by the bid-price cap under the uniform rule. The
sparse rival and positive-generator-hour representation matches the normal
EPEC. This is an experimental extension and does not replace the maintained
projection EPEC.

The experimental strategic-operation extension is implemented in
`model/epec_strategic_operation_diagonalization.py`. Investors choose hourly
charge/discharge quantity offers in addition to nodal MW/MWh, while the ISO
retains dispatch control inside those offers. The identically calibrated
four-investor unregularized run reached 60 iterations without convergence:
aggregate investment stabilized at about 195.451 MW / 632.085 MWh, but
ownership/fine siting cycled across price-equivalent N3/N9 and merchant offers
remained unstable. All MPEC solves were optimal and lower-level strong-duality
gaps were small, so this was an algorithmic/nonunique-best-response issue.

An optional proximal diagonalization penalty is available through
`--proximal-penalty-eur-per-mw2-day`. It penalizes changes from the investor's
previous MW/MWh capacity and withheld charge/discharge quantities; it does not
penalize withholding itself, is off by default, and is excluded from the
Jacobi initializer. A staircase continuation is also available with an
explicit configurable initial zero-penalty period. The unified Tikhonov
strategic EPEC uses 1 EUR/MW^2/day steps by default: iterations 1-10 use zero,
11-15 use one, 16-20 use two, and so forth. Resuming iteration 60 with a fixed
coefficient of 1 EUR/MW^2/day
produced a regularized fixed point in iteration 61 with realized penalties
below 0.001 EUR/day per investor. An immediate unpenalized sweep moved again
(`dP`, `dE`, and `dOffer` about 0.31) at effectively unchanged reported profit.
Therefore the penalized result is an equilibrium-selection/numerical candidate,
not evidence that the original raw strategy map has a unique fixed point.

The strategic driver now supports both full Gauss--Jacobi and Gauss--Seidel
updates. In strategic Jacobi mode every investor solves against one frozen
snapshot containing all rivals' MW, MWh, charge/discharge quantities, and bid
prices; damped proposals are applied simultaneously after the complete sweep.
Independent best responses can run in separate worker processes. A same-snapshot
serial-versus-four-worker verification matched capacities, hourly offers, bid
prices, objectives, termination states, strong-duality gaps, and joint price
bounds exactly. Checkpoints are written only after complete sweeps. The first
200 MW two-sided-price Jacobi run reached 100 iterations without convergence.
Aggregate investment ended at 177.129 MW / 594.405 MWh, almost entirely at N8
and N6, but quantity offers and especially bid prices remained unstable. The
run used no proximal or dispatch regularization; its only objective selector
was a direct `1e-6` squared-price epsilon whose realized cost was economically
tiny. All MPECs were optimal, strong-duality gaps were small, and no projection
occurred, so this is a stable physical-investment candidate rather than a
converged strategic price-and-quantity equilibrium.

An optional two-sided strategic bidding mode is enabled with
`--strategic-bid-prices`. Each investor then chooses an hourly charging buy-bid
price/quantity pair and discharging sell-offer price/quantity pair. Submitted
prices are complete market bids: they replace the investor's private
degradation coefficients in the ISO objective, while physical degradation is
still subtracted from realized investor profit. Charging bids enter the ISO
cost-minimization objective with a negative sign (willingness to pay), and
discharging offers enter with a positive sign. The active investor's prices are
upper-level variables in its MPEC; rival prices are frozen parameters during
each best response. A linear bid-consistency constraint prevents submitted
prices from rewarding a same-hour efficiency-loss cycle, and joint settlement
exports explicitly diagnose any simultaneous charging/discharging. Bid-price
changes participate in damping, convergence, checkpointing, and the optional
proximal penalty after normalization by a configurable EUR/MWh price scale.
The separate `--strategic-epsilon-penalty` is an unnormalized direct
epsilon-times-square selector on charging buy-bid and discharging sell-offer
prices only. It contains no direct MW, MWh, or quantity-offer term; those
strategies remain governed by profit and the optional proximal continuation.
Without the flag, the earlier quantity-only formulation is preserved.

The main four-investor specification is:

- I1: stand-alone merchant BESS, 8% WACC;
- I2: stand-alone merchant BESS, 12% WACC;
- I3: wind-heavy renewable portfolio, 8% WACC;
- I4: solar-heavy renewable portfolio, 8% WACC.

## Previous empirical result (before the clean-base refactor)

The following values were produced before separate-rival batteries, fixed
demand, and zero dispatch regularization became the explicit maintained base.
They remain useful historical diagnostics but must be rerun before being quoted
as results of the current base formulation.

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
access prices, or settled investor profits. The maintained EPEC exports the
investor-specific multipliers of its private headroom constraints, but it
contains no common access-price variable, payment, update, or clearing rule. If
nodal access allocation is revisited, use a separate explicit merit-order auction
design rather than extending the discarded tâtonnement code.

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
