# Project context: clean capacity-only baseline

Last updated: 2026-08-14

## Objective

The project studies strategic BESS capacity investment in a deterministic
nodal spot market. The maintained code has deliberately returned to the
smallest coherent formulation before any extensions are reconsidered.

## Maintained formulation

The common lower level is a fixed-demand 24-hour DC market-clearing LP with
generation, renewable curtailment through capacity availability, BESS
charge/discharge, periodic state of charge, PTDF line limits, and hard nodal
and system balances. In the first seven formulations, investors choose nodal
BESS power and energy capacity under a shared nodal MW connection limit. The
strategic-access formulation instead makes only awarded MW a lower-level
outcome of co-optimised nodal access bids; MWh remains an independent
upper-level investment choice. The strategic-operation formulation
additionally lets investors choose hourly charging bids and discharge offers.

Eight single-investor formulations are maintained:

1. `model/mpec_strong_duality.py`: primal feasibility, dual feasibility, and
   strong duality, solved as a nonconvex NLP.
2. `model/mpec_relaxed_kkt.py`: primal feasibility, dual feasibility, and
   stationarity with every nonnegative KKT complementarity product bounded by
   a selectable Scholtes epsilon (default `1e-3`), solved as a smooth NLP with
   Ipopt. Product violations and the primal-dual gap are exported for audit.
3. `model/mpec_kkt_bigm.py`: explicit KKT complementarity with binaries and a
   user-set dual Big-M, solved as a MILP. Continuous capacity is retained; the
   investor revenue and portfolio rent are linearised exactly.
4. `model/mpec_strategic_operation.py`: the strong-duality MPEC extended with
   hourly charging buy-bids and discharging sell-offers. Charging and
   discharging remain bounded by the full installed MW, so investors cannot
   withhold quantity. Submitted prices replace degradation in the ISO
   objective; LMP settlement and physical degradation remain in investor
   profit.

5. `model/mpec_strategic_price_relaxed_kkt.py`: the price-only extension
with full MW availability and Scholtes-relaxed KKT lower-level optimality.
The ISO uses one nonredundant shared-inverter constraint,
`P_charge + P_discharge <= X_power`. SOC transition, energy bounds, and
periodicity remain in the lower-level physical market. There are no strategic
quantity or offer-deliverability constraints.

6. `model/mpec_strategic_quantity_relaxed_kkt.py` is integrated into the
Jacobi runner as `strategic-quantity`.
The investor chooses hourly maximum charging and discharging MW in
addition to installed MW/MWh. The ISO selects realised dispatch below those
offers, enforces cyclic SOC and a shared inverter limit, and minimizes only
generation cost. Offered MW must be sustainable for the complete one-hour
interval from the anticipated beginning-of-hour SOC. Lower-level optimality
uses the same selectable Scholtes relaxation as the capacity-only relaxed-KKT
model. Jacobi convergence includes raw quantity-bid deviations, and format-v4
checkpoints retain both hourly quantity profiles.

7. The same module is exposed as `strategic-price-quantity` when both bid
dimensions are active. The investor chooses charging and discharging MW
ceilings together with charging willingness-to-pay and discharge offer prices.
The ISO minimizes generation cost plus discharge offer cost minus charging
willingness-to-pay; realized storage remains settled at nodal LMP. The moving
L1 proximal term extends to capacity, quantity, and scaled price changes.
Convergence requires both quantity and price residuals, and format-v5
checkpoints retain the four hourly profiles.

8. `model/mpec_strategic_access_relaxed_kkt.py` is exposed as
`strategic-access`. Each investor submits a requested MW quantity and a
non-negative pay-as-bid access price in EUR/MW-day, subject to a portfolio-wide
request cap, and independently chooses nodal MWh. The ISO lower level jointly
awards nodal connection MW and clears generation, storage dispatch, SOC, and
the DC network. Awarded MW uses one shared inverter bound; investor MWh obeys
the linear technical bounds of 2--8 hours relative to awarded MW. Access
allocation and every physical storage constraint are represented by
Scholtes-relaxed KKT conditions. After each simultaneous Jacobi strategy
update, a common exact HiGHS clearing derives the awarded fleet; format-v7
checkpoints retain requests, bids, MWh strategies, and MW awards.
Fresh strategic-access runs seed every investor-node access bid at
`1 EUR/MW-day`. Their first 10 sweeps use full, undamped simultaneous best
responses; from sweep 11 onward they use the configured Jacobi damping factor
(default `0.25`). The cutoff is configurable and follows the total sweep
number when resuming a checkpoint.

`model/jacobi_diagonalization.py` builds the configured investor models against one
frozen rival-capacity snapshot and applies all responses simultaneously.
`model/run_model.py` is the only executable for these games and owns all
solver, Big-M, regulariser, damping, and convergence flags. Convergence is
measured from raw best-response deviations, not damped iterate changes.
Strategic-operation convergence also includes raw charging-bid and
discharge-offer deviations; strategic-quantity convergence includes raw
hourly MW-bid deviations. The active MPEC is selected with `--formulation`.
The four independent best responses can be solved in separate processes with
`--parallel-workers 4`, which is the CLI default. Select one worker explicitly
for sequential execution.
Format-v2 capacity, format-v3 strategic-price, format-v4 strategic-quantity,
format-v5 combined price-quantity, and format-v7 strategic-access checkpoints
retain the complete Jacobi state and can be resumed with `--resume-from`; the
runner rejects changed game settings or input data.
Ipopt formulations use HSL MA57 by default, with MUMPS retained as the
selectable `--ipopt-linear-solver mumps` fallback.
Strategic-quantity best responses warm-start the active investor from its
previous damped capacity and quantity profile. A fixed-bid HiGHS solve supplies
an exact lower-level KKT point. If the LP's selected SOC trajectory cannot
fully support the previous quantities, only their initial values are reduced
to the corresponding one-hour SOC limits and the LP is resolved; the MPEC
equations and feasible set are unchanged.
An explicit `--allow-proximal-penalty-change` restart may change only the L1
proximal coefficient and resets the convergence streak, supporting staged
zero-penalty then positive-penalty runs without weakening other compatibility
checks.

The four investors are I1 (merchant, 8% WACC), I2 (merchant, 12% WACC), I3
(wind-heavy renewable portfolio, 8%), and I4 (solar-heavy portfolio, 8%). The
only active dataset is `model/input/market_data.json`.

The four-player population remains the default. Economic portfolio
sensitivities may instead be supplied to `model/run_model.py` with
`--investor-config`; this changes only investor parameters and player count, not
the market-clearing or MPEC equations. The maintained three-player sensitivity
`model/input/investors_merchant_wind_pv.json` contains an 8% WACC merchant, a
100% wind owner, and a 100% PV owner.

## Explicitly inactive work

The previous monolithic MPEC/Jacobi implementation, central-planner and export
helpers, Tikhonov and superseded relaxed-KKT variants, superseded strategic
quantity-withholding variants,
stochastic models, diagnostics, historical inputs, and previous outputs are
archived under `model/old/`. They are reference material, not maintained code.

## Current verification

- The standalone primal and dual LLP both solve at `394,587.851881 EUR/day`,
  with zero objective gap and matching LMPs.
- The reduced strong-duality MPEC reproduces the established single-investor
  result: `187.182 MW`, `547.195 MWh`, `18,818.92 EUR/day`, and a
  `2.4e-8 EUR/day` primal-dual gap.
- A one-sweep strong-duality Jacobi smoke test solves all four best responses
  optimally from a zero economic fleet.
- The same sweep was verified with `--parallel-workers 4`: all four worker
  processes solved optimally and reproduced the sequential aggregate update of
  `351.139 MW / 1,047.337 MWh`.
- The KKT model is a linear MILP (1,994 binaries for the no-rival first-sweep
  case). With the strong-duality active set fixed, it reproduces capacity and
  profit to numerical precision, both without rivals and with three fixed
  rivals plus portfolio generation rent.
- The KKT path now supplies HiGHS with a feasible lower-level MIP start. A
  120-second unrestricted test recovered the same `187.182 MW / 547.195 MWh`
  solution and profit, but did not finish the global optimality proof. Timed-out
  incumbents are reported but are never applied as Jacobi best responses.
- A one-sweep three-investor strategic-operation smoke test from zero capacity
  solved all three price-and-capacity best responses optimally with three
  parallel workers. Its model contains no quantity-offer variables; both
  charge and discharge dispatch remain bounded by installed MW.
- The standalone strategic-quantity relaxed-KKT model reproduces the
  fixed-bid lower-level optimum at `394,587.851881 EUR/day` with zero
  primal-dual gap and KKT products below `2e-12`. A no-rival single-investor
  Ipopt solve terminated optimally at `187.182 MW`, `547.195 MWh`, and
  `19,006.70 EUR/day`; with epsilon `1e-3`, its maximum product was
  `0.001003643` and its primal-dual gap was `0.975 EUR/day`.
- The combined price-quantity formulation reproduces a fixed nonzero-bid ISO
  clearing with zero primal-dual gap and maximum KKT product below `5e-13`.
  A one-investor Ipopt best response with `rho=0.01` terminated optimally at
  `187.182 MW / 547.195 MWh`; its epsilon-relaxed maximum product was
  `0.001003643`. A four-worker first sweep then solved all four combined best
  responses optimally in about 27 seconds; this is an integration check, not
  an equilibrium claim.
- The price-only relaxed-KKT formulation reproduces a fixed nonzero-price ISO
  clearing with zero primal-dual gap and maximum KKT product below `8e-13`.
  A one-investor Ipopt solve and a four-worker first-sweep integration test
  both terminated optimally; the latter retained full MW availability and no
  quantity variables.
- The corrected strategic-access model auctions only lower-level MW and keeps
  MWh as an independent upper-level investment subject to linear 2--8 hour
  bounds. Its 6,590-variable four-player best response contains 5,153
  quadratic relaxed-KKT constraints and no cubic constraints. The independent-
  MWh fixed-strategy ISO LP is reproduced from its embedded KKT start with zero
  primal-dual gap, products below `3e-12`, and initial constraint violation near
  `1e-12`. A corrected four-worker first sweep with MA57, `rho=0`, a 40 MW
  nodal limit, 5 MW/node requests, 20 EUR/MW-day bids, and 3-hour initial MWh
  solved all four responses optimally in about 110 seconds wall time. This is
  an integration check, not an equilibrium claim.
- A four-investor undamped first-sweep integration smoke test solved all four
  strategic-quantity best responses optimally in parallel. The simultaneous
  update produced `307.801 MW / 877.716 MWh`; checkpoint round-trips were also
  verified for the existing formats 2/3 and the new quantity format 4.
- The requested five-sweep strategic-quantity run with damping `0.5`, four
  parallel workers, and a 600-second per-response limit solved all 20 best
  responses optimally. It did not converge: the last raw deviations were
  `28.653 MW`, `88.784 MWh`, and `28.653 MW` in the quantity profiles. The
  aggregate power sequence oscillated rather than contracting.
- A diagnostic common ISO clearing of the final damped state found
  `212.874 MW / 628.147 MWh`, zero simultaneous charge/discharge, and
  `363,477.11 EUR/day` generation cost versus `394,587.85 EUR/day` without
  storage. Strategic quantities increased cost by only about `8.09 EUR/day`
  relative to full availability at the same installed fleet, so the final
  nonconverged iterate does not yet exhibit material system-level withholding.
- On the final-state I3 response, MA57 reduced solve time from the comparable
  MUMPS run's `164.7 s` to `116.0 s`. Reusing and SOC-projecting the preceding
  active quantity profile as the warm start reduced the MA57 solve to `52.0 s`
  and returned a higher local objective. Ipopt's adaptive barrier strategy was
  rejected after taking `281.1 s` on the same initialized problem. The local
  solution can change with the start because the MPEC is nonconvex, although
  its mathematics is unchanged.
- An isolated four-worker sweep from the final checkpoint verified the retained
  MA57 plus SOC-feasible warm start end to end. All responses were optimal in
  `112.4`, `172.9`, `53.6`, and `58.4 s`; the wall time was about `174 s`.
  Against the comparable MUMPS sweep, the maximum response time fell from
  `318.1 s` to `172.9 s` and the mean from `216.7 s` to `99.3 s`. The temporary
  benchmark output is under
  `model/output/_runtime_smoke_strategic_quantity_sweep6/` and is not an
  equilibrium claim.

No converged multi-investor equilibrium is currently claimed.
