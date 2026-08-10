# Project context: clean capacity-only baseline

Last updated: 2026-08-10

## Objective

The project studies strategic BESS capacity investment in a deterministic
nodal spot market. The maintained code has deliberately returned to the
smallest coherent formulation before any extensions are reconsidered.

## Maintained formulation

The common lower level is a fixed-demand 24-hour DC market-clearing LP with
generation, renewable curtailment through capacity availability, BESS
charge/discharge, periodic state of charge, PTDF line limits, and hard nodal
and system balances. All investors choose nodal BESS power and energy capacity
under a shared nodal MW connection limit. The strategic-operation formulation
additionally lets them choose hourly charging bids and discharge offers.

Three single-investor formulations are maintained:

1. `model/mpec_strong_duality.py`: primal feasibility, dual feasibility, and
   strong duality, solved as a nonconvex NLP.
2. `model/mpec_kkt_bigm.py`: explicit KKT complementarity with binaries and a
   user-set dual Big-M, solved as a MILP. Continuous capacity is retained; the
   investor revenue and portfolio rent are linearised exactly.
3. `model/mpec_strategic_operation.py`: the strong-duality MPEC extended with
   hourly charging buy-bids and discharging sell-offers. Charging and
   discharging remain bounded by the full installed MW, so investors cannot
   withhold quantity. Submitted prices replace degradation in the ISO
   objective; LMP settlement and physical degradation remain in investor
   profit.

`model/jacobi_diagonalization.py` builds the configured investor models against one
frozen rival-capacity snapshot and applies all responses simultaneously.
`model/run_model.py` is the only executable for these games and owns all
solver, Big-M, regulariser, damping, and convergence flags. Convergence is
measured from raw best-response deviations, not damped iterate changes.
Strategic-operation convergence also includes raw charging-bid and
discharge-offer deviations. The active MPEC is selected with `--formulation`.
The four independent best responses can be solved in separate processes with
`--parallel-workers 4`, which is the CLI default. Select one worker explicitly
for sequential execution.
Format-v2 capacity checkpoints and format-v3 strategic checkpoints retain the
complete Jacobi state and can be resumed with `--resume-from`; the runner
rejects changed game settings or input data.

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
helpers, Tikhonov and relaxed-KKT variants, strategic quantity withholding, auctions,
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

No converged multi-investor equilibrium is currently claimed.
