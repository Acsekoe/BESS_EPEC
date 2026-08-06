# Tikhonov exact strong-duality verification

## Objective

Replace the finite-epsilon relaxed KKT representation with the matched
finite-gamma soft-balance primal and Tikhonov dual, coupled through exact strong
duality and no complementarity-product constraints.

## Implemented

- `model/tikhonov_kkt/soft_balance_llp.py`: matched standalone primal/dual.
- `model/tikhonov_kkt/strong_duality_formulation.py`: exact single-investor
  builder, soft-market initialization, diagnostics, and same-fleet audit.
- `model/tikhonov_kkt/mpec_strong_duality.py`: exact MPEC runner.
- `model/tikhonov_kkt/compare_single_investor_formulations.py`: controlled
  exact-versus-relaxed comparison.
- `model/tikhonov_kkt/compare_four_investor_strong_duality.py`: reproducible
  comparison of four standalone exact MPECs, one for each thesis investor.

## Verification

- Fixed 50 MW N6 / 50 MW N8 fleet: matched objectives within 0.01 EUR/day and
  primal-versus-dual prices within `6e-7` EUR/MWh.
- Exact MPEC (`gamma=1e-3`): optimal, 187.182153 MW / 547.213300 MWh,
  18,819.57 EUR/day.
- Exact MPEC internal `h + gamma*lambda` residual: below `7e-18` MW; matched
  strong-duality gap: about `5e-8` EUR/day.
- Independent same-fleet audit: maximum MPEC-price difference below 0.019
  EUR/MWh.
- Relaxed KKT (`epsilon=1e-3`) with a common soft-market initialization:
  capacity differed by `2.7e-5` MW and profit by 0.014 EUR/day from the exact
  formulation. Its minimum product remained negative at about `-9e-5`, so the
  theoretical finite-epsilon uniqueness issue remains despite this empirical
  agreement.

### Four standalone investors (`gamma=1e-3`)

All four Ipopt solves terminated `optimal` (local NLP solutions):

| Investor | Profile | MW | MWh | Objective profit (EUR/day) |
|---|---|---:|---:|---:|
| I1 | merchant, 8% WACC | 187.182153 | 547.213300 | 18,819.57 |
| I2 | merchant, 12% WACC | 187.182152 | 547.213299 | 17,873.29 |
| I3 | wind-heavy portfolio, 8% WACC | 132.011869 | 465.242745 | 124,196.89 |
| I4 | solar-heavy portfolio, 8% WACC | 187.336132 | 547.931680 | 183,784.52 |

The portfolio profits include owned renewable-generator rents and therefore
are not pure BESS-profit comparisons against I1/I2. I1 and I2 anticipated LMP
vectors agree to `1.9e-7` EUR/MWh maximum despite different nodal allocations.
Maximum pairwise differences were 38.50 EUR/MWh for I1--I3 and 41.69 EUR/MWh
for I1--I4. The largest four-way spread is at hour 12, N8: approximately 0 for
I1/I2, 38.495 for I3, and 41.691 EUR/MWh for I4.

Each MPEC's prices were independently re-cleared at its own proposed fleet.
Maximum MPEC-versus-soft-dual price errors were 0.01840 (I1), 0.01855 (I2),
0.00634 (I3), and 0.00875 EUR/MWh (I4). This supports consistency within each
finite-gamma market; the large cross-investor differences reflect different
endogenous fleets, not alternative prices for one fixed fleet.

Outputs: `model/output/tikhonov_four_standalone_strong_duality/` contains
`summary.json`, all 216 anticipated hour-node prices, capacities, and pairwise
price-difference tables.

## Interpretation and next step

The exact formulation removes epsilon-driven price freedom for each fixed
fleet but retains finite-gamma physical imbalance. It has not yet been wired
into the four-investor Jacobi driver. The next modeling step is to add it as a
separate lower-level-optimality option and then test gamma continuation from a
zero-capacity economic state.
