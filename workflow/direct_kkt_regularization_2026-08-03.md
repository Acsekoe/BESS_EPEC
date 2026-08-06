# Direct KKT regularization diagnostic

## Objective

Test whether direct KKT regularization can reproduce the saved ISO minimum-norm
price rule without another embedded optimization layer. All comparisons used
I2 against the identical frozen Jacobi snapshot: I1, I3, and I4 each fixed at
10 MW / 40 MWh at every node, a 200 MW nodal limit, and dispatch regularization
of `1e-4`.

The saved explicit minimum-norm I2 best response is 72.1718 MW / 167.0633 MWh,
profit 1,895.63 EUR/day, and solve time 90.61 seconds.

## Implementation

`model/single_investor_mpec_relaxed_kkt.py` now supports two optional
experimental parameters while preserving the prior Scholtes formulation by
default:

- dual Tikhonov regularization `h + gamma * lambda == 0`, where `h` is the
  original nodal-balance residual;
- shifted central-path equalities `(slack + shift) * dual == epsilon`.

Diagnostics report original nodal imbalance, complementarity residuals, and
the original primal-dual objective gap. These options are not wired into the
EPEC CLI because the diagnostic does not yet justify a sweep.

## Results

1. Full I2 best response with `gamma=1e-6`, shifted central-path
   `epsilon=1e-6`, and `shift=1e-8` failed at the 180-second CPU limit.
2. The same shifted central-path system with I2 capacity fixed at the saved
   minimum-norm solution failed at the 60-second CPU limit. The equality
   smoothing itself is therefore the immediate numerical problem.
3. With `gamma=1e-6` and the existing Scholtes `slack*dual <= 1e-4` conditions,
   the fixed-capacity system solved in 35.07 seconds. It matched the saved
   minimum-norm prices within 0.02925 EUR/MWh and profit within 0.35 EUR/day.
   Maximum nodal imbalance was `6.01e-5` MW and total absolute node-hour
   imbalance was 0.0119 MWh.
4. Allowing I2 capacity to move from the uniform 10 MW / 40 MWh guess produced
   45.1089 MW / 99.9819 MWh and profit 2,606.94 EUR/day in 101.82 seconds.
   A warm start from the saved minimum-norm capacity instead produced a poorer
   local solution of 66.8026 MW / 134.3114 MWh and profit 1,641.06 EUR/day.
5. An exact fixed-capacity minimum-norm dual QP at the 45.1089 MW direct-KKT
   solution solved in 9.42 seconds and valued profit at only 2,213.72 EUR/day.
   The direct formulation therefore overstated profit by 393.22 EUR/day at its
   own chosen capacity despite the small physical imbalance.

## Assessment

The gamma equation can closely reproduce minimum-norm prices at one fixed
capacity, but gamma `1e-6` combined with Scholtes epsilon `1e-4` does not impose
the same price rule throughout the endogenous capacity problem. The shifted
central-path equality is slower and failed even with capacity fixed. Direct KKT
regularization should remain experimental; no EPEC sweep is warranted from
these results. A further test, if desired, should study the ratio of epsilon to
gamma on fixed-capacity cases before any additional capacity best response.
