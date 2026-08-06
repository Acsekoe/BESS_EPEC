# Fixed-demand N6/N8 capacity-discontinuity diagnostics

## Objective

Test whether the BESS payoff cliff is caused by excessive midday PV, by holding
N6 fixed, or by the chosen network limits. The maintained EPEC and market data
were not changed.

## Method

Added `model/diagnose_capacity_discontinuity.py`, which aggregates the saved
four-investor fleet into one physically equivalent storage unit and solves the
exact unregularized fixed-demand market with HiGHS. It ran 221 post-hoc clears:

- 81 points on a 0--200 MW joint N6/N8 grid;
- 140 points covering PV scales 0.8--1.2, joint corridor scales 0.8--1.2,
  separate N6/N8 corridor changes, and 0--2.25 times the saved N6/N8 fleet.

Saved outputs:

- `model/output/capacity_discontinuity_diagnostics_2026-08-05/joint_n6_n8_grid.csv`
- `model/output/capacity_discontinuity_diagnostics_2026-08-05/pv_corridor_screen.csv`
- `model/output/capacity_discontinuity_diagnostics_2026-08-05/run_summary.json`

The full screen took 74.5 seconds. Aggregate node-specific E/P ratios were held
at the saved fleet values. Reported unit margins are aggregate average margins,
not unilateral marginal best responses.

## Results

The saved fleet has 66.50 MW at N6 and 87.77 MW at N8. Along its joint capacity
ray:

| Fleet scale | N6 MW | N8 MW | PV curtailment | N8 average net margin |
|---:|---:|---:|---:|---:|
| 0.75 | 49.87 | 65.83 | 120.9 MWh/day | +88.95 EUR/MW/day |
| 1.00 | 66.50 | 87.77 | 0 | -17.71 EUR/MW/day |

The 106.66 EUR/MW/day change reproduces the discontinuity. The joint grid shows
a downward-sloping curtailment-elimination frontier: at 75 MW N6, total
curtailment disappears between 75 and 100 MW N8; at 100 MW N6 it disappears
between 50 and 75 MW N8. N6/N8 coupling moves the frontier but does not remove
it.

True no-storage PV sensitivity:

| PV scale | N6+N8 curtailment | Share of PV | Curtailed hours N6/N8 |
|---:|---:|---:|---:|
| 0.8 | 96.1 MWh | 2.6% | 3 / 2 |
| 0.9 | 326.1 MWh | 7.7% | 4 / 3 |
| 1.0 | 584.6 MWh | 12.5% | 5 / 4 |
| 1.1 | 908.3 MWh | 17.6% | 6 / 5 |
| 1.2 | 1,256.3 MWh | 22.4% | 6 / 6 |

At 0.8x PV, 0.25 of the saved congested-node fleet already eliminates the
remaining curtailment and storage returns are negative. At 1.2x PV, curtailment
remains even at 2.25 times the saved fleet and the N8 average margin stays
positive. Base PV is therefore not merely too high; it lies near the interior
regime transition. Changing PV mainly translates the threshold and produces
zero-investment or cap-seeking corners.

Increasing only L98/L78 by 20% was the most promising network sensitivity. It
reduced the largest adjacent N8 margin change from 106.66 to 30.20
EUR/MW/day by staggering price regimes, but a discontinuity remained. Tightening
corridors moved the terminal jump toward a cap-binding outcome.

## Interpretation and next step

The cliff is structural to the deterministic fixed-demand LP: storage earns
curtailment rents until the last curtailed MWh is absorbed, after which the
charging price moves to a different marginal regime. Network changes can split
the transition into smaller steps or move it to a corner, but cannot guarantee
continuity.

Before changing the calibration, run a focused duration sensitivity near the
joint frontier and, if possible, a weighted multi-day expected-profit screen.
Do not select line limits solely because they make the EPEC converge.
