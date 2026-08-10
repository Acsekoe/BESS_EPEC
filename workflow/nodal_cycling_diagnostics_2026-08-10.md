# Nodal cycling diagnostics — 2026-08-10

## Bottom line

The present network does **not** provide sufficiently distinct and persistent
locational storage values to identify a unique nodal BESS allocation.

- N3 and N9 have identical no-BESS LMPs in every hour.
- A controlled 1 MW or 5 MW transfer between N3 and N9 changes the market
  objective, aggregate storage operating profit, curtailment, and LMPs only at
  numerical roundoff.
- N6, N8, N3, and N9 have nearly identical price-taking storage values.
- The archived exact-strong-duality Jacobi run reaches essentially fixed total
  MW and MWh by sweep 3, while individual raw nodal responses continue moving.
- At sweep 20, I3 proposes moving capacity from N8 toward N9 while I4 proposes
  the opposite movement. These offset in aggregate.

The observed cycling is therefore primarily selection among a flat or nearly
flat locational best-response face. Simultaneous Jacobi and optimistic dual
selection can amplify the selection changes, but they are not the underlying
source of the N3/N9 indifference.

## Important formulation discrepancy

The intended setup in the diagnostic request says “current minimum-norm LMP
selection.” That is not the maintained model currently in `model/`.

- `mpec_strong_duality.py` embeds optimistic dual variables.
- The archived 20-sweep run records
  `dual_selection = optimistic_mpec_no_price_penalty`.
- Minimum-norm pricing exists only in archived experiments and is explicitly
  inactive in `project_context.md`.

The no-BESS prices below are the basic LP dual returned by HiGHS. They are not
claimed to be a minimum-norm price vector. The exact N3/N9 perturbation result
does not depend on calling this vector minimum norm.

## 1. No-BESS market baseline

The active input already contains zero BESS capacity. The exact fixed-demand LP
solves at `394,587.851881 EUR/day`.

### Generation and curtailment

| Unit | Node | Cost EUR/MWh | Dispatch MWh | Curtailment/unused availability MWh |
|---|---:|---:|---:|---:|
| G_IEEE1 | N1 | 60.00 | 1,781.02 | 4,218.98 |
| G_IEEE2 | N2 | 52.20 | 5,478.22 | 1,721.78 |
| G_IEEE3 | N3 | 67.15 | 26.26 | 6,453.74 |
| RES_Wind_N1 | N1 | 0.00 | 1,804.69 | 0.00 |
| RES_PV_N6 | N6 | 0.00 | 1,359.45 | 278.35 |
| RES_PV_N8 | N8 | 0.00 | 2,735.35 | 306.26 |

Total renewable curtailment is `584.61 MWh/day`.

### Binding transmission constraints

Only two lines bind:

- L46: 6 hours, hours 11–16, lower direction, limit 180 MW.
- L78: 5 hours, hours 12–16, lower direction, limit 210 MW.

All other lines are nonbinding. Maximum no-BESS utilization is:

| Line | Limit MW | Maximum utilization |
|---|---:|---:|
| L14 | 400 | 84.16% |
| L45 | 400 | 80.46% |
| L57 | 400 | 76.70% |
| L72 | 400 | 75.00% |
| L69 | 180 | 48.33% |
| L98 | 210 | 41.43% |
| L93 | 400 | 6.56% |

### Hours creating the price separation

- Hour 11: L46 binds without renewable curtailment. LMPs are N1 = 60.00,
  N3/N9 = 46.72, N6 = 41.33, and N8 = 49.92 EUR/MWh.
- Hours 12–15: L46 and L78 bind and PV is curtailed. N3, N6, N8, and N9
  all price at zero while the uncongested thermal/load side prices at
  52.20 EUR/MWh.
- Hour 16: both corridors bind and 6.39 MW of N6 PV is curtailed. N6 prices at
  zero, N3/N9 at 5.39, N8 at 8.59, and N1 at 60.00 EUR/MWh.

The active input contains PTDFs rather than line reactances. Raw reactances
cannot be audited from the maintained dataset. The topology enters through the
PTDF matrix.

## 2. Nodal LMP differentiation

“Materially different” is defined as an absolute difference above
`0.5 EUR/MWh`.

| Pair | Mean absolute difference | Maximum difference | Material hours | Correlation |
|---|---:|---:|---:|---:|
| N3–N9 | 0.000 | 0.000 | 0 | 1.0000 |
| N3–N8 | 0.266 | 3.196 | 2 | 0.9993 |
| N3–N6 | 0.449 | 5.390 | 2 | 0.9982 |
| N6–N8 | 0.716 | 8.586 | 2 | 0.9954 |
| N1–N3 | 11.529 | 54.610 | 6 | 0.6980 |

There are additional exact equivalences: N1 = N4 and N2 = N7 in every hour.

N3 and N9 differ in the PTDF matrix only on L93. Because L93 never approaches
its limit, this electrical distinction has zero marginal economic effect.

## 3. Price-taking marginal storage value

Each case is a 1 MW battery with 93.6% one-way efficiency and periodic SOC.
“Gross” excludes degradation; the net diagnostic applies the maintained
15 EUR/MWh throughput convention.

### Gross arbitrage value, EUR/day

| Node | 2 MWh | 4 MWh | 6 MWh |
|---|---:|---:|---:|
| N1 | 8.66 | 9.34 | 9.34 |
| N3 | 119.47 | 230.32 | 270.58 |
| N6 | 119.47 | 231.79 | 281.36 |
| N8 | 119.47 | 229.44 | 264.19 |
| N9 | 119.47 | 230.32 | 270.58 |

For 1 MW / 4 MWh, the values after degradation are N3 = 170.18, N6 =
171.66, N8 = 169.31, and N9 = 170.18 EUR/day. The largest difference among
these four locations is only 2.35 EUR/day.

This gives one strong siting distinction—renewable/congested region versus
N1/N2/load-side buses—but almost no identification within the attractive
region.

## 4. Direct Jacobi cycling

The archived clean run used exact strong duality, no Tikhonov, no proximal
penalty, four workers, damping 0.25, a 200 MW nonbinding nodal access limit,
and no projection events.

- Sweep 1 applied total: 173.38 MW.
- Sweep 2 applied total: 187.09 MW.
- Sweeps 3–18: approximately 187.182 MW / 547.195 MWh.
- Last ten sweeps: total MW standard deviation 0.098 MW, including the small
  sweep-20 increase to 187.510 MW.
- Last ten sweeps: nodal raw deviations remain 3.09–9.27 MW.

Investor totals in sweeps 16–18 are nearly fixed:

- I1 ≈ 50.269 MW;
- I2 ≈ 50.269 MW;
- I3 ≈ 36.376 MW;
- I4 ≈ 50.269 MW.

At sweep 20:

| Investor | Node | Previous MW | Raw best response MW | Damped MW |
|---|---|---:|---:|---:|
| I3 | N8 | 5.522 | 0.000 | 4.142 |
| I3 | N9 | 11.711 | 18.332 | 13.366 |
| I4 | N8 | 11.985 | 16.224 | 13.044 |
| I4 | N9 | 26.931 | 20.161 | 25.239 |

I3 and I4 therefore exchange N8/N9 exposure in opposite directions. Aggregate
capacity changes much less than the individual raw responses.

Re-clearing the common market at every saved applied fleet gives the same
1 MW / 4 MWh marginal values from sweep 3 onward: N3/N9 = 58.66, N6 = 60.13,
and N8 = 57.78 EUR/day. The market opportunity is effectively unchanged while
the optimizer selects different nodal points.

The archived output does not contain every iteration’s investor-specific
optimistic MPEC dual vector. It contains the final vectors only. At the final
sweep, I1/I2 prices differ from the common settlement by as much as
38.46 EUR/MWh, while I3/I4 differ by only about 0.061 EUR/MWh. Optimistic price
selection is therefore an additional concern, especially for the merchants.

## 5. Controlled N3/N9 perturbation

The sweep-20 aggregate fleet was re-cleared after transferring 1 MW and 5 MW
in both directions, retaining the source-node duration.

For all four perturbations:

- objective change: at most `5.8e-11 EUR/day`;
- storage operating-profit change: at most `3.6e-12 EUR/day`;
- maximum LMP change: exactly zero at reported precision;
- renewable curtailment change: zero;
- total installed MW and MWh: unchanged.

Some primal generator dispatch components move along a degenerate optimal
face, but cost, prices, curtailment, and storage value do not change.

N3 and N9 are economically exact substitutes for these perturbations.

## 6. One-at-a-time market-data sensitivity screen

The screen changed no optimization equations.

### Changes that did not improve locational identification

- Tightening L46 or L78 by 10% increases curtailment by 90–105 MWh/day but
  leaves the 4-hour marginal storage values effectively unchanged. The system
  remains in the same linear congestion regime.
- Tightening L14 or L45 by 10% has no effect because neither line binds.
- Reducing L93 from 400 MW to 50 MW has no effect. Reducing it to 20 MW makes
  the fixed-demand system infeasible because required evening G3 exports
  exceed the rating.
- Moving 5% of N5 load to N9 does not separate N3 and N9 and reduces the
  existing renewable-corridor differentiation.
- Reducing total PV by 5%, increasing N1 wind by 10%, reducing G2 capacity by
  10%, or increasing G2 cost by 2 EUR/MWh does not break N3/N9 equivalence.

### Changes that create some additional differentiation

| Scenario | N6 4 h value | N8 4 h value | N3/N9 4 h value | Assessment |
|---|---:|---:|---:|---|
| Base | 231.79 | 229.44 | 230.32 | Almost flat |
| PV split 40/60 | 231.79 | 224.24 | 227.05 | Small improvement; only 2 material hours |
| L45 = 310 MW | 229.66 | 242.09 | 237.47 | Best single-line separation tested; N3/N9 still exact |
| L14 = 330 MW | 238.94 | 236.59 | 237.47 | Separates N1 side, not N3/N9 |

L45 = 310 MW is the strongest one-parameter candidate in the screen. It makes
L45 bind for two hours and creates a 12.43 EUR/day difference between N6 and
N8 for a 1 MW / 4 MWh battery. However, it is a 22.5% rating reduction and does
not resolve N3/N9 cycling. It should not be adopted without an external
physical basis for the 310 MW rating.

## 7. Interpretation of investor totals

The roughly equal I1, I2, and I4 totals are a saturation/frontier result rather
than proof that their objectives are identical. Once enough capacity removes
the valuable PV-curtailment opportunity, the useful aggregate capacity is
almost fixed. WACC differences change profit but need not change the capacity
at that discontinuous frontier.

I3 is wind-heavy. N1 wind is never curtailed and a 1 MW / 4 MWh battery at N1
earns only 9.34 EUR/day gross in the no-BESS prices, versus about 230 EUR/day in
the PV/congestion region. I3 owns only 20% of each PV unit, so it internalizes
less PV-curtailment relief than solar-heavy I4 and selects a lower total around
36 MW.

This is a plausible mechanism, but a global-regret comparison would still be
needed before treating the local Ipopt responses as globally optimal.

## 8. Recommendation

Do not tune line limits merely to manufacture a unique nodal answer. The data
support an aggregate/equivalence-class result, not uniquely identified siting.

The preferred sequence is:

1. Decide whether the maintained game should actually use minimum-norm prices;
   currently it does not. This must be resolved before comparing candidate
   EPEC equilibria under the intended final setup.
2. Treat N3/N9 as one economic zone unless a source-based L93 rating or changed
   load/generation geography makes that corridor persistently relevant.
3. If a data-only sensitivity is required now, use L45 = 310 MW only as a
   labelled sensitivity, not a recalibrated baseline, until the rating is
   physically justified.
4. A more defensible route to unique siting is to use source-based line ratings
   and load locations, or to define eligible BESS interconnection buses from
   physical siting assumptions. The current all-node strategy space includes
   electrically redundant transmission/generator buses.

Full candidate-data Jacobi equilibrium runs were not launched because no
single tested modification removed the exact N3/N9 equivalence, and the
requested minimum-norm price selector is absent from the maintained model.
Running long optimistic-price EPECs would not answer the stated final-setup
question and could be mistaken for minimum-norm results.

## Exported artifacts

All CSVs and figures are in:

`model/output/nodal_diagnostics_2026-08-10/`

Key files:

- `no_bess_lmps.csv`
- `no_bess_generation.csv`
- `no_bess_line_flows.csv`
- `no_bess_renewable_curtailment.csv`
- `no_bess_lmp_profiles.png`
- `lmp_pair_metrics.csv`
- `fixed_price_storage_value.csv`
- `fixed_price_storage_dispatch.csv`
- `archived_jacobi_power_detail.csv`
- `archived_jacobi_sweep_metrics.csv`
- `archived_jacobi_nodal_cycling.png`
- `archived_jacobi_reconstructed_joint_lmps.csv`
- `archived_jacobi_reconstructed_storage_value.csv`
- `n3_n9_shift_perturbations.csv`
- `market_data_sensitivity_screen.csv`
- `sensitivity_storage_value_4h.csv`
- `sensitivity_binding_lines.csv`
