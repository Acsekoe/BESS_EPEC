# Strategic-power three-bus toy

This standalone toy removes the access auction and tests strategic BESS
operation directly. Four fixed-size batteries submit hourly discharge
availability ceilings. Charging availability is fixed at installed power by
default; pass `--two-sided` to make it strategic as well. The ISO decides actual dispatch, SOC,
generation and network flows. Investors receive nodal-price settlement.

The test system has three buses, three thermal generators, one wind unit, one
solar-PV unit, three constrained lines, six hours, and four 10 MW / 25 MWh
batteries. Splitting the original storage fleet this way preserves its total
40 MW / 100 MWh capacity:

- `I1` is a merchant investor with 8% WACC.
- `I2` is a merchant investor with 12% WACC.
- `I3` is wind-heavy: 80% of N1 wind and 20% of N3 PV.
- `I4` is solar-heavy: 20% of N1 wind and 80% of N3 PV.

All four batteries are located at the constrained N3 bus. This is the same
merchant/wind-heavy/solar-heavy ownership split as the maintained IEEE-9
model, reduced to one wind and one PV generator.

The generation ownership is deliberate: it provides a transparent incentive
for one investor to withhold storage discharge when higher prices benefit its
generator portfolio.

Small convex quadratic generation and storage-dispatch costs are included to
reduce dual-price ambiguity. Every best response is independently recleared;
the convergence test rejects candidates whose embedded prices or profit do not
match the exact reclear.

The Nash test also requires the independently recleared best-response profit
improvement to be at most EUR 0.50/day; a small MW strategy deviation is not
accepted when it crosses a large price discontinuity.

The default relaxed-KKT epsilon is `1e-5`. Convergence additionally requires
the independent reclear to agree within EUR 0.02/MWh for LMPs and EUR 0.25/day
for investor profit.

The market also contains a strongly penalized, two-sided demand adjustment
that pins the nodal-price selection at merit-order kinks. The adjustment is
exported in `final_market.csv` and is normally only hundredths of a MW.

A EUR 0.001/MW availability reward is used only as a lexicographic tie-breaker
when an offer is above realised dispatch and therefore economically
irrelevant. Reported investor profit excludes this term.

## Run

From this directory:

```powershell
python run_toy.py
```

IPOPT is used for best-response MPECs and HiGHS for exact market reclearing.
IPOPT is discovered from `IPOPT_EXECUTABLE`, `PATH`, or
`%LOCALAPPDATA%\idaes\bin\ipopt.exe`.

Useful diagnostic run:

```powershell
python run_toy.py --max-sweeps 3 --damping 0.25 --no-output
```

Two-sided availability experiment:

```powershell
python run_toy.py --two-sided
```

One-stage continuous investment-and-availability experiment:

```powershell
python run_joint_investment.py
```

In this alternative game, every investor chooses continuous installed power
and energy together with its hourly availability. Power is individually
bounded by 30 MW and duration is selected continuously between two and eight
hours. This is a simultaneous decision model, not the sequential two-stage
game. Its outputs are written to `output/joint_investment/`.

Outputs are written to `output/default/`:

- `summary.json`: convergence verdict and final profits
- `history.csv`: Gauss-Seidel and simultaneous Nash residuals
- `final_strategies.csv`: hourly availability offers
- `final_market.csv`: dispatch and LMPs
- `final_audit.csv`: final unpenalized best-response audit

Read [formulation.md](formulation.md) for the equations.

## Interpretation limits

This is a diagnostic model, not yet a thesis result. Capacity is fixed, each
hour is one hour long, there is no uncertainty or reserve market, and the MPEC
uses a relaxed complementarity tolerance. An IPOPT `optimal` status is a local
NLP result. A candidate equilibrium should still be tested with multiple
initializations and tighter epsilon values.
