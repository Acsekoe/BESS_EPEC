# Renewable-scenario experiment

This isolated experiment leaves the maintained deterministic models and market
data unchanged. It implements:

- an independent fixed-demand LLP clearing for each renewable scenario;
- a deterministic perfect-foresight planner benchmark for each scenario; and
- a two-stage stochastic planner with one shared BESS investment and
  scenario-specific dispatch, storage operation, flows, and prices.

The default low/base/high scenarios are illustrative rather than empirically
calibrated. They are shifted below the current deterministic calibration: the
current PV and wind profiles are the high case, while the weighted expected
scales are 0.90 for PV and 0.925 for wind.

From the repository root, run:

```powershell
python ".\model\stochstic\run_experiment.py"
```

Run only the base scenario as a deterministic consistency check:

```powershell
python ".\model\stochstic\run_experiment.py" --only-scenario base --no-export
```

Use another scenario file or output directory with `--scenarios` and
`--output-dir`. The default shared nodal BESS limit is 200 MW, matching the
current processed IEEE-9 data and the latest PV-sensitivity work.

This formulation assumes that renewable availability is known when each daily
spot market is dispatched. It does not model sequential forecast revelation,
day-ahead schedules, balancing, reserves, or nonanticipative hourly operation.

## Default illustrative result

With a 200 MW shared nodal limit and the scenarios in `scenarios.json`:

| Case | BESS MW | BESS MWh | Residual renewable curtailment MWh/day |
|---|---:|---:|---:|
| Low-scenario perfect foresight | 52.52 | 105.99 | 0.00 |
| Base-scenario perfect foresight | 120.40 | 305.22 | 0.00 |
| High-scenario perfect foresight | 187.18 | 547.19 | 0.00 |
| Shared stochastic investment | 187.18 | 547.19 | 0.00 expected |

The shared stochastic fleet costs 404,861.74 EUR/day in expectation, saves
9,441.99 EUR/day relative to no storage, and is 1,532.38 EUR/day above the
scenario-specific perfect-information lower bound. The high scenario still
drives the shared investment at its 25% probability, illustrating that scenario
averaging does not remove the curtailment-frontier discontinuity.

The high-only check reproduces the maintained aggregate planner result and
objective (187.18 MW / 547.19 MWh and 375,581.18 EUR/day). Fine allocation among
N3/N9 and the renewable nodes is LP-degenerate and can differ between HiGHS and
Ipopt while aggregate capacity and cost agree.

## Stochastic single-investor MPEC

Run the isolated capacity-only optimistic MPEC with:

```powershell
python ".\model\stochstic\stochastic_mpec.py"
```

The investor chooses one shared fleet, while every scenario embeds a separate
fixed-demand spot market using primal feasibility, dual feasibility, and exact
strong duality. The default 10 MW/node numerical start converged to a local
candidate with 52.52 MW / 187.22 MWh and expected profit of 4,945.31 EUR/day
under optimistic embedded prices. Independent HiGHS re-clears of the same fleet
give 4,013.06 EUR/day. The maximum embedded-versus-reference LMP difference is
29.91 EUR/MWh, so this remains an optimistic price/dispatch-selection result.

Initialization is material. A start from the stochastic planner fleet converged
to an inferior local candidate with 119.66 MW / 424.77 MWh and optimistic profit
of 4,173.49 EUR/day. Starts at 5 and 10 MW/node both reached the higher-profit
52.52 MW candidate. Ipopt optimal termination therefore establishes a local NLP
solution, not a global stochastic MPEC optimum.
