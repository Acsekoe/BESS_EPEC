# Clean BESS EPEC

This is the maintained IEEE-9 four-investor model. The clean baseline lets each
investor strategically choose nodal BESS power and energy capacity. The joint
game additionally lets every investor submit hourly charging bids and
discharge offers while continuing to choose investment. The ISO retains
control of all accepted quantities.

There is no grid-access auction, access request, award, access payment, or
strategic operational quantity in either formulation.

## Investors

- `I1`: merchant, 8% WACC.
- `I2`: merchant, 12% WACC.
- `I3`: 80% wind / 20% PV portfolio, 8% WACC.
- `I4`: 20% wind / 80% PV portfolio, 8% WACC.

## Formulation

For every node, investor `i` chooses

```text
X_power[i,n] >= 0
ratio_min[i] * X_power[i,n] <= X_energy[i,n]
X_energy[i,n] <= ratio_max[i] * X_power[i,n]
```

The shared nodal constraint is retained as an ordinary physical capacity cap,
not an auction:

```text
sum_i X_power[i,n] <= node_limit_mw
```

The clean baseline uses `node_limit_mw = 1000`, which is intended to remain
far above the equilibrium fleet and therefore nonbinding.

The maintained default represents the lower-level competitive market with
Scholtes-relaxed KKT conditions and solves it with IPOPT. Exact strong duality
remains available as a cross-check through `--formulation strong-duality`.
A small quadratic demand-adjustment penalty selects a unique LMP; this is an
ISO variable, not an investor decision, and the adjustment is exported for
inspection. Every final best response is independently recleared with the same
fixed-capacity market model.

All upper-level strategies can use a moving quadratic proximal penalty centred
on the preceding Jacobi profile. In the capacity game it applies to MW and MWh;
in the operational game it also applies to every node-hour charging bid and
discharge offer.

## Run

From this directory:

```powershell
python model/run_model.py `
  --node-limit-mw 1000 `
  --max-sweeps 60 `
  --damping 0.25 `
  --parallel-workers 4 `
  --output-dir model/output/capacity_only_high_limits
```

The output folder contains the run configuration, rolling checkpoint and
history, current/final capacities, exact final market, simultaneous final
best-response audit, and summary. It also contains
`capacity_by_investor_node_by_sweep.csv` and
`capacity_totals_by_investor_by_sweep.csv`. Sweep 0 is the initial profile;
each later entry is the damped capacity profile used for the next sweep.

For a fresh ten-sweep joint investment and hourly bid/offer run:

```powershell
python model/run_hourly_bid_game.py `
  --initial-power-mw 5 `
  --initial-ratio-hours 3 `
  --max-sweeps 10 `
  --parallel-workers 4 `
  --proximal-capacity-penalty 0.01 `
  --proximal-bid-penalty 0.1 `
    --output-dir model/output/joint_investment_hourly_bids_10_sweeps
```

The maintained physical inverter constraint is
`P_charge + P_discharge <= X_power`.  To isolate the effect of the older
independent directional bounds, use the explicit experiment flag
`--inverter-limit separate`; the default remains `shared`.

For a three-player ablation without the 12% WACC merchant investor, add
`--exclude-investor I2`.  The selected population is saved and restored when
the run is resumed.

Renewable-support sensitivities can override the offer costs without editing
the canonical input file.  Use `--pv-offer-cost -25` for both PV units and/or
`--wind-offer-cost -25` for the wind unit.  Effective overrides are written to
`run_config.json` and restored automatically when resuming a run.

This command does not read the previous capacity result. Add `--fixed-capacity`
and `--capacities <file>` only when intentionally running the diagnostic
fixed-investment subgame.

To continue an existing joint-game state without restarting, use a new output
directory. Economic, regularization, and solver settings are inherited from
the source run:

```powershell
python model/run_hourly_bid_game.py `
  --resume-from model/output/joint_investment_hourly_bids_10_sweeps `
  --additional-sweeps 10 `
  --parallel-workers 4 `
  --output-dir model/output/joint_investment_hourly_bids_sweeps_1_20
```

See `project_context.md` for interpretation and validation rules.
