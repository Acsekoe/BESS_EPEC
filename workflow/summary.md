# Strategic-operation IEEE-9 EPEC diagonalization

Last updated: 2026-07-28 17:55 Europe/Vienna

## Objective and status

The IEEE-9 storage-investment MPEC has been extended to a multi-investor EPEC
in which each investor chooses both nodal MW/MWh investment and hourly
charge/discharge quantity offers. A fresh run performs one common-snapshot
Gauss-Jacobi best-response sweep, projects the result onto the shared nodal
connection limits, and then enters a damped Gauss-Seidel loop.

The withholding and no-withholding cases remain independently runnable:

- `model/epec_strategic_operation_diagonalization.py`: investors choose
  capacity and hourly quantity offers; the ISO accepts operation within those
  offers.
- `model/epec_diagonalization.py`: investors choose capacity only; all
  installed battery power remains available to the ISO.

The former standalone `model/toy_strategic_storage_mpec.py` has been removed.

## Implemented model structure

`model/ieee9_strategic_operation_mpec.py` now accepts any number of distinct
rival battery owners. For every best response it freezes each rival's nodal
power, energy, hourly charge offer, hourly discharge offer, and degradation
cost. The active investor controls `Q_offer_charge[n,t]` and
`Q_offer_discharge[n,t]`, each bounded by its installed `X_power[n]`.

The ISO still operates every battery. Its lower-level constraints are

`P_charge[i,n,t] <= offered_charge[i,n,t]` and
`P_discharge[i,n,t] <= offered_discharge[i,n,t]`.

The active offer is a variable and every rival offer is a fixed parameter in
the active investor's MPEC. The installed-power terms in the lower-level dual
objective were replaced by the corresponding hourly offer terms, so primal
feasibility, dual feasibility/stationarity, and strong duality describe the
same offer-constrained ISO clearing problem.

This remains a quantity-withholding model. Investors do not choose offer
prices; the ISO objective continues to use physical degradation costs.

## Jacobi initialization and Gauss-Seidel loop

`model/epec_strategic_operation_diagonalization.py` maintains four strategy
maps: investor-node MW, investor-node MWh, investor-node-hour charge offers,
and investor-node-hour discharge offers.

For a fresh default run:

1. All investors solve a best response against the same economic snapshot
   (default 0 MW/node), using 10 MW/node only as the numerical Ipopt guess.
2. The desired Jacobi responses are combined and any overloaded node is
   projected proportionally. MW, MWh, and offers receive the same scale.
3. Gauss-Seidel begins in investor order. Later investors see capacities and
   offers updated by earlier investors in the same sweep.
4. Each update is damped, and offers are clipped to the updated installed MW.
5. Convergence requires relative changes in MW, MWh, and hourly offers all to
   be below tolerance. The capacity-only model checks MW and MWh only.

An economically slack offer is canonicalized to full installed availability.
This prevents arbitrary nonbinding offer values returned by Ipopt from
blocking numerical convergence. Offers that bind accepted dispatch are kept.

Strategic checkpoints contain capacities and both hourly offer maps. Resuming
continues directly with additional Seidel sweeps and does not repeat Jacobi.

## Common calibrated case and run commands

For an apples-to-apples comparison, both scripts must use the distributed
IEEE-9 data and the same generator calibration: +20 MW for each conventional
generator and a 200 MW peaker at N5 with a 95 EUR/MWh marginal cost.

Run the strategic-withholding EPEC from the repository root:

```powershell
python model\epec_strategic_operation_diagonalization.py --data model\data\processed\market_data_IEEE_9Bus_distributed_congestion.json --conventional-capacity-adder-mw 20 --peaker-node N5 --peaker-capacity-mw 200 --peaker-cost-eur-per-mwh 95 --max-cpu-time 180 --output-dir model\output\epec_strategic_operation\seidel_scarcity95
```

Run the capacity-only/no-withholding EPEC independently:

```powershell
python model\epec_diagonalization.py --data model\data\processed\market_data_IEEE_9Bus_distributed_congestion.json --update-rule seidel --conventional-capacity-adder-mw 20 --peaker-node N5 --peaker-capacity-mw 200 --peaker-cost-eur-per-mwh 95 --max-cpu-time 180 --output-dir model\output\epec\seidel_scarcity95_no_withholding
```

Both commands default to the four-investor portfolio, damping 0.7, nodal
settlement, fixed demand, a zero economic Jacobi snapshot, and a 100 MW shared
connection limit per node. Calibration metadata is written to `run_config.json`
and `summary.json` in both cases.

If a run reaches its iteration limit, use the same arguments plus
`--resume-from <output-directory>`. In both scripts, `--max-iters` then means
additional iterations.

## Strategic-specific outputs

The strategic driver reuses the standard EPEC outputs and adds:

- `strategic_quantity_offers.csv`: installed MW, charge/discharge offer,
  accepted charge/discharge, and joint LMP for every investor-node-hour;
- `offer_convergence_history.csv`: maximum relative offer change and aggregate
  charge/discharge offered capacity-hours by iteration;
- strategic offer maps in `checkpoint.json`;
- explicit experiment, strategy-space, and calibration metadata in the JSON
  summaries.

The shared joint settlement now detects offer-aware states and clears the final
ISO market subject to their hourly quantity offers. Capacity-only settlement is
unchanged because those states have no offer fields.

## What to compare with the other models

First verify that `run_config.json` matches on data path, calibration, investor
profiles/order, WACC, nodal limit, damping, tolerance, demand model, settlement
price basis, regularization, price/dual bounds, and solver tolerance. A result
is not comparable if these differ.

Then compare in this order:

1. **Convergence and feasibility.** Require `converged=true`, no shared-limit
   violation, small strong-duality gaps, and no unresolved solver failures.
2. **Investment.** Compare total and node-level MW/MWh, E/P ratios, ownership,
   and whether N8/N9 or the 100 MW connection limits remain dominant.
3. **Actual withholding.** In `strategic_quantity_offers.csv`, calculate
   installed MW minus offered MW. Distinguish binding withholding
   (`offer approximately accepted < installed`) from harmless unused capacity.
4. **ISO operation.** Compare accepted charge/discharge and SOC timing against
   `joint_storage_hour_operation.csv` in the capacity-only EPEC and against the
   central-planner dispatch.
5. **Prices.** Compare LMP duration distributions, hour-node heat maps, maxima,
   spatial spreads, and the number of hours near the 95 EUR/MWh peaker cost or
   the approximately 100.68 EUR/MWh congestion peak. Do not compare only the
   single maximum.
6. **Congestion and scarcity mechanism.** Check whether price increases occur
   when a strategic offer binds, whether the N5 peaker dispatch changes, and
   whether L46/L78 congestion or genuine capacity scarcity explains the LMP.
7. **Economics.** Use settled joint-market profit as the primary comparison.
   Also report optimistic MPEC profit minus settled profit; a large gap signals
   optimistic dual/price selection or a materially different final reclear.
8. **Algorithm robustness.** Compare trajectories, projection events,
   convergence speed, and sensitivity to investor order, damping, starting
   point, and local NLP solutions. This is a nonconvex EPEC, so one run is not
   proof of a unique equilibrium.
9. **Model scope.** The strategic case tests quantity withholding while the ISO
   retains dispatch control. It does not yet represent strategic offer prices,
   self-scheduling, reserve markets, or balancing-market actions.

If relevant-hour strategic offers converge to installed MW, the strategic
model should approach the capacity-only result, apart from nonconvex/local
solution effects. If investments remain similar but LMPs or profits differ,
the offer and accepted-operation files are the first place to locate the
mechanism.

## Verification completed

- Python compilation passed for both drivers, the strategic MPEC, and the
  shared results exporter.
- `git diff --check` passed for the changed model files.
- A two-owner construction test produced 18 storage owner-node pairs, 216
  active hourly offers, and 432 owner-node-hour charge bounds; the calibrated
  N5 peaker was present.
- A one-investor strategic end-to-end diagnostic completed one Seidel sweep,
  joint settlement, and all strategic CSV/JSON exports. It reproduced the
  calibrated 0 to 100.6836 EUR/MWh price range.
- A two-investor test completed the requested common-snapshot Jacobi sweep and
  one subsequent Seidel sweep. Jacobi desired investments were 176.538 MW /
  593.287 MWh for I1 and 139.382 MW / 503.265 MWh for I2. One initial solve hit
  the 60-second test limit; the built-in retry recovered.
- The identically calibrated capacity-only driver also completed an
  end-to-end one-sweep diagnostic and joint settlement.
- These one-sweep tests were deliberately nonconverged diagnostics. A full
  four-investor strategic equilibrium has not yet been run, so their capacities
  and profits must not be reported as equilibrium results.
- The temporary smoke-export directory was moved out of the workspace after
  validation, and the toy MPEC source was deleted as requested.

## Next steps

Run both four-investor commands to convergence, resume if needed, and only then
build the thesis comparison figures/tables. The most informative initial figure
set is: node-level investment, offer/accepted/installed operation for key hours,
LMP heat map or duration curves, and a settled-profit comparison with the
capacity-only EPEC and central planner.
