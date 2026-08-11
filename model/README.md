# Maintained BESS EPEC formulations

The maintained model uses fixed demand, DC market clearing, and endogenous
BESS MW/MWh capacity. Two formulations keep operation price-taking; a third
allows strategic two-sided operating prices without quantity withholding.

## Active files

- `primal_market_clearing_model.py` — primal lower-level market LP.
- `dual_market_clearing_model.py` — explicit dual LP and primal/dual check.
- `mpec_strong_duality.py` — single-investor MPEC using primal feasibility,
  dual feasibility, and strong duality.
- `mpec_kkt_bigm.py` — the same MPEC using explicit KKT complementarity and
  binary Big-M linearisation.
- `mpec_strategic_operation.py` — capacity-and-price MPEC with hourly charging
  bids and discharge offers, but no quantity withholding.
- `jacobi_diagonalization.py` — constructs fixed-rival best responses and
  applies simultaneous Gauss-Jacobi capacity and strategic-price updates.
- `run_model.py` — the executable and sole home of solver, Big-M,
  regularisation, damping, and convergence settings.
- `input/market_data.json` — the only active input.

All superseded models, scripts, datasets, and results are under `old/`.

## Run

```powershell
python model/primal_market_clearing_model.py
python model/dual_market_clearing_model.py --compare
python model/run_model.py --formulation strong-duality
python model/run_model.py --formulation strong-duality --parallel-workers 4
python model/run_model.py --formulation kkt-bigm --big-m-dual 800
python model/run_model.py --formulation strategic-operation --bid-price-bound 500
python model/run_model.py --investor-config model/input/investors_merchant_wind_pv.json --formulation strategic-operation --node-limit-mw 200 --initial-power-mw 0 --damping 0.25 --max-sweeps 20 --parallel-workers 3
python model/run_model.py --resume-from model/output/<run>/checkpoint.json --max-sweeps 60
python model/run_model.py --investor-config model/input/investors_merchant_wind_pv.json --parallel-workers 3
```

The `--formulation` selector exposes three maintained MPECs:

1. `strong-duality`: capacity-only nonlinear MPEC;
2. `kkt-bigm`: capacity-only KKT/Big-M MILP;
3. `strategic-operation`: capacity plus hourly two-sided storage prices,
   embedded by strong duality.

In `strategic-operation`, a charging bid is the maximum willingness to pay and
enters the ISO objective with a negative sign; a discharge offer enters with a
positive sign. Both are bounded by `--bid-price-bound`. Both initialize at zero
by default and can be changed with `--initial-bid-charge` and
`--initial-offer-discharge`. The restriction
`offer_discharge >= bid_charge / eta^2` excludes negative-cost same-hour
charge/discharge loops.

There are no bid-quantity variables. The ISO may dispatch charging and
discharging up to the investor's full installed MW in every node-hour. The
investor is settled at LMP and pays physical degradation; submitted prices are
not pay-as-bid settlement prices. Strategic runs write
`final_strategic_bids.csv`, and convergence also requires the raw bid-price
residual to satisfy `--bid-tolerance`.

Use `python model/run_model.py --help` for every numerical flag. The default
baseline has no proximal regularizer; `--proximal-penalty` enables an optional
L1 capacity best-response regularizer compatible with all three formulations.
The default Jacobi damping factor is `0.25`.

`--parallel-workers 4` solves the four best responses concurrently in separate
processes. All four still use the same frozen capacity snapshot, so this changes
runtime only—not the Gauss-Jacobi update rule. Four workers is the default;
select one explicitly for sequential execution.

Each run writes a configuration, compact history, checkpoint, final capacities,
and summary under `model/output/`; strategic runs additionally write final
two-sided bids. Use `--no-output` for a smoke test. Checkpoints retain the full
Jacobi state, convergence streak, projection count, and compact history. On
resume, `--max-sweeps` is the total target sweep number. The runner rejects
changed game settings or input data, while solver limits and the number of
parallel workers may be changed.

`--investor-config` accepts a JSON investor population without changing the
market-clearing or MPEC equations. The supplied
`input/investors_merchant_wind_pv.json` sensitivity has three 8% WACC investors:
one merchant, one 100% wind owner, and one 100% PV owner. Use
`--parallel-workers 3` so their three best responses solve concurrently.
`input/investors_original_without_i2.json` instead retains the original I1,
I3, and I4 definitions and removes only the 12% WACC merchant I2.
