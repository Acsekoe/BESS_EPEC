# Reproduction snapshot — strategic operation (4-investor and 3-investor)

Frozen copy of the exact script chain behind the
`comparison_4v3_2026-08-10_16-38-15` figure, in particular the
**"4 investors – strategic operation"** panel.

**This is a snapshot, not a second maintained tree.** The files are verbatim
copies of `model/` as of 2026-09-01. Edit `model/`; if it changes, re-copy here
rather than editing both. Nothing in `model/` imports this folder.

## The chain, in call order

```
input/market_data.json                  network, generators, demand, PTDF, renewables
input/investors_original_without_i2.json  3-player population (I1, I3, I4)
        |
        v
run_model.py                            CLI, all solver / damping / convergence flags
        |                               --formulation strategic-operation
        v
jacobi_diagonalization.py               Gauss-Jacobi sweeps; one frozen rival snapshot
        |                               per sweep, four best responses, applied together
        v
mpec_strategic_operation.py             the MPEC: capacity + hourly charge bids /
        |                               discharge offers, full MW always available
        v
mpec_strong_duality.py                  the capacity-only MPEC it extends
        |                               (primal + dual feasibility + strong duality)
        v
primal_market_clearing_model.py         24-hour DC market-clearing LP, MarketData loader
```

### Why the other five MPEC modules are here

`jacobi_diagonalization.py` imports the whole formulation registry at module
scope, so these must be importable even though the strategic-operation path
never calls them:

- `mpec_relaxed_kkt.py`
- `mpec_kkt_bigm.py`
- `mpec_strategic_price_relaxed_kkt.py`
- `mpec_strategic_quantity_relaxed_kkt.py`
- `mpec_strategic_access_relaxed_kkt.py` (also imported directly by `run_model.py`)

The import closure of `run_model.py` is exactly these ten `.py` files. The only
module in `model/` that is *not* needed is `dual_market_clearing_model.py`,
which is a standalone dual-LP verification script.

## Reproducing the figure's four panels

Run from inside this folder. `--data` and `--investor-config` default to paths
next to the scripts, so no path juggling is needed.

```powershell
# 4 investors - capacity only (top row)
python run_model.py --formulation strong-duality `
  --max-sweeps 20 --damping 0.25 --parallel-workers 4 `
  --output-dir out/four_capacity_only

# 4 investors - strategic operation (second row)
python run_model.py --formulation strategic-operation `
  --max-sweeps 20 --damping 0.25 --parallel-workers 4 `
  --output-dir out/four_strategic_operation

# 3 investors - capacity only (third row)
python run_model.py --formulation strong-duality `
  --investor-config input/investors_original_without_i2.json `
  --max-sweeps 20 --damping 0.25 --parallel-workers 3 `
  --output-dir out/three_capacity_only

# 3 investors - strategic operation (bottom row)
python run_model.py --formulation strategic-operation `
  --investor-config input/investors_original_without_i2.json `
  --max-sweeps 20 --damping 0.25 --parallel-workers 3 `
  --output-dir out/three_strategic_operation
```

The original run's flags were not recorded — the run directories and the
plotting script were deleted. `--damping 0.25` is the runner default and matches
the sibling figure `comparison_4v3_damping025_50sweeps_...`; 20 sweeps matches
this figure's x-axis. Everything else is left at defaults. Expect the
trajectories to resemble, not equal, the original figure.

### Where the plotted numbers live

`history.csv` in each output directory, one row per sweep per investor. The
figure's y-axes are the columns `new_power_mw` (MW panels) and `new_energy_mwh`
(MWh panels), plotted against `sweep`.

Each run also writes `run_config.json`, `final_capacities.csv`, `summary.json`,
and a resumable `checkpoint.json`. The strategic-operation runs additionally
write `final_strategic_bids.csv` (the hourly charge bids and discharge offers).

## Verified

A one-sweep 4-investor strategic-operation run from inside this folder
(`--max-sweeps 1 --damping 0.25 --parallel-workers 4`) completed with all four
best responses optimal, on the default MA57 linear solver: total
`192.243 MW / 557.507 MWh`, max raw deviation
`66.874 MW / 181.557 MWh / 499.672 EUR/MWh`. That confirms the folder is a
complete, self-contained chain; it is an integration check, not an equilibrium
claim.

## Environment

- Ipopt is required for every `strategic-operation` and `strong-duality` solve
  and is **not on PATH** in this workspace. It is installed under
  `C:\Users\Alexander\AppData\Local\idaes\bin`; prepend that to PATH first, or
  Pyomo reports `ipopt: False`.
- The runner defaults to HSL MA57 (`--ipopt-linear-solver ma57`). If the local
  Ipopt build has no HSL, use `--ipopt-linear-solver mumps`.
- HiGHS (`appsi_highs`) is available and is used by the MILP path only.

## Caveat on the result itself

No converged multi-investor equilibrium is claimed for any formulation. Only a
one-sweep strategic-operation smoke test is recorded as verified in
`project_context.md`. The investor separation visible in the figure is a
20-sweep non-converged trajectory, not an equilibrium.
